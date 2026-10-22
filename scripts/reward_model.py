"""
reward_model.py  —  GRPO Reward Function
------------------------------------------------------------------------
Implements the full utility function:

  U(o, e | s) = wR·R(o,s) + wV·V(e,s) − wC·C(e,s) − wP·P(o,s) − μ·Φ(o,y,α,η)

All terms are dimensionless: R(o,s) is normalised by the
maximum possible energy saving so every term lives in a comparable
numeric range.  No physical units appear in the reward scalar.

Outputs scored
──────────────
  offload_decision   o ∈ {true, false}
  escalation_flag    e ∈ {true, false}

Reward range (approximate)
───────────────────────────
  Parse failure / wrong schema          : −1.0  (hard floor)
  Valid schema, both decisions wrong    : −0.5 to 0
  Valid schema, one decision correct    : +0.1 to +0.5
  Both decisions correct, no violation : +0.7 to +1.0
  Both correct, escalation well-valued : up to +1.3

Simulation parameters
────────────────────────────────────────────────────────────────────
  beta_c        = 20%        critical battery threshold
  kappa_stable  = 1.0        network condition coefficient — stable
  kappa_cong    = 1.4        network condition coefficient — congested
  kappa_weak    = 2.5        network condition coefficient — weak signal
  gamma_min     = 0.10       minimum energy credit (normalised)
  gamma_esc     = 0.20       escalation credit threshold
  Z             = 1000 ms    max tolerable escalation round-trip latency
  G0            = 0.60       base expected gain from network LLM
  lambda_c      = 25.0       C1 penalty coefficient (raised 15→20→25)
  lambda_ws     = 12.0       C8 penalty coefficient (weak signal, unchanged)
  lambda_cws    = 45.0       C1∩C8 interaction penalty ≥ λ_C1+λ_C8=37 (raised 30→40→45)
  lambda_exp    = 1.0        credit penalty exponential sensitivity
  wR = wV = wC = wP = 1.0   equal objective weights (ablation baseline)
  mu            = 1.0        credit penalty weight

  All penalty coefficients are expressed in the same dimensionless scale
  as the normalised rewards so no unit conversion is needed.

"""

import json
import math
import re

# ═══════════════════════════════════════════════════════════════════════════════
# Simulation parameters
# ═══════════════════════════════════════════════════════════════════════════════

# Network condition coefficients κ(η)  — burn rate multipliers
KAPPA = {
    "stable":      1.0,
    "congested":   1.4,
    "weak signal": 2.5,
}

# Credit thresholds (normalised, matching generate_dataset.py)
GAMMA_MIN = 0.10
GAMMA_ESC = 0.20
GAMMA_MAX = 1.0   # B normalised to 1

# Constraint thresholds
BETA_C = 20        # critical battery (%) — C1 threshold

# Latency
Z_MS = 1000.0      # max tolerable round-trip latency [ms]

# Approximate round-trip latency by network condition [ms]
# T_UL(η) + T_inference(~200ms for 7B MEC model) + T_DL(~5ms)
C_LATENCY_MS = {
    "stable":      215.0,    # ~10 UL + 200 inference + 5 DL
    "congested":   350.0,    # ~145 UL + 200 inference + 5 DL
    "weak signal": 9999.0,   # C8 blocks escalation — set > Z
}

# Penalty coefficients (dimensionless, eq. 6)
# Must satisfy: lambda_cws >= lambda_c + lambda_ws
LAMBDA_C   = 25.0   # C1: critical battery violation  (was 20.0)
LAMBDA_WS  = 12.0   # C8: weak signal violation       (unchanged)
LAMBDA_CWS = 45.0   # C1∩C8 interaction (45 >= 25+12=37 )  (was 40.0)

# Credit penalty exponential sensitivity (eq. 9)
LAMBDA_EXP = 1.0

# Escalation gain
G0 = 0.60   # base expected gain from network LLM (offline benchmark estimate)

# Objective weights — equal baseline for first training run
W_R = 1.0   # offload utility weight
W_V = 1.0   # escalation value weight
W_C = 1.0   # escalation cost weight
W_P = 1.0   # constraint violation penalty weight
MU  = 1.0   # credit penalty weight

# App compute demand F(α) [GFLOP/s] — from generate_dataset.py
F_ALPHA = {
    "podcast":       1.0,
    "system update": 0.1,
    "video":         15.0,
    "game":          120.0,
    "ar/vr":         900.0,
}
F_ALPHA_MAX = 900.0 * 2.5   # AR/VR under weak signal — normalisation ceiling


# ═══════════════════════════════════════════════════════════════════════════════
# Context parser  (reads the four state variables from the prompt)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_context(prompt):
    """
    Extract (battery, app_key, condition, gamma) from the natural language prompt.

    Returns
    ───────
    battery   : int     β ∈ [1, 100]
    app_key   : str     matched app keyword
    condition : str     η ∈ {stable, congested, weak signal}
    gamma     : float   γ ∈ [0, 1]  — energy credit level
    """
    p = prompt.lower()

    # β — battery level
    m = re.search(r'battery is at (\d+)%', p)
    battery = int(m.group(1)) if m else 50

    # α — app type
    app_key = "podcast"   # default
    for key in F_ALPHA:
        if key in p:
            app_key = key
            break

    # η — network condition (check "weak signal" before "weak")
    if "weak signal" in p:
        condition = "weak signal"
    elif "congested" in p:
        condition = "congested"
    else:
        condition = "stable"

    # γ — energy credit (extract percentage from "(XX%)" pattern)
    m_credit = re.search(r'energy credit is \w[\w ]*\((\d+)%\)', p)
    gamma = int(m_credit.group(1)) / 100.0 if m_credit else 1.0

    return battery, app_key, condition, gamma


# ═══════════════════════════════════════════════════════════════════════════════
# Burn rate helper
# ═══════════════════════════════════════════════════════════════════════════════

def burn_rate_normalised(app_key, condition):
    """
    ρ(α,η) normalised to [0,1].
    ρ_raw = F(α) × κ(η)   [GFLOP/s]
    ρ_norm = ρ_raw / (F_max × κ_max)
    """
    raw = F_ALPHA.get(app_key, 1.0) * KAPPA.get(condition, 1.0)
    return raw / F_ALPHA_MAX


# ═══════════════════════════════════════════════════════════════════════════════
# Objective terms  (all dimensionless, Option B)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_R(o, battery, app_key, condition, gamma):
    """
    Offload utility R(o, s) normalised to [−1, +1].

    r_compute(α): normalised compute saving from offloading.
                  Proportional to app compute demand (heavy apps save more).
    r_tx(β,η):    normalised transmission energy cost.
                  Increases as battery drops or η deteriorates.
    r_local(α,β): normalised local execution cost.
                  High for heavy apps at low battery.

    Normalisation: divide all raw Joule estimates by max possible saving
    so R ∈ [−1, +1].
    """
    rho = burn_rate_normalised(app_key, condition)

    # Compute saving from offloading (proportional to app demand, normalised)
    r_compute = rho   # heavier app → more saving from offload

    # Transmission cost: higher when battery is low or link is degraded
    # (1 − β/100) captures battery scarcity; κ(η)/κ_max captures link cost
    battery_factor = 1.0 - battery / 100.0
    link_factor    = KAPPA.get(condition, 1.0) / KAPPA["weak signal"]
    r_tx = 0.3 * (battery_factor + link_factor) / 2.0   # normalised to ~[0,0.3]

    # Local execution cost: high for heavy apps at low battery
    r_local = rho * battery_factor   # normalised to [0, 1]

    if o:
        return r_compute - r_tx      # net saving from offloading
    else:
        return -r_local              # cost of running locally


def compute_V(e, battery, app_key, condition, gamma):
    """
    Escalation value V(e, s)  extended with missed-escalation penalty.

    When e = 1:  V = (1 − q(s)) · G_net(s)   [positive reward]
    When e = 0:  V = −(1 − q(s)) · G_net(s) · α_miss   [negative penalty]
                 where α_miss < 1 so missed escalation is penalised less
                 severely than the gain from escalating correctly.

    q(s): structural confidence proxy (training-time approximation of
          mean log-probability confidence score at inference time).
          Near C1 boundary or in credit zone → low q (uncertain).
          Safe state → high q (confident).

    G_net(s) = G0 · (1 − γ/B)  — eq. 5.
               More valuable when credits are scarce.
    """
    ALPHA_MISS = 0.4   # missed-escalation penalty is 40% of escalation gain

    # Structural confidence proxy
    near_c1     = BETA_C < battery <= 25
    credit_zone = GAMMA_MIN < gamma <= GAMMA_ESC
    heavy_cong  = (app_key in ("game", "ar/vr")
                   and condition == "congested"
                   and 40 <= battery <= 60)

    if near_c1 or credit_zone or heavy_cong:
        q = 0.30   # genuinely uncertain
    elif battery <= BETA_C or condition == "weak signal":
        q = 0.05   # constraint is clear
    else:
        q = 0.75   # safe state — model should be confident

    G_net = G0 * (1.0 - gamma / GAMMA_MAX)
    value = (1.0 - q) * G_net

    if e:
        return value                # reward for escalating when uncertain
    else:
        return -ALPHA_MISS * value  # penalty for NOT escalating when uncertain


def compute_C(e, condition):
    """
    Escalation cost C(e, s) normalised by Z.

    C(e,s) = e · c_net(η) = e · c_latency(η) / Z

    Ranges from 0 (no escalation) to >1 (latency exceeds budget, infeasible).
    """
    if not e:
        return 0.0
    c_lat = C_LATENCY_MS.get(condition, 9999.0)
    return min(c_lat / Z_MS, 1.5)   # cap at 1.5 to avoid extreme outliers


def compute_P(o, battery, condition):
    """
    Constraint violation penalty P(o, s) normalised.

    P(o,s) = o · [λ_c·𝟙[β≤β_c] + λ_ws·𝟙[η=weak signal]
                  + λ_cws·𝟙[β≤β_c AND η=weak signal]]

    Coefficients are in the same dimensionless scale as normalised rewards.
    Normalise by LAMBDA_CWS (the largest coefficient) so P ∈ [0, ~2].
    """
    if not o:
        return 0.0

    c1 = battery <= BETA_C
    c8 = condition == "weak signal"

    raw = (LAMBDA_C   * int(c1 and not c8)
         + LAMBDA_WS  * int(c8 and not c1)
         + LAMBDA_CWS * int(c1 and c8))

    return raw / LAMBDA_CWS   # normalise: worst case = 1.0 (c1∩c8 = 45/45)


def compute_Phi(o, gamma, app_key, condition):
    """
    Energy credit penalty Φ(o, y, α, η).

    Φ = o · ρ(α,η) · [exp(λ · (1 − γ/B)) − 1]

    Zero when γ = B (full credits) or o = 0 (local execution).
    Grows exponentially as credits deplete.
    Normalised by the maximum value of the exponential factor.
    """
    if not o:
        return 0.0

    rho        = burn_rate_normalised(app_key, condition)
    credit_use = 1.0 - gamma / GAMMA_MAX
    phi_raw    = rho * (math.exp(LAMBDA_EXP * credit_use) - 1.0)

    # Normalise by maximum possible value: rho_max=1 × (exp(λ)−1)
    phi_max    = 1.0 * (math.exp(LAMBDA_EXP) - 1.0)
    return phi_raw / phi_max   # Φ ∈ [0, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# JSON parsing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def dict_raise_on_duplicates(ordered_pairs):
    d = {}
    for k, v in ordered_pairs:
        if k in d:
            raise ValueError(f"Duplicate key: {k}")
        d[k] = v
    return d


def _repair_json(raw: str) -> str:
    """
    Best-effort repair of common LLM JSON mistakes.
      1. Python booleans/None → JSON literals
      2. Single-quoted strings → double-quoted
      3. Unquoted bare-word keys → double-quoted
      4. Trailing comma before } → removed
      5. Truncate at first }
      6. Append missing }
    """
    s = raw
    s = re.sub(r'\bTrue\b',  'true',  s)
    s = re.sub(r'\bFalse\b', 'false', s)
    s = re.sub(r'\bNone\b',  'null',  s)
    s = re.sub(r"'([^']*)'", r'"\1"', s)
    s = re.sub(r'(?<=[{,])\s*([A-Za-z_][A-Za-z_0-9]*)\s*:', r' "\1":', s)
    s = re.sub(r',\s*}', '}', s)
    if '}' in s:
        s = s.split('}')[0] + '}'
    else:
        s = s.rstrip() + '\n}'
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# Terminal printer
# ═══════════════════════════════════════════════════════════════════════════════

def _print_step(prompt, raw, repaired, reward, components=None, parse_error=None):
    bat  = re.search(r'Battery is at (\d+)%',       prompt)
    app  = re.search(r'opened a (.+?)\.',            prompt)
    net  = re.search(r'Network is (\w[\w ]*?)\.',   prompt)
    cred = re.search(r'Energy credit is (\w[\w ]*?)\(', prompt)

    ctx = "  ".join(filter(None, [
        f"bat={bat.group(1)}%"         if bat  else "bat=?",
        f"app={app.group(1)}"          if app  else "app=?",
        f"net={net.group(1).strip()}"  if net  else "net=?",
        f"cred={cred.group(1).strip()}" if cred else "cred=?",
    ]))

    bar   = "─" * 68
    color = "\033[92m" if reward >= 0 else "\033[91m"
    reset = "\033[0m"

    print(f"\n{bar}")
    print(f"  Context  : {ctx}")
    print(f"  Raw out  : {raw[:120].strip()}")
    if repaired and repaired != raw:
        print(f"  Repaired : {repaired}")
    if parse_error:
        print(f"  x Parse  : {parse_error}")
    if components:
        R, V, C, P, Phi = components
        print(f"  U terms  : wR·R={W_R*R:+.3f}  wV·V={W_V*V:+.3f}"
              f"  wC·C={W_C*C:+.3f}  wP·P={W_P*P:+.3f}  μ·Φ={MU*Phi:+.3f}")
    print(f"  Reward   : {color}{reward:+.4f}{reset}")
    print(bar)


# ═══════════════════════════════════════════════════════════════════════════════
# Main reward function
# ═══════════════════════════════════════════════════════════════════════════════

REQUIRED_KEYS = {"offload_decision", "escalation_flag"}


def calculate_ue_reward(prompt, generated_text, verbose=True,
                        return_components=False):
    """
    Score the model's two-output decision against eq. 10.

    Pipeline
    ────────
    1. Parse the prompt to extract state (β, α, η, γ)
    2. Repair and parse the generated JSON
    3. Extract o and e from the JSON
    4. Check schema validity (partial credit if incomplete)
    5. Compute all five objective terms
    6. Apply hard feasibility overrides for C1, C8, credit floor
    7. Return U(o,e|s) as the GRPO reward scalar

    Returns
    ───────
    float   The reward scalar passed to GRPOTrainer.
            Positive means a good decision; negative means a bad one.
            Range: approximately [−1.0, +1.3]
    """
    raw = generated_text.strip()
    if not raw.startswith("{"):
        raw = "{" + raw
    clean_json = _repair_json(raw)

    # ── Step 1: parse context from prompt ────────────────────────────────────
    battery, app_key, condition, gamma = parse_context(prompt)

    # ── Step 2: parse generated JSON ─────────────────────────────────────────
    try:
        config = json.loads(clean_json,
                            object_pairs_hook=dict_raise_on_duplicates)
    except ValueError as e:
        if verbose:
            _print_step(prompt, raw, clean_json, reward=-1.0,
                        parse_error=str(e))
        if return_components: return -1.0, -1.0
        return -1.0   # parse failure — hard floor

    # ── Step 3: schema check ──────────────────────────────────────────────────
    keys_found  = set(config.keys())
    extra_keys  = keys_found - REQUIRED_KEYS
    missing_keys = REQUIRED_KEYS - keys_found

    # Start with a small base for producing valid JSON
    schema_reward = 0.1

    # Partial credit: +0.1 per correct key found
    for key in REQUIRED_KEYS:
        if key in config:
            val = config[key]
            if isinstance(val, bool):
                schema_reward += 0.1   # correct type
            else:
                schema_reward -= 0.05  # wrong type

    # Penalise hallucinated keys
    schema_reward -= 0.15 * len(extra_keys)

    # If schema is incomplete or types are wrong, return schema reward only
    if missing_keys or extra_keys:
        reward = schema_reward
        if verbose:
            _print_step(prompt, raw, clean_json, reward)
        if return_components: return reward, reward - schema_reward
        return reward

    # ── Step 4: extract decisions ─────────────────────────────────────────────
    o = bool(config["offload_decision"])
    e = bool(config["escalation_flag"])

    # ── Step 5: hard feasibility overrides ───────────────────────────────────
    # These mirror C1, C8, and the credit floor from the constraint hierarchy.
    # A feasibility violation gets a strong negative reward regardless of
    # what the utility function would otherwise score.

    c1 = battery <= BETA_C
    c8 = condition == "weak signal"
    credit_floor_violated = gamma <= GAMMA_MIN and o

    # C1 ∩ C8: o must be 0, e must be 0
    if c1 and c8:
        feasibility_reward = 0.0
        if o:  feasibility_reward -= 0.5
        if e:  feasibility_reward -= 0.3
        reward = schema_reward + feasibility_reward
        if verbose:
            _print_step(prompt, raw, clean_json, reward)
        if return_components: return reward, reward - schema_reward
        return reward

    # C1 only: o must be 0
    if c1 and o:
        reward = schema_reward - 0.5   # strong penalty for C1 violation
        if verbose:
            _print_step(prompt, raw, clean_json, reward)
        if return_components: return reward, reward - schema_reward
        return reward

    # C8 only: o must be 0, e must be 0
    if c8 and (o or e):
        penalty = 0.0
        if o: penalty -= 0.5
        if e: penalty -= 0.3
        reward = schema_reward + penalty
        if verbose:
            _print_step(prompt, raw, clean_json, reward)
        if return_components: return reward, reward - schema_reward
        return reward

    # Credit floor: o must be 0 when gamma <= gamma_min
    if credit_floor_violated:
        reward = schema_reward - 0.4
        if verbose:
            _print_step(prompt, raw, clean_json, reward)
        if return_components: return reward, reward - schema_reward
        return reward

    # (o=1, e=1) is always invalid
    if o and e:
        reward = schema_reward - 0.4
        if verbose:
            _print_step(prompt, raw, clean_json, reward)
        if return_components: return reward, reward - schema_reward
        return reward

    # ── Step 6: compute all five objective terms ──────────────────────────────
    R   = compute_R(o,       battery, app_key, condition, gamma)
    V   = compute_V(e,       battery, app_key, condition, gamma)
    C   = compute_C(e,       condition)
    P   = compute_P(o,       battery, condition)
    Phi = compute_Phi(o,     gamma,   app_key, condition)

    # ── Step 7: assemble U(o,e|s) ───────────────────────────────────
    U = (W_R * R
       + W_V * V
       - W_C * C
       - W_P * P
       - MU  * Phi)

    # Add schema reward on top — keeps gradient signal alive even when
    # the output format is correct but the decision is wrong
    reward = schema_reward + U

    if verbose:
        _print_step(prompt, raw, clean_json, reward,
                    components=(R, V, C, P, Phi))

    # return_components=True: return (total, U) so the training loop can
    # log schema_reward and U separately for the reward decomposition chart.
    if return_components:
        return reward, U
    return reward