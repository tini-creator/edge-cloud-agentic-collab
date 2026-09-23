"""
generate_dataset.py  —  Dataset generator for the agentic UE resource manager
------------------------------------------------------------------------------
Reflects the two-output optimisation formulation with three mobility
strategies for generating physically grounded η trajectories.

  Inputs  (state space  s = (β, α, η, γ))
  ─────────────────────────────────────────
    β   battery level       [1–100 %]
    α   app type            {podcast, system update, 4K video, game, AR/VR}
    η   network condition   {stable, congested, weak signal}
    γ   energy credit       [0–1]  normalised fraction of operator budget Γ

  Outputs  (action space  a = (o, e))
  ────────────────────────────────────
    o   offload_decision    {True, False}
    e   escalation_flag     {True, False}

  Hard constraints
  ────────────────
    C1   β ≤ 20%            →  o = False
    C8   η = "weak signal"  →  o = False, e = False (uplink unavailable)
    C1 ∩ C8                 →  o = False, e = False

  Escalation conditions  (e = True when uplink available AND any of):
    1. γ_min < γ ≤ γ_esc                      credit running low
    2. 20 < β ≤ 25                             near C1 boundary
    3. heavy app + congested + 40≤β≤60         structural ambiguity
    4. credit_cost_epoch > (γ − γ_esc)         one epoch exhausts headroom
    5. post-handover recovery                  NEW — mobility-triggered

  η generation — three strategies
  ─────────────────────────────────
  This script uses three complementary approaches to generate η values:

  Strategy 1 — Direct episode extraction from ns-3 trace
    Real Gauss-Markov mobility traces from ns-3 (7-cell hexagonal grid,
    UE velocity 80 km/hr).  Episodes of T=30 steps extracted with stride=10.
    SINR mapped to η using data-calibrated thresholds:
      θ_stable = 10 dB  (from trace median/percentile analysis)
      θ_weak   =  2 dB  (from trace percentile analysis)
    CellID rounded to detect handover events.
    Source: radio_parameters_moving_clean.csv
    Allocation: ~5% of dataset ("mobility_real" stratum)

  Strategy 2 — Synthetic Gauss-Markov episodes
    Parameters fitted from the ns-3 trace:
      α_gm  = 0.873   (lag-1 SINR autocorrelation)
      v_bar = 9.82 dB  (mean SINR)
      σ_gm  = 6.16 dB  (noise std = trace_std × √(1 − α²))
    Same SINR → η mapping and handover detection as Strategy 1.
    Generates as many episodes as needed without being limited to
    the 450 steps of real data.
    Allocation: ~5% of dataset ("mobility_synthetic" stratum)

  Strategy 3 — i.i.d. stratified sampling (original method)
    Independent samples from each constraint region.  Retained alongside
    the mobility strategies because the model still needs to handle
    state combinations that don't arise from smooth trajectories.
    Allocation: ~90% of dataset (all other strata)

  Gauss-Markov parameter fitting (from ns-3 trace analysis)
  ──────────────────────────────────────────────────────────
    lag-1 SINR autocorr : 0.873  →  α_gm = 0.873
    SINR mean           : 9.82 dB
    SINR std            : 11.78 dB
    σ_gm = 11.78 × √(1 − 0.873²) = 6.16 dB

    These are cited as: "GM parameters fitted from ns-3 mobility traces
    of a single UE at 80 km/hr across a 7-cell hexagonal grid."

  η thresholds  (calibrated from ns-3 trace)
  ─────────────────────────────────────────
    θ_stable = 10 dB  →  η = "stable"      (33% of trace steps)
    θ_weak   =  2 dB  →  η = "congested"   (40% of trace steps)
                       →  η = "weak signal" (27% of trace steps)

  κ(η) validation from TBLER
  ───────────────────────────
    TBLER by η from ns-3 trace: stable=0.503, congested=0.580, weak=0.682
    Ratios 1.0 : 1.15 : 1.36 confirm the κ ordering (1.0, 1.4, 2.5).
    κ_weak=2.5 is a conservative upper bound covering the tail of very
    degraded handover gaps.

  FLOPs-based burn rate  ρ(α, η) = F(α) × κ(η)  [GFLOP/s]
  ──────────────────────────────────────────────────────────
    F(α): podcast=1.0, update=0.1, video=15.0, game=120.0, AR/VR=900.0
    κ(η): stable=1.0, congested=1.4, weak signal=2.5
    Normalised by RHO_MAX = 2250 GFLOP/s (AR/VR, weak signal)

  Stratified sampling groups
  ───────────────────────────
    Group                   Target region                     Allocation
    ──────────────────────  ──────────────────────────────────  ──────────
    general                 No constraint, γ>γ_esc, β>30         30%
    c1_only                 β ≤ 20%, stable/congested            16%
    c8_only                 Weak signal, β > 20%                 11%
    c1_and_c8               β ≤ 20% AND weak signal               8%
    esc_zone                γ_min < γ ≤ γ_esc, usable uplink      8%
    boundary                20 < β ≤ 25                           7%
    low_confidence_proxy    Structurally ambiguous states          7%
    mobility_real           ns-3 trace episodes (T=30, stride=10) 5%
    mobility_synthetic      Synthetic GM episodes                  8%

Usage
─────
  python generate_dataset.py                    # 800 samples
  python generate_dataset.py --n 2000          # custom count
  python generate_dataset.py --trace path.csv  # custom trace file
  python generate_dataset.py --no-meta         # omit metadata
"""

import json
import math
import random
import argparse
import os

# ── Seed ──────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)

# ── Credit constants ──────────────────────────────────────────────────────────
GAMMA_MIN = 0.10
GAMMA_ESC = 0.20
PSI_0     = 1.0 / 3600.0
DELTA_T   = 1.0

# ── κ(η) network condition multipliers ───────────────────────────────────────
KAPPA = {"stable": 1.0, "congested": 1.4, "weak signal": 2.5}
RHO_MAX = 2250.0   # AR/VR, weak signal [GFLOP/s]

# ── SINR → η thresholds  (calibrated from ns-3 trace) ────────────────────────
THETA_STABLE = 10.0   # dB  —  SINR > 10 dB  →  stable
THETA_WEAK   =  2.0   # dB  —  SINR ≤  2 dB  →  weak signal
                      #        2 < SINR ≤ 10  →  congested

# ── Gauss-Markov parameters  (fitted from ns-3 trace, lag-1 autocorr=0.873) ──
GM_ALPHA  = 0.873    # memory parameter
GM_V_BAR  = 9.82    # mean SINR [dB]
GM_SIGMA  = 6.16    # noise std [dB]  = 11.78 × √(1 − 0.873²)

# ── Episode parameters ────────────────────────────────────────────────────────
EPISODE_T      = 30    # steps per episode
EPISODE_STRIDE = 10    # stride for real trace extraction

# ── Battery drain rate by η (% per epoch) ────────────────────────────────────
# At cell edge the UE raises TX power, draining battery faster.
BETA_DRAIN = {"stable": 0.10, "congested": 0.15, "weak signal": 0.25}

# ── App definitions ───────────────────────────────────────────────────────────
app_types = [
    {"name": "background podcast",        "compute": "low",       "F_alpha_gflops":   1.0},
    {"name": "system update",             "compute": "low",       "F_alpha_gflops":   0.1},
    {"name": "4K video stream",           "compute": "medium",    "F_alpha_gflops":  15.0},
    {"name": "mobile MOBA game",          "compute": "high",      "F_alpha_gflops": 120.0},
    {"name": "AR/VR headset application", "compute": "very_high", "F_alpha_gflops": 900.0},
]
network_conditions = ["stable", "congested", "weak signal"]
HEAVY_APPS = {"mobile MOBA game", "AR/VR headset application"}

# Default trace path — can be overridden via --trace argument
DEFAULT_TRACE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "radio_parameters_moving_clean.csv"
)


# ═══════════════════════════════════════════════════════════════════════════════
# Credit helpers
# ═══════════════════════════════════════════════════════════════════════════════

def burn_rate(app, condition):
    """ρ(α,η) = F(α) × κ(η) / RHO_MAX — normalised to [0,1]."""
    return (app["F_alpha_gflops"] * KAPPA[condition]) / RHO_MAX

def credit_cost(app, condition):
    return burn_rate(app, condition) * DELTA_T

def simulate_credit(gamma, offload, app, condition):
    cost = credit_cost(app, condition) if offload else 0.0
    return max(0.0, min(1.0, gamma - cost + PSI_0 * DELTA_T))


# ═══════════════════════════════════════════════════════════════════════════════
# Constraint checkers
# ═══════════════════════════════════════════════════════════════════════════════

def c1_active(battery):
    return battery <= 20

def c8_active(condition):
    return condition == "weak signal"

def uplink_available(condition):
    return condition != "weak signal"

def offload_feasible(battery, condition, gamma):
    return not c1_active(battery) and not c8_active(condition) and gamma > GAMMA_MIN


# ═══════════════════════════════════════════════════════════════════════════════
# Escalation condition checkers
# ═══════════════════════════════════════════════════════════════════════════════

def in_credit_escalation_zone(gamma):
    return GAMMA_MIN < gamma <= GAMMA_ESC

def near_c1_boundary(battery):
    return 20 < battery <= 25

def congested_heavy_mid_battery(app, condition, battery):
    return (app["name"] in HEAVY_APPS
            and condition == "congested"
            and 40 <= battery <= 60)

def single_epoch_exhausts_headroom(app, condition, gamma):
    return (gamma > GAMMA_ESC
            and credit_cost(app, condition) > (gamma - GAMMA_ESC))

def post_handover(eta_history, window=2):
    """
    Condition 5  —  NEW mobility-triggered escalation.

    True when a handover completed within the last `window` decision epochs:
    η was "weak signal" recently and has now recovered to stable/congested.

    After a handover the UE has re-attached to a new serving cell.  The
    local model's cached policy is conditioned on its training distribution —
    it has no knowledge of the new cell's load, interference profile, or
    slice configuration.  Escalating to the network LLM gives access to
    fresh global cell-level context.

    Requires eta_history (list of recent η values, oldest first) —
    only available in trajectory-based samples, not i.i.d. samples.
    """
    if not eta_history or len(eta_history) < 2:
        return False
    recently_weak  = any(h == "weak signal" for h in eta_history[-window-1:-1])
    currently_ok   = eta_history[-1] in ("stable", "congested")
    return recently_weak and currently_ok


def should_escalate(battery, app, condition, gamma, eta_history=None):
    """
    Master escalation rule — five structural conditions.
    e = True when uplink is available AND any condition holds.
    """
    if not uplink_available(condition):
        return False
    return (
        in_credit_escalation_zone(gamma)
        or near_c1_boundary(battery)
        or congested_heavy_mid_battery(app, condition, battery)
        or single_epoch_exhausts_headroom(app, condition, gamma)
        or post_handover(eta_history or [])
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Optimal decision rules
# ═══════════════════════════════════════════════════════════════════════════════

def determine_optimal_decision(battery, app, condition, gamma, eta_history=None):
    """Return (offload, escalate) following the two-level constraint hierarchy."""
    c1 = c1_active(battery)
    c8 = c8_active(condition)

    if c1 and c8:
        return False, False
    if c1:
        return False, (near_c1_boundary(battery) and uplink_available(condition))
    if c8:
        return False, False
    if gamma <= GAMMA_MIN:
        return False, False
    if gamma <= GAMMA_ESC:
        return False, True
    if battery <= 30:
        return False, should_escalate(battery, app, condition, gamma, eta_history)
    if app["compute"] in ("high", "very_high"):
        return True, should_escalate(battery, app, condition, gamma, eta_history)
    return False, should_escalate(battery, app, condition, gamma, eta_history)


# ═══════════════════════════════════════════════════════════════════════════════
# SINR → η  mapping  (calibrated from ns-3 trace)
# ═══════════════════════════════════════════════════════════════════════════════

def sinr_to_eta(sinr_db):
    """
    Map SINR [dB] to discrete η using data-calibrated thresholds.

    Thresholds calibrated from ns-3 trace analysis:
      θ_stable = 10 dB  (P50 of trace SINR is 4.6 dB; 10 dB gives 33% stable)
      θ_weak   =  2 dB  (P10 = 0.94 dB; 2 dB gives 27% weak — matches HO gap freq)

    Reference: 3GPP TR 38.901 — SINR operating ranges for urban macro.
    """
    if sinr_db > THETA_STABLE:
        return "stable"
    elif sinr_db > THETA_WEAK:
        return "congested"
    else:
        return "weak signal"


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 1 — Real ns-3 trace episode extraction
# ═══════════════════════════════════════════════════════════════════════════════

def load_trace(trace_path):
    """
    Load the ns-3 mobility trace CSV and return a cleaned DataFrame.

    Fixes applied:
      - CellID_RSRP rounded to nearest integer (ns-3 averaging artefact)
      - Single TBLER > 1.0 value clipped to 1.0
    """
    try:
        import pandas as pd
        df = pd.read_csv(trace_path)
        df["CellID_int"] = df["CellID_RSRP"].round().astype(int)
        df["TBLER"]      = df["TBLER"].clip(upper=1.0)
        return df
    except Exception as e:
        print(f"  [WARN] Could not load trace: {e}")
        return None


def extract_real_episodes(trace_df, T=EPISODE_T, stride=EPISODE_STRIDE):
    """
    Extract episodes of length T from the ns-3 SINR trace with given stride.

    Each episode is a list of η values derived from the SINR column, with
    handover events (CellID change) annotated.

    Returns
    ───────
    list of dicts:
      sinr_seq   : list[float]  — raw SINR values [dB]
      eta_seq    : list[str]    — mapped η values
      ho_steps   : set[int]     — indices within episode where handover occurred
    """
    episodes = []
    n = len(trace_df)
    for start in range(0, n - T + 1, stride):
        sub = trace_df.iloc[start:start + T].reset_index(drop=True)
        sinr_seq = sub["SINR"].tolist()
        eta_seq  = [sinr_to_eta(s) for s in sinr_seq]
        # Detect handovers: CellID changes within episode
        ho_steps = set()
        for i in range(1, T):
            if sub.loc[i, "CellID_int"] != sub.loc[i-1, "CellID_int"]:
                ho_steps.add(i)
        episodes.append({
            "sinr_seq": sinr_seq,
            "eta_seq":  eta_seq,
            "ho_steps": ho_steps,
            "source":   "real",
        })
    return episodes


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy 2 — Synthetic Gauss-Markov episodes
# ═══════════════════════════════════════════════════════════════════════════════

def generate_gm_sinr(T, alpha=GM_ALPHA, v_bar=GM_V_BAR, sigma=GM_SIGMA,
                     v0=None):
    """
    Generate one SINR trajectory of length T using the 1D Gauss-Markov model.

    v(t+1) = α·v(t) + (1−α)·v̄ + √(1−α²)·σ·N(0,1)

    Parameters fitted from ns-3 trace:
      α = 0.873  (lag-1 autocorrelation of SINR)
      v̄ = 9.82 dB
      σ = 6.16 dB  = 11.78 × √(1 − 0.873²)

    Returns list[float] of SINR values [dB], all ≥ 0.
    """
    noise_scale = math.sqrt(1.0 - alpha ** 2) * sigma
    v = v0 if v0 is not None else random.gauss(v_bar, sigma)
    seq = []
    for _ in range(T):
        v = alpha * v + (1.0 - alpha) * v_bar + noise_scale * random.gauss(0, 1)
        seq.append(max(0.0, v))
    return seq


def generate_synthetic_episode(T=EPISODE_T):
    """
    Generate one synthetic Gauss-Markov episode.

    Handovers are detected as SINR dips below THETA_WEAK followed by
    recovery above THETA_WEAK — a proxy for the handover gap condition
    visible in the real trace.
    """
    sinr_seq = generate_gm_sinr(T)
    eta_seq  = [sinr_to_eta(s) for s in sinr_seq]

    # Detect pseudo-handovers: transitions from weak signal back to
    # stable/congested (mirrors real trace handover pattern)
    ho_steps = set()
    for i in range(1, T):
        if eta_seq[i-1] == "weak signal" and eta_seq[i] != "weak signal":
            ho_steps.add(i)

    return {
        "sinr_seq": sinr_seq,
        "eta_seq":  eta_seq,
        "ho_steps": ho_steps,
        "source":   "synthetic",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Episode → samples
# ═══════════════════════════════════════════════════════════════════════════════

def episode_to_samples(episode, app, stratum):
    """
    Convert one trajectory episode into T individual training samples.

    Each sample sees only the current state (β, α, η, γ) and the η history
    up to the current step — mimicking what the UE agent observes at runtime.

    Battery evolves with mobility-aware drain (higher at cell edge).
    Credit carries forward from step to step via eq. 7.
    App type is fixed for the episode (session continuity).
    """
    eta_seq  = episode["eta_seq"]
    ho_steps = episode["ho_steps"]
    T        = len(eta_seq)

    # Initial state: random battery (31–100 to avoid C1 at episode start)
    # and random credit above escalation zone
    battery = random.randint(31, 100)
    gamma   = round(random.uniform(GAMMA_ESC + 0.05, 1.0), 3)

    samples  = []
    eta_hist = []

    for t in range(T):
        condition = eta_seq[t]
        eta_hist.append(condition)

        # Determine optimal decision with full η history
        offload, escalate = determine_optimal_decision(
            battery, app, condition, gamma, eta_history=eta_hist
        )

        # Build sample
        sample = {
            "prompt":   build_prompt(battery, app, condition, gamma),
            "response": json.dumps(
                build_response(offload, escalate, battery, app, condition,
                               gamma, eta_history=eta_hist)
            ),
            "meta": {
                "battery":   battery,
                "app":       app["name"],
                "condition": condition,
                "gamma":     round(gamma, 4),
                "stratum":   stratum,
                "t":         t,
                "sinr":      round(episode["sinr_seq"][t], 2),
                "handover":  t in ho_steps,
                "source":    episode["source"],
            },
        }
        samples.append(sample)

        # Advance state for next step
        # Battery drains: base metabolic + radio TX overhead by η
        battery = max(1, battery - BETA_DRAIN.get(condition, 0.10))
        battery = int(round(battery))
        gamma   = simulate_credit(gamma, offload, app, condition)

    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_prompt(battery, app, condition, gamma):
    credit_pct = round(gamma * 100)
    if gamma >= 0.8:
        credit_desc = "full"
    elif gamma >= 0.5:
        credit_desc = "moderate"
    elif gamma > GAMMA_ESC:
        credit_desc = "low"
    elif gamma > GAMMA_MIN:
        credit_desc = "critically low"
    else:
        credit_desc = "depleted"
    return (
        f"Battery is at {battery}%. "
        f"User just opened a {app['name']}. "
        f"Network is {condition}. "
        f"Energy credit is {credit_desc} ({credit_pct}%)."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Response builder
# ═══════════════════════════════════════════════════════════════════════════════

def _escalation_reason(escalate, battery, app, condition, gamma, eta_history):
    if not escalate:
        return "none"
    if c1_active(battery) and not c8_active(condition):
        return "near_c1_boundary"
    if in_credit_escalation_zone(gamma):
        return "credit_escalation_zone"
    if post_handover(eta_history or []):
        return "post_handover"
    if congested_heavy_mid_battery(app, condition, battery):
        return "congested_heavy_mid_battery"
    if single_epoch_exhausts_headroom(app, condition, gamma):
        return "single_epoch_exhausts_headroom"
    return "structural_ambiguity"


def build_response(offload, escalate, battery, app, condition, gamma,
                   eta_history=None):
    return {
        "offload_decision":   offload,
        "escalation_flag":    escalate,
        "burn_rate_gflops_s": round(app["F_alpha_gflops"] * KAPPA[condition], 2),
        "credit_cost_epoch":  round(credit_cost(app, condition), 6),
        "c1_active":          c1_active(battery),
        "c8_active":          c8_active(condition),
        "offload_feasible":   offload_feasible(battery, condition, gamma),
        "escalation_reason":  _escalation_reason(
            escalate, battery, app, condition, gamma, eta_history
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# i.i.d. stratified sample generator  (Strategy 3)
# ═══════════════════════════════════════════════════════════════════════════════

def make_iid_sample(battery, app, condition, gamma, stratum):
    offload, escalate = determine_optimal_decision(battery, app, condition, gamma)
    return {
        "prompt":   build_prompt(battery, app, condition, gamma),
        "response": json.dumps(
            build_response(offload, escalate, battery, app, condition, gamma)
        ),
        "meta": {
            "battery":   battery,
            "app":       app["name"],
            "condition": condition,
            "gamma":     gamma,
            "stratum":   stratum,
            "source":    "iid",
        },
    }


def generate_iid_strata(num_samples):
    """Generate all i.i.d. stratified samples (Strategies 3)."""
    # Use explicit int() for each stratum then assign the rounding remainder
    # to "general" so the total always equals num_samples exactly.
    n_c1    = int(num_samples * 0.16)
    n_c8    = int(num_samples * 0.11)
    n_c18   = int(num_samples * 0.08)
    n_esc   = int(num_samples * 0.08)
    n_bnd   = int(num_samples * 0.07)
    n_lcp   = int(num_samples * 0.07)
    n_gen   = num_samples - n_c1 - n_c8 - n_c18 - n_esc - n_bnd - n_lcp
    strata = {
        "general":              n_gen,
        "c1_only":              n_c1,
        "c8_only":              n_c8,
        "c1_and_c8":            n_c18,
        "esc_zone":             n_esc,
        "boundary":             n_bnd,
        "low_confidence_proxy": n_lcp,
    }

    def sg(low=GAMMA_ESC+0.01, high=1.0):
        return round(random.uniform(low, high), 3)

    def sg_esc():
        return round(random.uniform(GAMMA_MIN+0.001, GAMMA_ESC), 3)

    dataset = []

    for _ in range(strata["general"]):
        dataset.append(make_iid_sample(
            random.randint(31, 100), random.choice(app_types),
            random.choice(["stable","congested"]), sg(GAMMA_ESC+0.05,1.0),
            "general"))

    for _ in range(strata["c1_only"]):
        dataset.append(make_iid_sample(
            random.randint(1, 20), random.choice(app_types),
            random.choice(["stable","congested"]), sg(0.05,1.0),
            "c1_only"))

    for _ in range(strata["c8_only"]):
        dataset.append(make_iid_sample(
            random.randint(21, 100), random.choice(app_types),
            "weak signal", sg(0.05,1.0),
            "c8_only"))

    for _ in range(strata["c1_and_c8"]):
        dataset.append(make_iid_sample(
            random.randint(1, 20), random.choice(app_types),
            "weak signal", sg(0.05,1.0),
            "c1_and_c8"))

    for _ in range(strata["esc_zone"]):
        dataset.append(make_iid_sample(
            random.randint(21, 100), random.choice(app_types),
            random.choice(["stable","congested"]), sg_esc(),
            "esc_zone"))

    for _ in range(strata["boundary"]):
        dataset.append(make_iid_sample(
            random.randint(21, 25), random.choice(app_types),
            random.choice(network_conditions), sg(0.05,1.0),
            "boundary"))

    # low_confidence_proxy — conditions 3 and 4
    heavy_apps = [a for a in app_types if a["compute"] in ("high","very_high")]
    lcp_n = strata["low_confidence_proxy"]
    half  = lcp_n // 2
    arvr  = next(a for a in app_types if a["compute"]=="very_high")

    for _ in range(half):
        app  = random.choice(heavy_apps)
        dataset.append(make_iid_sample(
            random.randint(40,60), app, "congested",
            sg(GAMMA_ESC+0.05,1.0), "low_confidence_proxy"))

    for _ in range(lcp_n - half):
        cond = random.choice(["stable","congested"])
        cpe  = credit_cost(arvr, cond)
        glo  = GAMMA_ESC + 0.001
        ghi  = min(1.0, GAMMA_ESC + cpe)
        if ghi <= glo: ghi = glo + 0.01
        dataset.append(make_iid_sample(
            random.randint(31,100), arvr, cond,
            round(random.uniform(glo,ghi),4), "low_confidence_proxy"))

    return dataset, strata


# ═══════════════════════════════════════════════════════════════════════════════
# Master dataset generator
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dataset(num_samples=800, trace_path=None):
    """
    Generate a dataset combining all three η strategies.

    Allocation:
      ~5%  mobility_real       (real ns-3 episodes)
      ~8%  mobility_synthetic  (synthetic GM episodes)
      ~87% i.i.d. strata       (original stratified sampling)
    """
    n_real      = int(num_samples * 0.05)
    n_synthetic = int(num_samples * 0.08)
    n_iid       = num_samples - n_real - n_synthetic

    dataset = []

    # ── Strategy 1: real ns-3 trace episodes ──────────────────────────────────
    trace_df = None
    if trace_path and os.path.exists(trace_path):
        trace_df = load_trace(trace_path)

    if trace_df is not None:
        real_episodes = extract_real_episodes(trace_df, T=EPISODE_T,
                                              stride=EPISODE_STRIDE)
        print(f"  Real episodes extracted   : {len(real_episodes)}"
              f"  (T={EPISODE_T}, stride={EPISODE_STRIDE})")

        # Collect samples from episodes, stopping at n_real
        real_samples = []
        for ep in real_episodes:
            app = random.choice(app_types)
            real_samples.extend(episode_to_samples(ep, app, "mobility_real"))
        # Sample n_real from all generated real episode samples
        random.shuffle(real_samples)
        dataset.extend(real_samples[:n_real])
        print(f"  Real mobility samples     : {min(n_real,len(real_samples))}"
              f"  (target {n_real})")
    else:
        print(f"  [INFO] No trace file — skipping real episodes, "
              f"replacing with synthetic.")
        n_synthetic += n_real
        n_real = 0

    # ── Strategy 2: synthetic Gauss-Markov episodes ───────────────────────────
    n_episodes_needed = math.ceil(n_synthetic / EPISODE_T) + 1
    syn_samples = []
    for _ in range(n_episodes_needed):
        app = random.choice(app_types)
        ep  = generate_synthetic_episode(T=EPISODE_T)
        syn_samples.extend(episode_to_samples(ep, app, "mobility_synthetic"))

    random.shuffle(syn_samples)
    dataset.extend(syn_samples[:n_synthetic])
    print(f"  Synthetic GM samples      : {n_synthetic}")

    # ── Strategy 3: i.i.d. stratified samples ────────────────────────────────
    iid_samples, strata = generate_iid_strata(n_iid)
    dataset.extend(iid_samples)
    print(f"  i.i.d. stratified samples : {len(iid_samples)}")

    random.shuffle(dataset)
    return dataset


# ═══════════════════════════════════════════════════════════════════════════════
# Statistics
# ═══════════════════════════════════════════════════════════════════════════════

def print_stats(dataset):
    n = len(dataset)

    def count(key, val=True):
        return sum(1 for d in dataset if json.loads(d["response"]).get(key)==val)

    offload_true  = count("offload_decision", True)
    escalate_true = count("escalation_flag",  True)
    c1_count      = count("c1_active",        True)
    c8_count      = count("c8_active",        True)

    strata_counts  = {}
    esc_reasons    = {}
    source_counts  = {}
    handover_count = 0

    for d in dataset:
        m = d.get("meta", {})
        s = m.get("stratum","unknown")
        strata_counts[s] = strata_counts.get(s, 0) + 1
        src = m.get("source","iid")
        source_counts[src] = source_counts.get(src, 0) + 1
        if m.get("handover", False):
            handover_count += 1
        r = json.loads(d["response"])
        reason = r.get("escalation_reason","none")
        esc_reasons[reason] = esc_reasons.get(reason, 0) + 1

    c1c8_esc = sum(
        1 for d in dataset
        if json.loads(d["response"])["c1_active"]
        and json.loads(d["response"])["c8_active"]
        and json.loads(d["response"])["escalation_flag"]
    )

    print(f"\n  Dataset statistics ({n} samples)")
    print(f"  {'─'*56}")
    print(f"  offload_decision = True  : {offload_true:>5}  ({100*offload_true/n:.1f}%)")
    print(f"  escalation_flag  = True  : {escalate_true:>5}  ({100*escalate_true/n:.1f}%)")
    print(f"  C1 active (β ≤ 20%)     : {c1_count:>5}  ({100*c1_count/n:.1f}%)")
    print(f"  C8 active (weak signal) : {c8_count:>5}  ({100*c8_count/n:.1f}%)")
    print(f"  Handover steps          : {handover_count:>5}  ({100*handover_count/n:.1f}%)")
    print(f"  C1∩C8 escalation (=0)   : {c1c8_esc}")

    print(f"\n  Source breakdown:")
    for src, cnt in sorted(source_counts.items()):
        print(f"    {src:<24}: {cnt:>5}  ({100*cnt/n:.1f}%)")

    print(f"\n  Sampling group breakdown:")
    for stratum, cnt in sorted(strata_counts.items()):
        print(f"    {stratum:<28}: {cnt:>5}  ({100*cnt/n:.1f}%)")

    print(f"\n  Escalation reason breakdown:")
    for reason, cnt in sorted(esc_reasons.items(), key=lambda x:-x[1]):
        print(f"    {reason:<36}: {cnt:>5}  ({100*cnt/n:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate UE resource management dataset with mobility"
    )
    parser.add_argument("--n",       type=int, default=800)
    parser.add_argument("--out",     default="train_data.jsonl")
    parser.add_argument("--trace",   default=DEFAULT_TRACE,
                        help="Path to ns-3 mobility trace CSV")
    parser.add_argument("--no-meta", action="store_true")
    args = parser.parse_args()

    print(f"Generating {args.n} samples …")
    print(f"  Trace: {args.trace}")
    data = generate_dataset(args.n, trace_path=args.trace)
    print_stats(data)

    with open(args.out, "w", encoding="utf-8") as f:
        for item in data:
            record = {"prompt": item["prompt"], "response": item["response"]}
            if not args.no_meta:
                record["meta"] = item["meta"]
            f.write(json.dumps(record) + "\n")

    print(f"\n  Saved {len(data)} samples → {args.out}")
    print(f"  Next step: python run_sft.py")