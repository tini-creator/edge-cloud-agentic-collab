"""
evaluate_robustness.py  —  Robustness evaluation for the UE AI Agent
---------------------------------------------------------------------
Evaluates the GRPO step-800 checkpoint against five baselines across
three robustness axes, producing a results JSON and a printed report.

Baselines
─────────
  random          Uniformly random valid (o,e) pair — lower bound
  always_offload  Always (o=True,e=False) when feasible, else (False,False)
  always_escalate Always (o=False,e=True) when uplink available, else (False,False)
  sft             SFT-only model  (models/sft-ue-merged)
  few_shot        Untuned SmolLM2-135M with k-shot examples in the prompt
  exhaustive      Deterministic ground-truth policy — theoretical upper bound
  grpo            GRPO checkpoint-800 — the model being evaluated

Robustness axes
───────────────
  Axis 1 — OOD generalisation (state-space coverage)
    1a  Novel app names        same compute tier, different surface form
    1b  Prose battery          "battery is nearly dead" instead of "Battery is at 9%"
    1c  Prose credit           "offload budget nearly exhausted" instead of standard
    1d  Reordered prompt       network condition stated before battery
    1e  Compound edge cases    multiple boundary conditions simultaneously

  Axis 3 — Credit level sensitivity
    3a  γ sweep (1%–100%)     escalation_flag vs γ at fixed β=50, stable, AR/VR
    3b  Credit-app interaction same γ=25% for light (podcast) vs heavy (AR/VR)

  Axis 6 — Linguistic robustness
    6a  Synonym substitution   "launched" / "started" / "is running" instead of "opened"
    6b  Formal condensed       dense formal prose instead of templated format
    6c  Passive voice          "an AR/VR session is active" instead of "User just opened"

Metrics reported per axis per baseline
───────────────────────────────────────
  accuracy      both decisions correct
  cvr           constraint violation rate (all types combined)
  offload_f1    offload precision/recall/F1
  esc_f1        escalation precision/recall/F1
  mnu           mean normalised utility from reward_model.py
  parse_fail    fraction of unparseable outputs

Output
──────
  models/robustness_results.json   — full per-axis per-baseline results
  models/robustness_report.txt     — formatted comparison table

Usage
─────
  python scripts/evaluate_robustness.py
  python scripts/evaluate_robustness.py --grpo-checkpoint models/grpo-ue-agent/checkpoint-800
  python scripts/evaluate_robustness.py --baselines random exhaustive sft grpo
  python scripts/evaluate_robustness.py --few-shot-k 3
  python scripts/evaluate_robustness.py --quiet
  python scripts/evaluate_robustness.py --axes 1a 1b 3a 6a
"""

import os
import sys
import json
import random
import argparse
import itertools

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SFT_MODEL  = os.path.join(ROOT, "models", "sft-ue-merged")
GRPO_CKPT  = os.path.join(ROOT, "models", "grpo-ue-agent", "checkpoint-800")
OUT_JSON   = os.path.join(ROOT, "models", "robustness_results.json")
OUT_REPORT = os.path.join(ROOT, "models", "robustness_report.txt")

sys.path.insert(0, ROOT)

# ── Shared imports from existing scripts ─────────────────────────────────────
from scripts.reward_model import (
    _repair_json, dict_raise_on_duplicates, calculate_ue_reward,
    GAMMA_MIN, GAMMA_ESC,
)
from data.generate_dataset import (
    app_types, network_conditions,
    determine_optimal_decision,
    c1_active, c8_active, offload_feasible,
    KAPPA,
)
import importlib.util as _ilu
# Locate evaluate.py — same directory as this script
_ev_path = os.path.join(SCRIPT_DIR, "evaluate.py")
if not os.path.exists(_ev_path):
    # Fallback for when scripts/ layout differs
    for _candidate in [
        os.path.join(ROOT, "scripts", "evaluate.py"),
        os.path.join(os.path.dirname(SCRIPT_DIR), "scripts", "evaluate.py"),
    ]:
        if os.path.exists(_candidate):
            _ev_path = _candidate
            break
_ev_spec = _ilu.spec_from_file_location("evaluate", _ev_path)
_ev = _ilu.module_from_spec(_ev_spec)
_ev_spec.loader.exec_module(_ev)

# Re-export helpers from evaluate.py
violates_c1          = _ev.violates_c1
violates_c8          = _ev.violates_c8
violates_credit_floor= _ev.violates_credit_floor
violates_any         = _ev.violates_any
score_sample         = _ev.score_sample
generate_config      = _ev.generate_config
load_model           = _ev.load_model
safe_div             = _ev.safe_div
build_eval_prompt    = _ev.build_eval_prompt
_credit_desc         = _ev._credit_desc
_sg                  = _ev._sg

# Lookup: app key → index in app_types
APP_BY_NAME = {a["name"]: a for a in app_types}


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  BASELINES
# ═══════════════════════════════════════════════════════════════════════════════

def random_pred():
    """Uniform random valid (o,e) — never (1,1)."""
    pairs = [(False, False), (True, False), (False, True)]
    o, e = random.choice(pairs)
    return {"offload_decision": o, "escalation_flag": e}


def always_offload_pred(battery, condition, gamma):
    """Offload whenever feasible, otherwise do nothing."""
    if offload_feasible(battery, condition, gamma):
        return {"offload_decision": True,  "escalation_flag": False}
    return     {"offload_decision": False, "escalation_flag": False}


def always_escalate_pred(condition):
    """Escalate whenever uplink is available, otherwise do nothing."""
    if condition != "weak signal":
        return {"offload_decision": False, "escalation_flag": True}
    return     {"offload_decision": False, "escalation_flag": False}


def exhaustive_pred(battery, app, condition, gamma):
    """Ground-truth policy from generate_dataset.py — theoretical ceiling."""
    o, e = determine_optimal_decision(battery, app, condition, gamma)
    return {"offload_decision": o, "escalation_flag": e}


def _parse_output(raw_text):
    """Parse model output to dict or None."""
    raw = raw_text.strip()
    if not raw.startswith("{"):
        raw = "{" + raw
    try:
        return json.loads(_repair_json(raw),
                          object_pairs_hook=dict_raise_on_duplicates)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  FEW-SHOT BASELINE
# ═══════════════════════════════════════════════════════════════════════════════

# Fixed few-shot examples covering the main constraint regions.
# These are ground-truth (o,e) pairs with diverse states — chosen so that
# the model sees examples of each constraint region before the test prompt.
FEW_SHOT_EXAMPLES = [
    # C1: critical battery — never offload
    {
        "prompt": ("Battery is at 8%. User just opened a mobile MOBA game. "
                   "Network is stable. Energy credit is full (90%)."),
        "response": {"offload_decision": False, "escalation_flag": False},
        "rationale": "Battery below 20% — C1 prohibits offloading.",
    },
    # C8: weak signal — never offload or escalate
    {
        "prompt": ("Battery is at 75%. User just opened a 4K video stream. "
                   "Network is weak signal. Energy credit is moderate (60%)."),
        "response": {"offload_decision": False, "escalation_flag": False},
        "rationale": "Weak signal — uplink unavailable, cannot offload or escalate.",
    },
    # Credit escalation zone
    {
        "prompt": ("Battery is at 65%. User just opened a AR/VR headset application. "
                   "Network is stable. Energy credit is critically low (15%)."),
        "response": {"offload_decision": False, "escalation_flag": True},
        "rationale": "Credits critically low — escalate to network LLM for guidance.",
    },
    # Heavy app, full credits, stable — offload
    {
        "prompt": ("Battery is at 80%. User just opened a AR/VR headset application. "
                   "Network is stable. Energy credit is full (95%)."),
        "response": {"offload_decision": True, "escalation_flag": False},
        "rationale": "Heavy compute, good conditions, full credits — offload.",
    },
    # Light app — execute locally
    {
        "prompt": ("Battery is at 55%. User just opened a background podcast. "
                   "Network is congested. Energy credit is moderate (70%)."),
        "response": {"offload_decision": False, "escalation_flag": False},
        "rationale": "Light app — local execution is efficient.",
    },
]


def build_few_shot_prompt(test_prompt, k=3, examples=None):
    """
    Build a k-shot prompt by prepending k examples before the test query.

    Format mirrors the training format exactly so the model sees the same
    JSON output structure it would produce after SFT/GRPO fine-tuning.
    The examples are chosen to cover the main constraint regions so the
    model has in-context demonstrations of C1, C8, credit escalation, etc.
    """
    if examples is None:
        examples = FEW_SHOT_EXAMPLES
    chosen = examples[:k]

    parts = []
    for ex in chosen:
        parts.append(
            f"UE Context: {ex['prompt']}\n"
            f"Required keys (offload_decision, escalation_flag):\n"
            f"Decision:\n"
            f"{json.dumps(ex['response'])}"
        )
    # Append the test query with the standard JSON-prefill
    parts.append(
        f"UE Context: {test_prompt}\n"
        f"Required keys (offload_decision, escalation_flag):\n"
        f"Decision:\n{{"
    )
    return "\n\n".join(parts)


def few_shot_pred(model, tokenizer, test_prompt, k=3, max_new_tokens=40):
    """
    Run k-shot inference on the base SmolLM2-135M.
    The few-shot examples are prepended in-context; no weights are updated.
    """
    import torch
    full_prompt = build_few_shot_prompt(test_prompt, k=k)
    inputs = tokenizer(full_prompt, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = ids[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return _parse_output(raw)


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TEST-SET BUILDERS — Axes 1, 3, 6
# ═══════════════════════════════════════════════════════════════════════════════

# ── Axis 1: OOD generalisation ────────────────────────────────────────────────

def build_axis_1a(seed=101):
    """
    1a — Novel app names.
    Same compute tier as training apps but lexically different names.
    Tests whether the model generalised the compute-tier concept or
    memorised the specific app name strings.
    """
    random.seed(seed)
    # (display_name, backing_app_index, description)
    mappings = [
        ("Spotify music stream",              0, "low-compute audio"),
        ("Apple Music background playback",   0, "low-compute audio"),
        ("WhatsApp video call",               2, "medium-compute video"),
        ("Netflix 4K streaming",              2, "medium-compute video"),
        ("Call of Duty: Mobile",              3, "high-compute game"),
        ("Genshin Impact",                    3, "high-compute game"),
        ("PUBG Mobile",                       3, "high-compute game"),
        ("Meta Quest 3 VR session",           4, "very-high AR/VR"),
        ("Apple Vision Pro immersive app",    4, "very-high AR/VR"),
        ("Microsoft HoloLens AR workspace",   4, "very-high AR/VR"),
        ("background OS update",              1, "very-low system update"),
        ("silent app patch download",         1, "very-low system update"),
    ]
    samples = []
    for display, idx, desc in mappings:
        backing   = app_types[idx]
        battery   = random.randint(5, 95)
        condition = random.choice(network_conditions)
        gamma     = _sg(0.05, 1.0)
        o, e = determine_optimal_decision(battery, backing, condition, gamma)
        prompt = (
            f"Battery is at {battery}%. "
            f"User just opened {display}. "
            f"Network is {condition}. "
            f"Energy credit is {_credit_desc(gamma)} ({round(gamma*100)}%)."
        )
        samples.append({
            "prompt": prompt, "label": {"offload_decision": o, "escalation_flag": e},
            "battery": battery, "app": backing, "condition": condition,
            "gamma": gamma, "axis": "1a", "meta": desc,
        })
    return samples


def build_axis_1b(seed=102):
    """
    1b — Prose battery descriptions.
    Natural language descriptions of battery level instead of "Battery is at X%".
    Tests whether the model learned the numeric threshold semantics or
    just the "Battery is at N%" pattern.
    """
    random.seed(seed)
    # (prose description, actual battery %)
    prose_cases = [
        ("The phone battery is nearly dead",              9),
        ("Almost completely out of charge",               5),
        ("Battery critically low, may shut down soon",   14),
        ("Battery saver mode just activated",            18),
        ("Just plugged in the charger, at 22%",          22),
        ("Battery about a quarter full",                 25),
        ("Halfway charged",                              50),
        ("Battery doing fine",                           70),
        ("Fully charged",                                98),
        ("Battery just topped off",                      99),
        ("Running low, should charge soon",              28),
        ("Battery in good shape",                        82),
    ]
    samples = []
    for prose, battery in prose_cases:
        app       = random.choice(app_types)
        condition = random.choice(network_conditions)
        gamma     = _sg(0.05, 1.0)
        o, e = determine_optimal_decision(battery, app, condition, gamma)
        prompt = (
            f"{prose}. "
            f"User just opened a {app['name']}. "
            f"Network is {condition}. "
            f"Energy credit is {_credit_desc(gamma)} ({round(gamma*100)}%)."
        )
        samples.append({
            "prompt": prompt, "label": {"offload_decision": o, "escalation_flag": e},
            "battery": battery, "app": app, "condition": condition,
            "gamma": gamma, "axis": "1b", "meta": f"battery={battery}%",
        })
    return samples


def build_axis_1c(seed=103):
    """
    1c — Prose credit descriptors.
    OOD phrasing of energy credit level instead of the standard
    "Energy credit is <desc> (<pct>%)." format.
    Tests credit-level generalisation independent of keyword matching.
    """
    random.seed(seed)
    # (prose, actual_gamma)
    credit_cases = [
        ("the offload budget is almost exhausted",   0.12),
        ("nearly out of MEC credits",                0.08),
        ("credit account is running on fumes",       0.15),
        ("only a sliver of network credit left",     0.11),
        ("network quota almost used up",             0.18),
        ("credit balance is comfortable",            0.60),
        ("plenty of offload budget remaining",       0.85),
        ("fresh credit allocation just received",    0.98),
        ("credit at the threshold",                  0.21),
        ("just enough credit for one more session",  0.22),
        ("credit balance is zero",                   0.05),
        ("no offload credits left",                  0.03),
    ]
    samples = []
    for prose, gamma in credit_cases:
        battery   = random.randint(25, 90)
        app       = random.choice(app_types)
        condition = random.choice(["stable", "congested"])
        o, e = determine_optimal_decision(battery, app, condition, gamma)
        prompt = (
            f"Battery is at {battery}%. "
            f"User just opened a {app['name']}. "
            f"Network is {condition}. "
            f"Note: {prose}."
        )
        samples.append({
            "prompt": prompt, "label": {"offload_decision": o, "escalation_flag": e},
            "battery": battery, "app": app, "condition": condition,
            "gamma": gamma, "axis": "1c", "meta": f"gamma={gamma}",
        })
    return samples


def build_axis_1d(seed=104):
    """
    1d — Reordered prompt structure.
    The four state variables appear in different orders than training.
    Training order: battery, app, network, credit.
    Tests whether the model is robust to positional information
    or relies on a fixed template structure.
    """
    random.seed(seed)
    # Each tuple: (template_fn, description)
    # All four orderings are tested
    def order_netfirst(battery, app, condition, gamma):
        return (f"Network is {condition}. "
                f"Battery is at {battery}%. "
                f"User just opened a {app['name']}. "
                f"Energy credit is {_credit_desc(gamma)} ({round(gamma*100)}%).")

    def order_creditfirst(battery, app, condition, gamma):
        return (f"Energy credit is {_credit_desc(gamma)} ({round(gamma*100)}%). "
                f"Battery is at {battery}%. "
                f"Network is {condition}. "
                f"User just opened a {app['name']}.")

    def order_appfirst(battery, app, condition, gamma):
        return (f"User just opened a {app['name']}. "
                f"Battery is at {battery}%. "
                f"Network is {condition}. "
                f"Energy credit is {_credit_desc(gamma)} ({round(gamma*100)}%).")

    def order_reversed(battery, app, condition, gamma):
        return (f"Energy credit is {_credit_desc(gamma)} ({round(gamma*100)}%). "
                f"Network is {condition}. "
                f"User just opened a {app['name']}. "
                f"Battery is at {battery}%.")

    orderings = [
        (order_netfirst,    "network-first"),
        (order_creditfirst, "credit-first"),
        (order_appfirst,    "app-first"),
        (order_reversed,    "reversed"),
    ]
    samples = []
    # 5 samples per ordering = 20 total
    for order_fn, order_name in orderings:
        for _ in range(5):
            battery   = random.randint(5, 95)
            app       = random.choice(app_types)
            condition = random.choice(network_conditions)
            gamma     = _sg(0.05, 1.0)
            o, e = determine_optimal_decision(battery, app, condition, gamma)
            prompt = order_fn(battery, app, condition, gamma)
            samples.append({
                "prompt": prompt, "label": {"offload_decision": o, "escalation_flag": e},
                "battery": battery, "app": app, "condition": condition,
                "gamma": gamma, "axis": "1d", "meta": order_name,
            })
    return samples


def build_axis_1e(seed=105):
    """
    1e — Compound edge cases.
    Multiple boundary conditions simultaneously active.
    These states are structurally the hardest — they sit at the intersection
    of multiple constraint regions and were rare in training.
    """
    random.seed(seed)
    # (battery, condition, gamma, description)
    edge_cases = [
        # C1 boundary + credit escalation zone
        (21, "stable",    0.15, "just above C1 + credit esc zone"),
        (19, "stable",    0.15, "just below C1 + credit esc zone"),
        (20, "stable",    0.15, "exactly at C1 + credit esc zone"),
        # C1 boundary + congested network
        (21, "congested", 0.60, "just above C1 + congested"),
        (19, "congested", 0.60, "just below C1 + congested"),
        # credit floor boundary
        (50, "stable",    0.10, "exactly at credit floor"),
        (50, "stable",    0.09, "just below credit floor"),
        (50, "stable",    0.11, "just above credit floor"),
        # triple boundary: C1 + credit esc zone + congested
        (21, "congested", 0.15, "C1-boundary + esc-zone + congested"),
        (19, "congested", 0.15, "below-C1 + esc-zone + congested"),
        # C8 + credit escalation zone (escalation impossible due to C8)
        (50, "weak signal", 0.15, "C8 + credit esc zone — esc blocked"),
        # maximum stress: low battery + weak signal + low credit
        (12, "weak signal", 0.08, "C1∩C8 + credit floor — all prohibitions"),
    ]
    samples = []
    for battery, condition, gamma, desc in edge_cases:
        app = random.choice(app_types)
        o, e = determine_optimal_decision(battery, app, condition, gamma)
        prompt = build_eval_prompt(battery, app, condition, gamma)
        samples.append({
            "prompt": prompt, "label": {"offload_decision": o, "escalation_flag": e},
            "battery": battery, "app": app, "condition": condition,
            "gamma": gamma, "axis": "1e", "meta": desc,
        })
    return samples


# ── Axis 3: Credit sensitivity ────────────────────────────────────────────────

def build_axis_3a(seed=301):
    """
    3a — γ sweep from 1% to 100% in 5% steps.
    Fixed state: β=50%, AR/VR, stable network.
    Expected: escalation_flag=True for γ ∈ (10%, 20%], False elsewhere.
    Tests whether the model learned the credit threshold semantics.
    """
    random.seed(seed)
    battery   = 50
    app       = next(a for a in app_types if a["compute"] == "very_high")
    condition = "stable"
    samples   = []
    for pct in range(1, 101, 2):          # 1%, 3%, 5%, … 99%
        gamma = round(pct / 100, 2)
        o, e  = determine_optimal_decision(battery, app, condition, gamma)
        prompt = build_eval_prompt(battery, app, condition, gamma)
        samples.append({
            "prompt": prompt, "label": {"offload_decision": o, "escalation_flag": e},
            "battery": battery, "app": app, "condition": condition,
            "gamma": gamma, "axis": "3a", "meta": f"gamma={pct}%",
        })
    return samples


def build_axis_3b(seed=302):
    """
    3b — Credit-app interaction at γ=25% (just above γ_esc=20%).
    At this credit level, the heavy AR/VR app drains credits much faster
    than the light podcast, so the model should behave differently:
    AR/VR: credit_cost_epoch ≈ 0.40 >> (γ - γ_esc) = 0.05 → condition 4 fires
    Podcast: credit_cost_epoch ≈ 0.0004 << 0.05 → condition 4 doesn't fire
    Tests whether the model captures the asymmetric credit pressure.
    """
    random.seed(seed)
    battery   = 60
    condition = "stable"
    gamma     = 0.25     # just above γ_esc=0.20
    samples   = []
    for app in app_types:
        o, e = determine_optimal_decision(battery, app, condition, gamma)
        prompt = build_eval_prompt(battery, app, condition, gamma)
        samples.append({
            "prompt": prompt, "label": {"offload_decision": o, "escalation_flag": e},
            "battery": battery, "app": app, "condition": condition,
            "gamma": gamma, "axis": "3b", "meta": app["name"],
        })
    # Also test at γ=15% (in esc zone) — all apps should escalate
    gamma_esc = 0.15
    for app in app_types:
        o, e = determine_optimal_decision(battery, app, condition, gamma_esc)
        prompt = build_eval_prompt(battery, app, condition, gamma_esc)
        samples.append({
            "prompt": prompt, "label": {"offload_decision": o, "escalation_flag": e},
            "battery": battery, "app": app, "condition": condition,
            "gamma": gamma_esc, "axis": "3b", "meta": f"{app['name']} (esc-zone)",
        })
    return samples


# ── Axis 6: Linguistic robustness ─────────────────────────────────────────────

def build_axis_6a(seed=601):
    """
    6a — Synonym substitution.
    Replace "User just opened a" with semantically equivalent alternatives.
    The model should not change its decision based on this surface form.
    """
    random.seed(seed)
    templates = [
        "User just launched {app}.",
        "User has started {app}.",
        "{app} is now running.",
        "User is actively using {app}.",
        "User just switched to {app}.",
        "{app} was just opened by the user.",
        "The user has initiated {app}.",
    ]
    samples = []
    # 3 samples per template × 7 templates = 21 samples
    for tmpl in templates:
        for _ in range(3):
            battery   = random.randint(5, 95)
            app       = random.choice(app_types)
            condition = random.choice(network_conditions)
            gamma     = _sg(0.05, 1.0)
            o, e = determine_optimal_decision(battery, app, condition, gamma)
            app_phrase = f"a {app['name']}"
            prompt = (
                f"Battery is at {battery}%. "
                f"{tmpl.format(app=app_phrase)} "
                f"Network is {condition}. "
                f"Energy credit is {_credit_desc(gamma)} ({round(gamma*100)}%)."
            )
            samples.append({
                "prompt": prompt, "label": {"offload_decision": o, "escalation_flag": e},
                "battery": battery, "app": app, "condition": condition,
                "gamma": gamma, "axis": "6a",
                "meta": tmpl.split("{")[0].strip(),
            })
    return samples


def build_axis_6b(seed=602):
    """
    6b — Formal condensed register.
    All four state variables presented in dense formal prose rather than
    the four-sentence template used in training.
    """
    random.seed(seed)

    def formal_prompt(battery, app, condition, gamma):
        return (
            f"UE status: β={battery}%, α={app['name']}, "
            f"η={condition}, γ={round(gamma*100)}%. "
            f"Determine optimal resource action."
        )

    def technical_prompt(battery, app, condition, gamma):
        return (
            f"Device report — battery level: {battery}%, "
            f"active application: {app['name']}, "
            f"radio condition: {condition}, "
            f"remaining energy credit: {round(gamma*100)}%."
        )

    def clinical_prompt(battery, app, condition, gamma):
        cred_pct = round(gamma * 100)
        return (
            f"Battery: {battery}%. "
            f"Application: {app['name']}. "
            f"Network: {condition}. "
            f"Credit: {cred_pct}%."
        )

    builders = [
        (formal_prompt,    "formal_symbolic"),
        (technical_prompt, "technical_report"),
        (clinical_prompt,  "clinical_terse"),
    ]
    samples = []
    for builder_fn, style_name in builders:
        for _ in range(8):
            battery   = random.randint(5, 95)
            app       = random.choice(app_types)
            condition = random.choice(network_conditions)
            gamma     = _sg(0.05, 1.0)
            o, e = determine_optimal_decision(battery, app, condition, gamma)
            prompt = builder_fn(battery, app, condition, gamma)
            samples.append({
                "prompt": prompt, "label": {"offload_decision": o, "escalation_flag": e},
                "battery": battery, "app": app, "condition": condition,
                "gamma": gamma, "axis": "6b", "meta": style_name,
            })
    return samples


def build_axis_6c(seed=603):
    """
    6c — Passive voice and third-person descriptions.
    Replaces the active "User just opened X" with passive/third-person forms.
    """
    random.seed(seed)
    passive_templates = [
        "An {app} session is currently active.",
        "{app} has been launched on the device.",
        "The device is running {app}.",
        "{app} was opened moments ago.",
        "An {app} workload is being processed.",
        "The current foreground application is {app}.",
    ]
    samples = []
    for tmpl in passive_templates:
        for _ in range(4):
            battery   = random.randint(5, 95)
            app       = random.choice(app_types)
            condition = random.choice(network_conditions)
            gamma     = _sg(0.05, 1.0)
            o, e = determine_optimal_decision(battery, app, condition, gamma)
            app_phrase = f"a {app['name']}"
            prompt = (
                f"Battery is at {battery}%. "
                f"{tmpl.format(app=app_phrase)} "
                f"Network is {condition}. "
                f"Energy credit is {_credit_desc(gamma)} ({round(gamma*100)}%)."
            )
            samples.append({
                "prompt": prompt, "label": {"offload_decision": o, "escalation_flag": e},
                "battery": battery, "app": app, "condition": condition,
                "gamma": gamma, "axis": "6c",
                "meta": tmpl.split("{")[0].strip(),
            })
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PER-SAMPLE SCORER
# ═══════════════════════════════════════════════════════════════════════════════

def score(pred, sample):
    """Thin wrapper around evaluate.py score_sample."""
    return score_sample(
        pred,
        sample["label"],
        sample["battery"],
        sample["condition"],
        sample["gamma"],
        sample["prompt"],
    )


def aggregate(scored_list):
    """Aggregate a list of score_sample dicts into summary metrics."""
    n = len(scored_list)
    if n == 0:
        return {}

    def s(k):  return sum(r.get(k, 0) for r in scored_list)

    accuracy   = safe_div(s("both_correct"), n)
    cvr        = safe_div(s("viol_c1") + s("viol_c8") + s("viol_credit"), n)
    parse_fail = safe_div(s("parse_fail"), n)
    mnu        = sum(r.get("mnu", 0) for r in scored_list) / n

    feas   = [r for r in scored_list if r.get("feasible")]
    tp_o   = sum(r["offload_tp"] for r in feas)
    fp_o   = sum(r["offload_fp"] for r in feas)
    fn_o   = sum(r["offload_fn"] for r in feas)
    off_p  = safe_div(tp_o, tp_o + fp_o)
    off_r  = safe_div(tp_o, tp_o + fn_o)
    off_f1 = safe_div(2 * off_p * off_r, off_p + off_r)

    tp_e  = s("esc_tp"); fp_e = s("esc_fp"); fn_e = s("esc_fn")
    ver   = safe_div(tp_e, tp_e + fp_e)
    er    = safe_div(tp_e, tp_e + fn_e)
    esc_f1= safe_div(2 * ver * er, ver + er)

    return {
        "n":          n,
        "accuracy":   round(accuracy,   4),
        "cvr":        round(cvr,        4),
        "offload_p":  round(off_p,      4),
        "offload_r":  round(off_r,      4),
        "offload_f1": round(off_f1,     4),
        "ver":        round(ver,        4),
        "er":         round(er,         4),
        "esc_f1":     round(esc_f1,     4),
        "mnu":        round(mnu,        4),
        "parse_fail": round(parse_fail, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  AXIS RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_axis(axis_name, samples, get_pred_fns, quiet=False):
    """
    Evaluate all baselines on one axis's samples.

    get_pred_fns: dict of {baseline_name: callable(sample) -> pred_dict | None}
    Returns: {baseline_name: aggregated_metrics}
    """
    if not quiet:
        print(f"  Axis {axis_name}: {len(samples)} samples × "
              f"{len(get_pred_fns)} baselines …", flush=True)

    results = {name: [] for name in get_pred_fns}

    for s in samples:
        for name, fn in get_pred_fns.items():
            pred = fn(s)
            results[name].append(score(pred, s))

    return {name: aggregate(scores) for name, scores in results.items()}


def run_axis_3a_sweep(samples, get_pred_fns, quiet=False):
    """
    Special runner for axis 3a: returns per-γ escalation decisions AND
    per-sample MNU so the caller can plot the credit-sweep boundary and
    the escalation-benefit-vs-confidence chart.

    Each row contains:
      gamma         : float — credit level for this sample
      label_esc     : bool  — ground-truth escalation flag
      label_off     : bool  — ground-truth offload decision
      <bl>_esc      : bool | None — model escalation decision
      <bl>_off      : bool | None — model offload decision
      <bl>_mnu      : float| None — normalised utility for this sample
      <bl>_correct  : bool | None — whether both decisions matched label
    """
    if not quiet:
        print(f"  Axis 3a sweep: {len(samples)} γ values …", flush=True)

    sweep = []
    for s in samples:
        row = {
            "gamma":     s["gamma"],
            "label_esc": s["label"]["escalation_flag"],
            "label_off": s["label"]["offload_decision"],
        }
        for name, fn in get_pred_fns.items():
            pred = fn(s)
            if pred is None:
                row[f"{name}_esc"]     = None
                row[f"{name}_off"]     = None
                row[f"{name}_mnu"]     = None
                row[f"{name}_correct"] = None
            else:
                esc = pred.get("escalation_flag")
                off = pred.get("offload_decision")
                sc  = score(pred, s)
                row[f"{name}_esc"]     = esc
                row[f"{name}_off"]     = off
                row[f"{name}_mnu"]     = sc.get("mnu")
                row[f"{name}_correct"] = bool(sc.get("both_correct"))
        sweep.append(row)
    return sweep


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  REPORT FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

def format_robustness_report(all_results):
    """Print a multi-axis comparison table."""
    SEP  = "=" * 90
    SEP2 = "-" * 90
    lines = [SEP, "  ROBUSTNESS EVALUATION REPORT — UE AI Agent (GRPO checkpoint-800)", SEP]

    for axis_id, axis_data in sorted(all_results.items()):
        if axis_id == "3a_sweep":
            continue   # sweep data printed separately
        lines.append(f"\n  Axis {axis_id}")
        lines.append("  " + SEP2)

        baselines = list(next(iter(axis_data.values())).keys()) \
            if axis_data else []
        if not baselines:
            continue

        # Header
        hdr = f"  {'Baseline':<16}  {'Acc':>6}  {'CVR':>6}  " \
              f"{'Off-F1':>7}  {'VER':>6}  {'ER':>6}  " \
              f"{'EscF1':>7}  {'MNU':>8}  {'Parse':>7}"
        lines.append(hdr)
        lines.append("  " + "-" * 76)

        # One line per baseline — show metrics aggregated across the whole axis
        for bl, metrics in axis_data.items():
            lines.append(
                f"  {bl:<16}  "
                f"{metrics.get('accuracy',0):>6.1%}  "
                f"{metrics.get('cvr',0):>6.1%}  "
                f"{metrics.get('offload_f1',0):>7.3f}  "
                f"{metrics.get('ver',0):>6.1%}  "
                f"{metrics.get('er',0):>6.1%}  "
                f"{metrics.get('esc_f1',0):>7.3f}  "
                f"{metrics.get('mnu',0):>+8.4f}  "
                f"{metrics.get('parse_fail',0):>7.1%}"
            )

    lines.append("\n" + SEP)
    return "\n".join(lines)


def print_3a_sweep(sweep_data):
    """Print the credit sweep boundary table."""
    if not sweep_data:
        return
    baselines = [k for k in sweep_data[0] if k.endswith("_esc")
                 and k not in ("label_esc",)]
    bl_names  = [k.replace("_esc", "") for k in baselines]

    print(f"\n  Axis 3a — Credit sweep (β=50%, AR/VR, stable)")
    print(f"  γ% | Label | " + " | ".join(f"{b[:8]:>8}" for b in bl_names))
    print("  " + "-" * (14 + 11 * len(bl_names)))
    for row in sorted(sweep_data, key=lambda r: r["gamma"]):
        pct = round(row["gamma"] * 100)
        lbl = "esc=T" if row["label_esc"] else "esc=F"
        vals= [f"{'T' if row.get(k) else 'F' if row.get(k) is False else '?':>8}"
               for k in baselines]
        # Highlight rows near the thresholds
        mark = ""
        if 9 <= pct <= 11:  mark = " ← γ_min boundary"
        if 19 <= pct <= 21: mark = " ← γ_esc boundary"
        print(f"  {pct:>3}% | {lbl} | " + " | ".join(vals) + mark)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

ALL_AXES = ["1a", "1b", "1c", "1d", "1e", "3a", "3b", "6a", "6b", "6c"]

AXIS_BUILDERS = {
    "1a": build_axis_1a,
    "1b": build_axis_1b,
    "1c": build_axis_1c,
    "1d": build_axis_1d,
    "1e": build_axis_1e,
    "3a": build_axis_3a,
    "3b": build_axis_3b,
    "6a": build_axis_6a,
    "6b": build_axis_6b,
    "6c": build_axis_6c,
}


def main():
    parser = argparse.ArgumentParser(
        description="Robustness evaluation — Axes 1, 3, 6 with five baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  Full evaluation (all axes, all baselines):
    python scripts/evaluate_robustness.py

  Specific axes only:
    python scripts/evaluate_robustness.py --axes 1a 1b 3a

  Specific baselines only:
    python scripts/evaluate_robustness.py --baselines random exhaustive grpo

  Use a specific GRPO checkpoint:
    python scripts/evaluate_robustness.py --grpo-checkpoint models/grpo-ue-agent/checkpoint-800

  Control few-shot k:
    python scripts/evaluate_robustness.py --few-shot-k 5
        """
    )
    parser.add_argument("--baselines", nargs="+",
        choices=["random","always_offload","always_escalate",
                 "sft","few_shot","exhaustive","grpo"],
        default=["random","always_offload","always_escalate",
                 "sft","few_shot","exhaustive","grpo"],
        help="Which baselines to run (default: all)")
    parser.add_argument("--axes", nargs="+",
        choices=ALL_AXES, default=ALL_AXES,
        help="Which robustness axes to evaluate (default: all)")
    parser.add_argument("--grpo-checkpoint", default=GRPO_CKPT,
        help=f"Path to GRPO checkpoint (default: {GRPO_CKPT})")
    parser.add_argument("--sft-model", default=SFT_MODEL,
        help=f"Path to SFT model (default: {SFT_MODEL})")
    parser.add_argument("--few-shot-k", type=int, default=3,
        help="Number of few-shot examples for the few_shot baseline (default: 3)")
    parser.add_argument("--out-json",   default=OUT_JSON)
    parser.add_argument("--out-report", default=OUT_REPORT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    print("=== UE AI Agent — Robustness Evaluation ===")
    print(f"  Axes      : {args.axes}")
    print(f"  Baselines : {args.baselines}")
    print()

    # ── Load neural models ────────────────────────────────────────────────────
    sft_model, sft_tok     = None, None
    grpo_model, grpo_tok   = None, None
    fs_model,  fs_tok      = None, None

    if "sft" in args.baselines:
        sft_model, sft_tok = load_model(args.sft_model, quiet=args.quiet)

    if "grpo" in args.baselines:
        # GRPO checkpoint: load base SFT + merge LoRA adapter
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            print(f"  Loading GRPO checkpoint: {args.grpo_checkpoint} …")
            base = AutoModelForCausalLM.from_pretrained(
                args.sft_model, torch_dtype=torch.float32, device_map={"": "cpu"}
            )
            peft_m     = PeftModel.from_pretrained(base, args.grpo_checkpoint)
            grpo_model = peft_m.merge_and_unload()
            grpo_model.eval()
            grpo_tok = AutoTokenizer.from_pretrained(args.sft_model)
            grpo_tok.pad_token = grpo_tok.eos_token
            print("  GRPO model ready.")
        except Exception as e:
            print(f"  [WARN] Could not load GRPO checkpoint: {e}")

    if "few_shot" in args.baselines:
        # Base SmolLM2-135M — no fine-tuning
        BASE_ID = "HuggingFaceTB/SmolLM2-135M"
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            print(f"  Loading few-shot base model ({BASE_ID}) …")
            fs_model = AutoModelForCausalLM.from_pretrained(
                BASE_ID, torch_dtype=torch.float32, device_map={"": "cpu"}
            )
            fs_model.eval()
            fs_tok = AutoTokenizer.from_pretrained(BASE_ID)
            fs_tok.pad_token = fs_tok.eos_token
            print(f"  Few-shot model ready (k={args.few_shot_k}).")
        except Exception as e:
            print(f"  [WARN] Could not load few-shot model: {e}")

    # ── Build baseline dispatch ───────────────────────────────────────────────
    def make_pred_fns(samples_peek=None):
        """
        Return {name: fn(sample)->pred} for the requested baselines.
        samples_peek: optional first sample, used to pre-check sample structure.
        """
        fns = {}

        if "random" in args.baselines:
            fns["random"] = lambda s: random_pred()

        if "always_offload" in args.baselines:
            fns["always_offload"] = lambda s: always_offload_pred(
                s["battery"], s["condition"], s["gamma"])

        if "always_escalate" in args.baselines:
            fns["always_escalate"] = lambda s: always_escalate_pred(
                s["condition"])

        if "exhaustive" in args.baselines:
            fns["exhaustive"] = lambda s: exhaustive_pred(
                s["battery"], s["app"], s["condition"], s["gamma"])

        if "sft" in args.baselines and sft_model is not None:
            fns["sft"] = lambda s: generate_config(sft_model, sft_tok, s["prompt"])

        if "grpo" in args.baselines and grpo_model is not None:
            fns["grpo"] = lambda s: generate_config(grpo_model, grpo_tok, s["prompt"])

        if "few_shot" in args.baselines and fs_model is not None:
            k = args.few_shot_k
            fns["few_shot"] = lambda s: few_shot_pred(
                fs_model, fs_tok, s["prompt"], k=k)

        return fns

    # ── Run axes ──────────────────────────────────────────────────────────────
    all_results = {}

    for axis_id in args.axes:
        if not args.quiet:
            print(f"\nBuilding axis {axis_id} …")
        samples = AXIS_BUILDERS[axis_id]()
        fns = make_pred_fns()

        if axis_id == "3a":
            # Credit sweep — also store the per-γ table
            sweep = run_axis_3a_sweep(samples, fns, quiet=args.quiet)
            all_results["3a_sweep"] = sweep
            print_3a_sweep(sweep)
            # Also aggregate for the report table
            all_results[axis_id] = run_axis(axis_id, samples, fns,
                                             quiet=args.quiet)
        else:
            all_results[axis_id] = run_axis(axis_id, samples, fns,
                                             quiet=args.quiet)

    # ── Write results ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    # Merge with existing results so partial runs don't overwrite other axes
    if os.path.exists(args.out_json):
        try:
            with open(args.out_json, "r", encoding="utf-8") as f:
                existing = json.load(f)
            existing.update(all_results)   # new results overwrite matching axes
            all_results = existing
        except Exception:
            pass  # if load/parse fails, just use what we have

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults → {args.out_json}")

    report = format_robustness_report(all_results)
    print("\n" + report)
    with open(args.out_report, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"Report  → {args.out_report}")


if __name__ == "__main__":
    main()