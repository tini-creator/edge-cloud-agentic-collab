"""
evaluate.py  —  Full system evaluation for the UE AI Agent
------------------------------------------------------------
Evaluates three baselines against all four metric families defined
in the paper, covering the two-output decision schema (offload_decision,
escalation_flag) with energy credit as a state variable.

  Baselines
  ─────────
    random   Uniformly random valid (o, e) pair — lower bound
    sft      SFT-only model  (models/sft-ue-merged)
    grpo     GRPO fine-tuned (models/grpo-ue-agent)

  Prompt format  (identical to run_sft.py and run_grpo.py)
  ─────────────────────────────────────────────────────────
    UE Context: Battery is at β%. User just opened a <app>.
                Network is <η>. Energy credit is <desc> (<γ>%).
    Required keys (offload_decision, escalation_flag):
    Decision:
    {

  Metric families
  ───────────────
  Family 1 — Feasibility safety (FVR)
    FVR_C1       offload=True when β ≤ 20%            (C1 violation)
    FVR_C8       offload=True or escalate=True         (C8 violation)
                 when η = weak signal
    FVR_C1∩C8   either=True when β ≤ 20% AND          (compound)
                 η = weak signal
    FVR_credit   offload=True when γ ≤ γ_min           (credit floor)
    Parse failures count as violations in all FVR metrics.

  Family 2 — Offload quality (in feasible region only)
    COR          Correct offload rate — fraction where o_pred = o_label
    Offload P    Precision of offload=True predictions
    Offload R    Recall of offload=True labels
    Offload F1   Harmonic mean of P and R

  Family 3 — Escalation quality
    VER          Valuable escalation rate — escalations in warranted states
    ER           Escalation recall — warranted states that were escalated
    MER_HO       Missed escalation rate on post-handover mobility steps
    UER          Unnecessary escalation rate = 1 − VER
    Escalation F1

  Family 4 — Utility
    MNU          Mean normalised utility U(o,e|s) from reward_model.py
    ΔU           MNU_GRPO − MNU_SFT
    CCR          Credit spent per offload decision (mobility episodes)
                 = total_credit_spent / n_offload_steps  (↓ lower = cheaper per offload)
    credit_per_step  Mean credit spent per time step = total_credit_spent / T  (↓ lower)
    Parse fail   Fraction of outputs that could not be parsed as JSON

  Test sets
  ─────────
    id_set         200 samples — in-distribution, includes energy credit
    c1_set         100 samples — β ≤ 20%, stable/congested
    c8_set         100 samples — weak signal, β > 20%
    combined_set   150 samples — C1 only / C8 only / C1∩C8 / control
    credit_set      80 samples — γ ≤ γ_min
    ood_set             OOD across 5 axes (novel app / prose battery /
                        compound / boundary / credit descriptor)
    mobility_set    ns-3 trace episodes — for MER_HO and CCR

Usage
─────
  python scripts/evaluate.py
  python scripts/evaluate.py --baselines random sft
  python scripts/evaluate.py --grpo-model models/grpo-ue-agent
  python scripts/evaluate.py --checkpoint models/grpo-ue-agent/checkpoint-200
  python scripts/evaluate.py --quiet
"""

import os
import sys
import json
import time
import math
import random
import argparse
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SFT_MODEL  = os.path.join(ROOT, "models", "sft-ue-merged")
GRPO_MODEL = os.path.join(ROOT, "models", "grpo-ue-agent")
TRACE_FILE = os.path.join(ROOT, "data",   "radio_parameters_moving_clean.csv")
OUT_JSON   = os.path.join(ROOT, "models", "eval_results.json")
OUT_REPORT = os.path.join(ROOT, "models", "eval_report.txt")

sys.path.insert(0, ROOT)
from scripts.reward_model import (
    _repair_json, dict_raise_on_duplicates, calculate_ue_reward,
    parse_context, compute_R, GAMMA_MIN, GAMMA_ESC,
)
from data.generate_dataset import (
    determine_optimal_decision, app_types, network_conditions,
    should_escalate, c1_active, c8_active, offload_feasible,
    extract_real_episodes, load_trace, episode_to_samples,
    simulate_credit, credit_cost,
    EPISODE_T, EPISODE_STRIDE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PROMPT BUILDER  (identical format to run_sft.py / run_grpo.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _credit_desc(gamma):
    if gamma >= 0.8:   return "full"
    if gamma >= 0.5:   return "moderate"
    if gamma > GAMMA_ESC: return "low"
    if gamma > GAMMA_MIN: return "critically low"
    return "depleted"


def build_eval_prompt(battery, app, condition, gamma):
    """Build the natural-language prompt including energy credit."""
    credit_pct  = round(gamma * 100)
    credit_desc = _credit_desc(gamma)
    return (
        f"Battery is at {battery}%. "
        f"User just opened a {app['name']}. "
        f"Network is {condition}. "
        f"Energy credit is {credit_desc} ({credit_pct}%)."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CONSTRAINT CHECKERS
# ═══════════════════════════════════════════════════════════════════════════════

def violates_c1(pred, battery):
    """C1: offload=True when battery ≤ 20%. Parse failure = violation."""
    if battery > 20:
        return False
    if pred is None:
        return True
    return bool(pred.get("offload_decision", True))


def violates_c8(pred, condition):
    """C8: offload=True or escalate=True when η=weak signal."""
    if condition != "weak signal":
        return False
    if pred is None:
        return True
    return (bool(pred.get("offload_decision", True))
            or bool(pred.get("escalation_flag", True)))


def violates_c1_and_c8(pred, battery, condition):
    """C1∩C8: either output True when both prohibitions active."""
    if battery > 20 or condition != "weak signal":
        return False
    if pred is None:
        return True
    return (bool(pred.get("offload_decision", True))
            or bool(pred.get("escalation_flag", True)))


def violates_credit_floor(pred, gamma):
    """Credit floor: offload=True when γ ≤ γ_min."""
    if gamma > GAMMA_MIN:
        return False
    if pred is None:
        return True
    return bool(pred.get("offload_decision", True))


def violates_any(pred, battery, condition, gamma):
    return (violates_c1(pred, battery)
            or violates_c8(pred, condition)
            or violates_credit_floor(pred, gamma))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODEL INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_path, quiet=False):
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        if not quiet:
            print(f"  Loading {os.path.basename(model_path)} …")
        tok = AutoTokenizer.from_pretrained(model_path)
        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float32, device_map="cpu"
        )
        model.eval()
        return model, tok
    except Exception as e:
        print(f"  [WARN] Could not load {model_path}: {e}")
        return None, None


def generate_config(model, tokenizer, prompt_text, max_new_tokens=40):
    """
    Run one forward pass and return parsed dict or None.
    max_new_tokens=40: two-field JSON is ~18 tokens; 40 gives safe headroom.
    """
    import torch
    full_prompt = (
        f"UE Context: {prompt_text}\n"
        f"Required keys (offload_decision, escalation_flag):\n"
        f"Decision:\n{{"
    )
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
    if not raw.startswith("{"):
        raw = "{" + raw
    repaired = _repair_json(raw)
    try:
        return json.loads(repaired, object_pairs_hook=dict_raise_on_duplicates)
    except Exception:
        return None


def random_decision():
    """Random baseline: valid (o, e) pair — never (1,1)."""
    pairs = [(False, False), (True, False), (False, True)]
    return {"offload_decision": random.choice(pairs)[0],
            "escalation_flag":  random.choice(pairs)[1]}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TEST-SET BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def _make_sample(battery, app, condition, gamma, meta=None):
    o, e = determine_optimal_decision(battery, app, condition, gamma)
    prompt = build_eval_prompt(battery, app, condition, gamma)
    label  = {"offload_decision": o, "escalation_flag": e}
    s = {"prompt": prompt, "label": label,
         "battery": battery, "app": app, "condition": condition, "gamma": gamma}
    if meta:
        s.update(meta)
    return s


def _sg(low=GAMMA_ESC + 0.01, high=1.0):
    return round(random.uniform(low, high), 3)


def build_id_test_set(n=200, seed=99):
    """In-distribution: full state space including credit, fresh seed."""
    random.seed(seed)
    samples = []
    for _ in range(n):
        battery   = random.randint(1, 100)
        app       = random.choice(app_types)
        condition = random.choice(network_conditions)
        gamma     = _sg(0.05, 1.0)
        samples.append(_make_sample(battery, app, condition, gamma))
    return samples


def build_c1_set(n=100, seed=77):
    """C1 only: β ≤ 20%, usable network, any credit."""
    random.seed(seed)
    samples = []
    while len(samples) < n:
        battery   = random.randint(1, 20)
        app       = random.choice(app_types)
        condition = random.choice(["stable", "congested"])
        gamma     = _sg(0.05, 1.0)
        samples.append(_make_sample(battery, app, condition, gamma,
                                    {"constraint": "c1_only"}))
    return samples


def build_c8_set(n=100, seed=55):
    """C8 only: weak signal, β > 20%."""
    random.seed(seed)
    samples = []
    while len(samples) < n:
        battery   = random.randint(21, 100)
        app       = random.choice(app_types)
        condition = "weak signal"
        gamma     = _sg(0.05, 1.0)
        samples.append(_make_sample(battery, app, condition, gamma,
                                    {"constraint": "c8_only"}))
    return samples


def build_combined_constraint_set(seed=33):
    """
    Combined C1∪C8 with four sub-groups:
      c1_only    50 samples
      c8_only    50 samples
      c1_and_c8  30 samples  (hardest — both prohibitions)
      none       20 samples  (control — neither active)
    """
    random.seed(seed)
    samples = []

    for _ in range(50):   # C1 only
        battery   = random.randint(1, 20)
        app       = random.choice(app_types)
        condition = random.choice(["stable", "congested"])
        gamma     = _sg(0.05, 1.0)
        samples.append(_make_sample(battery, app, condition, gamma,
                                    {"constraint": "c1_only"}))

    for _ in range(50):   # C8 only
        battery   = random.randint(21, 100)
        app       = random.choice(app_types)
        condition = "weak signal"
        gamma     = _sg(0.05, 1.0)
        samples.append(_make_sample(battery, app, condition, gamma,
                                    {"constraint": "c8_only"}))

    for _ in range(30):   # C1 ∩ C8
        battery   = random.randint(1, 20)
        app       = random.choice(app_types)
        condition = "weak signal"
        gamma     = _sg(0.05, 1.0)
        samples.append(_make_sample(battery, app, condition, gamma,
                                    {"constraint": "c1_and_c8"}))

    for _ in range(20):   # control
        battery   = random.randint(31, 100)
        app       = random.choice(app_types)
        condition = random.choice(["stable", "congested"])
        gamma     = _sg(GAMMA_ESC + 0.05, 1.0)
        samples.append(_make_sample(battery, app, condition, gamma,
                                    {"constraint": "none"}))

    return samples


def build_credit_floor_set(n=80, seed=44):
    """Credit floor: γ ≤ γ_min, usable network, battery > 20%."""
    random.seed(seed)
    samples = []
    while len(samples) < n:
        battery   = random.randint(21, 100)
        app       = random.choice(app_types)
        condition = random.choice(["stable", "congested"])
        gamma     = round(random.uniform(0.0, GAMMA_MIN), 4)
        samples.append(_make_sample(battery, app, condition, gamma,
                                    {"constraint": "credit_floor"}))
    return samples


def build_ood_test_set(seed=42):
    """
    OOD test set across five axes:
      A. Novel app names (same compute tier, different name)
      B. Prose battery descriptions
      C. Compound conditions (battery + network together)
      D. Boundary battery values (20, 21, 30, 31)
      E. Prose credit descriptors (OOD phrasing of energy credit)
    """
    random.seed(seed)
    samples = []

    # Axis A: novel app names
    novel_apps = [
        ("Spotify music stream",           0),
        ("WhatsApp video call",            2),
        ("Call of Duty Mobile",            3),
        ("Meta Quest virtual environment", 4),
        ("background file sync",           1),
        ("Netflix 4K download",            2),
        ("Pokemon GO AR camera",           4),
        ("idle weather widget",            0),
    ]
    for display_name, idx in novel_apps:
        backing   = app_types[idx]
        battery   = random.randint(10, 95)
        condition = random.choice(network_conditions)
        gamma     = _sg(0.05, 1.0)
        o, e = determine_optimal_decision(battery, backing, condition, gamma)
        prompt = (
            f"Battery is at {battery}%. "
            f"User just opened a {display_name}. "
            f"Network is {condition}. "
            f"Energy credit is {_credit_desc(gamma)} ({round(gamma*100)}%)."
        )
        samples.append({"prompt": prompt,
                         "label": {"offload_decision": o, "escalation_flag": e},
                         "battery": battery, "app": backing,
                         "condition": condition, "gamma": gamma,
                         "ood_axis": "novel_app"})

    # Axis B: prose battery descriptions (with credit)
    prose_battery = [
        ("battery is nearly dead",      10),
        ("almost out of charge",         8),
        ("battery is critically low",   15),
        ("phone is fully charged",      95),
        ("battery at half",             50),
        ("running low on battery",      25),
        ("battery saver kicks in soon", 22),
        ("battery is fine",             65),
    ]
    for prose, battery in prose_battery:
        app       = random.choice(app_types)
        condition = random.choice(network_conditions)
        gamma     = _sg(0.05, 1.0)
        o, e = determine_optimal_decision(battery, app, condition, gamma)
        prompt = (
            f"The {prose}. "
            f"User just opened a {app['name']}. "
            f"Network is {condition}. "
            f"Energy credit is {_credit_desc(gamma)} ({round(gamma*100)}%)."
        )
        samples.append({"prompt": prompt,
                         "label": {"offload_decision": o, "escalation_flag": e},
                         "battery": battery, "app": app,
                         "condition": condition, "gamma": gamma,
                         "ood_axis": "prose_battery"})

    # Axis C: compound conditions including C1∩C8
    compound = [
        (15, "congested"),
        ( 8, "weak signal"),
        (18, "congested"),
        (75, "congested"),
        (35, "weak signal"),
        (50, "congested"),
        (12, "weak signal"),
        ( 5, "weak signal"),
    ]
    for battery, condition in compound:
        app   = random.choice(app_types)
        gamma = _sg(0.05, 1.0)
        o, e  = determine_optimal_decision(battery, app, condition, gamma)
        prompt = (
            f"Battery is at {battery}%. "
            f"Network is {condition}. "
            f"User just opened a {app['name']}. "
            f"Energy credit is {_credit_desc(gamma)} ({round(gamma*100)}%)."
        )
        samples.append({"prompt": prompt,
                         "label": {"offload_decision": o, "escalation_flag": e},
                         "battery": battery, "app": app,
                         "condition": condition, "gamma": gamma,
                         "ood_axis": "compound"})

    # Axis D: boundary battery values
    for battery in [20, 21, 30, 31, 40, 41]:
        app       = random.choice(app_types)
        condition = random.choice(network_conditions)
        gamma     = _sg(0.05, 1.0)
        o, e = determine_optimal_decision(battery, app, condition, gamma)
        prompt = build_eval_prompt(battery, app, condition, gamma)
        samples.append({"prompt": prompt,
                         "label": {"offload_decision": o, "escalation_flag": e},
                         "battery": battery, "app": app,
                         "condition": condition, "gamma": gamma,
                         "ood_axis": "boundary"})

    # Axis E: OOD credit descriptors — different phrasing for credit level
    ood_credit_phrases = [
        ("energy is almost gone",                    0.08),
        ("network credits nearly exhausted",         0.12),
        ("offload budget running low",               0.18),
        ("credit tank is full",                      0.95),
        ("halfway through monthly offload budget",   0.50),
    ]
    for phrase, gamma in ood_credit_phrases:
        battery   = random.randint(31, 90)
        app       = random.choice(app_types)
        condition = random.choice(["stable", "congested"])
        o, e = determine_optimal_decision(battery, app, condition, gamma)
        prompt = (
            f"Battery is at {battery}%. "
            f"User just opened a {app['name']}. "
            f"Network is {condition}. "
            f"The {phrase}."
        )
        samples.append({"prompt": prompt,
                         "label": {"offload_decision": o, "escalation_flag": e},
                         "battery": battery, "app": app,
                         "condition": condition, "gamma": gamma,
                         "ood_axis": "credit_descriptor"})

    return samples


def build_mobility_set(trace_path, seed=17):
    """
    Mobility test set from ns-3 trace episodes.
    Used for: MER_HO (missed escalation on handover) and CCR (credit conservation).
    Falls back to synthetic GM episodes if trace file is unavailable.
    """
    random.seed(seed)
    samples = []

    trace_df = None
    if trace_path and os.path.exists(trace_path):
        trace_df = load_trace(trace_path)

    if trace_df is not None:
        episodes = extract_real_episodes(trace_df, T=EPISODE_T,
                                         stride=EPISODE_STRIDE)
        for ep in episodes[:10]:   # use first 10 episodes
            app = random.choice(app_types)
            ep_samples = episode_to_samples(ep, app, "mobility_real")
            for s in ep_samples:
                m   = s["meta"]
                r   = json.loads(s["response"])
                samples.append({
                    "prompt":    s["prompt"],
                    "label":     {"offload_decision": r["offload_decision"],
                                  "escalation_flag":  r["escalation_flag"]},
                    "battery":   m["battery"],
                    "app":       app,
                    "condition": m["condition"],
                    "gamma":     m["gamma"],
                    "handover":  m.get("handover", False),
                    "source":    "real",
                })
    else:
        print("  [INFO] Trace file not found — using synthetic GM episodes for mobility set.")
        from data.generate_dataset import generate_synthetic_episode
        for _ in range(5):
            app = random.choice(app_types)
            ep  = generate_synthetic_episode(T=EPISODE_T)
            ep_samples = episode_to_samples(ep, app, "mobility_synthetic")
            for s in ep_samples:
                m   = s["meta"]
                r   = json.loads(s["response"])
                samples.append({
                    "prompt":    s["prompt"],
                    "label":     {"offload_decision": r["offload_decision"],
                                  "escalation_flag":  r["escalation_flag"]},
                    "battery":   m["battery"],
                    "app":       app,
                    "condition": m["condition"],
                    "gamma":     m["gamma"],
                    "handover":  m.get("handover", False),
                    "source":    "synthetic",
                })

    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SINGLE-SAMPLE SCORER
# ═══════════════════════════════════════════════════════════════════════════════

def score_sample(pred, label, battery, condition, gamma, prompt):
    """Return per-sample metric dict for aggregation."""
    o_pred = None if pred is None else bool(pred.get("offload_decision"))
    e_pred = None if pred is None else bool(pred.get("escalation_flag"))
    o_label = bool(label["offload_decision"])
    e_label = bool(label["escalation_flag"])

    # escalation warranted = ground-truth label from the dataset.
    # The label is produced by determine_optimal_decision() which implements
    # all five conditions C1-C5 (credit zone, battery boundary, congested-heavy,
    # headroom exhaustion, post-handover).  Using label["escalation_flag"]
    # directly is correct and avoids reimplementing the criterion here, which
    # previously missed C3, C4, C5 and had a duplicate third clause.
    esc_warranted = e_label   # ground truth from dataset label

    mnu = calculate_ue_reward(prompt,
                              json.dumps(pred) if pred is not None else "{}",
                              verbose=False)

    # R(o,s): offload utility component only.
    # parse_context recovers battery, app_key, condition, gamma
    # from the prompt text so we can call compute_R without
    # re-running the model.
    try:
        _bat, _app_key, _cond, _gam = parse_context(prompt)
        _o   = bool(pred.get("offload_decision")) if pred else False
        mean_r = compute_R(_o, _bat, _app_key, _cond, _gam)
    except Exception:
        mean_r = 0.0

    return {
        "parse_fail":    pred is None,
        "o_correct":     o_pred == o_label if o_pred is not None else False,
        "e_correct":     e_pred == e_label if e_pred is not None else False,
        "both_correct":  (o_pred == o_label and e_pred == e_label)
                         if pred is not None else False,

        # offload TP/FP/FN (in feasible region)
        "feasible":      offload_feasible(battery, condition, gamma),
        "offload_tp":    o_pred is True  and o_label is True,
        "offload_fp":    o_pred is True  and o_label is False,
        "offload_fn":    o_pred is False and o_label is True,

        # escalation TP/FP/FN against ground-truth label
        # esc_tp: model escalated and escalation was warranted
        # esc_fp: model escalated but escalation was NOT warranted
        # esc_fn: model did NOT escalate but escalation WAS warranted
        "esc_warranted": esc_warranted,
        "esc_tp":        e_pred is True  and esc_warranted,
        "esc_fp":        e_pred is True  and not esc_warranted,
        "esc_fn":        e_pred is False and esc_warranted,

        # constraint violations
        "viol_c1":         violates_c1(pred, battery),
        "viol_c8":         violates_c8(pred, condition),
        "viol_c1_and_c8":  violates_c1_and_c8(pred, battery, condition),
        "viol_credit":     violates_credit_floor(pred, gamma),

        # utility
        "mnu":           mnu,
        "mean_r":        mean_r,
    }


def safe_div(a, b, default=0.0):
    return a / b if b > 0 else default


# ═══════════════════════════════════════════════════════════════════════════════
# 6. BASELINE EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_baseline(name, get_pred_fn,
                      id_set, c1_set, c8_set, combined_set,
                      credit_set, ood_set, mobility_set,
                      quiet=False):
    if not quiet:
        print(f"\n  Evaluating [{name}] …")

    def run(samples, label=""):
        results = []
        t0 = time.time()
        for s in samples:
            pred = get_pred_fn(s)  # full sample dict; fn extracts prompt or uses eta_history
            sc   = score_sample(pred, s["label"],
                                s["battery"], s["condition"], s["gamma"],
                                s["prompt"])
            sc["pred"]      = pred
            sc["label"]     = s["label"]
            sc["battery"]   = s["battery"]
            sc["condition"] = s["condition"]
            sc["gamma"]     = s["gamma"]
            sc["prompt"]    = s["prompt"]
            sc["handover"]  = s.get("handover", False)
            sc["pred"]      = pred  # needed for CCR simulation
            sc["ood_axis"]  = s.get("ood_axis", "id")
            sc["constraint"]= s.get("constraint", "")
            results.append(sc)
        return results, time.time() - t0

    id_res,       id_t   = run(id_set,       "ID")
    c1_res,       _      = run(c1_set,       "C1")
    c8_res,       _      = run(c8_set,       "C8")
    comb_res,     _      = run(combined_set, "Combined")
    credit_res,   _      = run(credit_set,   "Credit")
    ood_res,      ood_t  = run(ood_set,      "OOD")
    mob_res,      _      = run(mobility_set, "Mobility")

    n_id  = len(id_res)
    n_ood = len(ood_res)
    n_mob = len(mob_res)

    # ── Family 1: FVR ─────────────────────────────────────────────────────────
    fvr_c1       = safe_div(sum(r["viol_c1"]        for r in c1_res),   len(c1_res))
    fvr_c8       = safe_div(sum(r["viol_c8"]        for r in c8_res),   len(c8_res))
    fvr_credit   = safe_div(sum(r["viol_credit"]    for r in credit_res), len(credit_res))

    def fvr_group(group_name):
        g = [r for r in comb_res if r["constraint"] == group_name]
        if not g:
            return None
        return safe_div(
            sum(violates_any(r["pred"], r["battery"], r["condition"], r["gamma"])
                for r in g), len(g)
        )

    fvr_c1_only   = fvr_group("c1_only")
    fvr_c8_only   = fvr_group("c8_only")
    fvr_c1_and_c8 = fvr_group("c1_and_c8")
    fvr_control   = fvr_group("none")
    fvr_combined  = safe_div(
        sum(violates_any(r["pred"], r["battery"], r["condition"], r["gamma"])
            for r in comb_res if r["constraint"] != "none"),
        len([r for r in comb_res if r["constraint"] != "none"]),
    )

    # ── Family 2: Offload quality (feasible region only) ──────────────────────
    feas = [r for r in id_res if r["feasible"]]
    cor  = safe_div(sum(r["o_correct"] for r in feas), len(feas))
    tp_o = sum(r["offload_tp"] for r in feas)
    fp_o = sum(r["offload_fp"] for r in feas)
    fn_o = sum(r["offload_fn"] for r in feas)
    off_prec = safe_div(tp_o, tp_o + fp_o)
    off_rec  = safe_div(tp_o, tp_o + fn_o)
    off_f1   = safe_div(2 * off_prec * off_rec, off_prec + off_rec)

    # ── Family 3: Escalation quality ──────────────────────────────────────────
    warranted      = [r for r in id_res if r["esc_warranted"]]
    escalated      = [r for r in id_res if
                      r["pred"] is not None
                      and r["pred"].get("escalation_flag")]
    tp_e = sum(r["esc_tp"] for r in id_res)
    fp_e = sum(r["esc_fp"] for r in id_res)
    fn_e = sum(r["esc_fn"] for r in id_res)
    ver  = safe_div(tp_e, tp_e + fp_e)
    er   = safe_div(tp_e, tp_e + fn_e)
    uer  = 1.0 - ver
    esc_f1 = safe_div(2 * ver * er, ver + er)

    # MER_HO: missed escalation on post-handover steps
    ho_steps = [r for r in mob_res if r["handover"]]
    mer_ho   = safe_div(
        sum(1 for r in ho_steps
            if r["pred"] is not None
            and not r["pred"].get("escalation_flag")),
        len(ho_steps),
    )

    # ── Family 4: Utility ─────────────────────────────────────────────────────
    mnu         = sum(r["mnu"]    for r in id_res) / n_id
    mean_r      = sum(r["mean_r"] for r in id_res) / n_id
    parse_fail  = safe_div(sum(r["parse_fail"] for r in id_res), n_id)
    latency_ms  = (id_t / n_id) * 1000

    # CCR and credit_per_step — two complementary credit-efficiency metrics.
    #
    # Episode grouping: build_mobility_set produces flat sample dicts with no
    # "meta" sub-key, so we split by fixed episode length (EPISODE_T = 30).
    #
    # Option A — CCR (credit spent per offload decision):
    #   CCR = total_credit_spent / n_offload_decisions
    #   = (sum of credit_cost(app, condition) over steps where o_pred=True)
    #     / (number of steps where o_pred=True)
    #   Lower = each offload decision is cheaper (e.g. light apps, good signal).
    #   Policies that never offload → CCR = 0 (no offload, no spend).
    #   Removes the "offload less = win" artefact of ratio-based CCR.
    #
    # Option C — credit_per_step (Δγ/step, mean credit spent per time step):
    #   credit_per_step = total_credit_spent / total_steps
    #   Directly interpretable: how credit-intensive is this policy on average?
    #   Pairs naturally with R(o,s): high credit_per_step + high R = efficient;
    #   high credit_per_step + low R = wasteful offloading.
    if mob_res and mobility_set:
        episodes_sim = []
        ep_len = EPISODE_T          # 30 steps per episode, from generate_dataset
        pairs  = list(zip(mobility_set, mob_res))
        for start in range(0, len(pairs), ep_len):
            chunk = pairs[start : start + ep_len]
            if chunk:
                episodes_sim.append(chunk)

        total_credit_spent  = 0.0
        total_offload_steps = 0
        total_steps         = 0

        for ep in episodes_sim:
            for s, r in ep:
                app       = s.get("app")       if isinstance(s, dict) else None
                condition = s.get("condition") if isinstance(s, dict) else "stable"
                o_pred    = r["pred"].get("offload_decision", False) \
                            if r.get("pred") else False
                total_steps += 1
                if app is not None and o_pred:
                    total_credit_spent  += credit_cost(app, condition)
                    total_offload_steps += 1

        # Option A: credit spent per offload (0 when policy never offloads)
        ccr             = (total_credit_spent / total_offload_steps
                           if total_offload_steps > 0 else 0.0)
        # Option C: credit spent per step
        credit_per_step = (total_credit_spent / total_steps
                           if total_steps > 0 else 0.0)
    else:
        ccr             = None
        credit_per_step = None

    # ── ID / OOD accuracy ─────────────────────────────────────────────────────
    id_acc  = safe_div(sum(r["both_correct"] for r in id_res),  n_id)
    ood_acc = safe_div(sum(r["both_correct"] for r in ood_res), n_ood)
    ood_gap = id_acc - ood_acc

    # Per-axis OOD
    axes = ["novel_app", "prose_battery", "compound", "boundary", "credit_descriptor"]
    ood_by_axis = {}
    for axis in axes:
        ax = [r for r in ood_res if r["ood_axis"] == axis]
        if ax:
            ood_by_axis[axis] = round(
                safe_div(sum(r["both_correct"] for r in ax), len(ax)), 4
            )

    return {
        "name": name,
        # Family 1
        "fvr_c1":        round(fvr_c1,        4),
        "fvr_c8":        round(fvr_c8,        4),
        "fvr_credit":    round(fvr_credit,    4),
        "fvr_combined":  round(fvr_combined,  4),
        "fvr_c1_only":   round(fvr_c1_only,   4) if fvr_c1_only   is not None else None,
        "fvr_c8_only":   round(fvr_c8_only,   4) if fvr_c8_only   is not None else None,
        "fvr_c1_and_c8": round(fvr_c1_and_c8, 4) if fvr_c1_and_c8 is not None else None,
        "fvr_control":   round(fvr_control,   4) if fvr_control   is not None else None,
        # Family 2
        "cor":           round(cor,      4),
        "offload_prec":  round(off_prec, 4),
        "offload_rec":   round(off_rec,  4),
        "offload_f1":    round(off_f1,   4),
        # Family 3
        "ver":           round(ver,     4),
        "er":            round(er,      4),
        "uer":           round(uer,     4),
        "esc_f1":        round(esc_f1,  4),
        "mer_ho":        round(mer_ho,  4),
        # Family 4
        "mnu":           round(mnu,         4),
        "mean_r":        round(mean_r,      4),
        "parse_fail":    round(parse_fail,  4),
        "latency_ms":    round(latency_ms,  2),
        "ccr":             round(ccr,             4) if ccr             is not None else None,
        "credit_per_step": round(credit_per_step, 4) if credit_per_step is not None else None,
        # Generalisation
        "id_accuracy":   round(id_acc,  4),
        "ood_accuracy":  round(ood_acc, 4),
        "ood_gap":       round(ood_gap, 4),
        "ood_by_axis":   ood_by_axis,
        # Sample counts
        "n_id":       n_id,
        "n_c1":       len(c1_res),
        "n_c8":       len(c8_res),
        "n_combined": len(comb_res),
        "n_credit":   len(credit_res),
        "n_ood":      n_ood,
        "n_mobility": n_mob,
        "n_ho_steps": len(ho_steps),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. REPORT FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

def format_report(all_results, delta_u=None):
    SEP  = "=" * 82
    SEP2 = "-" * 82
    lines = []

    def pct(v):
        return f"{v:.1%}" if v is not None else "n/a"

    r0 = all_results[0]
    lines += [
        SEP,
        "  EVALUATION REPORT — Agentic UE Resource Manager (two-output schema)",
        SEP,
        f"  Baselines   : {', '.join(r['name'] for r in all_results)}",
        f"  ID set      : {r0['n_id']} samples",
        f"  C1 set      : {r0['n_c1']} samples  (β ≤ 20%, usable network)",
        f"  C8 set      : {r0['n_c8']} samples  (weak signal, β > 20%)",
        f"  Combined    : {r0['n_combined']} samples  (C1/C8/C1∩C8/control)",
        f"  Credit set  : {r0['n_credit']} samples  (γ ≤ γ_min)",
        f"  OOD set     : {r0['n_ood']} samples  (5 axes)",
        f"  Mobility    : {r0['n_mobility']} samples  ({r0['n_ho_steps']} handover steps)",
        SEP2,
    ]

    # Family 1
    lines += ["", "  FAMILY 1 — Constraint Safety (↓ lower is better, target 0%)", ""]
    hdr = f"  {'Baseline':<12}  {'FVR C1':>8}  {'FVR C8':>8}  {'FVR Credit':>11}  {'FVR Combined':>13}"
    lines += [hdr, "  " + "-" * 60]
    for r in all_results:
        lines.append(
            f"  {r['name']:<12}  "
            f"{pct(r['fvr_c1']):>8}  "
            f"{pct(r['fvr_c8']):>8}  "
            f"{pct(r['fvr_credit']):>11}  "
            f"{pct(r['fvr_combined']):>13}"
        )

    lines += ["", "  FVR sub-case breakdown:", ""]
    hdr2 = f"  {'Baseline':<12}  {'C1 only':>9}  {'C8 only':>9}  {'C1∩C8':>8}  {'control':>9}"
    lines += [hdr2, "  " + "-" * 52]
    for r in all_results:
        lines.append(
            f"  {r['name']:<12}  "
            f"{pct(r['fvr_c1_only']):>9}  "
            f"{pct(r['fvr_c8_only']):>9}  "
            f"{pct(r['fvr_c1_and_c8']):>8}  "
            f"{pct(r['fvr_control']):>9}"
        )

    # Family 2
    lines += ["", "  FAMILY 2 — Offload Quality (feasible region only, ↑ higher is better)", ""]
    hdr3 = f"  {'Baseline':<12}  {'COR':>7}  {'Precision':>10}  {'Recall':>8}  {'F1':>7}"
    lines += [hdr3, "  " + "-" * 52]
    for r in all_results:
        lines.append(
            f"  {r['name']:<12}  "
            f"{pct(r['cor']):>7}  "
            f"{pct(r['offload_prec']):>10}  "
            f"{pct(r['offload_rec']):>8}  "
            f"{pct(r['offload_f1']):>7}"
        )

    # Family 3
    lines += ["", "  FAMILY 3 — Escalation Quality", ""]
    hdr4 = f"  {'Baseline':<12}  {'VER':>7}  {'ER':>7}  {'UER':>7}  {'Esc F1':>8}  {'MER_HO':>8}"
    lines += [hdr4, "  " + "-" * 60]
    for r in all_results:
        lines.append(
            f"  {r['name']:<12}  "
            f"{pct(r['ver']):>7}  "
            f"{pct(r['er']):>7}  "
            f"{pct(r['uer']):>7}  "
            f"{pct(r['esc_f1']):>8}  "
            f"{pct(r['mer_ho']):>8}"
        )

    # Family 4
    lines += ["", "  FAMILY 4 — Utility & Efficiency", ""]
    hdr5 = f"  {'Baseline':<12}  {'MNU':>8}  {'CCR':>7}  {'Parse Fail':>11}  {'Latency':>9}"
    lines += [hdr5, "  " + "-" * 58]
    for r in all_results:
        lines.append(
            f"  {r['name']:<12}  "
            f"{r['mnu']:>+8.4f}  "
            f"{pct(r['ccr']):>7}  "
            f"{pct(r['parse_fail']):>11}  "
            f"{r['latency_ms']:>7.1f}ms"
        )
    if delta_u is not None:
        lines.append(f"\n  ΔU (GRPO − SFT) = {delta_u:+.4f}")

    # Generalisation
    lines += ["", "  GENERALISATION — ID vs OOD accuracy", ""]
    hdr6 = (f"  {'Baseline':<12}  {'ID Acc':>8}  {'OOD Acc':>9}  "
            f"{'OOD Gap':>9}  {'novel':>7}  {'prose':>7}  {'compound':>9}  "
            f"{'boundary':>9}  {'credit':>8}")
    lines += [hdr6, "  " + "-" * 88]
    for r in all_results:
        d = r["ood_by_axis"]
        lines.append(
            f"  {r['name']:<12}  "
            f"{pct(r['id_accuracy']):>8}  "
            f"{pct(r['ood_accuracy']):>9}  "
            f"{pct(r['ood_gap']):>9}  "
            f"{pct(d.get('novel_app')):>7}  "
            f"{pct(d.get('prose_battery')):>7}  "
            f"{pct(d.get('compound')):>9}  "
            f"{pct(d.get('boundary')):>9}  "
            f"{pct(d.get('credit_descriptor')):>8}"
        )

    lines += ["", SEP]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main(baselines_to_run, grpo_model_path, quiet):
    print("Building test sets …")
    id_set       = build_id_test_set(n=200)
    c1_set       = build_c1_set(n=100)
    c8_set       = build_c8_set(n=100)
    combined_set = build_combined_constraint_set()
    credit_set   = build_credit_floor_set(n=80)
    ood_set      = build_ood_test_set()
    mobility_set = build_mobility_set(TRACE_FILE)

    if not quiet:
        print(f"  ID          : {len(id_set)}")
        print(f"  C1          : {len(c1_set)}")
        print(f"  C8          : {len(c8_set)}")
        print(f"  Combined    : {len(combined_set)}")
        print(f"  Credit floor: {len(credit_set)}")
        print(f"  OOD         : {len(ood_set)}")
        print(f"  Mobility    : {len(mobility_set)}")

    eval_kwargs = dict(
        id_set=id_set, c1_set=c1_set, c8_set=c8_set,
        combined_set=combined_set, credit_set=credit_set,
        ood_set=ood_set, mobility_set=mobility_set, quiet=quiet,
    )

    all_results = []

    if "random" in baselines_to_run:
        random.seed(0)
        all_results.append(evaluate_baseline(
            "random", lambda _: random_decision(), **eval_kwargs
        ))

    if "sft" in baselines_to_run:
        model, tok = load_model(SFT_MODEL, quiet=quiet)
        if model:
            all_results.append(evaluate_baseline(
                "sft", lambda p: generate_config(model, tok, p), **eval_kwargs
            ))
        else:
            print("  [SKIP] SFT model not found.")

    if "grpo" in baselines_to_run:
        model, tok = load_model(grpo_model_path, quiet=quiet)
        if model:
            all_results.append(evaluate_baseline(
                "grpo", lambda p: generate_config(model, tok, p), **eval_kwargs
            ))
        else:
            print("  [SKIP] GRPO model not found.")

    if not all_results:
        print("No baselines evaluated.")
        return

    # ΔU = MNU_GRPO − MNU_SFT
    delta_u = None
    mnu_map = {r["name"]: r["mnu"] for r in all_results}
    if "grpo" in mnu_map and "sft" in mnu_map:
        delta_u = mnu_map["grpo"] - mnu_map["sft"]

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    print(f"\nResults   → {OUT_JSON}")

    report = format_report(all_results, delta_u=delta_u)
    print("\n" + report)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"Report    → {OUT_REPORT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full four-family evaluation for the UE AI Agent"
    )
    parser.add_argument("--baselines", nargs="+",
                        choices=["random", "sft", "grpo"],
                        default=["random", "sft", "grpo"])
    parser.add_argument("--grpo-model",  default=GRPO_MODEL)
    parser.add_argument("--checkpoint",  default=None,
                        help="Evaluate a specific checkpoint instead of the final model")
    parser.add_argument("--quiet",       action="store_true")
    args = parser.parse_args()

    model_path = args.checkpoint if args.checkpoint else args.grpo_model
    main(args.baselines, model_path, args.quiet)