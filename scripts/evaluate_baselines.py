"""
evaluate_baselines.py  —  Baseline comparison for the UE AI Agent
------------------------------------------------------------------
Evaluates six policies on all four metric families and produces
paper-quality figures in the TUM colour palette.

Policies evaluated
──────────────────
  random          Uniform random valid (o,e) — lower bound
  always_offload  Always offload unconditionally (o=1, e=0) — upper bound on FVR
  always_escalate Escalate whenever the uplink is available
  few_shot        Untuned SmolLM2-135M with k in-context examples
  sft             SFT-only model  (models/sft-ue-merged, step 200)
  grpo            GRPO checkpoint-800  (models/grpo-ue-agent/checkpoint-800)
  exhaustive      Deterministic ground-truth policy — theoretical upper bound

Metrics reported (four families)
─────────────────────────────────
  Family 1 — Constraint safety
    FVR_C1        offload=True when β ≤ 20%
    FVR_C8        offload/escalate=True when η=weak signal
    FVR_combined  either violation type

  Family 2 — Offload quality (feasible region only)
    COR           correct offload rate
    Off-Precision
    Off-Recall
    Off-F1

  Family 3 — Escalation quality
    VER           valuable escalation rate (escalation precision)
    ER            escalation recall
    Esc-F1
    MER-HO        missed escalation on post-handover steps (not plotted)

  Family 4 — Utility
    MNU           mean normalised utility  U(o,e|s)
    ΔU            MNU_policy − MNU_SFT
    CCR           credit conservation ratio
    Parse fail

Output
──────
  models/baseline_eval_results.json    raw numbers
  models/fig_baseline_family1.{fmt}    constraint safety
  models/fig_baseline_family2.{fmt}    offload quality
  models/fig_baseline_family3.{fmt}    escalation quality
  models/fig_baseline_family4.{fmt}    utility
  models/fig_baseline_overview.{fmt}   radar + summary heatmap

Usage
─────
  python scripts/evaluate_baselines.py
  python scripts/evaluate_baselines.py --grpo-checkpoint models/grpo-ue-agent/checkpoint-800
  python scripts/evaluate_baselines.py --few-shot-k 3
  python scripts/evaluate_baselines.py --skip-eval        # plot from existing JSON
  python scripts/evaluate_baselines.py --fmt pdf --dpi 300
  python scripts/evaluate_baselines.py --quiet
"""

import os
import sys
import json
import time
import random
import argparse
import importlib.util

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
SFT_MODEL  = os.path.join(ROOT, "models", "sft-ue-merged")
GRPO_CKPT  = os.path.join(ROOT, "models", "grpo-ue-agent", "checkpoint-800")
TRACE_FILE = os.path.join(ROOT, "data",   "radio_parameters_moving_clean.csv")
OUT_JSON   = os.path.join(ROOT, "models", "baseline_eval_results.json")
OUT_DIR    = os.path.join(ROOT, "models")
BASE_MODEL = "HuggingFaceTB/SmolLM2-135M"

# ── Import evaluate.py helpers ────────────────────────────────────────────────
def _load_ev():
    path = os.path.join(SCRIPT_DIR, "evaluate.py")
    spec = importlib.util.spec_from_file_location("evaluate", path)
    m    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

ev = _load_ev()


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Baseline prediction functions
# ═══════════════════════════════════════════════════════════════════════════════

def pred_random(_sample):
    pairs = [(False, False), (True, False), (False, True)]
    o, e  = random.choice(pairs)
    return {"offload_decision": o, "escalation_flag": e}


def pred_always_offload(sample):
    # Unconditional offload — violates C1/C8 whenever constraints are active.
    # This is intentional: the baseline stress-tests the constraint penalty
    # and should show high FVR.
    return {"offload_decision": True, "escalation_flag": False}


def pred_always_escalate(sample):
    # Unconditional escalation — violates C8 whenever weak signal is active.
    # Symmetric with pred_always_offload: no constraint check, deliberately bad.
    return {"offload_decision": False, "escalation_flag": True}


def pred_exhaustive(sample):
    """
    True Optimal† oracle.  Uses full sample dict so C5 (post-handover
    escalation via eta_history) is evaluated correctly — MER-HO = 0%.
    """
    from data.generate_dataset import determine_optimal_decision

    # ── C5: post-handover step → always escalate ─────────────────────────────
    # Prefer the pre-computed "handover" boolean from build_mobility_set;
    # fall back to eta_history sequence comparison.
    post_handover = sample.get("handover", False)
    if not post_handover:
        eta_hist = sample.get("eta_history") or sample.get("network_history") or []
        post_handover = (len(eta_hist) >= 2 and eta_hist[-2] != eta_hist[-1])
    if post_handover:
        return {"offload_decision": False, "escalation_flag": True}

    # ── Optimal action for non-handover steps ────────────────────────────────
    app_raw = sample.get("app") or sample.get("app_type")
    # app_raw may already be the app dict — pass it through as-is
    o, e = determine_optimal_decision(
        sample["battery"],
        app_raw,
        sample["condition"],
        sample["gamma"],
    )
    return {"offload_decision": o, "escalation_flag": e}


# ── Few-shot examples (same as evaluate_robustness.py) ───────────────────────
FEW_SHOT_EXAMPLES = [
    {"prompt": ("Battery is at 8%. User just opened a mobile MOBA game. "
                "Network is stable. Energy credit is full (90%)."),
     "response": {"offload_decision": False, "escalation_flag": False}},
    {"prompt": ("Battery is at 75%. User just opened a 4K video stream. "
                "Network is weak signal. Energy credit is moderate (60%)."),
     "response": {"offload_decision": False, "escalation_flag": False}},
    {"prompt": ("Battery is at 65%. User just opened a AR/VR headset application. "
                "Network is stable. Energy credit is critically low (15%)."),
     "response": {"offload_decision": False, "escalation_flag": True}},
    {"prompt": ("Battery is at 80%. User just opened a AR/VR headset application. "
                "Network is stable. Energy credit is full (95%)."),
     "response": {"offload_decision": True, "escalation_flag": False}},
    {"prompt": ("Battery is at 55%. User just opened a background podcast. "
                "Network is congested. Energy credit is moderate (70%)."),
     "response": {"offload_decision": False, "escalation_flag": False}},
]


def build_few_shot_prompt(test_prompt, k=3):
    parts = []
    for ex in FEW_SHOT_EXAMPLES[:k]:
        parts.append(
            f"UE Context: {ex['prompt']}\n"
            f"Required keys (offload_decision, escalation_flag):\n"
            f"Decision:\n{json.dumps(ex['response'])}"
        )
    parts.append(
        f"UE Context: {test_prompt}\n"
        f"Required keys (offload_decision, escalation_flag):\n"
        f"Decision:\n{{"
    )
    return "\n\n".join(parts)


def make_few_shot_fn(base_model, tokenizer, k=3, max_new_tokens=40):
    import torch
    from scripts.reward_model import _repair_json, dict_raise_on_duplicates

    def _pred(prompt):
        full = build_few_shot_prompt(prompt, k=k)
        inputs = tokenizer(full, return_tensors="pt")
        with torch.no_grad():
            ids = base_model.generate(
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
        try:
            return json.loads(_repair_json(raw),
                              object_pairs_hook=dict_raise_on_duplicates)
        except Exception:
            return None

    return _pred


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Evaluation runner
# ═══════════════════════════════════════════════════════════════════════════════

def build_test_sets(n_id=200):
    """Build all six test sets used by evaluate_baseline."""
    print("  Building test sets …")
    return dict(
        id_set       = ev.build_id_test_set(n=n_id),
        c1_set       = ev.build_c1_set(n=100),
        c8_set       = ev.build_c8_set(n=100),
        combined_set = ev.build_combined_constraint_set(),
        credit_set   = ev.build_credit_floor_set(n=80),
        ood_set      = ev.build_ood_test_set(),
        mobility_set = ev.build_mobility_set(TRACE_FILE),
    )


def run_all(policies, test_sets, quiet=False):
    """
    Run evaluate_baseline for every policy.

    policies: list of (name, get_pred_fn) where get_pred_fn(sample) → dict|None
    Returns:  list of result dicts (one per policy), in input order.
    """
    results = []
    for name, fn in policies:
        r = ev.evaluate_baseline(
            name         = name,
            get_pred_fn  = fn,
            quiet        = quiet,
            **test_sets,
        )
        results.append(r)
        if not quiet:
            print(f"    FVR={r['fvr_combined']:.1%}  COR={r['cor']:.3f}  "
                  f"VER={r['ver']:.3f}  ER={r['er']:.3f}  MNU={r['mnu']:+.4f}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  TUM colour palette
# ═══════════════════════════════════════════════════════════════════════════════

# One solid colour per policy — same assignment logic as plot_robustness.py
# so all paper figures use a consistent visual vocabulary.
PALETTE = {
    "random":          "#888888",   # neutral grey
    "always_offload":  "#5E94D4",   # consistent always-offload colour
    "always_escalate": "#14519A",   # TUM blue dark-4
    "few_shot":        "#6955E2",   # TUM blue bright-dark
    "sft":             "#9B468D",   # TUM pink-dark
    "grpo":            "#0A2D57",   # TUM blue dark-1  (primary result)
    "exhaustive":      "#B6ACF1",   # TUM blue bright-1 (ceiling)
}

LABELS = {
    "random":          "Random",
    "always_offload":  "Always-offload",
    "always_escalate": "Always-escalate",
    "few_shot":        "Few-shot (k=3)",
    "sft":             "SFT",
    "grpo":            "GRPO",
    "exhaustive":      "Exhaustive",
}

# Canonical display order for all charts
POLICY_ORDER = [
    "random", "always_offload", "always_escalate",
    "few_shot", "sft", "grpo", "exhaustive",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Matplotlib style
# ═══════════════════════════════════════════════════════════════════════════════

def apply_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size":          8,
        "axes.titlesize":     8,
        "axes.labelsize":     8,
        "xtick.labelsize":    7,
        "ytick.labelsize":    7,
        "legend.fontsize":    7,
        "axes.linewidth":     0.6,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "grid.linewidth":     0.4,
        "grid.alpha":         0.4,
        "patch.linewidth":    0.0,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Plot helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _results_to_dict(results):
    """Index results by policy name for easy lookup."""
    return {r["name"]: r for r in results}


def _get(rd, name, metric):
    v = rd.get(name, {}).get(metric)
    return v if v is not None else 0.0


def _bar_chart(ax, metric, results_dict, title, ylabel,
               ylim=(0, 1.05), pct=True,
               lower_is_better=False,
               policies=None):
    """
    Horizontal bar chart: one bar per policy, coloured by PALETTE.
    GRPO bar is outlined in dark navy to make it stand out.
    """
    import matplotlib.ticker as mtick
    import numpy as np

    if policies is None:
        policies = [p for p in POLICY_ORDER if p in results_dict]

    vals   = [_get(results_dict, p, metric) for p in policies]
    labels = [LABELS.get(p, p)              for p in policies]
    colors = [PALETTE.get(p, "#888888")     for p in policies]
    y      = np.arange(len(policies))

    bars = ax.barh(y, vals, color=colors, edgecolor="none", linewidth=0, zorder=3)

    # Outline GRPO
    if "grpo" in policies:
        idx = policies.index("grpo")
        bars[idx].set_edgecolor("#FFFFFF")
        bars[idx].set_linewidth(0.9)
        bars[idx].set_zorder(4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(xlim := (ylim[0], ylim[1]))
    ax.set_xlabel(ylabel, fontsize=8)
    ax.xaxis.grid(True, zorder=0, color="#CCCCCC", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.invert_yaxis()   # top = first policy

    if pct:
        ax.xaxis.set_major_formatter(
            mtick.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Value labels on bars
    for bar, val in zip(bars, vals):
        w = bar.get_width()
        if w > 0.01:
            ax.text(
                min(w + 0.01, ylim[1] - 0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}" if pct else f"{val:+.3f}",
                va="center", ha="left", fontsize=6,
                color="#333333",
            )


def _vbar_chart(ax, metric, results_dict, title, ylabel,
               ylim=(0, 1.05), pct=True, lower_is_better=False, policies=None):
    """
    Vertical grouped bar chart: algorithms on x-axis, one bar per policy.
    Used for single-metric panels where policy comparison is the primary story.
    """
    import matplotlib.ticker as mtick
    import numpy as np

    if policies is None:
        policies = [p for p in POLICY_ORDER if p in results_dict]

    vals   = [_get(results_dict, p, metric) for p in policies]
    labels = [LABELS.get(p, p)              for p in policies]
    colors = [PALETTE.get(p, "#888888")     for p in policies]
    x      = np.arange(len(policies))
    w      = 0.65

    bars = ax.bar(x, vals, width=w, color=colors,
                  edgecolor="none", linewidth=0, zorder=3)

    # Outline GRPO
    if "grpo" in policies:
        idx = policies.index("grpo")
        bars[idx].set_edgecolor("#FFFFFF")
        bars[idx].set_linewidth(0.9)
        bars[idx].set_zorder(4)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=25, ha="right")
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.yaxis.grid(True, zorder=0, color="#CCCCCC", linewidth=0.4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if pct:
        ax.yaxis.set_major_formatter(
            mtick.FuncFormatter(lambda v, _: f"{v:.0%}"))

    # Value labels above bars — guard against axis edge clip
    for bar, val in zip(bars, vals):
        h = bar.get_height()
        if h > 0.005:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(h + 0.015, ylim[1] - 0.02),
                f"{val:.1%}" if pct else f"{val:+.3f}",
                ha="center", va="bottom", fontsize=6, color="#333333",
            )


def _add_legend(fig, policies, ypos=-0.04):
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(facecolor=PALETTE.get(p, "#888"),
                       edgecolor="none", label=LABELS.get(p, p))
        for p in policies
    ]
    fig.legend(handles, [LABELS.get(p, p) for p in policies],
               ncol=len(policies), loc="lower center",
               bbox_to_anchor=(0.5, ypos),
               fontsize=7, frameon=True, framealpha=0.9,
               edgecolor="#CCCCCC", columnspacing=1.0,
               handlelength=1.2, handleheight=0.9)


def _save(fig, name, out_dir, fmt, dpi):
    path = os.path.join(out_dir, f"{name}.{fmt}")
    fig.savefig(path, dpi=dpi)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  Saved → {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Four metric-family figures
# ═══════════════════════════════════════════════════════════════════════════════

def fig_family1(rd, out_dir, fmt, dpi, policies):
    """Family 1 — Feasibility safety: FVR combined only, algorithms on x-axis."""
    import matplotlib.pyplot as plt
    apply_style()

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    fig.subplots_adjust(bottom=0.32, top=0.96, left=0.14, right=0.96)

    _vbar_chart(ax, "fvr_combined", rd,
                "FVR — Feasibility Violation Rate  (↓ lower is better)",
                "FVR", ylim=(0, 0.70), lower_is_better=True, policies=policies)

    return _save(fig, "fig_baseline_family1", out_dir, fmt, dpi)


def fig_family2(rd, out_dir, fmt, dpi, policies):
    """Family 2 — Offload quality: COR only, algorithms on x-axis."""
    import matplotlib.pyplot as plt
    apply_style()

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    fig.subplots_adjust(bottom=0.32, top=0.96, left=0.14, right=0.96)

    _vbar_chart(ax, "cor", rd,
                "Correct Offload Rate  (feasible region, ↑ higher is better)",
                "COR", ylim=(0, 1.12), policies=policies)

    return _save(fig, "fig_baseline_family2", out_dir, fmt, dpi)


def fig_family3(rd, out_dir, fmt, dpi, policies):
    """
    Family 3 — Escalation quality: VER only, algorithms on x-axis.
    MER-HO is excluded from the figure (reported in the text if needed).
    """
    import matplotlib.pyplot as plt
    apply_style()

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    fig.subplots_adjust(bottom=0.32, top=0.96, left=0.12, right=0.97)

    _vbar_chart(ax, "ver", rd,
                "Valuable Escalation Rate (↑)",
                "VER", ylim=(0, 1.12), policies=policies)

    return _save(fig, "fig_baseline_family3", out_dir, fmt, dpi)


def fig_credit_r(rd, out_dir, fmt, dpi, policies):
    """
    Three-panel figure: CCR, credit_per_step, and mean R(o,s).

    CCR (Option A) = total_credit_spent / n_offload_steps
        Credit cost per offload decision; 0 for non-offloading policies.
        Lower = each offload is cheaper (lighter app / better network).

    credit_per_step (Option C) = total_credit_spent / total_steps
        Mean credit intensity per time step, independent of offload frequency.
        Pairs with R(o,s): high credit_per_step + high R = efficient spending;
        high credit_per_step + low R = wasteful offloading.

    mean_r = mean R(o,s) offload utility over the ID test set.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np
    apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.0),
                             gridspec_kw={"wspace": 0.46})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.96, bottom=0.26, left=0.07, right=0.97)

    policies_use = [p for p in policies if p in rd]
    labels  = [LABELS.get(p, p)       for p in policies_use]
    colors  = [PALETTE.get(p, "#888") for p in policies_use]
    x = np.arange(len(policies_use))
    w = 0.65

    def _outline_grpo(bars_list):
        if "grpo" in policies_use:
            idx = policies_use.index("grpo")
            bars_list[idx].set_edgecolor("#FFFFFF")
            bars_list[idx].set_linewidth(0.9)
            bars_list[idx].set_zorder(4)

    def _xticks(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=25, ha="right")
        ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    def _val_labels(ax, bars_list, vals, fmt_str):
        for bar, val in zip(bars_list, vals):
            h = bar.get_height()
            if abs(h) > 1e-6:
                off = max(abs(h) * 0.06, 0.0001)
                ypos = h + off if val >= 0 else h - off
                va   = "bottom" if val >= 0 else "top"
                ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                        fmt_str.format(val),
                        ha="center", va=va, fontsize=5.8,
                        color="#333333", zorder=7)

    # ── Panel 1: CCR (credit per offload step) ────────────────────────────────
    ccr_vals = [rd.get(p, {}).get("ccr") or 0.0 for p in policies_use]
    bars1 = axes[0].bar(x, ccr_vals, width=w, color=colors,
                        edgecolor="none", zorder=3)
    _outline_grpo(bars1)
    _xticks(axes[0])
    ccr_max = max(max(ccr_vals) * 1.18, 0.001)
    axes[0].set_ylim(0, ccr_max)
    axes[0].set_ylabel("Credit / offload step", fontsize=8)
    axes[0].yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f"{v:.4f}"))
    _val_labels(axes[0], bars1, ccr_vals, fmt_str="{:.4f}")

    # ── Panel 2: credit_per_step ──────────────────────────────────────────────
    cps_vals = [rd.get(p, {}).get("credit_per_step") or 0.0 for p in policies_use]
    bars2 = axes[1].bar(x, cps_vals, width=w, color=colors,
                        edgecolor="none", zorder=3)
    _outline_grpo(bars2)
    _xticks(axes[1])
    cps_max = max(max(cps_vals) * 1.18, 0.001)
    axes[1].set_ylim(0, cps_max)
    axes[1].set_ylabel("Credit / step", fontsize=8)
    axes[1].yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f"{v:.4f}"))
    _val_labels(axes[1], bars2, cps_vals, fmt_str="{:.4f}")

    # ── Panel 3: mean R(o,s) ─────────────────────────────────────────────────
    r_vals = [rd.get(p, {}).get("mean_r") or 0.0 for p in policies_use]
    bars3 = axes[2].bar(x, r_vals, width=w, color=colors,
                        edgecolor="none", zorder=3)
    _outline_grpo(bars3)
    _xticks(axes[2])
    axes[2].axhline(0, color="#888888", linewidth=0.6, linestyle=":", zorder=2, alpha=0.7)
    r_min = min(min(r_vals) - 0.05, -0.15)
    r_max = max(max(r_vals) + 0.05,  0.15)
    axes[2].set_ylim(r_min, r_max)
    axes[2].set_ylabel("R(o, s)", fontsize=8)
    axes[2].yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f"{v:+.3f}"))
    _val_labels(axes[2], bars3, r_vals, fmt_str="{:+.3f}")

    return _save(fig, "fig_credit_r", out_dir, fmt, dpi)


def fig_mnu(rd, out_dir, fmt, dpi, policies):
    """
    Standalone MNU figure — Mean Normalised Utility U(o,e|s) per policy.
    Separated from R(o,s) to avoid the previous conflation of the two.
    MNU = mean over the ID test set of the full objective U(o,e|s):
        U = wR·R + wV·V − wC·C − wP·P − μ·Φ
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    apply_style()

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    fig.subplots_adjust(bottom=0.32, top=0.96, left=0.16, right=0.96)

    _vbar_chart(ax, "mnu", rd,
                "Mean Normalised Utility  U(o, e | s)",
                "MNU", ylim=(-0.05, 0.28), pct=False, policies=policies)
    ax.axhline(0, color="#888888", linewidth=0.6, linestyle=":", zorder=0)
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f"{v:+.2f}"))

    return _save(fig, "fig_mnu", out_dir, fmt, dpi)


def fig_family4(rd, out_dir, fmt, dpi, policies):
    """Family 4 — Utility: MNU, ΔU, parse-fail."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np
    apply_style()

    # ΔU = MNU_policy - MNU_SFT
    sft_mnu = _get(rd, "sft", "mnu")
    for name in rd:
        rd[name]["delta_u"] = round(_get(rd, name, "mnu") - sft_mnu, 4)

    all_mnu = [_get(rd, p, "mnu") for p in policies]
    all_du  = [_get(rd, p, "delta_u") for p in policies]
    mnu_min = min(min(all_mnu) - 0.05, -0.15)
    mnu_max = max(max(all_mnu) + 0.08,  0.40)
    du_min  = min(min(all_du)  - 0.05, -0.15)
    du_max  = max(max(all_du)  + 0.08,  0.25)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
    fig.subplots_adjust(wspace=0.65, bottom=0.20, top=0.88)

    # MNU
    _bar_chart(axes[0], "mnu", rd, "Mean Normalised Utility (MNU)",
               "MNU", ylim=(mnu_min, mnu_max), pct=False, policies=policies)
    axes[0].axvline(0, color="#888", linewidth=0.6, zorder=0)
    axes[0].xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f"{v:+.2f}"))

    # ΔU
    _bar_chart(axes[1], "delta_u", rd, "ΔU vs SFT baseline",
               "ΔU", ylim=(du_min, du_max), pct=False, policies=policies)
    axes[1].axvline(0, color="#888", linewidth=0.6, zorder=0)
    axes[1].xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f"{v:+.2f}"))

    # Parse fail (lower is better)
    _bar_chart(axes[2], "parse_fail", rd, "Parse Fail Rate  (↓)",
               "Parse fail", ylim=(0, 0.55),
               lower_is_better=True, policies=policies)

    return _save(fig, "fig_baseline_family4", out_dir, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# 6b.  Combined paper figures (two SVG outputs for the paper)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_mnu_cor_ver(rd, out_dir, fmt="svg", dpi=300, policies=None):
    """
    SVG 1 — Three panels side by side: MNU | COR | VER.

    Paper-quality single-file figure, IEEE double-column width (7.2 in).
    Vertical bar charts; policies on x-axis; one bar per policy.
    Suitable for the System Performance section of the paper.
    """
    import matplotlib
    matplotlib.use("svg" if fmt == "svg" else "Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np

    apply_style()
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family":      "serif",
    })

    if policies is None:
        policies = [p for p in POLICY_ORDER if p in rd]

    labels = [LABELS.get(p, p)       for p in policies]
    colors = [PALETTE.get(p, "#888") for p in policies]
    x = np.arange(len(policies))
    w = 0.65

    FIG_W = 7.2   # IEEE double-column
    FIG_H = 2.9

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H),
                             gridspec_kw={"wspace": 0.42})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.90, bottom=0.30, left=0.08, right=0.98)

    panel_specs = [
        ("mnu",  r"MNU  $U(o,e\,|\,s)$",              (-0.05, 0.30), False),
        ("cor",  r"COR  (feasible region, $\uparrow$)", (0.00,  1.12), True),
        ("ver",  r"VER  ($\uparrow$)",                   (0.00,  1.12), True),
    ]

    for ax, (metric, ylabel, ylim, is_pct) in zip(axes, panel_specs):
        vals = [rd.get(p, {}).get(metric) or 0.0 for p in policies]

        bars = ax.bar(x, vals, width=w, color=colors,
                      edgecolor="none", linewidth=0, zorder=3)

        # Outline GRPO bar for visual emphasis
        if "grpo" in policies:
            idx = policies.index("grpo")
            bars[idx].set_edgecolor("#FFFFFF")
            bars[idx].set_linewidth(0.9)
            bars[idx].set_zorder(4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5, rotation=28, ha="right")
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if is_pct:
            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(lambda v, _: f"{v:.0%}"))
        else:
            ax.axhline(0, color="#888888", linewidth=0.5,
                       linestyle=":", zorder=0, alpha=0.7)
            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(lambda v, _: f"{v:+.2f}"))

        # Value labels above bars
        for bar, val in zip(bars, vals):
            h = bar.get_height()
            if abs(h) > 1e-4:
                off = max(abs(ylim[1] - ylim[0]) * 0.025, 1e-4)
                ypos = h + off if val >= 0 else h - off * 1.5
                va   = "bottom" if val >= 0 else "top"
                label_str = f"{val:.1%}" if is_pct else f"{val:+.3f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(ypos, ylim[1] * 0.97),
                    label_str,
                    ha="center", va=va, fontsize=5.5, color="#333333", zorder=7,
                )

    return _save(fig, "fig_mnu_cor_ver", out_dir, fmt, dpi)


def fig_credit_fvr(rd, out_dir, fmt="svg", dpi=300, policies=None):
    """
    SVG 2 — Three panels: credit_per_step | mean_R | FVR.

    Credit evaluation (two panels) + feasibility violation rate (one panel).
    CCR (credit per offload step) removed; credit_per_step is kept.
    Paper-quality, IEEE double-column width (7.2 in).
    """
    import matplotlib
    matplotlib.use("svg" if fmt == "svg" else "Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np

    apply_style()
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family":      "serif",
    })

    if policies is None:
        policies = [p for p in POLICY_ORDER if p in rd]

    labels = [LABELS.get(p, p)       for p in policies]
    colors = [PALETTE.get(p, "#888") for p in policies]
    x = np.arange(len(policies))
    w = 0.65

    FIG_W = 7.2
    FIG_H = 2.9

    fig, axes = plt.subplots(1, 3, figsize=(FIG_W, FIG_H),
                             gridspec_kw={"wspace": 0.48})
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.90, bottom=0.30, left=0.08, right=0.98)

    # Collect data ranges for dynamic ylim
    cps_vals = [rd.get(p, {}).get("credit_per_step") or 0.0 for p in policies]
    r_vals   = [rd.get(p, {}).get("mean_r")          or 0.0 for p in policies]
    fvr_vals = [rd.get(p, {}).get("fvr_combined")    or 0.0 for p in policies]

    panel_data = [
        (cps_vals, "Credit / step",
         (0, max(max(cps_vals) * 1.22, 0.001)),
         "{:.4f}", False),
        (r_vals,   r"$R(o,\,s)$",
         (min(min(r_vals) - 0.06, -0.12), max(max(r_vals) + 0.06, 0.12)),
         "{:+.3f}", False),
        (fvr_vals, r"FVR ($\downarrow$)",
         (0, min(max(max(fvr_vals) * 1.22, 0.05), 1.0)),
         "{:.1%}", True),
    ]

    for ax, (vals, ylabel, ylim, fmt_str, is_pct) in zip(axes, panel_data):
        bars = ax.bar(x, vals, width=w, color=colors,
                      edgecolor="none", linewidth=0, zorder=3)

        if "grpo" in policies:
            idx = policies.index("grpo")
            bars[idx].set_edgecolor("#FFFFFF")
            bars[idx].set_linewidth(0.9)
            bars[idx].set_zorder(4)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.5, rotation=28, ha="right")
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel, fontsize=7.5)
        ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if is_pct:
            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(lambda v, _: f"{v:.0%}"))
        elif "+" in fmt_str:
            # signed axis — add zero line
            ax.axhline(0, color="#888888", linewidth=0.5,
                       linestyle=":", zorder=0, alpha=0.7)
            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(lambda v, _: f"{v:+.3f}"))
        else:
            ax.yaxis.set_major_formatter(
                mtick.FuncFormatter(lambda v, _: f"{v:.4f}"))

        # Value labels
        for bar, val in zip(bars, vals):
            h = bar.get_height()
            if abs(h) > 1e-6:
                span = ylim[1] - ylim[0]
                off  = span * 0.030
                ypos = h + off if val >= 0 else h - off * 1.5
                va   = "bottom" if val >= 0 else "top"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(ypos, ylim[1] * 0.97) if val >= 0 else max(ypos, ylim[0] * 0.97),
                    fmt_str.format(val),
                    ha="center", va=va, fontsize=5.5, color="#333333", zorder=7,
                )

    return _save(fig, "fig_credit_fvr", out_dir, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Overview figure: radar + grouped bar summary
# ═══════════════════════════════════════════════════════════════════════════════

def fig_overview(rd, out_dir, fmt, dpi, policies):
    """
    Overview figure: grouped vertical bar chart of all key metrics,
    one cluster per metric, one bar per policy.

    FVR and MER-HO are inverted (1 - value) so that all bars point
    upward = better, keeping the visual consistent.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import numpy as np
    apply_style()

    # Seven key metrics; FVR is inverted so taller = better (MER-HO excluded)
    key_metrics = [
        ("fvr_combined", "FVR\n(↓ inv.)",  True),
        ("cor",          "COR",             False),
        ("offload_f1",   "Off-F1",          False),
        ("ver",          "VER",             False),
        ("er",           "ER",              False),
        ("esc_f1",       "Esc-F1",          False),
        ("mnu",          "MNU",             False),
    ]
    n_m  = len(key_metrics)
    n_p  = len(policies)
    w    = 0.72 / n_p
    x    = np.arange(n_m)
    offs = np.linspace(-(n_p - 1) / 2, (n_p - 1) / 2, n_p) * w

    # Figure is wider now (no radar) so bars have more room
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    fig.subplots_adjust(bottom=0.20, top=0.97, left=0.07, right=0.98)

    for i, p in enumerate(policies):
        r    = rd.get(p, {})
        vals = [(1 - (r.get(m) or 0)) if inv else (r.get(m) or 0)
                for m, _, inv in key_metrics]
        bars = ax.bar(
            x + offs[i], vals,
            width=w * 0.92,
            color=PALETTE.get(p, "#888"),
            edgecolor="none",
            label=LABELS.get(p, p),
            zorder=3,
        )
        # White separator outline on GRPO bars to make them pop
        if p == "grpo":
            for b in bars:
                b.set_edgecolor("#FFFFFF")
                b.set_linewidth(0.9)
                b.set_zorder(4)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [m[1] for m in key_metrics],
        fontsize=7.5, ha="center")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score  (FVR inverted: taller = better)",
                   fontsize=7.5)
    ax.yaxis.grid(True, color="#CCCCCC", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend in a single row tight under the bars
    _add_legend(fig, policies, ypos=-0.19)

    return _save(fig, "fig_baseline_overview", out_dir, fmt, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and plot baseline comparison for the UE AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--grpo-checkpoint", default=GRPO_CKPT)
    parser.add_argument("--sft-model",       default=SFT_MODEL)
    parser.add_argument("--few-shot-k",      type=int, default=3)
    parser.add_argument("--n-id",            type=int, default=200,
                        help="ID test set size (default: 200)")
    parser.add_argument("--skip-eval",       action="store_true",
                        help="Skip evaluation; load existing JSON and plot only")
    parser.add_argument("--out",             default=OUT_DIR)
    parser.add_argument("--fmt",             default="svg",
                        choices=["pdf","png","svg"])
    parser.add_argument("--dpi",             type=int, default=300)
    parser.add_argument("--quiet",           action="store_true")
    parser.add_argument("--no-exhaustive",   action="store_true",
                        help="Skip the exhaustive baseline (faster evaluation)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ── Load or run evaluation ────────────────────────────────────────────────
    if args.skip_eval and os.path.exists(OUT_JSON):
        print(f"Loading existing results from {OUT_JSON}")
        with open(OUT_JSON, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        test_sets = build_test_sets(n_id=args.n_id)

        # ── Load neural models ────────────────────────────────────────────────
        sft_model, sft_tok   = None, None
        grpo_model, grpo_tok = None, None
        fs_model, fs_tok     = None, None

        print("\nLoading models …")

        sft_model, sft_tok = ev.load_model(args.sft_model, quiet=args.quiet)

        try:
            from peft import PeftModel
            print(f"  Loading GRPO checkpoint: {args.grpo_checkpoint}")
            base = AutoModelForCausalLM.from_pretrained(
                args.sft_model, torch_dtype=torch.float32,
                device_map={"": "cpu"})
            grpo_model = PeftModel.from_pretrained(
                base, args.grpo_checkpoint).merge_and_unload()
            grpo_model.eval()
            grpo_tok = AutoTokenizer.from_pretrained(args.sft_model)
            grpo_tok.pad_token = grpo_tok.eos_token
            print("  GRPO ready.")
        except Exception as e:
            print(f"  [WARN] Could not load GRPO: {e}")

        try:
            print(f"  Loading few-shot base ({BASE_MODEL})")
            fs_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, torch_dtype=torch.float32,
                device_map={"": "cpu"})
            fs_model.eval()
            fs_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
            fs_tok.pad_token = fs_tok.eos_token
            print(f"  Few-shot ready (k={args.few_shot_k}).")
        except Exception as e:
            print(f"  [WARN] Could not load few-shot model: {e}")

        # ── Build policy list ─────────────────────────────────────────────────
        # All policy functions receive the full sample dict.
        # Non-LLM baselines use structured fields directly.
        # LLM-based policies extract s["prompt"] internally.
        policies = [
            ("random", pred_random),
        ]

        def _always_offload_fn(s):
            return pred_always_offload(s)

        def _always_escalate_fn(s):
            return pred_always_escalate(s)

        policies += [
            ("always_offload",  _always_offload_fn),
            ("always_escalate", _always_escalate_fn),
        ]

        if not args.no_exhaustive:
            from data.generate_dataset import determine_optimal_decision, app_types

            # Build app lookup by display name for the structured sample fields.
            # The sample dict already carries app_type as a display name
            # (e.g. "AR/VR headset application") so no parse_context needed.
            APP_BY_DISPLAY = {a["name"]: a for a in app_types}
            # Short-key fallback (e.g. "ar/vr") in case app_type is stored
            # as a key rather than a display name.
            _KEY_FRAGMENTS = [
                ("ar/vr",   "AR/VR headset application"),
                ("game",    "mobile MOBA game"),
                ("video",   "4K video stream"),
                ("update",  "system update"),
                ("podcast", "background podcast"),
            ]
            APP_BY_KEY = {k: APP_BY_DISPLAY[v] for k, v in _KEY_FRAGMENTS}

            def _oracle_fn(s):
                """
                True Optimal oracle — receives full sample dict, so C5
                (post-handover escalation) can be evaluated correctly using
                eta_history.  MER-HO should now be 0% for this policy.
                """
                battery   = s["battery"]
                condition = s["condition"]
                gamma     = s["gamma"]
                app_raw   = s.get("app_type") or s.get("app")

                # app_raw may already be the app dict (from generate_dataset),
                # or a display-name string, or a short key like "ar/vr".
                if isinstance(app_raw, dict):
                    app = app_raw
                elif isinstance(app_raw, str):
                    app = (
                        APP_BY_DISPLAY.get(app_raw)
                        or APP_BY_KEY.get(app_raw)
                    )
                    if app is None:
                        app_low = app_raw.lower()
                        app = next(
                            (a for a in app_types if a["name"].lower() in app_low),
                            app_types[0],
                        )
                else:
                    app = app_types[0]   # fallback: podcast

                # ── C5: post-handover step → always escalate ─────────────────
                # build_mobility_set stores a pre-computed boolean "handover"
                # field in each sample dict (m.get("handover", False)).
                # Fall back to eta_history sequence comparison if that field
                # is absent (future dataset versions).
                handover_flag = s.get("handover", False)
                if not handover_flag:
                    eta_hist = s.get("eta_history") or s.get("network_history") or []
                    handover_flag = (
                        len(eta_hist) >= 2 and eta_hist[-2] != eta_hist[-1]
                    )
                if handover_flag:
                    return {"offload_decision": False, "escalation_flag": True}

                # ── C1 / C8 / optimal action ─────────────────────────────────
                o, e = determine_optimal_decision(battery, app, condition, gamma)
                return {"offload_decision": o, "escalation_flag": e}

            policies.append(("exhaustive", _oracle_fn))

        if fs_model is not None:
            _fs_fn = make_few_shot_fn(fs_model, fs_tok, k=args.few_shot_k)
            policies.append(("few_shot", lambda s, _f=_fs_fn: _f(s["prompt"])))

        if sft_model is not None:
            policies.append(
                ("sft",  lambda s: ev.generate_config(sft_model,  sft_tok,  s["prompt"])))

        if grpo_model is not None:
            policies.append(
                ("grpo", lambda s: ev.generate_config(grpo_model, grpo_tok, s["prompt"])))

        # ── Run ───────────────────────────────────────────────────────────────
        print("\nEvaluating …")
        results = run_all(policies, test_sets, quiet=args.quiet)

        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
        print(f"\nResults → {OUT_JSON}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")

    rd       = _results_to_dict(results)
    policies = [p for p in POLICY_ORDER if p in rd]

    print(f"\nGenerating figures ({args.fmt}, {args.dpi} DPI) → {args.out}")
    out_paths = []
    out_paths.append(fig_family1   (rd, args.out, args.fmt, args.dpi, policies))
    out_paths.append(fig_family2   (rd, args.out, args.fmt, args.dpi, policies))
    out_paths.append(fig_family3   (rd, args.out, args.fmt, args.dpi, policies))
    out_paths.append(fig_family4   (rd, args.out, args.fmt, args.dpi, policies))
    out_paths.append(fig_credit_r  (rd, args.out, args.fmt, args.dpi, policies))
    out_paths.append(fig_mnu       (rd, args.out, args.fmt, args.dpi, policies))
    out_paths.append(fig_overview  (rd, args.out, args.fmt, args.dpi, policies))
    # Combined paper figures
    out_paths.append(fig_mnu_cor_ver(rd, args.out, args.fmt, args.dpi, policies))
    out_paths.append(fig_credit_fvr (rd, args.out, args.fmt, args.dpi, policies))

    print(f"\nDone. {len(out_paths)} figure(s) saved.")


if __name__ == "__main__":
    main()