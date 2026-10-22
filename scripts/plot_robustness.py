"""
plot_robustness.py  —  Paper-quality robustness result figures
--------------------------------------------------------------
Reads robustness_report.txt (and optionally robustness_results.json)
and generates publication-ready figures using the TU Munich colour palette.

  Figure 1  Axis 1 — OOD generalisation (1a–1e)
            2×2 panel: accuracy, CVR, offload F1, escalation F1

  Figure 2  Axis 3 — Credit sensitivity (3a, 3b)
            1×3 panel: accuracy, escalation F1, MNU

  Figure 3  Axis 6 — Linguistic robustness (6a–6c)
            1 panel: accuracy only  (CVR, parse-fail reported in text)

  Figure 4  Escalation benefit near γ_esc boundary
            Line plot: accuracy + escalation rate vs. distance from γ_esc=20%
            Reads 3a_sweep from robustness_results.json

  Figure 5  Summary heatmap — all baselines × all axes (accuracy)

TUM colour palette (from brand guide)
──────────────────────────────────────
  TUM blue dark variants:   #0A2D57  #0E396E  #114584  #14519A
  TUM pink variants:        #B55CA5  #9B468D  #C680BB  #D6A4CE
  TUM blue bright variants: #8F81EA  #6955E2  #B6ACF1
  Neutral greys for exhaustive and random baselines.

Usage
─────
  python scripts/plot_robustness.py
  python scripts/plot_robustness.py --report models/robustness_report.txt
  python scripts/plot_robustness.py --out    models/ --fmt pdf --dpi 300
  python scripts/plot_robustness.py --figs   axis1 axis3
"""

import os
import re
import sys
import argparse
import numpy as np

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT           = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_REPORT = os.path.join(ROOT, "models", "robustness_report.txt")
DEFAULT_OUT    = os.path.join(ROOT, "models")

# ── Column order in the report ────────────────────────────────────────────────
METRIC_COLS = ["acc", "cvr", "off_f1", "ver", "er", "esc_f1", "mnu", "parse_fail"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Report parser
# ═══════════════════════════════════════════════════════════════════════════════

def _pct_or_float(s):
    s = s.strip()
    return float(s.rstrip("%")) / 100.0 if s.endswith("%") else float(s)


def parse_report(path):
    """
    Parse robustness_report.txt into a nested dict:
        {axis_id: {baseline_name: {metric: float}}}

    Tolerant of mixed line endings (\r\n / \n / bare \r) and any amount
    of leading/trailing whitespace on axis-header lines.
    Column order is fixed by METRIC_COLS.
    Also supports loading from robustness_results.json when the .txt
    file cannot be found or parsed (pass the .json path instead).
    """
    import json as _json

    # ── JSON fallback: if path points to a .json, load aggregated data ────────
    if path.endswith(".json"):
        return _parse_report_json(path)

    # ── Also try the sibling .json first for robustness ───────────────────────
    json_path = path.replace(".txt", ".json").replace(
        "robustness_report", "robustness_results")
    if not os.path.exists(path) and os.path.exists(json_path):
        return _parse_report_json(json_path)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    # Normalise all line endings to plain \n, strip stray \r throughout
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    data         = {}
    current_axis = None
    in_data      = False

    for line in text.split("\n"):
        # Strip trailing whitespace including any residual \r
        line = line.rstrip()

        # Axis header — tolerate any leading whitespace, e.g. "  Axis 6a"
        m = re.match(r"^\s*Axis\s+(\w+)\s*$", line)
        if m:
            current_axis = m.group(1)
            data[current_axis] = {}
            in_data = False
            continue

        # Column-header row signals start of data block
        if "Baseline" in line and "Acc" in line:
            in_data = True
            continue

        # Separator / blank lines — skip but stay in data block
        if re.match(r"^\s*[=\-]+\s*$", line) or not line.strip():
            continue

        # Data row: split on 2+ spaces to preserve underscore names
        if in_data and current_axis:
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) >= len(METRIC_COLS) + 1:
                bl   = parts[0]
                # Backward-compat: reports generated before oracle→exhaustive rename
                if bl == "oracle":
                    bl = "exhaustive"
                vals = parts[1: len(METRIC_COLS) + 1]
                try:
                    data[current_axis][bl] = {
                        k: _pct_or_float(v)
                        for k, v in zip(METRIC_COLS, vals)
                    }
                except ValueError:
                    pass  # skip malformed rows silently

    return data


def _parse_report_json(json_path):
    """
    Build the same {axis: {baseline: {metric: float}}} dict directly from
    robustness_results.json, mapping the JSON metric names to METRIC_COLS.
    """
    import json as _json

    KEY_MAP = {
        # JSON key        → plot short key
        "accuracy":   "acc",
        "cvr":        "cvr",
        "offload_f1": "off_f1",
        "ver":        "ver",
        "er":         "er",
        "esc_f1":     "esc_f1",
        "mnu":        "mnu",
        "parse_fail": "parse_fail",
    }
    with open(json_path, "r", encoding="utf-8") as f:
        raw = _json.load(f)

    data = {}
    for axis_id, axis_data in raw.items():
        if axis_id == "3a_sweep":
            continue
        if not isinstance(axis_data, dict):
            continue
        data[axis_id] = {}
        for bl, metrics in axis_data.items():
            if not isinstance(metrics, dict):
                continue
            canonical_bl = "exhaustive" if bl == "oracle" else bl
            data[axis_id][canonical_bl] = {
                short: metrics.get(long, 0.0)
                for long, short in KEY_MAP.items()
            }
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  TUM colour palette
# ═══════════════════════════════════════════════════════════════════════════════
#
# Baseline → solid fill colour from the TUM brand palette.
# Seven baselines, seven distinct colours, no hatching needed.
#
# Assignment rationale:
#   random          → mid-grey (neutral, lowest-information baseline)
#   always_offload  → TUM blue light (clearly distinct from GRPO's near-black)
#   always_escalate → TUM blue dark-4 (related hue, lighter shade)
#   few_shot        → TUM blue bright (perceptually distinct from dark blues)
#   sft             → TUM pink-dark (warm counterpoint to cool blues)
#   grpo            → TUM blue dark-1 (darkest, primary result — stands out)
#   exhaustive      → light grey (theoretical ceiling, intentionally recessive)

BASELINES = [
    # (key,             display label,       solid colour)
    ("random",          "Random",            "#888888"),
    ("always_offload",  "Always-offload",    "#5E94D4"),   # TUM blue light (well-separated from GRPO dark)
    ("always_escalate", "Always-escalate",   "#14519A"),   # TUM blue dark-4
    ("few_shot",        "Few-shot (k=3)",    "#6955E2"),   # TUM blue bright-dark
    ("sft",             "SFT",               "#9B468D"),   # TUM pink-dark
    ("grpo",            "GRPO (ckpt-800)",   "#0A2D57"),   # TUM blue dark-1
    ("exhaustive",      "Exhaustive",        "#B6ACF1"),   # TUM blue bright-1
]

BL_KEYS  = [b[0] for b in BASELINES]
BL_LABEL = {b[0]: b[1] for b in BASELINES}
BL_COLOR = {b[0]: b[2] for b in BASELINES}


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Matplotlib style  (IEEE Transactions)
# ═══════════════════════════════════════════════════════════════════════════════

def apply_ieee_style():
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
        "patch.linewidth":    0.0,   # no bar outlines — clean solid bars
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Core bar-chart helper
# ═══════════════════════════════════════════════════════════════════════════════

def grouped_bars(ax, axis_ids, metric, data,
                 title, ylabel,
                 ylim=(0, 1.05), pct=True,
                 invert_bg=False,
                 grpo_outline=True):
    """
    Draw a grouped bar chart — solid TUM-palette colours, no hatching.

    axis_ids   : list of axis keys, used as x-axis groups
    metric     : key into each baseline metric dict
    invert_bg  : light-red background for "lower is better" panels (CVR, parse)
    grpo_outline: draw a dark border around GRPO bars to make them pop
    """
    n_groups = len(axis_ids)
    n_bl     = len(BASELINES)
    width    = 0.78 / n_bl
    offsets  = np.linspace(-(n_bl - 1) / 2, (n_bl - 1) / 2, n_bl) * width
    x        = np.arange(n_groups)

    for i, (bl_key, bl_label, color) in enumerate(BASELINES):
        vals = [
            (data.get(ax_id, {}).get(bl_key, {}).get(metric) or 0)
            for ax_id in axis_ids
        ]
        bars = ax.bar(
            x + offsets[i], vals,
            width=width * 0.92,
            color=color,
            edgecolor="none",
            linewidth=0,
            label=bl_label,
            zorder=3,
        )
        # Emphasise GRPO with a thin white separator and dark outline
        if bl_key == "grpo" and grpo_outline:
            for bar in bars:
                bar.set_edgecolor("#FFFFFF")
                bar.set_linewidth(0.8)
                bar.set_zorder(4)

    ax.set_xticks(x)
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
    ax.yaxis.grid(True, zorder=0, color="#CCCCCC", linewidth=0.4)
    ax.set_axisbelow(True)

    if pct:
        import matplotlib.ticker as mtick
        ax.yaxis.set_major_formatter(
            mtick.FuncFormatter(lambda v, _: f"{v:.0%}"))

    if invert_bg:
        ax.set_facecolor("#FFF5F5")


def set_x_labels(ax, axis_ids, display_map, fontsize=7):
    ax.set_xticklabels(
        [display_map.get(k, k) for k in axis_ids],
        fontsize=fontsize, ha="center",
    )


def add_shared_legend(fig, ncol=7, ypos=-0.04):
    import matplotlib.patches as mpatches
    handles = [
        mpatches.Patch(facecolor=color, edgecolor="none", label=label)
        for _, label, color in BASELINES
    ]
    fig.legend(
        handles, [b[1] for b in BASELINES],
        ncol=ncol, loc="lower center",
        bbox_to_anchor=(0.5, ypos),
        fontsize=7, frameon=True, framealpha=0.9,
        edgecolor="#CCCCCC", columnspacing=1.0,
        handlelength=1.2, handleheight=0.9,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Figure 1 — Axis 1: OOD generalisation
# ═══════════════════════════════════════════════════════════════════════════════

AXIS1_DISPLAY = {
    "1a": "Novel\napps",
    "1b": "Prose\nbattery",
    "1c": "Prose\ncredit",
    "1d": "Reordered\nprompt",
    "1e": "Compound\nedge",
}


def fig_axis1(data, out_dir, fmt="pdf", dpi=300):
    import matplotlib.pyplot as plt
    apply_ieee_style()

    ax_ids = ["1a", "1b", "1c", "1d", "1e"]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    fig.subplots_adjust(hspace=0.50, wspace=0.32, bottom=0.18, top=0.95)

    grouped_bars(axes[0, 0], ax_ids, "acc",    data,
                 "Accuracy", "Accuracy", ylim=(0, 1.15))
    set_x_labels(axes[0, 0], ax_ids, AXIS1_DISPLAY)

    grouped_bars(axes[0, 1], ax_ids, "cvr",    data,
                 "CVR (↓ lower is better)", "CVR",
                 ylim=(0, 0.68), invert_bg=True)
    set_x_labels(axes[0, 1], ax_ids, AXIS1_DISPLAY)

    grouped_bars(axes[1, 0], ax_ids, "off_f1", data,
                 "Offload F1", "F1", ylim=(0, 1.15))
    set_x_labels(axes[1, 0], ax_ids, AXIS1_DISPLAY)

    grouped_bars(axes[1, 1], ax_ids, "esc_f1", data,
                 "Escalation F1", "F1", ylim=(0, 1.15))
    set_x_labels(axes[1, 1], ax_ids, AXIS1_DISPLAY)

    add_shared_legend(fig, ncol=4, ypos=-0.04)

    out = os.path.join(out_dir, f"fig_robustness_axis1.{fmt}")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Figure 2 — Axis 3: Credit sensitivity
# ═══════════════════════════════════════════════════════════════════════════════

AXIS3_DISPLAY = {
    "3a": "γ-sweep\n(credit thresh.)",
    "3b": "Credit-app\ninteraction",
}


def fig_axis3(data, out_dir, fmt="pdf", dpi=300):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    apply_ieee_style()

    ax_ids = ["3a", "3b"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))
    fig.subplots_adjust(wspace=0.42, bottom=0.30, top=0.88)

    grouped_bars(axes[0], ax_ids, "acc",    data,
                 "Accuracy", "Accuracy", ylim=(0, 1.15))
    set_x_labels(axes[0], ax_ids, AXIS3_DISPLAY)

    grouped_bars(axes[1], ax_ids, "esc_f1", data,
                 "Escalation F1", "F1", ylim=(0, 1.15))
    set_x_labels(axes[1], ax_ids, AXIS3_DISPLAY)

    # MNU can be negative
    all_mnu = [
        (data.get(ax, {}).get(bl[0], {}).get("mnu") or 0)
        for ax in ax_ids for bl in BASELINES
    ]
    ymin = min(min(all_mnu) - 0.06, -0.15)
    ymax = max(max(all_mnu) + 0.06,  0.42)

    grouped_bars(axes[2], ax_ids, "mnu",    data,
                 "MNU", "MNU", ylim=(ymin, ymax), pct=False)
    axes[2].axhline(0, color="#888888", linewidth=0.6, zorder=0)
    axes[2].yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f"{v:+.2f}"))
    set_x_labels(axes[2], ax_ids, AXIS3_DISPLAY)

    add_shared_legend(fig, ncol=4, ypos=-0.05)

    out = os.path.join(out_dir, f"fig_robustness_axis3.{fmt}")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Figure 3 — Axis 6: Linguistic robustness
# ═══════════════════════════════════════════════════════════════════════════════

AXIS6_DISPLAY = {
    "6a": "Synonym\nsubstitution",
    "6b": "Formal\ncondensed",
    "6c": "Passive\nvoice",
}


def fig_axis6(data, out_dir, fmt="pdf", dpi=300):
    """
    Axis 6 — Linguistic robustness: accuracy only.
    CVR and parse-fail are excluded from the figure (reported in text).
    """
    import matplotlib.pyplot as plt
    apply_ieee_style()

    ax_ids = ["6a", "6b", "6c"]
    # ── Diagnostic: print what the parser gave us for axis 6 ──────────────
    for ax_id in ax_ids:
        ax_data = data.get(ax_id, {})
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.0))
    fig.subplots_adjust(bottom=0.32, top=0.92, left=0.13, right=0.97)

    grouped_bars(ax, ax_ids, "acc", data,
                 "Accuracy — Linguistic robustness (Axis 6)", "Accuracy",
                 ylim=(0, 1.15))
    set_x_labels(ax, ax_ids, AXIS6_DISPLAY)

    add_shared_legend(fig, ncol=4, ypos=-0.10)

    out = os.path.join(out_dir, f"fig_robustness_axis6.{fmt}")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Figure — Escalation benefit for low-confidence (near-boundary) decisions
# ═══════════════════════════════════════════════════════════════════════════════

# γ thresholds from the reward model
GAMMA_ESC = 0.20   # below this, escalation is optimal
GAMMA_MIN = 0.10   # below this, no escalation (credit floor)

# Baselines to show in this figure (ordered weakest → strongest)
ESC_BEN_BASELINES = ["sft", "grpo", "exhaustive"]

# Bin edges — distance from the nearest threshold (γ_esc = 0.20)
# Negative distance → inside escalation zone (γ < 0.20)
# Positive distance → outside escalation zone (γ > 0.20)
BIN_EDGES = [-0.20, -0.12, -0.06, -0.02, 0.02, 0.06, 0.12, 0.20, 0.40]
BIN_LABELS = [
    "≤−12%", "−12–−6%", "−6–−2%", "−2–+2%",
    "+2–+6%", "+6–+12%", "+12–+20%", ">+20%",
]


def _distance_to_esc_boundary(gamma):
    """Signed distance from γ_esc=0.20: negative means inside esc zone."""
    return gamma - GAMMA_ESC


def _bin_index(gamma):
    d = _distance_to_esc_boundary(gamma)
    for i, edge in enumerate(BIN_EDGES[1:]):
        if d < edge:
            return i
    return len(BIN_EDGES) - 2   # last bin


def fig_escalation_benefit(sweep_data, out_dir, fmt="pdf", dpi=300):
    """
    2×2 precision/recall grid: rows = zone (inside / outside γ_esc),
    columns = decision type (escalation / offload).

    Each cell shows precision (solid) and recall (hatched) bars for
    SFT, GRPO and Exhaustive.

    Story:
      Inside zone (γ < γ_esc):
        Escalation — all three escalate perfectly (precision=recall=1).
        Offload    — SFT outputs both flags → offload precision drops;
                     GRPO/Exhaustive cleanly suppress offload.
      Outside zone (γ ≥ γ_esc):
        Escalation — SFT keeps escalating (recall=1) but half are wrong
                     (precision=0.5); GRPO stops escalating entirely
                     (N/A) because it chooses offload instead.
        Offload    — GRPO offloads correctly (precision=recall=1);
                     SFT also offloads but pairs it with escalation,
                     penalised by the task.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    import matplotlib.patches as mpatches
    apply_ieee_style()

    if not sweep_data:
        print("  [SKIP] fig_escalation_benefit: no 3a_sweep data.")
        return None

    # ── Accumulate TP / FP / FN for escalation AND offload, by zone ──────────
    def _empty():
        return {bl: {"tp": 0, "fp": 0, "fn": 0} for bl in ESC_BEN_BASELINES}

    esc_counts = {"inside": _empty(), "outside": _empty()}
    off_counts = {"inside": _empty(), "outside": _empty()}

    for row in sweep_data:
        if row.get("label_esc") is None:
            continue
        zone = "inside" if row["gamma"] < GAMMA_ESC else "outside"
        le = bool(row["label_esc"])
        lo = bool(row.get("label_off", False))

        for bl in ESC_BEN_BASELINES:
            pe = row.get(f"{bl}_esc")
            po = row.get(f"{bl}_off")
            if pe is None or po is None:
                continue
            pe, po = bool(pe), bool(po)

            # escalation confusion
            ec = esc_counts[zone][bl]
            if pe and le:   ec["tp"] += 1
            elif pe:        ec["fp"] += 1
            elif le:        ec["fn"] += 1

            # offload confusion
            oc = off_counts[zone][bl]
            if po and lo:   oc["tp"] += 1
            elif po:        oc["fp"] += 1
            elif lo:        oc["fn"] += 1

    def _prec(c):
        return c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else None

    def _rec(c):
        return c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else None

    # ── Diagnostics ──────────────────────────────────────────────────────────
    def _fmt(v):
        return "N/A" if v is None else f"{v:.0%}"

    for zone in ("inside", "outside"):
        print(f"  [ESC_BEN] {zone} zone:")
        for bl in ESC_BEN_BASELINES:
            ec = esc_counts[zone][bl]
            oc = off_counts[zone][bl]
            print(f"    {bl:12s}  esc: P={_fmt(_prec(ec))} R={_fmt(_rec(ec))}"
                  f"  |  off: P={_fmt(_prec(oc))} R={_fmt(_rec(oc))}")

    # ── Layout: 2 rows × 2 cols ───────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharey=True,
                             sharex="col")
    fig.subplots_adjust(bottom=0.18, top=0.88, left=0.09, right=0.97,
                        hspace=0.38, wspace=0.10)

    fmt_pct = mtick.FuncFormatter(lambda v, _: f"{v:.0%}")

    n_bl      = len(ESC_BEN_BASELINES)
    group_gap = 0.50
    bar_w     = 0.18
    p_centre  = 0.0                        # precision group
    r_centre  = n_bl * bar_w + group_gap   # recall group
    offsets   = np.arange(n_bl) * bar_w - (n_bl - 1) * bar_w / 2

    # rows: inside (0) / outside (1);  cols: escalation (0) / offload (1)
    row_cfg = [
        ("inside",  "#EEF3FA",
         "(a) Escalation — inside zone  (γ < γ_esc)",
         "(b) Offload — inside zone  (γ < γ_esc)"),
        ("outside", "#FFFFFF",
         "(c) Escalation — outside zone  (γ ≥ γ_esc)",
         "(d) Offload — outside zone  (γ ≥ γ_esc)"),
    ]

    def _draw_cell(ax, zone, dim_counts, bg, title, p_label, r_label):
        ax.set_facecolor(bg)
        for i, bl in enumerate(ESC_BEN_BASELINES):
            c     = dim_counts[zone][bl]
            color = BL_COLOR.get(bl, "#888888")
            ec_   = "#333333" if bl == "grpo" else "none"
            lw    = 0.8       if bl == "grpo" else 0.0

            pv = _prec(c)   # None = no predictions at all
            rv = _rec(c)    # None = no positive labels at all

            # precision bar (solid)
            ax.bar(p_centre + offsets[i], pv if pv is not None else 0.0,
                   width=bar_w * 0.88, color=color,
                   edgecolor=ec_, linewidth=lw, zorder=3)
            if pv is None:
                ax.text(p_centre + offsets[i], 0.03, "N/A",
                        ha="center", va="bottom", fontsize=5,
                        color=color, fontweight="bold", rotation=90, zorder=4)

            # recall bar (hatched)
            ax.bar(r_centre + offsets[i], rv if rv is not None else 0.0,
                   width=bar_w * 0.88, color=color,
                   edgecolor="#FFFFFF", linewidth=0.5,
                   hatch="///", zorder=3, alpha=0.85)
            if rv is None:
                ax.text(r_centre + offsets[i], 0.03, "N/A",
                        ha="center", va="bottom", fontsize=5,
                        color=color, fontweight="bold", rotation=90, zorder=4)

        ax.set_xticks([p_centre, r_centre])
        ax.set_xticklabels([p_label, r_label], fontsize=6.5)
        ax.set_xlim(p_centre - n_bl * bar_w, r_centre + n_bl * bar_w)
        ax.yaxis.set_major_formatter(fmt_pct)
        ax.set_ylim(0, 1.18)
        ax.yaxis.grid(True, zorder=0, color="#CCCCCC", linewidth=0.4)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=6.8, fontweight="bold", pad=3)
        ax.tick_params(axis="x", length=0)

    for r, (zone, bg, title_esc, title_off) in enumerate(row_cfg):
        _draw_cell(axes[r, 0], zone, esc_counts, bg, title_esc,
                   "Precision", "Recall")
        _draw_cell(axes[r, 1], zone, off_counts, bg, title_off,
                   "Precision", "Recall")
        axes[r, 0].set_ylabel("Metric value", fontsize=7.5)

    fig.suptitle(
        "Decision quality by action type and credit zone",
        fontsize=8.5, fontweight="bold", y=0.95,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    bl_handles = [
        mpatches.Patch(facecolor=BL_COLOR.get(bl, "#888"), edgecolor="none",
                       label=BL_LABEL.get(bl, bl))
        for bl in ESC_BEN_BASELINES
    ]
    hatch_handles = [
        mpatches.Patch(facecolor="#AAAAAA", edgecolor="none",
                       label="Precision  (solid)"),
        mpatches.Patch(facecolor="#AAAAAA", edgecolor="#FFFFFF",
                       hatch="///", label="Recall  (hatched)"),
    ]
    fig.legend(
        bl_handles + hatch_handles,
        [h.get_label() for h in bl_handles + hatch_handles],
        ncol=len(ESC_BEN_BASELINES) + 2,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        fontsize=6.5, frameon=True, framealpha=0.9,
        edgecolor="#CCCCCC", columnspacing=0.9,
        handlelength=1.1, handleheight=0.9,
    )

    out = os.path.join(out_dir, f"fig_escalation_benefit.{fmt}")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 9.  Figure — Summary heatmap (all baselines × all axes)
# ═══════════════════════════════════════════════════════════════════════════════

ALL_AXES_DISPLAY = {
    "1a": "Novel apps",   "1b": "Prose batt.",
    "1c": "Prose credit", "1d": "Reorder",
    "1e": "Compound",     "3a": "γ-sweep",
    "3b": "Credit-app",   "6a": "Synonym",
    "6b": "Formal",       "6c": "Passive",
}

# Row order for heatmap: weakest → GRPO → oracle
HEATMAP_ORDER = [
    "random", "few_shot", "always_offload",
    "always_escalate", "sft", "grpo", "exhaustive",
]


def fig_summary_heatmap(data, out_dir, fmt="pdf", dpi=300):
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    apply_ieee_style()

    all_axes = ["1a","1b","1c","1d","1e","3a","3b","6a","6b","6c"]

    matrix = np.zeros((len(HEATMAP_ORDER), len(all_axes)))
    for j, ax_id in enumerate(all_axes):
        for i, bl in enumerate(HEATMAP_ORDER):
            matrix[i, j] = (data.get(ax_id, {})
                            .get(bl, {}).get("acc") or 0)

    # TUM-palette diverging: white → TUM blue dark-1
    from matplotlib.colors import LinearSegmentedColormap
    tum_cmap = LinearSegmentedColormap.from_list(
        "tum_blue",
        ["#FFFFFF", "#B6ACF1", "#6955E2", "#14519A", "#0A2D57"],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    fig.subplots_adjust(left=0.18, right=0.96, bottom=0.24, top=0.93)

    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap=tum_cmap)

    # Annotate each cell
    for i in range(len(HEATMAP_ORDER)):
        for j in range(len(all_axes)):
            v    = matrix[i, j]
            tcol = "white" if v > 0.55 else "#0A2D57"
            ax.text(j, i, f"{v:.0%}",
                    ha="center", va="center",
                    fontsize=6.5, color=tcol, fontweight="bold")

    ax.set_xticks(range(len(all_axes)))
    ax.set_xticklabels(
        [ALL_AXES_DISPLAY[a] for a in all_axes],
        fontsize=6.5, rotation=30, ha="right",
    )
    ax.set_yticks(range(len(HEATMAP_ORDER)))
    ax.set_yticklabels(
        [BL_LABEL[b] for b in HEATMAP_ORDER], fontsize=7,
    )

    # Highlight GRPO row with TUM pink border
    grpo_row = HEATMAP_ORDER.index("grpo")
    for y_off in [-0.5, 0.5]:
        ax.axhline(grpo_row + y_off, color="#9B468D",
                   linewidth=1.4, zorder=5)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.008)
    cbar.set_label("Accuracy", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.outline.set_linewidth(0)

    ax.set_title(
        "Accuracy heatmap — all baselines × robustness axes  "
        "(GRPO row in pink border)",
        fontsize=7.5, pad=4,
    )

    out = os.path.join(out_dir, f"fig_robustness_summary.{fmt}")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  Saved → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Entry point
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_RESULTS_JSON = os.path.join(ROOT, "models", "robustness_results.json")


def main():
    parser = argparse.ArgumentParser(
        description="Plot robustness evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python scripts/plot_robustness.py
  python scripts/plot_robustness.py --report models/robustness_report.txt
  python scripts/plot_robustness.py --fmt pdf --dpi 300
  python scripts/plot_robustness.py --figs axis1 axis6 esc_benefit
        """,
    )
    parser.add_argument("--report", default=DEFAULT_REPORT,
                        help=f"Path to robustness_report.txt (default: {DEFAULT_REPORT})")
    parser.add_argument("--results-json", default=DEFAULT_RESULTS_JSON,
                        help=f"Path to robustness_results.json (for esc_benefit figure; "
                             f"default: {DEFAULT_RESULTS_JSON})")
    parser.add_argument("--out",    default=DEFAULT_OUT,
                        help=f"Output directory (default: {DEFAULT_OUT})")
    parser.add_argument("--fmt",    default="pdf",
                        choices=["pdf", "png", "svg"],
                        help="Output format (default: pdf)")
    parser.add_argument("--dpi",    type=int, default=300,
                        help="DPI for raster output (default: 300)")
    parser.add_argument("--figs",   nargs="+",
                        choices=["axis1", "axis3", "axis6", "esc_benefit",
                                 "summary", "all"],
                        default=["all"],
                        help="Which figures to generate (default: all)")
    args = parser.parse_args()

    # ── Parse the report ──────────────────────────────────────────────────────
    if not os.path.exists(args.report):
        sys.exit(f"[ERROR] Report not found: {args.report}\n"
                 f"        Run evaluate_robustness.py first, or pass --report <path>.")

    print(f"Parsing: {args.report}")
    data = parse_report(args.report)

    # ── JSON fallback: if the .txt gave us nothing useful, try the JSON ───────
    expected_axes = {"1a","1b","1c","1d","1e","3a","3b","6a","6b","6c"}
    if not expected_axes.intersection(data.keys()):
        print(f"  [WARN] No recognised axes in report; falling back to {args.results_json}")
        if os.path.exists(args.results_json):
            data = _parse_report_json(args.results_json)
            print(f"  Axes loaded from JSON: {sorted(data.keys())}")
        else:
            sys.exit(f"[ERROR] Neither the report nor the JSON yielded usable data.\n"
                     f"        Re-run evaluate_robustness.py to regenerate both.")

    # ── Load per-sample sweep data for esc_benefit figure ────────────────────
    sweep_data = []
    want = set(args.figs)
    make_all = "all" in want

    if make_all or "esc_benefit" in want:
        if os.path.exists(args.results_json):
            import json as _json
            with open(args.results_json, "r", encoding="utf-8") as f:
                raw_results = _json.load(f)
            sweep_data = raw_results.get("3a_sweep", [])
            if sweep_data:
                print(f"  Loaded 3a_sweep: {len(sweep_data)} rows from {args.results_json}")
            else:
                print(f"  [WARN] No 3a_sweep key in {args.results_json}. "
                      f"Re-run evaluate_robustness.py to generate it.")
        else:
            print(f"  [WARN] Results JSON not found: {args.results_json}. "
                  f"Skipping esc_benefit figure.")

    # ── Verify TUM palette is used ────────────────────────────────────────────
    print(f"  Colour palette: TUM brand")
    for bl_key, bl_label, color in BASELINES:
        print(f"    {bl_label:<22} {color}")

    # ── Generate figures ──────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        sys.exit("[ERROR] matplotlib required:  pip install matplotlib")

    os.makedirs(args.out, exist_ok=True)

    print(f"\nGenerating figures → {args.out}")
    out_paths = []

    if make_all or "axis1"       in want:
        out_paths.append(fig_axis1(data, args.out, args.fmt, args.dpi))
    if make_all or "axis3"       in want:
        out_paths.append(fig_axis3(data, args.out, args.fmt, args.dpi))
    if make_all or "axis6"       in want:
        out_paths.append(fig_axis6(data, args.out, args.fmt, args.dpi))
    if (make_all or "esc_benefit" in want) and sweep_data:
        out_paths.append(
            fig_escalation_benefit(sweep_data, args.out, args.fmt, args.dpi))
    if make_all or "summary"     in want:
        out_paths.append(fig_summary_heatmap(data, args.out, args.fmt, args.dpi))

    print(f"\nDone. {len(out_paths)} figure(s) saved.")
    return out_paths


if __name__ == "__main__":
    main()