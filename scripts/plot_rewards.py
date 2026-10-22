"""
plot_rewards.py  —  Per-step and smoothed GRPO reward visualiser
-----------------------------------------------------------------
Reads grpo_metrics.json and produces a six-panel figure:

  Panel 1  Reward  — raw per-step (faint) + EMA smoothed (bold) + ±std band
  Panel 2  Reward logic U(o,e|s)  — if reward_logic was logged (run 5+)
           otherwise: reward_std per step (how diverse completions are)
  Panel 3  Zero-std fraction  — per-step + smoothed
  Panel 4  KL divergence  — per-step + smoothed + 0.05 reference
  Panel 5  Loss  — per-step + smoothed (if available)
  Panel 6  Grad norm  — per-step + smoothed (if available)

Smoothing
---------
Exponential Moving Average (EMA) with configurable alpha:
  EMA(t) = alpha * value(t) + (1 - alpha) * EMA(t-1)
  alpha = 2 / (window + 1)  where window = --smooth (default 50)

A window of 50 steps means the EMA gives ~86% weight to the last
50 observations — a good balance between responsiveness and noise
suppression for runs of 900–9000 log entries.

Checkpoint overlay
------------------
If --checkpoints path is given, vertical lines mark each checkpoint
step on all panels, coloured by CVR (green=low, red=high).

Usage
-----
  # Basic — reads default paths
  python scripts/plot_rewards.py

  # Custom paths
  python scripts/plot_rewards.py \\
      --metrics  models/grpo_metrics.json \\
      --out      models/reward_plot.png

  # Paper-quality single-panel SVG (no target line)
  python scripts/plot_rewards.py --paper \\
      --metrics  models/grpo_metrics.json \\
      --out      figures/fig_reward_curve.svg

  # With checkpoint overlay
  python scripts/plot_rewards.py \\
      --checkpoints models/grpo_checkpoint_eval.json

  # Faster EMA (less smoothing)
  python scripts/plot_rewards.py --smooth 20

  # Higher DPI for paper figures
  python scripts/plot_rewards.py --dpi 300

Dependencies
------------
  pip install matplotlib numpy
"""

import os
import sys
import json
import argparse
import numpy as np

# ── Path defaults ─────────────────────────────────────────────────────────────
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
ROOT             = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_METRICS  = os.path.join(ROOT, "models", "grpo_metrics.json")
DEFAULT_CKPT     = os.path.join(ROOT, "models", "grpo_checkpoint_eval.json")
DEFAULT_OUT      = os.path.join(ROOT, "models", "reward_plot.svg")


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [d for d in data if "reward" in d]


def load_checkpoints(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        ckpts = json.load(f)
    return sorted(ckpts, key=lambda c: c["step"])


# ═══════════════════════════════════════════════════════════════════════════════
# Smoothing
# ═══════════════════════════════════════════════════════════════════════════════

def ema(values, window=50):
    """
    Exponential Moving Average.
    alpha = 2/(window+1) so the EMA gives 86% weight to the last `window`
    observations — standard convention matching TensorBoard's smoothing.
    """
    alpha  = 2.0 / (window + 1)
    result = []
    s      = None
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            result.append(None)
            continue
        s = v if s is None else alpha * v + (1 - alpha) * s
        result.append(s)
    return result


def valid(lst):
    """Return (indices, values) for non-None entries."""
    idx = [i for i, v in enumerate(lst) if v is not None]
    val = [lst[i] for i in idx]
    return idx, val


# ═══════════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

# Colour palette — works on both white and dark backgrounds
C = {
    "reward":        "#4C8BF5",   # blue
    "reward_smooth": "#1A56CC",   # dark blue
    "logic":         "#00B4A6",   # teal
    "logic_smooth":  "#007A70",   # dark teal
    "rstd":          "#A0A0A0",   # grey
    "zero_std":      "#E84545",   # red
    "zero_smooth":   "#B52020",   # dark red
    "kl":            "#F5A623",   # amber
    "kl_smooth":     "#C47A00",   # dark amber
    "loss":          "#9B59B6",   # purple
    "loss_smooth":   "#6C3483",   # dark purple
    "grad":          "#2ECC71",   # green
    "grad_smooth":   "#1A7A42",   # dark green
    "band":          "#4C8BF5",   # blue band
    "ckpt_safe":     "#2ECC71",   # green vline
    "ckpt_risky":    "#E84545",   # red vline
    "ref":           "#888888",   # reference lines
}

ALPHA_RAW  = 0.20   # per-step series transparency
ALPHA_BAND = 0.10   # reward ± std band transparency


def _add_checkpoint_vlines(ax, checkpoints, ymin, ymax):
    """
    Draw vertical lines at each checkpoint step.
    Colour: green when CVR ≤ 5%, amber when CVR ≤ 15%, red otherwise.
    """
    for ck in checkpoints:
        cvr = ck.get("cvr_combined", 1.0)
        if   cvr <= 0.05: col = C["ckpt_safe"]
        elif cvr <= 0.15: col = C["kl"]
        else:             col = C["ckpt_risky"]
        ax.axvline(ck["step"], color=col, alpha=0.25, linewidth=0.8, zorder=0)


def _style(ax, title, ylabel, xlim=None, ylim=None):
    ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel("gradient step", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    ax.legend(fontsize=7, framealpha=0.5, loc="best")


def plot(metrics, checkpoints, out_path, smooth_window=50, dpi=150):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[ERROR] matplotlib is required:  pip install matplotlib")
        sys.exit(1)

    # ── Extract series ────────────────────────────────────────────────────────
    steps    = [d["step"]                      for d in metrics]
    reward   = [d["reward"]                    for d in metrics]
    rstd     = [d["reward_std"]                for d in metrics]
    zero_std = [d["frac_reward_zero_std"]      for d in metrics]
    kl       = [d["kl"]                        for d in metrics]
    loss     = [d.get("loss")                  for d in metrics]
    grad_n   = [d.get("grad_norm")             for d in metrics]
    logic    = [d.get("reward_logic")          for d in metrics]
    logic_std= [d.get("reward_logic_std")      for d in metrics]

    has_logic = any(v is not None for v in logic)
    has_loss  = any(v is not None for v in loss)
    has_grad  = any(v is not None for v in grad_n)

    # Reward ± std band
    rew_hi = [r + s for r, s in zip(reward, rstd)]
    rew_lo = [r - s for r, s in zip(reward, rstd)]

    # EMA smoothed series
    sm_reward   = ema(reward,   smooth_window)
    sm_zero_std = ema(zero_std, smooth_window)
    sm_kl       = ema(kl,       smooth_window)
    sm_loss     = ema(loss,     smooth_window)
    sm_grad     = ema(grad_n,   smooth_window)
    sm_logic    = ema(logic,    smooth_window)

    # Summary stats for title bar
    n       = len(reward)
    q       = max(1, n // 4)
    q1_mean = np.mean(reward[:q])
    q4_mean = np.mean(reward[3*q:])
    zs_mean = np.mean(zero_std)
    rising  = q4_mean > q1_mean + 0.01

    xlim = (min(steps), max(steps))

    # ── Layout: 2 columns × 3 rows ────────────────────────────────────────────
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    fig.patch.set_facecolor("#FAFAFA")
    for ax in axes.flat:
        ax.set_facecolor("#FAFAFA")

    run_label = os.path.basename(os.path.dirname(out_path)) or "GRPO run"
    curve_str = f"RISING ↑" if rising else "FLAT →"
    suptitle  = (f"{run_label}  |  "
                 f"{n} log entries  ·  steps {steps[0]}–{steps[-1]}  ·  "
                 f"smooth={smooth_window}  ·  "
                 f"Q1={q1_mean:.3f} → Q4={q4_mean:.3f} ({curve_str})  ·  "
                 f"zero-std mean={zs_mean:.2f}  ·  "
                 f"checkpoints={len(checkpoints)}")
    fig.suptitle(suptitle, fontsize=8, y=0.995, color="#333")

    # ── Panel 1: Reward ───────────────────────────────────────────────────────
    ax = axes[0, 0]
    if checkpoints:
        _add_checkpoint_vlines(ax, checkpoints, min(rew_lo), max(rew_hi))

    # ±std band
    ax.fill_between(steps, rew_lo, rew_hi,
                    color=C["band"], alpha=ALPHA_BAND, label="reward ± std")
    # Per-step (faint)
    ax.plot(steps, reward, color=C["reward"], alpha=ALPHA_RAW,
            linewidth=0.6, label="_raw")
    # EMA (bold)
    si, sv = valid(sm_reward)
    ax.plot([steps[i] for i in si], sv,
            color=C["reward_smooth"], linewidth=1.8,
            label=f"EMA-{smooth_window}")
    # Reference line at zero only (target line removed for paper)
    ax.axhline(0.0, color=C["ref"], linewidth=0.5, linestyle=":", alpha=0.4)
    # Annotate peak
    peak_val = max(reward)
    peak_idx = reward.index(peak_val)
    ax.annotate(f"peak {peak_val:.3f}",
                xy=(steps[peak_idx], peak_val),
                xytext=(steps[peak_idx], peak_val + 0.03),
                fontsize=6.5, color=C["reward_smooth"],
                arrowprops=dict(arrowstyle="-", color=C["reward_smooth"],
                                linewidth=0.6))
    _style(ax, "Reward (total = schema + U)", "reward",
           xlim=xlim, ylim=(min(rew_lo) - 0.05, max(rew_hi) + 0.05))

    # ── Panel 2: Reward logic  OR  reward_std ─────────────────────────────────
    ax = axes[0, 1]
    if checkpoints:
        _add_checkpoint_vlines(ax, checkpoints, -1, 1)

    if has_logic:
        # Logic reward U(o,e|s) — shows true learning without schema floor
        if any(s is not None for s in logic_std):
            lo_hi = [r+s if r is not None and s is not None else None
                     for r,s in zip(logic, logic_std)]
            lo_lo = [r-s if r is not None and s is not None else None
                     for r,s in zip(logic, logic_std)]
            li, lv_hi = valid(lo_hi); _, lv_lo = valid(lo_lo)
            ax.fill_between([steps[i] for i in li], lv_lo, lv_hi,
                            color=C["logic"], alpha=ALPHA_BAND,
                            label="U(o,e|s) ± std")
        li2, lv = valid(logic)
        ax.plot([steps[i] for i in li2], lv,
                color=C["logic"], alpha=ALPHA_RAW, linewidth=0.6)
        si2, sv2 = valid(sm_logic)
        ax.plot([steps[i] for i in si2], sv2,
                color=C["logic_smooth"], linewidth=1.8,
                label=f"U EMA-{smooth_window}")
        ax.axhline(0.0, color=C["ref"], linewidth=1.0, linestyle="--",
                   alpha=0.7, label="U=0 breakeven")
        title2  = "Logic reward U(o,e|s)  [schema excluded]"
        ylabel2 = "U(o,e|s)"
    else:
        # Fallback: reward_std per step
        ax.plot(steps, rstd, color=C["rstd"], alpha=ALPHA_RAW,
                linewidth=0.6, label="reward_std (raw)")
        sm_rstd = ema(rstd, smooth_window)
        si3, sv3 = valid(sm_rstd)
        ax.plot([steps[i] for i in si3], sv3,
                color="#505050", linewidth=1.8,
                label=f"EMA-{smooth_window}")
        ax.axhline(0.05, color=C["ref"], linewidth=0.8, linestyle="--",
                   alpha=0.6, label="diversity floor 0.05")
        title2  = ("Reward std dev  [diversity proxy — run reward_logic "
                   "available in run 5+]")
        ylabel2 = "reward_std"

    _style(ax, title2, ylabel2, xlim=xlim)

    # ── Panel 3: Zero-std ─────────────────────────────────────────────────────
    ax = axes[1, 0]
    if checkpoints:
        _add_checkpoint_vlines(ax, checkpoints, 0, 1)

    ax.plot(steps, zero_std, color=C["zero_std"], alpha=ALPHA_RAW,
            linewidth=0.6)
    si, sv = valid(sm_zero_std)
    ax.plot([steps[i] for i in si], sv,
            color=C["zero_smooth"], linewidth=1.8,
            label=f"EMA-{smooth_window}")
    ax.axhline(0.30, color="#2ECC71", linewidth=0.8, linestyle="--",
               alpha=0.7, label="healthy < 0.30")
    ax.axhline(0.70, color=C["zero_std"], linewidth=0.8, linestyle="--",
               alpha=0.5, label="collapse > 0.70")
    # Shade danger zone
    ax.axhspan(0.70, 1.05, color=C["zero_std"], alpha=0.04)
    ax.fill_between([], [], color=C["zero_std"], alpha=0.08,
                    label="collapse zone")
    _style(ax, "Zero-std fraction  [mode collapse indicator]",
           "fraction of batches with σ=0",
           xlim=xlim, ylim=(-0.02, 1.05))

    # ── Panel 4: KL divergence ────────────────────────────────────────────────
    ax = axes[1, 1]
    if checkpoints:
        _add_checkpoint_vlines(ax, checkpoints, 0, 0.12)

    ax.plot(steps, kl, color=C["kl"], alpha=ALPHA_RAW, linewidth=0.6)
    si, sv = valid(sm_kl)
    ax.plot([steps[i] for i in si], sv,
            color=C["kl_smooth"], linewidth=1.8,
            label=f"EMA-{smooth_window}")
    ax.axhline(0.05, color="#2ECC71", linewidth=0.8, linestyle="--",
               alpha=0.7, label="healthy > 0.05")
    # Shade healthy zone
    ax.axhspan(0.05, max(kl) * 1.1, color="#2ECC71", alpha=0.04)
    # Annotate peak
    kl_peak = max(kl)
    kl_pidx = kl.index(kl_peak)
    ax.annotate(f"peak {kl_peak:.4f}",
                xy=(steps[kl_pidx], kl_peak),
                xytext=(steps[kl_pidx], kl_peak * 0.85),
                fontsize=6.5, color=C["kl_smooth"],
                arrowprops=dict(arrowstyle="-", color=C["kl_smooth"],
                                linewidth=0.6))
    _style(ax, "KL divergence  [policy drift from SFT reference]",
           "KL", xlim=xlim)

    # ── Panel 5: Loss ─────────────────────────────────────────────────────────
    ax = axes[2, 0]
    if checkpoints:
        _add_checkpoint_vlines(ax, checkpoints, 0, 0.01)

    if has_loss:
        li, lv = valid(loss)
        ax.plot([steps[i] for i in li], lv,
                color=C["loss"], alpha=ALPHA_RAW, linewidth=0.6)
        si, sv = valid(sm_loss)
        ax.plot([steps[i] for i in si], sv,
                color=C["loss_smooth"], linewidth=1.8,
                label=f"EMA-{smooth_window}")
        ax.axhline(0.002, color=C["ref"], linewidth=0.8, linestyle="--",
                   alpha=0.6, label="active update floor 0.002")
        _style(ax, "Training loss", "loss", xlim=xlim)
    else:
        ax.text(0.5, 0.5, "loss not logged",
                ha="center", va="center", transform=ax.transAxes,
                color="#888", fontsize=10)
        _style(ax, "Training loss", "loss", xlim=xlim)

    # ── Panel 6: Gradient norm ────────────────────────────────────────────────
    ax = axes[2, 1]
    if checkpoints:
        _add_checkpoint_vlines(ax, checkpoints, 0, 5)

    if has_grad:
        gi, gv = valid(grad_n)
        ax.plot([steps[i] for i in gi], gv,
                color=C["grad"], alpha=ALPHA_RAW, linewidth=0.6)
        si, sv = valid(sm_grad)
        ax.plot([steps[i] for i in si], sv,
                color=C["grad_smooth"], linewidth=1.8,
                label=f"EMA-{smooth_window}")
        # Annotate gradient spikes (> 3× the smoothed value)
        sm_arr = np.array([v if v is not None else 0 for v in sm_grad])
        raw_arr= np.array(gv)
        raw_idx= np.array(gi)
        spikes = raw_idx[raw_arr > 3 * np.maximum(sm_arr[gi], 0.01)]
        for s_idx in spikes[:5]:   # annotate at most 5 spikes
            ax.axvline(steps[s_idx], color=C["ckpt_risky"], alpha=0.4,
                       linewidth=0.8, linestyle=":")
        _style(ax, "Gradient norm  [training stability]",
               "grad norm", xlim=xlim)
    else:
        ax.text(0.5, 0.5, "grad_norm not logged",
                ha="center", va="center", transform=ax.transAxes,
                color="#888", fontsize=10)
        _style(ax, "Gradient norm", "grad norm", xlim=xlim)

    # ── Checkpoint legend ─────────────────────────────────────────────────────
    if checkpoints:
        patches = [
            mpatches.Patch(color=C["ckpt_safe"],  alpha=0.6, label="checkpoint CVR ≤ 5%"),
            mpatches.Patch(color=C["kl"],         alpha=0.6, label="checkpoint CVR ≤ 15%"),
            mpatches.Patch(color=C["ckpt_risky"], alpha=0.6, label="checkpoint CVR > 15%"),
        ]
        fig.legend(handles=patches, loc="lower center", ncol=3,
                   fontsize=7, framealpha=0.5,
                   bbox_to_anchor=(0.5, -0.005))

    plt.tight_layout(rect=[0, 0.02 if checkpoints else 0, 1, 0.995])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# Paper figure  —  single-panel reward curve, SVG, no target line
# ═══════════════════════════════════════════════════════════════════════════════

def plot_paper(metrics, out_path, smooth_window=50):
    """
    Single-panel, paper-quality reward curve.

    Reproduces the style of the training monitor figure:
      · faint per-step values shown as a ±std fill band
      · EMA-50 smoothed curve in dark blue
      · peak annotated with a leader line
      · no target / oracle reference line

    Output format is determined by the file extension (.svg recommended
    for LaTeX inclusion; .png also works).

    Parameters
    ----------
    metrics       : list of dicts from grpo_metrics.json
    out_path      : destination path (e.g. "figures/fig_reward_curve.svg")
    smooth_window : EMA window size (default 50)
    """
    import matplotlib
    ext = os.path.splitext(out_path)[1].lower().lstrip(".")
    matplotlib.use("svg" if ext == "svg" else "Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.rcParams.update({
        "text.usetex":        False,
        "mathtext.fontset":   "cm",
        "font.family":        "serif",
        "axes.labelsize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "axes.linewidth":     0.7,
        "xtick.major.width":  0.5,
        "ytick.major.width":  0.5,
        "grid.linewidth":     0.4,
        "grid.alpha":         0.4,
    })

    steps  = [d["step"]       for d in metrics]
    reward = [d["reward"]     for d in metrics]
    rstd   = [d["reward_std"] for d in metrics]

    rew_hi    = [r + s for r, s in zip(reward, rstd)]
    rew_lo    = [r - s for r, s in zip(reward, rstd)]
    sm_reward = ema(reward, smooth_window)

    # Figure dimensions: double-column IEEE width
    fig, ax = plt.subplots(figsize=(3.5, 2.2))   # single-column width

    # ±std band (faint fill + raw per-step within it)
    ax.fill_between(steps, rew_lo, rew_hi,
                    color=C["band"], alpha=0.18, label="reward $\\pm$ std")
    ax.plot(steps, reward, color=C["reward"], alpha=0.18,
            linewidth=0.5, zorder=2)

    # EMA smoothed
    si, sv = valid(sm_reward)
    ax.plot([steps[i] for i in si], sv,
            color=C["reward_smooth"], linewidth=1.6,
            label=f"EMA-{smooth_window}", zorder=3)

    # Zero baseline
    ax.axhline(0.0, color=C["ref"], linewidth=0.5, linestyle=":",
               alpha=0.5, zorder=1)

    # Peak annotation
    peak_val = max(reward)
    peak_idx = reward.index(peak_val)
    ax.annotate(f"peak {peak_val:.3f}",
                xy=(steps[peak_idx], peak_val),
                xytext=(steps[peak_idx] - len(steps) * 0.12, peak_val + 0.03),
                fontsize=7, color=C["reward_smooth"],
                arrowprops=dict(arrowstyle="-",
                                color=C["reward_smooth"],
                                linewidth=0.6))

    ax.set_xlabel("gradient step", fontsize=9)
    ax.set_ylabel("reward", fontsize=9)
    ax.set_title("Reward (total = schema + U)", fontsize=9, pad=4)
    ax.set_xlim(min(steps), max(steps))
    y_pad = 0.05
    ax.set_ylim(min(rew_lo) - y_pad, max(rew_hi) + y_pad)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    ax.tick_params(direction="in", length=3, pad=2)
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    fig.tight_layout(pad=0.4)

    save_kwargs = {"bbox_inches": "tight"}
    if ext == "svg":
        save_kwargs["format"] = "svg"
        save_kwargs["metadata"] = {"Creator": "plot_rewards.py"}
    else:
        save_kwargs["format"] = ext or "png"
        save_kwargs["dpi"] = 300

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, **save_kwargs)
    plt.close(fig)
    print(f"Saved → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Plot per-step and EMA-smoothed GRPO reward curves"
    )
    parser.add_argument("--metrics",
                        default=DEFAULT_METRICS,
                        help=f"Path to grpo_metrics.json (default: {DEFAULT_METRICS})")
    parser.add_argument("--checkpoints",
                        default=None,
                        help="Path to grpo_checkpoint_eval.json (optional overlay)")
    parser.add_argument("--out",
                        default=DEFAULT_OUT,
                        help=f"Output path (default: {DEFAULT_OUT})")
    parser.add_argument("--smooth",
                        type=int, default=50, metavar="N",
                        help="EMA window size (default: 50 steps)")
    parser.add_argument("--dpi",
                        type=int, default=150,
                        help="Output DPI for PNG (default: 150; use 300 for paper)")
    parser.add_argument("--paper",
                        action="store_true",
                        help=("Single-panel paper figure. "
                              "Use --out with .svg extension for LaTeX inclusion."))
    args = parser.parse_args()

    print(f"Loading metrics  : {args.metrics}")
    metrics = load_metrics(args.metrics)
    print(f"  {len(metrics)} entries  "
          f"steps {metrics[0]['step']}–{metrics[-1]['step']}")

    if args.paper:
        plot_paper(metrics, args.out, smooth_window=args.smooth)
        return

    checkpoints = []
    if args.checkpoints:
        checkpoints = load_checkpoints(args.checkpoints)
        print(f"Loading checkpoints: {args.checkpoints}")
        print(f"  {len(checkpoints)} checkpoints  "
              f"steps {checkpoints[0]['step']}–{checkpoints[-1]['step']}"
              if checkpoints else "  none found")

    plot(metrics, checkpoints, args.out,
         smooth_window=args.smooth,
         dpi=args.dpi)


if __name__ == "__main__":
    main()