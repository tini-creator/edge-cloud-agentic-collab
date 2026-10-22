"""
monitor_grpo.py  —  Live GRPO Training Monitor
-----------------------------------------------
Serves a single-page dashboard at http://localhost:8050 that reads
grpo_metrics.json and grpo_checkpoint_eval.json as they are written
during training and renders auto-refreshing charts.

No training is started or interrupted — this script is read-only.
It can be run on any machine that has access to the model directory.

Usage
─────
  python scripts/monitor_grpo.py                        # default paths
  python scripts/monitor_grpo.py --port 8080            # custom port
  python scripts/monitor_grpo.py --metrics path/to/grpo_metrics.json
  python scripts/monitor_grpo.py --no-browser           # don't open browser

Dependencies  (standard + one lightweight server)
──────────────────────────────────────────────────
  pip install flask

Charts rendered
───────────────
  Row 1  Reward curve (mean ± std) and zero-std fraction
  Row 2  KL divergence and loss
  Row 3  Checkpoint metrics — CVR and offload F1 / escalation F1
  Row 4  Checkpoint metrics — MNU/ΔU and ID/OOD accuracy

Refresh rate:  30 seconds (configurable via --interval)
"""

import os
import sys
import json
import argparse
import math
import threading
import webbrowser
import time

# ── Path defaults (mirror run_grpo.py layout) ─────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ROOT         = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_METRICS  = os.path.join(ROOT, "models", "grpo_metrics.json")
DEFAULT_CKPT     = os.path.join(ROOT, "models", "grpo_checkpoint_eval.json")


# ═══════════════════════════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_metrics(path):
    """Load grpo_metrics.json — returns empty list on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [d for d in data if "reward" in d]
    except Exception:
        return []


def load_checkpoints(path):
    """Load grpo_checkpoint_eval.json — returns empty list on any error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def summarise(metrics, checkpoints):
    """Compute the summary statistics shown in the KPI header bar."""
    if not metrics:
        return {}

    rewards   = [d["reward"]              for d in metrics]
    zero_stds = [d["frac_reward_zero_std"] for d in metrics]
    kls       = [d["kl"]                  for d in metrics]

    n = len(rewards); q = max(1, n // 4)
    q1 = sum(rewards[:q]) / q
    q3 = sum(rewards[2*q:3*q]) / max(1, len(rewards[2*q:3*q]))

    out = {
        "n_steps":       metrics[-1]["step"],
        "n_entries":     n,
        "reward_max":    max(rewards),
        "reward_mean":   sum(rewards) / n,
        "reward_final":  rewards[-1],
        "zero_std_mean": sum(zero_stds) / n,
        "zero_std_final":zero_stds[-1],
        "kl_max":        max(kls),
        "kl_final":      kls[-1],
        "q1":            q1,
        "q3":            q3,
        "curve":         "RISING" if q3 > q1 + 0.05 else "FLAT",
    }

    if checkpoints:
        best = max(checkpoints, key=lambda c: c.get("mnu", -999))
        out["best_step"]   = best["step"]
        out["best_mnu"]    = best.get("mnu", 0)
        out["best_cvr"]    = best.get("cvr_combined", 1)
        out["best_offf1"]  = best.get("offload_f1", 0)
        out["best_ver"]    = best.get("ver", 0)
        out["n_ckpts"]     = len(checkpoints)
        out["delta_u"]     = best.get("delta_u")

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# HTML page builder
# ═══════════════════════════════════════════════════════════════════════════════

def _js_array(values, decimals=4):
    """Format a Python list as a compact JS array literal."""
    fmt = f"{{:.{decimals}f}}"
    return "[" + ",".join(fmt.format(v) if v is not None else "null"
                          for v in values) + "]"


def build_page(metrics, checkpoints, interval, metrics_path, ckpt_path):
    """Render the complete HTML page as a string."""

    S = summarise(metrics, checkpoints)

    # ── Training series ───────────────────────────────────────────────────────
    steps    = [d["step"]                   for d in metrics]
    reward   = [d["reward"]                 for d in metrics]
    rstd     = [d["reward_std"]             for d in metrics]
    zero_std = [d["frac_reward_zero_std"]   for d in metrics]
    kl       = [d["kl"]                     for d in metrics]
    loss     = [d.get("loss")               for d in metrics]
    grad_n   = [d.get("grad_norm")          for d in metrics]

    # Reward ± std bands
    rew_hi = [r + s for r, s in zip(reward, rstd)]
    rew_lo = [r - s for r, s in zip(reward, rstd)]

    # ── Checkpoint series ─────────────────────────────────────────────────────
    ck_steps   = [c["step"]                         for c in checkpoints]
    ck_cvr     = [c.get("cvr_combined", None)       for c in checkpoints]
    ck_offf1   = [c.get("offload_f1", None)         for c in checkpoints]
    ck_ver     = [c.get("ver", None)                for c in checkpoints]
    ck_er      = [c.get("er", None)                 for c in checkpoints]
    ck_escf1   = [c.get("esc_f1", None)             for c in checkpoints]
    ck_mnu     = [c.get("mnu", None)                for c in checkpoints]
    ck_delta_u = [c.get("delta_u", None)            for c in checkpoints]
    ck_id_acc  = [c.get("id_accuracy", None)        for c in checkpoints]
    ck_ood_acc = [c.get("ood_accuracy", None)       for c in checkpoints]
    ck_mer_ho  = [c.get("mer_ho", None)             for c in checkpoints]

    # ── Status indicators ─────────────────────────────────────────────────────
    if not metrics:
        status_msg   = "Waiting for training to start…"
        status_color = "#BA7517"
    elif S.get("curve") == "RISING":
        status_msg   = f"Training — step {S['n_steps']}  |  Exploration curve: RISING ✓"
        status_color = "#639922"
    else:
        status_msg   = f"Training — step {S['n_steps']}  |  Exploration curve: FLAT  (zero-std {S['zero_std_mean']:.1%})"
        status_color = "#E24B4A"

    n_ckpts   = S.get("n_ckpts", 0)
    last_ckpt = f"step {ckpt_path and checkpoints[-1]['step'] if checkpoints else '—'}"

    # ── KPI bar values ────────────────────────────────────────────────────────
    def kpi(label, value, sub="", color="var(--fg)"):
        return (f'<div class="kpi">'
                f'<div class="kpi-lbl">{label}</div>'
                f'<div class="kpi-val" style="color:{color}">{value}</div>'
                f'<div class="kpi-sub">{sub}</div>'
                f'</div>')

    kpi_html = ""
    if S:
        crv_col = "#639922" if S.get("curve") == "RISING" else "#E24B4A"
        kpi_html += kpi("Max reward",    f"{S.get('reward_max',0):.4f}",
                        f"mean {S.get('reward_mean',0):.4f}",
                        "#BA7517" if S.get('reward_max',0) < 0.5 else "#639922")
        kpi_html += kpi("Zero-std mean", f"{S.get('zero_std_mean',0):.1%}",
                        f"final {S.get('zero_std_final',0):.1%}",
                        "#E24B4A" if S.get('zero_std_mean',0) > 0.5 else "#639922")
        kpi_html += kpi("KL max",        f"{S.get('kl_max',0):.4f}",
                        f"final {S.get('kl_final',0):.4f}",
                        "#639922" if S.get('kl_max',0) > 0.05 else "#BA7517")
        kpi_html += kpi("Curve",         S.get("curve","—"), f"Q1={S.get('q1',0):.3f} → Q3={S.get('q3',0):.3f}", crv_col)
        if "best_mnu" in S:
            kpi_html += kpi("Best MNU",  f"+{S['best_mnu']:.4f}",
                            f"step {S['best_step']}  ΔU=+{S.get('delta_u',0):.3f}" if S.get("delta_u") else f"step {S['best_step']}",
                            "#639922")
            kpi_html += kpi("Best CVR",  f"{S['best_cvr']:.1%}",
                            f"Off-F1={S['best_offf1']:.3f}  VER={S['best_ver']:.3f}",
                            "#639922" if S["best_cvr"] < 0.05 else "#BA7517")
            kpi_html += kpi("Checkpoints", str(S["n_ckpts"]), last_ckpt)

    # ── JS data injection ─────────────────────────────────────────────────────
    has_metrics  = "true"  if metrics     else "false"
    has_ckpts    = "true"  if checkpoints else "false"

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GRPO Monitor</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<style>
  :root{{
    --bg0:#0f1117; --bg1:#181c24; --bg2:#1e2330;
    --border:#2a3040; --fg:#e2e8f0; --fg2:#94a3b8;
    --green:#4ade80; --amber:#fbbf24; --red:#f87171; --blue:#60a5fa;
    --purple:#c084fc;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg0);color:var(--fg);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;font-size:13px;min-height:100vh}}
  header{{background:var(--bg1);border-bottom:1px solid var(--border);padding:12px 20px;display:flex;align-items:center;gap:16px;position:sticky;top:0;z-index:10}}
  header h1{{font-size:15px;font-weight:600;letter-spacing:0.02em}}
  .status{{font-size:12px;padding:3px 10px;border-radius:20px;background:var(--bg2);border:1px solid var(--border);color:{status_color}}}
  .refresh-tag{{margin-left:auto;font-size:11px;color:var(--fg2)}}
  .kpi-bar{{display:flex;gap:8px;padding:12px 20px;background:var(--bg1);border-bottom:1px solid var(--border);overflow-x:auto;flex-wrap:wrap}}
  .kpi{{background:var(--bg2);border:1px solid var(--border);border-radius:8px;padding:8px 14px;min-width:130px;flex-shrink:0}}
  .kpi-lbl{{font-size:10px;color:var(--fg2);text-transform:uppercase;letter-spacing:.06em;margin-bottom:2px}}
  .kpi-val{{font-size:18px;font-weight:600}}
  .kpi-sub{{font-size:10px;color:var(--fg2);margin-top:1px}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:14px 20px}}
  .card{{background:var(--bg1);border:1px solid var(--border);border-radius:10px;padding:14px}}
  .card h2{{font-size:11px;font-weight:500;color:var(--fg2);text-transform:uppercase;letter-spacing:.06em;margin-bottom:10px}}
  .chart-wrap{{position:relative;height:200px}}
  .empty{{display:flex;align-items:center;justify-content:center;height:200px;color:var(--fg2);font-size:12px}}
  .file-bar{{padding:8px 20px;background:var(--bg1);border-top:1px solid var(--border);font-size:11px;color:var(--fg2);display:flex;gap:20px;flex-wrap:wrap}}
  .file-bar span{{color:var(--fg)}}
  @media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>

<header>
  <h1>⚡ GRPO Training Monitor</h1>
  <div class="status">{status_msg}</div>
  <div class="refresh-tag" id="countdown">Refreshing in {interval}s</div>
</header>

<div class="kpi-bar">{kpi_html if kpi_html else '<div style="color:var(--fg2);font-size:12px">No data yet — waiting for grpo_metrics.json…</div>'}</div>

<div class="grid">

  <!-- Reward curve -->
  <div class="card">
    <h2>Reward (mean ± std)  &amp;  Zero-std fraction</h2>
    {'<div class="chart-wrap"><canvas id="cReward"></canvas></div>' if metrics else '<div class="empty">No training data yet</div>'}
  </div>

  <!-- KL + loss -->
  <div class="card">
    <h2>KL divergence  &amp;  Loss</h2>
    {'<div class="chart-wrap"><canvas id="cKL"></canvas></div>' if metrics else '<div class="empty">No training data yet</div>'}
  </div>

  <!-- CVR + offload F1 evolution across checkpoints -->
  <div class="card">
    <h2>Checkpoint: CVR  &amp;  Offload F1  &amp;  Escalation F1</h2>
    {'<div class="chart-wrap"><canvas id="cCVR"></canvas></div>' if checkpoints else '<div class="empty">No checkpoint evals yet</div>'}
  </div>

  <!-- MNU + accuracy evolution across checkpoints -->
  <div class="card">
    <h2>Checkpoint: MNU  /  ΔU  /  ID acc  /  OOD acc</h2>
    {'<div class="chart-wrap"><canvas id="cMNU"></canvas></div>' if checkpoints else '<div class="empty">No checkpoint evals yet</div>'}
  </div>

  <!-- Reward std + grad norm -->
  <div class="card">
    <h2>Reward std dev  &amp;  Gradient norm</h2>
    {'<div class="chart-wrap"><canvas id="cStd"></canvas></div>' if metrics else '<div class="empty">No training data yet</div>'}
  </div>

  <!-- Escalation VER / ER / MER_HO -->
  <div class="card">
    <h2>Checkpoint: VER  /  ER  /  MER-HO</h2>
    {'<div class="chart-wrap"><canvas id="cEsc"></canvas></div>' if checkpoints else '<div class="empty">No checkpoint evals yet</div>'}
  </div>

</div>

<div class="file-bar">
  <div>Metrics file: <span>{metrics_path}</span></div>
  <div>Checkpoint eval: <span>{ckpt_path}</span></div>
  <div>Entries loaded: <span>{len(metrics)} training  /  {len(checkpoints)} checkpoint evals</span></div>
</div>

<script>
const HAS_METRICS = {has_metrics};
const HAS_CKPTS   = {has_ckpts};

// ── Training data ─────────────────────────────────────────────────────────────
const steps    = {_js_array(steps,   0)};
const reward   = {_js_array(reward)};
const rew_hi   = {_js_array(rew_hi)};
const rew_lo   = {_js_array(rew_lo)};
const zero_std = {_js_array(zero_std)};
const kl       = {_js_array(kl)};
const loss     = {_js_array(loss)};
const grad_n   = {_js_array(grad_n)};
const rstd     = {_js_array(rstd)};

// ── Checkpoint data ───────────────────────────────────────────────────────────
const ck_steps   = {_js_array(ck_steps,   0)};
const ck_cvr     = {_js_array(ck_cvr)};
const ck_offf1   = {_js_array(ck_offf1)};
const ck_ver     = {_js_array(ck_ver)};
const ck_er      = {_js_array(ck_er)};
const ck_escf1   = {_js_array(ck_escf1)};
const ck_mnu     = {_js_array(ck_mnu)};
const ck_delta_u = {_js_array(ck_delta_u)};
const ck_id_acc  = {_js_array(ck_id_acc)};
const ck_ood_acc = {_js_array(ck_ood_acc)};
const ck_mer_ho  = {_js_array(ck_mer_ho)};

// ── Chart defaults ────────────────────────────────────────────────────────────
const grid   = {{color:'rgba(255,255,255,0.06)'}};
const tick   = {{color:'#94a3b8', font:{{size:10}}}};
const legend = {{display:true, labels:{{boxWidth:8, font:{{size:10}}, color:'#94a3b8'}}}};

function opts(yExtra={{}}) {{
  return {{
    responsive:true, maintainAspectRatio:false, animation:false,
    plugins:{{legend, tooltip:{{}}}},
    scales:{{
      x:{{grid, ticks:{{...tick, maxTicksLimit:8}}}},
      y:{{grid, ticks:tick, ...yExtra}}
    }}
  }};
}}

function optsY2(yLeft={{}}, yRight={{}}) {{
  return {{
    responsive:true, maintainAspectRatio:false, animation:false,
    plugins:{{legend, tooltip:{{}}}},
    scales:{{
      x:{{grid, ticks:{{...tick, maxTicksLimit:8}}}},
      yL:{{type:'linear', position:'left',  grid, ticks:tick, ...yLeft}},
      yR:{{type:'linear', position:'right', grid:{{drawOnChartArea:false}}, ticks:tick, ...yRight}}
    }}
  }};
}}

// ── Colour palette ────────────────────────────────────────────────────────────
const C = {{
  blue:   '#60a5fa', green:  '#4ade80', amber:  '#fbbf24',
  red:    '#f87171', purple: '#c084fc', teal:   '#2dd4bf',
  band:   'rgba(96,165,250,0.12)',
}};

// ── Chart 1: Reward ± std  +  zero-std ───────────────────────────────────────
if (HAS_METRICS) {{
  new Chart('cReward', {{
    type:'line',
    data:{{
      labels: steps,
      datasets:[
        // Std band (fill between hi and lo)
        {{label:'reward+std', data:rew_hi, borderColor:'transparent',
          backgroundColor:C.band, pointRadius:0, fill:'+1', tension:0.3}},
        {{label:'reward-std', data:rew_lo, borderColor:'transparent',
          backgroundColor:C.band, pointRadius:0, fill:false, tension:0.3}},
        {{label:'reward',    data:reward,   borderColor:C.blue,  borderWidth:1.5,
          pointRadius:0, fill:false, tension:0.3, yAxisID:'yL'}},
        {{label:'zero-std ▶', data:zero_std, borderColor:C.red,  borderWidth:1.5,
          pointRadius:0, fill:false, tension:0.3, yAxisID:'yR', borderDash:[4,2]}},
        // Reference lines
        {{label:'target 0.5', data:steps.map(()=>0.5),  borderColor:'#4ade8044',
          borderWidth:1, borderDash:[6,3], pointRadius:0, yAxisID:'yL'}},
        {{label:'healthy<0.3', data:steps.map(()=>0.3), borderColor:'#f8717144',
          borderWidth:1, borderDash:[6,3], pointRadius:0, yAxisID:'yR'}},
      ]
    }},
    options: optsY2({{min:-0.05, max:1.3}}, {{min:0, max:1.05}})
  }});
}}

// ── Chart 2: KL  +  Loss ─────────────────────────────────────────────────────
if (HAS_METRICS) {{
  const validLoss = loss.some(v=>v!==null);
  new Chart('cKL', {{
    type:'line',
    data:{{
      labels: steps,
      datasets:[
        {{label:'KL',   data:kl,   borderColor:C.amber,  borderWidth:1.5,
          pointRadius:0, tension:0.3, fill:false, yAxisID:'yL'}},
        {{label:'target KL>0.05', data:steps.map(()=>0.05), borderColor:'#4ade8044',
          borderWidth:1, borderDash:[6,3], pointRadius:0, yAxisID:'yL'}},
        ...(validLoss ? [{{label:'loss ▶', data:loss, borderColor:C.purple, borderWidth:1.5,
          pointRadius:0, tension:0.3, fill:false, yAxisID:'yR'}}] : []),
      ]
    }},
    options: optsY2({{min:0}}, {{min:0}})
  }});
}}

// ── Chart 3: Checkpoint CVR / Offload F1 / Esc F1 ────────────────────────────
if (HAS_CKPTS) {{
  new Chart('cCVR', {{
    type:'line',
    data:{{
      labels: ck_steps,
      datasets:[
        {{label:'CVR_combined ↓', data:ck_cvr,   borderColor:C.red,    borderWidth:2,
          pointRadius:4, tension:0.2, fill:false}},
        {{label:'Offload F1 ↑',  data:ck_offf1, borderColor:C.blue,   borderWidth:2,
          pointRadius:4, tension:0.2, fill:false}},
        {{label:'Esc F1 ↑',      data:ck_escf1, borderColor:C.purple, borderWidth:2,
          pointRadius:4, tension:0.2, fill:false}},
        {{label:'CVR target 2%', data:ck_steps.map(()=>0.02), borderColor:'#f8717144',
          borderWidth:1, borderDash:[5,3], pointRadius:0}},
      ]
    }},
    options: opts({{min:0, max:1}})
  }});
}}

// ── Chart 4: MNU / ΔU / Accuracy ─────────────────────────────────────────────
if (HAS_CKPTS) {{
  new Chart('cMNU', {{
    type:'line',
    data:{{
      labels: ck_steps,
      datasets:[
        {{label:'MNU ↑',       data:ck_mnu,     borderColor:C.green,  borderWidth:2,
          pointRadius:4, tension:0.2, fill:false, yAxisID:'yL'}},
        {{label:'ΔU vs SFT ↑', data:ck_delta_u, borderColor:C.teal,  borderWidth:2,
          pointRadius:4, tension:0.2, fill:false, yAxisID:'yL', borderDash:[4,2]}},
        {{label:'ID acc ↑',    data:ck_id_acc,  borderColor:C.blue,  borderWidth:1.5,
          pointRadius:3, tension:0.2, fill:false, yAxisID:'yR'}},
        {{label:'OOD acc ↑',   data:ck_ood_acc, borderColor:C.amber, borderWidth:1.5,
          pointRadius:3, tension:0.2, fill:false, yAxisID:'yR'}},
        {{label:'SFT MNU',     data:ck_steps.map(()=>0.1676), borderColor:'#ffffff22',
          borderWidth:1, borderDash:[6,3], pointRadius:0, yAxisID:'yL'}},
      ]
    }},
    options: optsY2({{min:-0.05}}, {{min:0, max:1}})
  }});
}}

// ── Chart 5: Reward std + Grad norm ──────────────────────────────────────────
if (HAS_METRICS) {{
  const validGrad = grad_n.some(v=>v!==null);
  new Chart('cStd', {{
    type:'line',
    data:{{
      labels: steps,
      datasets:[
        {{label:'reward std',  data:rstd,  borderColor:C.blue,  borderWidth:1.5,
          pointRadius:0, tension:0.3, fill:false, yAxisID:'yL'}},
        ...(validGrad ? [{{label:'grad norm ▶', data:grad_n, borderColor:C.amber, borderWidth:1,
          pointRadius:0, tension:0.3, fill:false, yAxisID:'yR'}}] : []),
      ]
    }},
    options: optsY2({{min:0}}, {{min:0}})
  }});
}}

// ── Chart 6: VER / ER / MER_HO ───────────────────────────────────────────────
if (HAS_CKPTS) {{
  new Chart('cEsc', {{
    type:'line',
    data:{{
      labels: ck_steps,
      datasets:[
        {{label:'VER ↑',    data:ck_ver,    borderColor:C.green,  borderWidth:2,
          pointRadius:4, tension:0.2, fill:false}},
        {{label:'ER ↑',     data:ck_er,     borderColor:C.blue,   borderWidth:2,
          pointRadius:4, tension:0.2, fill:false}},
        {{label:'MER_HO ↓', data:ck_mer_ho, borderColor:C.red,    borderWidth:2,
          pointRadius:4, tension:0.2, fill:false, borderDash:[4,2]}},
        {{label:'ER target>40%', data:ck_steps.map(()=>0.40), borderColor:'#4ade8044',
          borderWidth:1, borderDash:[5,3], pointRadius:0}},
      ]
    }},
    options: opts({{min:0, max:1.05}})
  }});
}}

// ── Auto-refresh countdown ────────────────────────────────────────────────────
let secs = {interval};
const cd = document.getElementById('countdown');
setInterval(()=>{{
  secs--;
  if (secs <= 0) {{
    window.location.reload();
  }} else {{
    cd.textContent = `Refreshing in ${{secs}}s`;
  }}
}}, 1000);
</script>
</body>
</html>"""

    return page


# ═══════════════════════════════════════════════════════════════════════════════
# Flask server
# ═══════════════════════════════════════════════════════════════════════════════

def create_app(metrics_path, ckpt_path, interval):
    try:
        from flask import Flask, Response
    except ImportError:
        print("[ERROR] Flask is required: pip install flask")
        sys.exit(1)

    app = Flask(__name__)

    @app.route("/")
    def index():
        metrics     = load_metrics(metrics_path)
        checkpoints = load_checkpoints(ckpt_path)
        html        = build_page(metrics, checkpoints, interval,
                                  metrics_path, ckpt_path)
        return Response(html, mimetype="text/html")

    @app.route("/data")
    def data():
        """JSON endpoint for programmatic access."""
        metrics     = load_metrics(metrics_path)
        checkpoints = load_checkpoints(ckpt_path)
        return {
            "n_metric_entries": len(metrics),
            "n_checkpoints":    len(checkpoints),
            "summary":          summarise(metrics, checkpoints),
        }

    return app


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Live GRPO training monitor — serves a dashboard at localhost"
    )
    parser.add_argument("--metrics",    default=DEFAULT_METRICS,
                        help=f"Path to grpo_metrics.json (default: {DEFAULT_METRICS})")
    parser.add_argument("--checkpoints", default=DEFAULT_CKPT,
                        help=f"Path to grpo_checkpoint_eval.json (default: {DEFAULT_CKPT})")
    parser.add_argument("--port",       type=int, default=8050,
                        help="Port to serve on (default: 8050)")
    parser.add_argument("--interval",   type=int, default=30,
                        help="Page auto-refresh interval in seconds (default: 30)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open the browser automatically")
    args = parser.parse_args()

    metrics_path = os.path.abspath(args.metrics)
    ckpt_path    = os.path.abspath(args.checkpoints)

    print(f"\n  GRPO Monitor")
    print(f"  ────────────────────────────────────────────────────")
    print(f"  Metrics file    : {metrics_path}")
    print(f"  Checkpoint eval : {ckpt_path}")
    print(f"  Dashboard URL   : http://localhost:{args.port}")
    print(f"  Refresh rate    : every {args.interval}s")
    print(f"  ────────────────────────────────────────────────────")

    if not os.path.exists(metrics_path):
        print(f"  [INFO] Metrics file not found yet — dashboard will show")
        print(f"         'waiting' state until training starts writing it.")
    if not os.path.exists(ckpt_path):
        print(f"  [INFO] Checkpoint eval file not found yet — checkpoint")
        print(f"         charts will appear after the first eval sweep.")

    print(f"\n  Press Ctrl+C to stop the monitor.\n")

    app = create_app(metrics_path, ckpt_path, args.interval)

    if not args.no_browser:
        def open_browser():
            time.sleep(1.2)
            webbrowser.open(f"http://localhost:{args.port}")
        threading.Thread(target=open_browser, daemon=True).start()

    # Disable Flask's startup banner and request logging for cleaner output
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()