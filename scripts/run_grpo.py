"""
run_grpo.py  —  GRPO Reinforcement Learning for the UE AI Agent
----------------------------------------------------------------
Trains the SFT-initialised model using GRPO to optimise the joint
offload and escalation utility function:

  U(o, e | s) = wR·R(o,s) + wV·V(e,s) − wC·C(e,s)
               − wP·P(o,s) − μ·Φ(o,y,α,η)

Outputs scored
──────────────
  offload_decision   o ∈ {true, false}
  escalation_flag    e ∈ {true, false}

State variables in the prompt
──────────────────────────────
  β   battery level       ("Battery is at X%.")
  α   app type            ("User just opened a <app>.")
  η   network condition   ("Network is <stable|congested|weak signal>.")
  γ   energy credit       ("Energy credit is <desc> (X%).")

Prompt format  (identical to run_sft.py and evaluate.py)
─────────────────────────────────────────────────────────
  UE Context: <natural language state>
  Required keys (offload_decision, escalation_flag):
  Decision:
  {

Dataset and oversampling
─────────────────────────
  Mirrors the stratified structure of generate_dataset.py.
  Hard-constraint and ambiguity groups are oversampled 2× to
  maintain strong gradient signal throughout training:
    c1_only, c8_only, c1_and_c8, low_confidence_proxy  →  2×
    general, esc_zone, boundary                         →  1×

Checkpoint saving
─────────────────
  save_steps=20, save_total_limit=25
  With ~900 augmented samples, batch_size=4, grad_accum=4:
    steps_per_epoch ≈ 900 / 16 ≈ 56 steps
    total_steps     ≈ 56 × 8  ≈ 450 steps
  → ~22 step-based saves + 8 epoch saves = ~30 checkpoints total.

Checkpoint evaluation sweep
────────────────────────────
  After training, all saved checkpoints are evaluated against all
  four metric families from evaluate.py:

    Family 1  CVR_C1, CVR_C8, CVR_credit, CVR_combined
    Family 2  COR, offload precision, recall, F1
    Family 3  VER, ER, UER, escalation F1, MER_HO
    Family 4  MNU, CCR, parse fail

  Results are written to models/grpo_checkpoint_eval.json and a
  comparison table is printed at the end showing metric progression
  and the best checkpoint per criterion.

  Unlike run_sft.py, GRPO checkpoints save the full merged model
  (not just a LoRA adapter), so no merge step is needed during the
  sweep — each checkpoint is loaded directly as AutoModelForCausalLM.

Checkpoint resumption
─────────────────────
  Full resumption (weights + optimiser + LR schedule + RNG):
    python scripts/run_grpo.py --resume
    python scripts/run_grpo.py --checkpoint models/grpo-ue-agent/checkpoint-400

  Fresh start:
    python scripts/run_grpo.py

Flags
─────
  --live-eval          Evaluate checkpoints in the background
                       as training progresses (recommended for
                       long CPU runs — updates monitor in real time)
  --eval-interval N    Polling interval in seconds for --live-eval
                       (default: 300 s — do not set below 60 on CPU)
  --no-eval            Skip post-training checkpoint sweep
  --eval-only          Skip training, only run the checkpoint sweep
  --save-steps N       Checkpoint every N steps (default: 400)

Simulation parameters
───────────────────────────────────
  See reward_model.py for the full parameter set.
  Key GRPO hyperparameters:
    lr       = 5e-5   learning rate
    beta     = 0.03   KL penalty (low — allows constraint-boundary exploration)
    temp     = 0.8    sampling temperature
    n_gen    = 8      completions per prompt
    epochs   = 8
    warmup   = 5%     linear warmup → cosine decay
"""

import os
import re
import sys
import json
import glob
import argparse
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scripts.reward_model import calculate_ue_reward

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
SFT_MODEL     = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models", "sft-ue-merged"))
DATA_FILE     = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data",   "train_data.jsonl"))
OUTPUT_DIR    = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models", "grpo-ue-agent"))
METRICS_FILE  = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models", "grpo_metrics.json"))
CKPT_EVAL     = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "models", "grpo_checkpoint_eval.json"))
TRACE_FILE    = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "data",   "radio_parameters_moving_clean.csv"))

GRPO_METRIC_KEYS = [
    "reward", "reward_std", "reward_logic", "reward_logic_std",
    "kl", "loss", "grad_norm", "frac_reward_zero_std"
]
# reward_logic = mean of U(o,e|s) only (schema reward excluded).
# Plotting reward_logic separately reveals the true learning curve
# without the +0.30 schema floor that compresses the visible range.


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ═══════════════════════════════════════════════════════════════════════════════

def find_latest_checkpoint(output_dir):
    """Return the checkpoint with the highest step number, or None."""
    candidates = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not candidates:
        return None

    def step_of(path):
        try:
            return int(os.path.basename(path).split("-")[-1])
        except ValueError:
            return -1

    latest = max(candidates, key=step_of)
    return latest if step_of(latest) >= 0 else None


def list_checkpoints(output_dir):
    """Return all checkpoint directories sorted by step number ascending."""
    candidates = glob.glob(os.path.join(output_dir, "checkpoint-*"))

    def step_of(path):
        try:
            return int(os.path.basename(path).split("-")[-1])
        except ValueError:
            return -1

    return sorted([d for d in candidates if step_of(d) >= 0], key=step_of)


def validate_checkpoint(checkpoint_dir):
    """Verify a checkpoint has the required files for resumption."""
    required = ["adapter_model.safetensors", "adapter_config.json"]
    optional = ["optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"]

    missing_req = [f for f in required
                   if not os.path.exists(os.path.join(checkpoint_dir, f))]
    missing_opt = [f for f in optional
                   if not os.path.exists(os.path.join(checkpoint_dir, f))]

    if missing_req:
        raise FileNotFoundError(
            f"Checkpoint at {checkpoint_dir} is missing: {missing_req}"
        )
    if missing_opt:
        print(f"  [WARN] Optional files missing from checkpoint: {missing_opt}")
        print(f"         Optimizer/scheduler state will not be restored.")


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_prompt(example):
    """
    Format the dataset prompt for GRPO generation.
    Format is identical to run_sft.py and evaluate.py.
    The opening brace prefill forces the model to begin with a JSON object.
    """
    example["prompt"] = (
        f"UE Context: {example['prompt']}\n"
        f"Required keys (offload_decision, escalation_flag):\n"
        f"Decision:\n"
        f"{{"
    )
    return example


# ═══════════════════════════════════════════════════════════════════════════════
# Stratum filters for oversampling
# ═══════════════════════════════════════════════════════════════════════════════

def is_c1(example):
    r = json.loads(example["response"])
    return r.get("c1_active", False) and not r.get("c8_active", False)

def is_c8(example):
    r = json.loads(example["response"])
    return r.get("c8_active", False) and not r.get("c1_active", False)

def is_c1_and_c8(example):
    r = json.loads(example["response"])
    return r.get("c1_active", False) and r.get("c8_active", False)

def is_low_confidence_proxy(example):
    return (example.get("meta") or {}).get("stratum") == "low_confidence_proxy"


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset loading
# ═══════════════════════════════════════════════════════════════════════════════

def is_escalation_warranted(example):
    """True when the ground-truth label has escalation_flag=True."""
    try:
        return bool(json.loads(example["response"]).get("escalation_flag", False))
    except Exception:
        return False


def is_esc_zone(example):
    """esc_zone stratum: credits between γ_min and γ_esc."""
    return (example.get("meta") or {}).get("stratum") == "esc_zone"


def load_and_prepare_dataset():
    """
    Load dataset with escalation-focused augmentation for run 4.

    Changes vs run 3
    ─────────────────
    1. esc_zone and low_confidence_proxy oversampled at 3× (was 2×)
       — these strata contain all structurally ambiguous states where
       escalation is warranted.

    2. 50/50 escalation-balanced interleaving applied after augmentation
       — guarantees that each batch of 4 prompts contains at least 2
       escalation-warranted samples, raising the effective escalation
       gradient signal from ~23% to 50% of batch slots.
       Without this, most batches had 0–1 escalation-warranted prompts
       and the GRPO gradient was dominated by the non-escalation signal.

    3. Constraint strata (c1, c8, c1∩c8) remain at 2× — constraint
       safety is already learned (CVR=3.1% at best checkpoint); further
       upweighting would compete with the escalation signal.
    """
    from datasets import interleave_datasets

    print("Loading dataset …")
    raw    = load_dataset("json", data_files=DATA_FILE, split="train")
    n_base = len(raw)

    c1_set   = raw.filter(is_c1)
    c8_set   = raw.filter(is_c8)
    c18_set  = raw.filter(is_c1_and_c8)
    lcp_set  = raw.filter(is_low_confidence_proxy)
    esc_set  = raw.filter(is_esc_zone)

    print(f"  Base dataset              : {n_base}")
    print(f"  C1 only                   : {len(c1_set)}  ({100*len(c1_set)/n_base:.0f}%)")
    print(f"  C8 only                   : {len(c8_set)}  ({100*len(c8_set)/n_base:.0f}%)")
    print(f"  C1 ∩ C8                   : {len(c18_set)}  ({100*len(c18_set)/n_base:.0f}%)")
    print(f"  Low-confidence proxy      : {len(lcp_set)}  ({100*len(lcp_set)/n_base:.0f}%)")
    print(f"  Esc zone                  : {len(esc_set)}  ({100*len(esc_set)/n_base:.0f}%)")

    # Constraint strata: 2× | Escalation strata: 3× (change 1)
    augmented = concatenate_datasets([
        raw,
        c1_set,  c8_set,  c18_set,          # 2× constraint strata
        lcp_set, lcp_set,                    # 3× low-confidence proxy
        esc_set, esc_set,                    # 3× escalation zone
    ]).shuffle(seed=42)

    n_aug = len(augmented)
    print(f"  Augmented total           : {n_aug}")

    # 50/50 escalation-balanced interleaving (change 2)
    # Partition augmented dataset into escalation-warranted and not
    esc_warranted    = augmented.filter(is_escalation_warranted)
    esc_not_warranted= augmented.filter(lambda e: not is_escalation_warranted(e))
    print(f"  Esc-warranted samples     : {len(esc_warranted)}  ({100*len(esc_warranted)/n_aug:.0f}%)")
    print(f"  Non-esc samples           : {len(esc_not_warranted)}  ({100*len(esc_not_warranted)/n_aug:.0f}%)")

    balanced = interleave_datasets(
        [esc_warranted, esc_not_warranted],
        probabilities=[0.50, 0.50],
        seed=42,
    )
    print(f"  Balanced (50/50 esc mix)  : {len(balanced)}")

    cols_to_remove = [c for c in balanced.column_names if c != "prompt"]
    return balanced.map(build_prompt, remove_columns=cols_to_remove)


# ═══════════════════════════════════════════════════════════════════════════════
# Reward function
# ═══════════════════════════════════════════════════════════════════════════════

# Module-level buffer: stores the most recent batch's U-component values
# so that the custom_metrics callback can log reward_logic to trainer state.
_last_U_values: list = []


def ue_reward_fn(prompts, completions, **kwargs):
    """
    GRPO reward function.
    Returns one scalar per (prompt, completion) pair.

    Also populates _last_U_values with the logic-only U(o,e|s) component
    so that the custom_metrics_callback can log reward_logic separately
    from the schema reward.  This enables the reward decomposition chart
    in monitor_grpo.py to show the true learning curve without the +0.30
    schema floor compressing the visible range.
    """
    global _last_U_values
    rewards = []
    U_vals  = []
    for prompt, completion in zip(prompts, completions):
        total, U = calculate_ue_reward(
            prompt, completion, verbose=True, return_components=True
        )
        rewards.append(total)
        U_vals.append(U)
    _last_U_values = U_vals
    return rewards


# ═══════════════════════════════════════════════════════════════════════════════
# Training metrics helpers
# ═══════════════════════════════════════════════════════════════════════════════

def save_metrics(entries):
    """Append entries to grpo_metrics.json (creates file if absent)."""
    os.makedirs(os.path.dirname(METRICS_FILE), exist_ok=True)
    existing = []
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
    existing.extend(entries)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4)


def print_training_summary(log_history):
    """Print reward curve summary and exploration arc assessment."""
    entries   = [e for e in log_history if "reward" in e]
    if not entries:
        print("  [no reward entries in log history]")
        return

    rewards   = [e["reward"]                      for e in entries]
    zero_stds = [e.get("frac_reward_zero_std", 0) for e in entries]
    n = len(rewards)
    q = max(1, n // 4)

    print(f"\n  [GRPO training summary]")
    print(f"    reward   — mean: {sum(rewards)/n:.4f}"
          f"  max: {max(rewards):.4f}  final: {rewards[-1]:.4f}")
    print(f"    zero-std — mean: {sum(zero_stds)/n:.3f}"
          f"  final: {zero_stds[-1]:.3f}")
    print(f"    reward by quarter:")
    for i in range(4):
        chunk = rewards[i*q:(i+1)*q]
        if chunk:
            print(f"      Q{i+1}: mean={sum(chunk)/len(chunk):.4f}"
                  f"  max={max(chunk):.4f}")

    q1 = sum(rewards[:q]) / q
    q3 = sum(rewards[2*q:3*q]) / max(1, len(rewards[2*q:3*q]))
    if q3 > q1 + 0.05:
        print(f"    Exploration curve: YES  (Q1={q1:.4f} → Q3={q3:.4f})")
    else:
        print(f"    Exploration curve: FLAT (Q1={q1:.4f}, Q3={q3:.4f})")
        print(f"    HINT: check for zero-std collapse; increase num_generations")

    # reward_logic (U only) tells us if the logic reward exceeds 0 —
    # a positive mean U means the model earns more from correct decisions
    # than it loses from wrong ones, independent of the schema floor.
    logic_vals = [e.get("reward_logic") for e in entries
                  if e.get("reward_logic") is not None]
    if logic_vals:
        print(f"    reward_logic mean: {sum(logic_vals)/len(logic_vals):+.4f}"              f"  max: {max(logic_vals):+.4f}  final: {logic_vals[-1]:+.4f}")
        print(f"    reward_logic > 0 ever: "
              f"{'YES — U positive' if max(logic_vals) > 0 else 'NO — U always negative'}")
    print(f"    Reward > 0.5 ever: "
          f"{'YES — learning correct decisions' if max(rewards) > 0.5 else 'NO — check hyperparameters'}")


# ═══════════════════════════════════════════════════════════════════════════════
# LoRA config
# ═══════════════════════════════════════════════════════════════════════════════

LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Live checkpoint watcher  (runs as background thread during training)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_eval_state(tokenizer):
    """
    Import evaluate.py, build test sets, and evaluate the SFT baseline.
    Returns (ev, test_sets, sft_mnu) or (None, None, None) on failure.
    Called once by LiveCheckpointWatcher at startup.
    """
    import importlib.util
    try:
        eval_path = os.path.normpath(os.path.join(SCRIPT_DIR, "evaluate.py"))
        spec      = importlib.util.spec_from_file_location("evaluate", eval_path)
        ev        = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
    except Exception as e:
        print(f"  [watcher] Could not import evaluate.py: {e}")
        return None, None, None

    print("  [watcher] Building test sets …")
    test_sets = {
        "id_set":       ev.build_id_test_set(n=100),
        "c1_set":       ev.build_c1_set(n=60),
        "c8_set":       ev.build_c8_set(n=60),
        "combined_set": ev.build_combined_constraint_set(),
        "credit_set":   ev.build_credit_floor_set(n=50),
        "ood_set":      ev.build_ood_test_set(),
        "mob_set":      ev.build_mobility_set(TRACE_FILE),
    }
    print(f"  [watcher] Test sets ready "
          f"({sum(len(v) for v in test_sets.values())} samples per checkpoint)")

    sft_mnu = None
    try:
        print("  [watcher] Evaluating SFT baseline for ΔU …")
        sft_model = AutoModelForCausalLM.from_pretrained(
            SFT_MODEL, torch_dtype=torch.float32, device_map={"": "cpu"}
        )
        sft_model.eval()
        sft_res = ev.evaluate_baseline(
            name="_sft_",
            get_pred_fn=lambda p: ev.generate_config(sft_model, tokenizer, p),
            id_set=test_sets["id_set"], c1_set=test_sets["c1_set"],
            c8_set=test_sets["c8_set"], combined_set=test_sets["combined_set"],
            credit_set=test_sets["credit_set"], ood_set=test_sets["ood_set"],
            mobility_set=test_sets["mob_set"], quiet=True,
        )
        sft_mnu = sft_res["mnu"]
        print(f"  [watcher] SFT baseline MNU = {sft_mnu:+.4f}")
        del sft_model
    except Exception as e:
        print(f"  [watcher] SFT baseline eval failed: {e}")

    return ev, test_sets, sft_mnu


class LiveCheckpointWatcher:
    """
    Background thread that watches OUTPUT_DIR for new checkpoints and
    evaluates each one incrementally, appending results to
    grpo_checkpoint_eval.json as training progresses.

    This lets the monitoring dashboard (monitor_grpo.py) show metric
    evolution in near-real-time during a long CPU training run, rather
    than only after the full run completes.

    Design
    ──────
    - Polls OUTPUT_DIR every `interval` seconds for new checkpoint-N dirs.
    - Tracks already-evaluated steps in a set to avoid re-evaluation.
    - Loads existing grpo_checkpoint_eval.json on startup so resumptions
      correctly skip checkpoints that were evaluated in a previous run.
    - Each evaluation is done in the calling thread (not sub-threaded)
      to avoid BLAS contention with the training process; the training
      process is paused by GIL while the watcher holds the CPU between
      gradient steps.
    - Writes are atomic: results are appended to a temp file then renamed.

    Memory
    ──────
    Each checkpoint evaluation loads the base SFT model, applies the
    LoRA adapter, runs inference on the test sets, then explicitly
    deletes the merged model and calls gc.collect() to free memory
    before the next evaluation.  Peak memory during eval is ~2× the
    model size (base + merged copy).

    Usage
    ─────
    watcher = LiveCheckpointWatcher(tokenizer, ev, test_sets, sft_mnu,
                                    interval=300)
    watcher.start()
    trainer.train()          # blocking
    watcher.stop()
    watcher.join(timeout=60)
    """

    def __init__(self, tokenizer, ev, test_sets, sft_mnu,
                 interval=300, quiet=False):
        import threading
        self.tokenizer  = tokenizer
        self.ev         = ev
        self.test_sets  = test_sets
        self.sft_mnu    = sft_mnu
        self.interval   = interval
        self.quiet      = quiet
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._run,
                                           name="ckpt-watcher",
                                           daemon=True)

        # Pre-populate already-evaluated steps from existing file
        self._done_steps = set()
        if os.path.exists(CKPT_EVAL):
            try:
                with open(CKPT_EVAL, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                self._done_steps = {r["step"] for r in existing}
                print(f"  [watcher] Found {len(self._done_steps)} "
                      f"already-evaluated steps — will skip them.")
            except Exception:
                pass

    def start(self):
        self._thread.start()
        print(f"  [watcher] Started — polling every {self.interval}s "
              f"for new checkpoints.")

    def stop(self):
        self._stop.set()

    def join(self, timeout=60):
        self._thread.join(timeout=timeout)

    def _run(self):
        import gc
        header_printed = False

        while not self._stop.is_set():
            try:
                checkpoints = list_checkpoints(OUTPUT_DIR)
                new_ckpts   = [c for c in checkpoints
                               if int(os.path.basename(c).split("-")[-1])
                               not in self._done_steps]

                for ckpt_path in new_ckpts:
                    if self._stop.is_set():
                        break

                    step = int(os.path.basename(ckpt_path).split("-")[-1])

                    if not self.quiet:
                        print(f"\n  [watcher] Evaluating checkpoint-{step} …",
                              flush=True)

                    try:
                        merged  = load_checkpoint_model(ckpt_path,
                                                        quiet=self.quiet)
                        metrics = run_checkpoint_eval(merged, self.tokenizer,
                                                      self.ev, self.test_sets)
                        del merged
                        gc.collect()
                    except Exception as e:
                        print(f"  [watcher] Eval failed at step {step}: {e}")
                        self._done_steps.add(step)   # skip next time
                        continue

                    delta_u = ((metrics["mnu"] - self.sft_mnu)
                               if self.sft_mnu is not None else None)

                    record = {
                        "checkpoint": ckpt_path,
                        "step":       step,
                        "delta_u":    round(delta_u, 4) if delta_u is not None else None,
                        **metrics,
                    }

                    # Atomic append to grpo_checkpoint_eval.json
                    self._append_record(record)
                    self._done_steps.add(step)

                    # Print one-line summary to console
                    if not self.quiet:
                        if not header_printed:
                            print(f"  {'Step':>6}  {'CVR':>6}  {'ID':>6}"
                                  f"  {'Off-F1':>7}  {'VER':>6}  {'ER':>6}"
                                  f"  {'MNU':>8}  {'ΔU':>7}")
                            print("  " + "-" * 62)
                            header_printed = True
                        delu_s = (f"{delta_u:+.3f}"
                                  if delta_u is not None else "  n/a")
                        print(
                            f"  {step:>6}  "
                            f"{metrics['cvr_combined']:>6.3f}  "
                            f"{metrics['id_accuracy']:>6.3f}  "
                            f"{metrics['offload_f1']:>7.3f}  "
                            f"{metrics['ver']:>6.3f}  "
                            f"{metrics['er']:>6.3f}  "
                            f"{metrics['mnu']:>+8.4f}  "
                            f"{delu_s:>7}",
                            flush=True,
                        )

            except Exception as e:
                print(f"  [watcher] Polling error: {e}")

            # Wait for next poll (interruptible by stop event)
            self._stop.wait(timeout=self.interval)

        print("  [watcher] Stopped.")

    def _append_record(self, record):
        """
        Atomically append one record to grpo_checkpoint_eval.json.

        Reads the current file, appends the new record, writes to a
        temp file, then renames atomically.  This prevents the monitor
        from reading a half-written file.
        """
        existing = []
        if os.path.exists(CKPT_EVAL):
            try:
                with open(CKPT_EVAL, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception:
                existing = []

        # Remove any stale entry for this step (shouldn't happen, but safe)
        existing = [r for r in existing if r["step"] != record["step"]]
        existing.append(record)

        # Sort by step so the file is always in chronological order
        existing.sort(key=lambda r: r["step"])

        tmp = CKPT_EVAL + ".tmp"
        os.makedirs(os.path.dirname(CKPT_EVAL), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=4)
        os.replace(tmp, CKPT_EVAL)   # atomic on all platforms


# ═══════════════════════════════════════════════════════════════════════════════
# GRPO training
# ═══════════════════════════════════════════════════════════════════════════════

def run_grpo(dataset, tokenizer, checkpoint=None, save_steps=400,
             _live_eval_interval=None):
    """
    Run GRPO training from the SFT-merged model.

    Checkpoint saving
    ─────────────────
    save_steps=20 with ~450 total steps gives ~22 intermediate saves
    plus 8 end-of-epoch saves ≈ 30 checkpoints total.
    save_total_limit=25 retains the 25 most recent.
    Each GRPO checkpoint saves the full LoRA adapter (~20 MB),
    so 25 checkpoints ≈ 500 MB total.

    Hyperparameters
    ───────────────
    lr=5e-5        Tuned value from earlier runs
    beta=0.03      Low KL — allows exploration away from SFT reference
    temp=0.8       Diversity without instability
    num_gen=8      8 completions per prompt for reward variance
    epochs=8       Full exploration arc
    warmup=5%      DeepSeek R1-style flat start → cosine decay
    """
    print("\n" + "=" * 64)
    if checkpoint:
        step = os.path.basename(checkpoint).split("-")[-1]
        print(f"  GRPO  —  resuming from step {step}")
        print(f"  Checkpoint: {checkpoint}")
    else:
        print("  GRPO Reinforcement Learning")
        print("  Reward: eq. 10 — U(o,e|s) = wR·R + wV·V − wC·C − wP·P − μ·Φ")
    print("=" * 64)

    config = GRPOConfig(
        output_dir=OUTPUT_DIR,

        # Optimiser
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        # num_train_epochs raised 8 → 12 for run 4.
        # The 50/50 escalation interleave truncates the dataset to
        # ~786 samples (minority esc-warranted pool × 2), down from
        # ~1090 in run 3.  12 epochs restores the total gradient steps
        # to ~9300, matching run 3 for a fair comparison:
        #   786 / 16 (eff. batch) × 12 epochs ≈ 589 grad steps × 10
        #   logging_steps → ~5890 log steps ≈ run 3's 9000.
        #   (exact count depends on trainer internal expansion)
        num_train_epochs=12,

        # LR schedule
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",

        # Checkpoint saving
        # Checkpoint saving strategy
        # ─────────────────────────────────────────────────────────────
        # Actual run 2 measured ~9300 total gradient steps
        # (933 log entries × logging_steps=10).
        #
        # save_steps=400 (default) gives ~23 evenly-spaced checkpoints
        # across the full training horizon — enough to plot the evolution
        # of all four metric families without clustering at the end.
        #
        # save_total_limit=30 retains ALL checkpoints.
        # Previously limit=25 with save_steps=20 kept only the last
        # 25 of ~466 checkpoints (all clustered in the final epoch),
        # making metric evolution invisible across the training horizon.
        # With save_steps=400 and limit=30 every checkpoint is kept.
        #
        # Each LoRA adapter checkpoint is ~20 MB.
        # 30 checkpoints ≈ 600 MB total disk usage.
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=30,

        # Resumption
        resume_from_checkpoint=checkpoint,

        # GRPO-specific
        # num_generations=16: doubled from run 2 (was 8) to counteract
        # zero-std collapse.  More completions per prompt increases the
        # probability that at least two distinct outputs appear in a batch,
        # giving GRPO a non-zero gradient even when the policy has converged
        # toward a preferred mode.  Justified by zero-std mean=81.4% in run 2.
        num_generations=16,
        beta=0.03,
        loss_type="grpo",

        # Generation
        max_completion_length=40,
        # temperature=1.0: raised from 0.8 (run 2) to increase output
        # diversity and further reduce zero-std collapse.  The model has
        # already converged to a stable mode so there is headroom to
        # push diversity without introducing instability.
        temperature=1.0,
        top_k=50,
        top_p=0.95,

        # Logging
        logging_steps=10,
        report_to="none",
        bf16=False,
        fp16=False,
    )

    trainer = GRPOTrainer(
        model=SFT_MODEL,
        args=config,
        train_dataset=dataset,
        reward_funcs=ue_reward_fn,
        peft_config=LORA_CONFIG,
        processing_class=tokenizer,
    )

    # ── Live checkpoint watcher ──────────────────────────────────────────────
    # Start the background watcher only when --live-eval is requested.
    # It evaluates each new checkpoint as it appears and appends results
    # to grpo_checkpoint_eval.json so the monitor can track metric
    # evolution during the training run.
    watcher = None
    if _live_eval_interval is not None and _live_eval_interval > 0:
        _ev, _test_sets, _sft_mnu = _build_eval_state(tokenizer)
        if _ev is not None:
            watcher = LiveCheckpointWatcher(
                tokenizer    = tokenizer,
                ev           = _ev,
                test_sets    = _test_sets,
                sft_mnu      = _sft_mnu,
                interval     = _live_eval_interval,
                quiet        = False,
            )
            watcher.start()

    # Custom metrics callback: logs reward_logic (U component only)
    # to trainer.state.log_history at each logging step.
    # GRPOTrainer does not natively log custom reward components, so we
    # hook into on_log to inject them from the _last_U_values buffer.
    import numpy as _np
    from transformers import TrainerCallback

    class RewardLogicCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None and _last_U_values:
                logs["reward_logic"]     = float(_np.mean(_last_U_values))
                logs["reward_logic_std"] = float(_np.std(_last_U_values))

    trainer.add_callback(RewardLogicCallback())

    trainer.train()

    if watcher is not None:
        watcher.stop()
        print("  [watcher] Waiting for final eval to finish …")
        watcher.join(timeout=120)

    print_training_summary(trainer.state.log_history)

    # ── Merge final adapter to OUTPUT_DIR ──────────────────────────────────────
    print(f"\n  Merging final adapter → {OUTPUT_DIR}")
    merged = trainer.model.merge_and_unload()
    if hasattr(merged, "peft_config"):
        del merged.peft_config
    if hasattr(merged, "active_adapter"):
        del merged.active_adapter
    merged.save_pretrained(OUTPUT_DIR)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.save_pretrained(OUTPUT_DIR)

    adapter_cfg = os.path.join(OUTPUT_DIR, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        os.remove(adapter_cfg)

    # ── Save training metrics ─────────────────────────────────────────────────
    entries = [
        {
            "step": e.get("step", i),
            **{k: e[k] for k in GRPO_METRIC_KEYS if k in e},
        }
        for i, e in enumerate(trainer.state.log_history)
        if any(k in e for k in GRPO_METRIC_KEYS)
    ]
    save_metrics(entries)
    print(f"  Training metrics → {METRICS_FILE}")

    return trainer


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint evaluation sweep  (all four metric families)
# ═══════════════════════════════════════════════════════════════════════════════

def load_checkpoint_model(ckpt_path, quiet=False):
    """
    Load a GRPO checkpoint for evaluation.

    GRPO checkpoints contain a LoRA adapter (adapter_model.safetensors +
    adapter_config.json).  We must load the base SFT model first, apply
    the adapter with PeftModel, then merge and return a plain
    AutoModelForCausalLM for inference.

    The merged model is kept in memory only — not saved to disk —
    so the checkpoint directory is not modified.
    """
    from peft import PeftModel

    if not quiet:
        step = os.path.basename(ckpt_path).split("-")[-1]
        print(f"    Loading checkpoint-{step} …", end="", flush=True)

    base       = AutoModelForCausalLM.from_pretrained(
        SFT_MODEL, torch_dtype=torch.float32, device_map={"": "cpu"}
    )
    peft_model = PeftModel.from_pretrained(base, ckpt_path)
    merged     = peft_model.merge_and_unload()
    merged.eval()

    if not quiet:
        print(" merged.")

    return merged


def run_checkpoint_eval(merged_model, tokenizer, ev, test_sets):
    """
    Evaluate one merged checkpoint against all four metric families.

    Uses pre-built test sets (built once before the sweep) to avoid
    rebuilding them for every checkpoint.

    Returns a flat dict of all scalar metrics.
    """
    def get_pred(prompt):
        return ev.generate_config(merged_model, tokenizer, prompt)

    result = ev.evaluate_baseline(
        name         = "_ckpt_",
        get_pred_fn  = get_pred,
        id_set       = test_sets["id_set"],
        c1_set       = test_sets["c1_set"],
        c8_set       = test_sets["c8_set"],
        combined_set = test_sets["combined_set"],
        credit_set   = test_sets["credit_set"],
        ood_set      = test_sets["ood_set"],
        mobility_set = test_sets["mob_set"],
        quiet        = True,
    )

    return {
        # Family 1 — constraint safety
        "cvr_c1":        result["cvr_c1"],
        "cvr_c8":        result["cvr_c8"],
        "cvr_credit":    result["cvr_credit"],
        "cvr_combined":  result["cvr_combined"],
        # Family 2 — offload quality
        "cor":           result["cor"],
        "offload_prec":  result["offload_prec"],
        "offload_rec":   result["offload_rec"],
        "offload_f1":    result["offload_f1"],
        # Family 3 — escalation quality
        "ver":           result["ver"],
        "er":            result["er"],
        "uer":           result["uer"],
        "esc_f1":        result["esc_f1"],
        "mer_ho":        result["mer_ho"],
        # Family 4 — utility
        "mnu":           result["mnu"],
        "ccr":           result["ccr"],
        "parse_fail":    result["parse_fail"],
        # Generalisation
        "id_accuracy":   result["id_accuracy"],
        "ood_accuracy":  result["ood_accuracy"],
        "ood_gap":       result["ood_gap"],
        "ood_by_axis":   result["ood_by_axis"],
    }


def sweep_checkpoints(output_dir, tokenizer, quiet=False):
    """
    Sweep all saved GRPO checkpoints and evaluate each against all four
    metric families defined in evaluate.py.

    Differences from run_sft.py sweep
    ───────────────────────────────────
    GRPO checkpoints save LoRA adapters, not merged weights.  Each
    checkpoint is loaded by applying the adapter on top of the SFT base
    model and merging in memory before evaluation.  This merge is done
    fresh for each checkpoint and the result is discarded afterwards
    to keep memory usage bounded.

    Writes results to grpo_checkpoint_eval.json and prints:
      - One-line per-checkpoint summary (all key metrics)
      - Comparison table vs SFT baseline at the end
      - Best checkpoint per criterion
    """
    checkpoints = list_checkpoints(output_dir)
    if not checkpoints:
        print("  [WARN] No checkpoints found — skipping evaluation sweep.")
        return

    print(f"\n{'='*68}")
    print(f"  GRPO checkpoint evaluation sweep — all four metric families")
    print(f"  {len(checkpoints)} checkpoints in {output_dir}")
    print(f"{'='*68}")

    # ── Import evaluate.py ────────────────────────────────────────────────────
    try:
        import importlib.util
        eval_path = os.path.normpath(os.path.join(SCRIPT_DIR, "evaluate.py"))
        spec      = importlib.util.spec_from_file_location("evaluate", eval_path)
        ev        = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
    except Exception as e:
        print(f"  [WARN] Could not import evaluate.py: {e}")
        print("         Run scripts/evaluate.py manually after training.")
        return

    # ── Build test sets once ──────────────────────────────────────────────────
    print("\n  Building test sets (once for all checkpoints) …")
    test_sets = {
        "id_set":       ev.build_id_test_set(n=100),
        "c1_set":       ev.build_c1_set(n=60),
        "c8_set":       ev.build_c8_set(n=60),
        "combined_set": ev.build_combined_constraint_set(),
        "credit_set":   ev.build_credit_floor_set(n=50),
        "ood_set":      ev.build_ood_test_set(),
        "mob_set":      ev.build_mobility_set(TRACE_FILE),
    }
    n_total = sum(len(v) for v in test_sets.values())
    print(f"  Total eval samples per checkpoint: {n_total}")

    # ── SFT baseline (for ΔU comparison) ─────────────────────────────────────
    # Load the SFT model once and evaluate it as the baseline reference.
    # This gives the ΔU = MNU_GRPO − MNU_SFT column in the comparison table.
    sft_mnu = None
    try:
        print("\n  Evaluating SFT baseline for ΔU reference …")
        sft_model = AutoModelForCausalLM.from_pretrained(
            SFT_MODEL, torch_dtype=torch.float32, device_map={"": "cpu"}
        )
        sft_model.eval()
        sft_result = ev.evaluate_baseline(
            name="_sft_", get_pred_fn=lambda p: ev.generate_config(sft_model, tokenizer, p),
            id_set=test_sets["id_set"], c1_set=test_sets["c1_set"],
            c8_set=test_sets["c8_set"], combined_set=test_sets["combined_set"],
            credit_set=test_sets["credit_set"], ood_set=test_sets["ood_set"],
            mobility_set=test_sets["mob_set"], quiet=True,
        )
        sft_mnu = sft_result["mnu"]
        print(f"  SFT baseline MNU = {sft_mnu:+.4f}")
        del sft_model
    except Exception as e:
        print(f"  [WARN] SFT baseline eval failed: {e} — ΔU will not be reported.")

    # ── Sweep ─────────────────────────────────────────────────────────────────
    records        = []
    header_printed = False

    for ckpt_path in checkpoints:
        step = int(os.path.basename(ckpt_path).split("-")[-1])

        try:
            merged = load_checkpoint_model(ckpt_path, quiet=quiet)
        except Exception as e:
            print(f"\n  [WARN] Could not load checkpoint-{step}: {e} — skipping.")
            continue

        try:
            metrics = run_checkpoint_eval(merged, tokenizer, ev, test_sets)
        except Exception as e:
            print(f"\n  [WARN] Eval failed at step {step}: {e} — skipping.")
            del merged
            continue

        del merged   # free memory before next checkpoint

        # ΔU vs SFT
        delta_u = (metrics["mnu"] - sft_mnu) if sft_mnu is not None else None

        record = {
            "checkpoint": ckpt_path,
            "step":       step,
            "delta_u":    round(delta_u, 4) if delta_u is not None else None,
            **metrics,
        }
        records.append(record)

        # Print one-line summary
        if not header_printed:
            print(f"\n  {'Step':>6}  {'CVR':>6}  {'ID':>6}  {'OOD':>6}  "
                  f"{'COR':>6}  {'Off-F1':>7}  {'VER':>6}  {'ER':>6}  "
                  f"{'EscF1':>7}  {'MER_HO':>7}  {'CCR':>6}  {'MNU':>8}  {'ΔU':>7}")
            print("  " + "-" * 98)
            header_printed = True

        ccr_s   = f"{metrics['ccr']:.2f}"   if metrics["ccr"]   is not None else "  n/a"
        delu_s  = f"{delta_u:+.3f}"          if delta_u           is not None else "  n/a"
        print(
            f"  {step:>6}  "
            f"{metrics['cvr_combined']:>6.3f}  "
            f"{metrics['id_accuracy']:>6.3f}  "
            f"{metrics['ood_accuracy']:>6.3f}  "
            f"{metrics['cor']:>6.3f}  "
            f"{metrics['offload_f1']:>7.3f}  "
            f"{metrics['ver']:>6.3f}  "
            f"{metrics['er']:>6.3f}  "
            f"{metrics['esc_f1']:>7.3f}  "
            f"{metrics['mer_ho']:>7.3f}  "
            f"{ccr_s:>6}  "
            f"{metrics['mnu']:>+8.4f}  "
            f"{delu_s:>7}"
        )

    if not records:
        print("  No checkpoints evaluated successfully.")
        return

    # ── Write results ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(CKPT_EVAL), exist_ok=True)
    with open(CKPT_EVAL, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)
    print(f"\n  Results → {CKPT_EVAL}")

    # ── Best checkpoint summary ───────────────────────────────────────────────
    safe = [r for r in records if r["cvr_combined"] == 0.0]
    safe_pool = safe if safe else records   # fall back to all if none are safe

    best_safety  = min(records,    key=lambda r: (r["cvr_combined"], -r["id_accuracy"]))
    best_offload = max(safe_pool,  key=lambda r: r["offload_f1"])
    best_esc     = max(safe_pool,  key=lambda r: r["esc_f1"])
    best_mnu     = max(records,    key=lambda r: r["mnu"])
    best_delta   = max(
        [r for r in records if r["delta_u"] is not None],
        key=lambda r: r["delta_u"]
    ) if any(r["delta_u"] is not None for r in records) else best_mnu

    # GRPO targets (from SFT analysis):
    # CVR → ≤2%, Offload F1 → >0.50, ER → >40%, MNU > +0.25
    targets_met = {
        r["step"]: (
            r["cvr_combined"] <= 0.02
            and r["offload_f1"] > 0.50
            and r["er"] > 0.40
            and r["mnu"] > 0.25
        )
        for r in records
    }

    print(f"\n  Best checkpoints:")
    print(f"    Lowest CVR          : step {best_safety['step']:>5}"
          f"  CVR={best_safety['cvr_combined']:.1%}"
          f"  Off-F1={best_safety['offload_f1']:.3f}"
          f"  MNU={best_safety['mnu']:+.4f}")
    print(f"    Best offload F1     : step {best_offload['step']:>5}"
          f"  Off-F1={best_offload['offload_f1']:.3f}"
          f"  Prec={best_offload['offload_prec']:.3f}"
          f"  Rec={best_offload['offload_rec']:.3f}"
          f"  CVR={best_offload['cvr_combined']:.1%}")
    print(f"    Best escalation F1  : step {best_esc['step']:>5}"
          f"  EscF1={best_esc['esc_f1']:.3f}"
          f"  VER={best_esc['ver']:.3f}"
          f"  ER={best_esc['er']:.3f}"
          f"  CVR={best_esc['cvr_combined']:.1%}")
    print(f"    Best MNU            : step {best_mnu['step']:>5}"
          f"  MNU={best_mnu['mnu']:+.4f}"
          f"  CVR={best_mnu['cvr_combined']:.1%}")
    if any(r["delta_u"] is not None for r in records):
        print(f"    Best ΔU (vs SFT)    : step {best_delta['step']:>5}"
              f"  ΔU={best_delta['delta_u']:+.4f}"
              f"  MNU={best_delta['mnu']:+.4f}")

    all_targets = [s for s, met in targets_met.items() if met]
    if all_targets:
        print(f"\n  Steps meeting ALL GRPO targets (CVR≤2%, F1>0.5, ER>40%, MNU>0.25):")
        for s in all_targets:
            r = next(x for x in records if x["step"] == s)
            print(f"    step {s:>5}  CVR={r['cvr_combined']:.1%}"
                  f"  F1={r['offload_f1']:.3f}  ER={r['er']:.3f}  MNU={r['mnu']:+.4f}")
    else:
        print(f"\n  No step meets all GRPO targets yet.")
        print(f"  Closest: step {best_mnu['step']} — increase epochs or check reward scale.")

    print(f"\n  Recommended checkpoint: step {best_safety['step']}"
          f"  (lowest CVR, use for deployment / further fine-tuning)")

    return records


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="GRPO training + checkpoint evaluation for the UE AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  Fresh start:
    python scripts/run_grpo.py

  Auto-resume from latest checkpoint:
    python scripts/run_grpo.py --resume

  Resume from a specific checkpoint:
    python scripts/run_grpo.py --checkpoint models/grpo-ue-agent/checkpoint-400

  Skip checkpoint sweep (run evaluate.py separately later):
    python scripts/run_grpo.py --no-eval

  Only run the checkpoint sweep on existing checkpoints:
    python scripts/run_grpo.py --eval-only

  Save checkpoints every 10 steps (maximum granularity):
    python scripts/run_grpo.py --save-steps 10
        """
    )
    parser.add_argument("--resume",     action="store_true",
                        help="Auto-resume from latest checkpoint.")
    parser.add_argument("--checkpoint", default=None, metavar="PATH",
                        help="Explicit checkpoint path to resume from.")
    parser.add_argument("--no-eval",    action="store_true",
                        help="Skip post-training checkpoint evaluation sweep.")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training; only run the checkpoint sweep.")
    parser.add_argument(
        "--live-eval", action="store_true",
        help=(
            "Evaluate checkpoints in the background as training progresses. "
            "Results are written to grpo_checkpoint_eval.json incrementally "
            "so monitor_grpo.py can plot metric evolution without waiting "
            "for training to finish. Adds ~8–10 min per checkpoint on CPU."
        ))
    parser.add_argument(
        "--eval-interval", type=int, default=300, metavar="SECS",
        help=(
            "How often (seconds) the background watcher polls for new "
            "checkpoints when --live-eval is active (default: 300 = 5 min). "
            "Set lower on fast hardware; keep at 300+ on CPU to avoid "
            "interfering with gradient steps."
        ))
    parser.add_argument("--save-steps", type=int, default=400, metavar="N",
                        help=(
                            "Save a checkpoint every N steps (default: 400). "
                            "With ~9300 total steps and default=400 this gives "
                            "~23 evenly-spaced checkpoints across the full "
                            "training horizon, all retained by save_total_limit=30."
                        ))
    args = parser.parse_args()

    if args.eval_only and args.no_eval:
        parser.error("--eval-only and --no-eval are mutually exclusive.")

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    checkpoint = None
    if args.checkpoint:
        checkpoint = os.path.normpath(args.checkpoint)
        if not os.path.isdir(checkpoint):
            parser.error(f"Checkpoint directory not found: {checkpoint}")
        validate_checkpoint(checkpoint)
        print(f"Resuming from: {checkpoint}")
    elif args.resume:
        checkpoint = find_latest_checkpoint(OUTPUT_DIR)
        if checkpoint is None:
            print(f"[WARN] --resume given but no checkpoints found. Starting fresh.")
        else:
            validate_checkpoint(checkpoint)
            step = os.path.basename(checkpoint).split("-")[-1]
            print(f"Auto-detected latest checkpoint: {checkpoint}  (step {step})")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    # Always load from SFT_MODEL to avoid version mismatches across checkpoints.
    print("\nLoading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(SFT_MODEL)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if not args.eval_only:
        # ── Dataset ───────────────────────────────────────────────────────────
        dataset = load_and_prepare_dataset()

        # ── Train ─────────────────────────────────────────────────────────────
        run_grpo(dataset, tokenizer,
                 checkpoint=checkpoint,
                 save_steps=args.save_steps,
                 _live_eval_interval=(
                     args.eval_interval if args.live_eval else None
                 ))

    # ── Checkpoint evaluation sweep ───────────────────────────────────────────
    if not args.no_eval:
        sweep_checkpoints(OUTPUT_DIR, tokenizer, quiet=False)
    else:
        print("\n[--no-eval] Skipping checkpoint evaluation sweep.")
        print(f"  Run manually: python scripts/run_grpo.py --eval-only")

    print("\nComplete!")
    if not args.no_eval:
        print(f"  Training metrics : {METRICS_FILE}")
        print(f"  Checkpoint eval  : {CKPT_EVAL}")


if __name__ == "__main__":
    main()