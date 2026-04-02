#!/usr/bin/env python3
"""
Metadata manager for Parameter Golf experiment runs.

Writes and updates ``meta.json`` files that capture run configuration,
hardware info, and final results.  Runs entirely *outside* the training
loop — zero overhead by design.  Called by ``launch_run.sh`` before and
after training, or from the CLI.

CLI
---
    python tools/golf_meta.py init     --run-dir <path> [--hypothesis "..."]
    python tools/golf_meta.py finalize --run-dir <path>

Python
------
    from tools.golf_meta import init_meta, finalize_meta
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Hyperparameter spec — mirrors the env-var interface of train_gpt.py.
# Maps env var name -> (type_converter, default_value).
# ---------------------------------------------------------------------------
HYPERPARAM_SPEC: dict[str, tuple[type, object]] = {
    # Data
    "DATA_PATH": (str, "./data/datasets/fineweb10B_sp1024"),
    "TOKENIZER_PATH": (str, "./data/tokenizers/fineweb_1024_bpe.model"),
    "SEED": (int, 1337),
    # Training length
    "ITERATIONS": (int, 20000),
    "WARMDOWN_ITERS": (int, 1200),
    "WARMUP_STEPS": (int, 20),
    "TRAIN_BATCH_TOKENS": (int, 524_288),
    "TRAIN_SEQ_LEN": (int, 1024),
    "MAX_WALLCLOCK_SECONDS": (float, 600.0),
    # Logging cadence
    "VAL_BATCH_SIZE": (int, 524_288),
    "VAL_LOSS_EVERY": (int, 1000),
    "TRAIN_LOG_EVERY": (int, 200),
    # Model shape
    "VOCAB_SIZE": (int, 1024),
    "NUM_LAYERS": (int, 9),
    "NUM_KV_HEADS": (int, 4),
    "MODEL_DIM": (int, 512),
    "NUM_HEADS": (int, 8),
    "MLP_MULT": (int, 2),
    "TIE_EMBEDDINGS": (int, 1),
    "ROPE_BASE": (float, 10000.0),
    "LOGIT_SOFTCAP": (float, 30.0),
    "QK_GAIN_INIT": (float, 1.5),
    # Optimizer
    "EMBED_LR": (float, 0.6),
    "HEAD_LR": (float, 0.008),
    "TIED_EMBED_LR": (float, 0.05),
    "TIED_EMBED_INIT_STD": (float, 0.005),
    "MATRIX_LR": (float, 0.04),
    "SCALAR_LR": (float, 0.04),
    "MUON_MOMENTUM": (float, 0.95),
    "MUON_BACKEND_STEPS": (int, 5),
    "MUON_MOMENTUM_WARMUP_START": (float, 0.85),
    "MUON_MOMENTUM_WARMUP_STEPS": (int, 500),
    "BETA1": (float, 0.9),
    "BETA2": (float, 0.95),
    "ADAM_EPS": (float, 1e-8),
    "GRAD_CLIP_NORM": (float, 0.0),
    # Weight-sharing (train_gpt_sharing.py)
    "SHARING_RANK_V": (int, 0),
    "SHARING_RANK_O": (int, 0),
    "SHARING_RANK_Q": (int, 0),
    "SHARING_RANK_MLP_FC": (int, 0),
    "SHARING_RANK_MLP_PROJ": (int, 0),
    "SHARING_NUM_PAIRS": (int, 1),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_cmd(cmd: list[str]) -> str:
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return ""


def _get_gpu_info() -> dict:
    """GPU info from nvidia-smi (no torch import)."""
    try:
        raw = _run_cmd([
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        if lines:
            parts = lines[0].split(", ")
            return {
                "name": parts[0],
                "memory_mib": int(float(parts[1])) if len(parts) > 1 else None,
                "count": len(lines),
            }
    except Exception:
        pass
    return {"name": "unknown", "memory_mib": None, "count": 0}


def _get_git_info() -> dict:
    info: dict[str, object] = {}
    info["commit"] = _run_cmd(["git", "rev-parse", "--short", "HEAD"])
    info["branch"] = _run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    try:
        info["dirty"] = bool(subprocess.run(
            ["git", "diff", "--quiet"], capture_output=True, timeout=5,
        ).returncode)
    except Exception:
        info["dirty"] = None
    return info


def _get_slurm_info() -> dict | None:
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return None
    return {
        "job_id": job_id,
        "partition": os.environ.get("SLURM_JOB_PARTITION"),
        "node": os.environ.get("SLURM_NODELIST"),
        "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        "gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
    }


def _collect_hyperparameters() -> tuple[dict, dict]:
    """Return (all_params, overrides_only)."""
    params: dict[str, object] = {}
    overrides: dict[str, dict] = {}
    for name, (conv, default) in HYPERPARAM_SPEC.items():
        env_val = os.environ.get(name)
        if env_val is not None:
            val = conv(env_val)
            params[name] = val
            if default is not None and val != default:
                overrides[name] = {"default": default, "value": val}
        elif default is not None:
            params[name] = default
    return params, overrides


def _parse_jsonl(run_dir: Path) -> list[dict]:
    """Read all JSONL files in a run directory."""
    events: list[dict] = []
    for jf in sorted(run_dir.glob("*.jsonl")):
        for line in jf.read_text().strip().split("\n"):
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def _extract_results(events: list[dict], run_dir: Path) -> dict:
    """Derive result metrics from parsed JSONL events."""
    train_steps = [e for e in events if e.get("event") == "train_step"]
    val_steps = [e for e in events if e.get("event") == "val_step"]
    final_rt = [e for e in events if e.get("event") == "final_int8_zlib_roundtrip"]

    results: dict[str, object] = {}

    if val_steps:
        last = val_steps[-1]
        results["final_val_bpb"] = last.get("val_bpb")
        results["final_val_loss"] = last.get("val_loss")

    if final_rt:
        rt = final_rt[-1]
        results["final_int8_bpb"] = rt.get("val_bpb")
        results["final_int8_loss"] = rt.get("val_loss")
        results["eval_time_ms"] = rt.get("eval_time_ms")

    if train_steps:
        results["total_steps"] = train_steps[-1].get("step", 0)
        results["total_train_time_ms"] = train_steps[-1].get("train_time_ms", 0)
        tok_vals = [e["tok_s"] for e in train_steps if e.get("tok_s")]
        if tok_vals:
            results["mean_tok_s"] = round(sum(tok_vals) / len(tok_vals), 1)
            results["peak_tok_s"] = round(max(tok_vals), 1)
        step_ms = [e["step_time_ms"] for e in train_steps if e.get("step_time_ms")]
        if step_ms:
            results["mean_step_ms"] = round(sum(step_ms) / len(step_ms), 2)

    # Artifact sizes
    for ptz in run_dir.glob("*.int8.ptz"):
        results["artifact_bytes"] = ptz.stat().st_size
    for pt in run_dir.glob("final_model.pt"):
        results["model_bytes"] = pt.stat().st_size

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_meta(
    run_dir: str | Path,
    hypothesis: str = "",
    train_script: str = "train_gpt.py",
    baseline_ref: str | None = None,
) -> dict:
    """Write initial meta.json before training starts."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    params, overrides = _collect_hyperparameters()

    meta = {
        "run_id": os.environ.get("RUN_ID", run_dir.name),
        "hypothesis": hypothesis,
        "train_script": train_script,
        "baseline_ref": baseline_ref,
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "status": "running",
        "gpu": _get_gpu_info(),
        "slurm": _get_slurm_info(),
        "git": _get_git_info(),
        "hyperparameters": params,
        "env_overrides": overrides,
        "results": None,
    }

    meta_path = run_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def finalize_meta(run_dir: str | Path) -> dict:
    """Update meta.json with results extracted from the JSONL log."""
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {"run_id": run_dir.name, "status": "unknown"}

    events = _parse_jsonl(run_dir)
    if not events:
        meta["status"] = "failed"
        meta["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        return meta

    meta["results"] = _extract_results(events, run_dir)
    final_rt = [e for e in events if e.get("event") == "final_int8_zlib_roundtrip"]
    meta["status"] = "completed" if final_rt else "incomplete"
    meta["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")

    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter Golf run metadata")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Write initial meta.json")
    p_init.add_argument("--run-dir", required=True)
    p_init.add_argument("--hypothesis", default="")
    p_init.add_argument("--train-script", default="train_gpt.py")
    p_init.add_argument("--baseline-ref", default=None)

    p_fin = sub.add_parser("finalize", help="Update meta.json with results")
    p_fin.add_argument("--run-dir", required=True)

    args = parser.parse_args()

    if args.command == "init":
        meta = init_meta(
            args.run_dir, args.hypothesis, args.train_script, args.baseline_ref,
        )
        print(json.dumps(meta, indent=2))
    elif args.command == "finalize":
        meta = finalize_meta(args.run_dir)
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
