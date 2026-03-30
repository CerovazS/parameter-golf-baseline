# Parameter Golf — Experiment Logging System

Local-first experiment tracking with zero training overhead.
No WandB, no TensorBoard, no external services required.

## Quick Start

```bash
# Launch an experiment
HYPOTHESIS="Test higher Muon LR" MATRIX_LR=0.06 \
  ./tools/launch_run.sh --gpus 1

# Generate dashboard
python3 tools/dashboard.py

# Open in browser (from login node)
firefox outputs/dashboard.html
```

## Architecture

```
tools/
  golf_meta.py      # Writes meta.json before/after training (CLI + library)
  launch_run.sh     # Creates dirs, calls golf_meta, launches torchrun
  dashboard.py      # Reads all runs, generates self-contained HTML

outputs/
  runs/             # Symlinks to scratch (or local dirs with --local)
    run_20260329_143000/  →  $FAST/parameter-golf/runs/run_20260329_143000/
    run_20260329_150000/  →  ...
  baseline/         # Existing baseline runs (auto-discovered by dashboard)
  dashboard.html    # Generated interactive report
```

**Storage**: artifacts live on `$FAST` scratch for I/O performance.
Symlinks in `outputs/runs/` provide seamless local access.
With `--local`, artifacts stay in `outputs/runs/` directly (no scratch).

## Components

### launch_run.sh

Orchestrates the full lifecycle of a run:

1. Generates a timestamped `RUN_ID`
2. Creates run directory on `$FAST` (or locally with `--local`)
3. Creates symlink in `outputs/runs/`
4. Writes `meta.json` via `golf_meta.py init`
5. Launches `torchrun`
6. Finalizes `meta.json` via `golf_meta.py finalize`

```bash
# Basic usage
HYPOTHESIS="Describe why" ./tools/launch_run.sh

# All options
HYPOTHESIS="..." ./tools/launch_run.sh \
  --script train_mario.py \
  --gpus 4 \
  --run-id my_experiment_name \
  --baseline baseline_sp1024_a100_20m \
  --local
```

**Environment variables**: all `train_gpt.py` env vars pass through.
Set them before calling the script:

```bash
HYPOTHESIS="Double model dim" MODEL_DIM=1024 NUM_HEADS=16 \
  ./tools/launch_run.sh --gpus 1
```

### golf_meta.py

Writes and updates `meta.json` — the structured metadata file for each run.
Runs entirely outside the training loop. Zero overhead.

**CLI usage** (called by `launch_run.sh`, rarely needed directly):

```bash
python3 tools/golf_meta.py init     --run-dir <path> --hypothesis "..."
python3 tools/golf_meta.py finalize --run-dir <path>
```

**Retroactive metadata for old runs** (no meta.json yet):

```bash
python3 tools/golf_meta.py finalize --run-dir outputs/baseline/some_old_run/
```

This reads the `.jsonl` log and creates `meta.json` with extracted results.

### dashboard.py

Generates a self-contained interactive HTML dashboard.

```bash
python3 tools/dashboard.py                          # all runs
python3 tools/dashboard.py --runs run_a run_b       # specific subset
python3 tools/dashboard.py --baseline <run_id>      # set explicit baseline
python3 tools/dashboard.py -o reports/compare.html  # custom output path
python3 tools/dashboard.py --scan-dirs /extra/path  # extra scan locations
```

**Auto-discovery**: scans `outputs/runs/`, `outputs/baseline/`, and
`logs/baseline/` for any directory containing `.jsonl` files.

**Dashboard contents**:
- Summary bar (baseline BPB, best BPB)
- Run comparison table with Δ vs baseline (BPB, throughput, time)
- Tabbed interactive charts: Loss, BPB, Throughput, Wall-clock
- Hyperparameter diff table (only parameters that differ)

**Requires**: `plotly` (`uv add plotly`).

## meta.json Schema

```json
{
  "run_id": "run_20260329_143000",
  "hypothesis": "Free-text description of what this run tests",
  "train_script": "train_gpt.py",
  "baseline_ref": "baseline_sp1024_a100_20m",
  "started_at": "2026-03-29T14:30:00",
  "finished_at": "2026-03-29T14:40:12",
  "status": "completed",

  "gpu": {"name": "NVIDIA A100-SXM4-64GB", "memory_mib": 65536, "count": 1},
  "slurm": {"job_id": "38590001", "partition": "boost_usr_prod", "node": "..."},
  "git": {"commit": "d76b1cd", "branch": "main", "dirty": false},

  "hyperparameters": { "...all env-var-configurable params..." },
  "env_overrides": {
    "MATRIX_LR": {"default": 0.04, "value": 0.06}
  },

  "results": {
    "final_val_bpb": 1.2172,
    "final_val_loss": 2.0606,
    "final_int8_bpb": 1.2409,
    "final_int8_loss": 2.0952,
    "total_steps": 13780,
    "total_train_time_ms": 600000,
    "mean_tok_s": 800000,
    "peak_tok_s": 850000,
    "mean_step_ms": 43.5,
    "artifact_bytes": 15000000,
    "model_bytes": 67000000
  }
}
```

## Integrating with a New Training Script

The logging system requires **zero changes** to training scripts.
It works with any script that:

1. Reads `OUT_DIR` and `RUN_ID` env vars for output path
2. Writes `.jsonl` with events matching this schema:

```jsonl
{"event":"train_step","step":1,"train_time_ms":43.2,"step_time_ms":43.2,"train_loss":6.93,"tok_s":803000}
{"event":"val_step","step":1000,"train_time_ms":43200,"val_loss":2.06,"val_bpb":1.22}
{"event":"final_int8_zlib_roundtrip","val_loss":2.09,"val_bpb":1.24,"eval_time_ms":5600}
```

**Required JSONL fields per event type**:

| Event | Required fields |
|-------|----------------|
| `train_step` | `step`, `train_loss` |
| `val_step` | `step`, `val_loss`, `val_bpb` |
| `final_int8_zlib_roundtrip` | `val_loss`, `val_bpb` |

**Optional but recommended fields**:
- `train_time_ms` — cumulative training wall-clock (enables time-axis charts)
- `step_time_ms` — per-step duration (enables throughput charts)
- `tok_s` — tokens per second
- `eval_time_ms` — evaluation duration

**To add meta.json support to a new script** (optional, `launch_run.sh` handles this):

If your new training script uses different env var names, update
`HYPERPARAM_SPEC` in `tools/golf_meta.py` to match. The spec maps env var
names to `(type, default_value)` pairs so the metadata writer can track
which parameters were explicitly overridden vs defaulted.

## Using with SLURM Batch Scripts

Integrate with existing SLURM scripts by replacing the direct `torchrun` call:

```bash
# In your SLURM script, instead of:
#   srun uv run torchrun --standalone --nproc_per_node=1 train_gpt.py

# Use:
export HYPOTHESIS="Production baseline run"
srun uv run bash ./tools/launch_run.sh --gpus 1
```

Or call `golf_meta.py` directly if you want more control:

```bash
# Setup
RUN_DIR="${OUT_DIR}/${RUN_ID}"
python3 tools/golf_meta.py init --run-dir "$RUN_DIR" --hypothesis "..."

# Training (unchanged)
srun uv run torchrun --standalone --nproc_per_node=1 train_gpt.py

# Finalize
python3 tools/golf_meta.py finalize --run-dir "$RUN_DIR"
```

## File Layout per Run

```
<run_dir>/
  meta.json              ← structured metadata (written by golf_meta.py)
  <run_id>.jsonl         ← training metrics (written by train_gpt.py)
  <run_id>.txt           ← human-readable log (written by train_gpt.py)
  final_model.pt         ← full-precision checkpoint
  final_model.int8.ptz   ← compressed submission artifact
  slurm-<jobid>.out      ← SLURM stdout (if batch job)
  slurm-<jobid>.err      ← SLURM stderr (if batch job)
```
