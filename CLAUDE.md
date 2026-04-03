# CLAUDE.md

This file provides project-specific guidance to Claude Code when working in this repository.

It overrides the global `~/.claude/CLAUDE.md` with instructions specific to the Parameter Golf challenge.

## What This Is

**Parameter Golf** is an OpenAI challenge to train the best language model that fits in a **16MB compressed artifact** and trains in **under 10 minutes on 8×H100 SXM GPUs**. Evaluation metric is **bits-per-byte (BPB)** on a frozen FineWeb validation set — tokenizer-agnostic compression quality.

## Key Commands

```bash
# Download dataset (sp1024 tokenizer, 80 train shards = 8B tokens)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80

# Smoke test with fewer shards
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# Train on 8×H100 (canonical challenge setup)
torchrun --standalone --nproc_per_node=8 train_gpt.py

# Train on 1 GPU with env overrides
RUN_ID=test ITERATIONS=200 MAX_WALLCLOCK_SECONDS=3600 torchrun --standalone --nproc_per_node=1 train_gpt.py

# Train on macOS Apple Silicon (MLX backend)
RUN_ID=test ITERATIONS=200 python3 train_gpt_mlx.py

# Submit to Leonardo SLURM cluster (1×A100)
sbatch scripts/train_1xA100.sh
```

## Architecture Overview

The codebase is intentionally flat — two self-contained training scripts plus data tooling:

- **`train_gpt.py`** — CUDA/PyTorch training script (~1100 lines, hard limit <1500). Contains the full model, optimizer, data loading, training loop, quantization, and evaluation in one file. All hyperparameters are configured via environment variables through the `Hyperparameters` class.
- **`train_gpt_mlx.py`** — Equivalent script for Apple Silicon using MLX instead of PyTorch.
- **`data/`** — Dataset download (`cached_challenge_fineweb.py`), custom tokenization (`download_hf_docs_and_tokenize.py`), tokenizer specs.
- **`records/`** — Past submissions organized into `track_10min_16mb/` (record-beating) and `track_non_record_16mb/` (non-record/unlimited compute). Each has its own `train_gpt.py`, `README.md`, and `submission.json`.

### Model Architecture (in train_gpt.py)

Transformer-based GPT with: RMSNorm, Group Query Attention (GQA), RoPE, relu² MLP, optional tied embeddings, tanh logit softcap.

### Optimizer Design

Per-parameter-type learning rates with two optimizers:
- **Muon** (custom Newton-Schulz orthogonalization) for matrix/weight parameters
- **Adam** for embeddings, LM head, and scalar/control parameters

### Quantization Pipeline

Post-training: float → int8 (per-row for matrices, per-tensor for vectors) → zlib level 9 compression. Small tensors (<65K elements) stay in float16. The compressed `.int8.ptz` artifact must be ≤16MB including all code bytes.

### Data Format

Binary shards (`.bin`): 256-int32 header (magic, version, num_tokens) followed by uint16 token values. Training reads shards sequentially via `TokenStream`/`DistributedTokenLoader`.

## Hard Constraints

- **16MB artifact limit** (16,000,000 bytes decimal) = compressed model + code bytes
- **10-minute training wallclock** on 8×H100 SXM (enforced by `MAX_WALLCLOCK_SECONDS=600`)
- **10-minute evaluation cap** additional
- **<1500 lines** for train_gpt.py / train_gpt_mlx.py
- No external downloads during evaluation; no validation data access during training
- Record submissions must beat SOTA by ≥0.005 nats with p<0.01 (3 seeds recommended)

## Leonardo HPC Workflow

### Interactive Development (PREFERRED for this project)

**When developing, debugging, or experimenting with Parameter Golf:**

1. **Start an interactive GPU session** instead of submitting batch jobs:
   ```bash
   srun -p boost_usr_prod -A IscrC_TVU --gres=gpu:1 --mem=40G --time=2:00:00 --pty bash
   ```

2. **Load CUDA**:
   ```bash
   module load cuda/12.2
   ```

3. **Set up internet access** (ask for the port):
   ```bash
   export http_proxy='http://login01:<PORT>'
   export https_proxy='http://login01:<PORT>'
   ```

4. **Navigate to project**:
   ```bash
   cd /leonardo_work/IscrC_YENDRI/lcerovaz/parameter-golf
   ```

5. **Run experiments interactively**:
   ```bash
   # Single GPU test
   RUN_ID=test ITERATIONS=200 MAX_WALLCLOCK_SECONDS=3600 torchrun --standalone --nproc_per_node=1 train_gpt.py
   
   # Multi-GPU if available
   torchrun --standalone --nproc_per_node=2 train_gpt.py
   ```

6. **Monitor GPU usage** (from another terminal):
   ```bash
   squeue -u $USER  # Get job ID
   srun --jobid <JOB_ID> --overlap --pty bash -lc 'watch -n 1 nvidia-smi'
   ```

### When to Use Batch Jobs

**Only use `sbatch` for:**
- Final production runs for submission
- Long training runs (>2 hours) after validation
- Multiple parallel experiments (hyperparameter sweeps)

### Development Workflow

1. **Implement changes** to `train_gpt.py` locally or via interactive session
2. **Test in interactive mode** with small iterations (e.g., `ITERATIONS=200`)
3. **Validate convergence** and check artifact size
4. **Only after validation**, submit batch job for full 10-minute run:
   ```bash
   sbatch scripts/train_1xA100.sh
   ```

### Cluster Configuration

- **Partition**: `boost_usr_prod`
- **Account**: `IscrC_TVU`
- **GPUs**: A100 64GB (typically request 1, up to 4 available per node)
- **SLURM logs**: `slurm/` directory
- **Internet**: Compute nodes require squid proxy on login01
- **Max walltime**: 24 hours (but challenge limit is 10 minutes + 10 minutes eval)

### Key Environment Variables for Parameter Golf

```bash
RUN_ID=<experiment_name>           # Default: generated timestamp
ITERATIONS=<num_iters>              # Default: defined in Hyperparameters class
MAX_WALLCLOCK_SECONDS=600          # Challenge limit: 600 seconds (10 min)
```

See `train_gpt.py` `Hyperparameters` class for all configurable environment variables.

## Experiment Logging

See `LOGGING.md` for full documentation.

### Default Launcher: `tools/launch_run.sh`

**Every experiment — interactive or batch — MUST go through `tools/launch_run.sh`.**
It handles run directory creation, code snapshot, `meta.json` init/finalize, and
output tee. Never call `torchrun` directly for a tracked experiment.

```bash
# Interactive (1 GPU, tracked)
HYPOTHESIS="Test higher Muon LR" MATRIX_LR=0.06 \
    ./tools/launch_run.sh --gpus 1

# Custom script (e.g. low-rank variant)
HYPOTHESIS="Spectron lr=0.04" MATRIX_LR=0.04 RANK_RATIO=0.25 \
    ./tools/launch_run.sh --gpus 4 --script train_gpt_low_rank.py

# With explicit run ID
HYPOTHESIS="..." RUN_ID=my_run_name \
    ./tools/launch_run.sh --gpus 1
```

### SLURM Batch Scripts

SLURM scripts in `scripts/` call `launch_run.sh` via `srun`. When writing or
editing a batch script, follow these rules to avoid the `.env` override trap:

**Critical: `.env` sets `RUN_ID` and `MAX_WALLCLOCK_SECONDS`. Always set
experiment-specific vars AFTER sourcing `.env` and WITHOUT `:-` fallback.**

```bash
# WRONG — .env will override these silently
export RUN_ID="${RUN_ID:-my_run}"          # .env already set RUN_ID
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-720}"  # same

# CORRECT — unconditional assignment after source
set -a; [ -f .env ] && source .env; set +a   # picks up DATA_PATH, PROXY_PORT, etc.
export RUN_ID="my_run_$(date +%Y%m%d)_job${SLURM_JOB_ID}"   # always fresh
export MAX_WALLCLOCK_SECONDS=720                               # always 12 min
export HYPOTHESIS="..."                                        # always set
```

Vars safe with `:-` (`.env` does not set them): `MATRIX_LR`, `RANK_RATIO`,
`VOCAB_SIZE`, `VAL_LOSS_EVERY`, `UV_PYTHON`, `UV_LINK_MODE`, thread counts.

**Generate dashboard**:
```bash
uv run python3 tools/dashboard.py
```

The system writes `meta.json` before/after training via `tools/golf_meta.py`,
stores artifacts on `$FAST` scratch with symlinks in `outputs/runs/`, and
produces a self-contained HTML dashboard. No training code modifications needed.
