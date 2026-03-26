# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Leonardo HPC Notes

- Compute nodes have no internet — set `http_proxy`/`https_proxy` to squid proxy on login01
- Partition: `boost_usr_prod`, account: `IscrC_TVU`, GPUs: A100 64GB
- SLURM logs go to `slurm/` directory
- Use `module load cuda/12.2` before training
