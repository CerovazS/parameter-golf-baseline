#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/parameter-golf
#SBATCH --job-name=pgolf-train
#SBATCH --output=./slurm/%x-%j.out
#SBATCH --error=./slurm/%x-%j.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_TVU

set -euo pipefail

module load cuda/12.2

mkdir -p slurm results

# Load .env if present (must come before proxy config)
set -a; [ -f .env ] && source .env; set +a

# Squid proxy for internet access on compute nodes
export http_proxy="http://login01:${PROXY_PORT}"
export https_proxy="http://login01:${PROXY_PORT}"

# ---- Configuration ----
export RUN_ID="${RUN_ID:-baseline_sp1024}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"

# Override wallclock for single-GPU (slower than 8xH100, so give more time)
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-3600}"

# Periodic validation logging
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"

# ---- Launch training ----
srun uv run torchrun --standalone --nproc_per_node=1 train_gpt.py
