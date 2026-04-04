#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_YENDRI/lcerovaz/parameter-golf
#SBATCH --job-name=pgolf-sweep
#SBATCH --output=./slurm/%x-%j.out
#SBATCH --error=./slurm/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
#SBATCH --account=IscrC_YENDRI
#SBATCH --cpus-per-task=32

set -euo pipefail

module load cuda/12.2

mkdir -p slurm

# Load .env if present (must come before proxy config)
set -a; [ -f .env ] && source .env; set +a

export UV_PYTHON="${UV_PYTHON:-$HOME/.local/bin/python3.11}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

# Squid proxy for internet access on compute nodes
export http_proxy="http://login01:${PROXY_PORT}"
export https_proxy="http://login01:${PROXY_PORT}"

# ── Configuration — set AFTER .env so these always override it ────────────
# Architecture params (safe with :- since .env doesn't set them)
export MODEL_DIM="${MODEL_DIM:-512}"
export NUM_LAYERS="${NUM_LAYERS:-9}"
export NUM_HEADS="${NUM_HEADS:-8}"
export NUM_KV_HEADS="${NUM_KV_HEADS:-4}"
export MLP_MULT="${MLP_MULT:-2}"
export RANK_RATIO="${RANK_RATIO:-0.30}"
export MATRIX_LR="${MATRIX_LR:-0.08}"
# Per-layer overrides (comma-separated, empty = use global default)
export PER_LAYER_NUM_HEADS="${PER_LAYER_NUM_HEADS:-}"
export PER_LAYER_NUM_KV_HEADS="${PER_LAYER_NUM_KV_HEADS:-}"
export PER_LAYER_RANK_RATIO="${PER_LAYER_RANK_RATIO:-}"

# Build descriptive tag from config
TAG="${EXP_TAG:-d${MODEL_DIM}_L${NUM_LAYERS}_nh${NUM_HEADS}_nkv${NUM_KV_HEADS}_r${RANK_RATIO//./}}"

# RUN_ID and MAX_WALLCLOCK_SECONDS: unconditional (no :- fallback)
export RUN_ID="sweep_${TAG}_lr${MATRIX_LR//./}_$(date +%Y%m%d)_job${SLURM_JOB_ID}"
export MAX_WALLCLOCK_SECONDS=720
export VAL_LOSS_EVERY=200
export HYPOTHESIS="${HYPOTHESIS:-Arch sweep: ${TAG}, lr=${MATRIX_LR}, 4xA100 12min}"

uv sync --python "${UV_PYTHON}" --active --locked

# ── Resolve run dir for SLURM log copy at end ────────────────────────────
FAST_BASE="/leonardo_scratch/fast/IscrC_YENDRI/parameter-golf/runs"
if [[ -d "/leonardo_scratch/fast" ]]; then
    _RUN_DIR="${FAST_BASE}/${RUN_ID}"
else
    _RUN_DIR="outputs/runs/${RUN_ID}"
fi

# ── Launch via the logging wrapper ────────────────────────────────────────
srun uv run bash ./tools/launch_run.sh --gpus 4 --run-id "$RUN_ID" \
    --script train_gpt_low_rank.py

# ── Copy SLURM logs into run dir ─────────────────────────────────────────
for ext in out err; do
    src="./slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.${ext}"
    [[ -f "$src" ]] && cp "$src" "${_RUN_DIR}/" 2>/dev/null || true
done
