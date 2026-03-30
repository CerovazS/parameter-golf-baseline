#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_YENDRI/lcerovaz/parameter-golf
#SBATCH --job-name=pgolf-4xA100
#SBATCH --output=./slurm/%x-%j.out
#SBATCH --error=./slurm/%x-%j.err
#SBATCH --time=00:25:00
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

# Prefer a uv-managed Python on compute nodes.
export UV_PYTHON="${UV_PYTHON:-$HOME/.local/bin/python3.11}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

# Squid proxy for internet access on compute nodes
export http_proxy="http://login01:${PROXY_PORT}"
export https_proxy="http://login01:${PROXY_PORT}"

# ── Configuration ─────────────────────────────────────────────────────────
export RUN_ID="${RUN_ID:-baseline_4xA100_$(date +%Y%m%d)_job${SLURM_JOB_ID:-local}}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-1200}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
export HYPOTHESIS="${HYPOTHESIS:-Baseline 4xA100 20min run}"

# Ensure the project environment is built against the intended interpreter.
uv sync --python "${UV_PYTHON}" --active --locked

# ── Resolve run dir for SLURM log copy at end ────────────────────────────
FAST_BASE="/leonardo_scratch/fast/IscrC_YENDRI/parameter-golf/runs"
if [[ -d "/leonardo_scratch/fast" ]]; then
    _RUN_DIR="${FAST_BASE}/${RUN_ID}"
else
    _RUN_DIR="outputs/runs/${RUN_ID}"
fi

# ── Launch via the logging wrapper ────────────────────────────────────────
# launch_run.sh handles: run dir creation, code snapshot, meta.json, finalize
srun uv run bash ./tools/launch_run.sh --gpus 4 --run-id "$RUN_ID"

# ── Copy SLURM logs into run dir ─────────────────────────────────────────
for ext in out err; do
    src="./slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.${ext}"
    [[ -f "$src" ]] && cp "$src" "${_RUN_DIR}/" 2>/dev/null || true
done
