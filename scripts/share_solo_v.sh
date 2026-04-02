#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_YENDRI/lcerovaz/parameter-golf
#SBATCH --job-name=pgolf-share-solo-v
#SBATCH --output=./slurm/%x-%j.out
#SBATCH --error=./slurm/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --mem=40G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=IscrC_YENDRI
#SBATCH --cpus-per-task=32

set -euo pipefail

module load cuda/12.2

mkdir -p slurm

set -a; [ -f .env ] && source .env; set +a

export UV_PYTHON="${UV_PYTHON:-$HOME/.local/bin/python3.11}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

export http_proxy="http://login01:${PROXY_PORT}"
export https_proxy="http://login01:${PROXY_PORT}"

# ── Experiment config ──────────────────────────────────────────────────────
export SHARING_RANK_V=128
export SHARING_RANK_O=0
export MAX_WALLCLOCK_SECONDS=500
export VAL_LOSS_EVERY=200
export HYPOTHESIS="SharedC solo V proj rank_v=128: test senza complicazione zero-init"
export RUN_ID="share_solo_v_$(date +%Y%m%d)_job${SLURM_JOB_ID:-local}"

uv sync --python "${UV_PYTHON}" --active --locked

FAST_BASE="/leonardo_scratch/fast/IscrC_YENDRI/parameter-golf/runs"
if [[ -d "/leonardo_scratch/fast" ]]; then
    _RUN_DIR="${FAST_BASE}/${RUN_ID}"
else
    _RUN_DIR="outputs/runs/${RUN_ID}"
fi

srun uv run bash ./tools/launch_run.sh --script train_gpt_sharing.py --gpus 1 --run-id "$RUN_ID"

for ext in out err; do
    src="./slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.${ext}"
    [[ -f "$src" ]] && cp "$src" "${_RUN_DIR}/" 2>/dev/null || true
done
