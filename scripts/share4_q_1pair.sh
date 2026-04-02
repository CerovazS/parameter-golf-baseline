#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_YENDRI/lcerovaz/parameter-golf
#SBATCH --job-name=pgolf-share4-q-1pair
#SBATCH --output=./slurm/%x-%j.out
#SBATCH --error=./slurm/%x-%j.err
#SBATCH --time=00:25:00
#SBATCH --ntasks=1
#SBATCH --mem=160G
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:4
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

export MAX_WALLCLOCK_SECONDS=700
export VAL_LOSS_EVERY=200
export SHARING_RANK_Q=256
export SHARING_NUM_PAIRS=1
export RUN_ID="share4_q_1pair_$(date +%Y%m%d)_job${SLURM_JOB_ID:-local}"

uv sync --python "${UV_PYTHON}" --active --locked

FAST_BASE="/leonardo_scratch/fast/IscrC_YENDRI/parameter-golf/runs"
_RUN_DIR="$( [[ -d /leonardo_scratch/fast ]] && echo "${FAST_BASE}/${RUN_ID}" || echo "outputs/runs/${RUN_ID}" )"

srun uv run bash ./tools/launch_run.sh \
    --script train_gpt_sharing.py \
    --gpus 4 \
    --run-id "$RUN_ID" \
    --hypothesis "4xA100 700s | Q rq=256 pairs=1 — Q projection sharing baseline"

for ext in out err; do
    src="./slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.${ext}"
    [[ -f "$src" ]] && cp "$src" "${_RUN_DIR}/" 2>/dev/null || true
done
