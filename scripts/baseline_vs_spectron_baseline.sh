#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_YENDRI/lcerovaz/parameter-golf
#SBATCH --job-name=pgolf-baseline-4xA100
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
# RUN_ID and MAX_WALLCLOCK_SECONDS must NOT use :- fallback here:
# .env sets RUN_ID=baseline_sp1024_a100_20m and MAX_WALLCLOCK_SECONDS=1500,
# so any :- fallback would silently keep those stale values.
export RUN_ID="baseline_4xA100_12min_$(date +%Y%m%d)_job${SLURM_JOB_ID}"
export MAX_WALLCLOCK_SECONDS=720
export VAL_LOSS_EVERY=200
export HYPOTHESIS="Baseline dense GPT on 4xA100, 12-min wallclock — reference for Spectron low-rank comparison"

uv sync --python "${UV_PYTHON}" --active --locked

# ── Resolve run dir for SLURM log copy at end ────────────────────────────
FAST_BASE="/leonardo_scratch/fast/IscrC_YENDRI/parameter-golf/runs"
if [[ -d "/leonardo_scratch/fast" ]]; then
    _RUN_DIR="${FAST_BASE}/${RUN_ID}"
else
    _RUN_DIR="outputs/runs/${RUN_ID}"
fi

# ── Launch via the logging wrapper ────────────────────────────────────────
srun uv run bash ./tools/launch_run.sh --gpus 4 --run-id "$RUN_ID"

# ── Copy SLURM logs into run dir ─────────────────────────────────────────
for ext in out err; do
    src="./slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.${ext}"
    [[ -f "$src" ]] && cp "$src" "${_RUN_DIR}/" 2>/dev/null || true
done
