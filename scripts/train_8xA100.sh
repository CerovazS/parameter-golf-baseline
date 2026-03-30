#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_YENDRI/lcerovaz/parameter-golf
#SBATCH --job-name=pgolf-8xA100
#SBATCH --output=./slurm/%x-%j.out
#SBATCH --error=./slurm/%x-%j.err
#SBATCH --time=00:25:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
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

# ── Configuration ─────────────────────────────────────────────────────────
SCRIPT="train_gpt.py"
export RUN_ID="${RUN_ID:-baseline_8xA100_$(date +%Y%m%d)_job${SLURM_JOB_ID:-local}}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
export VOCAB_SIZE="${VOCAB_SIZE:-1024}"
export MAX_WALLCLOCK_SECONDS="${MAX_WALLCLOCK_SECONDS:-600}"
export VAL_LOSS_EVERY="${VAL_LOSS_EVERY:-200}"
export HYPOTHESIS="${HYPOTHESIS:-Baseline 8xA100 10min canonical challenge run}"

# ── Setup run directory with code snapshot and metadata ───────────────────
PROJECT_DIR="/leonardo_work/IscrC_YENDRI/lcerovaz/parameter-golf"
FAST_BASE="/leonardo_scratch/fast/IscrC_YENDRI/parameter-golf/runs"

if [[ -d "/leonardo_scratch/fast" ]]; then
    RUN_DIR="${FAST_BASE}/${RUN_ID}"
    mkdir -p "$RUN_DIR"
    mkdir -p "${PROJECT_DIR}/outputs/runs"
    ln -sfn "$RUN_DIR" "${PROJECT_DIR}/outputs/runs/${RUN_ID}"
else
    RUN_DIR="${PROJECT_DIR}/outputs/runs/${RUN_ID}"
    mkdir -p "$RUN_DIR"
fi
export OUT_DIR="$(dirname "$RUN_DIR")"

# Code snapshot
CODE_DIR="${RUN_DIR}/code"
mkdir -p "$CODE_DIR"
cp "${PROJECT_DIR}/${SCRIPT}" "$CODE_DIR/"
[[ -f "${PROJECT_DIR}/triton_kernels.py" ]] && cp "${PROJECT_DIR}/triton_kernels.py" "$CODE_DIR/"
cp "$0" "$CODE_DIR/"
cp "${PROJECT_DIR}/tools/golf_meta.py" "$CODE_DIR/"

uv sync --python "${UV_PYTHON}" --active --locked

# Write initial metadata
uv run python3 tools/golf_meta.py init \
    --run-dir "$RUN_DIR" \
    --hypothesis "${HYPOTHESIS}" \
    --train-script "$SCRIPT" \
    > /dev/null

# ── Tee output into run dir ───────────────────────────────────────────────
exec > >(tee -a "${RUN_DIR}/output.log") 2> >(tee -a "${RUN_DIR}/error.log" >&2)

# ── Multi-node launch ────────────────────────────────────────────────────
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500

TRAIN_EXIT=0
srun uv run torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    "$SCRIPT" || TRAIN_EXIT=$?

# ── Finalize metadata ────────────────────────────────────────────────────
uv run python3 tools/golf_meta.py finalize --run-dir "$RUN_DIR" > /dev/null

if [[ $TRAIN_EXIT -ne 0 ]]; then
    uv run python3 -c "
import json, pathlib
p = pathlib.Path('${RUN_DIR}/meta.json')
if p.exists():
    m = json.loads(p.read_text()); m['status'] = 'failed'
    p.write_text(json.dumps(m, indent=2) + '\n')
"
    echo "Training exited with code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

# ── Copy SLURM logs into run dir ─────────────────────────────────────────
for ext in out err; do
    src="./slurm/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.${ext}"
    [[ -f "$src" ]] && cp "$src" "${RUN_DIR}/" 2>/dev/null || true
done

echo ""
echo "Run complete: ${RUN_ID}"
echo "Dashboard: uv run python3 tools/dashboard.py"
