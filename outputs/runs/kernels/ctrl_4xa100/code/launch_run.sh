#!/bin/bash
# tools/launch_run.sh — Parameter Golf experiment launcher
#
# Creates a run directory (on $FAST or locally), writes meta.json,
# launches training, and finalizes metadata.  Zero training overhead.
#
# Usage:
#   HYPOTHESIS="Test higher Muon LR" ./tools/launch_run.sh
#   HYPOTHESIS="..." MATRIX_LR=0.06 ./tools/launch_run.sh --gpus 4
#   HYPOTHESIS="..." ./tools/launch_run.sh --script train_mario.py
#   HYPOTHESIS="..." ./tools/launch_run.sh --local   # skip $FAST scratch
#
# Options:
#   --script <file>     Training script (default: train_gpt.py)
#   --gpus <N>          Number of GPUs (default: 1)
#   --run-id <name>     Explicit run ID (default: auto timestamp)
#   --baseline <id>     Baseline run ID for comparison
#   --local             Store artifacts in outputs/runs/ instead of $FAST
#
# Environment:
#   HYPOTHESIS          Strongly recommended: why this run exists
#   All train_gpt.py env vars (MATRIX_LR, NUM_LAYERS, etc.) pass through

set -euo pipefail

SCRIPT="train_gpt.py"
GPUS=1
BASELINE_REF=""
LOCAL_MODE=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --script)   SCRIPT="$2"; shift 2;;
        --gpus)     GPUS="$2"; shift 2;;
        --run-id)   RUN_ID="$2"; shift 2;;
        --baseline) BASELINE_REF="$2"; shift 2;;
        --local)    LOCAL_MODE=1; shift;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0;;
        *) echo "Unknown option: $1" >&2; exit 1;;
    esac
done

# ── Generate run ID ──────────────────────────────────────────────────────
if [[ -z "${RUN_ID:-}" ]]; then
    RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
fi
export RUN_ID

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FAST_BASE="/leonardo_scratch/fast/IscrC_YENDRI/parameter-golf/runs"

# ── Create run directory ─────────────────────────────────────────────────
if [[ "$LOCAL_MODE" == "1" ]] || [[ ! -d "/leonardo_scratch/fast" ]]; then
    RUN_DIR="${PROJECT_DIR}/outputs/runs/${RUN_ID}"
    mkdir -p "$RUN_DIR"
else
    RUN_DIR="${FAST_BASE}/${RUN_ID}"
    mkdir -p "$RUN_DIR"
    mkdir -p "${PROJECT_DIR}/outputs/runs"
    ln -sfn "$RUN_DIR" "${PROJECT_DIR}/outputs/runs/${RUN_ID}"
fi

# OUT_DIR must be the *parent* of RUN_ID (train_gpt.py does OUT_DIR/RUN_ID)
export OUT_DIR="$(dirname "$RUN_DIR")"

echo "╔══════════════════════════════════════════════════════╗"
echo "║  Parameter Golf — ${RUN_ID}"
echo "║  Script:     ${SCRIPT}"
echo "║  GPUs:       ${GPUS}"
echo "║  Run dir:    ${RUN_DIR}"
echo "║  Hypothesis: ${HYPOTHESIS:-<not set>}"
echo "╚══════════════════════════════════════════════════════╝"

# ── Tee all output into the run dir ──────────────────────────────────────
exec > >(tee -a "${RUN_DIR}/output.log") 2> >(tee -a "${RUN_DIR}/error.log" >&2)

# ── Snapshot source code into run dir ────────────────────────────────────
CODE_DIR="${RUN_DIR}/code"
mkdir -p "$CODE_DIR"
cp "${PROJECT_DIR}/${SCRIPT}" "$CODE_DIR/"
# Also copy triton kernels and any other .py imported by the script
for f in triton_kernels.py; do
    [[ -f "${PROJECT_DIR}/$f" ]] && cp "${PROJECT_DIR}/$f" "$CODE_DIR/"
done
# Copy the launch script itself and golf_meta for full reproducibility
cp "$0" "$CODE_DIR/launch_run.sh"
cp "${PROJECT_DIR}/tools/golf_meta.py" "$CODE_DIR/"

# ── Write initial metadata ───────────────────────────────────────────────
python3 "${PROJECT_DIR}/tools/golf_meta.py" init \
    --run-dir "$RUN_DIR" \
    --hypothesis "${HYPOTHESIS:-}" \
    --train-script "$SCRIPT" \
    ${BASELINE_REF:+--baseline-ref "$BASELINE_REF"} \
    > /dev/null

# ── Launch training ──────────────────────────────────────────────────────
TRAIN_EXIT=0
torchrun --standalone --nproc_per_node="$GPUS" "$SCRIPT" || TRAIN_EXIT=$?

# ── Finalize metadata ────────────────────────────────────────────────────
python3 "${PROJECT_DIR}/tools/golf_meta.py" finalize --run-dir "$RUN_DIR" > /dev/null

if [[ $TRAIN_EXIT -ne 0 ]]; then
    python3 -c "
import json, pathlib
p = pathlib.Path('${RUN_DIR}/meta.json')
if p.exists():
    m = json.loads(p.read_text())
    m['status'] = 'failed'
    p.write_text(json.dumps(m, indent=2) + '\n')
"
    echo "Training exited with code $TRAIN_EXIT"
    exit $TRAIN_EXIT
fi

echo ""
echo "Run complete.  Generate dashboard:"
echo "  python3 tools/dashboard.py"
