#!/bin/bash
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_YENDRI
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/kernel_bench_%j.out
#SBATCH --error=slurm/kernel_bench_%j.err

# Kernel benchmark run: 300 iterations, 1xA100
# Usage: HYPOTHESIS="..." GIT_BRANCH="kernels" sbatch scripts/kernel_bench.sh
#
# Required env vars:
#   HYPOTHESIS - what this run tests
#   GIT_BRANCH - which branch to checkout (default: current)

set -euo pipefail

module load cuda/12.2
cd /leonardo_work/IscrC_YENDRI/lcerovaz/parameter-golf
source .venv/bin/activate

# Checkout requested branch if specified
if [[ -n "${GIT_BRANCH:-}" ]]; then
    git checkout "${GIT_BRANCH}"
fi

HYPOTHESIS="${HYPOTHESIS:-kernel benchmark run}" \
ITERATIONS=300 \
MAX_WALLCLOCK_SECONDS=3600 \
  ./tools/launch_run.sh --gpus 1 --local
