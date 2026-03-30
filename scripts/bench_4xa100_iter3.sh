#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_YENDRI/lcerovaz/parameter-golf
#SBATCH --job-name=bench-iter3-4xa100
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export HYPOTHESIS="Iter-3 4xA100: ba_plus_cAA only (no XXT). PyTorch X@X.T + Triton fused b*A+c*A@A in Newton-Schulz. Isolates contribution of the second matmul fusion vs full symmetry trick."
export ITERATIONS=2000
export MAX_WALLCLOCK_SECONDS=1200

srun uv run bash ./tools/launch_run.sh --gpus 4 --script train_gpt_iter3.py --local
