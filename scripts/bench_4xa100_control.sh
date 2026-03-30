#!/bin/bash
#SBATCH -D /leonardo_work/IscrC_YENDRI/lcerovaz/parameter-golf
#SBATCH --job-name=bench-ctrl-4xa100
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
export HYPOTHESIS="Control 4xA100: main branch code, torch.compile on Newton-Schulz, no Triton kernels. Fair baseline for 4xA100 kernel comparison."
export ITERATIONS=2000
export MAX_WALLCLOCK_SECONDS=1200

srun uv run bash ./tools/launch_run.sh --gpus 4 --script train_gpt_main_ctrl.py --local
