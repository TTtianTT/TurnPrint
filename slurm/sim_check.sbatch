#!/bin/bash
#SBATCH --job-name=tp-simcheck
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out
# Usage: sbatch slurm/sim_check.sbatch RUN_DIR
set -euo pipefail
cd /dataset1/zailong/workspace/TurnPrint
source slurm/env.sh
srun --ntasks=1 $VLLM_PY scripts/sim_check.py "${1:?run dir}"
