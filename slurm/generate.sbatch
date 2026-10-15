#!/bin/bash
#SBATCH --job-name=tp-gen
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# Usage: sbatch slurm/generate.sbatch RUN_DIR PROFILE
# GPU1 serves the user simulator for the whole job; GPU0 serves one candidate
# at a time (clean phase for every model, then branch phase for every B).
set -euo pipefail
cd /dataset1/zailong/workspace/TurnPrint
RUN=${1:?run dir}; PROFILE=${2:-full}
srun --ntasks=1 bash slurm/generate_step.sh "$RUN" "$PROFILE"
