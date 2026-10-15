#!/bin/bash
#SBATCH --job-name=tp-analyze
#SBATCH --partition=4RTXq
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x-%j.out
# Usage: sbatch slurm/analyze.sbatch RUN_DIR PROFILE   (CPU only, no GPUs requested)
set -euo pipefail
cd /dataset1/zailong/workspace/TurnPrint
source slurm/env.sh
RUN=${1:?run dir}; PROFILE=${2:-full}
srun --ntasks=1 --exact $ANA_PY -m turnprint.analyze --run "$RUN" --profile "$PROFILE" &
srun --ntasks=1 --exact $ANA_PY -m turnprint.analyze --run "$RUN" --profile "$PROFILE" --exclude-truncated &
wait
