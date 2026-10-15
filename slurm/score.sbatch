#!/bin/bash
#SBATCH --job-name=tp-score
#SBATCH --partition=B300q
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
# Usage: sbatch slurm/score.sbatch RUN_DIR PROFILE
# D2 scoring: two parallel 1-GPU steps, each scoring two candidates in turn
# (balanced by size). Analysis runs separately on CPU nodes (slurm/analyze.sbatch).
set -euo pipefail
cd /dataset1/zailong/workspace/TurnPrint
source slurm/env.sh
RUN=${1:?run dir}; PROFILE=${2:-full}
read -r -a MODELS <<< "$($ANA_PY -c "from turnprint.common import load_config; print(' '.join(m['name'] for m in sorted(load_config(profile='$PROFILE')['models'], key=lambda m: -int(m['name'].split('-')[-1][:-1]))))")"
# largest + smallest on one GPU, the middle two on the other
LANE0=("${MODELS[0]}" "${MODELS[-1]}"); LANE1=("${MODELS[@]:1:${#MODELS[@]}-2}")
lane() { for M in "$@"; do $VLLM_PY -m turnprint.score --run "$RUN" --model "$M" --profile "$PROFILE"; done; }
export -f lane; export RUN PROFILE
srun --ntasks=1 --gres=gpu:1 --exact bash -c "lane ${LANE0[*]}" > "logs/tp-score-$SLURM_JOB_ID-lane0.log" 2>&1 &
P0=$!
srun --ntasks=1 --gres=gpu:1 --exact bash -c "lane ${LANE1[*]}" > "logs/tp-score-$SLURM_JOB_ID-lane1.log" 2>&1 &
P1=$!
wait $P0; wait $P1
