#!/bin/bash
# Usage: scripts/submit.sh RUN_NAME PROFILE [gen|score|analyze|all]
# Chains generation (2 GPUs) -> D2 scoring (2 GPUs) -> analysis (CPU only) with
# afterok dependencies, so at most 2 GPUs are ever held. Refuses if any of my jobs already hold/await GPUs.
set -euo pipefail
cd /dataset1/zailong/workspace/TurnPrint
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/etc/slurm/slurm.conf}
NAME=${1:?run name}; PROFILE=${2:-full}; WHAT=${3:-all}
RUN=$PWD/runs/$NAME
mine=$(squeue -u "$USER" -h -o "%i %T %b" | grep -i gpu || true)
if [[ -n $mine && $WHAT != analyze ]]; then echo "Refusing: GPU jobs already queued/running:"; echo "$mine"; exit 1; fi
mkdir -p "$RUN" logs
dep=()
if [[ $WHAT == gen || $WHAT == all ]]; then
  G=$(sbatch --parsable slurm/generate.sbatch "$RUN" "$PROFILE"); echo "generate job $G"
  dep=(--dependency=afterok:$G)
fi
if [[ $WHAT == score || $WHAT == all ]]; then
  S=$(sbatch --parsable "${dep[@]}" slurm/score.sbatch "$RUN" "$PROFILE"); echo "score job $S"
  dep=(--dependency=afterok:$S)
fi
if [[ $WHAT == analyze || $WHAT == all ]]; then
  A=$(sbatch --parsable "${dep[@]}" slurm/analyze.sbatch "$RUN" "$PROFILE"); echo "analyze job $A"
fi
