#!/bin/bash
set -euo pipefail
cd /dataset1/zailong/workspace/TurnPrint
source slurm/env.sh
RUN=$1; PROFILE=$2
mkdir -p "$RUN/server_logs"
MODELS=$($ANA_PY -c "from turnprint.common import load_config; print(' '.join(m['name'] for m in load_config(profile='$PROFILE')['models']))")
cfg() { $ANA_PY -c "from turnprint.common import load_config, model_spec; c=load_config(profile='$PROFILE'); print(model_spec(c,'$1')['$2'])"; }
MAXLEN=$($ANA_PY -c "from turnprint.common import load_config; print(load_config()['generation']['max_model_len'])")
BASE=$(( 20000 + (${SLURM_JOB_ID:-0} % 2000) * 2 ))
SIM_PORT=$BASE; CAND_PORT=$((BASE + 1))
echo "GPUs=${GPUS[*]} models=$MODELS ports=$SIM_PORT/$CAND_PORT"

serve() {  # serve GPU PORT NAME PATH MEMFRAC LOG
  # Gemma-3 is served as a multimodal model whose prefix-LM attention needs TRITON_ATTN;
  # FlashInfer (default) JIT-compiles with nvcc, which compute nodes lack.
  local extra=(--attention-backend FLASH_ATTN)
  [[ $4 == *gemma* ]] && extra=(--attention-backend TRITON_ATTN --limit-mm-per-prompt '{"image":0}')
  CUDA_VISIBLE_DEVICES=$1 $VLLM_BIN serve "$4" --served-model-name "$3" --port "$2" \
    --dtype bfloat16 --max-model-len "$MAXLEN" --gpu-memory-utilization "$5" \
    --seed 0 "${extra[@]}" > "$6" 2>&1 &
  SERVE_PID=$!
}
wait_up() {  # wait_up PORT PID
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$1/health" > /dev/null && return 0
    kill -0 "$2" 2>/dev/null || { echo "server on $1 died"; return 1; }
    sleep 5
  done
  echo "server on $1 timed out"; return 1
}
stop() {  # SIGTERM, wait for exit (frees GPU memory), SIGKILL after 120 s
  kill "$1" 2>/dev/null || true
  for _ in $(seq 1 24); do kill -0 "$1" 2>/dev/null || break; sleep 5; done
  kill -9 "$1" 2>/dev/null || true; wait "$1" 2>/dev/null || true; sleep 5
}

serve "${GPUS[1]}" $SIM_PORT simulator "$(cfg llama3.1-8b path)" 0.85 "$RUN/server_logs/simulator.log"; SIM_PID=$SERVE_PID
trap 'kill $SIM_PID 2>/dev/null || true; [[ -n ${CAND_PID:-} ]] && kill $CAND_PID 2>/dev/null || true' EXIT
wait_up $SIM_PORT $SIM_PID

REPLAY=$($ANA_PY -c "from turnprint.common import load_config; print(load_config(profile='$PROFILE')['profile'].get('user_mode', 'reactive'))")
if [[ $REPLAY == replay ]]; then
  echo "=== script $(date -Is)"
  $VLLM_PY -m turnprint.generate --run "$RUN" --profile "$PROFILE" --phase script --model simulator \
    --sim-url http://127.0.0.1:$SIM_PORT
  [[ ${SCRIPT_ONLY:-0} == 1 ]] && { echo "script only, done"; exit 0; }
fi

for PHASE in clean branch; do
  for M in $MODELS; do
    echo "=== $PHASE $M $(date -Is)"
    serve "${GPUS[0]}" $CAND_PORT "$M" "$(cfg $M path)" 0.88 "$RUN/server_logs/${PHASE}_$M.log"; CAND_PID=$SERVE_PID
    wait_up $CAND_PORT $CAND_PID
    $VLLM_PY -m turnprint.generate --run "$RUN" --profile "$PROFILE" --phase $PHASE --model "$M" \
      --cand-url http://127.0.0.1:$CAND_PORT --sim-url http://127.0.0.1:$SIM_PORT
    stop $CAND_PID; CAND_PID=
  done
done
echo "generation done $(date -Is)"
