# Shared environment for TurnPrint Slurm steps.
export VLLM_PY=/dataset1/zailong/envs/peft-sft-lab/bin/python
export VLLM_BIN=/dataset1/zailong/envs/peft-sft-lab/bin/vllm
export ANA_PY=/dataset1/zailong/envs/turnprint/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# node-local /tmp is small; keep temp files and compile caches on /dataset1
export TMPDIR=/dataset1/zailong/tmp/tp-${SLURM_JOB_ID:-local}
export TMP=$TMPDIR TEMP=$TMPDIR
export VLLM_CACHE_ROOT=/dataset1/zailong/cache/turnprint/vllm
export MPLCONFIGDIR=$TMPDIR/mpl
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT" "$MPLCONFIGDIR"
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
export VLLM_USE_FLASHINFER_SAMPLER=0  # flashinfer JIT needs nvcc, absent on compute nodes
