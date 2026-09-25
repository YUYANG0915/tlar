#!/bin/bash
# Source inside the job, before importing torch, transformers or vLLM.
configure_cache() {
  local base="${TLAR_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/tlar}"
  local run="$base/tmp/tlar_${SLURM_JOB_ID:-manual_$$}_${SLURM_ARRAY_TASK_ID:-0}"
  local key probe
  export HF_HOME="$base/hf_cache"
  export HF_HUB_CACHE="$HF_HOME/hub"
  export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
  export HF_DATASETS_CACHE="$HF_HOME/datasets"
  export HF_MODULES_CACHE="$HF_HOME/modules"
  export HF_XET_CACHE="$HF_HOME/xet"
  export TRANSFORMERS_CACHE="$HF_HOME/transformers"
  export TMPDIR="$run/tmp" TMP="$run/tmp" TEMP="$run/tmp"
  export XDG_CACHE_HOME="$run/cache"
  export XDG_CONFIG_HOME="$run/config" XDG_DATA_HOME="$run/data"
  export FLASHINFER_CACHE_DIR="$XDG_CACHE_HOME/flashinfer"
  export FLASHINFER_WORKSPACE_DIR="$FLASHINFER_CACHE_DIR/workspace"
  export FLASHINFER_JIT_CACHE_DIR="$FLASHINFER_CACHE_DIR/jit"
  export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
  export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
  export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch_extensions"
  export TORCH_HOME="$XDG_CACHE_HOME/torch"
  export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
  export VLLM_CACHE_ROOT="$XDG_CACHE_HOME/vllm"
  export VLLM_CONFIG_ROOT="$XDG_CONFIG_HOME/vllm"
  export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
  export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
  export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
  export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
  for key in HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE HF_MODULES_CACHE HF_XET_CACHE \
    TRANSFORMERS_CACHE TMPDIR XDG_CACHE_HOME XDG_CONFIG_HOME XDG_DATA_HOME \
    FLASHINFER_CACHE_DIR FLASHINFER_WORKSPACE_DIR FLASHINFER_JIT_CACHE_DIR \
    TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR TORCH_EXTENSIONS_DIR TORCH_HOME \
    CUDA_CACHE_PATH VLLM_CACHE_ROOT VLLM_CONFIG_ROOT NUMBA_CACHE_DIR MPLCONFIGDIR \
    PIP_CACHE_DIR UV_CACHE_DIR; do
    mkdir -p "${!key}" || return 1
    probe=$(mktemp "${!key}/.tlar_write_check.XXXXXX") || return 1
    rm -- "$probe" || return 1
  done
  printf 'Cache directories configured.\n'
}
configure_cache
