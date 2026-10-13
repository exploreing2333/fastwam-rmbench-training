#!/usr/bin/env bash
bundle_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
unset PYTHONHOME
export PYTHONNOUSERSITE=1
export FASTWAM_RUNTIME_ROOT="${FASTWAM_RUNTIME_ROOT:-${bundle_root}/.runtime}"
mkdir -p "${FASTWAM_RUNTIME_ROOT}" || return "$?"
FASTWAM_RUNTIME_ROOT="$(cd -- "${FASTWAM_RUNTIME_ROOT}" && pwd -P)" || return "$?"
export FASTWAM_RUNTIME_ROOT
bash "${bundle_root}/bootstrap.sh" || return "$?"
export FASTWAM_BUNDLE_ROOT="${bundle_root}"
export FASTWAM_ENV_PREFIX="${FASTWAM_RUNTIME_ROOT}/fastwam"
export HF_HOME="${FASTWAM_RUNTIME_ROOT}/cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_LEROBOT_HOME="${HF_HOME}/lerobot"
export TORCH_HOME="${FASTWAM_RUNTIME_ROOT}/cache/torch"
export XDG_CACHE_HOME="${FASTWAM_RUNTIME_ROOT}/cache/xdg"
export TRITON_CACHE_DIR="${FASTWAM_RUNTIME_ROOT}/cache/triton"
TMPDIR="$("${FASTWAM_ENV_PREFIX}/bin/python" "${bundle_root}/scripts/resolve_ipc_tmp.py")" || return "$?"
export TMPDIR
export FASTWAM_IPC_TMPDIR="${TMPDIR}"
export PATH="${FASTWAM_ENV_PREFIX}/bin:${FASTWAM_RUNTIME_ROOT}/cuda/bin:${PATH}"
export PYTHONPATH="${bundle_root}/src:${bundle_root}/vendor/curobo_src:${bundle_root}/vendor/LIBERO"
export CUDA_HOME="${FASTWAM_RUNTIME_ROOT}/cuda"
export LD_LIBRARY_PATH="${FASTWAM_ENV_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${bundle_root}/models"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export USE_TF=0
export USE_FLAX=0
export TORCH_EXTENSIONS_DIR="${FASTWAM_RUNTIME_ROOT}/cache/torch_extensions"
