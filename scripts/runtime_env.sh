#!/usr/bin/env bash
# Source after activating the project's environment. CUDA packages installed by
# pip contain nvcc, but do not put it on PATH for FlashInfer's JIT compiler.
_llm_cuda_root="$(python -c 'import sysconfig; from pathlib import Path; print(Path(sysconfig.get_path("purelib")) / "nvidia" / "cu13")')"
if [[ -z "${CUDA_HOME:-}" && -x "$_llm_cuda_root/bin/nvcc" ]]; then
  export CUDA_HOME="$_llm_cuda_root"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    printf 'CUDA_HOME has no executable nvcc: %s\n' "$CUDA_HOME" >&2
    unset _llm_cuda_root
    return 1
  fi
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
  export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
  if [[ -f "$CUDA_HOME/lib/libcudart.so.13" && ! -e "$CUDA_HOME/lib/libcudart.so" ]]; then
    # pip provides the versioned runtime; JIT linking requests -lcudart.
    _llm_cuda_link_dir="${XDG_CACHE_HOME:-$HOME/.cache}/llm-reasoning-grpo/cuda-link"
    mkdir -p "$_llm_cuda_link_dir"
    ln -sfn "$CUDA_HOME/lib/libcudart.so.13" "$_llm_cuda_link_dir/libcudart.so"
    export LIBRARY_PATH="$_llm_cuda_link_dir:$LIBRARY_PATH"
    unset _llm_cuda_link_dir
  fi
fi
if [[ -f /usr/lib/wsl/lib/libcuda.so ]]; then
  # WSL supplies the Windows driver's libcuda outside the pip toolkit.
  export LIBRARY_PATH="/usr/lib/wsl/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi
# Keep JIT compilation within the WSL host-memory budget.
export MAX_JOBS="${MAX_JOBS:-2}"
unset _llm_cuda_root
