#!/usr/bin/env bash
set -euo pipefail

# iGPU-friendly Vulkan entrypoint for setup/readiness checks/training.
#
# Usage:
#   bash scripts/igpu_vulkan.sh setup
#   bash scripts/igpu_vulkan.sh check
#   bash scripts/igpu_vulkan.sh train
#   bash scripts/igpu_vulkan.sh quickstart
#
# Optional environment variables:
#   VENV_PATH=.venv-vulkan
#   PYTHON_BIN=python3.11
#   WHEEL_PATH=dist/torch-*.whl
#   AUTO_BUILD_IF_MISSING=0
#   SKIP_VULKANINFO=0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv-vulkan}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
AUTO_BUILD_IF_MISSING="${AUTO_BUILD_IF_MISSING:-0}"
SKIP_VULKANINFO="${SKIP_VULKANINFO:-0}"

if [[ "${PYTHON_BIN}" == "python3" ]] && command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
fi

usage() {
  cat <<'EOF'
Usage: bash scripts/igpu_vulkan.sh <command>

Commands:
  setup       Ensure .venv-vulkan has Vulkan-enabled torch (from dist wheel or build)
  check       Check system Vulkan tooling and verify torch Vulkan operators
  train       Run train.py with iGPU-safe Vulkan defaults
  quickstart  setup + check + train
  help        Show this help

Tips:
  - Put a prebuilt wheel in dist/ or pass WHEEL_PATH to avoid rebuilds.
  - Set AUTO_BUILD_IF_MISSING=1 to auto-build torch when no wheel is found.
EOF
}

find_default_wheel() {
  if [[ ! -d "${ROOT_DIR}/dist" ]]; then
    return 0
  fi
  find "${ROOT_DIR}/dist" -maxdepth 1 -type f -name 'torch-*.whl' | sort | tail -n 1 || true
}

resolve_python_for_run() {
  if [[ -x "${VENV_PATH}/bin/python" ]]; then
    echo "${VENV_PATH}/bin/python"
    return
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    echo "${VIRTUAL_ENV}/bin/python"
    return
  fi
  if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    command -v "${PYTHON_BIN}"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  return 1
}

verify_vulkan_with_python() {
  local python_exec="$1"
  "${python_exec}" "${ROOT_DIR}/scripts/verify_vulkan_torch.py"
}

ensure_setup() {
  local wheel_path="${WHEEL_PATH:-}"

  if [[ -x "${VENV_PATH}/bin/python" ]]; then
    if verify_vulkan_with_python "${VENV_PATH}/bin/python"; then
      echo "[igpu-vulkan] Vulkan torch is already ready in ${VENV_PATH}"
      return 0
    fi
  fi

  if [[ -z "${wheel_path}" ]]; then
    wheel_path="$(find_default_wheel)"
  fi

  if [[ -n "${wheel_path}" ]]; then
    echo "[igpu-vulkan] Installing Vulkan torch wheel: ${wheel_path}"
    VENV_PATH="${VENV_PATH}" PYTHON_BIN="${PYTHON_BIN}" WHEEL_PATH="${wheel_path}" \
      INSTALL_RUNTIME_DEPS=1 bash "${ROOT_DIR}/scripts/install_pytorch_vulkan_wheel.sh"
    return 0
  fi

  if [[ "${AUTO_BUILD_IF_MISSING}" == "1" ]]; then
    echo "[igpu-vulkan] No local wheel found; building Vulkan-enabled torch..."
    VENV_PATH="${VENV_PATH}" PYTHON_BIN="${PYTHON_BIN}" INSTALL_RUNTIME_DEPS=1 \
      bash "${ROOT_DIR}/scripts/build_pytorch_vulkan.sh"
    return 0
  fi

  echo "[igpu-vulkan] ERROR: no Vulkan torch wheel found in ${ROOT_DIR}/dist"
  echo "[igpu-vulkan] Either:"
  echo "  - set WHEEL_PATH=/path/to/torch-*.whl and re-run setup, or"
  echo "  - run: bash scripts/build_pytorch_vulkan.sh"
  echo "  - or set AUTO_BUILD_IF_MISSING=1 and run setup"
  return 1
}

check_readiness() {
  local python_exec
  python_exec="$(resolve_python_for_run)"

  echo "[igpu-vulkan] Root: ${ROOT_DIR}"
  echo "[igpu-vulkan] Python: ${python_exec}"
  "${python_exec}" --version

  if [[ "${SKIP_VULKANINFO}" != "1" ]]; then
    if command -v vulkaninfo >/dev/null 2>&1; then
      echo "[igpu-vulkan] Running vulkaninfo --summary"
      if vulkaninfo --summary >/dev/null 2>&1; then
        echo "[igpu-vulkan] vulkaninfo: ok"
      else
        echo "[igpu-vulkan] WARNING: vulkaninfo command failed; driver/runtime may be incomplete"
      fi
    else
      echo "[igpu-vulkan] WARNING: 'vulkaninfo' not found (install vulkan-tools for deeper diagnostics)"
    fi
  fi

  verify_vulkan_with_python "${python_exec}"
}

run_training() {
  local python_exec
  python_exec="$(resolve_python_for_run)"

  verify_vulkan_with_python "${python_exec}"

  : "${AUTORESEARCH_DEVICE:=vulkan}"
  : "${AUTORESEARCH_CPU_THREADS:=4}"
  : "${AUTORESEARCH_INTEROP_THREADS:=1}"
  : "${AUTORESEARCH_TOKENIZER_THREADS:=2}"
  : "${AUTORESEARCH_NICE:=12}"
  : "${AUTORESEARCH_COMPILE:=0}"
  : "${AUTORESEARCH_AMP:=0}"
  : "${AUTORESEARCH_USE_MUON:=0}"
  : "${AUTORESEARCH_DTYPE:=float32}"
  : "${OMP_NUM_THREADS:=${AUTORESEARCH_CPU_THREADS}}"
  : "${OPENBLAS_NUM_THREADS:=${AUTORESEARCH_CPU_THREADS}}"
  : "${MKL_NUM_THREADS:=${AUTORESEARCH_CPU_THREADS}}"
  : "${NUMEXPR_NUM_THREADS:=${AUTORESEARCH_CPU_THREADS}}"
  : "${TOKENIZERS_PARALLELISM:=false}"

  export AUTORESEARCH_DEVICE
  export AUTORESEARCH_CPU_THREADS
  export AUTORESEARCH_INTEROP_THREADS
  export AUTORESEARCH_TOKENIZER_THREADS
  export AUTORESEARCH_NICE
  export AUTORESEARCH_COMPILE
  export AUTORESEARCH_AMP
  export AUTORESEARCH_USE_MUON
  export AUTORESEARCH_DTYPE
  export OMP_NUM_THREADS
  export OPENBLAS_NUM_THREADS
  export MKL_NUM_THREADS
  export NUMEXPR_NUM_THREADS
  export TOKENIZERS_PARALLELISM

  echo "[igpu-vulkan] Device: ${AUTORESEARCH_DEVICE}"
  echo "[igpu-vulkan] Thread limits: torch=${AUTORESEARCH_CPU_THREADS}, interop=${AUTORESEARCH_INTEROP_THREADS}, tokenizer=${AUTORESEARCH_TOKENIZER_THREADS}"
  echo "[igpu-vulkan] Host BLAS caps: OMP=${OMP_NUM_THREADS}, OPENBLAS=${OPENBLAS_NUM_THREADS}, MKL=${MKL_NUM_THREADS}, NUMEXPR=${NUMEXPR_NUM_THREADS}"
  echo "[igpu-vulkan] Safe defaults: compile=${AUTORESEARCH_COMPILE}, amp=${AUTORESEARCH_AMP}, muon=${AUTORESEARCH_USE_MUON}, dtype=${AUTORESEARCH_DTYPE}, nice=${AUTORESEARCH_NICE}"
  echo "[igpu-vulkan] Launching train.py with iGPU-safe Vulkan defaults"
  cd "${ROOT_DIR}"
  "${python_exec}" train.py
}

COMMAND="${1:-help}"
case "${COMMAND}" in
  setup)
    ensure_setup
    ;;
  check)
    check_readiness
    ;;
  train)
    run_training
    ;;
  quickstart)
    ensure_setup
    check_readiness
    run_training
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "[igpu-vulkan] ERROR: unknown command '${COMMAND}'"
    echo
    usage
    exit 1
    ;;
esac
