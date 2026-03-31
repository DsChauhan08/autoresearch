#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

usage() {
  cat <<'USAGE'
Usage: bash scripts/run_igpu.sh <command>

Commands:
  prepare     Run prepare.py (CPU/iGPU-safe setup path)
  train       Run train.py with conservative host/runtime defaults
  help        Show this help

Examples:
  bash scripts/run_igpu.sh prepare
  bash scripts/run_igpu.sh train
USAGE
}

run_prepare() {
  shift || true
  echo "[igpu-run] Running data/tokenizer preparation..."
  exec uv run prepare.py "$@"
}

run_train() {
  shift || true

  : "${AUTORESEARCH_DEVICE:=auto}"
  : "${AUTORESEARCH_CPU_THREADS:=4}"
  : "${AUTORESEARCH_INTEROP_THREADS:=1}"
  : "${AUTORESEARCH_TOKENIZER_THREADS:=2}"
  : "${AUTORESEARCH_RAM_FRACTION:=0.50}"
  : "${AUTORESEARCH_NICE:=12}"
  : "${AUTORESEARCH_COMPILE:=0}"
  : "${AUTORESEARCH_AMP:=0}"
  : "${AUTORESEARCH_USE_MUON:=0}"
  : "${OMP_NUM_THREADS:=${AUTORESEARCH_CPU_THREADS}}"
  : "${OPENBLAS_NUM_THREADS:=${AUTORESEARCH_CPU_THREADS}}"
  : "${MKL_NUM_THREADS:=${AUTORESEARCH_CPU_THREADS}}"
  : "${NUMEXPR_NUM_THREADS:=${AUTORESEARCH_CPU_THREADS}}"
  : "${TOKENIZERS_PARALLELISM:=false}"

  case "${AUTORESEARCH_DEVICE}" in
    auto|cuda|vulkan|cpu) ;;
    *)
      echo "[igpu-run] ERROR: invalid AUTORESEARCH_DEVICE='${AUTORESEARCH_DEVICE}'"
      echo "[igpu-run] Expected one of: auto, cuda, vulkan, cpu"
      exit 1
      ;;
  esac

  export AUTORESEARCH_DEVICE
  export AUTORESEARCH_CPU_THREADS
  export AUTORESEARCH_INTEROP_THREADS
  export AUTORESEARCH_TOKENIZER_THREADS
  export AUTORESEARCH_RAM_FRACTION
  export AUTORESEARCH_NICE
  export AUTORESEARCH_COMPILE
  export AUTORESEARCH_AMP
  export AUTORESEARCH_USE_MUON
  export OMP_NUM_THREADS
  export OPENBLAS_NUM_THREADS
  export MKL_NUM_THREADS
  export NUMEXPR_NUM_THREADS
  export TOKENIZERS_PARALLELISM

  echo "[igpu-run] Device preference: ${AUTORESEARCH_DEVICE}"
  echo "[igpu-run] Thread limits: torch=${AUTORESEARCH_CPU_THREADS}, interop=${AUTORESEARCH_INTEROP_THREADS}, tokenizer=${AUTORESEARCH_TOKENIZER_THREADS}"
  echo "[igpu-run] Host BLAS caps: OMP=${OMP_NUM_THREADS}, OPENBLAS=${OPENBLAS_NUM_THREADS}, MKL=${MKL_NUM_THREADS}, NUMEXPR=${NUMEXPR_NUM_THREADS}"
  echo "[igpu-run] RAM budget target: fraction=${AUTORESEARCH_RAM_FRACTION} (override with AUTORESEARCH_RAM_MB)"
  echo "[igpu-run] Safe defaults: compile=${AUTORESEARCH_COMPILE}, amp=${AUTORESEARCH_AMP}, muon=${AUTORESEARCH_USE_MUON}, nice=${AUTORESEARCH_NICE}"

  if [[ "${AUTORESEARCH_DEVICE}" == "vulkan" ]]; then
    echo "[igpu-run] AUTORESEARCH_DEVICE=vulkan requested, delegating to scripts/run_vulkan.sh"
    exec bash scripts/run_vulkan.sh
  fi

  echo "[igpu-run] Running training with uv..."
  exec uv run train.py
}

COMMAND="${1:-help}"
case "${COMMAND}" in
  prepare)
    run_prepare "$@"
    ;;
  train)
    run_train "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "[igpu-run] ERROR: unknown command '${COMMAND}'"
    echo
    usage
    exit 1
    ;;
esac
