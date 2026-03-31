#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

COMMAND="${1:-train}"

case "${COMMAND}" in
  prepare)
    echo "[run_vulkan] Running prepare.py with Vulkan environment"
    export AUTORESEARCH_DEVICE=vulkan
    exec uv run prepare.py
    ;;
  train|check|setup|quickstart|help|-h|--help)
    echo "[run_vulkan] Delegating to scripts/igpu_vulkan.sh ${COMMAND}"
    exec bash "${ROOT_DIR}/scripts/igpu_vulkan.sh" "${COMMAND}"
    ;;
  *)
    echo "[run_vulkan] ERROR: unknown command '${COMMAND}'"
    echo "[run_vulkan] Expected one of: prepare, train, check, setup, quickstart, help"
    exit 1
    ;;
esac
