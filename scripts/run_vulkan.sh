#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[run_vulkan] Delegating to scripts/igpu_vulkan.sh train"
exec bash "${ROOT_DIR}/scripts/igpu_vulkan.sh" train
