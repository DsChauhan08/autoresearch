#!/usr/bin/env bash
set -euo pipefail

# Install a previously built Vulkan-enabled torch wheel into a venv.
#
# Usage:
#   bash scripts/install_pytorch_vulkan_wheel.sh
#
# Optional environment variables:
#   VENV_PATH=.venv-vulkan
#   PYTHON_BIN=python3.11
#   WHEEL_PATH=dist/torch-*.whl
#   INSTALL_RUNTIME_DEPS=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv-vulkan}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WHEEL_PATH="${WHEEL_PATH:-}"
INSTALL_RUNTIME_DEPS="${INSTALL_RUNTIME_DEPS:-1}"

if [[ "${PYTHON_BIN}" == "python3" ]] && command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[vulkan-wheel] ERROR: Python interpreter '${PYTHON_BIN}' not found."
  exit 1
fi

if [[ -z "${WHEEL_PATH}" ]]; then
  WHEEL_PATH="$(find "${ROOT_DIR}/dist" -maxdepth 1 -type f -name 'torch-*.whl' | sort | tail -n 1 || true)"
fi

if [[ -z "${WHEEL_PATH}" ]]; then
  echo "[vulkan-wheel] ERROR: no prebuilt torch wheel found in ${ROOT_DIR}/dist"
  echo "[vulkan-wheel] Build one first with: bash scripts/build_pytorch_vulkan.sh"
  exit 1
fi

if [[ ! -f "${WHEEL_PATH}" ]]; then
  echo "[vulkan-wheel] ERROR: wheel not found: ${WHEEL_PATH}"
  exit 1
fi

echo "[vulkan-wheel] Venv: ${VENV_PATH}"
echo "[vulkan-wheel] Python: ${PYTHON_BIN}"
echo "[vulkan-wheel] Wheel: ${WHEEL_PATH}"

"${PYTHON_BIN}" -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install --force-reinstall "${WHEEL_PATH}"
if [[ "${INSTALL_RUNTIME_DEPS}" == "1" ]]; then
  PYTHON_BIN=python bash "${ROOT_DIR}/scripts/install_autoresearch_runtime_deps.sh"
fi
python "${ROOT_DIR}/scripts/verify_vulkan_torch.py"

echo
echo "[vulkan-wheel] Success."
echo "Activate with:"
echo "  source ${VENV_PATH}/bin/activate"
echo "Then run:"
echo "  bash scripts/igpu_vulkan.sh train"
