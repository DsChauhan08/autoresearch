#!/usr/bin/env bash
set -euo pipefail

# Build/install a local PyTorch wheel with Vulkan enabled.
# This script intentionally keeps settings explicit so users can reproduce builds.
#
# Usage:
#   bash scripts/build_pytorch_vulkan.sh
#
# Optional environment variables:
#   PYTORCH_REF=v2.9.1
#   BUILD_JOBS=4
#   VENV_PATH=.venv-vulkan
#   PYTHON_BIN=python3.10
#   WORK_DIR=.cache/pytorch-vulkan-src
#   CLONE_DEPTH=1
#   CLEAN_BUILD=1
#   EXPORT_WHEEL_DIR=dist
#   INSTALL_APT_DEPS=1
#   INSTALL_RUNTIME_DEPS=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${WORK_DIR:-${ROOT_DIR}/.cache/pytorch-vulkan-src}"
PYTORCH_REF="${PYTORCH_REF:-v2.9.1}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv-vulkan}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CLONE_DEPTH="${CLONE_DEPTH:-1}"
CLEAN_BUILD="${CLEAN_BUILD:-1}"
EXPORT_WHEEL_DIR="${EXPORT_WHEEL_DIR:-${ROOT_DIR}/dist}"
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-0}"
INSTALL_RUNTIME_DEPS="${INSTALL_RUNTIME_DEPS:-1}"

if [[ "${PYTHON_BIN}" == "python3" ]] && command -v python3.11 >/dev/null 2>&1; then
  # PyTorch source builds are currently more reliable on 3.11 than bleeding-edge interpreters.
  PYTHON_BIN="python3.11"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[vulkan-build] ERROR: Python interpreter '${PYTHON_BIN}' not found."
  echo "[vulkan-build] Set PYTHON_BIN explicitly, e.g. PYTHON_BIN=python3.11"
  exit 1
fi

echo "[vulkan-build] Root: ${ROOT_DIR}"
echo "[vulkan-build] Ref: ${PYTORCH_REF}"
echo "[vulkan-build] Jobs: ${BUILD_JOBS}"
echo "[vulkan-build] Workdir: ${WORK_DIR}"
echo "[vulkan-build] Venv: ${VENV_PATH}"
echo "[vulkan-build] Python: ${PYTHON_BIN}"
echo "[vulkan-build] Clean build dir: ${CLEAN_BUILD}"
"${PYTHON_BIN}" --version

if [[ "${INSTALL_APT_DEPS}" == "1" ]]; then
  echo "[vulkan-build] Installing apt build dependencies..."
  sudo apt-get update
  sudo apt-get install -y \
    build-essential \
    git \
    cmake \
    ninja-build \
    python3-dev \
    python3-venv \
    libvulkan-dev \
    vulkan-tools
fi

"${PYTHON_BIN}" -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

mkdir -p "${WORK_DIR}"
if [[ ! -d "${WORK_DIR}/.git" ]]; then
  if [[ "${CLONE_DEPTH}" == "0" ]]; then
    git clone --recursive https://github.com/pytorch/pytorch.git "${WORK_DIR}"
  else
    git clone --recursive --depth "${CLONE_DEPTH}" --shallow-submodules \
      https://github.com/pytorch/pytorch.git "${WORK_DIR}"
  fi
fi

cd "${WORK_DIR}"
git fetch --tags --force
git checkout "${PYTORCH_REF}"
git submodule sync --recursive
git submodule update --init --recursive

python -m pip install -r requirements.txt

if [[ "${CLEAN_BUILD}" == "1" ]]; then
  echo "[vulkan-build] Removing stale build artifacts..."
  rm -rf build dist
fi

PYTHON_EXE="$(command -v python)"
if [[ ! -x "${PYTHON_EXE}" ]]; then
  echo "[vulkan-build] ERROR: active venv python executable not found"
  exit 1
fi

export USE_CUDA=0
export USE_ROCM=0
export USE_XPU=0
export USE_MPS=0
export USE_VULKAN=1
export USE_VULKAN_FP16_INFERENCE=0
export USE_VULKAN_RELAXED_PRECISION=0
export BUILD_TEST=0
export BUILD_CAFFE2_OPS=0
export MAX_JOBS="${BUILD_JOBS}"
export PYTHON_EXECUTABLE="${PYTHON_EXE}"
export CMAKE_PREFIX_PATH="${VIRTUAL_ENV}${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"

echo "[vulkan-build] Building wheel (this can take a long time)..."
python setup.py bdist_wheel

WHEEL_PATH="$(find dist -maxdepth 1 -type f -name 'torch-*.whl' | sort | tail -n 1)"
if [[ -z "${WHEEL_PATH}" ]]; then
  echo "[vulkan-build] ERROR: no torch wheel produced under ${WORK_DIR}/dist"
  exit 1
fi

mkdir -p "${EXPORT_WHEEL_DIR}"
EXPORTED_WHEEL="${EXPORT_WHEEL_DIR}/$(basename "${WHEEL_PATH}")"
cp -f "${WHEEL_PATH}" "${EXPORTED_WHEEL}"

echo "[vulkan-build] Prebuilt wheel exported to ${EXPORTED_WHEEL}"
echo "[vulkan-build] Installing ${EXPORTED_WHEEL}"
python -m pip install --force-reinstall "${EXPORTED_WHEEL}"
if [[ "${INSTALL_RUNTIME_DEPS}" == "1" ]]; then
  PYTHON_BIN=python bash "${ROOT_DIR}/scripts/install_autoresearch_runtime_deps.sh"
fi

echo "[vulkan-build] Verifying Vulkan build..."
python "${ROOT_DIR}/scripts/verify_vulkan_torch.py"

echo
echo "[vulkan-build] Success."
echo "Activate this environment with:"
echo "  source ${VENV_PATH}/bin/activate"
echo "Then run autoresearch with:"
echo "  bash scripts/igpu_vulkan.sh train"
