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
#   INSTALL_APT_DEPS=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${ROOT_DIR}/.cache/pytorch-vulkan-src"
PYTORCH_REF="${PYTORCH_REF:-v2.9.1}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv-vulkan}"
INSTALL_APT_DEPS="${INSTALL_APT_DEPS:-0}"

echo "[vulkan-build] Root: ${ROOT_DIR}"
echo "[vulkan-build] Ref: ${PYTORCH_REF}"
echo "[vulkan-build] Jobs: ${BUILD_JOBS}"
echo "[vulkan-build] Venv: ${VENV_PATH}"

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

python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

mkdir -p "${WORK_DIR}"
if [[ ! -d "${WORK_DIR}/.git" ]]; then
  git clone --recursive https://github.com/pytorch/pytorch.git "${WORK_DIR}"
fi

cd "${WORK_DIR}"
git fetch --tags --force
git checkout "${PYTORCH_REF}"
git submodule sync --recursive
git submodule update --init --recursive

python -m pip install -r requirements.txt

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

echo "[vulkan-build] Building wheel (this can take a long time)..."
python setup.py bdist_wheel

WHEEL_PATH="$(ls -1 dist/torch-*.whl | tail -n 1)"
echo "[vulkan-build] Installing ${WHEEL_PATH}"
python -m pip install --force-reinstall "${WHEEL_PATH}"

echo "[vulkan-build] Verifying Vulkan build..."
python "${ROOT_DIR}/scripts/verify_vulkan_torch.py"

echo
echo "[vulkan-build] Success."
echo "Activate this environment with:"
echo "  source ${VENV_PATH}/bin/activate"
echo "Then run autoresearch with:"
echo "  AUTORESEARCH_DEVICE=vulkan uv run train.py"
