#!/bin/bash

# Purpose: Build vLLM and VERL components for ARM64 (GB200) platforms

# This script runs inside an enroot container started from nvidia/cuda:12.9.1-devel-ubuntu22.04 base image.
# Do this prior to running this script:
#   >> enroot import --output ./cuda12.9.1-base.sqsh   docker://nvidia/cuda:12.9.1-devel-ubuntu22.04
#   >> enroot create --name verl-build cuda12.9.1-base.sqsh
#   >> enroot start --rw verl-build bash


set -e  # Exit on error

export DEBIAN_FRONTEND=noninteractive
export PIP_NO_CACHE_DIR=1

echo "Installing system dependencies..."
apt-get update && apt-get install -y \
    git \
    wget \
    cmake \
    build-essential \
    libibverbs-dev \
    libnuma-dev \
    librdmacm-dev \
    numactl \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install -y \
    python3.12 \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

echo "Installing pip for Python 3.12..."
wget https://bootstrap.pypa.io/get-pip.py && \
    python3.12 get-pip.py && \
    rm get-pip.py

echo "Setting Python 3.12 as default..."
ln -sf /usr/bin/python3.12 /usr/bin/python3
ln -sf /usr/bin/python3.12 /usr/bin/python

echo "Installing PyTorch 2.9.1 with CUDA 12.9..."
pip install torch==2.9.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
sed -i '/nvidia-cudnn-cu12/d' /usr/local/lib/python3.12/dist-packages/torch-2.9.1+cu129.dist-info/METADATA
pip install --no-deps --force-reinstall nvidia-cudnn-cu12==9.16.0.29

# fix cudnn not found error
echo "Creating symlinks for cuDNN headers..."
ln -sf /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/include/*.h /usr/include/

echo "Installing vLLM..."
# GB200 Blackwell architecture only
export TORCH_CUDA_ARCH_LIST="10.0"
git clone --depth 1 -b v0.12.0 https://github.com/vllm-project/vllm.git && \
    cd vllm && \
    find requirements -name "*.txt" -print0 | xargs -0 sed -i '/torch/d' && \
    pip install -r requirements/build.txt && \
    pip install -e . --no-build-isolation --no-deps && \
    pip install -r requirements/cuda.txt

echo "Installing pybind11..."
pip install pybind11

echo "Installing CUDA keyring and cuDNN..."
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
else
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
fi
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update
# Install cuDNN with development headers
apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12
# Verify cuDNN headers are accessible
if [ ! -f "/usr/include/cudnn.h" ] && [ ! -f "/usr/local/cuda/include/cudnn.h" ]; then
    echo "ERROR: cudnn.h not found after installation!"
    exit 1
fi
rm -rf /var/lib/apt/lists/*

echo "Installing nvidia-mathdx..."
pip install nvidia-mathdx

echo "Installing Apex..."
MAX_JOBS=4 pip install -v --disable-pip-version-check --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex.git

echo "Installing TransformerEngine..."
export NVTE_FRAMEWORK=pytorch
MAX_JOBS=4 NVTE_BUILD_THREADS_PER_JOB=4 pip3 install --resume-retries 999 --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.10

echo "Installing transformers and tokenizers..."
pip uninstall transformers
# to fix model import issue
pip install --upgrade "transformers>=4.56.0,<5.0.0" tokenizers

echo "Installing additional packages..."
pip install codetiming tensordict mathruler pylatexenc qwen_vl_utils

echo "Installing flash_attn..."
pip install flash_attn==2.8.1 --no-build-isolation

echo "Installing Nsight Systems..."
NSIGHT_VERSION=$(if [ "$(uname -m)" = "aarch64" ]; then echo "2025.6.1_2025.6.1.190-1_arm64"; else echo "2025.6.1_2025.6.1.190-1_amd64"; fi)
wget https://developer.download.nvidia.com/devtools/nsight-systems/nsight-systems-${NSIGHT_VERSION}.deb
apt-get update && apt-get install -y libxcb-cursor0
apt-get install -y ./nsight-systems-${NSIGHT_VERSION}.deb
rm -rf /usr/local/cuda/bin/nsys
ln -s /opt/nvidia/nsight-systems/2025.6.1/nsys  /usr/local/cuda/bin/nsys
rm -rf /usr/local/cuda/bin/nsys-ui
ln -s /opt/nvidia/nsight-systems/2025.6.1/nsys-ui /usr/local/cuda/bin/nsys-ui

echo "Installing DeepEP..."
mkdir -p /home/dpsk_a2a
cd /home/dpsk_a2a
git clone -b v2.5.1 https://github.com/NVIDIA/gdrcopy.git
cd gdrcopy
make prefix=/usr/local lib_install
cd .. && rm -rf gdrcopy

export GDRCOPY_HOME=/usr/local

git clone -b v1.2.1 https://github.com/deepseek-ai/DeepEP.git
export NVSHMEM_DIR=/usr/local/lib/python3.12/dist-packages/nvidia/nvshmem
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"
cd ${NVSHMEM_DIR}/lib
ln -s libnvshmem_host.so.3 libnvshmem_host.so
cd /home/dpsk_a2a/DeepEP
python setup.py install

echo "Installing additional Python packages..."
pip3 install --no-deps trl
pip3 install nvtx matplotlib liger_kernel
pip install -U git+https://github.com/ISEEKYAN/mbridge.git

echo "Installing Megatron-LM..."
pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.15.0

echo "Installing VERL..."
pip install git+https://github.com/volcengine/verl.git@v0.6.0
# Note: Original Dockerfile uninstalls VERL here to allow mounting local dev version
# If you don't need that, comment out the line below:
# pip uninstall -y verl

echo "Installing curl..."
apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

echo "Build complete!"
echo ""
echo "================================================================"
echo "Build finished successfully!"
echo "================================================================"
echo "To save this container as a reusable image, run in another terminal:"
echo "  enroot export -o ~/enroot_data/verl-vllm-gb200.sqsh verl-build"
echo "================================================================"

