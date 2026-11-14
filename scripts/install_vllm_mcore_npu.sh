set -ex

USE_MEGATRON=${USE_MEGATRON:-1}
# Select the hardware-specific FLA NPU package before starting installation.
FLA_NPU_DEVICE=${FLA_NPU_DEVICE:-}
case "$FLA_NPU_DEVICE" in
    a2|a3|a5) ;;
    *)
        echo "Set FLA_NPU_DEVICE to a2 (Atlas A2), a3 (Atlas A3), or a5 (Ascend 950)." >&2
        exit 1
        ;;
esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "1. install basic packages"
pip uninstall -y triton triton-ascend
# Align with the vLLM-Ascend releases/v0.30.0 runtime requirements.
pip install torch==2.10.0 torch_npu==2.10.0.post7 torchvision==0.25.0 torchaudio==2.10.0
pip install triton-ascend==3.6.0.post2 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
pip install "transformers==5.14.1"
pip install setuptools-scm


echo "2. install vllm & vllm-ascend"
git clone --depth 1 --branch v0.30.0 https://github.com/vllm-project/vllm.git
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v -e .
cd ..
git clone -b releases/v0.30.0 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
pip install -r requirements.txt
COMPILE_CUSTOM_KERNELS=1 pip install --no-build-isolation --no-deps -v -e . --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
cd ..


if [ $USE_MEGATRON -eq 1 ]; then
    echo "3. install Megatron & MindSpeed"
    # 下载 MindSpeed，切换到指定 commit-id，并下载 Megatron 相关依赖
    git clone https://gitcode.com/Ascend/MindSpeed.git
    cd MindSpeed && git checkout 26.2.0_core_r0.18.0 && cd ..
    git clone --depth 1 --branch core_v0.18.2 https://github.com/NVIDIA/Megatron-LM.git
    git clone --depth 1 --branch 26.2.0_core_r0.18.0 https://gitcode.com/ascend/MegatronAdaptor.git
    git clone --depth 1 --branch 26.2.0_release_v2.17 https://gitcode.com/ascend/TransformerEngineNPU.git
    git clone --depth 1 --branch 26.2.0 https://gitcode.com/ascend/MindSpeed-Ops.git
    git clone --depth 1 --branch 26.2.0_v0.5.0 https://gitcode.com/ascend/MindSpeed-Bridge.git
    git clone --depth 1 --branch v0.5.0 https://github.com/NVIDIA-NeMo/Megatron-Bridge.git

    # Preserve Transformers' native Qwen3-ASR registration.
    git -C Megatron-Bridge apply --unidiff-zero \
        "${SCRIPT_DIR}/../docker/ascend/patches/megatron_bridge_qwen3_asr.patch"

    # 安装 Megatron 及 Ascend 适配组件
    pip install -e Megatron-LM
    pip install -e MegatronAdaptor
    pip install -e TransformerEngineNPU --no-build-isolation
    pip install -e MindSpeed
    pip install -e MindSpeed-Ops --no-build-isolation --no-deps
    pip install -r MindSpeed-Bridge/requirements.txt
    pip install -e MindSpeed-Bridge --no-deps
    pip install -e Megatron-Bridge --no-build-isolation --no-deps
fi

echo "4. install verl"
cd verl
pip install -v -e .
pip install -r requirements-npu.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
# （可选）提示：为了更佳的使用体验，最好将 recipe 子模块更新至最新 commit
cd recipe
git checkout main
cd ..

echo "5. May need to check other necessary packages"
# Restore the inference requirements after installing the training dependencies.
pip install torch_npu==2.10.0.post7 transformers==5.14.1 "xgrammar>=0.2.1,<1.0.0"
# Restore the vLLM Triton version after requirements-npu.txt installs its shared pin.
pip install triton-ascend==3.6.0.post2 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
# Install the hardware-specific FLA NPU backend used by vLLM-Ascend.
pip install --no-deps "flash-linear-attention-npu-${FLA_NPU_DEVICE}==26.9.1" \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi

echo "Successfully installed all packages"
