set -ex

USE_MEGATRON=${USE_MEGATRON:-1}

echo "1. install basic packages"
pip uninstall -y triton triton-ascend
# 安装与 vLLM-Ascend 0.23.0 对应的软件包
pip install torchvision==0.25.0
pip install torchaudio==2.10.0
pip install triton-ascend==3.2.1 --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
pip install "transformers==5.10.4" 
# vLLM v0.23.0's [build-system].requires pins torch==2.11.0. Default PEP 517
# build isolation would provision a build overlay by downloading torch 2.11
# and nvidia_* CUDA wheels, leaving pip check warnings and a dangling
# torch_npu 2.10.0 dependent. VLLM_TARGET_DEVICE=empty does not influence
# build isolation. Below we install the build backend deps up front and pass
# --no-build-isolation to the vLLM install, matching the pattern already
# used for the vLLM-Ascend install a few lines below.
pip install setuptools-scm setuptools-rust wheel


echo "2. install vllm & vllm-ascend"
git clone --depth 1 --branch v0.23.0 https://github.com/vllm-project/vllm.git
cd vllm
VLLM_TARGET_DEVICE=empty pip install -v --no-build-isolation -e .
cd ..
git clone -b releases/v0.23.0 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git submodule update --init --recursive
pip install -v -e . --no-build-isolation --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
cd ..


if [ $USE_MEGATRON -eq 1 ]; then
    echo "3. install Megatron & MindSpeed"
    # 下载 MindSpeed，切换到指定 commit-id，并下载 Megatron-LM
    git clone https://gitcode.com/Ascend/MindSpeed.git
    cd MindSpeed && git checkout core_r0.16.0 && cd ..
    git clone --depth 1 --branch core_r0.16.0 https://github.com/NVIDIA/Megatron-LM.git

    # 安装 Megatron & MindSpeed
    pip install -e Megatron-LM
    pip install -e MindSpeed

    # 安装 mbridge
    pip install mbridge
fi

echo "4. install verl"
cd verl
pip install -r requirements-npu.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
pip install -v -e .
# （可选）提示：为了更佳的使用体验，最好将 recipe 子模块更新至最新 commit
cd recipe
git checkout main
cd ..

echo "5. May need to check other neccessary packages"
pip install transformers==5.10.4 xgrammar==0.1.33
echo "Successfully installed all packages"