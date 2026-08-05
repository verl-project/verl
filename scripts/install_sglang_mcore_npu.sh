#!/bin/bash
set -e
USE_MEGATRON=${USE_MEGATRON:-1}
export MAX_JOBS=32

echo "1. install SGLang from source"
git clone -b v0.5.10 https://github.com/sgl-project/sglang.git
cd sglang
# set NPU install setting
mv python/pyproject.toml python/pyproject.toml.backup 
mv python/pyproject_npu.toml python/pyproject.toml 
pip install -e python[srt_npu]
cd ..

echo "2. install torch & torch_npu & other basic packages"
pip install torch==2.8.0 torch_npu==2.8.0.post2 torchvision==0.23.0 pyyaml
pip install pybind11 click==8.2.1 mbridge "numpy<2.0.0" cachetools


echo "3. install sgl-kernel-npu from release whl"
ARCH=$(uname -m) && wget --no-check-certificate https://github.com/sgl-project/sgl-kernel-npu/releases/download/2026.02.01/sgl-kernel-npu-2026.02.01-torch2.8.0-py311-cann8.5.0-a3-${ARCH}.zip
unzip sgl-kernel-npu*.zip
pip install torch_memory_saver*.whl
pip install sgl_kernel_npu*.whl
pip install deep_ep*.whl
cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so && cd -
python -c "import deep_ep; print(deep_ep.__path__)"
cd ..

if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install Megatron & MindSpeed"
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

echo "5. install verl "
cd verl/recipe && git checkout main && cd .. && \
pip install -r requirements-npu.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
pip install -v -e . && cd .. &&

echo "6. May need to uninstall timm & check other neccessary packages"
pip uninstall -y timm 
pip install pyyaml uvicorn fastapi pybase64 openai partial_json_parser python-multipart
echo "Successfully installed all packages"
