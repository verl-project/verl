#!/bin/bash
set -e
NPU_DEVICE=${NPU_DEVICE:=A3}

export MAX_JOBS=32

echo "1. install SGLang from source"
git clone -b v0.5.8 https://github.com/sgl-project/sglang.git
cd sglang
mv python/pyproject_other.toml python/pyproject.toml
pip install -e python[srt_npu]
cd ..

echo "2. install torch & torch_npu & triton_ascend & other basic packages"
pip install torch==2.7.1 torch_npu==2.7.1.post2 torchvision==0.22.1
pip install pybind11 click==8.2.1 mbridge "numpy<2.0.0" cachetools


echo "3. install sgl-kernel-npu form source, detailed readme in https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md"
git clone https://github.com/sgl-project/sgl-kernel-npu.git
cd sgl-kernel-npu
git checkout 46b73de
sed -i '101s/^/# /' build.sh
if [ "$NPU_DEVICE" = "A3" ]; then
    bash build.sh
fi
if [ "$NPU_DEVICE" = "A2" ]; then
    bash build.sh -a deepep2
fi
pip install output/torch_memory_saver*.whl
pip install output/sgl_kernel_npu*.whl
pip install output/deep_ep*.whl
cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so && cd -
python -c "import deep_ep; print(deep_ep.__path__)"
cd ..
# install sgl-kernel-npu from release whl
# if [ "$NPU_DEVICE" = "A3" ]; then
#     wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/2026.01.21/sgl-kernel-npu_2026.01.21_8.5.0_a3.zip
# fi
# if [ "$NPU_DEVICE" = "A2" ]; then
#     wget https://github.com/sgl-project/sgl-kernel-npu/releases/download/2026.01.21/sgl-kernel-npu_2026.01.21_8.5.0_910b.zip
# fi
# unzip sgl-kernel-npu*.zip
# pip install output/torch_memory_saver*.whl
# pip install output/sgl_kernel_npu*.whl
# pip install output/deep_ep*.whl

if [ $USE_MEGATRON -eq 1 ]; then
    echo "4. install Megatron and MindSpeed"
    git clone -b 2.3.0_core_r0.12.1 https://gitcode.com/Ascend/MindSpeed.git 
    pip install -e MindSpeed 
    pip install git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.12.1 
fi

echo "5. May need to uninstall timm & triton"
pip uninstall -y timm triton
echo "Successfully installed all packages"
