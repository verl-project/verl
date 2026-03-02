
#验证版本 CANN 8.5.0.B160 路径 /home/CANN/CANN8.5.0/ascend-toolkit
类似
#source /usr/local/Ascend/ascend-toolkit/set_env.sh
#source /usr/local/Ascend/nnal/atb/set_env.sh

# python3.11
conda create -n verl_qwen3.5_fsdp python=3.11
conda activate verl_qwen3.5_fsdp


# for torch-npu dev version or x86 machine [Optional]
# pip install 后缀需加上 --trusted-host download.pytorch.org --trusted-host mirrors.huaweicloud.com
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/   https://mirrors.huaweicloud.com/ascend/repos/pypi  "


# 安装vllm 0.16.0
git clone https://github.com/vllm-project/vllm.git  
cd vllm
# PR还未合入 跟踪：https://github.com/vllm-project/vllm/pull/34521  
git fetch origin pull/34521/head:pr-34521
git checkout pr-34521
pip install -r requirements/build.txt
VLLM_TARGET_DEVICE=empty pip install -v .
cd ..


# 安装vllm-ascend  0.14.0
git clone https://github.com/vllm-project/vllm-ascend.git  
cd vllm-ascend
# PR还未合入 跟踪：https://github.com/vllm-project/vllm-ascend/pull/6742  
git fetch origin pull/6742/head:pr-6742
git checkout pr-6742
pip install -r requirements.txt
export COMPILE_CUSTOM_KERNELS=1
python setup.py install
cd ..


# 源码安装transformers 5.2.0.dev0
cd transformers
# git checkout fc91372
git checkout 8e26f7e94b310dc9359bcf8bbe8fed453b4e8916
pip install -e .
cd ..

pip install triton-ascend==3.2.0 accelerate==1.2.0


# 安装verl
cd verl
# git checkout 9433f8a8f2771256ea4f8f94e4401bcfe9703228
pip install -e .
cd ..

# 因安装环境可能导致覆盖，需重新安装torch_npu
pip install torch_npu==2.9.0 torchvision==0.24.0






-----------------------------------------------------------------
# ray安装失败时执行
pip install grpcio==1.76.0
