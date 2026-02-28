#!/bin/bash
# 激活verl-qwen3.5环境脚本

# 设置代理
export proxy_addr=127.0.0.1
export proxy_http_port=7333
export proxy_socks_port=7333
export http_proxy=http://$proxy_addr:$proxy_http_port
export https_proxy=http://$proxy_addr:$proxy_http_port
export all_proxy=socks5://$proxy_addr:$proxy_socks_port
export no_proxy=127.0.0.1,.huawei.com,localhost,local,.local

# 激活CANN环境
source /home/CANN/CANN8.5.0/ascend-toolkit/set_env.sh

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate verl_qwen3.5_fsdp

# 设置Python路径
export PYTHONPATH=/root/.conda/envs/verl_qwen3.5_fsdp/lib/python3.11/site-packages:$PYTHONPATH
export PATH=/root/.conda/envs/verl_qwen3.5_fsdp/bin:$PATH

echo "verl-qwen3.5环境已激活"
echo "Python版本: $(python --version)"
echo "代理设置: $http_proxy"
