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

# 设置conda环境路径
export CONDA_PREFIX=/root/.conda/envs/verl_qwen3.5_fsdp
export PATH=$CONDA_PREFIX/bin:$PATH
export PYTHONPATH=$CONDA_PREFIX/lib/python3.11/site-packages:$PYTHONPATH

# 设置Python相关环境变量
export PYTHONHOME=$CONDA_PREFIX
export PYTHONEXECUTABLE=$CONDA_PREFIX/bin/python

echo "verl-qwen3.5环境已激活"
echo "Python路径: $(which python)"
echo "Python版本: $(python --version 2>&1)"
echo "代理设置: $http_proxy"
