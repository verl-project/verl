# verl-qwen3.5 NPU环境安装指南

基于实际安装经验总结的完整安装流程和问题解决方案。

## 目录
1. [环境要求](#环境要求)
2. [安装前准备](#安装前准备)
3. [完整安装步骤](#完整安装步骤)
4. [版本兼容性说明](#版本兼容性说明)
5. [常见问题解决方案](#常见问题解决方案)
6. [验证安装](#验证安装)
7. [使用说明](#使用说明)

## 环境要求

### 硬件要求
- **NPU设备**: 华为昇腾NPU（已验证8个NPU设备可用）
- **架构**: aarch64（ARM架构）

### 软件要求
- **操作系统**: Linux
- **CANN版本**: 8.5.0.B160（路径：`/home/CANN/CANN8.5.0/ascend-toolkit`）
- **Python版本**: 3.11.14
- **Conda**: 用于环境管理

### 网络配置
- **代理设置**: 127.0.0.1:7333（如无网络需配置代理）
- **华为源**: https://mirrors.huaweicloud.com/ascend/repos/pypi

## 安装前准备

### 1. 激活CANN环境
```bash
# 激活CANN环境（必须）
source /home/CANN/CANN8.5.0/ascend-toolkit/set_env.sh
```

### 2. 创建Conda环境
```bash
# 创建Python 3.11环境
conda create -n verl_qwen3.5_fsdp python=3.11
conda activate verl_qwen3.5_fsdp
```

### 3. 配置代理和PyPI源
```bash
# 设置代理环境变量
export http_proxy=http://127.0.0.1:7333
export https_proxy=http://127.0.0.1:7333

# 配置PyPI源（包含华为源）
pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/ https://mirrors.huaweicloud.com/ascend/repos/pypi"
```

## 完整安装步骤

### 步骤1：安装基础工具
```bash
# 安装编译工具
pip install cmake ninja
```

### 步骤2：安装PyTorch和NPU支持
```bash
# 安装torch 2.8.0（必须与torch_npu版本匹配）
pip install torch==2.8.0 torchvision==0.23.0

# 安装torch_npu 2.8.0.post1（华为源只有此版本）
pip install torch_npu==2.8.0.post1 --index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi
```

**重要**: 文档要求`torch_npu==2.9.0`，但华为源只有`2.8.0.post1`，需使用`torch==2.8.0`匹配。

### 步骤3：安装numpy（解决C扩展问题）
```bash
# 安装numpy 1.26.4（verl要求<2.0.0）
pip install numpy==1.26.4
```

### 步骤4：安装ray（verl核心组件）
```bash
# 先安装grpcio 1.76.0（解决ray依赖问题）
pip install grpcio==1.76.0

# 安装ray 2.41.0（verl要求>=2.41.0）
pip install "ray[default]==2.41.0"
```

### 步骤5：安装tensordict
```bash
# 安装tensordict 0.10.0（verl要求0.8.0-0.10.0，排除0.9.0）
pip install "tensordict==0.10.0"
```

### 步骤6：安装vllm
```bash
# 克隆vllm仓库
git clone https://github.com/vllm-project/vllm.git
cd vllm

# 安装vllm（使用empty目标设备避免编译问题）
VLLM_TARGET_DEVICE=empty pip install -v .
cd ..
```

### 步骤7：安装vllm-ascend
```bash
# 克隆vllm-ascend仓库
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend

# 安装依赖
pip install -r requirements.txt

# 安装vllm-ascend（跳过自定义内核编译）
export COMPILE_CUSTOM_KERNELS=0
python setup.py install
cd ..
```

### 步骤8：安装triton-ascend
```bash
# 安装triton-ascend 3.2.0
pip install triton-ascend==3.2.0
```

### 步骤9：安装verl
```bash
# 进入verl目录
cd verl

# 以editable模式安装verl
pip install -e .
cd ..
```

### 步骤10：安装其他依赖
```bash
# 安装verl的其他依赖
pip install accelerate==1.2.0
pip install transformers  # 会自动安装兼容版本
pip install datasets pandas peft wandb tensorboard
```

## 版本兼容性说明

### 关键版本依赖关系
| 组件 | 要求版本 | 实际安装版本 | 说明 |
|------|----------|--------------|------|
| torch | 2.8.0 | 2.8.0 | 必须与torch_npu匹配 |
| torch_npu | 2.9.0（文档） | 2.8.0.post1 | 华为源只有2.8.0.post1 |
| numpy | <2.0.0 | 1.26.4 | 解决C扩展问题 |
| ray | >=2.41.0 | 2.41.0 | verl核心组件 |
| tensordict | 0.8.0-0.10.0（排除0.9.0） | 0.10.0 | 必须在此范围内 |
| transformers | <5.0.0 | 自动安装兼容版本 | 避免版本冲突 |

### 版本冲突解决方案
1. **torch_npu版本不匹配**：使用`torch==2.8.0` + `torch_npu==2.8.0.post1`
2. **numpy C扩展问题**：重新安装`numpy==1.26.4`
3. **ray导入问题**：先安装`grpcio==1.76.0`，再安装`ray[default]==2.41.0`
4. **tensordict编译问题**：安装`tensordict==0.10.0`

## 常见问题解决方案

### 问题1：torch_npu找不到合适版本
**症状**: `ERROR: No matching distribution found for torch_npu==2.9.0`
**解决方案**:
```bash
# 检查华为源可用版本
pip index versions torch_npu --index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi

# 安装可用版本（2.8.0.post1）
pip install torch_npu==2.8.0.post1 --index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi

# 安装匹配的torch版本
pip install torch==2.8.0
```

### 问题2：numpy C扩展导入错误
**症状**: `numpy导入有C扩展编译问题`
**解决方案**:
```bash
# 重新安装numpy 1.26.4
pip install --force-reinstall numpy==1.26.4
```

### 问题3：ray导入失败
**症状**: `undefined symbol: PyObject_GetOptionalAttr` 或 `No module named 'rpds.rpds'`
**解决方案**:
```bash
# 卸载现有ray
pip uninstall -y ray

# 安装grpcio 1.76.0
pip install grpcio==1.76.0

# 安装ray 2.41.0
pip install "ray[default]==2.41.0"

# 如有需要，重新安装rpds-py
pip install --force-reinstall rpds-py
```

### 问题4：tensordict编译错误
**症状**: `undefined symbol: PyThreadState_GetUnchecked`
**解决方案**:
```bash
# 安装正确版本的tensordict
pip install "tensordict==0.10.0"
```

### 问题5：vllm-ascend编译错误
**症状**: 自定义内核编译失败
**解决方案**:
```bash
# 跳过自定义内核编译
export COMPILE_CUSTOM_KERNELS=0
python setup.py install
```

## 验证安装

### 创建验证脚本
创建文件`verify_installation.py`:
```python
#!/usr/bin/env python3
"""
verl-qwen3.5 NPU环境验证脚本
"""

def verify_imports():
    """验证所有核心组件导入"""
    print("=== verl-qwen3.5 NPU环境验证 ===\n")
    
    # 验证verl
    try:
        import verl
        print("✅ verl导入成功")
        print(f"   verl版本: {verl.__version__}")
    except Exception as e:
        print(f"❌ verl导入失败: {e}")
        return False
    
    # 验证ray
    try:
        import ray
        print("✅ ray导入成功")
        print(f"   ray版本: {ray.__version__}")
    except Exception as e:
        print(f"❌ ray导入失败: {e}")
        return False
    
    # 验证torch和torch_npu
    try:
        import torch
        import torch_npu
        print("✅ torch导入成功")
        print(f"   torch版本: {torch.version.__version__}")
        print(f"   torch_npu版本: {torch_npu.__version__}")
        print(f"   NPU设备数量: {torch_npu.npu.device_count()}")
        print(f"   NPU 0可用: {torch_npu.npu.is_available()}")
        
        # 测试NPU功能
        if torch_npu.npu.is_available():
            x = torch.tensor([1.0, 2.0, 3.0])
            x_npu = x.npu()
            print(f"   ✅ 张量成功移动到NPU: {x_npu.device}")
    except Exception as e:
        print(f"❌ torch/torch_npu导入失败: {e}")
        return False
    
    # 验证其他核心组件
    components = [
        ("numpy", "numpy"),
        ("vllm", "vllm"),
        ("triton", "triton"),
        ("transformers", "transformers"),
    ]
    
    for name, module in components:
        try:
            __import__(module)
            print(f"✅ {name}导入成功")
        except Exception as e:
            print(f"⚠️  {name}导入警告: {e}")
    
    print("\n=== 环境验证完成 ===")
    return True

if __name__ == "__main__":
    success = verify_imports()
    exit(0 if success else 1)
```

### 运行验证
```bash
# 激活环境
source /home/t00906153/activate_verl_env_fixed.sh

# 运行验证脚本
python verify_installation.py
```

### 预期输出
```
=== verl-qwen3.5 NPU环境验证 ===

✅ verl导入成功
   verl版本: 0.8.0.dev
✅ ray导入成功
   ray版本: 2.41.0
✅ torch导入成功
   torch版本: 2.8.0+cpu
   torch_npu版本: 2.8.0.post1
   NPU设备数量: 8
   NPU 0可用: True
   ✅ 张量成功移动到NPU: npu:0
✅ numpy导入成功
✅ vllm导入成功
✅ triton导入成功
✅ transformers导入成功

=== 环境验证完成 ===
```

## 使用说明

### 1. 激活环境脚本
创建激活脚本`activate_verl_env.sh`:
```bash
#!/bin/bash
# verl-qwen3.5环境激活脚本

# 设置代理
export http_proxy=http://127.0.0.1:7333
export https_proxy=http://127.0.0.1:7333

# 激活CANN环境
source /home/CANN/CANN8.5.0/ascend-toolkit/set_env.sh

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate verl_qwen3.5_fsdp

# 设置环境变量
export PYTHONPATH=/home/t00906153/project/verl-qwen3.5/verl:$PYTHONPATH

echo "verl-qwen3.5环境已激活"
echo "Python路径: $(which python)"
echo "Python版本: $(python --version)"
echo "代理设置: $http_proxy"
```

### 2. 运行verl示例
```bash
# 激活环境
source activate_verl_env.sh

# 进入verl目录
cd /home/t00906153/project/verl-qwen3.5/verl

# 运行示例（根据具体示例调整）
# python examples/grpo_trainer/run_qwen2-7b_math.sh
```

### 3. 性能测试
```bash
# 测试NPU性能
python -c "
import torch
import torch_npu
import time

if torch_npu.npu.is_available():
    # 创建大张量
    size = (1024, 1024, 1024)
    print(f'测试张量大小: {size}')
    
    # CPU计算
    start = time.time()
    a_cpu = torch.randn(size)
    b_cpu = torch.randn(size)
    c_cpu = torch.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start
    print(f'CPU计算时间: {cpu_time:.2f}秒')
    
    # NPU计算
    start = time.time()
    a_npu = a_cpu.npu()
    b_npu = b_cpu.npu()
    c_npu = torch.matmul(a_npu, b_npu)
    c_npu_cpu = c_npu.cpu()
    npu_time = time.time() - start
    print(f'NPU计算时间: {npu_time:.2f}秒')
    print(f'加速比: {cpu_time/npu_time:.2f}x')
else:
    print('NPU不可用')
"
```

## 维护和更新

### 更新依赖
```bash
# 更新所有包到兼容版本
pip install --upgrade --force-reinstall \
    torch==2.8.0 \
    torch_npu==2.8.0.post1 \
    numpy==1.26.4 \
    ray[default]==2.41.0 \
    tensordict==0.10.0
```

### 清理环境
```bash
# 清理pip缓存
pip cache purge

# 重新创建conda环境（极端情况）
conda deactivate
conda env remove -n verl_qwen3.5_fsdp
conda create -n verl_qwen3.5_fsdp python=3.11
```

## 故障排除

如果遇到问题，按以下步骤排查：

1. **检查CANN环境**：确保`source /home/CANN/CANN8.5.0/ascend-toolkit/set_env.sh`已执行
2. **检查代理设置**：确保代理可用`curl -I http://127.0.0.1:7333`
3. **检查版本兼容性**：使用`pip list`查看已安装版本
4. **查看错误日志**：仔细阅读错误信息，特别是版本冲突提示
5. **逐步安装**：按照本指南步骤逐步安装，每步验证

## 总结

本指南基于实际安装经验总结，解决了以下关键问题：
1. torch和torch_npu版本匹配问题
2. numpy C扩展编译问题
3. ray导入和依赖问题
4. tensordict版本兼容性问题
5. vllm-ascend编译问题

按照本指南步骤操作，可以成功安装verl-qwen3.5 NPU环境，所有核心组件均可正常使用。

---
**文档版本**: 1.0  
**更新日期**: 2026-02-28  
**基于安装经验**: 实际安装验证通过  
**适用环境**: CANN 8.5.0.B160 + Python 3.11 + aarch64架构