# Precision Alignment

在 VeRL 框架中进行强化学习（RL）训练时，**精度对齐**是确保训练过程可复现、可调试的关键环节。

本文档总结了在 VeRL 上对NPU和GPU进行精度对齐的方法以供参考。

Last updated: 05/09/2026.

## 1. 环境与权重对齐

### 1.1 依赖版本对齐

VeRL、transformers版本需要进行强对齐，否则会直接影响精度结果。

其他关键依赖（torch、megatron、vllm）如无法进行强对齐，需优先保持一致或相近。

### 1.2 模型权重对齐

检查模型的权重和config.json文件是否完全一致


## 2. 输入数据对齐

在verl训练启动脚本中添加如下配置：

```bash
data.shuffle=False
data.validation_shuffle=False
```


## 3. 配置对齐

在NPU与GPU做精度对齐时，需检查配置是否完全对齐。包含：
1. 直接对比脚本写入配置
2. 运行过程中保存日志，收集日志打屏中的配置进行对比，可比较默认参数配置是否一致，需保证关键参数对齐


## 4. 固定确定性

### 4.1 固定随机种子

在环境中安装 `msprobe` :

```bash
pip install mindstudio-probe
```

在 worker 文件开头添加确定性函数：

```python
from msprobe.pytorch import seed_all
seed_all(mode=True)
```

### 4.2 固定通信环境变量

在多卡通信情况下：

- HCCL通信下(默认场景):
  
  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export HCCL_DETERMINISTIC="true"
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

- LCCL通信下(通过export HCCL_OP_EXPANSION_MODE="AIV"使能):

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export LCCL_DETERMINISTIC=1
  -  export ATB_LLM_LCOC_ENABLE=0
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0

在单卡无通信情况下：

  -  export CLOSE_MATMUL_K_SHIFT=1
  -  export ATB_MATMUL_SHUFFLE_K_ENABLE=0
  -  export VLLM_ENABLE_V1_MULTIPROCESSING=0



## 5. 验证训练精度

### 5.1 训练打桩

**打桩**即保留当前阶段的输入输出数据，便于从结果上对比分析。在进行精度问题排查时，需要进行打桩辅助问题定位。常见的打桩方式是直接将rollout阶段的数据直接dump下来。

**第一步：在 GPU 环境生成基准数据**

先跑一次GPU脚本，开启如下配置：

```bash
trainer.rollout_data_dir='/path/dump/data_json'
```
可以保存每步推理结果为jsonl文件。

**第二步：在 NPU 环境复现验证**

NPU上开启如下参数，复用上一步生成的序列，端到端运行：

```bash
actor_rollout_ref.rollout.skip_rollout=True \
actor_rollout_ref.rollout.skip_dump_dir="/path/dump/data_json" \
```

**第三步：对比指标**

在打桩输入相同推理结果，训练配置保持一致，并且固定随机性的情况下，比较NPU与GPU的rewards/pg_loss/grad_norm值是否存在差异。


## 6. 验证推理精度

### 6.1 resharding

在推理正式开始前，vllm会进行**dummy run**，通过推理一个 token 来评估推理时的显存占用，进而分配显存。可以在 vLLM 的 LLM 初始化时指定参数 load_format 来指定 dummy run 的权重是随机初始化的（dummy）还是真实权重（safetensors）。在 VeRL 中，通过参数 **actor_rollout_ref.rollout.load_format** 指定该参数。

当出现推理乱码现象时，如果引擎初始化方式为**load_format=dummy**，则sharding高概率存在问题，即使换成了safetensors后吐字正常，sharding也是存在问题的，需要对比前向。


### 6.2  推理结果对齐

```bash
trainer.rollout_data_dir='/path/dump/data_json'
```

保存每步推理结果为jsonl文件，可以直接打开jsonl文件快速确认整网推理结果是否乱码，用于推理精度问题定界。


在dump推理数据之前，若复现推理精度问题占用的资源较多，可以先尝试缩小推理精度问题复现成本，减少复现的规模，减少需要dump和对比的数据。在多batch、长序列的场景下，可以通过发送单batch请求，减少序列长度尝试复现。


## 7. dump对比

[精度调试工具](./precision_debugger_zh.md)，定位到问题出现的阶段之后，可以通过msprobe工具进行数据dump来细致定位。

在推理或训练过程中，模型可能出现输出偏离预期、生成异常、甚至产生 NaN/Inf 等数值不稳定问题。要定位根因，需要对模型执行路径进行精细化监控，采集中间特征、权重、激活值以及各关键层的输入输出，并记录提示词、张量 dtype、硬件配置等上下文信息。通过捕获这些核心张量及元数据，可以系统性地追踪精度退化或数值错误的来源。




