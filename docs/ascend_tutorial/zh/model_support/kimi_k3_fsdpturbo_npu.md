# Kimi K3 FSDPTurbo NPU 使用指南

本文用于指导在 Ascend A3 上使用 verl + FSDPTurbo + vLLM-Ascend 运行 Kimi K3 多模态 MoE GRPO 训练。适配支持动态批量、训练权重同步和 rollout decode 图模式。

## 版本要求

| software | version |
| --- | --- |
| Python | 3.12 |
| PyTorch | 2.10.0（CPU wheel，配合 torch_npu） |
| torch_npu | 2.10.0.post7（官方 A3 Dockerfile 指定的 daily 包） |
| CANN | 9.2.1（配套 vLLM-Ascend `Dockerfile.a3`） |
| vLLM | v0.30.0，本次接口验证源码 `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| Transformers | 5.14.1（配套 vLLM-Ascend requirements） |
| Triton Ascend | 3.6.0.post2 |
| FSDPTurbo | 官方 main 基线 `1fa40ad350eb681ed413ca64bffedc166c986e0f` 及 Kimi 增量 |
| verl | 官方 main 基线 `2bcc98a9b64bdb3af5b88ac870ba0709ed709aef` 及 Kimi 增量 |
| vLLM-Ascend | 官方 main 基线 `cb629ea1925a280d57b41b48416a85097f6cdb6b` 及 Kimi 增量 |

请使用满足上述版本要求的 Ascend NPU 环境，并同时准备配套源码、Python 依赖及编译产物。源码版本与算子编译产物需要匹配。

容器还需提供与宿主驱动匹配的 `npu-smi`，可只读挂载宿主的
`/usr/local/sbin/npu-smi`。verl 使用它检查 IPC 支持；仅挂载驱动库并不能替代该工具。

main 的 AuxOutput 路由输出要求启用底层前缀缓存机制。配置
`rollout.enable_prefix_caching=False` 时，verl 会为每次生成分配独立的
`cache_salt`，保留不跨请求复用前缀的行为，同时支持完整路由记录。
权重更新后的缓存清理仍由 rollout 生命周期执行。图模式使用
`additional_config.rl_config.sleep_mode_extra_cleanup=False`；旧的顶层
`enable_sleep_mode_extra_cleanup` 字段不再受支持。
脚本通过 `additional_config.enable_flashcomm1` 和 `weight_nz_mode` 显式设置
FlashComm1 与权重格式，默认分别为开启和 `0`。兼容读取旧的
`VLLM_ASCEND_ENABLE_FLASHCOMM`、`VLLM_ASCEND_ENABLE_NZ` 启动参数；main
运行库本身已不再读取这两个旧变量。RL 权重同步必须禁用 NZ。

镜像与构建依赖以配套 vLLM-Ascend 的 `Dockerfile.a3` 和 `requirements.txt`
为准。KDA 与短卷积依赖 `flash-linear-attention-npu-a3==26.9.1`。
vLLM-Ascend 的 C++ 扩展及自定义 OPP 必须基于配套源码和 CANN 重新构建。

FSDPTurbo main 的 KDA 还需要源码依赖
[`triton-ascend-kernels`](https://gitcode.com/Ascend/triton-ascend-kernels)，
本次固定提交为 `7acb9c40a6ca827e885d902b7eaee5d45457b230`。
该模块与 Triton Ascend 编译器分别安装；运行前应确认 KDA 选择加速后端。
启动脚本默认 `VLLM_VERSION=0.30.0`，须与实际安装版本一致。

其他运行依赖包括 Ray、Hydra/OmegaConf、TransferQueue、datasets、pyarrow、pandas、NumPy、Pillow、qwen_vl_utils、tensordict、torchdata、accelerate、peft，以及 reward 使用的 mathruler、pylatexenc。其他库的精确版本尚未在本文锁定。不要直接用 verl 的 CUDA extras 覆盖这套 NPU 环境。

## 模型和脚本

| model | 本地权重 | script |
| --- | --- | --- |
| Kimi K3 top16 Geo3K SFT | `/path/to/models/kimi-k3-geo3k-sft` | [run_kimi_k3_fsdpturbo.sh](../../../../examples/ascend_extras/grpo_trainer/run_kimi_k3_fsdpturbo.sh) |


## 硬件和并行配置

以下为四机全层基线，共 64 张逻辑 NPU：

| nnodes | devices per node | FSDP | CP | EP | expert FSDP | rollout TP | rollout EP | rollout replicas |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | 16 | 64 | 1 | 16 | 4 | 16 | 16 | 4 |

global prompt batch 为 16，rollout n=4，生成 64 条训练样本，作为四机 bsz=1 的配置示例。开启动态批量后，实际 micro batch 还受序列长度及 token budget 影响，不能仅依据 `ppo_micro_batch_size_per_gpu` 判断实际批量。

Kimi 的 `gate_up_proj` 保持输入优先布局并使用 `Shard(2)`，`down_proj` 使用
`Shard(1)`。对于 `[E,3584,6144]` 和 `[E,3072,3584]` 的参数，expert FSDP
并行度必须同时整除 6144 和 3072，因此支持三机 48 卡、训练 EP=16、eFSDP=3。
同卡部署时，训练与推理 EP 不同的权重同步按专家、行、列区间直接交换分片。
separate async 使用完整的全局 HF 参数流，由接收端按自己的 TP/EP 布局加载；
训练卡的局部专家范围不能代替这个全局流。

## 模型适配和导出边界

### FSDP 原生模型

Kimi K3 的模型差异由 FSDPTurbo 原生模型持有：packed/HF 坐标转换、分片加载、
有界导出和图文展开都位于 `fsdp_turbo/models/kimi/`。
verl 使用通用 FSDP/FSDPTurbo 引擎，不注册 Kimi Engine，也不从 models 反向导入 workers。

训练显式请求 `output_raw_logits=True`。原生模型用实际图文合并产生的位置映射，
在 LM head 前选择原始 token 的 hidden states；图片占位符对应最后一个 image feature。
CP 模型先交换 hidden states 到标准 packed/SP ownership，再执行 head，避免跨卡搬运词表 logits。
原有输出对象通过 `logits_layout="packed"` 和 `logits_pad_size` 声明输出布局；输入仍可为 BSHD。
通用引擎据此复用已有 packed labels、temperature、log-prob、entropy 和 score-centering 流程。
默认 HF/SFT forward 不启用此输出契约，保留原展开序列输出；CP1 的 packed 输入也保留。

### ID-only R3（独立启用）

模型加载、前向和权重同步不依赖 R3；路由采集与回放单独配置。
R3 仅透传专家 IDs 和传输语义，模型在实际图文合并后对齐有效行，再切分 CP。
训练端使用 `actor.fsdp_config.router_replay.mode=R3`，复用现有回放配置；
`rollout.enable_rollout_routing_replay=True` 只控制推理端采集。默认 `disabled`，不支持 R2。
自然缺失的最后一个因果行继续使用当前 router 的选择，权重由当前 gate scores gather、
归一化和 scaling 得到。verl 中不保留 Kimi 展开几何或专家权重回放逻辑。

### Megatron 原生布局

Megatron 路径复用标准 GPT forward。MS-Bridge 的 Kimi 模型通过
`prepare_inputs_for_forward` 提供展开布局、有效路由行和原始 token logits 对齐；
verl 保留通用 BSHD padding、labels、temperature、logits processor 及 ID-only 回放。
Kimi 启动脚本通过已有 `model.external_lib` 加载
`mindspeed_bridge.runtime.kimi_k3_autoload`，并显式配置 `attention_backend=auto`。
该路径需要包含此布局接口的配套 MS-Bridge 版本。

### Rollout 输入与采集

rollout 使用标准 vLLM server，图像保持 PIL 输入。配套 vLLM-Ascend 的 Kimi
processor 通过原生 prompt-update 接口接受 HF processor 产生的媒体块。
脚本在 `rollout.custom.sampling_defaults.stop` 配置响应结束字符串，请求参数优先于这些
默认值；EOS 由 vLLM 从 tokenizer 和 HF generation config 读取，`ignore_eos` 只控制
EOS，显式配置的字符串 stop 仍然生效。`rollout.custom.routed_experts_source_semantics=model_rows`
显式保留展开序列的专家 IDs，未设置时保留原有逻辑 token 传输。
包含 tool mask 的多轮输出目前不支持 model-row 回放，会在输出校验时明确拒绝。

### 权重导出和同步

默认权重导出和 HF 保存逐专家生成标准 `w1/w3/w2.weight`，不聚合整块 MoE。
`naive` 同卡同步路径显式传入 rollout 拓扑，保留本地 packed 权重重分片优化；
separate async 和 HF checkpoint 使用完整全局 HF 流，不传入本地布局。
delta 的全量 seed 和后续更新共用相同的 HF 名称、转置和索引规则。
这些权重能力与 R3 的启用状态独立。

### FSDPTurbo Ulysses CP（关闭 TND）

设置 `TURBO_CP_SIZE=2` 或 `4`，启动脚本会同时配置 actor 和 ref 的
`fsdp_config.turbo_config.distributed.ulysses_parallel_size`、MLA/KDA 通信替换以及
Kimi 多模态融合后的序列切分。保持 `actor_rollout_ref.model.use_remove_padding=False`，
不要同时开启 verl 的 `ulysses_sequence_parallel_size`。

单机 16 卡示例：

```bash
NNODES=1 NPUS_PER_NODE=16 TURBO_EP_SIZE=8 TURBO_CP_SIZE=2 \
  bash examples/ascend_extras/grpo_trainer/run_kimi_k3_fsdpturbo.sh
```

CP 大小需要整除设备数和 attention/KDA 的 head 数。CP 分组内处理同一批样本，
数据并行大小为设备数除以 CP 大小；FSDP 仍覆盖全部训练设备。
本配置保持 rollout routing replay 开启。rollout 只传专家 IDs，原生模型将完整展开序列的
IDs 和有效行 mask 对齐到实际图文融合位置，再使用与隐藏状态相同的 CP padding 和切分。
训练前向按当前 router 参数重算所选专家的 gate 权重，沿用模型的归一化和 scaling；
重算复用相同的 IDs，不传输或回放采样时的 gate 权重。
模型先完成图文融合，再切分展开后的序列。原生模型在 LM head 前将 raw-token hidden states
重排到 packed/SP 分块；普通 log-prob、entropy、sum_pi_squared 和 score centering 均使用
同一通用 logits processor/标量汇总路径。词表 logits 始终留在本卡。

CPU 输出和梯度回归测试：

```bash
torchrun --nnodes=1 --nproc_per_node=4 --master_addr=127.0.0.1 --master_port=29458 \
  tests/workers/engine/test_packed_model_output_on_cpu.py
```

## 数据和模型准备

使用已准备好的 Geo3K Parquet 数据：

```bash
export BACKEND_ROOT=/path/to/kimi-k3
export MODEL_PATH=/path/to/models/kimi-k3-geo3k-sft
export TRAIN_FILE=/path/to/datasets/geo3k/train.parquet
export TEST_FILE=/path/to/datasets/geo3k/test.parquet
```

### FSDPTurbo 自动准备

FSDPTurbo 启动脚本会在后端源码根目录的 `models/fsdpturbo-<权重目录名>`
创建独立 checkpoint view。根目录由 `KIMI_BACKEND_ROOT` 指定，默认根据脚本位置推导；
其中应包含 `verl/` 和 `FSDPTurbo/`。准备脚本链接原始 safetensors，复制配置、
tokenizer 等文件，并覆盖为配套 FSDPTurbo 的 native modeling 和 processor 代码，
不修改原权重目录。只修改源码、仍加载旧 checkpoint 目录不能更新实际执行的 remote code。

原生 `KimiK3Processor` 直接支持标准 Hugging Face `text`（字符串或文本列表）和
`images` 参数，并声明自己的 `image_token`；原有 `medias/messages` API 继续保留。
verl 的通用 builder 和 AgentLoop 使用原 main 流程，没有 Kimi 转换或解码适配层。
完整媒体文本的处理、占位符计数和图片顺序由原生 processor 负责；AgentLoop 用
`skip_special_tokens=True` 重建文本时，即使媒体标记被移除，也可从保留的 PIL 图片
重新获得特征和 grid 元数据。准备数据时应检查过滤后的数据集非空。

### Megatron 显式准备

Megatron 启动脚本直接读取 `MODEL_PATH`，不会运行模型准备脚本，也不要求其
`BACKEND_ROOT` 下存在 `FSDPTurbo/`。使用独立位置的配套 FSDPTurbo 源码先生成
一个新 view，再将 `MODEL_PATH` 指向它：

```bash
export BACKEND_ROOT=/path/to/megatron-kimi-k3
export FSDPTURBO_SOURCE=/path/to/FSDPTurbo
export ORIGINAL_MODEL_PATH=/path/to/models/kimi-k3-geo3k-sft
export MODEL_PATH=/path/to/models/kimi-k3-native-view

python3 "$BACKEND_ROOT/verl/examples/ascend_extras/grpo_trainer/prepare_kimi_k3_model.py" \
  "$ORIGINAL_MODEL_PATH" "$MODEL_PATH" \
  --code "$FSDPTURBO_SOURCE/fsdp_turbo/models/kimi"
```

随后保持该 `MODEL_PATH` 运行
[run_kimi_k3_megatron.sh](../../../../examples/ascend_extras/grpo_trainer/run_kimi_k3_megatron.sh)。
view 与原目录必须不同；各节点都需要能访问 view 和其链接指向的原始权重。
准备脚本会覆盖 view 中的代码和元数据，原目录不变。

脚本同时在 view 配置中写入 `load_checkpoint_after_shard=True`。当前 verl 仅在
FSDP 加载流程读取此字段；配套 Kimi Bridge 从 `text_config` 映射其 provider 配置，
不读取该字段，因此它不会把 Megatron 切换为 FSDP 加载路径。
部署仍需满足 checkpoint remote code 的导入依赖：复制的 modeling 文件含有
`fsdp_turbo` 包导入，而 Megatron launcher 不自动把 `FSDPTURBO_SOURCE` 加入
`PYTHONPATH`。若部署中的 HF 类解析或转换需要导入这些文件，应在既有运行环境中
提供匹配的 FSDPTurbo 包；`--code` 只复制代码，不安装运行依赖。

## 启动训练

先在所有节点配置相同依赖并加载 CANN 环境。执行前将 `/path/to/...` 和 `<...>` 占位符替换为实际配置。

### 启动 Ray 集群

在主节点执行：

```bash
export MASTER_ADDR="<head-ip>"
export MASTER_PORT=6781
ray start --head --port="$MASTER_PORT" \
  --node-ip-address="$MASTER_ADDR" \
  --resources='{"NPU":16}'
```

在其余三个 worker 节点分别执行：

```bash
export MASTER_ADDR="<head-ip>"
export MASTER_PORT=6781
export NODE_IP="<current-node-ip>"
ray start --address="$MASTER_ADDR:$MASTER_PORT" \
  --node-ip-address="$NODE_IP" \
  --resources='{"NPU":16}'
```

在主节点执行 `ray status --address="$MASTER_ADDR:$MASTER_PORT"`，确认四个节点、共 64 个 NPU 资源已就绪。

### 启动 FSDPTurbo

以下命令仅在主节点执行：

```bash
export BACKEND_ROOT=/path/to/kimi-k3
export MODEL_PATH=/path/to/models/kimi-k3-geo3k-sft
export TRAIN_FILE=/path/to/datasets/geo3k/train.parquet
export TEST_FILE=/path/to/datasets/geo3k/test.parquet
export KIMI_RUN_MODE=multinode
export NNODES=4
export MASTER_ADDR="<head-ip>"
export MASTER_PORT=6781
export RAY_ADDRESS="$MASTER_ADDR:$MASTER_PORT"
export NPUS_PER_NODE=16
export TURBO_EP_SIZE=16
export ROLLOUT_TP=16
export TRAIN_BATCH_SIZE=16
export PPO_MINI_BATCH_SIZE=16
export ROLLOUT_N=4
export MAX_VISUAL_TOKENS=1024
export VERL_VLLM_MULTIMODAL_TOKEN_MARGIN=1024
export VERL_MAX_RESPONSE_LENGTH=2048
export VERL_VLLM_MAX_MODEL_LEN=4096
export VERL_VLLM_MAX_NUM_SEQS=16
export VERL_VLLM_MAX_NUM_BATCHED_TOKENS=4096
export VERL_TOTAL_TRAINING_STEPS=3
export VERL_SAVE_FREQ=-1
export VERL_VLLM_ENFORCE_EAGER=False

bash "$BACKEND_ROOT/verl/examples/ascend_extras/grpo_trainer/run_kimi_k3_fsdpturbo.sh" \
  "++ray_kwargs.ray_init.address=$RAY_ADDRESS" \
  data.max_prompt_length=1024 \
  trainer.balance_batch=True \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=4096 \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
  '++actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_memory_bytes=2147483648'
```

这是 prompt=1024、response=2048 的三步训练示例，不保存 checkpoint，也不主动开启 profiling。网络接口默认值与环境有关，迁移时应设置 `HCCL_SOCKET_IFNAME` 和 `GLOO_SOCKET_IFNAME` 为实际互通网卡。

## eFSDP 参数分片配置

`actor_rollout_ref.{actor,ref}.fsdp_config.turbo_config.distributed.ep_plan.efsdp_shard_placements` 接收 `(参数名, 维度)` 列表。
YAML/JSON用二元素数组，配置初始化时转为元组：

```yaml
distributed:
  ep_plan:
    efsdp_shard_placements:
      - [gate_up_proj, 2]
      - [down_proj, 1]
```

参数名相对于每个被eFSDP包装的专家模块，ModuleList则相对于每个成员，
可写成 `projection.weight`。维度从0开始；Kimi配置保持gate/up切列、down切行。
未配置的参数沿用二维及以上切dim1、一维切dim0；低精度all-gather仅支持dim0。
重复名称、负数和非整数维度在配置初始化时报错；不存在的参数、维度越界及
共享参数冲突在包装模型时报错。模型不再提供 `efsdp_shard_placement` 回调；
外部自定义启动配置需要显式传入这两项。

## 权重同步与上游能力复用

非量化权重同步使用 vLLM 的 `initialize_layerwise_reload` 和
`finalize_layerwise_reload`。verl 在接收 IPC bucket 时复制一次输入张量，
保证延迟到后续 bucket 才加载的权重不会被覆盖。Ascend 不再重复复制输入，
只保留 callable loader 和非 checkpoint 运行时状态的兼容处理。
训推 log-prob 差异直接使用上游指标 `training/rollout_logprobs_diff_mean`。
