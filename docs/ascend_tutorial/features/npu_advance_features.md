# NPU 高级特性指南

> 本文档整理自 [verl-project/verl 昇腾后端特性指南](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/features/ascend_backend_features.md)，提取昇腾 NPU 在 verl 生态中的高级特性与优化能力，供开发者参考。
>
> 最后更新: 2026/03/03

---

## 目录

- [1. 推理后端高级特性](#1-推理后端高级特性)
  - [1.1 vLLM 推理后端](#11-vllm-推理后端)
  - [1.2 SGLang 推理后端](#12-sglang-推理后端)
- [2. 训练后端高级特性](#2-训练后端高级特性)
  - [2.1 FSDP 训练后端](#21-fsdp-训练后端)
  - [2.2 Megatron 训练后端](#22-megatron-训练后端)
- [3. 性能优化特性](#3-性能优化特性)
  - [3.1 内存优化](#31-内存优化)
  - [3.2 计算加速](#32-计算加速)
  - [3.3 并行策略](#33-并行策略)
- [4. 混合专家模型 (MoE) 特性](#4-混合专家模型-moe-特性)
- [5. 限制与注意事项](#5-限制与注意事项)

---

## 1. 推理后端高级特性

当前 verl 支持 vLLM 和 SGLang 两种主流推理后端，均可在昇腾 NPU 上运行。以下列出各后端支持的高级特性参数。

### 1.1 vLLM 推理后端

昇腾通过 **vllm-ascend 插件** 支持 vLLM 推理后端。该插件遵循 [RFC](https://github.com/vllm-project/vllm/issues/11162)，提供可插拔接口将 Ascend NPU 与 vLLM 解耦。

#### 高级参数配置

| vLLM 参数 | verl 对应通用参数 | 功能说明 |
|:---|:---|:---|
| `enable_chunked_prefill` | `actor_rollout_ref.rollout.enable_chunked_prefill` | **分块预填充** — 将大预填充分割为更小的块，并与解码请求一起批处理，提升吞吐 |
| `enable_prefix_caching` | `actor_rollout_ref.rollout.enable_prefix_caching` | **前缀缓存** — 自动缓存共享前缀，减少重复计算，加速多轮对话场景 |
| `quantization` | `actor_rollout_ref.rollout.quantization` | **量化支持** — 支持多种量化方法，降低显存占用与提升推理速度 |
| `free_cache_engine` | `actor_rollout_ref.rollout.free_cache_engine` | **KV Cache 动态释放** — 在生成阶段后卸载 KV Cache，默认 `True`，节省显存 |
| `enforce_eager` | `actor_rollout_ref.rollout.enforce_eager` | **执行模式切换** — 强制使用 PyTorch Eager 模式（默认图模式），verl 默认 `False` |
| `max_num_batched_tokens` | `actor_rollout_ref.rollout.max_num_batched_tokens` | **批处理 Token 上限** — 单次 batch 可处理的最大总 Token 数，用于控制 batch size |
| `max_num_seqs` | `actor_rollout_ref.rollout.max_num_seqs` | **最大并发序列数** — 同时运行的最大请求数量，控制并发度 |
| `skip_tokenizer_init` | `actor_rollout_ref.rollout.skip_tokenizer_init` | **跳过分词器初始化** — 直接将 `input_ids` 传入推理请求，减少初始化开销 |
| `trust_remote_code` | `actor_rollout_ref.model.trust_remote_code` | **远程代码信任** — 允许加载 Hub 上的自定义模型定义 |

#### 并行配置

| vLLM 参数 | verl 对应通用参数 | 功能说明 |
|:---|:---|:---|
| `tp_size` | `actor_rollout_ref.rollout.tensor_model_parallel_size * data_parallel_size` | **张量并行 (TP)** 并行度 |
| `dp_size` | `actor_rollout_ref.rollout.data_parallel_size` | **数据并行 (DP)** 并行度 |
| `ep_size` | `actor_rollout_ref.rollout.expert_parallel_size` | **专家并行 (EP)** 并行度 |
| `gpu_memory_utilization` | `actor_rollout_ref.rollout.gpu_memory_utilization` | **显存利用率控制** — 取值 0.0~1.0，建议 0.8 预留缓冲 |

---

### 1.2 SGLang 推理后端

昇腾通过向 SGLang 社区持续建设与维护来支持相关功能，涉及以下核心组件：

| 组件 | 描述 |
|:---|:---|
| [sgl_kernel_npu](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md) | Ascend NPU 优化推理内核集合，含注意力机制、归一化、激活函数、LoRA 适配器等 |
| [deepep](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md) | DeepEP 的 Ascend 实现，为 MoE 模型提供高度优化的专家并行 (EP) 通信内核 |

#### 高级参数配置

| SGLang 参数 | verl 对应通用参数 | 功能说明 |
|:---|:---|:---|
| `enable_memory_saver` | 无（verl 默认 `True`） | **内存节省模式** — 支持 `release_memory_occupation` 和 `resume_memory_occupation` 动态释放/恢复显存 |
| `attention_backend` | `actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend` | **注意力后端选择** — NPU 上应设置为 `ascend` 以调用昇腾优化内核 |
| `disable_cuda_graph` | `actor_rollout_ref.rollout.enforce_eager` | **禁用图模式** — verl 默认 `False` |
| `quantization` | `actor_rollout_ref.rollout.quantization` | **量化支持** — 支持模型量化加载与推理 |
| `skip_tokenizer_init` | `actor_rollout_ref.rollout.skip_tokenizer_init` | **跳过分词器初始化** — 直接传入 `input_ids` |
| `skip_server_warmup` | 无（verl 默认 `True`） | **跳过预热** — 加速服务启动 |
| `max_running_requests` | `actor_rollout_ref.rollout.max_num_seqs` | **最大并发请求数** |
| `mem_fraction_static` | `actor_rollout_ref.rollout.gpu_memory_utilization` | **静态内存分配比例** — 控制模型权重与 KV Cache 内存池的分配比例 |

#### 资源调度参数

| SGLang 参数 | 功能说明 |
|:---|:---|
| `base_gpu_id` | 每个实例计算卡资源的初始分配 ID（自动计算） |
| `gpu_id_step` | 连续计算卡 ID 之间的差值（默认 1） |
| `dist_init_addr` | 分布式后端初始化主机地址（自动计算） |

> 更多 SGLang NPU 特性参数请参考 [sglang 社区 NPU 特性支持文档](https://docs.sglang.io/platforms/ascend_npu_support_features.html)

---

## 2. 训练后端高级特性

### 2.1 FSDP 训练后端

昇腾通过 `torch_npu` 提供 FSDP 相关支持能力。

#### FSDP1 专属高级特性

| verl 参数 | 功能说明 |
|:---|:---|
| `actor_rollout_ref.actor.fsdp_config.forward_prefetch` | **前向预取** — 在当前前向计算完成前预取下一次前向的 all-gather，减少通信等待，默认 `False` |
| `actor_rollout_ref.actor.fsdp_config.use_orig_params` | **原始参数初始化** — FSDP 使用 module 的原始参数进行初始化，默认 `False` |

#### FSDP1/FSDP2 通用高级特性

| verl 参数 | 功能说明 |
|:---|:---|
| `actor_rollout_ref.actor.fsdp_config.param_offload` | **参数 CPU 卸载** — 将模型权重卸载到 CPU，降低显存占用，默认 `False` |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | **优化器状态 CPU 卸载** — 将优化器状态卸载到 CPU，默认 `False` |
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` | **动态重分片** — 前向计算后重新分片参数，反向时重新全收集，平衡内存与通信，默认 `True` |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | **FSDP 分片组大小** — 每个 FSDP 分片组中的 NPU 数量，`-1` 表示自动，默认 `-1` |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size` | **Ulysses 序列并行** — 序列并行大小，用于长序列训练场景 |
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking` | **分块熵计算** — 通过分块计算熵值以降低显存峰值，默认 `False` |
| `actor_rollout_ref.actor.fsdp_config.entropy_checkpointing` | **熵计算重计算** — 训练时对熵计算启用重计算 (recompute)，降低显存峰值，默认 `False` |
| `actor_rollout_ref.actor.fsdp_config.forward_only` | **纯前向模式** — 仅进行前向计算（适用于推理评估场景），默认 `False` |

---

### 2.2 Megatron 训练后端

Megatron 是 NVIDIA 推出的模型并行训练框架。在 NPU 上运行需额外安装 **MindSpeed** 提供底层支持。MindSpeed 采用 **Monkey Patch** 技术无感替换 Megatron 关键组件，实现 NPU 适配。

#### MindSpeed Monkey Patch 框架原理

**触发入口:**
```python
from mindspeed.megatron_adaptor import repatch
```

**调用链:**
```
repatch
├── 执行 megatron_adaptor.py 模块导入
├── 导入 features_manager 模块
├── 执行 mindspeed/features_manager/__init__.py
├── @AutoExecuteFunction 装饰器触发
├── patch_features() 自动执行
└── 进行 apply_features_pre_patches 和 apply_features_patches 操作
```

**核心组件:**

| 组件 | 职责 |
|:---|:---|
| `Patch` 类 | 实现函数/类的动态替换，支持多层装饰器叠加 |
| `parse_path()` | 动态模块导入和创建 |
| `MindSpeedPatchesManager` | 全局单例管理所有 patch 注册 |
| `MindSpeedFeature` | Feature 基类，各特性通过继承集成 patch 系统 |

#### Megatron 高级参数配置

##### 内存与计算优化

| verl 参数 | 功能说明 |
|:---|:---|
| `actor_rollout_ref.actor.megatron.optimizer_offload` | **优化器 CPU 卸载** — 将优化器状态卸载到 CPU，默认 `False` |
| `actor_rollout_ref.actor.megatron.param_offload` | **参数 CPU 卸载** — 将模型权重卸载到 CPU，默认 `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs` | **流水线输出释放** — 张量发送到下一 PP stage 后释放输出数据，降低显存峰值，默认 `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity` | **重计算粒度控制** — 可选 `full` / `selective` / `none`。`full` 重算整个 Transformer 层，`selective` 仅重算注意力核心部分，默认 `none` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method` | **重计算方法** — 需 `recompute_granularity=full`，可选 `uniform` / `block`，默认 `None` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers` | **重计算层数** — 需 `recompute_granularity=full`，值越大显存占用越小、计算成本越高，需能被当前进程模型层数整除 |

##### 融合算子加速

| verl 参数 | 功能说明 |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn` | **Flash Attention** — 是否使用 Flash Attention 加速注意力计算，默认 `true` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb` | **融合旋转位置编码** — 使用融合算子加速 RoPE 计算，默认 `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu` | **融合 SwiGLU** — 使用融合算子加速 SwiGLU 激活函数，默认 `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm` | **持久化 LayerNorm** — 使用持久化策略优化 LayerNorm，默认 `False` |

##### 流水线并行优化

| verl 参数 | 功能说明 |
|:---|:---|
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split` | **Loss 层流水线划分** — 将 loss 层视为标准 Transformer 层参与划分，默认 `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split` | **Embedding 层流水线划分** — 将输入 embedding 层视为标准 Transformer 层参与划分，默认 `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage` | **首 stage 层数** — 指定第一个 pipeline stage 的层数，默认 `none` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage` | **末 stage 层数** — 指定最后一个 pipeline stage 的层数，默认 `none` |

##### 并行配置

| verl 参数 | 功能说明 |
|:---|:---|
| `actor_rollout_ref.actor.megatron.tensor_model_parallel_size` | **张量并行 (TP)** 大小，默认 `1` |
| `actor_rollout_ref.actor.megatron.pipeline_model_parallel_size` | **流水线并行 (PP)** 大小，默认 `1` |
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | **专家并行 (EP)** 大小，默认 `1` |
| `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size` | **TP 拓展 EP** 大小，默认 `null` |
| `actor_rollout_ref.actor.context_parallel_size` | **上下文/序列并行** 大小，默认 `False` |

##### 权重管理

| verl 参数 | 功能说明 |
|:---|:---|
| `actor_rollout_ref.actor.megatron.use_mbridge` | **MBridge 权重转换** — 启用 mbridge 进行权重格式转换 |
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` | **分布式 checkpoint** — 使用分布式格式保存/加载权重，默认 `False` |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_path` | **分布式权重路径** — 分布式 checkpoint 加载路径，默认 `null` |

---

## 3. 性能优化特性

### 3.1 内存优化

| 特性 | 推理/训练 | 说明 |
|:---|:---|:---|
| KV Cache 动态释放 (`free_cache_engine`) | 推理 (vLLM) | 生成阶段后自动卸载 KV Cache，默认启用 |
| 内存节省模式 (`enable_memory_saver`) | 推理 (SGLang) | 支持显存动态释放/恢复，verl 默认 `True` |
| 参数 CPU 卸载 (`param_offload`) | 训练 (FSDP/Megatron) | 将模型权重卸载到 CPU |
| 优化器 CPU 卸载 (`optimizer_offload`) | 训练 (FSDP/Megatron) | 将优化器状态卸载到 CPU |
| 分块熵计算 (`entropy_from_logits_with_chunking`) | 训练 (FSDP) | 分块计算熵值降低显存峰值 |
| 熵计算重计算 (`entropy_checkpointing`) | 训练 (FSDP) | 对熵计算启用重计算 |
| 流水线输出释放 (`deallocate_pipeline_outputs`) | 训练 (Megatron) | PP 场景下释放已传递的张量 |
| 激活重计算 (`recompute_granularity`) | 训练 (Megatron) | 支持 full/selective/none 三级粒度控制 |

### 3.2 计算加速

| 特性 | 推理/训练 | 说明 |
|:---|:---|:---|
| 分块预填充 (`enable_chunked_prefill`) | 推理 (vLLM) | 大预填充分块并与解码 batch 处理 |
| 前缀缓存 (`enable_prefix_caching`) | 推理 (vLLM) | 自动缓存共享前缀，减少重复计算 |
| Flash Attention | 训练 (Megatron) | 使用 Flash Attention 加速注意力计算，默认启用 |
| 融合旋转位置编码 (`use_fused_rotary_pos_emb`) | 训练 (Megatron) | 融合算子加速 RoPE |
| 融合 SwiGLU (`use_fused_swiglu`) | 训练 (Megatron) | 融合算子加速 SwiGLU 激活函数 |
| 持久化 LayerNorm (`persist_layer_norm`) | 训练 (Megatron) | 优化 LayerNorm 执行策略 |
| Group GEMM (`moe_grouped_gemm`) | 训练 (Megatron) | MoE 场景下的 Group GEMM 优化 |

### 3.3 并行策略

| 并行类型 | vLLM | SGLang | FSDP | Megatron | 说明 |
|:---|:---|:---|:---|:---|:---|
| 数据并行 (DP) | ✅ | ✅ | ✅ | ✅ | 数据维度并行 |
| 张量并行 (TP) | ✅ | ✅ | — | ✅ | 层内张量切分 |
| 流水线并行 (PP) | — | — | — | ✅ | 层间流水线切分 |
| 专家并行 (EP) | ✅ | ✅ | — | ✅ | MoE 专家维度并行 |
| 序列并行 (SP/Ulysses) | — | — | ✅ | ✅ | 序列维度切分，支持长序列 |
| 上下文并行 (CP) | — | — | — | ✅ | 上下文并行处理 |

---

## 4. 混合专家模型 (MoE) 特性

### vLLM/SGLang 推理 MoE 支持

- **专家并行 (EP)** — 通过 `ep_size` 参数配置，将不同专家分配到不同 NPU 设备
- SGLang 通过 [deepep](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md) 提供高度优化的 EP 通信内核

### Megatron 训练 MoE 支持

| verl 参数 | 功能说明 |
|:---|:---|
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | 专家并行 (EP) 大小，默认 `1` |
| `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size` | TP 拓展 EP 大小，默认 `null` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm` | **Group GEMM** — MoE 场景下使用 Group GEMM 优化专家计算，默认 `False` |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype` | **路由数据类型** — 路由与专家输出加权平均的数据类型，可选 `fp32`/`fp64`，默认 `fp32`，提高多专家场景稳定性 |

---

## 5. 限制与注意事项

1. **mbridge 与 VPP 互斥**
   - `actor_rollout_ref.actor.megatron.use_mbridge` 与 `actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size` (VPP) **暂不支持同时开启**
   - 由于 verl 默认开启 mbridge，使用 VPP 时需手动将 `use_mbridge` 置为 `False`

2. **FSDP1 vs FSDP2 差异**
   - `forward_prefetch` 和 `use_orig_params` 仅适用于 FSDP1
   - FSDP2 为默认推荐版本，API 支持度参照 [昇腾 PyTorch 版本说明](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-7-1/torch-distributed-fsdp.md)

3. **重计算参数依赖关系**
   - `recompute_method` 需 `recompute_granularity='full'` 才生效
   - `recompute_num_layers` 需 `recompute_granularity='full'` 才生效
   - 当 `recompute_method='uniform'` 时，`recompute_num_layers` 表示每个重计算单元的 Transformer 层数，需能被当前进程模型层数整除

4. **SGLang NPU 特有配置**
   - `attention_backend` 必须设置为 `ascend` 以调用昇腾优化内核
   - `enable_memory_saver` 在 verl 中默认启用，无需额外配置

---

## 附录：参数速查表

### 推理后端参数速查

| 参数类别 | vLLM 参数 | SGLang 参数 | verl 通用参数 |
|:---|:---|:---|:---|
| 模型路径 | `model_path` | `model_path` | `actor_rollout_ref.model.path` |
| 显存控制 | `gpu_memory_utilization` | `mem_fraction_static` | `actor_rollout_ref.rollout.gpu_memory_utilization` |
| 图模式 | `enforce_eager` | `disable_cuda_graph` | `actor_rollout_ref.rollout.enforce_eager` |
| 量化 | `quantization` | `quantization` | `actor_rollout_ref.rollout.quantization` |
| 最大序列长度 | `max_model_len` | — | `actor_rollout_ref.rollout.max_model_len` |
| 最大并发数 | `max_num_seqs` | `max_running_requests` | `actor_rollout_ref.rollout.max_num_seqs` |
| 分词器 | `skip_tokenizer_init` | `skip_tokenizer_init` | `actor_rollout_ref.rollout.skip_tokenizer_init` |
| 远程代码 | `trust_remote_code` | `trust_remote_code` | `actor_rollout_ref.model.trust_remote_code` |
| TP 并行 | `tp_size` | `tp_size` | `actor_rollout_ref.rollout.tensor_model_parallel_size` |
| DP 并行 | `dp_size` | `dp_size` | `actor_rollout_ref.rollout.data_parallel_size` |
| EP 并行 | `ep_size` | `ep_size` | `actor_rollout_ref.rollout.expert_parallel_size` |

### 训练后端参数速查

| 参数类别 | FSDP 参数 | Megatron 参数 |
|:---|:---|:---|
| 参数卸载 | `fsdp_config.param_offload` | `megatron.param_offload` |
| 优化器卸载 | `fsdp_config.optimizer_offload` | `megatron.optimizer_offload` |
| 序列并行 | `ulysses_sequence_parallel_size` | `context_parallel_size` |
| Flash Attention | — | `override_transformer_config.use_flash_attn` |
| 重计算粒度 | — | `override_transformer_config.recompute_granularity` |
| 分布式 Checkpoint | — | `use_dist_checkpointing` |
