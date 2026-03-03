# Ascend Backend Features Guide
==================================================================================

Last updated: 03/03/2026.

昇腾全面支持verl生态建设，本文将介绍NPU上对于verl的适配工作及后端特性支持供开发者进行参考

---

## 推理后端

当前verl支持vllm/sglang这两种主流推理后端，均可在昇腾NPU上运行。

### 1. vllm:

昇腾通过vllm-ascend插件来支持vllm推理后端，该插件是 vLLM 社区支持 Ascend 后端的推荐方法。它遵循[[RFC]](https://github.com/vllm-project/vllm/issues/11162)，提供了一个可插拔接口，将 Ascend NPU 与 vLLM 解耦。

##### 参数特性支持

| vllm参数| verl对应通用参数 | 简介|
| --- | --- | --- |
| `model_path` | `actor_rollout_ref.model.path` |模型权重文件的路径|
| `gpu_memory_utilization` | `actor_rollout_ref.rollout.gpu_memory_utilization` |用于控制每个阶段可使用的 GPU 内存量。它被指定为一个介于 0.0 和 1.0 之间的分数，其中：- 0.8 表示 GPU 总内存的 80%- 1.0 表示 GPU 总内存的 100%（不推荐，没有预留缓冲）|
| `enforce_eager`| `actor_rollout_ref.rollout.enforce_eager` |禁用图模式，verl默认为False|
| `enable_chunked_prefill`| `actor_rollout_ref.rollout.enable_chunked_prefill` | 分块预填充允许将大预填充分块成更小的块，并将它们与解码请求一起批处理。|
| `free_cache_engine`| `actor_rollout_ref.rollout.free_cache_engine`  |在部署生成阶段之后卸载 KVCache，默认值为 True。|
| `max_model_len` | `actor_rollout_ref.rollout.max_model_len` | 模型能够处理的最大序列长度。它限制了单个输入序列的最大长度 |
| `tp_size`|  `actor_rollout_ref.rollout.tensor_model_parallel_size * data_parallel_size`|TP并行度|
| `dp_size`| `actor_rollout_ref.rollout.data_parallel_size`|DP并行度|
| `ep_size`| `actor_rollout_ref.rollout.expert_parallel_size`|EP并行度|
| `node_rank`| `无，根据实际实例和卡数自动计算` |实例中的节点排序|
| `load_format`|  `actor_rollout_ref.rollout.load_format` |要加载的模型权重格式|
| `disable_log_stats`|  `actor_rollout_ref.rollout.disable_log_stats`|记录抢占请求的累积数量 |
| `nnodes `|  `无，根据实际实例和卡数自动计算` | 每个实例包含的节点数量` |
| `trust_remote_code`| `actor_rollout_ref.model.trust_remote_code`|是否允许在 Hub 上定义自定义模型，并将其写入自己的建模文件中|
| `max_num_seqs` | `actor_rollout_ref.rollout.max_num_seqs` |正在运行的请求的最大数量|
| `max_num_batched_tokens`| `actor_rollout_ref.rollout.max_num_batched_tokens` |在一次批处理（batch）中可以处理的最大总Token数|
| `skip_tokenizer_init`| `actor_rollout_ref.rollout.skip_tokenizer_init` |跳过初始化分词器并将 input_ids 传递到推理请求中|
| `enable_prefix_caching` | `actor_rollout_ref.rollout.enable_prefix_caching`|`用于启用自动前缀缓存` |
| `quantization`| `actor_rollout_ref.rollout.quantization，默认为None`|`量化方法`|
| `enforce_eager`|`actor_rollout_ref.rollout.enforce_eager`|标志用于强制使用PyTorch的eager执行模式，而非默认的图执行模式|

### 2. sglang:

对于sglang推理后端，昇腾通过直接向sglang社区进行持续建设与维护来支持相关功能。
此外在verl中使用sglang还涉及以下组件, 我们在[quick start](https://github.com/verl-project/verl/blob/main/docs/ascend_tutorial/ascend_sglang_quick_start.rst)中提供详细说明与一键安装脚本。

| 组件| 描述|
| --- | --- |
| [sgl_kernel_npu](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md) | Ascend NPU  SGL 优化推理内核集合，包括注意力机制、归一化、激活函数、LoRA 适配器等。 |
| [deepep](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md) |  DeepEP的 Ascend 实现，为MoE模型提供高度优化的专家并行 (EP) 通信内核 |

##### 参数特性支持

verl中通过rollout config管理推理后端参数使能，包含通用参数和engine_kwargs自定义传参。
以下列举在verl中常见设置的sglang特性参数，更多参数介绍请参考 [sglang社区NPU特性支持](https://docs.sglang.io/platforms/ascend_npu_support_features.html)

| sglang参数| verl对应通用参数 | 简介|
| --- | --- | --- |
| model_path | actor_rollout_ref.model.path|模型权重文件的路径|
| mem_fraction_static| actor_rollout_ref.rollout.gpu_memory_utilization |用于静态分配（模型权重和键值缓存内存池）的内存比例|
| disable_cuda_graph| actor_rollout_ref.rollout.enforce_eager|禁用图模式，verl默认为False|
| enable_memory_saver| 无，verl中默认设置为True | 允许使用 release_memory_occupation 和 resume_memory_occupation 来节省内存
| base_gpu_id| 无，根据实际实例和卡数自动计算  |用于分配每个实例上计算卡资源时的的初始ID
| gpu_id_step| 无，默认设置为1| 使用的连续计算卡ID 之间的差值
| tp_size|  actor_rollout_ref.rollout.tensor_model_parallel_size * data_parallel_size|TP并行度|
| dp_size| actor_rollout_ref.rollout.data_parallel_size|DP并行度|
| ep_size| actor_rollout_ref.rollout.expert_parallel_size|EP并行度|
| node_rank| 无，根据实际实例和卡数自动计算 |实例中的节点排序|
| load_format|  actor_rollout_ref.rollout.load_format|要加载的模型权重格式|
| dist_init_addr|  无，自动计算|用于初始化分布式后端的主机地址|
| nnodes| 无，根据实际实例和卡数自动计算|每个实例包含的节点数量|
| trust_remote_code| actor_rollout_ref.model.trust_remote_code|是否允许在 Hub 上定义自定义模型，并将其写入自己的建模文件中|
| max_running_requests| actor_rollout_ref.rollout.max_num_seqs |正在运行的请求的最大数量|
| log_level| 无，默认设置为error |日志记录器的日志级别|
| skip_tokenizer_init| actor_rollout_ref.rollout.skip_tokenizer_init |跳过初始化分词器并将 input_ids 传递到推理请求中|
| skip_server_warmup| 无，默认设置为True |跳过预热|
| quantization| actor_rollout_ref.rollout.quantization，默认为None|量化方法|
| attention_backend|actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend|attention内核,NPU应该设置为ascend|

---

## 训练后端

### 1. FSDP

昇腾通过torch_npu提供FSDP相关支持能力，当前pytorch api支持度参照[版本说明](https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/PyTorchNativeapi/docs/zh/native_apis/pytorch_2-7-1/torch-distributed-fsdp.md)。

#### FSDP1
##### 参数特性支持
| verl参数 | 简介|
| --- | --- |
| `actor_rollout_ref.actor.fsdp_config.param_offload` |是否卸载模型权重到CPU，默认值为False|
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` |是否卸载优化器状态到CPU，默认值为False|
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` |控制前向计算后的参数行为，平衡内存与通信。默认值为True：前向后重新分片参数，反向时重新全收集|
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | 每个FSDP分片组中的NPU数量；默认值-1表示自动。|
| `actor_rollout_ref.actor.fsdp_config.forward_prefetch`  |在前向计算完成前预取下一次前向传播的 all-gather，仅用于FSDP1，默认值为False|
| `actor_rollout_ref.actor.fsdp_config.use_orig_params` | FSDP是否会使用module的原始参数来初始化，仅用于FSDP1，默认值为False|
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size`|Ulysses序列并行大小|
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking`|通过分块计算熵以减少显存峰值，默认值为False|
| `actor_rollout_ref.actor.fsdp_config.entropy_checkpointing`|在训练时对熵计算启用重计算,降低显存峰值，默认值为False|
| `actor_rollout_ref.actor.fsdp_config.forward_only` |是否只进行前向计算，默认值为False|

#### FSDP2
##### 参数特性支持
| verl参数 | 简介|
| --- | --- |
| `actor_rollout_ref.actor.fsdp_config.param_offload` |是否卸载模型权重到CPU，默认值为False|
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` |是否卸载优化器状态到CPU，默认值为False|
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` |控制前向计算后的参数行为，平衡内存与通信。默认值为True：前向后重新分片参数，反向时重新全收集|
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | 每个FSDP分片组中的NPU数量；默认值-1表示自动。|
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size`|Ulysses序列并行大小|
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking`|通过分块计算熵以减少显存峰值，默认值为False|
| `actor_rollout_ref.actor.fsdp_config.entropy_checkpointing`|在训练时对熵计算启用重计算,降低显存峰值，默认值为False|
| `actor_rollout_ref.actor.fsdp_config.forward_only` |是否只进行前向计算，默认值为False|



### 2. Megatron

Megatron 是 NVIDIA 推出的一个专注于模型并行的训练框架仓库。如果一个仓库（例如 Verl）的训练后端使用了 Megatron，同时又希望在 NPU 上运行该仓库，那么就需要额外安装 MindSpeed 来提供底层支持。下文将介绍 MindSpeed 是如何实现无感替换 Megatron 中的关键组件，从而使其能够适配 NPU 的。

MindSpeed 底层的替换原理采用了 Monkey Patch 技术

* MindSpeed Moneky Patch框架

在verl里面通过`from mindspeed.megatron_adaptor import repatch  `触发patch，调用栈如下：

~~~
from mindspeed.megatron_adaptor import repatch
├── 执行 megatron_adaptor.py 模块导入
├── 导入 features_manager 模块
├── 执行 mindspeed/features_manager/__init__.py  
├── @AutoExecuteFunction 装饰器触发
├── patch_features() 自动执行
└── 进行`apply_features_pre_patches`和`apply_features_patches`操作
~~~

`Patch`类是整个patch系统的核心，实现了函数/类的动态替换

~~~python
class Patch
~~~

`parse_path`方法实现了动态模块导入和创建

~~~python
def parse_path(module_path, function_name, create_dummy)
~~~

patch系统支持多层装饰器叠加

~~~
def apply_patch(self):  
    final_patch_func = self.orig_func  
    if self.patch_func is not None:  
        final_patch_func = self.patch_func  

    # 应用所有装饰器  
    for wrapper in self.wrappers:  
        final_patch_func = wrapper(final_patch_func)
~~~

* MindSpeedPatchesManager类

`MindSpeedPatchesManager`作为全局单例管理所有patch

~~~python
class MindSpeedPatchesManager:  
    patches_info: Dict[str, Patch] = {}
~~~

* Feature集成模式

各个Feature通过继承`MindSpeedFeature`基类集成patch系统

~~~python
class MindSpeedFeature:
    """Base class for mindspeed features."""

    def __init__(self, feature_name: str, optimization_level: int = 2):
        self.feature_name = feature_name.lower().strip().replace('-', '_')
        self.optimization_level = optimization_level
        self.default_patches = self.optimization_level == 0

    def is_need_apply(self, args):
        """Check the feature is need to apply."""
        return (self.optimization_level <= args.optimization_level and getattr(args, self.feature_name, None)) \
            or self.default_patches

    def register_args(self, parser: ArgumentParser):
        """Register cli arguments to enable the feature."""
        pass

    def pre_validate_args(self, args: Namespace):
        """Validate the arguments of mindspeed before megatron args validation
        and store some arguments of the mindspeed temporarily,
        incase that megatron validate faile.
        for example:
            ```python
            origin_context_parallel_size = args.context_parallel_size
            args.context_parallel_size = 1
            ```
        """
        pass

    def validate_args(self, args: Namespace):
        """Restore the arguments of the mindspeed.

        for example:
        ```python
        args.context_parallel_size = origin_context_parallel_size
        ```
        """
        pass

    def post_validate_args(self, args: Namespace):
        """validate mindspeed arguments after megatron arguments validation."""
        pass

    def pre_register_patches(self, patch_manager: MindSpeedPatchesManager, args: Namespace):
        """Register all patch functions before import megatron"""
        pass

    def register_patches(self, patch_manager: MindSpeedPatchesManager, args: Namespace):
        """Register all patch functions the feature is related."""
        pass

    def incompatible_check(self, global_args, check_args):
        """Register all incompatible functions the feature is related."""
        if getattr(global_args, self.feature_name, None) and getattr(global_args, check_args, None):
            raise AssertionError('{} and {} are incompatible.'.format(self.feature_name, check_args))

    def dependency_check(self, global_args, check_args):
        """Register all dependency functions the feature is related."""
        if getattr(global_args, self.feature_name, None) and not getattr(global_args, check_args, None):
            raise AssertionError('{} requires {}.'.format(self.feature_name, check_args))

    @staticmethod
    def add_parser_argument_choices_value(parser, argument_name, new_choice):
        """Add a new choice value to the existing choices of a parser argument."""
        for action in parser._actions:
            exist_arg = isinstance(action, argparse.Action) and argument_name in action.option_strings
            if exist_arg and action.choices is not None and new_choice not in action.choices:
                action.choices.append(new_choice)
~~~

##### 参数特性支持
| verl参数 | 简介|
| --- | --- |
| `actor_rollout_ref.actor.megatron.optimizer_offload` |是否卸载模型优化器到CPU，默认值为False|
| `actor_rollout_ref.actor.megatron.use_mbridge` |是否使用mbridge进行权重转换|
| `actor_rollout_ref.actor.megatron.param_offload` |是否卸载模型权重到CPU，默认值为False|
| `actor_rollout_ref.actor.megatron.tensor_model_parallel_size` | 张量并行大小；默认值为1。|
| `actor_rollout_ref.actor.megatron.pipeline_model_parallel_size`  |流水并行大小，默认值为1|
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | 专家并行大小，默认值为1|
| `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size`|TP拓展EP大小，默认值为null|
| `actor_rollout_ref.actor.context_parallel_size`|序列并行大小，默认值为False|
| `actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs`|张量在发送到下一个pp stage后,输出数据被释放，降低显存峰值，默认值为False|
| `actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm` |是否使用持久化 LayerNorm，默认值为False|
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm` |是否使用持Group GEMM，默认值为False|
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype` |用于路由和专家输出加权平均的数据类型。使用 fp32 或 fp64 可以提高稳定性，尤其是在专家数量较多时，默认值为fp32|
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split` |如果设置为 True，在流水线并行的划分和放置策略中，loss 层会被视为一个标准的 Transformer 层来处理。默认为False。|
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split` |如果设置为 True，在流水线并行的划分和放置策略中，输入embedding 层会被视为一个标准的 Transformer 层来处理。默认为False。|
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity` |重新计算激活的粒度，可选项为'full', 'selective' and 'none'。其中full代表重新计算整个transformer layer，selective代表只计算transformer layer中的核心注意力部分。默认为'none'。|
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method` |该参数需将recompute_granularity设置为'full'才生效，可选项为'uniform', 'block'。默认为None。|
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers` |该参数需将recompute_granularity设置为'full'才生效，默认为None。若recompute_method设置为uniform，该参数含义为每个均匀划分的重新计算单元的transformer layers数量。例如你可以指定为--recompute_granularity full --recompute_method uniform --recompute_num_layers 4。recompute_num_layers越大，显存占用越小，计算成本越大。注意：当前进程中的模型层数需能被recompute_num_layers整除。默认为None。|
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` |是否使用分布式权重，默认值为False|
| `actor_rollout_ref.actor.megatron.dist_checkpointing_path` |分布式权重路径，默认值为null|
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn` |是否使用fa，默认值为true|
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb` |是否使用融合旋转位置编码，默认值为False|
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu` |是否使用融合swiglu，默认值为False|
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage` |第一个pipeline stage 的层数，默认值为none|
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage` |最后一个pipeline stage 的层数，默认值为none|
