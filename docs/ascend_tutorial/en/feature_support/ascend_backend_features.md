# Ascend Backend Feature Guide
==================================================================================

Last updated: 03/03/2026.

Ascend comprehensively supports the verl ecosystem. This document introduces the adaptation work and backend feature support for verl on NPU for developers' reference.

---

## Inference Backend

verl currently supports two mainstream inference backends: vLLM and SGLang, both of which can run on Ascend NPU.

### 1. vllm:

Ascend supports the vLLM inference backend through the vllm-ascend plugin, which is the recommended method for the vLLM community to support the Ascend backend. It follows [[RFC]](https://github.com/vllm-project/vllm/issues/11162), providing a pluggable interface that decouples Ascend NPU from vLLM.

#### Parameter Feature Support

| vllm Parameter | verl Corresponding Common Parameter | Description |
| --- | --- | --- |
| `model_path` | `actor_rollout_ref.model.path` | Path to the model weights file |
| `gpu_memory_utilization` | `actor_rollout_ref.rollout.gpu_memory_utilization` | Controls the amount of GPU memory that can be used at each stage. It is specified as a fraction between 0.0 and 1.0, where: - 0.8 means 80% of total GPU memory - 1.0 means 100% of total GPU memory (not recommended, no reserved buffer) |
| `enforce_eager`| `actor_rollout_ref.rollout.enforce_eager` | Disables graph mode, verl default is False |
| `enable_chunked_prefill`| `actor_rollout_ref.rollout.enable_chunked_prefill` | Chunked prefill allows splitting large prefill chunks into smaller ones and batching them together with decode requests. |
| `free_cache_engine`| `actor_rollout_ref.rollout.free_cache_engine`  | Unloads KVCache after the deployment generation phase, default value is True. |
| `max_model_len` | `actor_rollout_ref.rollout.max_model_len` | The maximum sequence length the model can process. It limits the maximum length of a single input sequence |
| `tp_size`|  `actor_rollout_ref.rollout.tensor_model_parallel_size * data_parallel_size`| TP parallelism degree |
| `dp_size`| `actor_rollout_ref.rollout.data_parallel_size`| DP parallelism degree |
| `ep_size`| `actor_rollout_ref.rollout.expert_parallel_size`| EP parallelism degree |
| `node_rank`| `None, auto-calculated based on actual instances and card count` | Node ranking within the instance |
| `load_format`|  `actor_rollout_ref.rollout.load_format` | The model weights format to load |
| `disable_log_stats`|  `actor_rollout_ref.rollout.disable_log_stats`| Controls whether to record rollout statistics logs |
| `nnodes`|  None, auto-calculated based on actual instances and card count | Number of nodes per instance |
| `trust_remote_code`| `actor_rollout_ref.model.trust_remote_code`| Whether to allow custom models defined on the Hub and write them into your own modeling files |
| `max_num_seqs` | `actor_rollout_ref.rollout.max_num_seqs` | The maximum number of running requests |
| `max_num_batched_tokens`| `actor_rollout_ref.rollout.max_num_batched_tokens` | The maximum total number of tokens that can be processed in a single batch |
| `skip_tokenizer_init`| `actor_rollout_ref.rollout.skip_tokenizer_init` | Skip tokenizer initialization and pass input_ids into the inference request |
| `enable_prefix_caching` | `actor_rollout_ref.rollout.enable_prefix_caching` | Used to enable automatic prefix caching |
| `quantization`| `actor_rollout_ref.rollout.quantization`, default is None | `Quantization method` |

### 2. sglang:

For the sglang inference backend, Ascend supports related functionality through continuous contribution and maintenance directly to the sglang community.
Additionally, using sglang in verl involves the following components:

| Component | Description |
| --- | --- |
| [sgl_kernel_npu](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md) | A collection of optimized inference kernels for Ascend NPU SGL, including attention, normalization, activation functions, LoRA adapters, and so on |
| [deepep](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md) |  Ascend implementation of DeepEP, providing highly optimized expert parallel (EP) communication kernels for MoE models |

#### Parameter Feature Support

In verl, inference backend parameters are enabled and managed through the rollout config, including common parameters and custom parameters through engine_kwargs.
The following lists common sglang feature parameters configured in verl. For more parameter details, refer to the [sglang community NPU feature support](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu_support_features)

| sglang Parameter | verl Corresponding Common Parameter | Description |
| --- | --- | --- |
| model_path | actor_rollout_ref.model.path| Path to the model weights file |
| mem_fraction_static| actor_rollout_ref.rollout.gpu_memory_utilization | The fraction of memory used for static allocation (model weights and KV cache memory pool) |
| disable_cuda_graph| actor_rollout_ref.rollout.enforce_eager| Disables graph mode, verl default is False |
| enable_memory_saver| None, default set to True in verl | Allows using release_memory_occupation and resume_memory_occupation to save memory
| base_gpu_id| None, auto-calculated based on actual instances and card count  | The initial ID used when allocating compute card resources on each instance
| gpu_id_step| None, default set to 1| The difference between consecutive compute card IDs used
| tp_size|  actor_rollout_ref.rollout.tensor_model_parallel_size * data_parallel_size| TP parallelism degree |
| dp_size| actor_rollout_ref.rollout.data_parallel_size| DP parallelism degree |
| ep_size| actor_rollout_ref.rollout.expert_parallel_size| EP parallelism degree |
| node_rank| None, auto-calculated based on actual instances and card count | Node ranking within the instance |
| load_format|  actor_rollout_ref.rollout.load_format| The model weights format to load |
| dist_init_addr|  None, auto-calculated | The host address used to initialize the distributed backend |
| nnodes| None, auto-calculated based on actual instances and card count| Number of nodes per instance |
| trust_remote_code| actor_rollout_ref.model.trust_remote_code| Whether to allow custom models defined on the Hub and write them into your own modeling files |
| max_running_requests| actor_rollout_ref.rollout.max_num_seqs | The maximum number of running requests |
| log_level| None, default set to error | The log level of the logger |
| skip_tokenizer_init| actor_rollout_ref.rollout.skip_tokenizer_init | Skip tokenizer initialization and pass input_ids into the inference request |
| skip_server_warmup| None, default set to True | Skip warmup |
| quantization| actor_rollout_ref.rollout.quantization, default is None| Quantization method |
| attention_backend|actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend| Attention kernel, should be set to ascend on NPU |

---

## Training Backend

### 1. FSDP

Ascend provides FSDP-related support through torch_npu. For the current PyTorch API support level, refer to the [version notes](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/native_apis/pytorch_2-12-0/torch-distributed-fsdp.md).

#### FSDP1
##### Parameter Feature Support
| verl Parameter | Description |
| --- | --- |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | Whether to offload model weights to CPU, default value is False |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | Whether to offload optimizer state to CPU, default value is False |
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` | Controls parameter behavior after forward computation, balancing memory and communication. Default value is True: reshard parameters after forward, re-all-gather during backward |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | Number of NPUs in each FSDP shard group; default value -1 means automatic. |
| `actor_rollout_ref.actor.fsdp_config.forward_prefetch`  | Prefetch the all-gather for the next forward pass before the current forward computation completes, only for FSDP1, default value is False |
| `actor_rollout_ref.actor.fsdp_config.use_orig_params` | Whether FSDP initializes using the module's original parameters, only for FSDP1, default value is False |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size`| Ulysses sequence parallelism size |
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking`| Compute entropy in chunks to reduce peak memory, default value is False |
| `actor_rollout_ref.actor.entropy_from_logits_chunk_size`| Entropy computation chunk size, default value is 2048 |
| `actor_rollout_ref.actor.fsdp_config.entropy_checkpointing`| Enable recomputation for entropy calculation during training to reduce peak memory, default value is False |
| `actor_rollout_ref.actor.fsdp_config.forward_only` | Whether to perform forward computation only, default value is False |

#### FSDP2
##### Parameter Feature Support
| verl Parameter | Description |
| --- | --- |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | Whether to offload model weights to CPU, default value is False |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | Whether to offload optimizer state to CPU, default value is False |
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` | Controls parameter behavior after forward computation, balancing memory and communication. Default value is True: reshard parameters after forward, re-all-gather during backward |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | Number of NPUs in each FSDP shard group; default value -1 means automatic. |
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size`| Ulysses sequence parallelism size |
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking`| Compute entropy in chunks to reduce peak memory, default value is False |
| `actor_rollout_ref.actor.fsdp_config.entropy_checkpointing`| Enable recomputation for entropy calculation during training to reduce peak memory, default value is False |
| `actor_rollout_ref.actor.fsdp_config.forward_only` | Whether to perform forward computation only, default value is False |



### 2. Megatron

Megatron is a training framework repository launched by NVIDIA that focuses on model parallelism. If a repository (such as Verl) uses Megatron as its training backend and also wants to run on NPU, MindSpeed must be additionally installed to provide underlying support. The following describes how MindSpeed transparently replaces key components in Megatron to adapt it to NPU.

MindSpeed's underlying replacement principle uses the Monkey Patch technique

* MindSpeed Monkey Patch Framework

In verl, the patch is triggered by `from mindspeed.megatron_adaptor import repatch  `, with the call stack as follows:

~~~
from mindspeed.megatron_adaptor import repatch
├── Execute megatron_adaptor.py module import
├── Import features_manager module
├── Execute mindspeed/features_manager/__init__.py  
├── @AutoExecuteFunction decorator triggers
├── patch_features() auto-executes
└── Perform `apply_features_pre_patches` and `apply_features_patches` operations
~~~

The `Patch` class is the core of the entire patch system, implementing dynamic replacement of functions/classes

~~~python
class Patch:
~~~

The `parse_path` method implements dynamic module import and creation

~~~python
def parse_path(module_path, function_name, create_dummy):
~~~

The patch system supports stacking multiple layers of decorators

~~~python
def apply_patch(self):  
    final_patch_func = self.orig_func  
    if self.patch_func is not None:  
        final_patch_func = self.patch_func  

    # Apply all decorators  
    for wrapper in self.wrappers:  
        final_patch_func = wrapper(final_patch_func)
~~~

* MindSpeedPatchesManager Class

`MindSpeedPatchesManager` serves as a global singleton to manage all patches

~~~python
class MindSpeedPatchesManager:  
    patches_info: Dict[str, Patch] = {}
~~~

* Feature Integration Pattern

Each Feature integrates into the patch system by inheriting from the `MindSpeedFeature` base class

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
        in case that megatron validate fails.
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

#### Parameter Feature Support
| verl Parameter | Description |
| --- | --- |
| `actor_rollout_ref.actor.megatron.optimizer_offload` | Whether to offload the model optimizer to CPU, default value is False |
| `actor_rollout_ref.actor.megatron.use_mbridge` | The engine uses Megatron-Bridge to construct the model and convert HuggingFace weights; Ascend adaptation is provided by MindSpeed and MindSpeed-Bridge. Can be enabled together with `use_dist_checkpointing`. |
| `actor_rollout_ref.actor.megatron.param_offload` | Whether to offload model weights to CPU, default value is False |
| `actor_rollout_ref.actor.megatron.tensor_model_parallel_size` | Tensor parallelism size; default value is 1. |
| `actor_rollout_ref.actor.megatron.pipeline_model_parallel_size`  | Pipeline parallelism size, default value is 1 |
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | Expert parallelism size, default value is 1 |
| `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size`| TP-extended EP size, default value is null |
| `actor_rollout_ref.actor.context_parallel_size`| Sequence parallelism size, default value is 1 |
| `actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs`| After tensors are sent to the next PP stage, the output data is released to reduce peak memory, default value is False |
| `actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm` | Whether to use persistent LayerNorm, default value is False |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm` | Whether to use Group GEMM, default value is False |
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype` | The data type used for routing and the weighted average of expert outputs. Using fp32 or fp64 can improve stability, especially when there are many experts, default value is fp32 |
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split` | If set to True, in the partitioning and placement strategy of pipeline parallelism, the loss layer is treated as a standard Transformer layer. Default is False. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split` | If set to True, in the partitioning and placement strategy of pipeline parallelism, the input embedding layer is treated as a standard Transformer layer. Default is False. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity` | The granularity of activation recomputation, options are 'full', 'selective' and 'none'. 'full' means recomputing the entire transformer layer, 'selective' means only recomputing the core attention part of the transformer layer. Default is 'none'. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method` | This parameter takes effect only when recompute_granularity is set to 'full', options are 'uniform', 'block'. Default is None. |
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers` | This parameter takes effect only when recompute_granularity is set to 'full', default is None. If recompute_method is set to uniform, this parameter means the number of transformer layers in each uniformly partitioned recomputation unit. For example, you can specify --recompute_granularity full --recompute_method uniform --recompute_num_layers 4. The larger recompute_num_layers is, the smaller the memory footprint and the higher the computation cost. Note: The number of model layers in the current process must be divisible by recompute_num_layers. Default is None. |
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` | When True, the `model` slot uses Megatron `dist_checkpointing` sharding (`model/dist_ckpt/`). Independent of `use_mbridge`: both can be True simultaneously to save/load sharded checkpoints + HF export. Default False |
| `actor_rollout_ref.actor.megatron.dist_checkpointing_path` | Distributed weights path, default value is null |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn` | Whether to use Flash Attention, default value is true |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb` | Whether to use fused rotary position embedding, default value is False |
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu` | Whether to use fused swiglu, default value is False |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage` | Number of layers in the first pipeline stage, default value is none |
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage` | Number of layers in the last pipeline stage, default value is none |

### 3. VeOmni

VeOmni is a unified reinforcement learning training backend designed for efficient training of large-scale models. Built on FSDP, it provides rich parallelism strategies and optimization features, and is particularly suitable for MoE models and large-scale distributed training scenarios.

#### Parameter Feature Support

| verl Parameter | Description |
| --- | --- |
| `actor_rollout_ref.actor.veomni.param_offload` | Whether to offload model weights to CPU, default value is False |
| `actor_rollout_ref.actor.veomni.optimizer_offload` | Whether to offload optimizer state to CPU, default value is False |
| `actor_rollout_ref.actor.veomni.fsdp_size` | Number of NPUs in each FSDP shard group; default value -1 means automatic |
| `actor_rollout_ref.actor.veomni.ulysses_parallel_size` | Ulysses sequence parallelism size, default value is 1 |
| `actor_rollout_ref.actor.veomni.expert_parallel_size` | Expert parallelism size, default value is 1 |
| `actor_rollout_ref.actor.veomni.mixed_precision` | Whether to enable mixed precision training, default value is true |
| `actor_rollout_ref.actor.veomni.enable_full_shard` | Whether to enable full sharding (ZeRO-3), default value is true |
| `actor_rollout_ref.actor.veomni.forward_prefetch` | Whether to prefetch the all-gather for the next forward pass before the current forward computation completes, default value is true |
| `actor_rollout_ref.actor.veomni.attn_implementation` | Attention implementation, supports eager, sdpa, flash_attention_2, flash_attention_3, veomni_flash_attention_2_with_sp, veomni_flash_attention_3_with_sp, native-sparse, and so on |
| `actor_rollout_ref.actor.veomni.moe_implementation` | MoE implementation, supports eager or fused, default value is fused |
| `actor_rollout_ref.actor.veomni.cross_entropy_loss_implementation` | Cross entropy loss implementation, default value is eager |
| `actor_rollout_ref.actor.veomni.rms_norm_implementation` | RMSNorm implementation, default value is eager |
| `actor_rollout_ref.actor.veomni.swiglu_mlp_implementation` | SwiGLU MLP implementation, default value is eager |
| `actor_rollout_ref.actor.veomni.rotary_pos_emb_implementation` | Rotary position embedding implementation, default value is eager |
| `actor_rollout_ref.actor.veomni.load_balancing_loss_implementation` | MoE load balancing loss implementation, default value is eager |
| `actor_rollout_ref.actor.veomni.use_torch_compile` | Whether to use torch compile, default value is false |
| `actor_rollout_ref.actor.veomni.forward_only` | Whether to perform forward computation only, default value is false |
| `actor_rollout_ref.actor.veomni.enable_fsdp_offload` | Whether to enable FSDP CPU offload, default value is false |
| `actor_rollout_ref.actor.veomni.enable_reentrant` | Whether to use reentrant gradient checkpointing, default value is false |
| `actor_rollout_ref.actor.veomni.ckpt_manager` | Checkpoint manager, default value is dcp |
| `actor_rollout_ref.actor.veomni.init_device` | Model weight initialization device, supports cpu, cuda, meta, npu, default value is meta |
| `actor_rollout_ref.actor.veomni.enable_async_activation_offload` | Whether to enable async activation offload based on a dedicated stream, default value is false; the VeOmni backend only supports this mode, setting `actor_rollout_ref.model.enable_activation_offload` will raise an error directly |
| `actor_rollout_ref.actor.veomni.activation_offload_modules` | Module name matching pattern for async activation offload, default value is [] (auto-discovered from the model's `_no_split_modules`) |
| `actor_rollout_ref.actor.veomni.activation_offload_host_cache_limit_gb` | Upper limit of idle pinned host buffer (GB) retained between steps for async activation offload, default value is 4.0 |
| `actor_rollout_ref.rollout.moe_load_balance_metrics_interval` | MoE expert load metrics reporting interval on the rollout side, default value is 0 (disabled); requires `actor_rollout_ref.rollout.enable_rollout_routing_replay` to be enabled simultaneously to record routing decisions |

#### Router Replay Support

The VeOmni backend supports the Router Replay feature for MoE models, configured through `actor_rollout_ref.actor.veomni.router_replay`:

| Parameter | Description |
| --- | --- |
| `mode` | Router replay mode, supports disabled (disable), R2 (record and replay routing decisions), R3 (record and replay on the rollout side) |
| `record_file` | File path for recording routing decisions, required in R2/R3 modes |
| `replay_file` | File path for loading routing decisions for replay, required in replay mode |

#### Usage Example

The VeOmni backend is particularly suitable for GRPO training of large-scale MoE models. A typical configuration is as follows:

```bash
# Configure the VeOmni training backend
model_engine=veomni

# Configure parallel strategy
actor_rollout_ref.actor.veomni.fsdp_size=16
actor_rollout_ref.actor.veomni.ulysses_parallel_size=1
actor_rollout_ref.actor.veomni.expert_parallel_size=1

# Configure memory optimization
actor_rollout_ref.actor.veomni.param_offload=True
actor_rollout_ref.actor.veomni.optimizer_offload=True

# Configure operator implementation
actor_rollout_ref.actor.veomni.attn_implementation=veomni_flash_attention_2_with_sp
actor_rollout_ref.actor.veomni.moe_implementation=fused
```

#### Key Features

- **Efficient parallelism strategies**: Supports flexible combinations of data parallelism, Ulysses sequence parallelism, and expert parallelism
- **Memory optimization**: Supports parameter offload, optimizer offload, and activation offload, effectively reducing memory usage
- **MoE optimization**: Provides fused MoE implementation and Router Replay functionality, improving MoE model training efficiency
- **Operator optimization**: Supports multiple attention and MLP operator implementations, allowing selection of the optimal implementation based on hardware
- **Flexible deployment**: Supports NVIDIA GPU and Huawei Ascend NPU, with good cross-platform compatibility

---

## Checkpoint Engine Backend

Checkpoint Engine is a unified abstraction layer in verl used to synchronize weights between the training side and the inference side. It is mainly applied in off-policy scenarios such as the v1 trainer's separate async (training/inference separation) mode, which requires configuring a non-`naive` backend. Currently, Ascend NPU supports two backends: `nccl` (automatically recognized as HCCL) and `mooncake`.

### 1. HCCL

When configuring `actor_rollout_ref.rollout.checkpoint_engine.backend=nccl` on Ascend NPU, verl automatically recognizes and uses the HCCL backend to complete weight synchronization, without requiring additional configuration changes. This backend is based on HCCL collective communication, where the training side sends weights to each rollout worker through broadcast.

#### Parameter Feature Support

| verl Parameter | Description |
| --- | --- |
| `actor_rollout_ref.rollout.checkpoint_engine.backend` | Checkpoint Engine backend, when set to `nccl` on NPU it is automatically recognized as the HCCL backend |
| `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes` | Bucket size (MB) for a single weight transmission, default value is 2048. HCCL uses double-buffered send/receive, with NPU memory overhead of 2x the bucket size; this value must be larger than the memory footprint of the largest weight tensor in the model, and can be increased (for example to 4096) if insufficient |
| `actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.group_name` | HCCL process group name, default value is default |
| `actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.nccl.rebuild_group` | Whether to rebuild the HCCL process group on each weight update, default value is False |

### 2. Mooncake

Ascend NPU also supports configuring `actor_rollout_ref.rollout.checkpoint_engine.backend=mooncake`, which synchronizes weights in a p2p manner based on the Mooncake Transfer Engine. On NPU, the device must be specified as npu through engine_kwargs, and the Transfer Engine uses the `ascend_direct` transport protocol.

Note: The Mooncake Transfer Engine does not have an Ascend precompiled package and must be built and installed from source. Refer to [transfer-engine: ascend direct](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/design/transfer-engine/transport/ascend_direct_transport.md).

#### Parameter Feature Support

| verl Parameter | Description |
| --- | --- |
| `actor_rollout_ref.rollout.checkpoint_engine.backend` | Checkpoint Engine backend, set to `mooncake` |
| `actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes` | Bucket size (MB) for a single weight transmission, default value is 2048 |
| `actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.mooncake.device` | Checkpoint Engine running device, must be set to `npu` on NPU |
| `actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.mooncake.device_name` | Mooncake device name filter, default value is empty |

### Usage Example

A typical configuration on NPU for the v1 trainer separate async scenario is as follows:

```bash
# Configure the v1 trainer separate async mode
trainer.use_v1=True
trainer.v1.trainer_mode=separate_async

# Option 1: nccl backend (automatically recognized as HCCL on NPU)
actor_rollout_ref.rollout.checkpoint_engine.backend=nccl

# Option 2: mooncake backend (must specify device as npu on NPU)
actor_rollout_ref.rollout.checkpoint_engine.backend=mooncake
+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.mooncake.device=npu
```
