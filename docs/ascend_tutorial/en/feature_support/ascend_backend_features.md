# Ascend backend features guide
==================================================================================

Last updated: 03/03/2026.

Ascend fully supports the verl ecosystem development. This document introduces the adaptation work and backend feature support for verl on NPUs for developer reference.

---

## Inference Backend

Currently, verl supports two mainstream inference backends: vllm and sglang. Both can run on Ascend NPUs.

### 1. vllm:

Ascend supports the vLLM inference backend through the vllm-ascend plugin. This plugin is the recommended method for the vLLM community to support the Ascend backend. It follows [[RFC]](https://github.com/vllm-project/vllm/issues/11162), providing a pluggable interface to decouple the Ascend NPU from vLLM.

#### Parameter Feature Support

| vllm parameter | Corresponding verl general parameter | Description |
| --- | --- | --- |
| `model_path` | `actor_rollout_ref.model.path` | Path to the model weight file |
| `gpu_memory_utilization` | `actor_rollout_ref.rollout.gpu_memory_utilization` | Controls the amount of GPU memory available for each stage. It is specified as a fraction between 0.0 and 1.0, where: - 0.8 indicates 80% of the total GPU memory - 1.0 indicates 100% of the total GPU memory (not recommended, no buffer reserved) |
| `enforce_eager` | `actor_rollout_ref.rollout.enforce_eager` | Disables graph mode. The default value in verl is False. |
| `enable_chunked_prefill` | `actor_rollout_ref.rollout.enable_chunked_prefill` | Chunked prefill allows splitting a large prefill into smaller chunks and batching them with decoding requests. |
| `free_cache_engine` | `actor_rollout_ref.rollout.free_cache_engine` | Unloads the KVCache after the deployment generation phase. The default value is True. |
| `max_model_len` | `actor_rollout_ref.rollout.max_model_len` | The maximum sequence length that the model can process. It limits the maximum length of a single input sequence. |
| `tp_size` | `actor_rollout_ref.rollout.tensor_model_parallel_size * data_parallel_size` | TP parallelism degree |
| `dp_size` | `actor_rollout_ref.rollout.data_parallel_size` | DP parallelism degree |
| `ep_size` | `actor_rollout_ref.rollout.expert_parallel_size` | EP parallelism degree |
| `node_rank` | `None, automatically calculated based on the actual number of instances and devices` | Node rank in the instance |
| `load_format` | `actor_rollout_ref.rollout.load_format` | Format of the model weights to load |
| `disable_log_stats` | `actor_rollout_ref.rollout.disable_log_stats` | Controls whether to log rollout statistics |
| `nnodes` | None, automatically calculated based on the actual number of instances and devices | Number of nodes in each instance |
| `trust_remote_code` | `actor_rollout_ref.model.trust_remote_code` | Whether to allow defining custom models on the Hub and writing them into your own modeling files |
| `max_num_seqs` | `actor_rollout_ref.rollout.max_num_seqs` | Maximum number of running requests |
| `max_num_batched_tokens` | `actor_rollout_ref.rollout.max_num_batched_tokens` | Maximum total number of tokens that can be processed in a single batch |
| `skip_tokenizer_init` | `actor_rollout_ref.rollout.skip_tokenizer_init` | Skips initializing the tokenizer and passes input_ids to the inference request |
| `enable_prefix_caching` | `actor_rollout_ref.rollout.enable_prefix_caching` | Enables automatic prefix caching |
| `quantization` | `actor_rollout_ref.rollout.quantization`, default is None | `Quantization method` |

### 2. sglang:

For the sglang inference backend, Ascend supports related features through continuous development and maintenance directly in the sglang community.
In addition, using sglang in verl involves the following components:

| Component| Description|
| --- | --- |
| [sgl_kernel_npu](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/sgl_kernel_npu/README.md) | A collection of SGL-optimized inference kernels for Ascend NPU, including attention mechanisms, normalization, activation functions, LoRA adapters, and so on. |
| [deepep](https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md) | The Ascend implementation of DeepEP, providing highly optimized expert parallelism (EP) communication kernels for MoE models |

#### Parameter Feature Support

In verl, you manage the inference backend parameters through the rollout config. This includes general parameters and custom parameters passed using `engine_kwargs`.
The following lists commonly set sglang feature parameters in verl. For more information about parameters, refer to [sglang community NPU feature support](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu_support_features).

| sglang parameter | Corresponding verl parameter | Description |
| --- | --- | --- |
| model_path | actor_rollout_ref.model.path | Path to the model weight file |
| mem_fraction_static | actor_rollout_ref.rollout.gpu_memory_utilization | Proportion of memory used for static allocation (model weights and key-value cache memory pool) |
| disable_cuda_graph | actor_rollout_ref.rollout.enforce_eager | Disables graph mode. The default value in verl is False. |
| enable_memory_saver | None. The default value in verl is set to True. | Allows using release_memory_occupation and resume_memory_occupation to save memory
| base_gpu_id | None. Automatically calculated based on the actual instances and number of devices. | Initial ID used for allocating compute device resources on each instance
| gpu_id_step | None. The default value is set to 1. | Difference between the used consecutive compute device IDs
| tp_size | actor_rollout_ref.rollout.tensor_model_parallel_size * data_parallel_size | TP parallelism degree |
| dp_size | actor_rollout_ref.rollout.data_parallel_size | DP parallelism degree |
| ep_size | actor_rollout_ref.rollout.expert_parallel_size | EP parallelism degree |
| node_rank | None. Automatically calculated based on the actual instances and number of devices. | Node ranking in the instance |
| load_format | actor_rollout_ref.rollout.load_format | Model weight format to load |
| dist_init_addr | None. Automatically calculated. | Host address used to initialize the distributed backend |
| nnodes | None. Automatically calculated based on the actual instances and number of devices. | Number of nodes contained in each instance |
| trust_remote_code | actor_rollout_ref.model.trust_remote_code | Whether to allow defining custom models on the Hub and writing them into your own modeling files |
| max_running_requests | actor_rollout_ref.rollout.max_num_seqs | Maximum number of running requests |
| log_level | None. The default value is set to error. | Log level of the logger |
| skip_tokenizer_init | actor_rollout_ref.rollout.skip_tokenizer_init | Skips initializing the tokenizer and passes input_ids to the inference request |
| skip_server_warmup | None. The default value is set to True. | Skips warmup |
| quantization | actor_rollout_ref.rollout.quantization. The default value is None. | Quantization method |
| attention_backend | actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend | Attention kernel. For NPU, it should be set to ascend. |

---

## Training Backend

### 1. FSDP

Ascend provides FSDP support capabilities through torch_npu. For the current PyTorch API support status, refer to the [Release Notes](https://www.hiascend.com/document/detail/en/Pytorch/latest/apiref/nativeapi/docs/en/native_apis/pytorch_2-12-0/torch-distributed-fsdp.md).

#### FSDP1
##### Parameter Feature Support
| verl parameter | Description |
| --- | --- |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | Whether to offload model weights to the CPU. The default value is False. |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | Whether to offload optimizer states to the CPU. The default value is False. |
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` | Controls the parameter behavior after the forward computation to balance memory and communication. The default value is True: parameters are resharded after the forward pass, and all-gather is performed again during the backward pass. |
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | The number of NPUs in each FSDP shard group. The default value -1 indicates automatic configuration. |

| `actor_rollout_ref.actor.fsdp_config.forward_prefetch`  |Prefetches the all-gather for the next forward pass before the current forward computation completes. Applies only to FSDP1. The default value is False.|
| `actor_rollout_ref.actor.fsdp_config.use_orig_params` | Whether FSDP uses the original parameters of the module for initialization. Applies only to FSDP1. The default value is False.|
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size`|Ulysses sequence parallelism size|
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking`|Computes entropy in chunks to reduce peak device memory. The default value is False.|
| `actor_rollout_ref.actor.entropy_from_logits_chunk_size`|Chunk size for entropy computation. The default value is 2048.|
| `actor_rollout_ref.actor.fsdp_config.entropy_checkpointing`|Enables recomputation for entropy computation during training to reduce peak device memory. The default value is False.|
| `actor_rollout_ref.actor.fsdp_config.forward_only` |Whether to perform forward computation only. The default value is False.|

#### FSDP2
##### Parameter Feature Support
| verl parameter | Description |
| --- | --- |
| `actor_rollout_ref.actor.fsdp_config.param_offload` |Whether to offload model weights to CPU. The default value is False.|
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` |Whether to offload optimizer states to CPU. The default value is False.|
| `actor_rollout_ref.actor.fsdp_config.reshard_after_forward` |Controls the parameter behavior after forward computation to balance memory and communication. The default value is True: reshard parameters after forward, and re-all-gather during backward.|
| `actor_rollout_ref.actor.fsdp_config.fsdp_size` | The number of NPUs in each FSDP sharding group. The default value -1 indicates automatic.|
| `actor_rollout_ref.actor.ulysses_sequence_parallel_size`|Ulysses sequence parallelism size|
| `actor_rollout_ref.actor.entropy_from_logits_with_chunking`|Reduces device memory peak by computing entropy in chunks. The default value is False.|
| `actor_rollout_ref.actor.fsdp_config.entropy_checkpointing`|Enables recomputation for entropy calculation during training to reduce device memory peak. The default value is False.|
| `actor_rollout_ref.actor.fsdp_config.forward_only` |Whether to perform forward computation only. The default value is False.|



### 2. Megatron

Megatron is a training framework repository introduced by NVIDIA that focuses on model parallelism. If a repository (for example, Verl) uses Megatron as its training backend and you want to run it on an NPU, you must also install MindSpeed. MindSpeed provides the underlying support. The following section describes how MindSpeed transparently replaces key components in Megatron to adapt it to the NPU.

MindSpeed uses the Monkey Patch technique for its underlying replacement mechanism.

* MindSpeed Monkey Patch framework

In verl, you trigger the patch by running `from mindspeed.megatron_adaptor import repatch  `. The call stack is as follows:

~~~
from mindspeed.megatron_adaptor import repatch
├── Execute the megatron_adaptor.py module import
├── Import the features_manager module
├── Execute mindspeed/features_manager/__init__.py  
├── The @AutoExecuteFunction decorator is triggered
├── patch_features() is automatically executed
└── Perform the `apply_features_pre_patches` and `apply_features_patches` operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The `Patch` class is the core of the entire patch system and implements dynamic replacement of functions and classes.

~~~python
class Patch:
~~~~~~~~~~~~

The `parse_path` method implements dynamic module import and creation.

~~~python
def parse_path(module_path, function_name, create_dummy):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The patch system supports stacking multiple layers of decorators.

~~~python
def apply_patch(self):  
    final_patch_func = self.orig_func  
    if self.patch_func is not None:  
        final_patch_func = self.patch_func  

    # Apply all decorators  
    for wrapper in self.wrappers:  
        final_patch_func = wrapper(final_patch_func)
~~~

* MindSpeedPatchesManager class

`MindSpeedPatchesManager` manages all patches as a global singleton.

~~~python
class MindSpeedPatchesManager:  
    patches_info: Dict[str, Patch] = {}
~~~

* Feature integration mode

Each feature integrates into the patch system by inheriting the `MindSpeedFeature` base class.

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
| verl Parameter | Description|
| --- | --- |
| `actor_rollout_ref.actor.megatron.optimizer_offload` |Whether to offload the model optimizer to CPU. The default value is False.|
| `actor_rollout_ref.actor.megatron.use_mbridge` |Whether to enable mbridge. When set to True (the default), the engine constructs a `bridge` and passes it to the checkpoint manager, enabling read and write access to `model/huggingface/`. When `save_contents` or `load_contents` includes `hf_model`, the manager requires a non-null `bridge` (typically meaning this option is True). This option can be enabled together with `use_dist_checkpointing` to write both the HF tree and `model/dist_ckpt/` shards in the same checkpoint. When set to False, there is generally no `hf_model`. If only the `model` slot uses `dist_checkpointing`, set `use_dist_checkpointing=True` as well.|
| `actor_rollout_ref.actor.megatron.param_offload` |Whether to offload model weights to CPU. The default value is False.|
| `actor_rollout_ref.actor.megatron.tensor_model_parallel_size` | Tensor parallelism size. The default value is 1.|
| `actor_rollout_ref.actor.megatron.pipeline_model_parallel_size`  |Pipeline parallelism size. The default value is 1.|
| `actor_rollout_ref.actor.megatron.expert_model_parallel_size` | Expert parallelism size. The default value is 1.|
| `actor_rollout_ref.actor.megatron.expert_tensor_parallel_size`|TP-extended EP size. The default value is null.|
| `actor_rollout_ref.actor.context_parallel_size`|Sequence parallelism size. The default value is 1.|
| `actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs`|After the tensor is sent to the next pipeline stage, the output data is released to reduce the device memory peak. The default value is False.|
| `actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm` |Whether to use persistent LayerNorm. The default value is False.|
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm` |Whether to use Group GEMM. The default value is False.|
| `actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype` |The data type used for routing and weighted average of expert outputs. Using fp32 or fp64 can improve stability, especially when the number of experts is large. The default value is fp32.|
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_loss_in_pipeline_split` |If set to True, the loss layer is treated as a standard Transformer layer in the pipeline parallelism partitioning and placement strategy. The default value is False.|
| `actor_rollout_ref.actor.megatron.override_transformer_config.account_for_embedding_in_pipeline_split` |If set to True, the input embedding layer is treated as a standard Transformer layer in the pipeline parallelism partitioning and placement strategy. The default value is False.|
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity` |The granularity for recomputing activations. Available options are 'full', 'selective', and 'none'. The 'full' option recomputes the entire transformer layer, and 'selective' recomputes only the core attention part of the transformer layer. The default value is 'none'.|
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method` |This parameter takes effect only when recompute_granularity is set to 'full'. Available options are 'uniform' and 'block'. The default value is None.|
| `actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers` |This parameter takes effect only when recompute_granularity is set to 'full'. The default value is None. If recompute_method is set to 'uniform', this parameter specifies the number of transformer layers in each uniformly partitioned recomputation unit. For example, you can specify --recompute_granularity full --recompute_method uniform --recompute_num_layers 4. A larger recompute_num_layers value reduces device memory usage but increases computation cost. Note: The number of model layers in the current process must be divisible by recompute_num_layers. The default value is None.|
| `actor_rollout_ref.actor.megatron.use_dist_checkpointing` |When set to True, the `model` slot uses Megatron `dist_checkpointing` shards (`model/dist_ckpt/`). This is independent of `use_mbridge`: both can be set to True simultaneously to save and load shards plus HF exports. The default value is False.|
| `actor_rollout_ref.actor.megatron.dist_checkpointing_path` |Distributed weights path. The default value is null.|
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn` |Whether to use Flash Attention. The default value is true.|
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb` |Whether to use fused rotary position embedding. The default value is False.|
| `actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu` |Whether to use fused SwiGLU. The default value is False.|
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage` |The number of layers in the first pipeline stage. The default value is none.|
| `actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage` |The number of layers in the last pipeline stage. The default value is none.|

Note: You cannot currently enable `actor_rollout_ref.actor.megatron.use_mbridge` and `actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size` (VPP) at the same time. Because verl enables mbridge by default, manually set `actor_rollout_ref.actor.megatron.use_mbridge` to False when you use the VPP parameter.

### 3. VeOmni

VeOmni is a unified reinforcement learning training backend designed for the efficient training of large-scale models. It is built on FSDP and provides a variety of parallel strategies and optimization features. It is particularly suitable for MoE models and large-scale distributed training scenarios.

#### Parameter Feature Support

| verl parameter | Description |
| --- | --- |
| `actor_rollout_ref.actor.veomni.param_offload` | Determines whether to offload model weights to the CPU. The default value is False. |
| `actor_rollout_ref.actor.veomni.optimizer_offload` | Determines whether to offload optimizer states to the CPU. The default value is False. |
| `actor_rollout_ref.actor.veomni.fsdp_size` | The number of NPUs in each FSDP shard group. The default value -1 indicates automatic configuration. |
| `actor_rollout_ref.actor.veomni.ulysses_parallel_size` | The Ulysses sequence parallelism size. The default value is 1. |
| `actor_rollout_ref.actor.veomni.expert_parallel_size` | The expert parallelism size. The default value is 1. |
| `actor_rollout_ref.actor.veomni.mixed_precision` | Determines whether to enable mixed precision training. The default value is true. |
| `actor_rollout_ref.actor.veomni.enable_full_shard` | Determines whether to enable full sharding (ZeRO-3). The default value is true. |
| `actor_rollout_ref.actor.veomni.forward_prefetch` | Determines whether to prefetch the all-gather for the next forward pass before the current forward computation completes. The default value is true. |
| `actor_rollout_ref.actor.veomni.attn_implementation` | The attention implementation method. Supported values include eager, sdpa, flash_attention_2, flash_attention_3, veomni_flash_attention_2_with_sp, veomni_flash_attention_3_with_sp, native-sparse, and so on. |
| `actor_rollout_ref.actor.veomni.moe_implementation` | The MoE implementation method. Supported values are eager or fused. The default value is fused. |
| `actor_rollout_ref.actor.veomni.cross_entropy_loss_implementation` | The cross-entropy loss implementation. The default value is eager. |
| `actor_rollout_ref.actor.veomni.rms_norm_implementation` | The RMSNorm implementation. The default value is eager. |
| `actor_rollout_ref.actor.veomni.swiglu_mlp_implementation` | The SwiGLU MLP implementation. The default value is eager. |
| `actor_rollout_ref.actor.veomni.rotary_pos_emb_implementation` | The rotary position embedding implementation. The default value is eager. |
| `actor_rollout_ref.actor.veomni.load_balancing_loss_implementation` | The MoE load balancing loss implementation. The default value is eager. |
| `actor_rollout_ref.actor.veomni.use_torch_compile` | Determines whether to use torch compile. The default value is false. |
| `actor_rollout_ref.actor.veomni.forward_only` | Determines whether to perform only forward computation. The default value is false. |
| `actor_rollout_ref.actor.veomni.enable_fsdp_offload` | Determines whether to enable CPU offloading for FSDP. The default value is false. |
| `actor_rollout_ref.actor.veomni.enable_reentrant` | Determines whether to use reentrant gradient checkpointing. The default value is false. |
| `actor_rollout_ref.actor.veomni.ckpt_manager` | The checkpoint manager. The default value is dcp. |
| `actor_rollout_ref.actor.veomni.init_device` | The device for initializing model weights. Supported values are cpu, cuda, meta, and npu. The default value is meta. |
| `actor_rollout_ref.actor.veomni.activation_gpu_limit` | The activation device memory limit allowed to be retained on the GPU during activation offloading (in GB). The default value is 0.0. |
| `actor_rollout_ref.rollout.moe_load_balance_metrics_interval` | The interval for reporting MoE expert load metrics on the rollout side. The default value is 0 (disabled). You must also enable `actor_rollout_ref.rollout.enable_rollout_routing_replay` to record routing decisions. |

#### Router Replay Support

The VeOmni backend supports the Router Replay feature for MoE models. You can configure it using `actor_rollout_ref.actor.veomni.router_replay`:

| Parameter | Description |
| --- | --- |
| `mode` | Router replay mode. Supported values are disabled, R2 (records and replays routing decisions), and R3 (records and replays on the rollout side).|
| `record_file` | The file path for recording routing decisions. Required in R2/R3 mode.|
| `replay_file` | The file path for loading routing decisions for replay. Required in replay mode.|

#### Usage examples

The VeOmni backend is particularly suitable for GRPO training of large-scale MoE models. A typical configuration is as follows:

```bash
# Set the VeOmni training backend
model_engine=veomni

# Configure the parallelism strategy
actor_rollout_ref.actor.veomni.fsdp_size=16
actor_rollout_ref.actor.veomni.ulysses_parallel_size=1
actor_rollout_ref.actor.veomni.expert_parallel_size=1

# Configure memory optimization
actor_rollout_ref.actor.veomni.param_offload=True
actor_rollout_ref.actor.veomni.optimizer_offload=True

# Configure the operator implementation
actor_rollout_ref.actor.veomni.attn_implementation=veomni_flash_attention_2_with_sp
actor_rollout_ref.actor.veomni.moe_implementation=fused
```

#### Key Features

- **Efficient parallelism strategies**: Supports flexible combinations of data parallelism, Ulysses sequence parallelism, and expert parallelism
- **Memory optimization**: Supports parameter offloading, optimizer offloading, and activation offloading to effectively reduce device memory usage
- **MoE optimization**: Provides a fused MoE implementation and the Router Replay feature to improve MoE model training efficiency
- **Operator optimization**: Supports multiple attention and MLP operator implementations, allowing you to select the optimal implementation based on the hardware
- **Flexible deployment**: Supports NVIDIA GPUs and Huawei Ascend NPUs, providing good cross-platform compatibility
