# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig

from ...utils.profiler import ProfilerConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = [
    "FSDPEngineConfig",
    "McoreEngineConfig",
    "TrainingWorkerConfig",
    "TorchtitanEngineConfig",
    "VeOmniEngineConfig",
    "AutomodelEngineConfig",
    "EngineConfig",
    "EngineRouterReplayConfig",
]


# TODO: rename to RouterReplayConfig after removing the legacy implementation
@dataclass
class EngineRouterReplayConfig(BaseConfig):
    """Configuration for router replay in MoE models.

    This configuration controls the routing behavior for Mixture of Experts (MoE) models,
    allowing for deterministic training through route recording and replay.

    Args:
        mode (str): Router replay mode. Options: 'disabled', 'R2', 'R3'.
            - 'disabled': No router replay functionality
            - 'R2': Use Router Replay routing strategy
            - 'R3': Use Rollout Router Replay routing strategy
        record_file (Optional[str]): File path to save recorded routing decisions.
            Required when mode is 'record', 'R2', or 'R3'.
        replay_file (Optional[str]): File path to load recorded routing decisions for replay.
            Required when mode is 'replay'.
    """

    mode: str = "disabled"
    record_file: Optional[str] = None
    replay_file: Optional[str] = None

    def __post_init__(self):
        """Validate router replay configuration."""
        valid_modes = ["disabled", "R2", "R3"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid router_replay mode: {self.mode}. Must be one of {valid_modes}")


@dataclass
class EngineConfig(BaseConfig):
    _mutable_fields = BaseConfig._mutable_fields | {
        "use_dynamic_bsz",
        "max_token_len_per_gpu",
        "micro_batch_size_per_gpu",
        "infer_max_token_len_per_gpu",
        "infer_micro_batch_size_per_gpu",
        "use_fused_kernels",
        "use_remove_padding",
        "forward_only",
        "param_offload",
    }
    # whether to offload param
    param_offload: bool = False
    # whether to offload optimizer
    optimizer_offload: bool = False
    # whether to offload grad
    grad_offload: bool = False
    # whether the engine is forward only (e.g., ref policy)
    forward_only: bool = False
    # the strategy (backend)
    strategy: str = None
    # model dtype
    dtype: str = "bfloat16"  # ["bfloat16", "float16"]
    # whether to use dynamic bsz
    use_dynamic_bsz: bool = True
    # for training
    max_token_len_per_gpu: int = None
    micro_batch_size_per_gpu: int = None
    # for inference
    infer_max_token_len_per_gpu: int = None
    infer_micro_batch_size_per_gpu: int = None
    # whether use fuse lm head kernel
    use_fused_kernels: bool = False
    # TODO (this may conflict with the one in model config)
    use_remove_padding: bool = True

    seed: int = 42

    full_determinism: bool = False
    router_replay: EngineRouterReplayConfig = field(default_factory=EngineRouterReplayConfig)

    def __post_init__(self):
        pass
        # TODO: turn on this check after we reorg config
        # if self.use_dynamic_bsz:
        #     assert self.max_token_len_per_gpu is not None
        # else:
        #     assert self.micro_batch_size_per_gpu is not None


@dataclass
class McoreEngineConfig(EngineConfig):
    """Configuration for Megatron parallelism.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        param_offload (bool): Whether to offload parameters to CPU.
        grad_offload (bool): Whether to offload gradients to CPU.
        optimizer_offload (bool): Whether to offload optimizer states to CPU.
        tensor_model_parallel_size (int): Tensor model parallel size.
        expert_model_parallel_size (int): Expert model parallel size for MoE models.
        expert_tensor_parallel_size (Optional[int]): Expert tensor parallel size for MoE models.
        pipeline_model_parallel_size (int): Pipeline model parallel size.
        virtual_pipeline_model_parallel_size (Optional[int]): Virtual pipeline model parallel size
            for interleaved scheduling.
        context_parallel_size (int): Context parallel size for long sequences.
        sequence_parallel (bool): Whether to enable sequence parallelism.
        use_distributed_optimizer (bool): Whether to use distributed optimizer.
        use_dist_checkpointing (bool): Whether to use distributed checkpointing.
        dist_checkpointing_path (Optional[str]): Path for distributed checkpointing.
        dist_ckpt_optim_fully_reshardable (bool): Use fully reshardable optimizer checkpoints.
        distrib_optim_fully_reshardable_mem_efficient (bool): Use memory-efficient fully reshardable format.
        seed (int): Random seed for reproducibility.
        override_ddp_config (dict[str, Any]): Override configuration for DDP.
        override_transformer_config (dict[str, Any]): Override configuration for transformer.
        use_mbridge (bool): Whether to use MBridge for communication.
        dtype (str): Mixed precision training param dtype, default "bfloat16"
    """

    # sequence_parallel is not listed as a frozen field for auto-correction purpose
    _mutable_fields = EngineConfig._mutable_fields | {"sequence_parallel"}
    # mcore parallelism
    tensor_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: Optional[int] = None
    pipeline_model_parallel_size: int = 1
    virtual_pipeline_model_parallel_size: Optional[int] = None
    context_parallel_size: int = 1
    sequence_parallel: bool = True
    use_distributed_optimizer: bool = True
    use_dist_checkpointing: bool = False
    dist_checkpointing_path: Optional[str] = None
    dist_checkpointing_prefix: str = ""
    dist_ckpt_optim_fully_reshardable: bool = False
    distrib_optim_fully_reshardable_mem_efficient: bool = False
    override_ddp_config: dict[str, Any] = field(default_factory=dict)
    override_transformer_config: dict[str, Any] = field(default_factory=dict)
    override_mcore_model_config: dict[str, Any] = field(default_factory=dict)
    use_mbridge: bool = True
    vanilla_mbridge: bool = True
    strategy: str = "megatron"

    def __post_init__(self) -> None:
        super().__post_init__()
        """config validation logics go here"""
        assert self.strategy == "megatron"
        assert self.dtype in ["bfloat16", "float16"], f"dtype {self.dtype} not supported"
        if self.tensor_model_parallel_size == 1:
            warnings.warn("set sequence parallel to false as TP size is 1", stacklevel=2)
            self.sequence_parallel = False


@dataclass
class FSDPEngineConfig(EngineConfig):
    """Configuration for FSDP (Fully Sharded Data Parallel).

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        wrap_policy (Dict[str, Any]): Configuration for FSDP wrap policy.
        param_offload (bool): Whether to offload parameters to CPU, default False
        optimizer_offload (bool): Whether to offload optimizer states to CPU, default False
        offload_policy (bool): Whether to offload policy model parameters, default False
        reshard_after_forward (bool): Whether to reshard parameters after forward pass, default True
        fsdp_size (int): FSDP group size. -1 means use all available GPUs.
        forward_prefetch (bool): Whether to prefetch parameters for next forward pass, default False
        model_dtype (str): Model data type used to initialize the transformers model. default "fp32"
        use_orig_params (bool): Whether to use original parameters when initialize FSDP1, default False
        seed (int): Random seed for reproducibility.
        full_determinism (bool): If true, enable_full_determinism is called to ensure reproducible results
            in distributed training. Important: this will negatively impact performance, so only use it for
            debugging.
        mixed_precision (Optional[dict[str, Any]]): Mixed precision configuration for FSDP, default None
        dtype (str): Mixed precision training param dtype, default "bfloat16"
    """

    # ulysses_sequence_parallel_size is mutable for backward compatibility
    _mutable_fields = EngineConfig._mutable_fields | {"ulysses_sequence_parallel_size"}

    # fsdp specific flags
    wrap_policy: dict[str, Any] = field(default_factory=dict)
    offload_policy: bool = False
    reshard_after_forward: bool = True
    fsdp_size: int = -1
    forward_prefetch: bool = False
    model_dtype: str = "fp32"
    use_orig_params: bool = False
    mixed_precision: Optional[dict[str, Any]] = None
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    use_torch_compile: bool = True
    entropy_checkpointing: bool = False
    strategy: str = "fsdp"

    def __post_init__(self):
        super().__post_init__()
        assert self.strategy in ["fsdp", "fsdp2"], f"strategy {self.strategy} not supported"


@dataclass
class VeOmniEngineConfig(EngineConfig):
    """Configuration for VeOmni.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        wrap_policy (Dict[str, Any]): Configuration for FSDP wrap policy.
        param_offload (bool): Whether to offload parameters to CPU, default False
        optimizer_offload (bool): Whether to offload optimizer states to CPU, default False
        offload_policy (bool): Whether to offload policy model parameters, default False
        reshard_after_forward (bool): Whether to reshard parameters after forward pass, default True
        fsdp_size (int): FSDP group size. -1 means use all available GPUs, default -1
        ulysses_parallel_size (int): Ulysses sequence parallel size, default 1
        expert_parallel_size (int): Expert parallel size, default 1
        init_device (str): Device to initialize model weights.
            1. `cpu`: Init parameters on CPU in rank0 only.
            2. `cuda`: Init parameters on GPU.
            3. `meta`: Init parameters on meta.
            4. `npu`: Init parameters on Ascend NPU.
            default "meta"
        enable_full_shard (bool): Enable fully shard for FSDP training (ZeRO-3), default False
        enable_fsdp_offload (bool): Enable CPU offload for FSDP1, default False
        enable_reentrant (bool): Use reentrant gradient checkpointing, default False
        attn_implementation (str): Attention implementation to use.
            1. `eager`
            2. `sdpa`
            3. `flash_attention_2`
            4. `flash_attention_3`
            5. `veomni_flash_attention_2_with_sp`
            6. `veomni_flash_attention_3_with_sp`
            7. `native-sparse`
            default "flash_attention_2"
            Note: In case VeOmni add more attn_implementation, please check https://github.com/ByteDance-Seed/VeOmni/
        moe_implementation (str): MoE implementation to use.
            1. `eager`
            2. `fused`
            default "fused"
            Note: In case VeOmni add more moe_implementation, please check https://github.com/ByteDance-Seed/VeOmni/
        force_use_huggingface (bool): Force loading model from huggingface, default False
        activation_gpu_limit (float): When enabling activation offload, `activation_gpu_limit` GB
            activations are allowed to reserve on GPU, default 0.0
        basic_modules (list[str]): List of basic modules to use, default None
        forward_prefetch (bool): Whether to prefetch parameters for next forward pass, default False
        model_dtype (str): Model data type used to initialize the transformers model. default "fp32"
        use_orig_params (bool): Whether to use original parameters when initialize FSDP1, default False
        seed (int): Random seed for reproducibility.
        full_determinism (bool): If true, enable_full_determinism is called to ensure reproducible results
            in distributed training. Important: this will negatively impact performance, so only use it for
            debugging.
        mixed_precision (Optional[dict[str, Any]]): Mixed precision configuration for FSDP, default None

    """

    wrap_policy: dict[str, Any] = field(default_factory=dict)
    offload_policy: bool = False
    reshard_after_forward: bool = True
    forward_prefetch: bool = False
    use_orig_params: bool = False
    entropy_from_logits_with_chunking: bool = False
    use_torch_compile: bool = True
    entropy_checkpointing: bool = False
    strategy: str = "veomni"
    fsdp_size: int = -1
    ulysses_parallel_size: int = 1
    expert_parallel_size: int = 1
    seed: int = 42
    full_determinism: bool = False
    mixed_precision: bool = False
    init_device: str = "meta"
    enable_full_shard: bool = False
    ckpt_manager: Literal["dcp"] = "dcp"
    load_checkpoint_path: Optional[str] = None
    enable_fsdp_offload: bool = False
    enable_reentrant: bool = False
    attn_implementation: str = "flash_attention_2"
    moe_implementation: str = "fused"
    force_use_huggingface: bool = False
    activation_gpu_limit: float = 0.0
    basic_modules: Optional[list[str]] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        assert self.strategy in ["veomni"], f"strategy {self.strategy} not supported"


@dataclass
class TorchtitanEngineConfig(EngineConfig):
    """Configuration for Torchtitan.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        wrap_policy (Dict[str, Any]): Configuration for FSDP wrap policy.
        reshard_after_forward (Literal["default", "always", "never"]): The policy for applying
            `reshard_after_forward` within an FSDP setup, default "default"
        forward_prefetch (bool): Whether to prefetch parameters for next forward pass, default False
        use_orig_params (bool): Whether to use original parameters when initialize FSDP, default False
        mixed_precision (bool): Mixed precision configuration for FSDP, default False
        offload_policy (bool): Whether to offload policy model parameters, default False
        data_parallel_size (int): Data parallel group size, default 1
        data_parallel_replicate_size (int): Data parallel replicate size, default 1
        data_parallel_shard_size (int): Data parallel shard degree, default 1
        tensor_parallel_size (int): Tensor parallel size, default 1
        expert_parallel_size (int): Expert parallel size, default 1
        expert_tensor_parallel_size (int): Expert tensor parallel size, default 1
        pipeline_parallel_size (int): Pipeline parallel size, default 1
        context_parallel_size (int): Context parallel size, default 1
        attn_type (str): Attention type for torchtitan's model (e.g., "sdpa", "flex", "varlen"),
            default "flex"
        strategy (str): Strategy to use for distributed training, default "torchtitan"
        seed (int): Random seed for reproducibility.
        full_determinism (bool): If true, enable_full_determinism is called to ensure reproducible results
            in distributed training. Important: this will negatively impact performance, so only use it for
            debugging.

    """

    wrap_policy: dict[str, Any] = field(default_factory=dict)
    reshard_after_forward: Literal["default", "always", "never"] = "default"
    forward_prefetch: bool = False
    use_orig_params: bool = False
    mixed_precision: bool = False
    offload_policy: bool = False
    use_torch_compile: bool = True
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    data_parallel_size: int = 1
    data_parallel_replicate_size: int = 1
    data_parallel_shard_size: int = 1
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    context_parallel_size: int = 1
    attn_type: str = "flex"
    max_seq_len: Optional[int] = None
    strategy: str = "torchtitan"
    seed: int = 42
    full_determinism: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert self.strategy in ["torchtitan"], f"strategy {self.strategy} not supported"


@dataclass
class AutomodelEngineConfig(EngineConfig):
    """Configuration for Automodel (nemo_automodel) backend.

    The Automodel backend uses NeMoAutoModelForCausalLM for model loading and
    supports FSDP2, MegatronFSDP, and DDP distributed strategies with optional
    TP, CP, and EP parallelism.

    Args:
        strategy (str): Backend strategy identifier, must be "automodel".
        distributed_strategy (str): Distributed training strategy: "fsdp2", "megatron_fsdp", or "ddp".
        tp_size (int): Tensor parallel size.
        pp_size (int): Pipeline parallel size (only pp_size=1 supported initially).
        cp_size (int): Context parallel size.
        ep_size (int): Expert parallel size for MoE models.
        dp_replicate_size (int): Data-parallel replicate size for HSDP. 1 = pure sharding.
        sequence_parallel (bool): Enable sequence parallelism in the TP plan.
        defer_fsdp_grad_sync (bool): Defer FSDP gradient sync to the final micro-batch.
        activation_checkpointing (bool): Whether to enable activation checkpointing.
        enable_fp8 (bool): Whether to enable FP8 training.
        enable_compile (bool): Whether to enable torch.compile for the model.
        model_dtype (str): Model data type for loading weights. "fp32" loads in float32
            (matching FSDP golden), "auto" uses the dtype from the model config.
        attn_implementation (str): Attention implementation to use ("sdpa", "flash_attention_2", "eager", "te").

    Backend settings (nemo_automodel BackendConfig):
        use_te_backend (bool): Use TransformerEngine attn/linear/rms_norm.
        rope_fusion (bool): Enable RoPE fusion (requires TransformerEngine).
        gate_precision (Optional[str]): Precision for MoE gate/router weights (e.g. "fp32", "bf16").
        enable_hf_state_dict_adapter (bool): Enable HuggingFace state dict compatibility.
        enable_fsdp_optimizations (bool): Enable FSDP-specific optimizations in TE layers.

    MoE / Expert Parallelism settings:
        enable_deepep (bool): Enable DeepEP for distributed expert parallelism.
        reshard_after_forward (bool): Reshard parameters after forward pass in MoE parallelizer.
        fake_balanced_gate (bool): Use balanced gate for performance analysis.
        ignore_router_for_ac (bool): Use selective activation checkpointing that saves router outputs.
        lm_head_precision (Optional[str]): Custom precision for lm_head layer (e.g. "fp32").
        wrap_outer_model (bool): Wrap outer model in FSDP if it differs from inner model.

    Mixed precision policy (FSDP2):
        mp_param_dtype (str): Parameter dtype for FSDP2 mixed precision policy.
        mp_reduce_dtype (str): Reduce dtype for FSDP2 mixed precision policy.
        mp_output_dtype (str): Output dtype for FSDP2 mixed precision policy.

    Entropy computation:
        entropy_from_logits_with_chunking (bool): Whether to use chunked entropy computation.
        use_torch_compile (bool): Whether to use torch.compile for entropy computation.
        entropy_checkpointing (bool): Whether to use checkpointing for entropy computation.
    """

    strategy: str = "automodel"
    distributed_strategy: str = "fsdp2"
    # Parallelism sizes
    tp_size: int = 1
    pp_size: int = 1
    cp_size: int = 1
    ep_size: int = 1
    dp_replicate_size: int = 1
    sequence_parallel: bool = False
    defer_fsdp_grad_sync: bool = True
    # Model settings
    activation_checkpointing: bool = False
    enable_fp8: bool = False
    enable_compile: bool = False
    model_dtype: str = "fp32"
    attn_implementation: str = "sdpa"
    # Backend settings (BackendConfig)
    use_te_backend: bool = False
    rope_fusion: bool = True
    gate_precision: Optional[str] = None
    enable_hf_state_dict_adapter: bool = True
    enable_fsdp_optimizations: bool = False
    # MoE / Expert Parallelism settings
    enable_deepep: bool = False
    reshard_after_forward: bool = False
    fake_balanced_gate: bool = False
    ignore_router_for_ac: bool = False
    lm_head_precision: Optional[str] = None
    wrap_outer_model: bool = True
    # Mixed precision policy
    mp_param_dtype: str = "bf16"
    mp_reduce_dtype: str = "fp32"
    mp_output_dtype: str = "bf16"
    # Entropy computation
    entropy_from_logits_with_chunking: bool = False
    use_torch_compile: bool = True
    entropy_checkpointing: bool = False

    def __post_init__(self):
        super().__post_init__()
        assert self.strategy == "automodel", f"strategy must be 'automodel', got {self.strategy}"
        assert self.distributed_strategy in ["fsdp2", "megatron_fsdp", "ddp"], (
            f"distributed_strategy {self.distributed_strategy} not supported"
        )
        assert self.pp_size == 1, "Pipeline parallelism (pp_size > 1) is not yet supported for automodel backend"


@dataclass
class TrainingWorkerConfig(BaseConfig):
    model_type: str = None  # model type (language_model/value_model)
    model_config: HFModelConfig = None
    engine_config: EngineConfig = None
    optimizer_config: OptimizerConfig = None
    checkpoint_config: CheckpointConfig = None
    profiler_config: ProfilerConfig = None
    # automatically select engine and optimizer function.
    # This function takes model config and the device name as parameter.
    # Users can pass in a higher-order function to take more parameters
    auto_select_engine_optim_fn: Callable[["HFModelConfig", str], tuple["EngineConfig", "OptimizerConfig"]] = None
