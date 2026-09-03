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

import inspect
import logging
import os
from functools import partial
from typing import Any, Callable, ContextManager, Iterator

import torch
import torch.distributed
from megatron.core import parallel_state as mpu
from megatron.core.package_info import __version__
from megatron.core.pipeline_parallel import get_forward_backward_func
from omegaconf import OmegaConf
from tensordict import TensorDict

import verl.utils.torch_functional as verl_F
from verl.models.mcore import get_mcore_weight_converter
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.dynamic_cp_scheduler import (
    DCP_GROUP_LEADER,
    DCP_LOCAL_NUM_TOKENS,
    DCP_PADDING_MASK,
    DCP_SAMPLE_IDS,
    DynamicCPScheduler,
    get_megatron_dynamic_cp_scheduler_cls,
    postprocess_dynamic_cp_batch,
)
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.router_replay_patch import RouterReplay, RouterReplayAction, apply_router_replay_patch
from verl.utils.megatron.router_replay_utils import (
    RouterReplayHelper,
    align_r3_router_replay_data,
    build_r3_replay_mask,
    merge_nested_router_maps,
    merge_router_topk_indices,
    pp_gather,
    reorder_and_merge_vpp_layers,
    set_model_router_replay_action,
    set_router_replay_data,
)
from verl.utils.megatron.tensor_parallel import (
    vocab_parallel_entropy,
    vocab_parallel_entropy_with_chunking,
    vocab_parallel_log_probs_from_logits,
    vocab_parallel_sum_pi_squared,
)
from verl.utils.megatron_peft_utils import build_peft_config_for_vllm
from verl.utils.megatron_utils import (
    check_mtp_config,
    get_megatron_module_device,
    get_megatron_mtp_loss,
    load_megatron_model_to_gpu,
    load_megatron_optimizer,
    offload_megatron_model_to_cpu,
    offload_megatron_optimizer,
    patch_engine_mtp,
    register_megatron_training_hooks,
    unwrap_model,
)
from verl.utils.model import extract_multi_modal_inputs, load_mcore_dist_weights
from verl.utils.seqlen_balancing import restore_dynamic_batch
from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import postprocess_batch_func, prepare_micro_batches
from .utils import set_random_seed

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _resolve_fused_temperature(temperature: float | torch.Tensor) -> float:
    """Return the scalar temperature required by fused linear cross entropy."""
    values = torch.as_tensor(temperature).detach().flatten()
    if values.numel() == 0 or not torch.isfinite(values).all().item():
        raise ValueError("temperature must contain finite values")
    if not torch.all(values == values[0]).item():
        raise NotImplementedError(
            "Megatron fused kernels require a uniform temperature; set use_fused_kernels=False "
            "for per-sample temperatures."
        )
    return max(float(values[0].item()), 1e-8)


def _validate_dcp_world_size(dpcp_size: int) -> None:
    """Validate the topology accepted by Megatron-Core's dynamic group builder."""
    if dpcp_size < 2 or dpcp_size % 2 != 0:
        raise ValueError(
            f"Dynamic CP requires the DPxCP world size to be an even integer of at least two, got DPxCP={dpcp_size}."
        )


def _check_dcp_unsupported_features(engine_config, model_config, tf_config=None, batch=None) -> None:
    """Validate DCP capabilities as model, transformer, and batch state become available."""
    if not engine_config.dynamic_context_parallel:
        return
    if not engine_config.use_remove_padding:
        raise ValueError("dynamic_context_parallel requires use_remove_padding=True")
    if engine_config.pad_to_length:
        raise NotImplementedError("dynamic_context_parallel does not support pad_to_length")
    if model_config.model_type == "value_model":
        raise NotImplementedError("Dynamic CP currently supports language models only")
    if hasattr(model_config.hf_config, "vision_config"):
        raise NotImplementedError("Dynamic CP does not support multimodal models")
    if engine_config.virtual_pipeline_model_parallel_size not in (None, 1):
        raise NotImplementedError("Dynamic CP does not support virtual pipeline parallelism")

    if tf_config is not None:
        if getattr(tf_config, "fp8", None) not in (None, False):
            raise NotImplementedError("Dynamic CP does not support FP8 training")
        if engine_config.router_replay.mode != "disabled" and getattr(tf_config, "moe_router_fusion", False):
            raise NotImplementedError("Dynamic CP router replay requires moe_router_fusion=False")

    if batch is not None:
        if tu.get_non_tensor_data(batch, key="distillation_use_topk", default=False) or tu.get_non_tensor_data(
            batch, key="distillation_only", default=False
        ):
            raise NotImplementedError("Dynamic CP does not support distillation")


def _attach_dcp_recorded_routes(losses_reduced: list[dict], layers_topk_idx: torch.Tensor) -> None:
    """Attach recorded routes to DCP micro-batches without dropping or inventing samples."""
    recorded_routes = list(layers_topk_idx.unbind())
    cursor = 0
    for micro_batch_index, micro_output in enumerate(losses_reduced):
        micro_sample_ids = micro_output.get(DCP_SAMPLE_IDS)
        if not micro_sample_ids:
            raise RuntimeError(f"DCP router replay micro-batch {micro_batch_index} has no sample ids")
        end = cursor + len(micro_sample_ids)
        if end > len(recorded_routes):
            raise RuntimeError(
                "DCP router replay produced fewer route tensors than scheduled samples: "
                f"need {end}, got {len(recorded_routes)}"
            )
        micro_output.setdefault("model_output", {})["routed_experts"] = torch.nested.as_nested_tensor(
            recorded_routes[cursor:end], layout=torch.jagged
        )
        cursor = end

    if cursor != len(recorded_routes):
        raise RuntimeError(
            f"DCP router replay produced unconsumed route tensors: consumed {cursor}, recorded {len(recorded_routes)}"
        )


class MegatronEngine(BaseEngine):
    # mcore keeps model-parallel-local params resident and moves large host
    # buffers every step; pinning the whole delta snapshot set on top of that
    # starves the node's cudaHostAlloc pool and surfaces as CUDA OOM in
    # unrelated allocations (observed at 30B/235B) -- pageable is the safe
    # default here.
    delta_pin_snapshots = False

    @staticmethod
    def _get_context_parallel_layout(unwrapped_model) -> str:
        if getattr(unwrapped_model.config, "experimental_attention_variant", None) == "dsv4_hybrid":
            return "contiguous"
        return "zigzag"

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: McoreEngineConfig,
        optimizer_config: McoreOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        assert self.engine_config.use_mbridge, "use_mbridge must be True"
        _check_dcp_unsupported_features(self.engine_config, self.model_config)
        self._init_device_mesh()

        set_random_seed(seed=self.engine_config.seed)

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload

        self.mode = None

        self.layer_name_mapping = {
            "qkv_layer_name": "self_attention.linear_qkv.",
            "gate_proj_layer_name": "linear_fc1.",
        }
        self.weight_converter = None
        self._hf_export_tasks = None

        # QAT configuration
        self._qat_config = getattr(self.engine_config, "qat", None)
        self._qat_enabled = self._qat_config is not None and getattr(self._qat_config, "enable", False)
        if self._qat_enabled:
            if self.engine_config.vanilla_mbridge:
                raise ValueError(
                    "QAT requires non-vanilla Megatron bridge. "
                    "Please set 'use_mbridge=True' and 'vanilla_mbridge=False'."
                )
            logger.info(f"QAT enabled in MegatronEngine: mode={self._qat_config.mode}")

        # Router replay configuration for MoE models
        self.enable_routing_replay = self.engine_config.router_replay.mode != "disabled"
        logger.info(f"enable_routing_replay in MegatronEngine: {self.enable_routing_replay}")
        if self.enable_routing_replay:
            apply_router_replay_patch()
            self.mini_layer_topk_idx_list = []
        # Apply checkpoint patch for MoE models
        from verl.utils.device import is_cuda_available, is_npu_available

        if is_npu_available and __version__ >= "0.16.0":
            from verl.models.mcore.patch import apply_mtp_inference_patch

            apply_mtp_inference_patch()

        if is_cuda_available:
            from verl.models.mcore.patch import apply_patch_megatron_recomputation_backward

            apply_patch_megatron_recomputation_backward()

    def _init_device_mesh(self):
        # TODO: set different parallelism for actor, critic, ref
        if self.engine_config.dynamic_context_parallel:
            get_megatron_dynamic_cp_scheduler_cls()

        if mpu.is_initialized():
            if self.engine_config.dynamic_context_parallel:
                dpcp_size = mpu.get_data_parallel_world_size(with_context_parallel=True)
                _validate_dcp_world_size(dpcp_size)
                try:
                    mpu.get_dynamic_data_context_parallel_groups(group_size=1)
                except (KeyError, AssertionError, AttributeError) as exc:
                    raise RuntimeError(
                        "Dynamic CP is enabled on this engine, but Megatron model-parallel state was already "
                        "initialized without dynamic DPxCP groups. Enable dynamic_context_parallel consistently "
                        "on every colocated Megatron engine (for example the reference model)."
                    ) from exc
            return

        extra_args = dict()

        if self.engine_config.dynamic_context_parallel:
            assert "dynamic_context_parallel" in inspect.signature(mpu.initialize_model_parallel).parameters, (
                "dynamic_context_parallel is not supported in your megatron version, "
                + "please update your megatron version to the latest version"
            )
            dpcp_size = torch.distributed.get_world_size() // (
                self.engine_config.tensor_model_parallel_size * self.engine_config.pipeline_model_parallel_size
            )
            _validate_dcp_world_size(dpcp_size)
            extra_args["dynamic_context_parallel"] = self.engine_config.dynamic_context_parallel

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.engine_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.engine_config.pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size=self.engine_config.virtual_pipeline_model_parallel_size,
            use_sharp=False,
            context_parallel_size=self.engine_config.context_parallel_size,
            expert_model_parallel_size=self.engine_config.expert_model_parallel_size,
            expert_tensor_parallel_size=self.engine_config.expert_tensor_parallel_size,
            nccl_communicator_config_path=None,
            **extra_args,
        )

    def _build_tf_config(self):
        from verl.utils.megatron_utils import mapping_string_to_attn_backend
        from verl.utils.torch_dtypes import PrecisionType

        self.is_value_model = self.model_config.model_type == "value_model"
        self.share_embeddings_and_output_weights = self.model_config.share_embeddings_and_output_weights

        check_mtp_config(self.model_config, self.engine_config)

        self.param_dtype = PrecisionType.to_dtype(self.engine_config.dtype)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        override_transformer_config = mapping_string_to_attn_backend({**self.engine_config.override_transformer_config})
        if self.is_value_model:
            # A value head cannot share weights with the vocabulary embedding. Force the HF
            # flag off (both bridges derive the model's tie-embeddings behavior from it) and
            # record it for the checkpoint manager. Do NOT push
            # share_embeddings_and_output_weights through override_transformer_config: the
            # vanilla-mbridge path forwards those into TransformerConfig(**kwargs) via
            # set_extra_args, and some Megatron builds (e.g. Ascend) reject it as an
            # unexpected kwarg. It is applied per-bridge below, mirroring routing-replay.
            self.model_config.hf_config.tie_word_embeddings = False
            self.share_embeddings_and_output_weights = False
        if self.engine_config.dynamic_context_parallel:
            override_transformer_config.update(
                {
                    "calculate_per_token_loss": True,
                    "context_parallel_size": self.engine_config.context_parallel_size,
                    # verl invokes the MCore scheduler before the pipeline schedule.
                    "dynamic_context_parallel": False,
                    "max_seqlen_per_dp_cp_rank": self.engine_config.max_seqlen_per_dp_cp_rank,
                }
            )
        if getattr(self.model_config.hf_config, "model_type", None) == "deepseek_v4" and (
            self.engine_config.context_parallel_size > 1 or self.engine_config.dynamic_context_parallel
        ):
            override_transformer_config.update(
                {
                    "cp_partition_mode": "contiguous",
                    "sequence_packing_scheduler": "default_dynamic_cp",
                }
            )
        self.provider = None
        self.vanilla_bridge = self.engine_config.vanilla_mbridge

        if self.vanilla_bridge:
            from verl.models.mcore.mbridge import AutoBridge

            bridge = AutoBridge.from_config(self.model_config.hf_config, dtype=self.param_dtype)
            bridge.set_extra_args(**override_transformer_config)
            tf_config = bridge.config
            tf_config.fp16 = self.param_dtype == torch.float16
            tf_config.bf16 = self.param_dtype == torch.bfloat16
            # Value head can't tie embeddings; set it on the built config rather than through
            # set_extra_args -> TransformerConfig(**kwargs), which some Megatron builds reject.
            if self.is_value_model and hasattr(tf_config, "share_embeddings_and_output_weights"):
                tf_config.share_embeddings_and_output_weights = False
        else:
            from verl.models.mcore.bridge import AutoBridge

            # Use Megatron-Bridge to convert HF config to Megatron config
            bridge = AutoBridge.from_hf_pretrained(
                self.model_config.local_path, trust_remote_code=self.model_config.trust_remote_code
            )
            # Get Megatron provider and configure it
            provider = bridge.to_megatron_provider(load_weights=False)

            # Match verl implementation (need variable_seq_lengths)
            from megatron.core.transformer.enums import AttnBackend

            virtual_pipeline_model_parallel_size = self.engine_config.virtual_pipeline_model_parallel_size
            provider_overrides = {
                "tensor_model_parallel_size": self.engine_config.tensor_model_parallel_size,
                "pipeline_model_parallel_size": self.engine_config.pipeline_model_parallel_size,
                "expert_model_parallel_size": self.engine_config.expert_model_parallel_size,
                "expert_tensor_parallel_size": self.engine_config.expert_tensor_parallel_size,
                "virtual_pipeline_model_parallel_size": virtual_pipeline_model_parallel_size,
                "context_parallel_size": self.engine_config.context_parallel_size,
                "sequence_parallel": self.engine_config.sequence_parallel,
                "overlap_p2p_comm": (
                    virtual_pipeline_model_parallel_size is not None and virtual_pipeline_model_parallel_size > 1
                ),
                "batch_p2p_comm": False,
                "variable_seq_lengths": True,
                "attention_backend": AttnBackend.flash,
                "moe_token_dispatcher_type": "alltoall",
                "moe_router_load_balancing_type": "none",
            }
            for key, value in override_transformer_config.items():
                provider_overrides[key] = value
            # Value head can't tie embeddings (see the value-model note above). Apply it on
            # the provider here rather than via override_transformer_config.
            if self.is_value_model and hasattr(provider, "share_embeddings_and_output_weights"):
                provider_overrides["share_embeddings_and_output_weights"] = False
            if (
                self.model_config.hf_config.model_type == "deepseek_v4"
                and not self.model_config.mtp.enable
                and getattr(provider, "mtp_num_layers", 0)
            ):
                provider_overrides["mtp_num_layers"] = 0
                csa_compress_ratios = getattr(provider, "csa_compress_ratios", None)
                if csa_compress_ratios is not None:
                    provider_overrides["csa_compress_ratios"] = csa_compress_ratios[: provider.num_layers]
            if self.enable_routing_replay:
                if hasattr(provider, "moe_enable_routing_replay"):
                    provider_overrides["moe_enable_routing_replay"] = True
                else:
                    provider_overrides["enable_routing_replay"] = True

            if self._qat_enabled:
                from megatron.bridge.models.gpt_provider import modelopt_transformer_layer_spec

                provider.transformer_layer_spec = modelopt_transformer_layer_spec

            # Megatron-Bridge >= v0.5.0 provides apply_overrides_and_finalize.
            # Megatron-Bridge <  v0.5.0 does not, so we fall back to manual setattr + finalize.
            if hasattr(provider, "apply_overrides_and_finalize"):
                provider.apply_overrides_and_finalize(
                    dtype=self.param_dtype,
                    overrides=provider_overrides,
                )
            else:
                provider.params_dtype = self.param_dtype
                provider.fp16 = self.param_dtype == torch.float16
                provider.bf16 = self.param_dtype == torch.bfloat16
                for name, value in provider_overrides.items():
                    setattr(provider, name, value)
                if hasattr(provider, "finalize"):
                    provider.finalize()
            self.provider = provider
            tf_config = None  # Will be set after model creation
        self.bridge = bridge

        if not self.bridge:
            self.weight_converter = get_mcore_weight_converter(self.model_config.hf_config, self.dtype)

        # Set router replay directly on tf_config instead of passing through
        # override_transformer_config, because dataclass subclasses like MLATransformerConfig
        # generate their own __init__ and may not accept compatibility kwargs.
        if self.enable_routing_replay and tf_config is not None:
            if hasattr(tf_config, "moe_enable_routing_replay"):
                tf_config.moe_enable_routing_replay = True
            else:
                tf_config.enable_routing_replay = True

        if torch.distributed.get_rank() == 0:
            if tf_config is not None:
                print(f"TF config: {tf_config}")
        self.tf_config = tf_config

        from verl.workers.config.megatron_peft import get_peft_cls

        self.peft_cls = get_peft_cls(
            model_config=self.model_config, bridge=self.bridge, provider=self.provider, dtype=self.param_dtype
        )

    def _resolve_override_ddp_config(self):
        """Keep the DDP grad-bucket dtype consistent with the optimizer's grad buffer.

        Default to FP32 gradient reduction for the conventional optimizer. When
        the precision-aware optimizer is opted into with a sub-FP32
        ``main_grads_dtype``, reduce gradients in that lower precision instead.
        An explicit ``override_ddp_config`` value always wins.

        For Muon + LayerWise, also enable ``use_layer_wise_param_layout`` on the
        DDP config so master weights live in the param buffer (not fp32 clones).
        """
        from verl.utils.megatron.optimizer import is_muon_layer_wise_config
        from verl.utils.torch_dtypes import PrecisionType

        override_ddp_config = dict(self.engine_config.override_ddp_config or {})
        opt_cfg = self.optimizer_config
        if opt_cfg is not None and "grad_reduce_in_fp32" not in override_ddp_config:
            use_low_precision_main_grads = (
                getattr(opt_cfg, "use_precision_aware_optimizer", False)
                and PrecisionType.to_dtype(getattr(opt_cfg, "main_grads_dtype", "fp32")) != torch.float32
            )
            override_ddp_config["grad_reduce_in_fp32"] = not use_low_precision_main_grads
        if opt_cfg is not None and is_muon_layer_wise_config(opt_cfg):
            override_ddp_config.setdefault("use_layer_wise_param_layout", True)
        return override_ddp_config

    def _build_megatron_module(self):
        from verl.utils.megatron.optimizer import is_muon_layer_wise_config
        from verl.utils.megatron_utils import McoreModuleWrapperConfig, make_megatron_module
        from verl.utils.model import print_model_size

        if self.engine_config.forward_only:
            wrap_with_ddp = False
        else:
            wrap_with_ddp = True

        layer_wise_muon = is_muon_layer_wise_config(self.optimizer_config)
        wrap_config = McoreModuleWrapperConfig(
            is_value_model=self.is_value_model,
            wrap_with_ddp=wrap_with_ddp,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            use_layer_wise_distributed_optimizer=layer_wise_muon,
            use_megatron_fsdp=self.engine_config.use_megatron_fsdp,
        )
        override_ddp_config = self._resolve_override_ddp_config()

        module, updated_tf_config = make_megatron_module(
            wrap_config=wrap_config,
            tf_config=self.tf_config,
            hf_config=self.model_config.hf_config,
            bridge=self.bridge,
            provider=self.provider,
            override_model_config=self.engine_config.override_mcore_model_config,
            override_ddp_config=override_ddp_config,
            peft_cls=self.peft_cls,
            peft_config=self.model_config.get("lora", None),
        )
        self.tf_config = updated_tf_config
        print(f"module: {len(module)}")

        if self.engine_config.use_dist_checkpointing:
            load_mcore_dist_weights(
                module, self.engine_config.dist_checkpointing_path, is_value_model=self.is_value_model
            )
        else:
            if self.vanilla_bridge:
                self.bridge.load_weights(module, self.model_config.local_path)
            else:
                allowed_mismatched_params = []
                if self.is_value_model:
                    allowed_mismatched_params = ["output_layer.weight"]
                self.bridge.load_hf_weights(
                    module, self.model_config.local_path, allowed_mismatched_params=allowed_mismatched_params
                )

        if torch.distributed.get_rank() == 0:
            print_model_size(module[0])

        if self.enable_routing_replay:
            print(f"routing replay layers: {len(RouterReplay.router_instances)}")

        return module

    def _maybe_enable_fused_kernels(self):
        if not self.engine_config.use_fused_kernels:
            return

        if not self.engine_config.use_remove_padding or self.is_value_model or self.model_config.mtp.enable:
            logger.warning_once(
                "Fused kernels require remove-padding and are not supported for value models or when MTP is enabled "
                "in Megatron engine; disabling."
            )
            self.engine_config.use_fused_kernels = False
            return

        from verl.models.mcore.model_forward_fused import patch_fused_forward

        for model in self.module:
            patch_fused_forward(model)

    def _build_optimizer(self):
        from verl.utils.megatron.optimizer import get_megatron_optimizer, init_megatron_optim_config

        optim_config_megatron = init_megatron_optim_config(
            self.optimizer_config,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            fp16=self.param_dtype == torch.float16,
            bf16=self.param_dtype == torch.bfloat16,
        )
        optimizer = get_megatron_optimizer(model=self.module, config=optim_config_megatron)
        register_megatron_training_hooks(self.module, optimizer)
        return optimizer

    def _build_lr_scheduler(self):
        from verl.utils.megatron.optimizer import get_megatron_optimizer_param_scheduler

        optimizer_scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer, config=self.optimizer_config
        )
        return optimizer_scheduler

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self):
        return (
            mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            and mpu.get_context_parallel_rank() == 0
            and (not self.engine_config.dynamic_context_parallel or mpu.get_data_parallel_rank() == 0)
        )

    def initialize(self):
        self._hf_export_tasks = None
        self._build_tf_config()
        _check_dcp_unsupported_features(self.engine_config, self.model_config, tf_config=self.tf_config)

        self.module = self._build_megatron_module()

        if self._qat_enabled and not self.engine_config.forward_only:
            from verl.utils.modelopt import apply_qat_to_modules

            self.module = apply_qat_to_modules(self.module, self._qat_config)

        # Freeze policy (after model build, before optimizer)
        if self.model_config.freeze_module_pattern:
            from verl.utils.freeze_utils import apply_freeze_to_module
            for model in self.module:
                apply_freeze_to_module(
                    model,
                    self.model_config.freeze_module_pattern,
                )

        self._maybe_enable_fused_kernels()

        if self.model_config.mtp.enable:
            patch_engine_mtp(self.module, self.model_config)
        elif (
            self.engine_config.forward_only
            and self.engine_config.override_transformer_config.get("mtp_num_layers") == 0
        ):
            from verl.models.mcore.mtp_patch import patch_postprocess

            for model in self.module:
                patch_postprocess(model)

        # For forward_only, we don't need optimizer, lr_scheduler, checkpoint_mananager
        if self.engine_config.forward_only:
            self.optimizer = None
            self.lr_scheduler = None
            self.to(device="cpu", model=self._is_offload_param, optimizer=False, grad=False)
            log_gpu_memory_usage("After offload model during init (forward_only)", logger=logger)
            return

        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()

        full_reshardable = self.engine_config.dist_ckpt_optim_fully_reshardable
        mem_eff = self.engine_config.distrib_optim_fully_reshardable_mem_efficient

        tmp_config = OmegaConf.create(
            {
                "model": {"path": self.model_config.local_path},
                "megatron": {
                    "dist_ckpt_optim_fully_reshardable": full_reshardable,
                    "distrib_optim_fully_reshardable_mem_efficient": mem_eff,
                },
            }
        )

        role = "actor" if not self.is_value_model else "critic"

        self.checkpoint_mananager = MegatronCheckpointManager(
            config=tmp_config,
            checkpoint_config=self.checkpoint_config,
            model_config=self.model_config.hf_config,
            transformer_config=self.tf_config,
            role=role,
            model=self.module,
            arch=self.model_config.architectures[0],
            hf_config=self.model_config.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            processing_class=self.model_config.get_processor(),
            optimizer=self.optimizer,
            optimizer_scheduler=self.lr_scheduler,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.optimizer_config.use_checkpoint_opt_param_scheduler,
            use_dist_checkpointing=self.engine_config.use_dist_checkpointing,
            bridge=self.bridge,
            provider=self.provider,
            peft_cls=self.peft_cls,
            use_megatron_fsdp=self.engine_config.use_megatron_fsdp,
        )

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def train_mode(self, **kwargs):
        """
        Context manager entry for switching the engine and model into training mode.

        Usage:
            with engine.train_mode():
                # runs in training mode
        """
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """
        Context manager entry for switching the engine and model into evaluation mode.

        Usage:
            with engine.eval_mode():
                # runs in evaluation mode
        """
        return EngineEvalModeCtx(self, **kwargs)

    def optimizer_zero_grad(self):
        """
        Zero out gradients of all parameters before starting a new backward pass.
        """
        self.optimizer.zero_grad()
        # use use_contiguous_buffers_in_local_ddp and no overlap_dp_param_comm
        for chunk in self.module:
            # if use distributed optimizer, zero grad buffer will be handled by optimizer
            chunk.zero_grad_buffer()

    def optimizer_step(self):
        """
        Perform an optimization step to update model parameters based on accumulated gradients.

        Returns:
            grad_norm (float): The norm of the gradients before clipping or update.
        """
        # forward_kl_topk leaves large fp32 vocab tensors until backward ends;
        # free cached blocks before grad-norm all_reduce to reduce OOM on tight VRAM.
        if getattr(self, "_distillation_use_topk_active", False):
            get_torch_device().empty_cache()
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()

        if update_successful:
            # allgather already execute in optimizer.step in new megatron
            pass
        else:
            raise NotImplementedError("Megatron optimizer step failed. This should not happen")

        return grad_norm

    def lr_scheduler_step(self):
        """
        Advance the learning rate scheduler by one step.

        Returns:
            current_lr (float or list[float]): Updated learning rate(s).
        """
        from verl.utils.megatron.optimizer import get_megatron_last_lr

        self.lr_scheduler.step(1)
        return get_megatron_last_lr(self.optimizer)

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.
        Note that this function executes irrespective of offload config. It serves as manual control

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        device_name = get_device_name()

        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_megatron_model_to_gpu(self.module, load_grad=grad)
            if optimizer and self.optimizer is not None:
                load_megatron_optimizer(self.optimizer)
        elif device == "cpu":
            if model:
                offload_megatron_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_megatron_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def get_data_parallel_rank(self):
        if self.engine_config.dynamic_context_parallel:
            # in order to let every dp-cp group has full data to split, we set dp=1
            return 0
        return mpu.get_data_parallel_rank()

    def get_data_parallel_size(self):
        if self.engine_config.dynamic_context_parallel:
            # in order to let every dp-cp group has full data to split, we set dp=1
            return 1
        return mpu.get_data_parallel_world_size()

    def get_data_parallel_group(self):
        if self.engine_config.dynamic_context_parallel:
            # The replicated DCP batch is one logical data-parallel replica.
            return mpu.get_dynamic_data_context_parallel_groups(group_size=1)
        return mpu.get_data_parallel_group()

    def get_model_parallel_group(self):
        return mpu.get_model_parallel_group()

    def get_context_parallel_group(self):
        return mpu.get_context_parallel_group()

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        max_ckpt_to_keep: int | None = None,
        **kwargs,
    ) -> None:
        """
        Save model, optimizer, and scheduler states to a checkpoint.

        Args:
            local_path: Local filesystem path to save checkpoint.
            hdfs_path: Optional HDFS path to copy checkpoint.
            global_step: Integer training step number for naming.
            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.
        """
        origin_module_device = get_megatron_module_device(self.module)
        if self._is_offload_param or origin_module_device == "cpu":
            load_megatron_model_to_gpu(self.module, load_grad=True)
        self.checkpoint_mananager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        torch.distributed.barrier()
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: str | None = None, del_local_after_load: bool = True, **kwargs
    ) -> None:
        """
        Load model, optimizer, and scheduler states from a checkpoint.

        Args:
            local_path: Local filesystem path of the checkpoint.
            hdfs_path: Optional HDFS path where checkpoint is stored.
            del_local_after_load: Whether to delete local copy after loading.
        """
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.module)
        self.checkpoint_mananager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )
        if self._is_offload_param:
            offload_megatron_model_to_cpu(self.module)
        if self._is_offload_optimizer:
            offload_megatron_optimizer(self.optimizer)

    def _routed_num_tokens(self, data: TensorDict) -> torch.Tensor:
        """Real (unpadded) tokens fed to the MoE router: attention_mask in the padded RL
        path, else the packed input_ids count in the no-padding SFT path. Not loss_mask,
        which counts response tokens only and would under-normalize the router loss."""
        attention_mask = data.get("attention_mask", None)
        if attention_mask is not None:
            return attention_mask.sum()
        input_ids = data["input_ids"]
        if input_ids.is_nested:
            return input_ids.offsets()[-1]
        return torch.tensor(input_ids.numel(), device=input_ids.device)

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        self._distillation_use_topk_active = tu.get_non_tensor_data(data, key="distillation_use_topk", default=False)
        tu.assign_non_tensor(data, sp_size=self.engine_config.context_parallel_size)

        _check_dcp_unsupported_features(self.engine_config, self.model_config, batch=data)

        # compute num_tokens in global batch for loss normalization
        loss_mask = data["loss_mask"]
        batch_num_tokens = (loss_mask.values().sum() if loss_mask.is_nested else loss_mask.sum()).to(get_device_id())
        if not self.engine_config.dynamic_context_parallel:
            torch.distributed.all_reduce(
                batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
            )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        # Global routed-token count for the per-token-loss regime (consumed in
        # postprocess_micro_batch_func). Static CP replicates real tokens inside
        # each CP group, so one DP all-reduce yields the global value; the DCP
        # batch is replicated across the whole DPxCP plane, so the local count
        # already is the global one.
        if self.tf_config is not None and self.tf_config.calculate_per_token_loss:
            routed_num_tokens = self._routed_num_tokens(data).to(get_device_id())
            if not self.engine_config.dynamic_context_parallel:
                torch.distributed.all_reduce(
                    routed_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
                )
            tu.assign_non_tensor(data, routed_num_tokens=routed_num_tokens.item())

        # BSHD path only: pad every micro-batch to the mini-batch's global max seq_len so the
        # padded `s_q` is shared -> cuDNN plan built once per shape. Raw (unaligned)
        # max; TP/CP/FP8 alignment is applied inside preprocess_bshd_engine.
        pad_bshd_to_minibatch_max = self.engine_config.pad_bshd_to_minibatch_max
        global_max_seqlen = None
        if pad_bshd_to_minibatch_max and not self.engine_config.use_remove_padding and "input_ids" in data.keys():
            input_ids_for_max = data["input_ids"]
            if input_ids_for_max.is_nested:
                global_max_seqlen = int(input_ids_for_max.offsets().diff().max().item())

        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        if vpp_size is not None and vpp_size > 1:
            num_batches_divided_by = self.tf_config.microbatch_group_size_per_vp_stage
        else:
            num_batches_divided_by = None

        dcp_group = None
        if self.engine_config.dynamic_context_parallel:
            dcp_group = mpu.get_data_parallel_group(with_context_parallel=True)
            cp_layout = (
                "contiguous"
                if getattr(self.tf_config, "experimental_attention_variant", None) == "dsv4_hybrid"
                else "zigzag"
            )
            min_local_rows = (
                self.tf_config.csa_window_size
                if cp_layout == "contiguous" and self.engine_config.use_fused_kernels
                else None
            )
            scheduler = DynamicCPScheduler(
                max_seqlen_per_dp_cp_rank=self.engine_config.max_seqlen_per_dp_cp_rank,
                dp_size=mpu.get_data_parallel_world_size(),
                cp_size=mpu.get_context_parallel_world_size(),
            )
            micro_batches = scheduler.schedule(
                data,
                dcp_group,
                tp_size=mpu.get_tensor_model_parallel_world_size(),
                cp_layout=cp_layout,
                min_local_rows=min_local_rows,
            )
            indices = None
        else:
            micro_batches, indices = prepare_micro_batches(
                data=data,
                dp_group=self.get_data_parallel_group(),
                num_batches_divided_by=num_batches_divided_by,
                same_micro_num_in_dp=True,
                min_num_micro_batch=None,
            )

        if num_batches_divided_by is not None:
            assert len(micro_batches) % num_batches_divided_by == 0, (
                f"micro_batches {micro_batches} must be divisible by num_batches_divided_by "
                f"{num_batches_divided_by} for megatron backend"
            )

        # compute input shapes for pp stages
        n_micro_batch = len(micro_batches)

        for micro_batch in micro_batches:
            tu.assign_non_tensor(micro_batch, num_micro_batch=n_micro_batch)
            if global_max_seqlen is not None:
                tu.assign_non_tensor(micro_batch, forced_max_seqlen=global_max_seqlen)

        forward_backward_func = get_forward_backward_func()

        postprocess_micro_batch_func = partial(
            self.postprocess_micro_batch_func,
            forward_only=forward_only,
            loss_function=loss_function,
        )

        tu.assign_non_tensor(data, num_micro_batch=n_micro_batch)

        forward_step = partial(
            self.forward_step,
            logits_processor_func=loss_function,
            postprocess_micro_batch_func=postprocess_micro_batch_func,
        )

        enable_routing_replay = tu.get_non_tensor_data(data, key="enable_routing_replay", default=False)

        if enable_routing_replay:
            # Set to REPLAY mode: for R3 mode or actor update phase in R2 mode
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            if forward_only and self.engine_config.router_replay.mode == "R2":
                # In R2 mode, forward_only calls (e.g., compute_log_probs) need to record routing information
                RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.module,
            num_microbatches=n_micro_batch,
            seq_length=1,  # the communication shape is obtained via p2p comm
            micro_batch_size=1,  # the communication shape is obtained via p2p comm
            forward_only=forward_only,
        )

        if self.model_config.mtp.enable and mpu.is_pipeline_last_stage(ignore_virtual=True):
            # All CP ranks must participate in the all_reduce inside get_megatron_mtp_loss,
            # because save_loss_to_tracker uses avg_group=DP+CP group.
            # Only collect metrics on the src rank afterward.
            metrics = get_megatron_mtp_loss(n_micro_batch)
            if self.is_mp_src_rank_with_outputs():
                if "metrics" not in losses_reduced[0]:
                    losses_reduced[0]["metrics"] = {}
                losses_reduced[0]["metrics"].update(metrics)

        if RouterReplayHelper.is_r2_record_action(self.tf_config):
            if self.tf_config.virtual_pipeline_model_parallel_size is not None:
                # config = self.actor_module[0].module.module.config
                vp_size = len(self.module)
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                bs = n_micro_batch
                topk_idx_td = reorder_and_merge_vpp_layers(
                    self.mini_layer_topk_idx_list, bs, vp_size, microbatch_group_size_per_vp_stage
                )
            else:
                topk_idx_td = merge_nested_router_maps(self.mini_layer_topk_idx_list)
            self.mini_layer_topk_idx_list = []

            layers_topk_idx = pp_gather(topk_idx_td, self.tf_config)
            use_dynamic_bsz = tu.get_non_tensor_data(data=data, key="use_dynamic_bsz", default=True)
            if use_dynamic_bsz and indices is not None:
                layers_topk_idx = restore_dynamic_batch(layers_topk_idx, indices)
            if dcp_group is not None and mpu.is_pipeline_last_stage(ignore_virtual=True):
                # This rank only recorded routes for its scheduled samples, in
                # micro-batch order. Hand them per micro-batch to the leader
                # collection below, which restores the original sample order.
                _attach_dcp_recorded_routes(losses_reduced, layers_topk_idx)

        output = {}
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            if dcp_group is not None:
                output = postprocess_dynamic_cp_batch(
                    losses_reduced, len(data), dcp_group, include_model_output=self.is_mp_src_rank_with_outputs()
                )
            else:
                output = postprocess_batch_func(output_lst=losses_reduced, indices=indices, data=data)
            if RouterReplayHelper.is_r2_record_action(self.tf_config) and dcp_group is None:
                output["model_output"]["routed_experts"] = layers_topk_idx
        if enable_routing_replay:
            RouterReplay.clear_global_indices()
            RouterReplay.clear_global_router_replay_action()
        return output

    def _mbridge_export_tasks(self):
        """Cache static export tasks while preserving Bridge's specialized FP8 task planning."""
        if getattr(self.bridge, "export_weight_dtype", None) == "fp8":
            return None
        if self._hf_export_tasks is None:
            self._hf_export_tasks = self.bridge.get_conversion_tasks(self.module)
        return self._hf_export_tasks

    def get_per_tensor_param(self, base_sync_done=False, **kwargs):
        peft_config = None
        non_merge_lora_sync = self.peft_cls is not None and not self.model_config.lora.get("merge", False)
        adapter_only = base_sync_done and non_merge_lora_sync
        if non_merge_lora_sync:
            peft_config = build_peft_config_for_vllm(self.model_config.lora)
        # when lora adapter only, we only load adapter weights when base sync is done, otherwise load all weights
        load_megatron_model_to_gpu(self.module, load_grad=False, load_frozen_params=not adapter_only)
        if self.vanilla_bridge:
            per_tensor_param = self.bridge.export_weights(self.module)
        elif adapter_only:
            per_tensor_param = self.bridge.export_adapter_weights(self.module)
        else:
            conversion_tasks = self._mbridge_export_tasks()
            per_tensor_param = (
                self.bridge.export_hf_weights(
                    self.module,
                    conversion_tasks=conversion_tasks,
                    merge_adapter_weights=False,
                )
                if non_merge_lora_sync
                else self.bridge.export_hf_weights(self.module, conversion_tasks=conversion_tasks)
            )

        # QAT: process weights through QATWeightExporter for quantized weight sync to vLLM
        if self._qat_enabled:
            from verl.utils.modelopt import export_qat_weights

            per_tensor_param = export_qat_weights(per_tensor_param, self.module, self._qat_config.mode, self.bridge)

        return per_tensor_param, peft_config

    def _mcore_export_index(self):
        """Build (once) the per-parameter delta export index: geometry specs and
        form-B probes derived from the bridge's conversion tasks. See
        :mod:`verl.workers.engine.megatron.delta_export`."""
        index = getattr(self, "_delta_export_index", None)
        if index is None:
            from .delta_export import build_export_index

            assert not self.vanilla_bridge, (
                "megatron delta_sharded is built on Megatron-Bridge param mappings; "
                "the deprecated vanilla mbridge flavor is not supported"
            )
            assert self.peft_cls is None, "megatron delta_sharded does not support LoRA"
            self._delta_slot_cache: dict = {}
            index = build_export_index(self.bridge, self.module, self._delta_slot_cache)
            self._delta_export_index = index
            self._delta_export_by_name = {rec.megatron_name: rec for rec in index}
        return index

    def get_per_tensor_param_shard(self, **kwargs):
        """Yield each rank's *local* mcore shard ``(name, local_flat_bf16,
        ShardSpec)`` -- the spec carries only the wire merge group (the
        comm-stubbed probe owns all shard geometry). Pure export, no side
        effects. Params owned by another pipeline stage yield an empty shard
        (zero-count lockstep rows; see the index builder)."""
        load_megatron_model_to_gpu(self.module, load_grad=False)
        index = self._mcore_export_index()

        def _gen():
            empty = torch.empty(0, dtype=torch.bfloat16, device=get_device_id())
            for rec in index:
                if rec.param is None:
                    yield rec.megatron_name, empty, rec.spec
                    continue
                local = rec.param.data
                if local.is_floating_point() and local.dtype != torch.bfloat16:
                    local = local.to(torch.bfloat16)
                yield rec.megatron_name, local.reshape(-1), rec.spec

        return _gen(), None

    def _hf_delta_entry(self, name, spec, place, lidx, lval):
        """mcore's per-param entry builder: probe the bridge's own converter
        (comm-stubbed copy -- real group sizes, gathers synthesized locally)
        with NaN sentinels -- see
        :func:`verl.workers.engine.megatron.delta_export.mcore_hf_delta_entry`."""
        from .delta_export import mcore_hf_delta_entry

        rec = self._delta_export_by_name[name]
        return mcore_hf_delta_entry(rec, place, lidx, lval, self._delta_slot_cache)

    def get_per_tensor_param_delta_shard(self, **kwargs):
        """Yield the delta engine's steady payloads -- FINAL HF-coordinate entries
        per mcore parameter (see the FSDP engine's method of the same name for the
        contract; MegatronEngine extends BaseEngine directly, so the thin wrapper
        is repeated here). Requires a prior :meth:`prime_delta_snapshots` call."""
        from ..utils import hf_delta_export

        self._delta_shard_snap = getattr(self, "_delta_shard_snap", {})
        gen, _ = self.get_per_tensor_param_shard()
        return hf_delta_export(gen, self._delta_shard_snap, self._hf_delta_entry), None

    def disable_adapter(self) -> ContextManager:
        return self.peft_cls.disable_adapter(self.module)

    def forward_step(self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def postprocess_micro_batch_func(self, output, data: TensorDict, forward_only: bool, loss_function):
        raise NotImplementedError("postprocess_micro_batch_func must be implemented in subclass")


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: MegatronEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, MegatronEngine)
        super().__enter__()
        # mcore module is a list of model chunk in each vpp stage
        for module in self.engine.module:
            module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, MegatronEngine)
        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: MegatronEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, MegatronEngine)
        super().__enter__()
        # mcore module is a list of model chunk in each vpp stage
        for module in self.engine.module:
            module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, MegatronEngine)
        if self.zero_grad_on_exit or exc_type is not None:
            self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend="megatron")
class MegatronEngineWithLMHead(MegatronEngine):
    def prepare_model_inputs(self, batch: TensorDict):
        input_ids = batch["input_ids"]
        loss_mask = batch["loss_mask"].to(bool)
        multi_modal_inputs = extract_multi_modal_inputs(batch.get("multi_modal_inputs", []))

        routed_experts = batch.get("routed_experts", None)

        return {
            "input_ids": input_ids,
            "attention_mask": batch.get("attention_mask", None),
            "loss_mask": loss_mask,
            "multi_modal_inputs": multi_modal_inputs,
            "routed_experts": routed_experts,
        }

    def prepare_model_outputs(self, output: dict, data: TensorDict):
        return output

    def _lm_head_logits_processor(
        self,
        logits,
        label,
        temperature,
        *,
        calculate_sum_pi_squared: bool,
        calculate_entropy: bool,
        distillation_use_topk: bool,
        distillation_only: bool,
        logits_processor_func: Callable,
        batch: TensorDict,
        data_format: str,
    ):
        assert logits.shape[:2] == label.shape[:2]
        # avoid non-positive temperature such as padding
        temperature[temperature <= 0] = 1e-8
        assert torch.all(temperature > 0).item(), f"temperature tensor must be positive. Got {temperature}"
        logits.div_(temperature.unsqueeze(dim=-1).to(logits.dtype))
        ret = {}
        # sum_pi_squared is non-destructive — must run before vocab_parallel_entropy.
        if calculate_sum_pi_squared:
            ret["sum_pi_squared"] = vocab_parallel_sum_pi_squared(logits)
        if calculate_entropy:
            logits_bak = logits.clone()
            # # disable the hint until the fused_kernel is optimized for triton>=3.3
            # if torch.distributed.get_rank() == 0:
            #     logger.warning_once(
            #         "For memory-efficient computation, enable fused kernels via "
            #         "`actor_rollout_ref.model.use_fused_kernels=True`. "
            #         "The current `clone()` operation ensures correctness but increases memory usage."
            #     )
            if self.engine_config.entropy_from_logits_with_chunking:
                entropy = vocab_parallel_entropy_with_chunking(
                    logits,
                    chunk_size=self.engine_config.entropy_from_logits_chunk_size,
                )
            else:
                entropy = vocab_parallel_entropy(logits)

            ret["entropy"] = entropy
        else:
            logits_bak = logits

        # logits_processor_func return tensors with shape (1, total_nnz/cp_size)
        if distillation_use_topk:
            ret.update(logits_processor_func(student_logits=logits_bak, data=batch, data_format=data_format))
        if not distillation_only:
            ret["log_probs"] = vocab_parallel_log_probs_from_logits(logits_bak, label)

        return ret

    def forward_step(
        self, batch_iter: Iterator[TensorDict], model, logits_processor_func, postprocess_micro_batch_func
    ):
        batch: TensorDict = next(batch_iter)

        batch = batch.to(get_device_id())
        use_fused_kernels = tu.get_non_tensor_data(
            batch, key="use_fused_kernels", default=self.engine_config.use_fused_kernels
        )
        if bool(use_fused_kernels) != bool(self.engine_config.use_fused_kernels):
            raise RuntimeError(
                "Per-batch use_fused_kernels must match the Megatron engine setting because fused forward is "
                "installed when the model is initialized."
            )
        calculate_entropy = tu.get_non_tensor_data(batch, key="calculate_entropy", default=False)
        calculate_sum_pi_squared = tu.get_non_tensor_data(batch, key="calculate_sum_pi_squared", default=False)
        distillation_use_topk = tu.get_non_tensor_data(batch, key="distillation_use_topk", default=False)
        distillation_only = tu.get_non_tensor_data(batch, key="distillation_only", default=False)
        pad_to_length_bucket = (
            self.engine_config.pad_to_length_bucket
            if self.engine_config.pad_to_length and self.engine_config.use_remove_padding
            else None
        )

        if pad_to_length_bucket is not None and distillation_use_topk:
            raise RuntimeError("pad_to_length is not supported with top-K distillation")
        if pad_to_length_bucket is not None and self.enable_routing_replay:
            raise RuntimeError("pad_to_length is not supported with router replay")

        if calculate_sum_pi_squared and use_fused_kernels:
            raise NotImplementedError(
                "calculate_sum_pi_squared=True is not supported with use_fused_kernels=True: "
                "fused kernels do not materialize the full logits tensor needed for Σπ²."
            )
        pad_mode = tu.get_non_tensor_data(batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        temperature = batch["temperature"]
        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]
        local_cp_size = tu.get_non_tensor_data(data=batch, key="local_cp_size", default=None)
        if self.engine_config.dynamic_context_parallel and local_cp_size is None:
            raise RuntimeError("Dynamic CP micro-batch is missing local_cp_size")
        router_padding_mask = None
        if self.engine_config.dynamic_context_parallel:
            router_padding_mask = tu.get_non_tensor_data(data=batch, key=DCP_PADDING_MASK, default=None)
            if router_padding_mask is None:
                raise RuntimeError("Dynamic CP micro-batch is missing its router padding mask")
            router_padding_mask = router_padding_mask.to(input_ids.device, non_blocking=True).unsqueeze(0)
        loss_mask = model_inputs["loss_mask"]

        unwrapped_model = unwrap_model(model)
        cp_layout = self._get_context_parallel_layout(unwrapped_model)
        if hasattr(unwrapped_model, "vp_stage"):
            vp_rank = unwrapped_model.vp_stage
        else:
            vp_rank = 0

        if RouterReplayHelper.is_replay_backward_action(self.tf_config, vp_rank):
            router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
            for router in router_instance_list:
                router.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            set_model_router_replay_action(unwrapped_model, RouterReplayAction.REPLAY_FORWARD)

        if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
            layers_topk_idx = model_inputs["routed_experts"]
            replay_mask = None
            if self.engine_config.router_replay.mode == "R3":
                layers_topk_idx = align_r3_router_replay_data(layers_topk_idx, input_ids)
                replay_mask = build_r3_replay_mask(input_ids, batch["response_mask"])
            set_router_replay_data(
                layers_topk_idx,
                None,
                self.tf_config,
                vp_rank,
                replay_mask=replay_mask,
                local_cp_size=local_cp_size,
                model=unwrapped_model,
            )

        if pad_mode == DatasetPadMode.NO_PADDING:
            label = input_ids.clone()
        else:
            raise NotImplementedError(f"Pad mode {pad_mode} is not supported for megatron engine")

        if use_fused_kernels:
            temperature_value = _resolve_fused_temperature(temperature)

        if use_fused_kernels:
            from verl.models.mcore import get_mcore_forward_fused_model_engine_fn

            fused_forward_fn = get_mcore_forward_fused_model_engine_fn(self.model_config.hf_config)
            output = fused_forward_fn(
                model=model,
                input_ids=input_ids,
                labels=label,
                multi_modal_inputs=multi_modal_inputs,
                temperature=temperature_value,
                calculate_entropy=calculate_entropy,
                pad_token_id=self.model_config.tokenizer.pad_token_id,
                cp_layout=cp_layout,
                local_cp_size=local_cp_size,
                router_padding_mask=router_padding_mask,
                pad_to_length_bucket=pad_to_length_bucket,
            )
        else:
            if not isinstance(temperature, torch.Tensor):
                temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

            temperature = temperature.to(torch.float32)
            assert temperature.shape[0] == input_ids.shape[0]
            temperature = verl_F.expand_as_nested(temperature, input_ids)  # (bsz, j1)
            from verl.models.mcore import get_mcore_engine_forward_fn

            forward_fn = get_mcore_engine_forward_fn(self.model_config.hf_config)
            data_format = "thd" if self.engine_config.use_remove_padding else "bshd"

            logits_processor = partial(
                self._lm_head_logits_processor,
                calculate_sum_pi_squared=calculate_sum_pi_squared,
                calculate_entropy=calculate_entropy,
                distillation_use_topk=distillation_use_topk,
                distillation_only=distillation_only,
                logits_processor_func=logits_processor_func,
                batch=batch,
                data_format=data_format,
            )

            response_attention_mask = None
            if attention_mask is not None and not loss_mask.is_nested:
                response_attention_mask = attention_mask[:, -loss_mask.shape[-1] :]
            logits_processor_args = {
                "label": label,
                "temperature": temperature,
                "loss_mask": loss_mask,
                "response_attention_mask": response_attention_mask,
            }

            mtp_loss_normalization_factor = None
            if (
                self.model_config.mtp.enable
                and self.model_config.mtp.enable_train
                and self.tf_config.calculate_per_token_loss
            ):
                batch_num_tokens = tu.get_non_tensor_data(batch, key="batch_num_tokens", default=0)
                routed_num_tokens = tu.get_non_tensor_data(batch, key="routed_num_tokens", default=0)
                if batch_num_tokens > 0:
                    mtp_loss_normalization_factor = routed_num_tokens / batch_num_tokens

            output = forward_fn(
                model,
                input_ids,
                multi_modal_inputs,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
                vision_model=hasattr(self.model_config.hf_config, "vision_config"),
                pad_token_id=self.model_config.tokenizer.pad_token_id,
                data_format=data_format,
                mtp_enable_train=self.model_config.mtp.enable and self.model_config.mtp.enable_train,
                local_cp_size=local_cp_size,
                router_padding_mask=router_padding_mask,
                mtp_loss_normalization_factor=mtp_loss_normalization_factor,
                forced_max_seqlen=tu.get_non_tensor_data(data=batch, key="forced_max_seqlen", default=None),
                pad_to_length_bucket=pad_to_length_bucket,
                cp_layout=cp_layout,
            )

        # Router replay: record routing decisions for R2 mode
        if RouterReplayHelper.is_r2_record_action(self.tf_config, vp_rank):
            merge_router_topk_indices(
                None, input_ids, self.mini_layer_topk_idx_list, self.tf_config, vp_rank, local_cp_size=local_cp_size
            )

        # Router replay: switch to backward replay mode for next backward pass
        if RouterReplayHelper.is_replay_forward_action(self.tf_config, vp_rank):
            router_instance_list = RouterReplayHelper.get_micro_batch_router_list(self.tf_config, vp_rank)
            for router in router_instance_list:
                router.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
            set_model_router_replay_action(unwrapped_model, RouterReplayAction.REPLAY_BACKWARD)

        return output, partial(postprocess_micro_batch_func, data=batch, local_cp_size=local_cp_size)

    def postprocess_micro_batch_func(
        self, output, data: TensorDict, forward_only: bool, loss_function, local_cp_size=None
    ):
        # For memory efficiency
        # We move calculation of entropy to compute_log_probs, forward_only == True
        device = data["input_ids"].device
        model_output = self.prepare_model_outputs(output, data)

        if loss_function is not None:
            # TODO(baiyan): How to support hybrid context parallel with dp_group,
            # now the dp_group is not used, so just leave it as is, but what if we need to use it?
            loss, metrics = loss_function(model_output=model_output, data=data, dp_group=self.get_data_parallel_group())
            # scale loss by num_micro_batch because megatron will scale loss
            # by n_micro_batch inside pp schedule
            scaled_loss = loss * data["num_micro_batch"]
        else:
            assert forward_only, "forward_only must be True when loss_function is None"
            loss = torch.tensor(1.0, device=device)
            scaled_loss = loss
            metrics = {}
        output = {"loss": loss.detach().item(), "metrics": metrics}
        if forward_only or not self.engine_config.dynamic_context_parallel:
            output["model_output"] = model_output
        if self.engine_config.dynamic_context_parallel:
            output[DCP_SAMPLE_IDS] = tu.get_non_tensor_data(data, key=DCP_SAMPLE_IDS, default=None)
            output[DCP_GROUP_LEADER] = tu.get_non_tensor_data(data, key=DCP_GROUP_LEADER, default=False)

        # calculate_per_token_loss=True (auto-enabled by Megatron-Bridge at CP>1) puts
        # Megatron in its per-token regime: loss_func must return (loss_sum, num_tokens,
        # output), and finalize_model_grads divides every gradient by the accumulated
        # total_num_tokens. That division is what cancels the MoE router's pre-multiplication
        # of the aux/z loss by num_tokens; a 2-tuple leaves total_num_tokens=0, so the factor
        # is never cancelled (the ~1e4 grad_norm blow-up at CP>1).
        if self.tf_config is not None and self.tf_config.calculate_per_token_loss and loss_function is not None:
            # Static CP cannot compose per-sequence token means from local output shards.
            # DCP reconstructs each sequence inside its dynamic CP group before applying
            # the native loss, so all aggregation modes remain valid there.
            if hasattr(loss_function, "keywords") and "config" in loss_function.keywords:
                _agg_mode = getattr(loss_function.keywords["config"], "loss_agg_mode", None)
                if _agg_mode == "seq-mean-token-mean" and not self.engine_config.dynamic_context_parallel:
                    raise ValueError(
                        "loss_agg_mode='seq-mean-token-mean' is incompatible with "
                        "calculate_per_token_loss=True (auto-enabled by Megatron-Bridge "
                        "under CP>1). The per-sequence inner division by n_s requires "
                        "local-shard counts that diverge from global under CP. Use one "
                        "of: 'token-mean', 'seq-mean-token-sum', 'seq-mean-token-sum-norm'."
                    )
            # The static BSHD path does not pass a router padding mask, so the MoE router
            # normalizes aux/z loss by B*S while gradients are divided by real tokens.
            if not self.engine_config.use_remove_padding:
                raise ValueError(
                    "calculate_per_token_loss=True requires use_remove_padding=True. "
                    "verl does not pass a padding_mask to the MoE router, so in BSHD it "
                    "normalizes the aux/z loss by the padding-inclusive token count (B*S) "
                    "while gradients are divided by the real token count. Use THD "
                    "(use_remove_padding=True) or disable CP."
                )
            if self.engine_config.dynamic_context_parallel:
                local_num_tokens = tu.get_non_tensor_data(data, key=DCP_LOCAL_NUM_TOKENS, default=None)
                if local_num_tokens is None:
                    raise ValueError("Dynamic CP micro-batch is missing its local token count")
                local_num_tokens = torch.as_tensor(local_num_tokens, device=device, dtype=torch.int)
            else:
                # Static CP replicas contain the same real tokens, so report one CP share.
                cp_size = self.engine_config.context_parallel_size
                local_num_tokens = (self._routed_num_tokens(data) // cp_size).to(torch.int)
            # n_i is the global routed-token count (all-reduced in forward_backward_batch);
            # scaling loss by the same value makes Sum(L_i)/Sum(n_i) recover the loss. Falls
            # back to local counts when not plumbed (single-rank / tests).
            routed_num_tokens = tu.get_non_tensor_data(data, key="routed_num_tokens", default=None)
            if routed_num_tokens is None:
                routed_num_tokens = self._routed_num_tokens(data)
            dp_size = tu.get_non_tensor_data(data, key="dp_size", default=1)
            local_sum = loss * routed_num_tokens / dp_size
            return local_sum, local_num_tokens, output

        # return loss and stats
        return scaled_loss, output


@EngineRegistry.register(model_type="value_model", backend="megatron")
class MegatronEngineWithValueHead(MegatronEngineWithLMHead):
    # for value head
    def forward_step(self, batch_iter, model, logits_processor_func, postprocess_micro_batch_func):
        batch: TensorDict = next(batch_iter)
        batch = batch.to(get_device_id())
        model_inputs = self.prepare_model_inputs(batch)
        input_ids = model_inputs["input_ids"]
        multi_modal_inputs = model_inputs["multi_modal_inputs"]
        cp_layout = self._get_context_parallel_layout(unwrap_model(model))

        from verl.models.mcore import get_mcore_engine_forward_fn

        forward_fn = get_mcore_engine_forward_fn(self.model_config.hf_config)

        output = forward_fn(
            model,
            input_ids,
            multi_modal_inputs,
            value_model=True,
            vision_model=hasattr(self.model_config.hf_config, "vision_config"),
            pad_token_id=self.model_config.tokenizer.pad_token_id,
            data_format="thd" if self.engine_config.use_remove_padding else "bshd",
            forced_max_seqlen=tu.get_non_tensor_data(data=batch, key="forced_max_seqlen", default=None),
            pad_to_length_bucket=(
                self.engine_config.pad_to_length_bucket
                if self.engine_config.pad_to_length and self.engine_config.use_remove_padding
                else None
            ),
            cp_layout=cp_layout,
        )

        return output, partial(postprocess_micro_batch_func, data=batch)

    def prepare_model_outputs(self, output: dict | torch.Tensor, data: TensorDict):
        return {"values": output}
