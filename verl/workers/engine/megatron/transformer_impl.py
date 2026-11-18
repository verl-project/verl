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
from verl.models.mcore.util import get_thd_min_local_rows, get_thd_quant_align_size
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
from verl.utils.megatron.memory_profile import finish_memory_history, start_memory_history
from verl.utils.megatron.memory_profile import log_parts as log_memory_parts
from verl.utils.megatron.memory_profile import log_phase as log_memory_phase
from verl.utils.megatron.memory_profile import write_snapshot as write_memory_snapshot
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
from ..utils import detach_tree, postprocess_batch_func, prepare_micro_batches
from .chunk_loss_mask import chunk_loss_data, reset_chunk_loss_mask_state
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


def _resolve_cp_seq_chunks(engine_config) -> int:
    """Prefer the declarative count, retaining the legacy switch only for older configs."""
    chunks = getattr(engine_config, "cp_seq_chunks", None)
    if chunks is None:
        chunks = int(os.environ.get("DSV41_SEQ_CHUNKS", "1") or 1)
    if not isinstance(chunks, int) or isinstance(chunks, bool) or chunks < 1:
        raise ValueError(f"cp_seq_chunks must be an integer >= 1, got cp_seq_chunks={chunks!r}")
    return chunks


def _check_cp_seq_chunk_config(engine_config, tf_config=None) -> None:
    """Resolve verl settings before checking the Core schedule's requirements."""
    if _resolve_cp_seq_chunks(engine_config) == 1:
        return

    from megatron.core.pipeline_parallel.outer_chunk import validate_outer_chunk_config

    overlap = not engine_config.forward_only and (engine_config.override_ddp_config or {}).get(
        "overlap_grad_reduce", engine_config.use_megatron_fsdp
    )
    validate_outer_chunk_config(
        context_parallel_size=engine_config.context_parallel_size,
        pipeline_model_parallel_size=engine_config.pipeline_model_parallel_size,
        dynamic_context_parallel=engine_config.dynamic_context_parallel,
        use_remove_padding=engine_config.use_remove_padding,
        use_fused_kernels=engine_config.use_fused_kernels,
        overlap_grad_reduce=overlap,
        router_replay_mode=engine_config.router_replay.mode,
        forward_only=engine_config.forward_only,
        transformer_config=tf_config,
    )


def _check_cp_seq_chunk_models(models, *, engine_config):
    """Validate actual routers while Bridge constructs models, before wrapping."""
    if _resolve_cp_seq_chunks(engine_config) == 1:
        return models

    from megatron.core.pipeline_parallel.outer_chunk import validate_outer_chunk_models

    _check_cp_seq_chunk_config(engine_config)
    return validate_outer_chunk_models(models, forward_only=engine_config.forward_only)


def _check_cp_seq_chunk_batch(batch, model) -> None:
    """Check the prerequisites that depend on this step's payload."""
    # Rule 6's batch replay flag and rule 8's video/image presence cannot be
    # known at construction: even a VLM can receive a text-only microbatch.
    batch_replay = tu.get_non_tensor_data(batch, key="enable_routing_replay", default=False)
    if batch_replay:
        raise NotImplementedError(
            f"Outer-sequence chunking does not support router replay: enable_routing_replay={batch_replay!r}"
        )
    inputs = extract_multi_modal_inputs(batch.get("multi_modal_inputs", []))
    videos = {key: value.numel() for key, value in inputs.items() if key in ("pixel_values_videos", "video_grid_thw")}
    if any(videos.values()):
        raise NotImplementedError(f"Outer-sequence chunking does not support video inputs: video_numel={videos!r}")
    images = {
        key: value.numel()
        for key, value in inputs.items()
        if key in ("pixel_values", "image_grid_hws", "image_grid_thw")
    }
    image_support = getattr(unwrap_model(model), "supports_cp_chunk_images", False)
    if any(images.values()) and not image_support:
        raise NotImplementedError(
            "Outer-sequence chunking with images requires a paired Megatron Core model "
            "with supports_cp_chunk_images=True; upgrade Core for chunk-aware image placement: "
            f"supports_cp_chunk_images={image_support!r}, image_numel={images!r}"
        )


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
        if (
            tu.get_non_tensor_data(batch, key="distillation_use_topk", default=False)
            or tu.get_non_tensor_data(batch, key="distillation_only", default=False)
            or tu.get_non_tensor_data(batch, key="needs_vocab_logits", default=False)
        ):
            # Same constraint for both entries: both keep per-position vocab tensors resident
            # past the forward, and the dynamic-CP reshuffle would move rows out from under them.
            raise NotImplementedError(
                "Dynamic CP does not support distillation or any other resident-vocab-logits loss"
            )


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
        _check_cp_seq_chunk_config(self.engine_config)
        _check_dcp_unsupported_features(self.engine_config, self.model_config)
        self._init_device_mesh()

        set_random_seed(seed=self.engine_config.seed)

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload

        self.mode = None

        self._hf_export_tasks = None

        # QAT configuration
        self._qat_config = getattr(self.engine_config, "qat", None)
        self._qat_enabled = self._qat_config is not None and getattr(self._qat_config, "enable", False)
        if self._qat_enabled:
            if self.engine_config.override_transformer_config.get("fp8"):
                raise ValueError(
                    "QAT and Transformer Engine FP8 training are mutually exclusive: QAT fake-quantizes weights "
                    "inside bf16 GEMMs, while override_transformer_config.fp8 switches the GEMMs themselves to "
                    "FP8. Disable one of them."
                )
            logger.info(f"QAT enabled in MegatronEngine: mode={self._qat_config.mode}")
        self._rollout_weight_quantizer = None

        # override_transformer_config.fp4 trains the routed experts in NVFP4.
        from verl.utils.megatron.low_precision import bf16_layers_at_start_end
        from verl.utils.real_nvfp4 import real_nvfp4_enabled

        overrides = self.engine_config.override_transformer_config
        self._real_nvfp4_enabled = real_nvfp4_enabled(overrides)
        self._real_nvfp4_bf16_layers = bf16_layers_at_start_end(overrides)
        if self._real_nvfp4_enabled:
            logger.info("NVFP4 routed-expert training: BF16 layers at start/end=%d/%d", *self._real_nvfp4_bf16_layers)

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

        if _resolve_cp_seq_chunks(self.engine_config) > 1:
            # Resolve Bridge defaults and overrides during construction, rather
            # than discovering an unsupported attention/loss at the first step.
            self._build_tf_config()
            _check_cp_seq_chunk_config(self.engine_config, self.provider)

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
        if self._real_nvfp4_enabled:
            from verl.utils.real_nvfp4.megatron import apply_real_nvfp4_overrides

            apply_real_nvfp4_overrides(
                override_transformer_config,
                self.model_config.hf_config,
                pipeline_size=self.engine_config.pipeline_model_parallel_size,
                virtual_pipeline_size=self.engine_config.virtual_pipeline_model_parallel_size,
            )
        if self.is_value_model:
            # A value head cannot share weights with the vocabulary embedding. Force the HF
            # flag off and record it for the checkpoint manager. Apply the
            # override to the provider after it is created.
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
        self.bridge = bridge

        self.tf_config = None  # Set after the provider creates the model.

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

        if _resolve_cp_seq_chunks(self.engine_config) > 1:
            # Hash routing and router.training exist only on constructed modules.
            # Validate in Bridge's construction hook before DDP and weight loading.
            self.provider.register_pre_wrap_hook(partial(_check_cp_seq_chunk_models, engine_config=self.engine_config))

        module, updated_tf_config = make_megatron_module(
            wrap_config=wrap_config,
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
            allowed_mismatched_params = ["output_layer.weight"] if self.is_value_model else []
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

        from verl.models.mcore.model_forward_fused import patch_fused_forward, unpatch_fused_forward

        if all(
            patch_fused_forward(model, self.model_config, engine_config=self.engine_config) for model in self.module
        ):
            return

        # The fused engine forward would pass kwargs the model cannot take. Restore any patched chunks.
        for model in self.module:
            unpatch_fused_forward(model)
        logger.warning_once("Fused kernels are not supported for this Megatron model; disabling.")
        self.engine_config.use_fused_kernels = False

    def _build_optimizer(self):
        from verl.utils.megatron.optimizer import get_megatron_optimizer, init_megatron_optim_config

        optim_config_megatron = init_megatron_optim_config(
            self.optimizer_config,
            use_distributed_optimizer=self.engine_config.use_distributed_optimizer,
            fp16=self.param_dtype == torch.float16,
            bf16=self.param_dtype == torch.bfloat16,
        )
        # LoRA+: lr_B = lora_plus_ratio * lr_A (read from the lora config dict).
        lora_cfg = getattr(self.model_config, "lora", {})
        lora_plus_ratio = float(lora_cfg.get("lora_plus_ratio", 1.0))
        if not 0 < lora_plus_ratio < float("inf"):  # also rejects nan
            raise ValueError(
                f"model.lora.lora_plus_ratio must be positive and finite (1.0 disables LoRA+), got {lora_plus_ratio}"
            )
        if lora_plus_ratio != 1.0 and self.optimizer_config.lr_warmup_init > 0:
            raise ValueError(
                "model.lora.lora_plus_ratio requires optim.lr_warmup_init=0: Megatron warms every param group up "
                "from one shared init lr, so the LoRA+ B/A ratio would not hold during warmup."
            )
        optimizer = get_megatron_optimizer(
            model=self.module,
            config=optim_config_megatron,
            lora_plus_ratio=lora_plus_ratio,
        )
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
        if _resolve_cp_seq_chunks(self.engine_config) == 1:
            self._build_tf_config()
        _check_dcp_unsupported_features(self.engine_config, self.model_config, tf_config=self.tf_config)

        # Reserve headroom for allocations the PyTorch caching allocator cannot see or
        # release: NCCL's own buffers are plain cudaMalloc/cudaHostAlloc calls, so with
        # the device full a 608-byte NCCL allocation fails and takes the rank down
        # while every other rank waits on a collective that rank will never issue
        # (observed at 768K). Capping the torch pool leaves that headroom.
        _frac = __import__("os").environ.get("DSV41_CUDA_MEM_FRACTION")
        if _frac:
            import torch as _torch

            _torch.cuda.set_per_process_memory_fraction(float(_frac))
        self.module = self._build_megatron_module()
        if __import__("os").environ.get("DSV41_MEM_PROBE", "0") == "1":
            import torch as _t
            import torch.distributed as _d

            _gb = 1 << 30
            print(
                f"[memprobe] rank={_d.get_rank()} after_build "
                f"alloc={_t.cuda.memory_allocated() / _gb:.2f}GB "
                f"resv={_t.cuda.memory_reserved() / _gb:.2f}GB "
                f"peak={_t.cuda.max_memory_allocated() / _gb:.2f}GB "
                f"free={_t.cuda.mem_get_info()[0] / _gb:.2f}GB "
                f"total={_t.cuda.mem_get_info()[1] / _gb:.2f}GB "
                f"nontorch="
                f"{(_t.cuda.mem_get_info()[1] - _t.cuda.mem_get_info()[0] - _t.cuda.memory_reserved()) / _gb:.2f}GB",
                flush=True,
            )
        # Profile-only NVTX instrumentation (DSV41_NVTX=1); no-op otherwise.

        if self._real_nvfp4_enabled:
            from verl.utils.real_nvfp4.megatron import attest_real_nvfp4_runtime, finish_real_nvfp4_module_init

            models = [unwrap_model(model_chunk) for model_chunk in self.module]
            finish_real_nvfp4_module_init(models)
            attest_real_nvfp4_runtime(
                models,
                self.tf_config,
                self.model_config.hf_config,
                *self._real_nvfp4_bf16_layers,
            )

        if self._qat_enabled and not self.engine_config.forward_only:
            from verl.utils.qat.modelopt import apply_qat_to_modules

            self.module = apply_qat_to_modules(self.module, self._qat_config)

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

        if os.environ.get("DSV41_MEM_PROFILE") == "1":
            log_memory_phase("model_build_done")

        # For forward_only, we don't need optimizer, lr_scheduler, checkpoint_mananager
        if self.engine_config.forward_only:
            self.optimizer = None
            self.lr_scheduler = None
            self.to(device="cpu", model=self._is_offload_param, optimizer=False, grad=False)
            log_gpu_memory_usage("After offload model during init (forward_only)", logger=logger)
            return

        self.optimizer = self._build_optimizer()
        if os.environ.get("DSV41_MEM_PROFILE") == "1":
            log_memory_phase("optimizer_init_done")
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
        # The logits-processor loss entries (top-k distillation and any other
        # resident-vocab-logits loss) leave large fp32 vocab tensors until backward ends;
        # free cached blocks before grad-norm all_reduce to reduce OOM on tight VRAM.
        if getattr(self, "_vocab_logits_active", False):
            get_torch_device().empty_cache()
        try:
            update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()
        finally:
            # Dump even when the optimizer step raises (for example, the OOM
            # that the history profiler is intended to diagnose).
            finish_memory_history(os.environ.get("DSV41_MEM_HISTORY"))

        if update_successful:
            # allgather already execute in optimizer.step in new megatron
            pass
        else:
            raise NotImplementedError("Megatron optimizer step failed. This should not happen")

        if os.environ.get("DSV41_MEM_PROFILE") == "1":
            log_memory_phase("step_end")
            log_memory_parts(self.module, self.optimizer)
            write_memory_snapshot(os.environ.get("DSV41_MEM_SNAPSHOT"))

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
        if self._is_offload_optimizer:
            # The optimizer was offloaded right after construction. Megatron's
            # HybridDeviceOptimizer rebuilds its sub-optimizers inside
            # ``load_state_dict`` and classifies every parameter by ``is_cuda``:
            # a shard that is still parked on the CPU gets no pinned copy and is
            # treated as a CPU-resident parameter, so once it is moved back to the
            # GPU the first ``step`` fails (KeyError / "assign a gradient with
            # device type 'cpu' to a tensor with device type 'cuda'"). Bring the
            # optimizer back to the GPU first; it is offloaded again below.
            load_megatron_optimizer(self.optimizer)
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
        if not forward_only:
            start_memory_history(os.environ.get("DSV41_MEM_HISTORY"))
        if __import__("os").environ.get("DSV41_MEM_STEP", "0") == "1":
            import torch as _t
            import torch.distributed as _d

            _gb = 1 << 30
            _free, _total = _t.cuda.mem_get_info()
            _resv = _t.cuda.memory_reserved()
            print(
                f"[memstep] rank={_d.get_rank()} "
                f"alloc={_t.cuda.memory_allocated() / _gb:.2f}GB "
                f"resv={_resv / _gb:.2f}GB "
                f"peak={_t.cuda.max_memory_allocated() / _gb:.2f}GB "
                f"free={_free / _gb:.2f}GB "
                f"nontorch={(_total - _free - _resv) / _gb:.2f}GB",
                flush=True,
            )
            if __import__("os").environ.get("DSV41_MEM_SAMPLER", "0") == "1":
                import threading as _th
                import time as _tm

                def _sample():
                    _last = 0.0
                    while True:
                        _tm.sleep(0.5)
                        try:
                            _a = _t.cuda.memory_allocated() / _gb
                            _r = _t.cuda.memory_reserved() / _gb
                            if _a - _last > 1.0 or _a > 100.0:
                                _last = _a
                                _f, _tt = _t.cuda.mem_get_info()
                                print(
                                    f"[memsample] rank={_d.get_rank()} t={_tm.time():.1f} "
                                    f"alloc={_a:.2f}GB resv={_r:.2f}GB "
                                    f"free={_f / _gb:.2f}GB "
                                    f"nontorch={(_tt - _f - _r) / _gb:.2f}GB",
                                    flush=True,
                                )
                        except Exception:
                            return

                if not getattr(_t.cuda, "_dsv41_sampler", None):
                    _t.cuda._dsv41_sampler = _th.Thread(target=_sample, daemon=True)
                    _t.cuda._dsv41_sampler.start()
        # Any loss entry that is computed from the vocab logits keeps them resident until
        # backward ends (see optimizer_step), so both flags arm the same cache flush.
        self._vocab_logits_active = tu.get_non_tensor_data(data, key="distillation_use_topk", default=False) or (
            tu.get_non_tensor_data(data, key="needs_vocab_logits", default=False)
        )
        tu.assign_non_tensor(data, sp_size=self.engine_config.context_parallel_size)

        _check_dcp_unsupported_features(self.engine_config, self.model_config, batch=data)

        # compute num_tokens in global batch for loss normalization
        loss_mask = data["loss_mask"]
        batch_num_tokens = (loss_mask.values().sum() if loss_mask.is_nested else loss_mask.sum()).to(get_device_id())
        # Static CP replicates each DP batch across its CP group. Reduce over DP
        # only: replicas can have different numbers of supervised tokens even when
        # they contain the same number of samples. DCP already replicates the whole
        # logical batch across the DP x CP plane.
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
        # Training only: validation/data-parallel inference forwards reuse this
        # path with their own batch size and a trailing partial batch, so the
        # accumulation contract does not apply to them.
        gas = tu.get_non_tensor_data(data, key="gradient_accumulation_steps", default=1)
        if not forward_only and gas and gas > 1 and n_micro_batch != gas:
            # The trainer derives gradient_accumulation_steps from
            # train_batch_size / (micro_batch_size_per_gpu * dp_size); the schedule
            # below accumulates one micro-batch at a time and steps the optimizer
            # once, so a different split would train a different effective batch
            # than the one requested. (prepare_micro_batches also divides the chunk
            # count by force_group_size, so a non-default value shows up here.)
            raise ValueError(
                f"gradient_accumulation_steps={gas} but the batch split produced "
                f"{n_micro_batch} micro-batches; check micro_batch_size_per_gpu, "
                "force_group_size and use_dynamic_bsz (accumulation requires "
                "use_dynamic_bsz=False)."
            )

        enable_routing_replay = tu.get_non_tensor_data(data, key="enable_routing_replay", default=False)
        record_r2_routes = (
            enable_routing_replay
            and forward_only
            and self.engine_config.router_replay.mode == "R2"
            and data.get("routed_experts", None) is None
        )

        for micro_batch in micro_batches:
            tu.assign_non_tensor(micro_batch, num_micro_batch=n_micro_batch)
            tu.assign_non_tensor(micro_batch, record_r2_routes=record_r2_routes)
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

        if enable_routing_replay:
            # Set to REPLAY mode: for R3 mode or actor update phase in R2 mode
            RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
            if record_r2_routes:
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

        if record_r2_routes:
            if not self.mini_layer_topk_idx_list:
                raise RuntimeError("R2 RECORD completed without capturing any router routes.")
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
            if record_r2_routes and dcp_group is None:
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
        if adapter_only:
            export_kwargs = {}
            if (
                self.model_config.lora.get("experts_shared_outer_loras", False)
                and getattr(self, "rollout_name", "vllm") == "vllm"
            ):
                # Serving wrappers choose the adapter format independently of the
                # base model's expert layout. Shared and mixed modes use 2D adapters.
                vllm_shared = getattr(self, "vllm_enable_moe_shared_loras", False)
                is_3d_moe = False
                if not vllm_shared and not getattr(self, "vllm_enable_mixed_moe_lora_format", False):
                    try:
                        from verl.workers.rollout.vllm_rollout.utils import is_3d_moe_vllm_model

                        is_3d_moe = is_3d_moe_vllm_model(self.model_config.hf_config)
                    except ModuleNotFoundError:
                        pass
                if vllm_shared:
                    export_kwargs["moe_shared_loras"] = True
                elif is_3d_moe:
                    export_kwargs["stack_3d_moe"] = True
                # No third branch: the serving export has no way to replicate a shared-outer
                # factor across experts (megatron-bridge removed the knob), so an adapter that
                # is neither in the 2D shared layout nor stacked 3D exports in the default
                # compact form. A consumer that cannot read that form fails at load time rather
                # than being handed an expanded copy that silently differs from what was trained.
            per_tensor_param = self.bridge.export_adapter_weights(self.module, **export_kwargs)
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

        if self._qat_enabled:
            per_tensor_param = ((name, tensor) for name, tensor in per_tensor_param if "_quantizer." not in name)
        if self._rollout_weight_quantizer is not None and not adapter_only:
            # QAT-trained master weights quantize to the same values the fake quant used in training.
            per_tensor_param = self._rollout_weight_quantizer(per_tensor_param)

        return per_tensor_param, peft_config

    def set_rollout_weight_quantizer(self, quantizer) -> None:
        self._rollout_weight_quantizer = quantizer

    def _mcore_export_index(self):
        """Build (once) the per-parameter delta export index: geometry specs and
        form-B probes derived from the bridge's conversion tasks. See
        :mod:`verl.workers.engine.megatron.delta_export`."""
        index = getattr(self, "_delta_export_index", None)
        if index is None:
            from .delta_export import build_export_index

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


# A micro-batch carries at most one resident top-k payload, and it enters the forward under a
# single pair of slot names (see forward_step). The teacher payload has two spellings: this
# fork's batch format uses `teacher_topk_*`, the distillation trainer's uses `teacher_*`. The
# sampler payload has one.
_TEACHER_PAYLOAD_KEYS = ("teacher_topk_token_ids", "teacher_topk_logprobs")
_TEACHER_PAYLOAD_ALIASES = ("teacher_ids", "teacher_logprobs")
_SAMPLER_PAYLOAD_KEYS = ("sampler_topk_token_ids", "sampler_topk_logprobs")


def _declares_keyword(func, name: str) -> bool:
    """Whether ``func`` can be called with ``name=``.

    The loss callable is a boundary to code this engine does not own (mint's closures, a
    user's loss, the distillation entries). Passing an argument a consumer's signature does
    not declare is a TypeError at the call site — after the forward has already run — so the
    capability is checked here rather than assumed. A callable that merely takes ``**kwargs``
    does *not* count: see the note below the parameter lookup.
    """
    import inspect

    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):  # builtins and some C callables expose no signature
        return False
    if name not in parameters:
        return False
    # A consumer that merely swallows `**kwargs` has not opted IN: a wrapper could forward
    # the value somewhere that treats it as supervision, or drop it silently, and neither
    # shows up as an error. Only an explicitly declared keyword counts.
    return parameters[name].kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


def _resident_logits_payload(batch, *, distillation_use_topk: bool, needs_vocab_logits: bool):
    """Pick the top-k payload that arms this micro-batch's logits processor.

    `distillation_use_topk` asks for the teacher's top-k; `needs_vocab_logits` asks for whatever
    distribution the loss prices against (a sampler's, for score centering). A micro-batch
    carries only one payload, so the distillation contract wins when both flags are set.

    Returns:
        ``(ids, logprobs)``, or ``None`` when neither flag is armed.
    """
    if distillation_use_topk:
        flag, spellings = "distillation_use_topk", (_TEACHER_PAYLOAD_KEYS, _TEACHER_PAYLOAD_ALIASES)
    elif needs_vocab_logits:
        flag, spellings = "needs_vocab_logits", (_SAMPLER_PAYLOAD_KEYS,)
    else:
        # No flag: whatever the batch carries is not this engine's payload, and it is not
        # ambiguous either — the batch builder is free to carry both for its own reasons.
        return None

    # A micro-batch carrying BOTH a teacher and a sampler top-k is ambiguous, and the two
    # mean different things to a consumer: one is a target distribution, the other is the
    # distribution the loss prices against. Preferring one silently would hand a consumer
    # the wrong KIND under a name it trusts, so the ambiguity is refused instead.
    def _first_present(keys):
        for key in keys:
            if batch.get(key, None) is not None:
                return key
        return None

    sampler_key = _first_present(_SAMPLER_PAYLOAD_KEYS[:1])
    # BOTH teacher spellings: the canonical one AND the alias. Checking only the canonical
    # key let a batch carrying `teacher_ids` next to a sampler payload through, which is the
    # same ambiguity under a different name.
    teacher_key = _first_present((*_TEACHER_PAYLOAD_KEYS[:1], *_TEACHER_PAYLOAD_ALIASES[:1]))
    if sampler_key is not None and teacher_key is not None:
        raise RuntimeError(
            f"the micro-batch carries BOTH a teacher top-k ({teacher_key}) and a sampler "
            f"top-k ({sampler_key}); this engine "
            "has one payload slot and the two are different distributions (teacher = target, "
            "sampler = the one the loss prices against). Arm one of them: send the batch "
            "with a single payload, or split it into two steps"
        )

    for ids_key, logprobs_key in spellings:
        ids = batch.get(ids_key, None)
        if ids is None:
            continue
        logprobs = batch.get(logprobs_key, None)
        if logprobs is None:
            raise RuntimeError(f"{flag} is set but the micro-batch carries {ids_key} without {logprobs_key}")
        return ids, logprobs

    if needs_vocab_logits and not distillation_use_topk and teacher_key is None:
        # Teacher export and lookup-based SFT need logits without an incoming
        # distribution. A partial sampler payload is still an error.
        if batch.get(_SAMPLER_PAYLOAD_KEYS[1], None) is not None:
            raise RuntimeError(f"{flag} carries {_SAMPLER_PAYLOAD_KEYS[1]} without {_SAMPLER_PAYLOAD_KEYS[0]}")
        return None
    expected = " or ".join(f"{ids_key}/{logprobs_key}" for ids_key, logprobs_key in spellings)
    raise RuntimeError(f"{flag} is set but the micro-batch carries no top-k payload (expected {expected})")


@EngineRegistry.register(model_type="language_model", backend="megatron")
class MegatronEngineWithLMHead(MegatronEngine):
    def prepare_model_inputs(self, batch: TensorDict):
        input_ids = batch["input_ids"]
        loss_mask = batch["loss_mask"].to(bool)
        multi_modal_inputs = extract_multi_modal_inputs(
            batch.get("multi_modal_inputs", []), input_ids=input_ids if input_ids.is_nested else None
        )

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
        needs_vocab_logits: bool = False,
        teacher_topk_ids: torch.Tensor | None = None,
        teacher_topk_logprobs: torch.Tensor | None = None,
        cp_layout: str = "zigzag",
        local_cp_size: int | None = None,
    ):
        if logits.shape[:2] != label.shape[:2]:
            # A bare assert hides which side is wrong; these are the two facts that matter.
            raise AssertionError(
                f"logits/label row mismatch: logits={tuple(logits.shape)} "
                f"label={tuple(label.shape)} "
                f"chunk={getattr(self, '_dsv41_chunk_ctx', None) and self._dsv41_chunk_ctx[:2]}"
            )
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
        if distillation_use_topk or needs_vocab_logits:
            # The payload arrives ALREADY sharded by gptmodel_forward_model_engine with the
            # label's cp_layout / use_fp8_padding / local_cp_size (it rode
            # logits_processor_args, see forward_step). Re-deriving the shard inside the loss is
            # the divergence this replaces: those three values are derived from the model config
            # inside the forward and are unreachable from the loss closure.
            processor_kwargs = {}
            if teacher_topk_ids is not None and teacher_topk_logprobs is not None:
                # Only advertise the slot when the batch actually armed a payload: a loss entry
                # that takes no payload keeps its existing signature.
                processor_kwargs["teacher_topk_ids"] = teacher_topk_ids
                processor_kwargs["teacher_topk_logprobs"] = teacher_topk_logprobs
            if (
                needs_vocab_logits
                and not distillation_use_topk
                and teacher_topk_ids is not None  # this lane armed a sampler payload
                and label is not None
                and _declares_keyword(logits_processor_func, "target_ids")
            ):
                # The SAMPLER-head lane (`needs_vocab_logits` without the distillation flag).
                # `label` here is the sharded, padded label that `logits` was built against,
                # which is the only copy a consumer can compare a head against: the batch's
                # own labels are unsharded and 32x the rows at CP=32. A loss that wants to
                # prove its head is aligned with the supervised position (rather than
                # correlating with it by accident) needs exactly this.
                processor_kwargs["target_ids"] = label
            # The distillation losses pack the teacher's top-k payload the way the student's
            # inputs were packed, and that information lives here, not in the loss -- but only
            # those losses declare the keywords. A policy loss plugged into the same slot does
            # not, so the triple is offered only to a target that can take it.
            if _declares_keyword(logits_processor_func, "quant_align_size"):
                processor_kwargs["quant_align_size"] = get_thd_quant_align_size(self.tf_config)
            if _declares_keyword(logits_processor_func, "cp_layout"):
                processor_kwargs["cp_layout"] = cp_layout
            if _declares_keyword(logits_processor_func, "local_cp_size"):
                processor_kwargs["local_cp_size"] = local_cp_size
            if _declares_keyword(logits_processor_func, "min_local_rows"):
                processor_kwargs["min_local_rows"] = get_thd_min_local_rows(self.tf_config, local_cp_size)
            chunk_ctx = getattr(self, "_dsv41_chunk_ctx", None)
            for name, value in (
                ("cp_chunk_index", chunk_ctx[0] if chunk_ctx else 0),
                ("cp_chunk_count", chunk_ctx[1] if chunk_ctx else 1),
            ):
                if _declares_keyword(logits_processor_func, name):
                    processor_kwargs[name] = value
            ret.update(
                logits_processor_func(
                    student_logits=logits_bak,
                    data=batch,
                    data_format=data_format,
                    **processor_kwargs,
                )
            )
        if not distillation_only:
            ret["log_probs"] = vocab_parallel_log_probs_from_logits(logits_bak, label)

        return ret

    def _dsv41_chunked_forward_step(
        self, batch, model, logits_processor_func, postprocess_micro_batch_func, chunk_count, *, forward_only=False
    ):
        """Expose the engine's single-pass THD forward to the Core schedule."""
        from megatron.core.pipeline_parallel.outer_chunk import forward_backward_outer_chunks

        reset_chunk_loss_mask_state(self)

        def forward_pass(index, tape):
            self._dsv41_chunk_ctx = (index, chunk_count, tape)
            self._dsv41_in_chunk = True
            try:
                return self.forward_step(iter([batch]), model, logits_processor_func, postprocess_micro_batch_func)
            finally:
                self._dsv41_in_chunk = False
                self._dsv41_chunk_ctx = None

        return forward_backward_outer_chunks(
            model=model,
            forward_pass=forward_pass,
            chunk_count=chunk_count,
            num_microbatches=tu.get_non_tensor_data(batch, key="num_micro_batch", default=1),
            cp_group_size=torch.distributed.get_world_size(self.get_context_parallel_group()),
            forward_only=forward_only,
            profile_phase=log_memory_phase if os.environ.get("DSV41_MEM_PROFILE") == "1" else None,
        )

    def forward_step(
        self, batch_iter: Iterator[TensorDict], model, logits_processor_func, postprocess_micro_batch_func
    ):
        batch: TensorDict = next(batch_iter)

        # Outer-sequence chunking is off unless a run asks for it: K=1 is the whole
        # sequence in one pass, exactly the path this engine ran before the feature
        # existed. The door is `_chunk_count > 1`, so the default below is the
        # single pass and nothing downstream changes.
        _chunk_count = _resolve_cp_seq_chunks(self.engine_config)
        if _chunk_count > 1 and not getattr(self, "_dsv41_in_chunk", False):
            _check_cp_seq_chunk_batch(batch, model)
            return self._dsv41_chunked_forward_step(
                batch,
                model,
                logits_processor_func,
                postprocess_micro_batch_func,
                _chunk_count,
                forward_only=postprocess_micro_batch_func.keywords["forward_only"],
            )

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
        needs_vocab_logits = tu.get_non_tensor_data(batch, key="needs_vocab_logits", default=False)
        pad_to_length_bucket = (
            self.engine_config.pad_to_length_bucket
            if self.engine_config.pad_to_length and self.engine_config.use_remove_padding
            else None
        )

        if pad_to_length_bucket is not None and (distillation_use_topk or needs_vocab_logits):
            raise RuntimeError(
                "pad_to_length is not supported with top-K distillation or any other resident-vocab-logits loss"
            )
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
        data_format = "thd" if self.engine_config.use_remove_padding else "bshd"
        forced_max_seqlen = tu.get_non_tensor_data(data=batch, key="forced_max_seqlen", default=None)
        if hasattr(unwrapped_model, "vp_stage"):
            vp_rank = unwrapped_model.vp_stage
        else:
            vp_rank = 0

        if self.enable_routing_replay and RouterReplayHelper.is_replay_backward_action(
            self.tf_config, vp_rank, model=unwrapped_model
        ):
            set_model_router_replay_action(unwrapped_model, RouterReplayAction.REPLAY_FORWARD)

        if self.enable_routing_replay and RouterReplayHelper.is_replay_forward_action(
            self.tf_config, vp_rank, model=unwrapped_model
        ):
            layers_topk_idx = model_inputs["routed_experts"]
            if layers_topk_idx is None:
                raise RuntimeError(
                    "router_replay REPLAY: micro_batch has no routed_experts. "
                    "R2 compute_log_prob must record and preserve routes before actor update."
                )
            replay_mask = None
            if self.engine_config.router_replay.mode == "R3":
                layers_topk_idx = align_r3_router_replay_data(
                    layers_topk_idx,
                    input_ids,
                    multi_modal_inputs=multi_modal_inputs,
                )
                replay_mask = build_r3_replay_mask(
                    input_ids,
                    batch["response_mask"],
                    multi_modal_inputs=multi_modal_inputs,
                )
            set_router_replay_data(
                layers_topk_idx,
                attention_mask,
                self.tf_config,
                vp_rank,
                replay_mask=replay_mask,
                local_cp_size=local_cp_size,
                model=unwrapped_model,
                data_format=data_format,
                forced_max_seqlen=forced_max_seqlen,
                use_fused_kernels=use_fused_kernels,
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
                position_ids=batch.get("position_ids", None),
            )
        else:
            if not isinstance(temperature, torch.Tensor):
                temperature = torch.tensor([temperature] * input_ids.shape[0], device=input_ids.device)

            temperature = temperature.to(torch.float32)
            assert temperature.shape[0] == input_ids.shape[0]
            temperature = verl_F.expand_as_nested(temperature, input_ids)  # (bsz, j1)
            from verl.models.mcore import get_mcore_engine_forward_fn

            forward_fn = get_mcore_engine_forward_fn(self.model_config.hf_config)
            logits_processor = partial(
                self._lm_head_logits_processor,
                calculate_sum_pi_squared=calculate_sum_pi_squared,
                calculate_entropy=calculate_entropy,
                distillation_use_topk=distillation_use_topk,
                distillation_only=distillation_only,
                needs_vocab_logits=needs_vocab_logits,
                logits_processor_func=logits_processor_func,
                batch=batch,
                data_format=data_format,
                # The distillation losses repack the teacher's top-k payload the way the student's
                # inputs were packed. Without these two the loss would fall back to the defaults
                # (zigzag, static CP) while the student packs contiguous / dynamic CP, and the KL
                # would be computed on misaligned tokens without any assertion firing.
                cp_layout=cp_layout,
                local_cp_size=local_cp_size,
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
            # Every value of this dict is sharded by gptmodel_forward_model_engine
            # (verl/models/mcore/model_forward.py) with the forward's own cp_layout /
            # use_fp8_padding / local_cp_size, i.e. exactly like input_ids and label. Putting the
            # payload here is therefore the whole feature: no sharding code is added, and
            # need_roll=(k == "label") keeps it unrolled, which is the contract (row t is the
            # distribution for the token row t predicts).
            payload = _resident_logits_payload(
                batch, distillation_use_topk=distillation_use_topk, needs_vocab_logits=needs_vocab_logits
            )
            if payload is not None:
                # `teacher_topk_*` is the slot name the logits processor sees for whichever
                # payload armed this micro-batch (distillation_use_topk or needs_vocab_logits).
                logits_processor_args["teacher_topk_ids"], logits_processor_args["teacher_topk_logprobs"] = payload

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

            _chunk_ctx = getattr(self, "_dsv41_chunk_ctx", None)
            _chunk_kwargs = (
                {"cp_chunk_index": _chunk_ctx[0], "cp_chunk_count": _chunk_ctx[1], "chunk_tape": _chunk_ctx[2]}
                if _chunk_ctx is not None
                else {}
            )
            output = forward_fn(
                model,
                input_ids,
                multi_modal_inputs,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
                pad_token_id=self.model_config.tokenizer.pad_token_id,
                data_format=data_format,
                mtp_enable_train=self.model_config.mtp.enable and self.model_config.mtp.enable_train,
                local_cp_size=local_cp_size,
                router_padding_mask=router_padding_mask,
                mtp_loss_normalization_factor=mtp_loss_normalization_factor,
                forced_max_seqlen=forced_max_seqlen,
                pad_to_length_bucket=pad_to_length_bucket,
                cp_layout=cp_layout,
                position_ids=batch.get("position_ids", None),
                **_chunk_kwargs,
            )

        # Router replay: record routing decisions for R2 mode
        if tu.get_non_tensor_data(batch, key="record_r2_routes", default=False):
            merge_router_topk_indices(
                attention_mask,
                input_ids,
                self.mini_layer_topk_idx_list,
                self.tf_config,
                vp_rank,
                local_cp_size=local_cp_size,
                model=unwrapped_model,
                data_format=data_format,
                use_fused_kernels=use_fused_kernels,
            )

        # Router replay: switch to backward replay mode for next backward pass
        if self.enable_routing_replay and RouterReplayHelper.is_replay_forward_action(
            self.tf_config, vp_rank, model=unwrapped_model
        ):
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
            loss_data = chunk_loss_data(self, model_output, data)
            loss, metrics = loss_function(
                model_output=model_output, data=loss_data, dp_group=self.get_data_parallel_group()
            )

            # scale loss by num_micro_batch because megatron will scale loss
            # by n_micro_batch inside pp schedule
            scaled_loss = loss * loss_data["num_micro_batch"]
        else:
            assert forward_only, "forward_only must be True when loss_function is None"
            loss = torch.tensor(1.0, device=device)
            scaled_loss = loss
            metrics = {}
        output = {"loss": loss.detach().item(), "metrics": metrics}
        if forward_only or not self.engine_config.dynamic_context_parallel:
            # Detach before this reaches Megatron's forward_data_store; see detach_tree.
            output["model_output"] = detach_tree(model_output)
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
                # Finalization sums these counts across DP and CP. Split the
                # exact logical count across the CP replicas, including any
                # remainder from removing padding.
                cp_size = self.engine_config.context_parallel_size
                real_tokens = self._routed_num_tokens(data)
                local_num_tokens = real_tokens // cp_size
                if cp_size > 1:
                    local_num_tokens = local_num_tokens + (real_tokens % cp_size > mpu.get_context_parallel_rank())
                local_num_tokens = local_num_tokens.to(torch.int)
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
            pad_token_id=self.model_config.tokenizer.pad_token_id,
            data_format="thd" if self.engine_config.use_remove_padding else "bshd",
            forced_max_seqlen=tu.get_non_tensor_data(data=batch, key="forced_max_seqlen", default=None),
            pad_to_length_bucket=(
                self.engine_config.pad_to_length_bucket
                if self.engine_config.pad_to_length and self.engine_config.use_remove_padding
                else None
            ),
            cp_layout=cp_layout,
            position_ids=batch.get("position_ids", None),
        )

        return output, partial(postprocess_micro_batch_func, data=batch)

    def prepare_model_outputs(self, output: dict | torch.Tensor, data: TensorDict):
        return {"values": output}
