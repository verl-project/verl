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


import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from veomni.arguments import MixedPrecisionConfig, OpsImplementationConfig
from veomni.distributed import parallel_state
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models.auto import build_foundation_model
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids

import verl.utils.torch_functional as verl_F
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import fsdp_version
from verl.utils.model import convert_weight_keys
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.ulysses import (
    get_ulysses_sequence_parallel_group,
    set_ulysses_sequence_parallel_group,
)
from verl.utils.veomni.router_replay import RouterReplayAction, VeOmniRouterReplay
from verl.workers.config import HFModelConfig, VeOmniEngineConfig, VeOmniOptimizerConfig

from ..base import BaseEngineCtx, EngineRegistry
from ..fsdp.transformer_impl import FSDPEngine, FSDPEngineWithLMHead, FSDPEngineWithValueHead
from ..utils import enable_full_determinism, postprocess_batch_func, prepare_micro_batches
from .utils import (
    MOE_PARAM_HANDERS,
    VL_TYPE2INDEX,
    default_moe_param_handler,
    load_veomni_model_to_gpu,
    load_veomni_optimizer,
    offload_veomni_model_to_cpu,
    offload_veomni_optimizer,
)

logger = logging.getLogger(__file__)


class VeOmniEngine(FSDPEngine):
    _veomni_handles_position_ids = True

    def _apply_veomni_input_transforms(self, model_inputs: dict, micro_batch: TensorDict):
        """Apply VeOmni-specific input transforms shared by LM and value heads.

        Handles vision-language model masks, sequence parallel sharding,
        and flash attention kwargs from position_ids.
        """
        input_ids_rmpad = model_inputs["input_ids"]
        sp_enabled = parallel_state.get_parallel_state().sp_enabled
        sp_shard_collator = OmniSequenceShardCollator() if sp_enabled else None

        if self.module.config.model_type in VL_TYPE2INDEX.keys():
            image_mask = input_ids_rmpad == VL_TYPE2INDEX[self.module.config.model_type]["IMAGE_INPUT_INDEX"]
            video_mask = input_ids_rmpad == VL_TYPE2INDEX[self.module.config.model_type]["VIDEO_INPUT_INDEX"]
            model_inputs.update({"image_mask": image_mask, "video_mask": video_mask})

            if sp_enabled:
                sp_shard_collator(model_inputs)

        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        if use_remove_padding and model_inputs.get("position_ids", None) is not None:
            model_inputs.update(_prepare_veomni_flash_attention_kwargs(model_inputs["position_ids"]))
            if sp_enabled:
                model_inputs["position_ids"] = sp_shard_collator.sp_slice(model_inputs["position_ids"], dim=-1)

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: VeOmniEngineConfig,
        optimizer_config: VeOmniOptimizerConfig,
        checkpoint_config: CheckpointConfig,
        **kwargs,
    ):
        """
        Initialize the VeOmniEngine.

        Sets up distributed device meshes, LoRA, and offload policies based on config.

        Args:
            config: Configuration object with VeOmni and model settings.
        """

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config
        # VeOmniEngine only supports fsdp2.
        self.data_parallel_mode = "fsdp2"
        self.rank = dist.get_rank()

        fsdp_size = self.engine_config.fsdp_size
        world_size = dist.get_world_size()
        dp_size = world_size // self.engine_config.ulysses_parallel_size

        if fsdp_size < 0 or fsdp_size >= dp_size:
            data_parallel_replicate_size = 1
            data_parallel_shard_size = dp_size
        else:
            if dp_size % fsdp_size != 0:
                raise ValueError(
                    f"Data parallel size ({dp_size}) must be divisible by fsdp_size ({fsdp_size}). "
                    "Please adjust your parallel configuration."
                )
            data_parallel_replicate_size = dp_size // fsdp_size
            data_parallel_shard_size = fsdp_size

        parallel_state.init_parallel_state(
            dp_size=dp_size,
            dp_replicate_size=data_parallel_replicate_size,
            dp_shard_size=data_parallel_shard_size,
            extra_parallel_sizes=(self.engine_config.expert_parallel_size,),
            ulysses_size=self.engine_config.ulysses_parallel_size,
            dp_mode=self.data_parallel_mode,
        )

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

        self.use_remove_padding = self.model_config.use_remove_padding

        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload
        self._is_lora = self.model_config.lora_rank > 0

        self.use_ulysses_sp = parallel_state.get_parallel_state().sp_enabled
        self.ulysses_sequence_parallel_size = self.engine_config.ulysses_parallel_size

        if self.use_ulysses_sp:
            self.ulysses_parallel_group = parallel_state.get_parallel_state().device_mesh["sp"].get_group()
        else:
            self.ulysses_parallel_group = None

        if self.engine_config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.engine_config.use_torch_compile  #  use torch compile by default
            else entropy_from_logits
        )

        # Router replay (R2 / R3) for MoE models. Controller is attached in
        # initialize() after the model is built; here we only record intent.
        self._router_replay_mode: str = self.engine_config.router_replay.mode
        self.enable_routing_replay: bool = self._router_replay_mode != "disabled"
        self._router_replay: VeOmniRouterReplay | None = None
        if self.enable_routing_replay:
            logger.info("VeOmniEngine: router_replay enabled, mode=%s", self._router_replay_mode)

    def initialize(self):
        """
        Build the model, optimizer, and learning rate scheduler under VeOmni.

        Applies device, dtype, and precision configurations, including mixed precision.
        Sets up checkpoint manager and FLOPs counter.
        """
        self._build_model_optimizer()

        if self.enable_routing_replay:
            # Defense in depth: the VeOmniActorConfig check is the primary
            # fail-fast point and runs *before* engine init. By the time we get here,
            # ``_build_model_optimizer()`` has already finished — this
            # second check exists to catch direct ``VeOmniEngine``
            # instantiation paths that bypass the worker (e.g. unit tests,
            # standalone debug scripts) so the user gets a typed config
            # error instead of an opaque mid-step ``AttributeError`` on
            # ``input_ids.offsets()``.
            if not self.engine_config.use_remove_padding:
                raise RuntimeError(
                    "router_replay requires use_remove_padding=True. In VeOmni engine, "
                    "the non-remove-padding path also disables Ulysses SP slicing and "
                    "the fused-kernel log_probs path, and is not a tested production "
                    "configuration for MoE routing replay. Set "
                    "actor.model.use_remove_padding=True or "
                    "router_replay.mode='disabled'."
                )
            self._router_replay = VeOmniRouterReplay(sp_group=self.ulysses_parallel_group)
            # Fails loudly if the VeOmni build in the environment does not
            # export `set_active_replay` yet (plan requires upgrading VeOmni
            # or disabling router_replay).
            self._router_replay.install(self.module)

        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.module,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.model_config.get_processor(),
            checkpoint_config=self.checkpoint_config,
            trust_remote_code=self.model_config.trust_remote_code,
        )

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_optimizer,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def _build_optimizer(self, module):
        optimizer = build_optimizer(
            module,
            lr=self.optimizer_config.lr,
            betas=self.optimizer_config.betas,
            weight_decay=self.optimizer_config.weight_decay,
            optimizer_type=self.optimizer_config.optimizer,
        )
        get_optimizer_pre_hook = getattr(module, "get_optimizer_pre_hook", None)
        if get_optimizer_pre_hook is not None:
            optimizer_pre_hook = get_optimizer_pre_hook(module, module.config, self.data_parallel_mode)
            optimizer.register_step_pre_hook(optimizer_pre_hook)

        return optimizer

    def _build_lr_scheduler(self, optimizer):
        optim_config = self.optimizer_config
        lr_scheduler = build_lr_scheduler(
            optimizer,
            train_steps=optim_config.total_training_steps,
            lr=optim_config.lr,
            lr_min=optim_config.lr_min,
            lr_decay_style=optim_config.lr_scheduler_type,
            lr_decay_ratio=optim_config.lr_decay_ratio,
            lr_warmup_ratio=optim_config.lr_warmup_steps_ratio,
            lr_start=optim_config.lr_start,
        )

        return lr_scheduler

    def _get_model_config_path(self):
        """Return the config path (or PretrainedConfig) for build_foundation_model.

        Subclasses can override to modify the HF config before model construction
        (e.g. VeOmniEngineWithValueHead rewrites architectures to ForTokenClassification).
        """
        return self.model_config.local_hf_config_path

    def _build_model_optimizer(self):
        # build_foundation_model runs apply_ops_config(ops_implementation)
        # before constructing the model, so per-model device_patch files see
        # the resolved kernel backends.
        ops_implementation = OpsImplementationConfig(
            attn_implementation=self.engine_config.attn_implementation,
            moe_implementation=self.engine_config.moe_implementation,
            cross_entropy_loss_implementation=self.engine_config.cross_entropy_loss_implementation,
            rms_norm_implementation=self.engine_config.rms_norm_implementation,
            swiglu_mlp_implementation=self.engine_config.swiglu_mlp_implementation,
            rotary_pos_emb_implementation=self.engine_config.rotary_pos_emb_implementation,
            load_balancing_loss_implementation=self.engine_config.load_balancing_loss_implementation,
            rms_norm_gated_implementation=self.engine_config.rms_norm_gated_implementation,
            causal_conv1d_implementation=self.engine_config.causal_conv1d_implementation,
            chunk_gated_delta_rule_implementation=self.engine_config.chunk_gated_delta_rule_implementation,
        )

        veomni_mixed_precision_config = MixedPrecisionConfig(enable=self.engine_config.mixed_precision)

        # Load base model with specified configuration and dtype
        module = build_foundation_model(
            config_path=self._get_model_config_path(),
            weights_path=self.model_config.local_path,
            torch_dtype="float32" if veomni_mixed_precision_config.enable else "bfloat16",
            attn_implementation=self.engine_config.attn_implementation,
            ops_implementation=ops_implementation,
            init_device=self.engine_config.init_device,
        )
        log_gpu_memory_usage("After load base model", logger=logger)

        # Applies parallel strategies to the model.
        log_gpu_memory_usage("Before parallelize model", logger=logger)
        module = build_parallelize_model(
            module,
            init_device=self.engine_config.init_device,
            weights_path=self.model_config.local_path,
            enable_full_shard=self.engine_config.enable_full_shard,
            mixed_precision=veomni_mixed_precision_config,
            enable_gradient_checkpointing=self.model_config.enable_gradient_checkpointing,
            enable_fsdp_offload=self.engine_config.enable_fsdp_offload,
            basic_modules=list(
                set(getattr(module, "_no_split_modules", None) or []) | set(self.engine_config.basic_modules)
            ),
            enable_reentrant=self.engine_config.enable_reentrant,
            enable_forward_prefetch=self.engine_config.forward_prefetch,
        )
        log_gpu_memory_usage("After parallelize model", logger=logger)

        if not self.engine_config.forward_only:
            # Initialize optimizer with model parameters and config settings
            optimizer = self._build_optimizer(module)
            # Create learning rate scheduler with warmup and decay settings
            lr_scheduler = self._build_lr_scheduler(optimizer)
        else:
            optimizer = None
            lr_scheduler = None

        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.model_fwd_context, self.model_bwd_context = build_activation_offloading_context(
            self.model_config.enable_activation_offload,
            self.model_config.enable_gradient_checkpointing,
            self.engine_config.activation_gpu_limit,
        )

    def optimizer_step(self):
        """
        Perform an optimization step using the optimizer.
        """
        if hasattr(self.module, "clip_grad_norm_"):
            grad_norm = self.module.clip_grad_norm_(self.optimizer_config.clip_grad)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.module.parameters(), self.optimizer_config.clip_grad)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        return grad_norm.item()

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False) -> Any:
        """
        Perform a forward pass and optionally a backward pass on a batch of data.

        Args:
            data: The input data for the forward pass, typically containing tensors and metadata.
            loss_function: The loss function to optimize. See `verl.workers.roles.utils.losses` for examples.
            forward_only: If True, perform only the forward pass. If False, perform forward and backward pass.

        Returns:
            Any: The output of the forward pass, which can be used for loss computation or other purposes.
        """
        tu.assign_non_tensor(data, sp_size=parallel_state.get_parallel_state().ulysses_size)

        # compute num_tokens in global batch for loss normalization
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        torch.distributed.all_reduce(
            batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=self.get_data_parallel_group()
        )
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data, dp_group=self.get_data_parallel_group(), same_micro_num_in_dp=True
        )

        # Router replay state machine: decide RECORD vs REPLAY for this step.
        # RECORD: R2 compute_log_prob (forward_only=True).
        # REPLAY: R2 actor update, or R3 always (forward_only=True and False).
        rr_active = self.enable_routing_replay and tu.get_non_tensor_data(data, "enable_routing_replay", default=False)
        if rr_active:
            assert self._router_replay is not None
            if self._router_replay_mode == "R2" and forward_only:
                self._router_replay.begin_record()
            else:
                self._router_replay.begin_replay()

        # Wrap the per-step body in try/finally so the controller is always
        # reset to DISABLED even if forward / backward / postprocess raises.
        # Without this, an exception leaves _recorded / _targets pinned
        # (GPU memory) until the next successful step's begin_record/replay
        # clears them, which may never happen if the caller (Ray actor) tears
        # down the worker after the failure.
        try:
            output_lst = []
            # Per-microbatch metadata for RECORD aggregation (pad_size for SP pad trim,
            # cu_seqlens for per-sample split). Collected via side-channel on the
            # micro_batch TensorDict during prepare_model_inputs.
            pad_size_per_mb: list[int] = []
            cu_seqlens_per_mb: list[torch.Tensor] = []

            for micro_batch in micro_batches:
                if rr_active:
                    # Singular form: stash the entire list as one NonTensorData.
                    # The plural ``assign_non_tensor`` auto-dispatches lists to a
                    # NonTensorStack (batch_size=[len(list)]), which mutates the
                    # lazy-stacked micro_batch's batch_size and is rejected.
                    tu.assign_non_tensor_data(micro_batch, "_router_replay_pad_size_out", pad_size_per_mb)
                    tu.assign_non_tensor_data(micro_batch, "_router_replay_cu_seqlens_out", cu_seqlens_per_mb)

                with self.model_fwd_context:
                    loss, meta_info = self.forward_step(
                        micro_batch, loss_function=loss_function, forward_only=forward_only
                    )
                if not forward_only:
                    with self.model_bwd_context:
                        loss.backward()

                output_lst.append(meta_info)

                # Advance the per-mb counter on the controller. RECORD bumps
                # ``_mb_index`` so the next micro-batch's first router fire writes
                # into the next slot (recompute-in-backward of *this* mb has
                # already finished by here, so its detection window is closed).
                # Skip the bump after the last micro-batch: collect_recorded
                # cross-checks that every layer has exactly num_micro_batches
                # entries, so we must not advance past the last one.
                if rr_active and self._router_replay.action is RouterReplayAction.RECORD:
                    if len(output_lst) < len(micro_batches):
                        self._router_replay.advance_record_microbatch()

            if rr_active and self._router_replay.action is RouterReplayAction.RECORD:
                # ``collect_recorded`` already checks pad_size_per_mb internally;
                # cu_seqlens_per_mb is engine-local, so we cross-check here for
                # a clear error if anything (e.g. a deep-copying TD op) breaks
                # the by-reference side-channel contract.
                n_mb = len(micro_batches)
                if not (len(pad_size_per_mb) == len(cu_seqlens_per_mb) == n_mb):
                    raise RuntimeError(
                        f"router_replay RECORD aggregation: side-channel lengths "
                        f"diverge — pad_size_per_mb={len(pad_size_per_mb)}, "
                        f"cu_seqlens_per_mb={len(cu_seqlens_per_mb)}, "
                        f"num_micro_batches={n_mb}."
                    )
                # Per-microbatch recorded indices -> per-sample nested tensors,
                # attached to each output_lst entry so postprocess_batch_func can
                # unbind + restore the original batch order, matching log_probs /
                # entropy flow.
                per_mb_flat = self._router_replay.collect_recorded(
                    pad_size_per_mb=pad_size_per_mb,
                    num_micro_batches=n_mb,
                )
                for i, (flat, cu) in enumerate(zip(per_mb_flat, cu_seqlens_per_mb, strict=True)):
                    output_lst[i].setdefault("model_output", {})["routed_experts"] = (
                        torch.nested.nested_tensor_from_jagged(flat, offsets=cu)
                    )

            result = postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)
            return result
        finally:
            if rr_active:
                self._router_replay.clear()

    def get_data_parallel_rank(self):
        return parallel_state.get_parallel_state().device_mesh.get_local_rank("dp")

    def get_data_parallel_size(self):
        return torch.distributed.get_world_size() // parallel_state.get_parallel_state().ulysses_size

    def get_data_parallel_group(self):
        if parallel_state.get_parallel_state().ulysses_size > 1:
            return parallel_state.get_parallel_state().device_mesh.get_group(mesh_dim="dp")
        else:
            return torch.distributed.group.WORLD

    def get_model_parallel_group(self):
        raise NotImplementedError

    def get_context_parallel_group(self):
        raise NotImplementedError

    def is_mp_src_rank_with_outputs(self):
        """
        Whether the current rank is the first rank in model parallel group that contains model outputs
        """
        if parallel_state.get_parallel_state().ulysses_size > 1:
            is_collect = parallel_state.get_parallel_state().device_mesh["ulysses"].get_local_rank() == 0
        else:
            is_collect = True
        return is_collect

    def train_mode(self, **kwargs):
        """
        Return a context manager that switches to training mode with VeOmni-specific handling.

        Includes parameter and optimizer offload entry/exit.
        """
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """
        Return a context manager that switches to evaluation mode with VeOmni-specific handling.

        Includes activation offload entry/exit.
        """
        return EngineEvalModeCtx(self, **kwargs)

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.
        Note that this function executes irrespective of offload config. It serves as manual control.

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
        """
        super(FSDPEngine, self).to(device=device, model=model, optimizer=optimizer, grad=grad)

        device_name = get_device_name()

        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                load_veomni_model_to_gpu(self.module)
            if optimizer and self.optimizer is not None:
                load_veomni_optimizer(self.optimizer, device)
        elif device == "cpu":
            if model:
                offload_veomni_model_to_cpu(self.module)
            if optimizer and self.optimizer is not None:
                offload_veomni_optimizer(self.optimizer)
        else:
            raise ValueError(f"Invalid device type: {device}")

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Save VeOmni checkpoint, handling parameter offload as needed.
        """
        origin_module_device = next(self.module.parameters()).device.type
        if self._is_offload_param or origin_module_device == "cpu":
            load_veomni_model_to_gpu(self.module)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_veomni_model_to_cpu(self.module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        """
        Load VeOmni checkpoint, restoring parameters and optimizer state.
        """
        if self._is_offload_param:
            load_veomni_model_to_gpu(self.module)

        self.checkpoint_manager.load_checkpoint(
            local_path=local_path, hdfs_path=hdfs_path, del_local_after_load=del_local_after_load
        )

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_veomni_model_to_cpu(self.module)

        if self._is_offload_optimizer:
            offload_veomni_optimizer(self.optimizer)

    def get_per_tensor_param(self, **kwargs):
        load_veomni_model_to_gpu(self.module)

        params = self.module.state_dict()
        params = convert_weight_keys(params, getattr(self.module, "_fsdp_wrapped_module", self.module))

        if self._is_offload_param:
            offload_veomni_model_to_cpu(self.module)

        ps = parallel_state.get_parallel_state()
        model_type = getattr(self.module.config, "model_type", "default")
        process_func = MOE_PARAM_HANDERS.get(model_type, default_moe_param_handler)

        def param_generator():
            for name, param in params.items():
                unsharded_tensor = param.full_tensor() if isinstance(param, DTensor) else param

                is_expert_layer = "mlp.experts." in name
                is_proj = any(p in name for p in ["down_proj", "gate_proj", "up_proj", "gate_up_proj"])

                if is_expert_layer and is_proj and ps.ep_enabled:
                    ep_rank, ep_size = ps.ep_rank, ps.ep_size
                    buffer = torch.empty_like(unsharded_tensor)  # [num_experts/ep_size, H, I]
                    for src_ep_rank in range(ep_size):
                        tensor = unsharded_tensor if src_ep_rank == ep_rank else buffer
                        torch.distributed.broadcast(tensor, group_src=src_ep_rank, group=ps.ep_group)
                        yield from process_func(name, tensor, ep_rank=src_ep_rank)

                else:
                    if is_expert_layer:
                        yield from process_func(name, unsharded_tensor, ep_rank=0)
                    else:
                        yield name, unsharded_tensor

        # TODO: support VeOmni LoRA
        return param_generator(), None


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: VeOmniEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, VeOmniEngine)
        super().__enter__()
        self.prev_sp_group = get_ulysses_sequence_parallel_group()
        set_ulysses_sequence_parallel_group(self.engine.ulysses_parallel_group)
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, VeOmniEngine)
        set_ulysses_sequence_parallel_group(self.prev_sp_group)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if parallel_state.get_parallel_state().dp_shard_size > 1:
            if fsdp_version(self.engine.module) == 1:
                self.engine.module._handle.reshard(True)
            elif fsdp_version(self.engine.module) == 2:
                self.engine.module.reshard()

        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: VeOmniEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, VeOmniEngine)
        super().__enter__()
        self.prev_sp_group = get_ulysses_sequence_parallel_group()
        set_ulysses_sequence_parallel_group(self.engine.ulysses_parallel_group)
        # TODO: Switch to eval mode after Integrating the CI environment
        # VeOmni (ref: https://github.com/ByteDance-Seed/VeOmni/pull/421)
        self.engine.module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, VeOmniEngine)
        set_ulysses_sequence_parallel_group(self.prev_sp_group)
        if self.zero_grad_on_exit or exc_type is not None:
            self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@dataclass
class OmniSequenceShardCollator:
    """
    Data collator to chunk inputs along the sequence length.
    """

    # features to slice sequence dimension
    sp_slice_features: dict[str, int] = field(
        default_factory=lambda: {
            "input_ids": -1,
            "labels": -1,
            "pixel_values": 0,
            "pixel_values_videos": 0,
        },
        metadata={"help": "features to slice sequence dimension."},
    )

    # features to padding sequence dimension
    padding_features: dict[str, int] = field(
        default_factory=lambda: {
            "pixel_values": 0,
            "pixel_values_videos": 0,
        },
        metadata={"help": "features to padding sequence dimension."},
    )

    # padding scale for padding features
    padding_scale: dict[str, int] = field(
        default_factory=lambda: {"pixel_values": 4, "pixel_values_videos": 4},
        metadata={"help": "padding scale for padding features."},
    )

    def __post_init__(self):
        self.sp_size = parallel_state.get_parallel_state().sp_size
        self.sp_rank = parallel_state.get_parallel_state().sp_rank

    def sp_slice(self, feature: torch.Tensor, dim: int = -1) -> dict[str, "torch.Tensor"]:
        seq_length = feature.size(dim)
        sp_chunk_size = (seq_length + self.sp_size - 1) // self.sp_size
        return feature.narrow(dim, self.sp_rank * sp_chunk_size, sp_chunk_size)

    def sp_padding(
        self, tensor: "torch.Tensor", dim: int = -1, pad_value: int = 0, pad_scale: int = 1
    ) -> "torch.Tensor":
        """
        Pads a tensor with pad_length to aligns tensor with sp size.
        """
        seq_length = tensor.size(dim)
        scale_sp_size = self.sp_size * pad_scale

        sp_chunk_size = (seq_length + scale_sp_size - 1) // scale_sp_size
        pad_size = sp_chunk_size * scale_sp_size - seq_length
        if pad_size == 0:
            return tensor

        pad_shape = list(tensor.shape)
        pad_shape[dim] = pad_size
        pad = torch.full(pad_shape, fill_value=pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat((tensor, pad), dim=dim)

    def __call__(self, batch: Sequence[dict[str, "torch.Tensor"]]) -> dict[str, "torch.Tensor"]:
        for key in batch.keys():
            if key in self.padding_features.keys():
                batch[key] = self.sp_padding(
                    batch[key],
                    dim=self.sp_slice_features.get(key, -1),
                    pad_value=self.padding_features[key],
                    pad_scale=self.padding_scale.get(key, 1),
                )

        # sp slice
        for key in batch.keys():
            if key in self.sp_slice_features.keys():
                batch[key] = self.sp_slice(batch[key], dim=self.sp_slice_features[key])

        return batch


def _prepare_veomni_flash_attention_kwargs(position_ids: torch.Tensor) -> dict[str, torch.Tensor | int]:
    """Normalize packed position_ids layout and derive varlen FlashAttention kwargs.

    Supported formats for use_remove_padding=true:
        - 2D: (1, total_nnz) - standard packed format
        - 3D: (rope_dim, 1, total_nnz) - VeRL mRoPE packed format
    """
    if position_ids.dim() == 2:
        # (1, total_nnz) - standard packed format
        fa_position_ids = position_ids
    elif position_ids.dim() == 3:
        # (rope_dim, 1, total_nnz) - VeRL mRoPE packed format
        if position_ids.shape[1] == 1:
            fa_position_ids = position_ids[0]
        else:
            raise ValueError(
                f"Unsupported 3D position_ids shape: {tuple(position_ids.shape)}, expected (rope_dim, 1, total_nnz)"
            )
    else:
        raise ValueError(
            f"Unsupported position_ids rank: {position_ids.dim()}, "
            f"expected 2 (1, total_nnz) or 3 (rope_dim, 1, total_nnz)"
        )

    (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(fa_position_ids)
    return {
        "cu_seq_lens_q": cu_seq_lens_q,
        "cu_seq_lens_k": cu_seq_lens_k,
        "max_length_q": max_length_q,
        "max_length_k": max_length_k,
    }


@EngineRegistry.register(model_type="language_model", backend=["veomni"], device=["cuda", "npu"])
class VeOmniEngineWithLMHead(VeOmniEngine, FSDPEngineWithLMHead):
    def prepare_model_inputs(self, micro_batch: TensorDict):
        model_inputs, output_args = super().prepare_model_inputs(micro_batch)
        self._apply_veomni_input_transforms(model_inputs, micro_batch)

        # Activate VeOmni's chunk_logprobs path: ForCausalLMLoss short-circuits
        # to per-token log_probs/entropy on return_log_probs=True. Pass the
        # already-rolled labels as shift_labels so chunk_logprobs skips its
        # internal causal shift and the output seq length matches the input —
        # prepare_model_outputs().squeeze(0) then lands at (total_nnz,).
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        use_fused_kernels = tu.get_non_tensor_data(data=micro_batch, key="use_fused_kernels", default=False)
        if use_fused_kernels and use_remove_padding:
            input_ids_rmpad = model_inputs["input_ids"]
            shift_labels = output_args["input_ids_rmpad_rolled"].unsqueeze(0)
            model_inputs["labels"] = input_ids_rmpad
            model_inputs["shift_labels"] = shift_labels
            model_inputs["return_log_probs"] = True

            # Pass teacher top-K tensors so ForCausalLMLoss routes to
            # chunk_topk_distill_function for fused distillation. TD keys
            # teacher_ids / teacher_logprobs are populated by verl's native
            # distillation pipeline (see verl/trainer/distillation/losses.py).
            distillation_use_topk = tu.get_non_tensor_data(data=micro_batch, key="distillation_use_topk", default=False)
            if distillation_use_topk and "teacher_ids" in micro_batch.keys():
                if "teacher_logprobs" not in micro_batch.keys():
                    raise ValueError(
                        "teacher_ids present without teacher_logprobs; "
                        "both must be provided together for fused top-K distillation."
                    )
                # Kernel kwarg names follow veomni's chunk_topk_distill_function API.
                teacher_topk_ids = micro_batch["teacher_ids"].values().unsqueeze(0)
                teacher_topk_log_probs = micro_batch["teacher_logprobs"].values().unsqueeze(0)
                # SP-slice along seqlen (dim=1); teacher tensors are 3D
                # (1, total_nnz, K) so use slice_input_tensor directly —
                # ulysses_pad_and_slice_inputs hardcodes 2D.
                if self.use_ulysses_sp:
                    from verl.utils.ulysses import slice_input_tensor

                    teacher_topk_ids = slice_input_tensor(teacher_topk_ids, dim=1, padding=True)
                    teacher_topk_log_probs = slice_input_tensor(teacher_topk_log_probs, dim=1, padding=True)
                model_inputs["teacher_topk_ids"] = teacher_topk_ids
                model_inputs["teacher_topk_log_probs"] = teacher_topk_log_probs

        # Router replay plumbing. Two responsibilities:
        #   (1) snapshot the ulysses pad_size for this micro-batch so
        #       forward_backward_batch can trim it during RECORD aggregation;
        #   (2) during REPLAY, slice this micro-batch's routed_experts along
        #       the same pad+SP rule that super().prepare_model_inputs used
        #       for input_ids, then feed per-layer targets to the controller.
        self._maybe_push_router_replay_state(micro_batch, output_args)

        return model_inputs, output_args

    def _maybe_push_router_replay_state(self, micro_batch: TensorDict, output_args: dict) -> None:
        rr = self._router_replay
        if rr is None:
            return

        # RR's pack/unpack rule assumes ``input_ids`` is a jagged NestedTensor
        # built by ``left_right_2_no_padding`` — both the cu_seqlens snapshot
        # below and ``slice_microbatch_replay_targets`` rely on the same
        # pad+slice rule that ``super().prepare_model_inputs`` applies to it.
        # The engine-init guard already rejects ``use_remove_padding=False``,
        # but this check defends against future callers that bypass the guard
        # (debug tools, sub-engine instantiations) and converts an opaque
        # ``AttributeError: 'Tensor' object has no attribute 'offsets'`` into
        # a clear invariant failure.
        input_ids = micro_batch["input_ids"]
        if not (isinstance(input_ids, torch.Tensor) and input_ids.is_nested):
            raise RuntimeError(
                "router_replay: micro_batch['input_ids'] must be a jagged "
                "NestedTensor (produced by left_right_2_no_padding). Got "
                f"type={type(input_ids).__name__}, is_nested="
                f"{getattr(input_ids, 'is_nested', False)}. RR currently "
                "supports only the use_remove_padding=True data path."
            )

        pad_sink = tu.get_non_tensor_data(micro_batch, "_router_replay_pad_size_out", default=None)
        if pad_sink is not None:
            pad_sink.append(int(output_args.get("pad_size", 0)))

        # Snapshot cu_seqlens for per-sample split during RECORD aggregation.
        cu_sink = tu.get_non_tensor_data(micro_batch, "_router_replay_cu_seqlens_out", default=None)
        if cu_sink is not None:
            cu_sink.append(input_ids.offsets().clone())

        if rr.action is RouterReplayAction.REPLAY:
            routed = micro_batch.get("routed_experts", None)
            if routed is None:
                # Strict: no silent fallback. A missing routed_experts in
                # REPLAY means the trainer-side plumbing (compute_log_prob ->
                # update_actor for R2, or rollout -> compute_log_prob for R3)
                # has dropped the field somewhere upstream — silently running
                # the actor update on native router decisions would re-introduce
                # exactly the floating-point divergence RR is meant to remove.
                raise RuntimeError(
                    "router_replay REPLAY: micro_batch missing 'routed_experts'. "
                    "Verify that compute_log_prob (R2) or the rollout path (R3) "
                    "attached routed_experts to the batch before this engine "
                    "call, and that left_right_2_no_padding preserved it."
                )
            # Nested-jagged [bs, seq, L, topk] → rmpad values [mb_nnz, L, topk].
            # slice_microbatch_replay_targets reuses slice_input_tensor to
            # mirror the exact pad+slice rule super().prepare_model_inputs
            # already applied to input_ids.
            flat = routed.values() if hasattr(routed, "values") else routed
            per_layer = rr.slice_microbatch_replay_targets(flat)

            # Per-token replay mask — R3 only.
            #
            # R2 RECORD captures the actor's full-sequence routing
            # (prompt + response) in compute_log_prob, so REPLAY
            # substitutes uniformly. Applying a response-only mask
            # would let prompt tokens fall through to a fresh native
            # call — atomic-add nondeterminism in the fused MoE
            # experts means that call may pick different indices than
            # RECORD, breaking the bit-equal forward guarantee R2
            # exists for. The divergence then propagates through
            # attention KV into response logits and gradients.
            #
            # R3 RECORD runs at the rollout backend, which only
            # captures response-token routing during generation;
            # prefill is not instrumented and prompt-token positions
            # carry zero placeholders. Substituting those zeros sends
            # every prompt token's topk slots to expert 0, corrupting
            # the EP all-to-all token distribution. R3 must mask
            # prompt tokens out and let them go through native routing.
            replay_mask = None
            if self._router_replay_mode == "R3":
                response_mask = micro_batch.get("response_mask", None)
                if response_mask is None:
                    raise RuntimeError(
                        "router_replay R3: micro_batch missing 'response_mask'. "
                        "R3 needs the response_mask to know which tokens have "
                        "real recorded routing (response) vs. zero placeholders "
                        "(prompt). Verify left_right_2_no_padding preserved it."
                    )
                # Build a per-rmpad-token bool mask in the SAME [total_nnz]
                # layout as ``input_ids.values()`` (also matches
                # ``routed_experts.values()`` since they share the same
                # ``index_first_axis(unpad_input(input_ids).indices)``
                # transform): per sample i, ``prompt_lens[i]`` zeros
                # followed by ``response_lens[i]`` ones.
                #
                # We CANNOT use ``micro_batch['loss_mask']`` directly —
                # after ``left_right_2_no_padding`` it's still a strided
                # ``(bs, max_response_len)`` tensor (not nested), which
                # neither has the right shape nor a valid ``.values()``
                # for a strided layout.
                total_lens = input_ids.offsets().diff()  # (bs,)
                response_lens = response_mask.sum(dim=-1).to(total_lens.dtype)  # (bs,)
                prompt_lens = total_lens - response_lens  # (bs,)
                # Defensive: response_lens > total_lens means the
                # response_mask describes more tokens than the input has,
                # which is data corruption. Failing here surfaces a clear
                # message instead of letting repeat_interleave silently
                # produce a malformed mask.
                if torch.any(prompt_lens < 0):
                    raise RuntimeError(
                        f"router_replay R3: response_mask sum exceeds total token "
                        f"count for some samples — prompt_lens={prompt_lens.tolist()}. "
                        "Likely cause: response_mask was not aligned with the "
                        "input_ids the actor sees (rollout/trainer plumbing bug)."
                    )
                bs = total_lens.size(0)
                # values=[0, 1, 0, 1, ...] (length 2*bs), counts=[p_0, r_0, p_1, r_1, ...]
                values = torch.tensor([False, True], dtype=torch.bool, device=total_lens.device).repeat(bs)
                counts = torch.stack([prompt_lens, response_lens], dim=1).flatten()
                mask_flat = torch.repeat_interleave(values, counts)
                # Defensive: the constructed mask must align with
                # routed_experts at the rmpad layer, otherwise the
                # downstream ``torch.where(mask, target, native)`` would
                # silently misalign and the EP all-to-all would still
                # blow up. Fail-fast here with a clearer message.
                if mask_flat.numel() != flat.size(0):
                    raise RuntimeError(
                        f"router_replay R3: constructed replay_mask has "
                        f"{mask_flat.numel()} entries but routed_experts.values() "
                        f"has {flat.size(0)}. response_mask + input_ids.offsets() "
                        "do not describe the same total token count."
                    )
                # Mirror the same pad+slice rule used for routed_experts.
                replay_mask = rr.slice_microbatch_replay_mask(mask_flat)

            rr.set_microbatch_targets(per_layer, replay_mask=replay_mask)


@EngineRegistry.register(model_type="value_model", backend=["veomni"], device=["cuda", "npu"])
class VeOmniEngineWithValueHead(VeOmniEngine, FSDPEngineWithValueHead):
    """Value model engine using VeOmni's FSDP2 + sequence parallelism.

    Combines VeOmniEngine (model init, parallel state, activation offloading)
    with FSDPEngineWithValueHead (TokenClassification output -> per-token values).
    """

    def _get_model_config_path(self):
        """Return a modified HF config that loads ForTokenClassification(num_labels=1).

        Uses HF's AutoModelForTokenClassification model mapping to resolve the
        canonical ForTokenClassification class name for this model family, then
        sets config.architectures so VeOmni's MODELING_REGISTRY dispatches to it.
        """
        from transformers import AutoModelForTokenClassification
        from veomni.models.auto import build_config

        config = build_config(self.model_config.local_hf_config_path)
        config.num_labels = 1
        config.classifier_dropout = 0.0
        config.hidden_dropout = "0"
        config.summary_dropout_prob = 0.0
        config.tie_word_embeddings = False
        token_cls = AutoModelForTokenClassification._model_mapping.get(type(config), None)
        if token_cls is None:
            raise ValueError(f"No ForTokenClassification class in transformers for {type(config).__name__}.")
        config.architectures = [token_cls.__name__]
        return config

    def prepare_model_inputs(self, micro_batch: TensorDict):
        model_inputs, output_args = super().prepare_model_inputs(micro_batch)
        self._apply_veomni_input_transforms(model_inputs, micro_batch)
        return model_inputs, output_args
