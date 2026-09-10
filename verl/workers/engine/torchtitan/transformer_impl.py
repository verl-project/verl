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
"""
The concrete Engine implementation using PyTorch TorchTitan parallelism (FSDP2 + TP + PP)
"""

import gc
import importlib
import logging
import os
import re
from contextlib import nullcontext
from typing import Any, Callable, Optional

import torch
import torch.distributed
from tensordict import TensorDict
from torch.distributed.tensor import DTensor
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.config import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.distributed.context_parallel import prepare_context_parallel_input
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.train import Trainer

import verl.utils.torch_functional as verl_F
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import (
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.model import extract_multi_modal_inputs
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.config import HFModelConfig, TorchtitanEngineConfig, TorchtitanOptimizerConfig
from verl.workers.engine.torchtitan.utils import (
    NoOpDataLoader,
    derive_torchtitan_name_and_flavor,
    enable_fsdp_gradient_division,
    get_attention_masks,
)

from ..base import BaseEngine, BaseEngineCtx, EngineRegistry
from ..utils import enable_full_determinism, postprocess_batch_func, prepare_micro_batches


def _hf_entry_row_slots(name, spec, place, lidx, lval):
    """Dim-0 identity slot profile: slot ``e`` IS ``full[e]``, so a position's slot is one divmod away."""
    from ..spec import translate_flat_indices
    from ..utils import _prodshape

    slots = spec.hf_slots
    n_slots = len(slots)
    rows = int(spec.full_shape[0])
    assert n_slots == rows, (
        f"{name}: dim-0 identity slots need one slot per dim-0 row, got {n_slots} slots for {rows} rows; "
        "a converter that emits several HF tensors per row belongs on the to_hf_chunk path"
    )
    dtype_str = str(lval.dtype).replace("torch.", "")
    if lidx.numel() == 0:
        return slots, dtype_str, torch.zeros(n_slots, dtype=torch.int64), lidx.to(torch.int32), lval

    row_numel = max(_prodshape(spec.full_shape[1:]), 1)
    gidx = translate_flat_indices(lidx, place)
    edges = torch.arange(n_slots + 1, device=gidx.device, dtype=gidx.dtype) * row_numel
    # One D2H for the run lengths; the diff's nonzero already synced this stream.
    counts = torch.searchsorted(gidx, edges).diff().cpu()
    return slots, dtype_str, counts, (gidx % row_numel).to(torch.int32), lval


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


class TorchTitanEngine(BaseEngine):
    """
    Concrete Engine implementation using PyTorch TorchTitan parallelism.

    Supports model sharding with FSDP2, tensor parallelism, activation/optimizer offloading,
    LoRA, and sequence parallelism following the TorchTitan design.
    """

    def __init__(
        self,
        model_config: HFModelConfig,
        engine_config: TorchtitanEngineConfig,
        optimizer_config: TorchtitanOptimizerConfig,
        checkpoint_config: CheckpointConfig,
    ):
        """
        Initialize the TorchTitanEngine.

        Sets up distributed device meshes for tensor and data parallelism, LoRA, and offload policies.

        Args:
            model_config: Configuration for HuggingFace model.
            engine_config: Configuration for FSDP/TorchTitan engine (uses FSDP2).
            optimizer_config: Configuration for optimizer.
            checkpoint_config: Configuration for checkpointing.
        """
        super().__init__()

        self.model_config = model_config
        self.engine_config = engine_config
        self.optimizer_config = optimizer_config
        self.checkpoint_config = checkpoint_config

        # Derive torchtitan model name and flavor from HF config
        torchtitan_name, torchtitan_flavor = derive_torchtitan_name_and_flavor(self.model_config.hf_config)

        # Get ModelSpec from model registry
        model_module = importlib.import_module(f"torchtitan.models.{torchtitan_name}")
        model_spec = model_module.model_registry(torchtitan_flavor, attn_backend=self.engine_config.attn_type)

        optimizer = OptimizersContainer.Config(
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name=self.optimizer_config.name,
                    optimizer_kwargs={
                        "lr": self.optimizer_config.lr,
                        "eps": self.optimizer_config.eps,
                        "betas": (self.optimizer_config.betas[0], self.optimizer_config.betas[1]),
                        "weight_decay": self.optimizer_config.weight_decay,
                    },
                )
            ],
        )

        total_steps = self.optimizer_config.total_training_steps
        lr_warmup_steps = self.optimizer_config.lr_warmup_steps
        if lr_warmup_steps is None or lr_warmup_steps <= 0:
            lr_warmup_steps = int(self.optimizer_config.lr_warmup_steps_ratio * total_steps)

        lr_scheduler = LRSchedulersContainer.Config(
            warmup_steps=lr_warmup_steps,
            decay_type=self.optimizer_config.decay_type,
            min_lr_factor=self.optimizer_config.min_lr_factor,
        )
        parallelism = ParallelismConfig(
            data_parallel_replicate_degree=self.engine_config.data_parallel_replicate_size,
            data_parallel_shard_degree=self.engine_config.data_parallel_shard_size,
            fsdp_reshard_after_forward=self.engine_config.reshard_after_forward,
            tensor_parallel_degree=self.engine_config.tensor_parallel_size,
            pipeline_parallel_degree=self.engine_config.pipeline_parallel_size,
            context_parallel_degree=self.engine_config.context_parallel_size,
            expert_parallel_degree=self.engine_config.expert_parallel_size,
            spmd_backend=self.engine_config.spmd_backend,
        )
        checkpoint = CheckpointManager.Config(
            enable=True,
            initial_load_in_hf=True,
            initial_load_model_only=True,
            initial_load_path=model_config.path,
        )
        compile_config = CompileConfig(enable=self.engine_config.use_torch_compile)
        training_kwargs = {}
        if self.engine_config.max_seq_len is not None:
            training_kwargs["seq_len"] = self.engine_config.max_seq_len
        if self.engine_config.offload_policy or self.engine_config.forward_only:
            training = TrainingConfig(enable_cpu_offload=True, **training_kwargs)
        else:
            training = TrainingConfig(**training_kwargs)

        # Activation checkpointing mode. Note: under spmd_backend="spmd_types" with
        # eager execution (use_torch_compile=False), selective/full AC recompute runs
        # on the autograd backward thread where the thread-local SPMD mesh is inactive,
        # so spmd.assert_type() raises "no current mesh". Set activation_checkpoint="none"
        # (or enable torch.compile, which recomputes in-graph) in that configuration.
        activation_checkpoint = {
            "selective": SelectiveAC.Config,
            "full": FullAC.Config,
            "none": lambda: None,
        }[self.engine_config.activation_checkpoint]()

        # Construct Torchtitan's Trainer.Config
        self.config = Trainer.Config(
            model_spec=model_spec,
            hf_assets_path=self.model_config.path,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            parallelism=parallelism,
            checkpoint=checkpoint,
            compile=compile_config,
            training=training,
            activation_checkpoint=activation_checkpoint,
            # Use a no-op dataloader since verl has its own data loading
            dataloader=NoOpDataLoader.Config(),
            # Provide a concrete loss so Trainer.__init__ can build it;
            # verl uses its own loss function and ignores this one.
            loss=CrossEntropyLoss.Config(),
        )
        self.trainer = Trainer(self.config)

        self._init_device_mesh()

        # Re-enable FSDP's gradient division for verl's loss scaling.
        # TorchTitan disables gradient division by default (for global token normalization),
        # but verl's loss function multiplies by dp_size to compensate for gradient averaging.
        if self.engine_config.data_parallel_shard_size > 1:
            dp_size = self.get_data_parallel_size()
            for model_part in self.trainer.model_parts:
                enable_fsdp_gradient_division(model_part, dp_size)

        if self.engine_config.full_determinism:
            enable_full_determinism(seed=self.engine_config.seed)

        # set FSDP offload params
        self._is_offload_param = self.engine_config.param_offload
        self._is_offload_optimizer = self.engine_config.optimizer_offload

        if self.engine_config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.engine_config.use_torch_compile
            else entropy_from_logits
        )

    @property
    def is_param_offload_enabled(self) -> bool:
        return self._is_offload_param

    @property
    def is_optimizer_offload_enabled(self) -> bool:
        return self._is_offload_optimizer

    def is_mp_src_rank_with_outputs(self):
        """
        Whether the current rank is the first rank in model parallel group that contains model outputs
        """
        is_collect = True
        # TP: outputs are on TP rank 0
        if self.parallel_dims.tp > 1:
            tp_mesh = self.parallel_dims.get_optional_mesh("tp")
            is_collect = is_collect and (tp_mesh.get_local_rank() == 0)
        # PP: outputs are on the last PP rank
        if self.parallel_dims.pp > 1:
            pp_mesh = self.parallel_dims.get_optional_mesh("pp")
            is_collect = is_collect and (pp_mesh.get_local_rank() == self.parallel_dims.pp - 1)
        # CP: outputs are on CP rank 0
        if self.parallel_dims.cp > 1:
            cp_mesh = self.parallel_dims.get_optional_mesh("cp")
            is_collect = is_collect and (cp_mesh.get_local_rank() == 0)
        return is_collect

    def initialize(self):
        """
        Build the model, optimizer, and learning rate scheduler with TorchTitan parallelism.

        Applies device, dtype, and precision configurations, including mixed precision.
        Sets up checkpoint manager.
        """
        self.module = self.trainer.model_parts
        self.checkpointer = self.trainer.checkpointer
        # load initial HF weights
        self.checkpointer.load()

        if not self.engine_config.forward_only:
            self.optimizer = self.trainer.optimizers
            self.lr_scheduler = self.trainer.lr_schedulers
        else:
            self.optimizer = None
            self.lr_scheduler = None

        self.to(
            device="cpu",
            model=self._is_offload_param,
            optimizer=self._is_offload_optimizer,
            grad=self._is_offload_param,
        )

        log_gpu_memory_usage("After offload model/optimizer/grad during init", logger=logger)

    def _init_device_mesh(self):
        """Initialize the device mesh for TorchTitan style parallelism."""
        world_size = torch.distributed.get_world_size()
        self.parallel_dims = ParallelDims(
            dp_shard=self.engine_config.data_parallel_shard_size,
            dp_replicate=self.engine_config.data_parallel_replicate_size,
            cp=self.engine_config.context_parallel_size,
            tp=self.engine_config.tensor_parallel_size,
            pp=self.engine_config.pipeline_parallel_size,
            ep=self.engine_config.expert_parallel_size,
            world_size=world_size,
        )
        self.device_mesh = self.parallel_dims.build_mesh()

        # Mirror torchtitan's init_distributed (which verl bypasses): disable autograd
        # multithreading so backward-thread activation-checkpoint recompute can access the
        # thread-local SPMD mesh / process groups (e.g. current_spmd_mesh().get_group(...)).
        torch.autograd.set_multithreading_enabled(False)

    def train_mode(self, **kwargs):
        """Return a context manager for training mode."""
        return EngineTrainModeCtx(self, **kwargs)

    def eval_mode(self, **kwargs):
        """Return a context manager for evaluation mode."""
        return EngineEvalModeCtx(self, **kwargs)

    def get_data_parallel_rank(self):
        mesh = self._get_data_parallel_mesh()
        return 0 if mesh is None else mesh.get_local_rank()

    def get_data_parallel_size(self):
        return self.engine_config.data_parallel_shard_size * self.engine_config.data_parallel_replicate_size

    def get_data_parallel_group(self):
        mesh = self._get_data_parallel_mesh()
        if mesh is not None:
            return mesh.get_group()
        # If world_size == dp_size (e.g. single GPU, or all ranks are DP),
        # return WORLD so that collective ops in _postprocess_output
        # (allgather_dict_into_dict, all_reduce) still run and produce the
        # correct metric aggregation format.
        if torch.distributed.get_world_size() == self.get_data_parallel_size():
            return torch.distributed.group.WORLD
        return None

    def get_model_parallel_group(self):
        raise NotImplementedError

    def get_context_parallel_group(self):
        raise NotImplementedError

    def _get_data_parallel_mesh(self):
        """Get the data parallel mesh, handling hybrid/fully/replicate shard modes."""
        mesh = self.parallel_dims.get_optional_mesh("loss")
        if mesh is None:
            mesh = self.parallel_dims.get_optional_mesh("fsdp")
        if mesh is None:
            mesh = self.parallel_dims.get_optional_mesh("dp_replicate")
        return mesh

    def forward_backward_batch(self, data: TensorDict, loss_function: Callable, forward_only=False):
        """Perform forward and optionally backward pass on a batch."""
        tu.assign_non_tensor(data, sp_size=self.engine_config.tensor_parallel_size)

        # Compute num_tokens in global batch for loss normalization
        batch_num_tokens = data["loss_mask"].sum().to(get_device_id())
        dp_group = self.get_data_parallel_group()
        if dp_group is not None:
            torch.distributed.all_reduce(batch_num_tokens, op=torch.distributed.ReduceOp.SUM, group=dp_group)
        tu.assign_non_tensor(data, batch_num_tokens=batch_num_tokens.item())
        tu.assign_non_tensor(data, dp_size=self.get_data_parallel_size())

        micro_batches, indices = prepare_micro_batches(
            data=data,
            dp_group=self.get_data_parallel_group(),
            same_micro_num_in_dp=True,
        )

        output_lst = []

        ctx = torch.no_grad() if forward_only else nullcontext()

        # train_context activates the (thread-local) SPMD mesh required by spmd_types; it must
        # span backward too, since activation-checkpoint recompute re-runs the forward there.
        # record_function names each micro-batch so a forward-only stage (compute_log_prob /
        # compute_ref_log_prob) shows "micro_batch<i>" rows instead of a single anonymous forward.
        for micro_batch_idx, micro_batch in enumerate(micro_batches):
            with self.trainer.train_context(), ctx, torch.profiler.record_function(f"micro_batch{micro_batch_idx}"):
                loss, output = self.forward_step(micro_batch, loss_function=loss_function, forward_only=forward_only)
                if not forward_only:
                    loss.backward()
            output_lst.append(output)

        return postprocess_batch_func(output_lst=output_lst, indices=indices, data=data)

    def model_forward_step(
        self,
        *,
        inputs: torch.Tensor,
        extra_inputs: dict[str, torch.Tensor] | None = None,
        extra_kwargs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Perform a forward pass through the trainer model without backward.
        """
        model_parts = self.module
        parallel_dims = self.parallel_dims

        if parallel_dims.pp_enabled:
            raise NotImplementedError(
                "Pipeline parallelism is not yet supported in model_forward_step. "
                "This will be implemented in a follow-up PR."
            )
        else:
            # Non-PP forward. train_context (SPMD mesh) is set by the caller.
            assert len(model_parts) == 1
            pred = model_parts[0](inputs, **extra_inputs, **extra_kwargs)

        if isinstance(pred, DTensor):
            pred = pred.full_tensor()
        return pred

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        raise NotImplementedError("forward_step must be implemented in subclass")

    def optimizer_zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()

    def optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.module for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
            ep_enabled=self.parallel_dims.ep_enabled,
        )

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            logger.warning(f"grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()
        return grad_norm.item()

    def lr_scheduler_step(self):
        """Advance learning rate scheduler."""
        self.lr_scheduler.step()
        lr = self.lr_scheduler.schedulers[0].get_last_lr()[0]
        return lr

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """Move model and/or optimizer to CPU or GPU."""
        super().to(device=device, model=model, optimizer=optimizer, grad=grad)

        if self.engine_config.forward_only:
            return

        device_name = get_device_name()
        assert device in (device_name, "cpu")
        if device == device_name:
            if model:
                for module in self.module:
                    load_fsdp_model_to_gpu(module)
            if optimizer and self.optimizer is not None:
                load_fsdp_optimizer(self.optimizer, device)
            gc.collect()
        elif device == "cpu":
            if model:
                for module in self.module:
                    offload_fsdp_model_to_cpu(module)
            if optimizer and self.optimizer is not None:
                offload_fsdp_optimizer(self.optimizer)
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
        """Save checkpoint."""
        if self._is_offload_param:
            for module in self.module:
                load_fsdp_model_to_gpu(module)

        # Override TorchTitan's folder to use verl's path
        parent_dir = os.path.dirname(local_path)
        self.checkpointer.folder = parent_dir

        if max_ckpt_to_keep is not None:
            self.checkpointer.keep_latest_k = max_ckpt_to_keep

        self.checkpointer.save(curr_step=global_step)

        torch.distributed.barrier()
        if self._is_offload_param:
            for module in self.module:
                offload_fsdp_model_to_cpu(module)

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: int = True, **kwargs
    ) -> None:
        """Load checkpoint."""
        if self._is_offload_param:
            for module in self.module:
                load_fsdp_model_to_gpu(module)

        # Override TorchTitan's folder to use verl's path
        parent_dir = os.path.dirname(local_path)
        self.checkpointer.folder = parent_dir

        # Extract step number from path (verl uses global_step_N format)
        match = re.search(r"global_step_(\d+)", local_path)
        if match:
            step = int(match.group(1))
            self.checkpointer.load(step=step)
        else:
            # Fallback to latest
            self.checkpointer.load(step=-1)

        torch.distributed.barrier()
        if self._is_offload_param:
            for module in self.module:
                offload_fsdp_model_to_cpu(module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.optimizer)

    def _to_hf_named_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Re-key a torchtitan state dict to HuggingFace names, values untouched."""
        # Convert TorchTitan key names to HuggingFace key names (expected by vLLM)
        sd_adapter = self.checkpointer.sd_adapter
        if sd_adapter is not None:
            params = sd_adapter.to_hf(params)

        # When weight tying is enabled, the sd_adapter skips lm_head.weight during
        # to_hf() conversion (since it's the same tensor as embed_tokens.weight in
        # the torchtitan model). But vLLM needs lm_head.weight explicitly, so we
        # add it back as a reference to embed_tokens.weight.
        if "model.embed_tokens.weight" in params and "lm_head.weight" not in params:
            params["lm_head.weight"] = params["model.embed_tokens.weight"]
        return params

    def _expert_stack_slots(self, name: str, param: Any) -> Optional[list[tuple[str, tuple]]]:
        """If ``name`` is a fused expert stack, enumerate ALL its experts' HF tensors, else ``None``."""
        sd_adapter = self.checkpointer.sd_adapter
        from_hf_map = getattr(sd_adapter, "from_hf_map", None) if sd_adapter is not None else None
        if not from_hf_map or param.ndim < 2:
            return None
        abstract = re.sub(r"(\d+)", "{}", name, count=1)
        hf_template = next((hf for hf, titan in from_hf_map.items() if titan == abstract), None)
        if hf_template is None or hf_template.count("{}") != 2:
            return None
        layer_id = re.search(r"\d+", name).group(0)
        slot_shape = tuple(int(d) for d in param.shape[1:])
        return [(hf_template.format(layer_id, e), slot_shape) for e in range(int(param.shape[0]))]

    def _hf_delta_entry(self, name, spec, place, lidx, lval):
        """Build one parameter's final HF-coordinate delta entry: expert stack via slots, else identity."""
        from ..utils import _hf_entry_identity

        if spec.hf_slots is not None:
            return _hf_entry_row_slots(name, spec, place, lidx, lval)
        assert spec.to_hf_chunk is None, (
            f"{name}: the torchtitan engine has no opaque to-HF converters; "
            "a spec carrying one did not come from this engine"
        )
        return _hf_entry_identity(name, spec, place, lidx, lval)

    def get_per_tensor_param_delta_shard(self, **kwargs):
        """Yield FINAL HF-coordinate delta entries per parameter; needs a prior :meth:`prime_delta_snapshots`."""
        from ..utils import hf_delta_export

        self._delta_shard_snap = getattr(self, "_delta_shard_snap", {})
        gen, _ = self.get_per_tensor_param_shard()
        return hf_delta_export(gen, self._delta_shard_snap, self._hf_delta_entry), None

    def _assert_shard_export_supported(self) -> None:
        """Reject PP, the one layout whose local shard is not a block; every FSDP2 layout is one."""
        pd = self.parallel_dims
        if pd.pp_enabled:
            raise NotImplementedError(
                "the torchtitan sharded delta export does not support pipeline parallelism: "
                "each stage holds a disjoint slice of the model, so the export order is not "
                "identical across ranks the way the delta engine's lockstep gather requires "
                f"(pipeline_parallel_size={pd.pp})"
            )
        # TP, EP and HSDP replicate all stay blocks, so derive_dtensor_placement handles them.

    def get_per_tensor_param_shard(self, **kwargs):
        """Yield this rank's *local* shard ``(hf_name, local_flat_bf16, ShardSpec)`` instead of the full tensor."""
        self._assert_shard_export_supported()
        raw = {}
        for module in self.module:
            raw.update(module.state_dict())

        # Expert stacks go WHOLE with a slot table; to_hf would name only the local experts, breaking lockstep.
        stacks = {}
        for name, param in raw.items():
            slots = self._expert_stack_slots(name, param)
            if slots is not None:
                stacks[name] = slots
        params = self._to_hf_named_params({k: v for k, v in raw.items() if k not in stacks})
        device = get_device_id()  # local shards live on CPU under an offload policy

        from ..spec import ShardSpec

        def _shard(param):
            p = param.to(device, non_blocking=True)
            # The wire speaks bf16, but mixed precision keeps fp32 masters: cast before the diff sees them.
            if p.is_floating_point():
                p = p.to(torch.bfloat16, non_blocking=True)
            local = p.to_local() if isinstance(p, DTensor) else p
            return local.reshape(-1)

        def _gen():
            for name, param in params.items():
                yield name, _shard(param), ShardSpec.from_param(param)
            # Keyed by the torchtitan name: the entry is the stack, not any one HF tensor.
            for name, slots in stacks.items():
                spec = ShardSpec.from_param(raw[name])
                spec.hf_slots = slots
                yield name, _shard(raw[name]), spec

        return _gen(), None

    def get_per_tensor_param(self, **kwargs):
        for module in self.module:
            load_fsdp_model_to_gpu(module)

        params = {}
        for module in self.module:
            module_params = module.state_dict()
            params.update(module_params)

        if self._is_offload_param:
            for module in self.module:
                offload_fsdp_model_to_cpu(module)

        # Gather the stack before splitting it: to_hf splits first and drops every expert EFSDP holds elsewhere.
        stacks = {}
        for name, param in params.items():
            slots = self._expert_stack_slots(name, param)
            if slots is not None:
                stacks[name] = slots
        expert_stacks = [(params[name], slots) for name, slots in stacks.items()]
        dense = self._to_hf_named_params({k: v for k, v in params.items() if k not in stacks})
        params.clear()

        device = get_device_id()  # used when fsdp2 set cpu_offload_policy

        def _gen():
            # TODO: cast fp32 to bf16 to reduce weight sync overhead, need more fine-grained control, e.g MoE gate
            for name, param in dense.items():
                if isinstance(param, DTensor):
                    yield name, param.to(device, non_blocking=True).full_tensor().to(torch.bfloat16, non_blocking=True)
                else:
                    yield name, param
            # One stack at a time: the gathered (num_experts, ...) tensor is the peak allocation here.
            for stack, slots in expert_stacks:
                full = stack.to(device, non_blocking=True)
                if isinstance(full, DTensor):
                    full = full.full_tensor()
                full = full.to(torch.bfloat16, non_blocking=True)
                for e, (hf_name, _shape) in enumerate(slots):
                    yield hf_name, full[e].clone()
                del full

        # TODO: support Torchtitan PEFT
        return _gen(), None


class EngineEvalModeCtx(BaseEngineCtx):
    def __init__(self, engine: TorchTitanEngine, **kwargs):
        super().__init__(engine=engine, mode="eval", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, TorchTitanEngine)
        super().__enter__()
        for module in self.engine.module:
            module.eval()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, TorchTitanEngine)

        # Reshard the root FSDP module
        if self.engine.engine_config.data_parallel_shard_size > 1:
            for module in self.engine.module:
                module.reshard()

        super().__exit__(exc_type, exc_value, traceback)


class EngineTrainModeCtx(BaseEngineCtx):
    def __init__(self, engine: TorchTitanEngine, **kwargs):
        super().__init__(engine=engine, mode="train", **kwargs)

    def __enter__(self):
        assert isinstance(self.engine, TorchTitanEngine)
        super().__enter__()
        for module in self.engine.module:
            module.train()

    def __exit__(self, exc_type, exc_value, traceback):
        assert isinstance(self.engine, TorchTitanEngine)
        if self.zero_grad_on_exit or exc_type is not None:
            self.engine.optimizer_zero_grad()
        super().__exit__(exc_type, exc_value, traceback)


@EngineRegistry.register(model_type="language_model", backend=["torchtitan"], device=["cuda", "npu"])
class TorchTitanEngineWithLMHead(TorchTitanEngine):
    """TorchTitan engine implementation for language models with LM head."""

    def prepare_model_inputs(self, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        multi_modal_inputs = extract_multi_modal_inputs(micro_batch.get("multi_modal_inputs", []))
        input_ids = micro_batch["input_ids"]
        position_ids = micro_batch["position_ids"]
        output_args = {}

        if use_remove_padding:
            input_ids = input_ids.values().unsqueeze(0)
            if position_ids.dim() == 3:
                position_ids = position_ids.values().unsqueeze(1)
            else:
                position_ids = position_ids.values().unsqueeze(0)

            # Pad sequence length to be divisible by cp_world_size * BLOCK_SIZE
            # for context parallel block mask alignment requirement
            if self.parallel_dims.cp_enabled:
                cp_world_size = self.parallel_dims.get_mesh("cp").size(0)
                # FlexAttention CP uses BLOCK_SIZE=128 for block masks
                alignment = cp_world_size * 128
                seq_len = input_ids.shape[-1]
                output_args["cp_original_seq_len"] = seq_len
                if seq_len % alignment != 0:
                    pad_len = alignment - (seq_len % alignment)
                    input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)
                    if position_ids.dim() == 2:
                        # position_ids shape: [1, total_tokens]
                        position_ids = torch.nn.functional.pad(position_ids, (0, pad_len), value=0)
                    else:
                        # position_ids shape: [total_tokens, 1, num_heads]
                        position_ids = torch.nn.functional.pad(position_ids, (0, 0, 0, 0, 0, pad_len), value=0)

            labels = torch.roll(input_ids, shifts=-1, dims=1)
            attn_type = self.engine_config.attn_type
            attention_mask = get_attention_masks(
                input_batch=input_ids,
                positions=position_ids,
                attn_type=attn_type,
            )
        else:
            loss_mask = micro_batch["loss_mask"]
            pad_token_id = tu.get_non_tensor_data(data=micro_batch, key="pad_token_id", default=0)
            batch_size = micro_batch.batch_size[0]
            max_seq_len = max(input_ids.offsets().diff())

            # Round up max_seq_len to CP alignment requirement
            if self.parallel_dims.cp_enabled:
                cp_world_size = self.parallel_dims.get_mesh("cp").size(0)
                # FlexAttention CP uses BLOCK_SIZE=128 for block masks
                alignment = cp_world_size * 128
                max_seq_len = (max_seq_len + alignment - 1) // alignment * alignment

            labels = torch.roll(input_ids.values(), shifts=-1, dims=0)
            input_ids = torch.nested.to_padded_tensor(
                input_ids, padding=pad_token_id, output_size=(batch_size, max_seq_len)
            )

            if position_ids.dim() == 3:
                position_ids = torch.nested.to_padded_tensor(
                    position_ids, padding=0, output_size=(batch_size, 4, max_seq_len)
                ).transpose(0, 1)
            else:
                position_ids = torch.nested.to_padded_tensor(
                    position_ids, padding=0, output_size=(batch_size, max_seq_len)
                )

            attention_mask_list = [torch.ones_like(t, dtype=torch.int32) for t in loss_mask]
            attention_mask = torch.nested.as_nested_tensor(attention_mask_list, layout=torch.jagged)
            attention_mask = torch.nested.to_padded_tensor(
                attention_mask, padding=0, output_size=(batch_size, max_seq_len)
            )

        extra_inputs = {
            "positions": position_ids,
        }
        # For arguments, like attention_masks, we have to put them in a separate
        # dict as extra_inputs are not forwarded to other stages in PP, but
        # extra_kwargs are.
        extra_kwargs: dict[str, Any] = {"attention_masks": attention_mask}
        if self.parallel_dims.cp_enabled:
            input_ids, labels, extra_kwargs = prepare_context_parallel_input(
                input_ids,
                labels,
                extra_kwargs,
                self.parallel_dims.get_mesh("cp"),
                self.trainer.device,
                self.trainer.config.parallelism.context_parallel_load_balancer,
            )
            # prepare_context_parallel_input adds CP-sharded 'positions' to
            # extra_kwargs; move it to extra_inputs to avoid duplicate kwarg
            # when both dicts are unpacked in model_forward_step
            extra_inputs["positions"] = extra_kwargs.pop("positions")

        # TODO(jessicazhong): multimodal is not yet supported for Torchtitan engine
        extra_inputs.update(multi_modal_inputs)
        output_args["labels"] = labels
        return input_ids, extra_inputs, extra_kwargs, output_args

    def _cp_unshard_1d(self, sharded_tensor: torch.Tensor, original_seq_len: int) -> torch.Tensor:
        """All-gather a 1D CP-sharded tensor and undo load balancer reordering + padding.

        Args:
            sharded_tensor: 1D tensor of shape [local_seq_len] from this CP rank's shard.
            original_seq_len: The sequence length before CP alignment padding was added.

        Returns:
            1D tensor of shape [original_seq_len] with elements in original order.
        """
        from torch.distributed.tensor.experimental._context_parallel._load_balancer import _HeadTailLoadBalancer

        cp_mesh = self.parallel_dims.get_mesh("cp")
        cp_world_size = cp_mesh.size(0)
        local_seq_len = sharded_tensor.shape[0]
        padded_seq_len = local_seq_len * cp_world_size

        # All-gather across CP ranks
        gathered = torch.empty(padded_seq_len, dtype=sharded_tensor.dtype, device=sharded_tensor.device)
        torch.distributed.all_gather_into_tensor(gathered, sharded_tensor.contiguous(), group=cp_mesh.get_group())

        # Undo load balancer reordering
        load_balancer = _HeadTailLoadBalancer(padded_seq_len, cp_world_size, sharded_tensor.device)
        restore_indices = load_balancer._generate_indices(restore=True).squeeze(0)
        gathered = gathered[restore_indices]

        # Trim CP alignment padding
        return gathered[:original_seq_len]

    def prepare_model_outputs(self, logits, output_args, micro_batch: TensorDict):
        use_remove_padding = tu.get_non_tensor_data(data=micro_batch, key="use_remove_padding", default=True)
        pad_mode = tu.get_non_tensor_data(data=micro_batch, key="pad_mode", default=DatasetPadMode.NO_PADDING)
        assert pad_mode == DatasetPadMode.NO_PADDING, f"pad_mode {pad_mode} not supported"

        temperature = micro_batch["temperature"]
        calculate_entropy = tu.get_non_tensor_data(data=micro_batch, key="calculate_entropy", default=False)
        labels = output_args["labels"]
        model_output = {}

        input_ids = micro_batch["input_ids"]
        cu_seqlens = input_ids.offsets()
        if use_remove_padding:
            labels = labels.squeeze(0)
            logits_rmpad = logits.squeeze(0)
            # PyTorch's autograd doesn't allow in-place modification of views when gradients need to flow back
            logits_rmpad = logits_rmpad / temperature

            inplace_backward = True
            if calculate_entropy:
                inplace_backward = False
            log_probs = logprobs_from_logits(
                logits=logits_rmpad,
                labels=labels,
                inplace_backward=inplace_backward,
            )

            if calculate_entropy:
                if not self.engine_config.entropy_checkpointing:
                    if self.engine_config.entropy_from_logits_with_chunking:
                        entropy_rmpad = self.compute_entropy_from_logits(
                            logits_rmpad,
                            chunk_size=self.engine_config.entropy_from_logits_chunk_size,
                        )  # ((total_nnz / sp) + pad)
                    else:
                        entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)
                else:
                    entropy_rmpad = torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)

            # With CP, log_probs are for the local shard only. All-gather and
            # undo load balancer reordering to reconstruct the full sequence.
            if self.parallel_dims.cp_enabled:
                original_seq_len = output_args["cp_original_seq_len"]
                log_probs = self._cp_unshard_1d(log_probs.squeeze(0), original_seq_len)
                if calculate_entropy:
                    entropy_rmpad = self._cp_unshard_1d(entropy_rmpad, original_seq_len)

            log_probs = torch.nested.nested_tensor_from_jagged(log_probs.squeeze(0), cu_seqlens)
            if calculate_entropy:
                entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)
        else:
            logits.div_(temperature)
            if calculate_entropy:
                if not self.engine_config.entropy_checkpointing:
                    entropy = verl_F.entropy_from_logits(logits)
                else:
                    entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            seq_lengths = cu_seqlens.diff()
            starts = torch.zeros_like(seq_lengths, dtype=torch.int64)
            logits = torch.nested.narrow(logits, 1, starts, seq_lengths, layout=torch.jagged)
            logits_rmpad = torch.cat([t for t in logits.unbind()])
            log_probs = logprobs_from_logits(logits=logits_rmpad, labels=output_args["labels"])
            log_probs = torch.nested.nested_tensor_from_jagged(log_probs, cu_seqlens)
            if calculate_entropy:
                entropy = torch.nested.narrow(entropy, 1, starts, seq_lengths, layout=torch.jagged)
                entropy_rmpad = torch.cat([t for t in entropy.unbind()])
                entropy = torch.nested.nested_tensor_from_jagged(entropy_rmpad, cu_seqlens)

        model_output["log_probs"] = log_probs
        if calculate_entropy:
            model_output["entropy"] = entropy

        return model_output

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        device_name = get_device_name()
        micro_batch = micro_batch.to(get_device_id())
        input_ids, extra_inputs, extra_kwargs, output_args = self.prepare_model_inputs(micro_batch=micro_batch)

        with torch.autocast(device_type=device_name, dtype=torch.bfloat16):
            logits = self.model_forward_step(inputs=input_ids, extra_inputs=extra_inputs, extra_kwargs=extra_kwargs)

            model_output = self.prepare_model_outputs(logits=logits, output_args=output_args, micro_batch=micro_batch)

            if loss_function is not None:
                loss, metrics = loss_function(
                    model_output=model_output, data=micro_batch, dp_group=self.get_data_parallel_group()
                )
            else:
                assert forward_only, "forward_only must be True when loss_function is None"
                loss = torch.tensor(1.0, device=device_name)
                metrics = {}

            output = {
                "model_output": model_output,
                "loss": loss.detach().item(),
                "metrics": metrics,
            }

            return loss, output
