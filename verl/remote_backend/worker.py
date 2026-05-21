"""Backend-agnostic forwarder worker.

Owns single-controller dispatch annotations, FlopsCounter / MFU, and
metric aggregation. Every payload encoding decision (loss, wire format,
parallelism) lives behind ``RemoteBackend.compute_log_prob`` and
``RemoteBackend.update_actor`` — the worker stays backend-agnostic.

Generic tensor / metric helpers live in
:mod:`verl.remote_backend.worker_utils` so backends and other modules can
import them without pulling in the Worker class or its dispatch
decorators.
"""

from __future__ import annotations

from typing import Any

import torch
from codetiming import Timer
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.remote_backend.base import RemoteBackend, RemoteBackendRegistry
from verl.remote_backend.worker_utils import make_njt, normalize_backend_metrics
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils import hf_tokenizer
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.flops_counter import FlopsCounter
from verl.utils.profiler import DistProfiler, DistProfilerExtension
from verl.utils.py_functional import append_to_dict
from verl.workers.config import ActorConfig, HFModelConfig


# ---------------------------------------------------------------------- #
# Generic forwarder worker
# ---------------------------------------------------------------------- #

class RemoteBackendActorRolloutRefWorker(Worker, DistProfilerExtension):
    """CPU-only forwarder; assumes ``data["input_ids"]`` is nested-jagged."""

    def __init__(self, config: DictConfig, role: str, **kwargs):
        Worker.__init__(self)
        self.config = config
        self.role = role
        self._is_actor = self.role in ("actor", "actor_rollout", "actor_rollout_ref")
        self._is_rollout = self.role in ("rollout", "actor_rollout", "actor_rollout_ref")
        self._is_ref = self.role in ("ref", "actor_rollout_ref")

        self.main_config: DictConfig = kwargs["main_config"]
        backend_handle: dict = kwargs["backend_handle"]

        backend_name = self.main_config.trainer.get("remote_backend")
        if backend_name is None:
            raise ValueError(
                "RemoteBackendActorRolloutRefWorker requires "
                "main_config.trainer.remote_backend to be set."
            )

        # `RemoteBackendRegistry.get` lazy-imports the adapter module
        # (Ray child procs don't inherit the driver's import side-effects).
        # `from_config(handle=...)` is the sole public constructor; the
        # handle is what makes this a re-attach rather than a fresh init.
        backend_cls = RemoteBackendRegistry.get(backend_name)
        self.backend: RemoteBackend = backend_cls.from_config(
            self.main_config, handle=backend_handle
        )

        DistProfilerExtension.__init__(
            self, DistProfiler(rank=self.rank, config=None, tool_config=None)
        )

        if self._is_actor:
            self.model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model)
            self.actor_config: ActorConfig = omega_conf_to_dataclass(self.config.actor)
            self.actor_config.model_config = self.model_config

            trust_remote_code = self.config.model.get("trust_remote_code", False)
            self.tokenizer = hf_tokenizer(self.config.model.path, trust_remote_code=trust_remote_code)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.pad_token_id = self.tokenizer.pad_token_id

            self.flops_counter = FlopsCounter(self.model_config.hf_config)

    # ------------------------------------------------------------------ #
    # Lifecycle (ONE_TO_ALL on a single forwarder => safe; multiple
    # forwarders are forbidden by `RemoteBackendTrainer`'s assert)
    # ------------------------------------------------------------------ #

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        self._register_dispatch_collect_info("actor", dp_rank=self.rank, is_collect=True)
        self._register_dispatch_collect_info("ref", dp_rank=self.rank, is_collect=True)
        self._register_dispatch_collect_info("rollout", dp_rank=self.rank, is_collect=True)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def to(self, device, model=True, optimizer=True, grad=True):
        return  # backend owns device residency

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_loss_fn(self, loss_fn):
        return  # loss is owned by backend (or sent in-band per call)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def reset(self):
        return  # backend owns engine state

    # ------------------------------------------------------------------ #
    # Core dispatched ops (delegate to backend.compute_log_prob / update_actor)
    # ------------------------------------------------------------------ #

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="ref"))
    @DistProfiler.annotate(color="olive", role="ref_compute_log_prob")
    async def compute_ref_log_prob(self, data: TensorDict) -> TensorDict:
        return await self._run_log_prob(data, ref=True, calculate_entropy=False)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="blue", role="actor_compute_log_prob")
    async def compute_log_prob(self, data: TensorDict) -> TensorDict:
        return await self._run_log_prob(data, ref=False, calculate_entropy=True)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    async def update_actor(self, data: TensorDict) -> TensorDict:
        return await self._run_update_actor(data)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, global_steps: int = None):
        await self.backend.update_weights()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        assert self._is_actor, "save_checkpoint only supported on actor role"
        await self.backend.save_checkpoint()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def sleep_replicas(self):
        return  # backend owns rollout lifecycle

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    async def destroy(self):
        if self.backend is not None:
            await self.backend.destroy()
            self.backend = None

    # ------------------------------------------------------------------ #
    # Generic glue: dispatch -> backend.{compute_log_prob,update_actor} -> TensorDict
    # ------------------------------------------------------------------ #
    # Static, config-derived engineering values (e.g. `max_token_len_per_gpu`)
    # are read directly from `self.config` by the backend at init time —
    # this forwarder no longer re-injects them into every batch.

    async def _run_log_prob(self, data: TensorDict, *, ref: bool, calculate_entropy: bool) -> TensorDict:
        result = await self.backend.compute_log_prob(
            data,
            ref=ref,
            calculate_entropy=calculate_entropy,
            rollout_n=self.actor_config.rollout_n,
            temperature=data.get("temperature"),
            pad_token_id=self.pad_token_id,
        )

        model_output_raw = result.get("model_output", {})
        model_output: dict[str, Any] = {}
        if "log_probs" in model_output_raw:
            model_output["log_probs"] = make_njt(data, model_output_raw["log_probs"])
        if calculate_entropy and "entropy" in model_output_raw:
            model_output["entropy"] = make_njt(data, model_output_raw["entropy"])

        metrics = dict(result.get("metrics", {}))
        metrics.setdefault("mfu", 0.0)

        return tu.get_tensordict(tensor_dict=model_output, non_tensor_dict={"metrics": metrics})

    async def _run_update_actor(self, data: TensorDict) -> TensorDict:
        from tensordict import NonTensorData

        # `train_batch` style fields the backend may want to read from data.
        if "global_token_num" not in data.keys():
            tu.assign_non_tensor(
                data,
                global_token_num=NonTensorData(data["input_ids"].offsets().diff().tolist()),
                update_lr_scheduler=True,
                disable_auto_offload=tu.get(data, key="disable_auto_offload", default=False),
            )

        with Timer(name="train_batch", logger=None) as timer:
            result = await self.backend.update_actor(
                data,
                actor_config=self.actor_config,
                pad_token_id=self.pad_token_id,
                rollout_n=self.actor_config.rollout_n,
                temperature=data.get("temperature"),
            )
        delta_time = timer.last

        loss_raw = result["loss"]
        metrics_raw = dict(result.get("metrics", {}))
        global_token_num = result.get("global_token_num")
        images_seqlens = data.get("images_seqlens", None) if hasattr(data, "get") else None

        loss = torch.sum(torch.tensor(loss_raw if isinstance(loss_raw, list) else [loss_raw]))

        grad_norm = metrics_raw.pop("grad_norm", None)
        lr = metrics_raw.pop("lr", None)
        last_lr = metrics_raw.pop("last_lr", None)

        final_metrics: dict[str, Any] = dict(metrics_raw)
        final_metrics["loss"] = loss
        if grad_norm is not None:
            final_metrics["grad_norm"] = grad_norm
        if lr is not None:
            final_metrics["lr"] = lr
        elif last_lr is not None:
            final_metrics["lr"] = last_lr

        if global_token_num is not None:
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_token_num, delta_time, images_seqlens=images_seqlens
            )
            final_metrics["mfu"] = estimated_flops / promised_flops

        wrapped = normalize_backend_metrics(final_metrics)

        out: dict[str, Any] = {}
        append_to_dict(out, wrapped)

        return tu.get_tensordict(tensor_dict={}, non_tensor_dict={"metrics": out}).cpu()
