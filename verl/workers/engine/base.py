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
The abstract base class defining the interface for model training engines.
"""

import os
from abc import abstractmethod
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Generator, Optional

import torch
from tensordict import TensorDict

from verl.utils.device import get_device_name, get_vendor
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids


class BaseEngine:
    """
    Abstract base class defining the interface for model training engines. Interface is subject to
    change before release.

    Engine implementations must subclass BaseEngine and provide concrete behavior for all methods.
    """

    def initialize(self):
        """
        Instantiate or load the model, optimizer, and learning rate scheduler.

        Should prepare all components necessary for training or evaluation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_param_offload_enabled(self) -> bool:
        """Whether parameter offloading is enabled."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_optimizer_offload_enabled(self) -> bool:
        """Whether optimizer offloading is enabled."""
        raise NotImplementedError

    @property
    def is_grad_offload_enabled(self) -> bool:
        """Whether gradient offloading is enabled.

        Existing engines historically move gradient buffers together with
        parameters.  They inherit this compatibility default; engines with an
        independent gradient policy should override it.
        """

        return self.is_param_offload_enabled

    def train_mode(self, **kwargs):
        """
        Context manager entry for switching the engine and model into training mode.

        Usage:
            with engine.train_mode():
                # runs in training mode
        """
        raise NotImplementedError

    def eval_mode(self, **kwargs):
        """
        Context manager entry for switching the engine and model into evaluation mode.

        Usage:
            with engine.eval_mode():
                # runs in evaluation mode
        """
        raise NotImplementedError

    def optimizer_zero_grad(self):
        """
        Zero the gradients of the optimizer.
        """
        raise NotImplementedError

    def optimizer_step(self):
        """
        Perform an optimization step using the optimizer.
        """
        raise NotImplementedError

    def lr_scheduler_step(self):
        """
        Advance the learning rate scheduler by one step.

        Returns:
            current_lr (float or list[float]): Updated learning rate(s).
        """
        raise NotImplementedError

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
        raise NotImplementedError

    def train_batch(self, data: TensorDict, loss_function: Callable) -> Any:
        """
        Perform a training step on a batch of data.

        Args:
            data: The input data for training, typically containing tensors and metadata.
            loss_function: A function that computes the loss and metrics given a batch and predictions.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the aggregated training metrics for the batch.
        """
        maybe_fix_3d_position_ids(data)

        self.optimizer_zero_grad()
        outputs = self.forward_backward_batch(data, loss_function, forward_only=False)
        grad_norm = self.optimizer_step()
        self.mark_parameters_updated()
        if self.is_mp_src_rank_with_outputs():
            assert "grad_norm" not in outputs["metrics"]
            outputs["metrics"]["grad_norm"] = grad_norm
        return outputs

    def mark_parameters_updated(self) -> None:
        """Advance the local parameter version after any operation that may mutate weights.

        Disk-offload backends use this version to decide whether a previously
        committed parameter generation still represents the live model. Custom
        weight-loading integrations must call this method before mutating model
        parameters.
        """

        self._parameter_version = getattr(self, "_parameter_version", 0) + 1

    def _record_disk_param_snapshot(self) -> None:
        """Record that the current parameter version has committed to disk."""

        self._disk_param_snapshot_version = getattr(self, "_parameter_version", 0)

    def _is_disk_param_snapshot_current(self) -> bool:
        """Whether the committed disk generation matches the live parameters."""

        return getattr(self, "_disk_param_snapshot_version", None) == getattr(self, "_parameter_version", 0)

    def infer_batch(self, data: TensorDict, loss_function: Optional[Callable] = None) -> Any:
        """
        Perform inference on a batch of data.

        Args:
            data: The input data for inference, typically containing tensors and metadata.

        Returns:
            Any: The output of the inference, which can be used for predictions or other purposes.
        """
        # see comments from train_batch
        maybe_fix_3d_position_ids(data)

        with torch.no_grad():
            outputs = self.forward_backward_batch(data, loss_function, forward_only=True)
        return outputs

    def get_per_tensor_param(self) -> tuple[Generator[tuple[str, torch.Tensor], None, None], Optional[dict]]:
        """
        Get a generator that yields per-tensor parameters and optional peft config.

        Returns:
            Generator[tuple[str, torch.Tensor]]: A generator that yields tuples of parameter names and tensors.
            Optional[dict]: None without an adapter, else a PEFT config carrying peft_type,
                task_type, r, lora_alpha, target_modules, lora_dropout and bias -- the first
                two as enum members, not strings.
        """
        raise NotImplementedError

    def get_per_tensor_param_shard(self, **kwargs) -> tuple[Generator, Optional[dict]]:
        """
        Like :meth:`get_per_tensor_param`, but yields each rank's *local* parameter shard
        instead of all-gathering full tensors.

        Used by checkpoint engines that operate on shards (e.g. the ``delta_sharded``
        backends, which byte-diff each rank's shard locally and only gather the changed
        elements), so the export never materializes full tensors on any rank.

        Implementations yield ``(name, local_shard, ShardSpec)`` -- the spec (see
        :mod:`verl.workers.engine.spec`) carries all placement knowledge
        (offset translation or a dense rebuild callable), so the consuming checkpoint
        engine stays trainer-agnostic.

        Returns:
            Generator: A generator that yields per-parameter local shards with placement metadata.
            Optional[dict]: Optional peft config.
        """
        raise NotImplementedError

    # Host-memory policy for the delta diff base: pinned (cudaHostAlloc) gives a
    # faster H2D on the diff read-back but competes with every other pinned pool
    # on the node; backends whose memory profile makes that competition
    # dangerous override this to False (see MegatronEngine).
    delta_pin_snapshots: bool = True

    def prime_delta_snapshots(self) -> None:
        """
        Snapshot this rank's CURRENT shards as the delta diff base. Called right after the
        seed sync (which streams :meth:`get_per_tensor_param`'s full HF export):
        weights do not move during the sync, so the snapshots equal exactly what the
        rollout side received and the first
        :meth:`get_per_tensor_param_delta_shard` diff is correct.

        Concrete here: it only consumes :meth:`get_per_tensor_param_shard`, so any
        engine that implements the shard export gets it for free.
        """
        from verl.utils.device import is_cuda_available
        from verl.workers.engine.utils import prime_delta_snapshots

        self._delta_shard_snap = getattr(self, "_delta_shard_snap", {})
        gen, _ = self.get_per_tensor_param_shard()
        prime_delta_snapshots(gen, self._delta_shard_snap, pin=is_cuda_available and self.delta_pin_snapshots)

    def get_per_tensor_param_delta_shard(self, **kwargs) -> tuple[Generator, Optional[dict]]:
        """
        Yield the delta engine's per-parameter payloads in FINAL HF coordinates:
        ``(slots, dtype_str, counts, hf_idx, hf_val, gather_group)``, in an order
        identical on every rank (non-contributing replicas yield zero-count entries
        but stay in the lockstep sequence). ``slots`` enumerates the HF tensors this
        parameter maps to (identity params: itself; fused params: their hf_slots),
        and ``hf_idx``/``hf_val`` are the changed elements since the previous export,
        already converted to HF coordinates.

        Everything backend-specific lives behind this call: the weight->HF naming,
        the to-HF conversion, the diff and its base. A backend that already keeps the
        previous step's weights (e.g. Decoupled PPO) can diff against that instead of
        a dedicated snapshot; :func:`verl.workers.engine.utils.hf_delta_export`
        implements the default pinned-CPU-snapshot strategy (primed by
        :meth:`prime_delta_snapshots`). The consuming delta engine only batches,
        gathers and ships.

        Returns:
            Generator: A generator that yields per-parameter HF-coordinate delta entries.
            Optional[dict]: Optional peft config.
        """
        raise NotImplementedError

    def get_data_parallel_size(self):
        raise NotImplementedError

    def get_data_parallel_rank(self):
        raise NotImplementedError

    def get_data_parallel_group(self):
        raise NotImplementedError

    def to(self, device: str, model: bool = True, optimizer: bool = True, grad: bool = True):
        """
        Move model parameters, optimizer states, or both to the specified device.

        Args:
            device: Target device identifier.
            model: If True, move the model.
            optimizer: If True, move the optimizer states.
            grad: If True, move the gradient buffer.
        """
        if grad:
            assert model, "Gradient buffers must be moved to device along with model parameters"

    def offload(
        self,
        *,
        model: bool = True,
        optimizer: bool = True,
        grad: bool = True,
        preserve_grad: bool = True,
    ) -> None:
        """Move selected state to its configured offload target.

        The base implementation preserves the legacy CPU behavior.  Backends
        supporting non-device targets such as disk override this method.
        """

        del preserve_grad
        self.to(device="cpu", model=model, optimizer=optimizer, grad=grad)

    def offload_after_read(self, *, model: bool = True) -> None:
        """Offload state after a read-only operation.

        Backends with reusable disk snapshots override this method to reclaim
        resident parameter storage without rewriting an unchanged generation.
        Other targets preserve the regular offload behavior.
        """

        self.offload(model=model, optimizer=False, grad=False, preserve_grad=True)

    def onload(self, *, model: bool = True, optimizer: bool = True, grad: bool = True) -> None:
        """Restore selected offloaded state to the active accelerator."""

        self.to(device=get_device_name(), model=model, optimizer=optimizer, grad=grad)

    @property
    def has_disk_offload_store(self) -> bool:
        """Whether this engine owns a store that can report disk I/O statistics."""

        return getattr(self, "_disk_store", None) is not None

    def pop_disk_offload_stats(self):
        """Return and clear rank-local disk I/O statistics."""

        store = getattr(self, "_disk_store", None)
        return store.pop_io_stats() if store is not None else {}

    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: Optional[str] = None,
        global_step: int = 0,
        max_ckpt_to_keep: Optional[int] = None,
        **kwargs,
    ) -> None:
        """
        Save model, optimizer, and scheduler states to a checkpoint.

        Args:
            local_path: Local filesystem path to save checkpoint.
            hdfs_path: Optional HDFS path to copy checkpoint.
            global_step: Integer training step number for naming.
            max_ckpt_to_keep: Maximum number of recent checkpoints to retain.
            **kwargs: Arbitrary keyword arguments.
        """
        raise NotImplementedError

    def load_checkpoint(
        self, local_path: str, hdfs_path: Optional[str] = None, del_local_after_load: bool = True, **kwargs
    ) -> None:
        """
        Load model, optimizer, and scheduler states from a checkpoint.

        Args:
            local_path: Local filesystem path of the checkpoint.
            hdfs_path: Optional HDFS path where checkpoint is stored.
            del_local_after_load: Whether to delete local copy after loading.
            **kwargs: Arbitrary keyword arguments.
        """
        raise NotImplementedError

    def is_mp_src_rank_with_outputs(self):
        """
        Whether the current rank is the first rank in model parallel group that contains model outputs
        """
        raise NotImplementedError

    def disable_adapter(self) -> ContextManager:
        """
        Disable all adapters temporarily under the context in the model for LoRA
        """
        return nullcontext()


class BaseEngineCtx:
    def __init__(self, engine: BaseEngine, mode, **kwargs):
        """Base Engine context that handles load and offload

        Args:
            engine:
            **kwargs:
        """
        self.engine = engine
        self.mode = mode
        assert self.mode in ("train", "eval")
        self.disable_auto_offload = kwargs.pop("disable_auto_offload", False)
        self.zero_grad_on_exit = kwargs.pop("zero_grad_on_exit", True)

    def _context_switch(self, device, *, preserve_grad=None):
        if self.disable_auto_offload:
            return
        if not (
            self.engine.is_param_offload_enabled
            or self.engine.is_grad_offload_enabled
            or self.engine.is_optimizer_offload_enabled
        ):
            return
        is_onload = device != "cpu"
        if self.mode == "eval":
            if is_onload:
                self.engine.onload(model=self.engine.is_param_offload_enabled, optimizer=False, grad=False)
            else:
                self.engine.offload_after_read(model=self.engine.is_param_offload_enabled)
        elif self.mode == "train":
            kwargs = {
                "model": self.engine.is_param_offload_enabled,
                "optimizer": self.engine.is_optimizer_offload_enabled,
                "grad": self.engine.is_grad_offload_enabled,
            }
            if is_onload:
                self.engine.onload(**kwargs)
            else:
                if preserve_grad is None:
                    preserve_grad = not self.zero_grad_on_exit
                self.engine.offload(**kwargs, preserve_grad=preserve_grad)

    def __enter__(self):
        self.engine.mode = self.mode
        self._context_switch(get_device_name())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context_switch("cpu", preserve_grad=not self.zero_grad_on_exit and exc_type is None)
        self.engine.mode = None


class EngineRegistry:
    """
    A registry for managing and instantiating different types of training engines.

    This class uses a dictionary to store engine classes, mapping a string key to each class.
    It provides a decorator `register` to add new engines to the registry and a `new` method
    to create an instance of a registered engine.
    """

    _engines = {}

    @classmethod
    def register(
        cls,
        model_type: str,
        backend: list[str] | str,
        device: list[str] | str = "cuda",
        vendor: list[str] | str | None = None,
    ):
        """
        A class method decorator that registers an engine class with a given key.

        This allows for dynamic instantiation of engine classes by their registered key.

        Args:
            model_type (str): The type of the model
            backend (list[str] | str): The backend to use for the model type
            device (list[str] | str): The device type (e.g., "cuda", "npu", "cpu") this engine supports,
                default is "cuda"
            vendor (list[str] | str | None): The hardware vendor (e.g., "nvidia", "metax") this engine
                supports. If None, the engine is registered as the default for the device type.

        Returns:
            A decorator function that takes an engine class and registers it.
        """

        def decorator(engine_class):
            assert issubclass(engine_class, BaseEngine)
            if model_type not in cls._engines:
                cls._engines[model_type] = {}

            backends = backend if isinstance(backend, list) else [backend]
            devices = device if isinstance(device, list) else [device]
            vendors = vendor if isinstance(vendor, list) else ([vendor] if vendor else [None])
            for current_backend in backends:
                for current_device in devices:
                    if current_backend not in cls._engines[model_type]:
                        cls._engines[model_type][current_backend] = {}
                    for current_vendor in vendors:
                        key = (current_device, current_vendor) if current_vendor else current_device
                        assert key not in cls._engines[model_type][current_backend], (
                            f"The key(device-vendor: {key}) has been already registed!"
                        )
                        cls._engines[model_type][current_backend][key] = engine_class

            return engine_class

        return decorator

    @classmethod
    def get_engine_cls(cls, model_type: str, backend: str):
        assert model_type in cls._engines, f"Unknown model_type: {model_type}"
        assert backend in cls._engines[model_type], f"Unknown backend: {backend}"
        device = get_device_name()
        vendor = get_vendor()
        # Allow environment variables to override detected device and vendor for engine selection, if set
        if os.getenv("VERL_ENGINE_DEVICE"):
            device = os.getenv("VERL_ENGINE_DEVICE")
        if os.getenv("VERL_ENGINE_VENDOR"):
            vendor = os.getenv("VERL_ENGINE_VENDOR")
        registry = cls._engines[model_type][backend]

        # Try vendor-specific lookup: (device, vendor)
        vendor_key = (device, vendor)
        if vendor_key in registry:
            return registry[vendor_key]

        # Fallback to device-only key (registered without vendor)
        if device in registry:
            return registry[device]

        # For cuda-compatible vendors without a specific registration, try nvidia
        if device == "cuda" and vendor != "nvidia":
            nvidia_key = (device, "nvidia")
            if nvidia_key in registry:
                return registry[nvidia_key]

        raise ValueError(
            f"No engine registered for device={device!r}, vendor={vendor!r}, "
            f"model_type={model_type!r}, backend={backend!r}"
        )

    @classmethod
    def new(cls, model_type, backend, *args, **kwargs):
        """
        Function to create a new training engine instance based on the provided config.
        Args:
            key: A configuration object containing the engine key and other settings.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            engine: An instance of the training engine corresponding to the config.
        Raises:
            NotImplementedError: If the engine key in the config does not match any known engines.
        """
        engine_cls = cls.get_engine_cls(model_type, backend)
        return engine_cls(*args, **kwargs)
