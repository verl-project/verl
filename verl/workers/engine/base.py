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

from abc import abstractmethod
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Generator, Optional

import torch
from tensordict import TensorDict

from verl.utils.device import get_device_name
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
        if self.is_mp_src_rank_with_outputs():
            assert "grad_norm" not in outputs["metrics"]
            outputs["metrics"]["grad_norm"] = grad_norm
        return outputs

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
        if not model:
            assert not optimizer and not grad, "Model must be moved to device along with optimizer and grad"

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

    def _context_switch(self, device):
        if self.disable_auto_offload:
            return
        should_move_model = self.engine.is_param_offload_enabled if device == "cpu" else True
        should_move_optimizer = self.engine.is_optimizer_offload_enabled if device == "cpu" else True
        if self.mode == "eval":
            self.engine.to(device=device, model=should_move_model, optimizer=False, grad=False)
        elif self.mode == "train":
            self.engine.to(
                device=device,
                model=should_move_model,
                optimizer=should_move_optimizer,
                grad=should_move_model,
            )

    def __enter__(self):
        self._context_switch(get_device_name())
        self.engine.mode = self.mode

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._context_switch("cpu")
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
    def register(cls, model_type: str, backend: list[str] | str, device: list[str] | str = "cuda"):
        """
        A class method decorator that registers an engine class with a given key.

        This allows for dynamic instantiation of engine classes by their registered key.

        Args:
            model_type (str): The type of the model
            backend (list[str] | str): The backend to use for the model type
            device (list[str] | str): The device type (e.g., "cuda", "npu", "cpu") this engine supports,
                default is "cuda"

        Returns:
            A decorator function that takes an engine class and registers it.
        """

        def decorator(engine_class):
            assert issubclass(engine_class, BaseEngine)
            if model_type not in cls._engines:
                cls._engines[model_type] = {}

            backends = backend if isinstance(backend, list) else [backend]
            devices = device if isinstance(device, list) else [device]
            for current_backend in backends:
                for current_device in devices:
                    if current_backend not in cls._engines[model_type]:
                        cls._engines[model_type][current_backend] = {}
                    if current_device not in cls._engines[model_type][current_backend]:
                        cls._engines[model_type][current_backend][current_device] = engine_class

            return engine_class

        return decorator

    @classmethod
    def get_engine_cls(cls, model_type: str, backend: str):
        assert model_type in cls._engines, f"Unknown model_type: {model_type}"
        assert backend in cls._engines[model_type], f"Unknown backend: {backend}"
        device = get_device_name()
        assert device in cls._engines[model_type][backend], (
            f"Unknown device: {device} for model_type: {model_type} and backend: {backend}"
        )
        return cls._engines[model_type][backend][device]

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
