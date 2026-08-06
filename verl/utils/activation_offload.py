# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Functionality for CPU offloading of tensors saved for backward pass."""

from __future__ import annotations

import functools
import logging
import os
from typing import Any, Optional

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl.utils.device import get_torch_device
from verl.utils.fsdp_utils import FSDPModule as FSDP2

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _get_unique_tensor_key(tensor):
    key = (tensor.untyped_storage().data_ptr() + tensor.storage_offset(), tensor.dtype)
    return key


class FSDPParameterFilter:
    def __init__(self):
        self.model_parameters_storage = set()

    def __call__(self, tensor):
        return tensor.untyped_storage().data_ptr() not in self.model_parameters_storage

    def update_model_parameters(self, model):
        new_storage = set()
        for p in model.parameters():
            new_storage.add(p.data.untyped_storage().data_ptr())
        self.model_parameters_storage = new_storage


class CpuOffloadHookWithOffloadHandler:
    """Context-manager that offloads/recovers tensors through an offload hander.

    The hook just offloads/recovers the tensor object to the handler through `tensor_push`
    and `tensor_pop` interface. How the offload-handler manages the offloading, recovering
    or prefetching timing is transparent to this hook.
    """

    def __init__(
        self,
        offload_handler: OffloadHandler,
        handler_extra_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        if handler_extra_kwargs is None:
            handler_extra_kwargs = {}
        self.offload_handler: OffloadHandler = offload_handler
        self.handler_extra_kwargs: dict[str, Any] = handler_extra_kwargs
        self.inside_context = False

    def __enter__(self):
        self.inside_context = True
        torch._C._autograd._push_saved_tensors_default_hooks(self.on_save_for_backward, self.on_get_saved_tensor)

    def __exit__(self, *args: Any):
        self.inside_context = False
        torch._C._autograd._pop_saved_tensors_default_hooks()

    def on_save_for_backward(self, tensor: torch.Tensor) -> Any:
        retrieve_identifier = self.offload_handler.tensor_push(tensor, **self.handler_extra_kwargs)
        return retrieve_identifier

    def on_get_saved_tensor(self, saved_state: Any) -> torch.Tensor:
        tensor = self.offload_handler.tensor_pop(saved_state, **self.handler_extra_kwargs)
        return tensor


class OffloadHandler:
    """A base class for CPU offload-handler."""

    def __init__(self) -> None:
        pass

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        """Tensor push."""
        raise NotImplementedError(
            "`tensor_push is not implented in OffloadHandler class. Inherit this class and implement your "
            "custom tensor_push."
        )

    def tensor_pop(self, tensor_tag: Any, **kwargs):
        """Tensor pop."""
        raise NotImplementedError(
            "`tensor_pop is not implented in OffloadHandler class. Inherit this class and implement your "
            "custom tensor_pop."
        )


class GroupCommitFunction(torch.autograd.Function):
    """this is a dummy op with output identical to input.
    However, it is necessary for marking a timepoint for offload handler to
    accomplish all synchronizations. Implementing it as a function is necessary
    because we need to actions in both forward and backward.
    """

    @staticmethod
    def forward(ctx, tensor, cpu_offload_handler):
        # pylint: disable=missing-function-docstring
        cpu_offload_handler.on_group_commit_forward()
        ctx.cpu_offload_handler = cpu_offload_handler
        # return the identical tensor
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=missing-function-docstring
        cpu_offload_handler = ctx.cpu_offload_handler
        cpu_offload_handler.on_group_commit_backward()
        return grad_output, None


group_prefetch_offload_commit = GroupCommitFunction.apply


class SynchronizedGroupOffloadHandler(OffloadHandler):
    """Offload Handler that offloads/reloads in a synchronized way.
    The device-to-host and host-to-device copying happen in the same stream
    as the computation kernels, thus the copying will block computation.
    """

    def __init__(self, num_offload_group, tensor_need_offloading_checker=(lambda _: True)) -> None:
        super().__init__()

        self.num_offload_group = num_offload_group
        self.tensor_need_offloading_checker = tensor_need_offloading_checker

        self.groupid_reset()

    def groupid_reset(self):
        """Groupid reset."""
        # Data structures to label saved tensors and book-keep their cpu copies.
        # Currently, on push, create a new cpu tensor and copies; on pop, copies
        # the tensor back to gpu and deletes the cpu tensor.
        # These will increment whenever `group_commit()` is invoked
        self.current_group, self.tensor_count_current_group = (0, 0)
        self.torch_tensor_count = 0
        self.tensor_tag_to_state = {}

    def on_group_commit_forward(self):
        """On group commit forward."""
        # finishing up with updating current group and tensor count
        self.current_group += 1  # increment
        self.tensor_count_current_group = 0  # reset

    def on_group_commit_backward(self):
        """On group commit backward."""
        self.current_group -= 1
        assert self.current_group >= 0

    @staticmethod
    def offload(src_tensor, pin_memory=True):
        """Offload."""

        cpu_backup = torch.empty(
            src_tensor.size(),
            dtype=src_tensor.dtype,
            layout=src_tensor.layout,
            device="cpu",
            pin_memory=pin_memory,
        )
        cpu_backup.copy_(src_tensor, non_blocking=True)
        state = (src_tensor.device, cpu_backup)
        return state

    @staticmethod
    def reload(state, non_blocking=None):
        """Reload."""
        dev, cpu_backup = state
        if non_blocking is None:
            non_blocking = cpu_backup.is_pinned()
        return cpu_backup.to(dev, non_blocking=non_blocking)

    def tensor_push(self, tensor: torch.Tensor, **kwargs):
        """Tensor push."""
        # obtain a unique tensor tag
        tensor_tag = (self.current_group, self.tensor_count_current_group)
        self.tensor_count_current_group += 1
        assert tensor_tag not in self.tensor_tag_to_state
        if self.current_group < self.num_offload_group and self.tensor_need_offloading_checker(tensor):
            state = SynchronizedGroupOffloadHandler.offload(tensor)
            self.tensor_tag_to_state[tensor_tag] = state
        else:
            # will be offloaded together after group commit
            self.tensor_tag_to_state[tensor_tag] = tensor

        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        """Tensor pop."""
        assert tensor_tag in self.tensor_tag_to_state
        state = self.tensor_tag_to_state.pop(tensor_tag)
        if isinstance(state, tuple):
            tensor = SynchronizedGroupOffloadHandler.reload(state)
        else:
            tensor = state
        return tensor


class AsyncDoubleBufferGroupOffloadHandler(SynchronizedGroupOffloadHandler):
    """Compared to synchronize, this uses more memory because of the buffer but
    achieves better performance due to the overlapping. D2h and h2d copying are
    completely hidden behind computation if computation time of a layer is longer
    than host-device communication time. Bulk offloading with delay and bulk reloading
    with prefetch are implemented."""

    def __init__(
        self,
        num_offload_group,  # must be <= actual number of groups (number of commits)
        num_model_group,
        tensor_need_offloading_checker=(lambda t: True),
    ) -> None:
        super().__init__(
            num_offload_group=num_offload_group,
            tensor_need_offloading_checker=tensor_need_offloading_checker,
        )
        # Number of layers in the model
        self.num_layers = num_model_group
        # Data Structure to maintain reference to activation tensors
        self.tensor_tag_to_buf = {}
        # Tracking the number of layers offloaded
        self.offloaded_group_count = 0
        # Core data structure that decides the window for offloading
        self.layer_window_map = {}
        self.group_offload_mapping = {}

        # Logic to make offloading load balance across computation
        # for optimal CPU/GPU interconnect usage
        constant = 0
        for i in range(self.num_offload_group):
            self.layer_window_map[i] = ((self.num_layers // self.num_offload_group) * (i + 1)) - 1
            if i < (self.num_layers % self.num_offload_group):
                self.layer_window_map[i] += i + 1
                constant = i + 1
            else:
                self.layer_window_map[i] += constant

        # allocate streams and events for synchronization
        self.d2h_stream = get_torch_device().Stream()
        self.h2d_stream = get_torch_device().Stream()

    def tensor_push(self, tensor: torch.Tensor, **kwargs) -> Any:
        torch_stray_tensor = isinstance(
            tensor,
            torch._subclasses.fake_tensor.FakeTensor | torch._subclasses.functional_tensor.FunctionalTensor,
        )
        need_offload = not torch_stray_tensor
        need_offload = need_offload and self.tensor_need_offloading_checker(tensor)

        if need_offload:
            # obtain a unique tensor tag
            tensor_tag = (self.current_group, self.tensor_count_current_group)
            self.tensor_count_current_group += 1

            assert tensor_tag not in self.tensor_tag_to_state
            self.tensor_tag_to_state[tensor_tag] = tensor

            if self.current_group < self.num_offload_group:
                self.tensor_tag_to_buf[tensor_tag] = tensor
        else:
            tensor_tag = tensor
        return tensor_tag

    def tensor_pop(self, tensor_tag, **kwargs):
        """Tensor pop."""
        if isinstance(tensor_tag, torch.Tensor):
            return tensor_tag
        assert tensor_tag in self.tensor_tag_to_state
        tensor = self.tensor_tag_to_state.pop(tensor_tag)
        self.tensor_tag_to_buf.pop(tensor_tag, None)

        # the tensor should have been copied back in on_group_commit_backward()
        # which invokes bulk_reload_group.
        assert not isinstance(tensor, tuple)
        return tensor

    def bulk_offload_group(self, group_to_offload):
        """Bulk offload group."""
        offload_mapping = {}
        offload_size = 0
        with get_torch_device().stream(self.d2h_stream):
            for tensor_tag, state in self.tensor_tag_to_state.items():
                group_id, _ = tensor_tag
                if group_id == group_to_offload:
                    assert not isinstance(state, tuple)
                    key = _get_unique_tensor_key(state)
                    if key not in offload_mapping:
                        offload_mapping[key] = state
                    # if offload, return the reference to cpu copy
                    self.tensor_tag_to_state[tensor_tag] = (key, state.shape)
            for key, tensor in offload_mapping.items():
                state = SynchronizedGroupOffloadHandler.offload(tensor)
                offload_size += tensor.numel() * tensor.element_size()
                offload_mapping[key] = state

            self.group_offload_mapping[group_to_offload] = offload_mapping

    def synchronize_on_group_commit_forward(self, current_group):
        """Synchronize on group commit forward."""

        # For the first group, kickstart the offload after we have
        # the first compute completion
        if current_group == 0:
            self.d2h_stream.wait_stream(get_torch_device().current_stream())
            self.bulk_offload_group(current_group)

        # Window map data structure helps us synchronize based on number
        # of layers offloaded
        if self.layer_window_map[self.offloaded_group_count] == current_group:
            # Stream synchronization both ways
            self.d2h_stream.wait_stream(get_torch_device().current_stream())
            get_torch_device().current_stream().wait_stream(self.d2h_stream)

            # Time to free the activation memory after usage
            for tensor_tag, _ in self.tensor_tag_to_buf.items():
                if tensor_tag[0] == self.offloaded_group_count:
                    self.tensor_tag_to_buf[tensor_tag] = None

            # Time to offload the next group
            if self.offloaded_group_count < (self.num_offload_group - 1):
                self.bulk_offload_group(self.offloaded_group_count + 1)

            # Increment the offload group count to keep track
            self.offloaded_group_count += 1

    def on_group_commit_forward(self):
        """This function will cause host device synchronization"""
        # handle synchronization events
        self.synchronize_on_group_commit_forward(self.current_group)

        super().on_group_commit_forward()

    @torch.no_grad
    def bulk_reload_group(self, group_to_reload):
        """Bulk reload group."""
        assert group_to_reload < self.num_offload_group

        with get_torch_device().stream(self.h2d_stream):
            # move back tensors
            offload_mapping = self.group_offload_mapping.pop(group_to_reload)
            assert offload_mapping is not None
            for key, state in offload_mapping.items():
                offload_mapping[key] = SynchronizedGroupOffloadHandler.reload(state)
            for tensor_label, state in self.tensor_tag_to_state.items():
                group_id, _ = tensor_label
                if group_id == group_to_reload and not isinstance(state, torch.Tensor):
                    assert isinstance(state, tuple), f"{group_id} {state}"
                    key, shape = state
                    recovered_tensor = offload_mapping[key].view(shape)
                    self.tensor_tag_to_state[tensor_label] = recovered_tensor

    def on_group_commit_backward(self):
        # first decrement the current group.
        # after last commit in forward, the group will +1; in backward it -1.
        # Finally it should be decremented to 0.
        self.current_group -= 1
        assert self.current_group >= 0

        # Layer window data structure helps us to reload at right times
        if self.layer_window_map[self.offloaded_group_count - 1] == self.current_group:
            # Stream synchronization both ways
            self.h2d_stream.wait_stream(get_torch_device().current_stream())
            get_torch_device().current_stream().wait_stream(self.h2d_stream)

            # Time to reload the next group
            self.bulk_reload_group(self.offloaded_group_count - 1)

            # Decrease the offloading group counter
            self.offloaded_group_count -= 1 if self.offloaded_group_count > 1 else 0

        # Last group computation needs to wait till all the reloads complete
        if self.current_group == 0:
            get_torch_device().current_stream().wait_stream(self.h2d_stream)
            self.offloaded_group_count = 0


def get_activation_offload_context(
    num_layers: int = 1, model_layers: int = 1, tensor_need_offloading_checker=(lambda t: True)
):
    cpu_offload_handler = AsyncDoubleBufferGroupOffloadHandler(
        num_offload_group=num_layers,
        num_model_group=model_layers,
        tensor_need_offloading_checker=tensor_need_offloading_checker,
    )

    def group_prefetch_offload_commit_async(tensor):
        return group_prefetch_offload_commit(tensor, cpu_offload_handler)

    return (
        CpuOffloadHookWithOffloadHandler(offload_handler=cpu_offload_handler),
        group_prefetch_offload_commit_async,
    )


class ActivationHandler:
    def __init__(self, offload_ctx, sync_func, tensor_filter, enable_ckpt):
        self._offload_ctx = offload_ctx
        self._sync_func = sync_func
        self._enable_ckpt = enable_ckpt
        self._tensor_filter = tensor_filter
        if enable_ckpt:
            self.checkpoint_fn = functools.partial(
                torch.utils.checkpoint.checkpoint,
                use_reentrant=True,
            )

    def pre_forward(self, module):
        if module.training:
            self._offload_ctx.__enter__()
            self._tensor_filter.update_model_parameters(module)

    def post_forward(self, module):
        if module.training:
            self._offload_ctx.__exit__(None, None, None)

    def _pack_kwargs(self, *args, **kwargs):
        kwarg_keys = []
        flat_args = list(args)
        for k, v in kwargs.items():
            kwarg_keys.append(k)
            flat_args.append(v)

        return tuple(flat_args), tuple(kwarg_keys)

    def _unpack_kwargs(self, flat_args, kwarg_keys):
        assert len(kwarg_keys) <= len(flat_args), f"too many keys {len(kwarg_keys)} vs. {len(flat_args)}"
        if len(kwarg_keys) == 0:
            return flat_args, {}
        args = flat_args[: -len(kwarg_keys)]
        kwargs = dict(zip(kwarg_keys, flat_args[-len(kwarg_keys) :], strict=True))
        return args, kwargs

    def _ckpt_forward(self, forward_method, *args, **kwargs):
        flat_args, kwarg_keys = self._pack_kwargs(*args, **kwargs)

        def my_function(*inputs):
            # unpack back into args and kwargs
            nonlocal forward_method, kwarg_keys
            unpacked_args, unpacked_kwargs = self._unpack_kwargs(inputs, kwarg_keys)
            # run original module
            return forward_method(*unpacked_args, **unpacked_kwargs)

        return self.checkpoint_fn(
            my_function,
            *flat_args,
        )

    def forward(self, module, forward_method, *args, **kwargs):
        if not module.training:
            return forward_method(*args, **kwargs)
        if not self._enable_ckpt:
            ret = forward_method(*args, **kwargs)
        else:
            ret = self._ckpt_forward(forward_method, *args, **kwargs)
        binded_tensor = ret
        if isinstance(ret, tuple):
            binded_tensor = ret[0]
        binded_tensor = self._sync_func(binded_tensor)
        final_ret = binded_tensor
        if isinstance(ret, tuple):
            final_ret = (final_ret,) + ret[1:]
        return final_ret

    def wrap_module_forward_method(self, module):
        orig_method = module.forward
        handler = self

        @functools.wraps(orig_method)
        def wrapped_method(model_self, *args, **kwargs):
            nonlocal handler
            handler.pre_forward(model_self)
            out = handler.forward(model_self, orig_method, *args, **kwargs)
            handler.post_forward(model_self)
            return out

        module.forward = wrapped_method.__get__(module, type(module))


def enable_activation_offloading(model, strategy, enable_ckpt=False):
    """
    Enable activation offloading for the model. It groups activations by TransformerLayer and offloads activation
    groups asynchronously. This means that the offloading of the i-th activation group and the computation of the i+1-th
    activation group happen at the same time, and there are at most two activation groups in GPU memory.

    Args:
        model: the model to enable activation offloading
        strategy: the training strategy of the model, such as "fsdp"
        enable_ckpt: whether activation checkpointing(also called gradient checkpointing) has been enabled for the model

    Note:
        For best efficiency, activation offloading is usually combined with activation checkpointing. However, this
        implementation of activation offloading is conflicted with the implementation of activation checkpointing in
        some training strategies. This function resolves this conflict, and therefore requires the "strategy" and
        "enable_ckpt" arguments.

    Returns:

    """

    assert strategy == "fsdp" or strategy == "fsdp2", "activation offloading only supports fsdp strategy"
    layers = []

    def get_layers(module):
        for name, child in module.named_children():
            if not isinstance(child, FSDP | FSDP2):
                get_layers(child)
            else:
                wrapped_module = child
                if isinstance(child, FSDP):
                    wrapped_module = child._fsdp_wrapped_module
                # In some cases, torch.nn.Embedding is wrapped with FSDP alone. However, the activation
                # size of torch.nn.Embedding is small, so it's not necessary to offload it.
                if not isinstance(wrapped_module, torch.nn.Embedding):
                    layers.append(child)

    get_layers(model)
    if len(layers) < 3:
        logger.warning(f"Find only {len(layers)} fsdp layers, not necessary to enable async activation offloading")
        return

    tensor_filter = FSDPParameterFilter()
    context, sync_func = get_activation_offload_context(len(layers) - 1, len(layers), tensor_filter)
    if enable_ckpt:
        # The implementation of activation checkpointing in transformers library is incompatible with
        # activation offloading,
        # so it will be disabled, but this implementation supports another version of activation checkpointing, so that
        # these two features can be enabled at the same time.
        for module in model.modules():
            if hasattr(module, "gradient_checkpointing_disable"):
                module.gradient_checkpointing_disable()

    handler = ActivationHandler(context, sync_func, tensor_filter, enable_ckpt)
    for layer in layers:
        module = layer
        if isinstance(layer, FSDP):
            module = module._fsdp_wrapped_module
        handler.wrap_module_forward_method(module)
