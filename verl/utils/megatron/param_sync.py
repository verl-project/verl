# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Parameter readiness for fused consumers that bypass a module's DDP pre-hook."""

import weakref

try:
    from megatron.core.utils import PARAM_READY_CALLBACK_ATTR
    from megatron.core.utils import ensure_params_ready as _mcore_ensure_params_ready
except ImportError:
    # Released MCore 0.18 and older stacks do not yet expose this contract.
    PARAM_READY_CALLBACK_ATTR = None
    _mcore_ensure_params_ready = None

_VERL_PARAM_READY_CALLBACK_ATTR = "_verl_ensure_param_ready_callback"


def _is_graph_capturing():
    from megatron.core.transformer import cuda_graphs

    return cuda_graphs.is_graph_capturing()


class _DDPParamReadyCallback:
    """Use the owning DDP's synchronization policy without retaining its buffers."""

    def __init__(self, ddp, bucket_group):
        self._ddp = weakref.ref(ddp)
        self._bucket_group = weakref.ref(bucket_group)

    def __call__(self):
        ddp = self._ddp()
        bucket_group = self._bucket_group()
        if ddp is None or bucket_group is None:
            return
        if bucket_group.param_gather_dispatched and bucket_group.param_gather_handle is None:
            return
        # Match MCore's pre-hook: collectives must not be captured per microbatch.
        if _is_graph_capturing():
            return

        ddp_owns_schedule = bool(ddp.remove_forward_pre_hook_handles)
        if not ddp_owns_schedule:
            # An external schedule owns dispatch. Drain an existing gather, but
            # do not start a new gather or prefetch the next bucket on its behalf.
            if bucket_group.param_gather_handle is not None:
                bucket_group.finish_param_sync(skip_next_bucket_dispatch=True)
            return

        finish = getattr(ddp, "_finish_param_sync_for_bucket_group", None)
        if finish is not None:
            finish(bucket_group)
        else:
            # This is the policy used by older DDP forward pre-hooks.
            bucket_group.finish_param_sync(
                skip_next_bucket_dispatch=(
                    ddp.ddp_config.align_param_gather or ddp.overlap_param_gather_with_optimizer_step
                )
            )


def register_ddp_param_ready_callbacks(model_chunks):
    """Attach compatibility callbacks after wrapping model chunks in DDP.

    Native MCore callbacks take precedence. Re-wrapping replaces our callbacks
    (or removes them when overlap is disabled), so parameters cannot retain an
    old bucket owner. Frozen parameters without a DDP buffer are left unmarked.
    """
    from megatron.core.distributed import DistributedDataParallel

    if not isinstance(model_chunks, list | tuple):
        model_chunks = [model_chunks]
    for ddp in model_chunks:
        if not isinstance(ddp, DistributedDataParallel):
            continue
        for param in ddp.module.parameters():
            if hasattr(param, _VERL_PARAM_READY_CALLBACK_ATTR):
                delattr(param, _VERL_PARAM_READY_CALLBACK_ATTR)
        if not ddp.ddp_config.overlap_param_gather:
            continue
        if not hasattr(ddp, "param_to_bucket_group"):
            raise RuntimeError("Fused parameter readiness requires DDP.param_to_bucket_group with AG overlap.")

        callbacks = {}
        for param, bucket_group in ddp.param_to_bucket_group.items():
            if PARAM_READY_CALLBACK_ATTR is not None and callable(getattr(param, PARAM_READY_CALLBACK_ATTR, None)):
                continue
            key = id(bucket_group)
            if key not in callbacks:
                callbacks[key] = _DDPParamReadyCallback(ddp, bucket_group)
            setattr(param, _VERL_PARAM_READY_CALLBACK_ATTR, callbacks[key])


def ensure_fused_weight_ready(weight):
    """Publish the actual weight before a fused kernel reads its parameter storage."""
    if _mcore_ensure_params_ready is not None:
        _mcore_ensure_params_ready([weight])
    callback = getattr(weight, _VERL_PARAM_READY_CALLBACK_ATTR, None)
    if callback is not None:
        callback()
