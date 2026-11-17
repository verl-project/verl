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
from functools import partial, wraps
from types import FunctionType
from typing import Callable

from verl.protocol import DataProtoFuture
from verl.utils.py_functional import DynamicEnum
from verl.utils.transferqueue_utils import _find_meta, tqbridge

# here we add a magic number of avoid user-defined function already have this attribute
MAGIC_ATTR = "attrs_3141562937"


class Dispatch(DynamicEnum):
    """Enum class defining different dispatch modes for distributed computation.

    Each mode represents a specific strategy for distributing data across
    different ranks in a distributed system. The modes are used to control
    how data is partitioned and processed across different worker groups.
    """

    _registry = {}
    _next_value = 0


def init_predefined_dispatch_mode():
    Dispatch.register("RANK_ZERO")
    Dispatch.register("ONE_TO_ALL")
    Dispatch.register("ALL_TO_ALL")
    Dispatch.register("DP_COMPUTE")
    Dispatch.register("DP_COMPUTE_PROTO")
    Dispatch.register("DP_COMPUTE_PROTO_WITH_FUNC")
    Dispatch.register("DP_COMPUTE_METRIC")
    # This is a special dispatch mode for vllm ExternalRayDistributedExecutor
    Dispatch.register("DIRECT_ROLLOUT_METHOD")


class Execute(DynamicEnum):
    """Enum class defining different execution modes for distributed computation.

    These modes control how a function should be executed across different ranks
    in a distributed system.
    """

    _registry = {}
    _next_value = 0


def init_predefined_execute_mode():
    Execute.register("ALL")
    Execute.register("RANK_ZERO")


# Initialize the two Dynamic Enum Classes
init_predefined_dispatch_mode()
init_predefined_execute_mode()


def _split_args_kwargs_data_proto(chunks, *args, **kwargs):
    from verl.protocol import BatchData

    splitted_args = []
    for arg in args:
        assert BatchData(arg).is_chunkable(), f"arg of type {type(arg)} is not chunkable"
        chunked_arg = BatchData(arg).chunk(chunks=chunks)
        assert len(chunked_arg) == chunks
        splitted_args.append(chunked_arg)

    splitted_kwargs = {}
    for key, val in kwargs.items():
        assert BatchData(val).is_chunkable(), f"kwarg '{key}' of type {type(val)} is not chunkable"
        chunked_kwarg = BatchData(val).chunk(chunks=chunks)
        assert len(chunked_kwarg) == chunks
        splitted_kwargs[key] = chunked_kwarg

    return splitted_args, splitted_kwargs


def dispatch_one_to_all(worker_group, *args, **kwargs):
    args = tuple([arg] * worker_group.world_size for arg in args)
    kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
    return args, kwargs


def dummy_direct_rollout_call(worker_group, *args, **kwargs):
    raise NotImplementedError("Direct rollout call is forbidden.")


def dispatch_all_to_all(worker_group, *args, **kwargs):
    return args, kwargs


def collect_all_to_all(worker_group, output):
    return output


def _concat_data_proto_or_future(output: list):
    from verl.protocol import BatchData

    # make sure all the elements in output has the same type
    for o in output:
        assert type(o) is type(output[0])

    return BatchData(output).concat()


def dispatch_dp_compute(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    for arg in args:
        assert isinstance(arg, tuple | list) and len(arg) == worker_group.world_size
    for k, v in kwargs.items():
        assert isinstance(v, tuple | list) and len(v) == worker_group.world_size
    return args, kwargs


def collect_dp_compute(worker_group, output):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert len(output) == worker_group.world_size
    return output


def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(
        worker_group.world_size,
        *args,
        **kwargs,
    )
    return splitted_args, splitted_kwargs


def dispatch_dp_compute_data_proto_with_func(worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert isinstance(args[0], FunctionType)  # NOTE: The first one args is a function!

    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.world_size, *args[1:], **kwargs)
    splitted_args_with_func = [[args[0]] * worker_group.world_size] + splitted_args
    return splitted_args_with_func, splitted_kwargs


def collect_dp_compute_data_proto(worker_group, output):
    from verl.protocol import BatchData

    assert BatchData(output).is_concatable(), (
        f"expecting concatable output, but got element type {type(output[0]) if output else 'empty'}"
    )

    output = collect_dp_compute(worker_group, output)
    return _concat_data_proto_or_future(output)


def dispatch_nd_compute(
    dispatch_infos: list,
    dp_size,
    worker_group,
    *args,
    dispatch_mp_src_only: bool = False,
    **kwargs,
):
    import os

    from verl.single_controller.base.worker_group import WorkerGroup
    from verl.utils.ray_utils import parallel_put

    assert isinstance(worker_group, WorkerGroup)

    max_workers = max(1, min(dp_size, os.cpu_count() or 1))

    args = [parallel_put(arg, max_workers=max_workers) for arg in args]
    kwargs = {k: parallel_put(v, max_workers=max_workers) for k, v in kwargs.items()}

    all_args = []
    for arg in args:
        assert isinstance(arg, tuple | list) and len(arg) == dp_size
        transformed_args = []
        for i in range(worker_group.world_size):
            dispatch_info = dispatch_infos[i]
            transformed_args.append(
                arg[dispatch_info.dp_rank] if not dispatch_mp_src_only or dispatch_info.is_mp_src else None
            )
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in kwargs.items():
        assert isinstance(v, tuple | list) and len(v) == dp_size
        transformed_v = []
        for i in range(worker_group.world_size):
            dispatch_info = dispatch_infos[i]
            transformed_v.append(
                v[dispatch_info.dp_rank] if not dispatch_mp_src_only or dispatch_info.is_mp_src else None
            )
        all_kwargs[k] = transformed_v
    return all_args, all_kwargs


def collect_nd_compute(collect_infos: list, worker_group, output):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)
    assert len(output) == worker_group.world_size

    output_in_dp = []
    for global_rank in range(worker_group.world_size):
        if collect_infos[global_rank].is_collect:
            output_in_dp.append(output[global_rank])
    return output_in_dp


def dispatch_nd_compute_dataproto(
    dispatch_infos: list,
    dp_size,
    worker_group,
    *args,
    dispatch_mp_src_only: bool = False,
    **kwargs,
):
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(dp_size, *args, **kwargs)
    return dispatch_nd_compute(
        dispatch_infos,
        dp_size,
        worker_group,
        *splitted_args,
        dispatch_mp_src_only=dispatch_mp_src_only,
        **splitted_kwargs,
    )


def collect_nd_compute_dataproto(collect_infos: list, worker_group, output):
    output = collect_nd_compute(collect_infos, worker_group, output)

    from verl.protocol import BatchData

    assert BatchData(output).is_concatable(), (
        f"expecting concatable output, but got element type {type(output[0]) if output else 'empty'}"
    )
    return _concat_data_proto_or_future(output)


def _check_one_mp_src_per_dp_rank(mesh_name, dispatch_infos, dp_size):
    for dp_rank in range(dp_size):
        num_src_ranks = sum(info.dp_rank == dp_rank and info.is_mp_src for info in dispatch_infos)
        assert num_src_ranks == 1, (
            f"mesh {mesh_name} dp_rank {dp_rank} must have exactly one MP source rank, but got {num_src_ranks}"
        )


def dispatch_lazy_compute_data_proto(mesh_name, worker_group, *args, dispatch_mp_src_only: bool = False, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)

    # query dispatch info of the worker group
    if mesh_name not in worker_group._dispatch_info:
        worker_group._dispatch_info[mesh_name] = worker_group._query_dispatch_info(mesh_name)
        assert len(worker_group._dispatch_info[mesh_name]) == worker_group.world_size

    dispatch_infos = worker_group._dispatch_info[mesh_name]
    dp_size = max(info.dp_rank for info in dispatch_infos) + 1
    if dispatch_mp_src_only:
        _check_one_mp_src_per_dp_rank(mesh_name, dispatch_infos, dp_size)
    return dispatch_nd_compute_dataproto(
        dispatch_infos,
        dp_size,
        worker_group,
        *args,
        dispatch_mp_src_only=dispatch_mp_src_only,
        **kwargs,
    )


def collect_lazy_compute_data_proto(mesh_name, worker_group, *args, **kwargs):
    from verl.single_controller.base.worker_group import WorkerGroup

    assert isinstance(worker_group, WorkerGroup)

    # the dispatch info is stored in the worker group
    assert mesh_name in worker_group._dispatch_info

    if mesh_name not in worker_group._collect_info:
        worker_group._collect_info[mesh_name] = worker_group._query_collect_info(mesh_name)
        assert len(worker_group._collect_info[mesh_name]) == worker_group.world_size

    collect_infos = worker_group._collect_info[mesh_name]
    return collect_nd_compute_dataproto(collect_infos, worker_group, *args, **kwargs)


def make_nd_compute_dataproto_dispatch_fn(mesh_name, dispatch_mp_src_only: bool = False):
    return {
        "dispatch_fn": partial(
            dispatch_lazy_compute_data_proto,
            mesh_name,
            dispatch_mp_src_only=dispatch_mp_src_only,
        ),
        "collect_fn": partial(collect_lazy_compute_data_proto, mesh_name),
    }


# Global registry for dispatch mode.
DISPATCH_MODE_FN_REGISTRY = {
    Dispatch.ONE_TO_ALL: {
        "dispatch_fn": dispatch_one_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.ALL_TO_ALL: {
        "dispatch_fn": dispatch_all_to_all,
        "collect_fn": collect_all_to_all,
    },
    Dispatch.DP_COMPUTE: {"dispatch_fn": dispatch_dp_compute, "collect_fn": collect_dp_compute},
    Dispatch.DP_COMPUTE_PROTO: {
        "dispatch_fn": dispatch_dp_compute_data_proto,
        "collect_fn": collect_dp_compute_data_proto,
    },
    Dispatch.DP_COMPUTE_PROTO_WITH_FUNC: {
        "dispatch_fn": dispatch_dp_compute_data_proto_with_func,
        "collect_fn": collect_dp_compute_data_proto,
    },
    Dispatch.DP_COMPUTE_METRIC: {"dispatch_fn": dispatch_dp_compute_data_proto, "collect_fn": collect_dp_compute},
    Dispatch.DIRECT_ROLLOUT_METHOD: {
        "dispatch_fn": dummy_direct_rollout_call,
        "collect_fn": dummy_direct_rollout_call,
    },
}


def get_predefined_dispatch_fn(dispatch_mode):
    return DISPATCH_MODE_FN_REGISTRY[dispatch_mode]


def register_dispatch_mode(dispatch_mode_name, dispatch_fn, collect_fn):
    """
    Register a new dispatch mode.
    """
    dispatch_mode = Dispatch.register(dispatch_mode_name)
    _check_dispatch_mode(dispatch_mode)
    assert dispatch_mode not in DISPATCH_MODE_FN_REGISTRY, f"dispatch_mode_name {dispatch_mode_name} already exists"
    DISPATCH_MODE_FN_REGISTRY[dispatch_mode] = {"dispatch_fn": dispatch_fn, "collect_fn": collect_fn}


def update_dispatch_mode(dispatch_mode, dispatch_fn, collect_fn):
    """
    Update the dispatch mode.
    """
    _check_dispatch_mode(dispatch_mode)
    assert dispatch_mode in DISPATCH_MODE_FN_REGISTRY, f"dispatch_mode {dispatch_mode} not found"
    DISPATCH_MODE_FN_REGISTRY[dispatch_mode] = {"dispatch_fn": dispatch_fn, "collect_fn": collect_fn}


def get_predefined_execute_fn(execute_mode):
    """
    Note that here we only asks execute_all and execute_rank_zero to be implemented
    Leave the choice of how these two functions handle argument 'blocking' to users
    """
    predefined_execute_mode_fn = {
        Execute.ALL: {"execute_fn_name": "execute_all"},
        Execute.RANK_ZERO: {"execute_fn_name": "execute_rank_zero"},
    }
    return predefined_execute_mode_fn[execute_mode]


def _check_dispatch_mode(dispatch_mode):
    assert isinstance(dispatch_mode, Dispatch | dict), (
        f"dispatch_mode must be a Dispatch or a Dict. Got {dispatch_mode}"
    )
    if isinstance(dispatch_mode, dict):
        necessary_keys = ["dispatch_fn", "collect_fn"]
        for key in necessary_keys:
            assert key in dispatch_mode, f"key {key} should be in dispatch_mode if it is a dictionary"


def _check_execute_mode(execute_mode):
    assert isinstance(execute_mode, Execute), f"execute_mode must be a Execute. Got {execute_mode}"


def _is_async_worker_hook(hook) -> bool:
    if hook is None:
        return False
    return inspect.iscoroutinefunction(hook) or inspect.iscoroutinefunction(
        inspect.getattr_static(hook, "__call__", None)
    )


def _wrap_worker_hooks(func, worker_preprocess_fn, worker_postprocess_fn):
    if worker_preprocess_fn is None and worker_postprocess_fn is None:
        return func

    if (
        inspect.iscoroutinefunction(func)
        or _is_async_worker_hook(worker_preprocess_fn)
        or _is_async_worker_hook(worker_postprocess_fn)
    ):

        @wraps(func)
        async def async_inner(self, *args, **kwargs):
            if worker_preprocess_fn is not None:
                preprocessed = worker_preprocess_fn(self, *args, **kwargs)
                if inspect.isawaitable(preprocessed):
                    preprocessed = await preprocessed
                args, kwargs = preprocessed
            output = func(self, *args, **kwargs)
            if inspect.isawaitable(output):
                output = await output
            if worker_postprocess_fn is not None:
                output = worker_postprocess_fn(self, output)
                if inspect.isawaitable(output):
                    output = await output
            return output

        return async_inner

    @wraps(func)
    def inner(self, *args, **kwargs):
        if worker_preprocess_fn is not None:
            args, kwargs = worker_preprocess_fn(self, *args, **kwargs)
        output = func(self, *args, **kwargs)
        if worker_postprocess_fn is not None:
            output = worker_postprocess_fn(self, output)
        return output

    return inner


def _assert_no_worker_hooks_when_transfer_queue_enabled(worker_preprocess_fn, worker_postprocess_fn, *args, **kwargs):
    """Reject worker hooks when this call carries TransferQueue metadata.

    ``tqbridge`` reads metadata before worker hooks run. Hooks that redistribute data
    across ranks, such as a model-parallel broadcast, drop that metadata on ranks that
    were not the dispatch target.
    """
    if _find_meta(*args, **kwargs) is None:
        return
    assert worker_preprocess_fn is None and worker_postprocess_fn is None, (
        "worker_preprocess_fn and worker_postprocess_fn must be None when TransferQueue is enabled"
    )


def _materialize_futures(*args, **kwargs):
    new_args = []
    for arg in args:
        if isinstance(arg, DataProtoFuture):
            arg = arg.get()
        # add more type to materialize
        new_args.append(arg)
    for k, v in kwargs.items():
        if isinstance(v, DataProtoFuture):
            kwargs[k] = v.get()

    new_args = tuple(new_args)
    return new_args, kwargs


def register(
    dispatch_mode=Dispatch.ALL_TO_ALL,
    execute_mode=Execute.ALL,
    blocking=True,
    materialize_futures=True,
    worker_preprocess_fn: Callable | None = None,
    worker_postprocess_fn: Callable | None = None,
):
    """Register a function with distributed execution configuration.

    This decorator registers a function with specific dispatch and execution modes
    for distributed computation. It handles both synchronous and asynchronous
    functions, and optionally materializes futures before execution.

    Args:
        dispatch_mode:
            Dispatch mode for computation distribution. Default: Dispatch.ALL_TO_ALL.
        execute_mode:
            Execute mode for computation distribution. Default: Execute.ALL.
        blocking:
            Whether the execution should be blocking. Defaults to True.
        materialize_futures:
            Whether to materialize the data before dispatching. Defaults to True.
        worker_preprocess_fn:
            Optional function executed inside each worker before the registered method.
            It receives ``(self, *args, **kwargs)`` and returns ``(args, kwargs)``.
            Defaults to None.
        worker_postprocess_fn:
            Optional function executed inside each worker after the registered method.
            It receives ``(self, output)`` and returns the output to send back.
            Defaults to None. Must be None when the call carries TransferQueue metadata.


    Returns:
        A decorator that wraps the original function with distributed execution
        configuration.
    """

    _check_dispatch_mode(dispatch_mode=dispatch_mode)
    _check_execute_mode(execute_mode=execute_mode)
    assert worker_preprocess_fn is None or callable(worker_preprocess_fn), (
        f"worker_preprocess_fn must be callable or None. Got {worker_preprocess_fn}"
    )
    assert worker_postprocess_fn is None or callable(worker_postprocess_fn), (
        f"worker_postprocess_fn must be callable or None. Got {worker_postprocess_fn}"
    )

    def decorator(func):
        func = _wrap_worker_hooks(func, worker_preprocess_fn, worker_postprocess_fn)
        func = tqbridge(dispatch_mode=dispatch_mode)(func)

        def _prepare_call(args, kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            _assert_no_worker_hooks_when_transfer_queue_enabled(
                worker_preprocess_fn, worker_postprocess_fn, *args, **kwargs
            )
            return args, kwargs

        @wraps(func)
        def inner(*args, **kwargs):
            args, kwargs = _prepare_call(args, kwargs)
            return func(*args, **kwargs)

        @wraps(func)
        async def async_inner(*args, **kwargs):
            args, kwargs = _prepare_call(args, kwargs)
            return await func(*args, **kwargs)

        wrapper = async_inner if inspect.iscoroutinefunction(func) else inner
        attrs = {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode, "blocking": blocking}
        setattr(wrapper, MAGIC_ATTR, attrs)
        return wrapper

    return decorator
