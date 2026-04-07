# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import asyncio
import copy
import functools
import inspect
import logging
import os
import threading
import time
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

import torch
from tensordict.tensorclass import NonTensorData, NonTensorStack

if TYPE_CHECKING:
    from verl.single_controller.base.decorator import Dispatch

from tensordict import TensorDict

try:
    import transfer_queue as tq
    from transfer_queue import (
        BatchMeta,
        KVBatchMeta,
    )

except ImportError:

    class BatchMeta:
        pass

    class KVBatchMeta:
        pass

    # Mock transfer_queue module when not installed
    class _MockTQ:
        """Mock transfer_queue module that raises RuntimeError on any access."""

        def __getattr__(self, name: str) -> Any:
            def _raise(*args, **kwargs):
                raise RuntimeError(
                    f"transfer_queue is not installed. Cannot use tq.{name}(). "
                    "Please install it by calling `pip install TransferQueue==0.1.6`"
                )

            return _raise

    tq = _MockTQ()


from verl.utils import tensordict_utils as tu

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

TQ_INITIALIZED = False


# TODO (TQ): verl will make all actor async, so this can be cleanup later.
def _run_async_in_temp_loop(async_func: Callable[..., Any], *args, **kwargs) -> Any:
    # Use a temporary event loop in a new thread because event
    # loop may already exist in server mode
    tmp_event_loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=tmp_event_loop.run_forever,
        name="batchmeta tensordict converter",
        daemon=True,
    )

    def run_coroutine(coroutine):
        if not thread.is_alive():
            thread.start()
        future = asyncio.run_coroutine_threadsafe(coroutine, tmp_event_loop)
        return future.result()

    async def stop_loop():
        tmp_event_loop.stop()

    try:
        return run_coroutine(async_func(*args, **kwargs))
    finally:
        if thread.is_alive():
            asyncio.run_coroutine_threadsafe(stop_loop(), tmp_event_loop)
            thread.join()


def _find_meta(*args, **kwargs):
    for arg in args:
        if isinstance(arg, BatchMeta):
            return arg
    for v in kwargs.values():
        if isinstance(v, BatchMeta):
            return v
    return None


async def _async_meta_to_realdata(meta: BatchMeta) -> TensorDict:
    meta_info = copy.deepcopy(meta.extra_info)
    if meta.size == 0:
        empty_td = TensorDict({}, batch_size=(0,))
        tu.assign_non_tensor(empty_td, **meta_info)
        return empty_td

    tq_client = tq.get_client()
    tensordict = await tq_client.async_get_data(meta)

    for key, val in meta_info.items():
        if isinstance(val, (NonTensorData | NonTensorStack)):
            tensordict[key] = val
        else:
            tu.assign_non_tensor_data(tensor_dict=tensordict, key=key, val=val)
    return tensordict


def _meta_to_realdata(meta: BatchMeta) -> TensorDict:
    return _run_async_in_temp_loop(_async_meta_to_realdata, meta)


async def _async_update_meta_with_output(output: TensorDict, meta: BatchMeta, func_name=None) -> BatchMeta:
    fields, meta_data = [], {}
    for k, v in output.items():
        if isinstance(v, torch.Tensor | NonTensorStack):
            fields.append(k)
        elif isinstance(v, NonTensorData):
            meta_data[k] = v.data
        else:
            raise ValueError(f"Unsupported type {type(v)} for key {k} in output TensorDict.")

    if fields:
        t1 = time.time()
        tq_client = tq.get_client()
        meta = await tq_client.async_put(data=output.select(*fields), metadata=meta)
        t2 = time.time()

        logger.info(f"Task {func_name} (pid={os.getpid()}) is writing to TransferQueue, cost time: {t2 - t1}s)")
        meta.extra_info = meta_data
    return meta


def _update_meta_with_output(output: TensorDict, meta: BatchMeta, func_name=None) -> BatchMeta:
    updated_meta = _run_async_in_temp_loop(_async_update_meta_with_output, output, meta, func_name)
    return updated_meta


def _compute_need_collect(dispatch_mode: "dict | Dispatch", args: list) -> bool:
    """Compute whether data collection is needed for the current worker.

    This function determines whether the current worker should collect data based on
    the dispatch mode configuration and worker parameters. It's used to optimize
    distributed data collection by ensuring only the appropriate rank collects data.

    Args:
        dispatch_mode: Controls data collection logic for the current worker. Can be None,
                      a Dispatch instance, or a dict with 'collect_fn' key. If None or Dispatch,
                      always returns True (current worker should collect). If dict, checks
                      collect_fn for lazy compute optimization.
        args: List of arguments passed to the function. Should contain a Worker instance
             as the first argument when using lazy compute mode.

    Returns:
        bool: True if data collection is needed, False otherwise.

    Note:
        Only checks worker attributes when dispatch_mode is a dict with 'collect_fn',
        the collect_fn is 'collect_lazy_compute_data_proto', and args[0] is a Worker.
        Otherwise, returns True. For the lazy compute case, checks the worker's
        data parallel rank for the mesh specified in collect_fn.args[0] to determine
        if this worker should collect data.
    """
    from verl.single_controller.base.decorator import Dispatch
    from verl.single_controller.base.worker import Worker

    if dispatch_mode is None or isinstance(dispatch_mode, Dispatch):
        return True

    assert "collect_fn" in dispatch_mode.keys(), "collect_fn should be in dispatch_mode."

    collect_fn = dispatch_mode["collect_fn"]

    # Check if collect_fn is a functools.partial and handle gracefully
    if isinstance(collect_fn, functools.partial):
        collect_fn_name = collect_fn.func.__name__
        if collect_fn_name != "collect_lazy_compute_data_proto" or len(args) < 1 or not isinstance(args[0], Worker):
            return True

        collect_mesh_name = collect_fn.args[0] if collect_fn.args else None
        if collect_mesh_name is None:
            return True

        return args[0].query_collect_info(collect_mesh_name)
    else:
        # If collect_fn is not a partial, we can't extract mesh_name information
        # Fall back to default behavior (collect data)
        return True


def _postprocess_common(output, put_data, need_collect):
    """Common post-processing logic for function outputs in TransferQueue bridge.

    This function handles the final return value based on whether data should be
    put into storage (put_data) and whether collection is needed (need_collect).
    It ensures proper return types based on the execution context.

    Args:
        output: The original output from the decorated function. Can be any type.
        put_data: bool, indicating whether the output should be put into TransferQueue.
                 If True, output will be put to TQ and return the corresponding BatchMeta;
                 if False, output will not be put into TQ.
        need_collect: bool, indicating whether this process needs to collect data.
                     If False, the output will be replaced by an empty BatchMeta or DataProto
                     to avoid redundant communication.

    Returns:
        - BatchMeta.empty(): When put_data=True but need_collect=False, indicating
          no data should be stored but BatchMeta structure is expected.
        - DataProto(): When put_data=False, need_collect=False, and output is DataProto,
          returning an empty DataProto.
        - output: In all other cases, returns the original output unchanged.

    Note:
        This function is used in the tqbridge decorator to normalize return values
        across different execution paths and avoid redundant data operations in
        distributed scenarios.
    """
    from verl.protocol import DataProto

    if put_data and not need_collect:
        return BatchMeta()
    elif not put_data and not need_collect and isinstance(output, DataProto):
        return DataProto()
    elif not put_data and not need_collect and isinstance(output, TensorDict):
        return TensorDict({}, batch_size=(0,))
    else:
        return output


async def async_kv_batch_meta2batch_meta(meta: KVBatchMeta) -> BatchMeta:
    global TQ_INITIALIZED
    if not TQ_INITIALIZED:
        tq.init()
        TQ_INITIALIZED = True
    tq_client = tq.get_client()
    batch_meta = await tq_client.async_kv_retrieve_meta(keys=meta.keys, partition_id=meta.partition_id, create=False)
    fields = meta.fields
    if fields is not None:
        if isinstance(fields, str):
            fields = [fields]
        batch_meta = batch_meta.select_fields(fields)

    batch_meta.extra_info = meta.extra_info
    return batch_meta


def kv_batch_meta2batch_meta(meta: KVBatchMeta):
    return _run_async_in_temp_loop(async_kv_batch_meta2batch_meta, meta)


async def async_batch_meta2kv_batch_meta(meta: BatchMeta) -> KVBatchMeta:
    global TQ_INITIALIZED
    if not TQ_INITIALIZED:
        tq.init()
        TQ_INITIALIZED = True
    tq_client = tq.get_client()
    partition_id = meta.partition_ids[0]
    assert all([partition_id == pid for pid in meta.partition_ids])
    keys = await tq_client.async_kv_retrieve_keys(global_indexes=meta.global_indexes, partition_id=partition_id)

    kv_batch_meta = KVBatchMeta(
        keys=keys,
        tags=[{}] * meta.size,
        partition_id=partition_id,
        fields=meta.field_names,
        extra_info=meta.extra_info,
    )
    return kv_batch_meta


def batch_meta2kv_batch_meta(meta: BatchMeta):
    return _run_async_in_temp_loop(async_batch_meta2kv_batch_meta, meta)


def tqbridge(dispatch_mode: "dict | Dispatch" = None):
    """Creates a decorator for bridging KVBatchMeta and TensorDict.

    This decorator automatically handles conversions between `KVBatchMeta`
    and `TensorDict` in function parameters, and decides whether to sync function
    output back to `KVBatchMeta` based on configuration(`put_data`). It supports
    both synchronous and asynchronous functions (async def). When TQ is not enabled, it
    simply calls the original function as-is.

    Args:
        dispatch_mode: Controls data collection behavior for the current worker. Passed to
                      _compute_need_collect to determine if current worker should collect data.
                      If None, _compute_need_collect will return True to fallback default logics.


    Returns:
        A decorator function used to decorate target functions (synchronous or asynchronous).
    """
    # TODO: move to the top
    from verl.single_controller.base.decorator import _check_dispatch_mode

    _check_dispatch_mode(dispatch_mode)

    def decorator(func):
        pid = os.getpid()

        @wraps(func)
        def inner(*args, **kwargs):
            batch_meta = _find_meta(*args, **kwargs)
            if batch_meta is None:
                return func(*args, **kwargs)
            else:
                global TQ_INITIALIZED
                if not TQ_INITIALIZED:
                    tq.init()
                    TQ_INITIALIZED = True
                t1 = time.time()
                args = [_meta_to_realdata(arg) if isinstance(arg, BatchMeta) else arg for arg in args]
                kwargs = {k: _meta_to_realdata(v) if isinstance(v, BatchMeta) else v for k, v in kwargs.items()}
                t2 = time.time()
                logger.info(
                    f"Task {func.__name__} (pid={pid}) is getting len_samples={batch_meta.size}, cost time: {t2 - t1}"
                )

                output = func(*args, **kwargs)

                put_data = False
                if isinstance(output, TensorDict):
                    if output.batch_size:
                        assert output.batch_size[0] == batch_meta.size, (
                            f"output batch size {output.batch_size} != meta size {batch_meta.size}"
                        )
                        put_data = True

                need_collect = _compute_need_collect(dispatch_mode, args)
                if put_data and need_collect:
                    updated_meta = _update_meta_with_output(output, batch_meta, func.__name__)
                    return updated_meta
                return _postprocess_common(output, put_data, need_collect)

        @wraps(func)
        async def async_inner(*args, **kwargs):
            batch_meta = _find_meta(*args, **kwargs)
            if batch_meta is None:
                return await func(*args, **kwargs)
            else:
                global TQ_INITIALIZED
                if not TQ_INITIALIZED:
                    tq.init()
                    TQ_INITIALIZED = True
                t1 = time.time()
                args = [await _async_meta_to_realdata(arg) if isinstance(arg, BatchMeta) else arg for arg in args]
                kwargs = {
                    k: await _async_meta_to_realdata(v) if isinstance(v, BatchMeta) else v for k, v in kwargs.items()
                }
                t2 = time.time()
                logger.info(
                    f"Task {func.__name__} (pid={pid}) is getting len_samples={batch_meta.size}, cost time: {t2 - t1}"
                )

                output = await func(*args, **kwargs)

                put_data = False
                if isinstance(output, TensorDict):
                    if output.batch_size:
                        assert output.batch_size[0] == batch_meta.size, (
                            f"output batch size {output.batch_size} != meta size {batch_meta.size}"
                        )
                        put_data = True

                need_collect = _compute_need_collect(dispatch_mode, args)
                if put_data and need_collect:
                    updated_meta = await _async_update_meta_with_output(output, batch_meta, func.__name__)
                    return updated_meta
                return _postprocess_common(output, put_data, need_collect)

        wrapper_inner = inner
        wrapper_async_inner = async_inner

        wrapper = wrapper_async_inner if inspect.iscoroutinefunction(func) else wrapper_inner
        return wrapper

    return decorator
