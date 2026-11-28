# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import threading
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from verl.utils.checkpoint.megatron_thread_checkpoint import ThreadAsyncCaller, _isolate_cpu_snapshots

_Request = namedtuple("_Request", "async_fn async_fn_args async_fn_kwargs finalize_fns preload_fn")


def test_cpu_shared_storage_is_snapshotted_before_training_mutates_it():
    source = torch.tensor([1.0, 2.0]).share_memory_()
    bucket = ("path", "key", (["serialized-object"], [("item", source)]))
    request_type = namedtuple("Request", "preload_fn async_fn_args")
    request = request_type(lambda: [bucket], (0, [bucket], None))

    staged = _isolate_cpu_snapshots(request).preload_fn()
    source.add_(10)

    staged_tensor = staged[0][2][1][0][1]
    assert torch.equal(staged_tensor, torch.tensor([1.0, 2.0]))
    assert staged_tensor.data_ptr() != source.data_ptr()
    assert staged[0][2][0] == ["serialized-object"]


def test_gpu_native_host_snapshot_is_not_copied_twice():
    source = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    staged_tensor = torch.tensor([1.0, 2.0])
    source_bucket = ("path", "key", ([], [("item", source)]))
    staged_bucket = ("path", "key", ([], [("item", staged_tensor)]))
    request_type = namedtuple("Request", "preload_fn async_fn_args")
    request = request_type(lambda: [staged_bucket], (0, [source_bucket], None))

    staged = _isolate_cpu_snapshots(request).preload_fn()

    assert staged[0][2][1][0][1] is staged_tensor


def _raise_local_error(error):
    if error is not None:
        raise error


def test_thread_caller_stages_before_return_and_keeps_training_priority():
    events = []
    actor_thread = threading.get_ident()
    release = threading.Event()
    started = threading.Event()
    source = torch.tensor([1.0, 2.0])
    priority = os.getpriority(os.PRIO_PROCESS, 0)

    def preload():
        events.append(("preload", threading.get_ident()))
        return source.clone()

    def write(_rank, snapshot, _queue):
        started.set()
        assert release.wait(5)
        events.append(("write", threading.get_ident(), snapshot.tolist()))

    request = _Request(write, (None, None, None), {}, [], preload)
    with ThreadPoolExecutor(max_workers=1) as executor:
        caller = ThreadAsyncCaller(
            executor,
            synchronize=lambda: events.append(("synchronize", threading.get_ident())),
            sync_all_done=lambda alive: not alive,
            sync_failure=_raise_local_error,
        )
        try:
            caller.schedule_async_call(request)
            assert started.wait(5)
            assert caller.is_current_async_call_done(blocking=False) is False
            source.add_(10)
        finally:
            release.set()
        assert caller.is_current_async_call_done(blocking=True) is True
        caller.close()

    assert events[:2] == [("preload", actor_thread), ("synchronize", actor_thread)]
    assert events[2][0] == "write"
    assert events[2][1] != actor_thread
    assert events[2][2] == [1.0, 2.0]
    assert os.getpriority(os.PRIO_PROCESS, 0) == priority


def test_future_error_reaches_all_rank_guard_before_it_is_raised():
    events = []
    error = OSError("unexpected writer failure")

    def write():
        raise error

    def synchronize_failure(local_error):
        events.append(("failure_guard", local_error))
        if local_error is not None:
            raise RuntimeError("all-rank checkpoint failure")

    with ThreadPoolExecutor(max_workers=1) as executor:
        caller = ThreadAsyncCaller(
            executor,
            synchronize=lambda: None,
            sync_all_done=lambda alive: events.append(("readiness", alive)) or not alive,
            sync_failure=synchronize_failure,
        )
        caller.schedule_async_call(_Request(write, (), {}, [], None))
        with pytest.raises(RuntimeError, match="all-rank checkpoint failure"):
            caller.is_current_async_call_done(blocking=True)

    assert events[0] == ("failure_guard", None)
    assert events[1][0] == "readiness"
    assert events[2] == ("failure_guard", error)


def test_successful_local_future_still_checks_for_other_rank_failure():
    write_failed_remotely = False

    def remote_failure(local_error):
        assert local_error is None
        if write_failed_remotely:
            raise RuntimeError("another rank failed")

    with ThreadPoolExecutor(max_workers=1) as executor:
        caller = ThreadAsyncCaller(
            executor,
            synchronize=lambda: None,
            sync_all_done=lambda alive: not alive,
            sync_failure=remote_failure,
        )
        caller.schedule_async_call(_Request(lambda: None, (), {}, [], None))
        write_failed_remotely = True
        with pytest.raises(RuntimeError, match="another rank failed"):
            caller.is_current_async_call_done(blocking=True)


@pytest.mark.parametrize("local_error", [None, OSError("local native writer failed")])
def test_default_failure_guard_raises_checkpoint_exception_on_every_rank(monkeypatch, local_error):
    from torch.distributed.checkpoint import CheckpointException

    from verl.utils.checkpoint.megatron_thread_checkpoint import _raise_if_writer_failed

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda flag, **_kwargs: flag.fill_(1))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(messages, local_message):
        assert local_message == (repr(local_error) if local_error else None)
        messages[:] = [repr(local_error) if local_error else None, "OSError('remote native writer failed')"]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    with pytest.raises(CheckpointException, match="remote native writer failed") as captured:
        _raise_if_writer_failed(local_error)
    assert 1 in captured.value.failures


@pytest.mark.parametrize("initially_enabled", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_native_thread_writer_restores_process_gc_state(initially_enabled, fail):
    native = pytest.importorskip("megatron.core.dist_checkpointing.strategies.filesystem_async")
    previous_gc = gc.isenabled()
    observed = []

    class ResultsQueue:
        def put(self, _result):
            observed.append(gc.isenabled())
            if fail:
                raise OSError("results channel failed")

    if initially_enabled:
        gc.enable()
    else:
        gc.disable()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            caller = ThreadAsyncCaller(
                executor,
                synchronize=lambda: None,
                sync_all_done=lambda alive: not alive,
                sync_failure=_raise_local_error,
            )
            caller.schedule_async_call(
                _Request(
                    native.FileSystemWriterAsync.write_preloaded_data_multithread,
                    ([], False, 0, [], ResultsQueue()),
                    {},
                    [],
                    None,
                )
            )
            if fail:
                with pytest.raises(OSError, match="results channel failed"):
                    caller.is_current_async_call_done(blocking=True)
            else:
                caller.is_current_async_call_done(blocking=True)
        assert observed == [False]
        assert gc.isenabled() is initially_enabled
    finally:
        gc.enable() if previous_gc else gc.disable()


def test_native_thread_queue_keeps_collectives_on_actor_thread_and_drains_on_close(monkeypatch):
    native = pytest.importorskip("megatron.core.dist_checkpointing.strategies.async_utils")
    from verl.utils.checkpoint.megatron_thread_checkpoint import create_thread_checkpoint_queue

    actor_thread = threading.get_ident()
    events = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda *_args, **_kwargs: events.append(("all_reduce", threading.get_ident()))
    )
    queue = create_thread_checkpoint_queue(
        synchronize=lambda: events.append(("synchronize", threading.get_ident())),
        sync_all_done=lambda alive: not alive,
        sync_failure=_raise_local_error,
    )
    request = native.AsyncRequest(
        lambda: events.append(("write", threading.get_ident())),
        (),
        [lambda: events.append(("finalize", threading.get_ident()))],
    )
    queue.schedule_async_request(request)
    queue.close()

    assert [name for name, _ident in events] == ["synchronize", "write", "finalize", "all_reduce"]
    assert all(ident == actor_thread for name, ident in events if name != "write")
    assert next(ident for name, ident in events if name == "write") != actor_thread
    assert queue.get_num_unfinalized_calls() == 0
    with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
        queue.schedule_async_request(request)


def test_overlapping_gc_disabled_scopes_keep_gc_disabled_until_last_exit():
    from verl.utils.checkpoint.megatron_thread_checkpoint import _gc_disabled_scope

    previous_gc = gc.isenabled()
    gc.enable()
    try:
        with _gc_disabled_scope():
            assert not gc.isenabled()
            with _gc_disabled_scope():
                assert not gc.isenabled()
            assert not gc.isenabled()
        assert gc.isenabled()
    finally:
        gc.enable() if previous_gc else gc.disable()


def test_writer_close_shuts_down_executor_when_background_write_fails(monkeypatch):
    native = pytest.importorskip("megatron.core.dist_checkpointing.strategies.async_utils")
    from verl.utils.checkpoint.megatron_thread_checkpoint import (
        ThreadedCheckpointWriter,
        create_thread_checkpoint_queue,
    )

    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    queue = create_thread_checkpoint_queue(
        synchronize=lambda: None,
        sync_all_done=lambda alive: not alive,
        sync_failure=_raise_local_error,
    )
    writer = ThreadedCheckpointWriter(queue=queue)
    request = native.AsyncRequest(lambda: (_ for _ in ()).throw(OSError("background write boom")), (), [])
    writer.schedule([request], lambda: None)
    with pytest.raises(OSError, match="background write boom"):
        writer.close()
    assert not writer.pending and writer.failed
    with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
        queue._executor.submit(lambda: None)
