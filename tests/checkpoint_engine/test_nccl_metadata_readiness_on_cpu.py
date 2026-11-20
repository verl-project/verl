# Copyright 2026 Individual Contributor
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
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import zmq

from verl.checkpoint_engine import nccl_checkpoint_engine as nccl
from verl.checkpoint_engine.delta_checkpoint_engine import DeltaShardedCheckpointEngine


@pytest.fixture
def collective(monkeypatch):
    # Only the GPU operations are mocked. All metadata traffic uses real TCP sockets.
    mock = Mock()
    mock.is_group_initialized.return_value = False
    monkeypatch.setattr(nccl, "collective", mock)
    monkeypatch.setattr(nccl, "cp", np)
    monkeypatch.setattr(nccl.ray.util, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(nccl.NCCLCheckpointEngine, "get_node_id", staticmethod(lambda: "source"))
    monkeypatch.setattr(nccl, "_SUBSCRIPTION_TIMEOUT_S", 5, raising=False)
    return mock


def bucket(engine, seq=None):
    async def transfer():
        metadata = None if seq is None else {"length": 1, "test_seq": seq}
        op = nccl.BroadcastOperation(
            engine.rank, engine.group_name, bytearray(1), metadata, engine.socket, engine.topic
        )
        return await op.wait_for_complete()

    return asyncio.run(transfer())


@pytest.mark.parametrize("handshake", [False, True], ids=["original-pub-sub", "wait-for-subscriber"])
def test_bucket_zero_with_delayed_subscription(collective, monkeypatch, handshake):
    """Hold SUBSCRIBE in transit until both ranks have entered the group barrier."""
    if not handshake:
        monkeypatch.setattr(nccl.NCCLCheckpointEngine, "_confirm_bucket_subscriptions", False, raising=False)
    root = nccl.NCCLCheckpointEngine(1, is_master=True)
    receiver = nccl.NCCLCheckpointEngine(1)
    master = root.prepare().master
    at_barrier = threading.Event()
    barrier = threading.Barrier(2, action=at_barrier.set, timeout=5)
    collective.barrier.side_effect = lambda _: barrier.wait()

    # Hold exactly one subscription on an otherwise ordinary XSUB/XPUB forwarder.
    # The receiver has already called SUBSCRIBE when the downstream XPUB observes it.
    with zmq.Context() as ctx, ctx.socket(zmq.XSUB) as upstream, ctx.socket(zmq.XPUB) as downstream:
        upstream.linger = downstream.linger = 0
        upstream.rcvtimeo = downstream.rcvtimeo = 5000
        upstream.connect(f"tcp://127.0.0.1:{root.listen_port}")
        port = downstream.bind_to_random_port("tcp://127.0.0.1")
        try:
            with ThreadPoolExecutor(2) as pool:
                sender_init = pool.submit(root.init_process_group, 0, 2, master, 1)
                recv_init = pool.submit(receiver.init_process_group, 1, 2, replace(master, zmq_port=port), 1)
                subscription = downstream.recv()
                assert subscription == b"\x01" + receiver.topic.encode()
                assert at_barrier.wait(5)

                if handshake:
                    assert not sender_init.done()
                    upstream.send(subscription)
                    probe = upstream.recv_multipart()
                    assert probe == [receiver.topic.encode(), b"probe"]
                    downstream.send_multipart(probe)
                    ack = downstream.recv()
                    assert ack == f"\x01ready/{receiver.topic}/1".encode()
                    upstream.send(ack)
                    while True:
                        frames = upstream.recv_multipart()
                        downstream.send_multipart(frames)
                        if frames[-1] == b"complete":
                            break
                sender_init.result(5)
                recv_init.result(5)

            bucket(root, 0)
            if not handshake:
                # The original engine has already published bucket 0. A later
                # marker proves this very connection works once SUBSCRIBE propagates.
                upstream.send(subscription)
                deadline = time.monotonic() + 5
                while not upstream.poll(10):
                    assert time.monotonic() < deadline
                    bucket(root, 1)
            downstream.send_multipart(upstream.recv_multipart())
            receiver.socket.rcvtimeo = 5000
            assert bucket(receiver)["test_seq"] == (0 if handshake else 1)
        finally:
            root.socket.close(0)
            receiver.finalize()


@pytest.mark.parametrize("multi_sender, expected", [(False, [1, 2]), (True, [2, 3])])
@pytest.mark.parametrize("rebuild_group", [False, True])
def test_receiver_set_and_reconnect(collective, multi_sender, expected, rebuild_group):
    def worker(master=False):
        return nccl.NCCLCheckpointEngine(1, multi_sender=multi_sender, rebuild_group=rebuild_group, is_master=master)

    root = worker(master=True)
    actors = [root, worker(), worker()]
    receivers = [worker(), worker()]
    epochs = set()
    try:
        for update in range(2):
            metadata = [root.prepare(), nccl.WorkerMetadata("remote"), nccl.WorkerMetadata("source")]
            metadata += [nccl.WorkerMetadata("rollout")] * len(receivers)
            actor_args, recv_args = root.build_topology(len(actors), len(receivers), metadata)
            assert recv_args["rank"] == expected
            assert actor_args["rank"] == ([0, -1, 1] if multi_sender else [0, -1, -1])
            barrier = threading.Barrier(recv_args["world_size"][0], timeout=5)
            collective.barrier.side_effect = lambda _, current_barrier=barrier: current_barrier.wait()
            collective.is_group_initialized.return_value = bool(update) and not rebuild_group
            with ThreadPoolExecutor(len(actors) + len(receivers)) as pool:
                futures = [
                    pool.submit(engine.init_process_group, **{k: values[i] for k, values in args.items()})
                    for workers, args in [(actors, actor_args), (receivers, recv_args)]
                    for i, engine in enumerate(workers)
                ]
                for future in futures:
                    future.result(5)

            assert all(peer.socket is None for peer in actors[1:])
            epochs.add(root.topic)
            bucket(root, update)
            for receiver in receivers:
                receiver.socket.rcvtimeo = 5000
                assert bucket(receiver)["test_seq"] == update
            for engine in actors + receivers:
                engine.finalize()
        assert len(epochs) == 2
        assert collective.init_collective_group.call_count == barrier.parties * (2 if rebuild_group else 1)
    finally:
        root.socket.close(0)


@pytest.mark.parametrize("rebuild_group", [False, True])
def test_missing_receiver_ignores_stale_and_other_rank_acks(collective, monkeypatch, rebuild_group):
    root = nccl.NCCLCheckpointEngine(1, is_master=True, rebuild_group=rebuild_group)
    old = root.prepare().master
    current = root.prepare().master
    topic = f"bucket_metadata/{current.subscription_epoch.hex}"
    monkeypatch.setattr(nccl, "_SUBSCRIPTION_TIMEOUT_S", 1)

    def publish(master):
        root.init_process_group(0, 4, master, 2)
        bucket(root, 0)

    # Two rollout ranks (2, 3), one relay (1). Rank 3 never confirms this epoch.
    with zmq.Context() as ctx, ctx.socket(zmq.SUB) as sub, ThreadPoolExecutor(1) as pool:
        sub.linger, sub.rcvtimeo = 0, 5000
        sub.subscribe(topic)
        sub.connect(f"tcp://127.0.0.1:{root.listen_port}")
        try:
            future = pool.submit(publish, current)
            assert sub.recv_multipart() == [topic.encode(), b"probe"]
            for ack in [f"{topic}/2", f"{topic}/2", f"{topic}/1", f"bucket_metadata/{old.subscription_epoch.hex}/3"]:
                sub.subscribe(f"ready/{ack}")
            with pytest.raises(TimeoutError, match=r"never_acked=\[3\], acked_then_left=\[\]"):
                future.result(5)
            collective.broadcast.assert_not_called()

            # Retry without a successful finalize. Only fresh ACKs can release it.
            retry = root.prepare().master
            retry_topic = f"bucket_metadata/{retry.subscription_epoch.hex}"
            collective.is_group_initialized.return_value = not rebuild_group
            with ctx.socket(zmq.SUB) as peer:
                peer.linger, peer.rcvtimeo = 0, 5000
                peer.subscribe(retry_topic)
                peer.connect(f"tcp://127.0.0.1:{root.listen_port}")
                future = pool.submit(publish, retry)
                assert peer.recv_multipart() == [retry_topic.encode(), b"probe"]
                sub.subscribe(f"ready/{topic}/3")  # late ACK for the failed attempt
                peer.subscribe(f"ready/{retry_topic}/2")
                # A renewed probe on the same connection proves the gate is still shut.
                assert peer.recv_multipart() == [retry_topic.encode(), b"probe"]
                peer.subscribe(f"ready/{retry_topic}/3")
                future.result(5)
                collective.broadcast.assert_called_once()
            assert collective.destroy_collective_group.call_count == int(rebuild_group)
        finally:
            root.socket.close(0)


@pytest.mark.parametrize("rebuild_group", [False, True])
def test_failed_handshake_retry_bucket_peak(collective, monkeypatch, rebuild_group):
    # Sample while the new allocation AND any references from the previous prepare are live.
    refs = {"sender": [], "receiver": []}
    counts = {"sender": [], "receiver": []}

    def track(role, buffer):
        refs[role].append(weakref.ref(buffer))
        counts[role].append(sum(ref() is not None for ref in refs[role]))
        return buffer

    cpu_zeros = nccl.torch.zeros
    monkeypatch.setattr(
        nccl, "cp", SimpleNamespace(uint8=np.uint8, zeros=lambda n, dtype: track("sender", np.zeros(n, dtype)))
    )
    monkeypatch.setattr(nccl.torch, "zeros", lambda n, dtype, device: track("receiver", cpu_zeros(n, dtype=dtype)))
    root = nccl.NCCLCheckpointEngine(8, is_master=True, rebuild_group=rebuild_group)
    receiver = nccl.NCCLCheckpointEngine(8, rebuild_group=rebuild_group)
    barrier = threading.Barrier(2, timeout=5)
    collective.barrier.side_effect = lambda _: barrier.wait()
    try:
        with ThreadPoolExecutor(2) as pool:
            # A failed init skips finalize on both workers.
            for retry in (False, True):
                master = root.prepare().master
                receiver.prepare()
                collective.is_group_initialized.return_value = retry and not rebuild_group
                monkeypatch.setattr(nccl, "_SUBSCRIPTION_TIMEOUT_S", 5 if retry else 0)
                sends = pool.submit(root.init_process_group, 0, 2, master, 1)
                receives = pool.submit(receiver.init_process_group, 1, 2, master, 1)
                for future in (sends, receives):
                    if retry:
                        future.result(5)
                    else:
                        with pytest.raises(TimeoutError):
                            future.result(5)
        bucket(root, 0)
        receiver.socket.rcvtimeo = 5000
        assert bucket(receiver)["test_seq"] == 0
        assert collective.init_collective_group.call_count == (4 if rebuild_group else 2)
        assert collective.destroy_collective_group.call_count == (2 if rebuild_group else 0)
        assert {role: max(samples) for role, samples in counts.items()} == {"sender": 2, "receiver": 2}
        root.finalize()
        receiver.finalize()
        assert all(ref() is None for role_refs in refs.values() for ref in role_refs)
        assert collective.destroy_collective_group.call_count == (4 if rebuild_group else 0)
    finally:
        root.socket.close(0)


@pytest.mark.parametrize("second_acks_after_exit", [False, True])
def test_timeout_distinguishes_departed_receiver(collective, monkeypatch, second_acks_after_exit):
    root = nccl.NCCLCheckpointEngine(1, is_master=True)
    first = nccl.NCCLCheckpointEngine(1)
    master = root.prepare().master
    topic = f"bucket_metadata/{master.subscription_epoch.hex}"
    first_ack = f"ready/{topic}/1".encode()
    ack_seen, exit_seen, second_ack_seen = threading.Event(), threading.Event(), threading.Event()
    recv = zmq.Socket.recv

    def observe_control_event(socket, *args, **kwargs):
        event = recv(socket, *args, **kwargs)
        if socket is root.socket:
            if event == b"\x01" + first_ack:
                ack_seen.set()
            elif event == b"\x00" + first_ack:
                exit_seen.set()
            elif event == f"\x01ready/{topic}/2".encode():
                second_ack_seen.set()
        return event

    monkeypatch.setattr(zmq.Socket, "recv", observe_control_event)

    # Expire each worker's handshake only after the root has observed the relevant wire events.
    clock = threading.local()
    expire_sender = threading.Event()

    def monotonic():
        return time.monotonic() + (60 if clock.expire.is_set() else 0)

    monkeypatch.setattr(nccl, "time", SimpleNamespace(monotonic=monotonic))

    def init(engine, rank, expire):
        clock.expire = expire
        engine.init_process_group(rank, 3, master, 1)
        if rank == 0:
            bucket(engine, 0)

    root.socket.linger = 0
    with root.socket, zmq.Context() as ctx, ctx.socket(zmq.SUB) as second, ThreadPoolExecutor(2) as pool:
        second.linger, second.rcvtimeo = 0, 5000
        second.subscribe(topic)
        second.connect(f"tcp://127.0.0.1:{root.listen_port}")
        sender = pool.submit(init, root, 0, expire_sender)
        assert second.recv_multipart() == [topic.encode(), b"probe"]
        receiver = pool.submit(init, first, 1, ack_seen)
        assert ack_seen.wait(5)
        with pytest.raises(TimeoutError, match="waiting for completion"):
            receiver.result(5)
        assert exit_seen.wait(5)
        if second_acks_after_exit:
            second.subscribe(f"ready/{topic}/2")
            assert second_ack_seen.wait(5)
        expire_sender.set()
        never_acked = r"\[\]" if second_acks_after_exit else r"\[2\]"
        with pytest.raises(TimeoutError, match=rf"never_acked={never_acked}, acked_then_left=\[1\]"):
            sender.result(5)
        collective.broadcast.assert_not_called()


@pytest.mark.parametrize("send_probe", [False, True], ids=["missing-probe", "missing-completion"])
def test_receiver_timeout_closes_subscription(collective, monkeypatch, send_probe):
    root = nccl.NCCLCheckpointEngine(1, is_master=True)
    receiver = nccl.NCCLCheckpointEngine(1)
    master = root.prepare().master
    topic = f"bucket_metadata/{master.subscription_epoch.hex}".encode()
    monkeypatch.setattr(nccl, "_SUBSCRIPTION_TIMEOUT_S", 1)
    with ThreadPoolExecutor(1) as pool:
        try:
            future = pool.submit(receiver.init_process_group, 1, 2, master, 1)
            root.socket.rcvtimeo = 5000
            assert root.socket.recv() == b"\x01" + topic
            if send_probe:
                root.socket.send_multipart([topic, b"probe"])
                assert root.socket.recv() == b"\x01ready/" + topic + b"/1"
            with pytest.raises(TimeoutError, match="waiting for " + ("completion" if send_probe else "probe")):
                future.result(5)
            assert receiver.socket is None
            collective.broadcast.assert_not_called()
        finally:
            root.socket.close(0)


def test_delta_keeps_its_metadata_protocol(collective):
    root = DeltaShardedCheckpointEngine(1, is_master=True)
    try:
        root.init_process_group(0, 2, root.prepare().master, 1)
        assert root.socket.type == zmq.PUB
        assert root.topic == "bucket_metadata"
    finally:
        root.socket.close(0)
