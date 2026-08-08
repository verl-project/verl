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
"""On-CPU tests for the RemoteBackend registry + trainer/engine wiring.

No GPU, no Ray, no plugin required.
"""

from __future__ import annotations

import pytest


def test_remote_backend_registry_decorator_and_dupes():
    from verl.remote_backend.base import RemoteBackend, RemoteBackendRegistry

    @RemoteBackendRegistry.register("test_dummy_backend")
    class Dummy(RemoteBackend):
        @classmethod
        def from_config(cls, main_config, *, handle=None):
            return cls()

        def reconnect_handle(self):
            return {}

        def destroy(self):
            return None

        async def update_weights(self):
            return {}

        async def save_checkpoint(self):
            return {}

        def requires_single_forwarder(self):
            return True

    assert RemoteBackendRegistry.get("test_dummy_backend") is Dummy
    assert "test_dummy_backend" in RemoteBackendRegistry.list()

    # Same class re-registration is a no-op.
    RemoteBackendRegistry.register("test_dummy_backend")(Dummy)

    # Different class under the same name raises.
    class OtherDummy(Dummy):
        pass

    with pytest.raises(ValueError, match="already registered"):
        RemoteBackendRegistry.register("test_dummy_backend")(OtherDummy)


def test_remote_backend_worker_loader_is_lazy():
    from verl.remote_backend.base import RemoteBackendRegistry

    calls = {"n": 0}

    def _loader():
        calls["n"] += 1

        class DummyWorker:
            pass

        return DummyWorker

    RemoteBackendRegistry.register_worker("test_lazy_worker_backend", _loader)
    assert calls["n"] == 0  # loader not invoked at registration time
    assert RemoteBackendRegistry.get_worker("test_lazy_worker_backend").__name__ == "DummyWorker"
    assert calls["n"] == 1
    # cached
    assert RemoteBackendRegistry.get_worker("test_lazy_worker_backend").__name__ == "DummyWorker"
    assert calls["n"] == 1


def test_remote_backend_unknown_backend_raises():
    from verl.remote_backend.base import RemoteBackendRegistry

    with pytest.raises(KeyError, match="Unknown remote backend"):
        RemoteBackendRegistry.get("no_such_backend_xyz")


def test_checkpoint_engine_registry_has_remote_backend():
    import verl.checkpoint_engine  # noqa: F401 -- triggers registration
    from verl.checkpoint_engine.base import CheckpointEngineRegistry
    from verl.checkpoint_engine.remote_backend import RemoteBackendCheckpointEngine

    assert CheckpointEngineRegistry.get("remote_backend") is RemoteBackendCheckpointEngine

    engine = CheckpointEngineRegistry.new("remote_backend", bucket_size=0, is_master=True)
    assert isinstance(engine, RemoteBackendCheckpointEngine)
    # send/receive must raise -- the manager should short-circuit before them.
    with pytest.raises(NotImplementedError):
        list(engine.receive_weights())


def test_trainer_registry_has_remote_backend():
    from verl.trainer.ppo.v1 import PPOTrainerRemoteBackend, get_trainer_cls

    assert get_trainer_cls("remote_backend") is PPOTrainerRemoteBackend


def test_ppo_trainer_hooks_default_to_naive():
    """Extension hook defaults must preserve non-plugin V1 behaviour."""
    from verl.trainer.ppo.v1.trainer_base import PPOTrainer

    # Call the hooks unbound to avoid the cost of a real PPOTrainer instance.
    assert PPOTrainer._checkpoint_engine_backend(None) == "naive"
    assert PPOTrainer._actor_rollout_wg_extra_kwargs(None) == {}
    assert PPOTrainer._llm_server_replica_init_kwargs(None) == {}
