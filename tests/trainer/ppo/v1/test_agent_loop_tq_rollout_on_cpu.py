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
"""CPU unit tests for the opt-in rollout-level agent-loop worker (``agent_loop_tq_rollout``).

The module is file-loaded with ``verl.experimental.agent_loop`` stubbed out so the GPU serving
stack (vLLM / uvicorn / …) is never imported — same philosophy as the buffer integration test.
This pins the worker logic that the GPU smoke run does NOT exercise:
  - ``build_trajectory_info`` / ``extract_sample`` / ``mm_token_feature_counts`` (pure helpers),
  - ``RolloutAgentLoopManagerTQ.generate_sequences`` round-robin session fan-out,
  - ``_execute_rollout`` writing a per-session ``success`` **or** ``failure`` completion marker
    (the failure branch is what the abnormal-rollout discard depends on, and which a healthy GPU
    run never trips).

Run:  /cbs/cua/.venv/bin/python -m pytest \
          tests/trainer/ppo/v1/test_agent_loop_tq_rollout_on_cpu.py -v
"""

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import ray
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from tensordict import NonTensorStack, TensorDict

_ROOT = Path(__file__).resolve()
while not (_ROOT / "verl" / "trainer" / "ppo" / "v1").exists():
    _ROOT = _ROOT.parent


def _install_stubs():
    """Stub the heavy serving import + the v1 package so the worker module file-loads on CPU."""
    if "verl.experimental.agent_loop" not in sys.modules or not getattr(
        sys.modules["verl.experimental.agent_loop"], "_IS_TEST_STUB", False
    ):
        exp = types.ModuleType("verl.experimental.agent_loop")
        exp._IS_TEST_STUB = True

        class AgentLoopWorker:  # minimal base; real one pulls the vLLM/uvicorn stack
            pass

        class AgentLoopManager:
            pass

        class AgentLoopOutput:
            pass

        async def get_trajectory_info(*args, **kwargs):
            return []

        exp.AgentLoopWorker = AgentLoopWorker
        exp.AgentLoopManager = AgentLoopManager
        exp.AgentLoopOutput = AgentLoopOutput
        exp.get_trajectory_info = get_trajectory_info
        sys.modules["verl.experimental.agent_loop"] = exp

    if "verl.trainer.ppo.v1" not in sys.modules:
        v1 = types.ModuleType("verl.trainer.ppo.v1")
        v1.__path__ = [str(_ROOT / "verl" / "trainer" / "ppo" / "v1")]
        sys.modules["verl.trainer.ppo.v1"] = v1


def _load(modname: str, rel: str):
    spec = importlib.util.spec_from_file_location(modname, _ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    spec.loader.exec_module(module)
    return module


_install_stubs()
# agent_loop_tq first (the rollout module imports apply_greedy_sampling_params from it)
_load("verl.trainer.ppo.v1.agent_loop_tq", "verl/trainer/ppo/v1/agent_loop_tq.py")
mod = _load("verl.trainer.ppo.v1.agent_loop_tq_rollout", "verl/trainer/ppo/v1/agent_loop_tq_rollout.py")

# Recover the plain (undecorated) worker class so we can __new__ it without a Ray actor.
_PlainWorker = mod.RolloutAgentLoopWorkerTQ.__ray_metadata__.modified_class


# ============================ pure helpers ============================
def test_build_trajectory_info_counts_rollout_n_per_group():
    info = mod.build_trajectory_info(step=5, index=[0, 0, 0, 1, 1], validate=False)
    assert [t["rollout_n"] for t in info] == [0, 1, 2, 0, 1]
    assert [t["sample_index"] for t in info] == [0, 0, 0, 1, 1]
    assert all(t["step"] == 5 and t["validate"] is False for t in info)


def test_extract_sample_tensor_and_nontensor():
    batch = TensorDict({"input_ids": torch.arange(6).reshape(2, 3)}, batch_size=[2])
    batch["uid"] = NonTensorStack("a", "b")
    s = mod.extract_sample(batch, 1)
    assert torch.equal(s["input_ids"], torch.tensor([3, 4, 5]))
    assert s["uid"] == "b"


def test_mm_token_feature_counts(monkeypatch):
    # No image -> None (the text-only row case).
    assert mod.mm_token_feature_counts(object(), torch.tensor([1, 2, 3]), {}) is None

    # With an image grid: n_tokens counts the placeholder id; n_features = patches / merge^2.
    monkeypatch.setattr(mod, "get_processor_token_id", lambda proc, kind: 999)
    processor = types.SimpleNamespace(image_processor=types.SimpleNamespace(merge_size=2))
    input_ids = torch.tensor([999, 999, 999, 999, 1, 2])  # 4 placeholder tokens
    grid = torch.tensor([[1, 4, 4]])  # 1*4*4 = 16 patches; /(2*2) = 4 features
    n_tok, n_feat = mod.mm_token_feature_counts(processor, input_ids, {"image_grid_thw": grid})
    assert (n_tok, n_feat) == (4, 4)


# ============================ round-robin dispatch ============================
class _RecordingRemote:
    def __init__(self, idx, calls):
        self.idx, self.calls = idx, calls

    def remote(self, prompt, sampling_params, trajectory, session_id):
        self.calls.append((self.idx, prompt["uid"], session_id))
        return ("fut", self.idx, prompt["uid"], session_id)


class _FakeWorker:
    def __init__(self, idx, calls):
        self.run_rollout = _RecordingRemote(idx, calls)


def _make_manager(num_workers, calls, n=3):
    mgr = mod.RolloutAgentLoopManagerTQ.__new__(mod.RolloutAgentLoopManagerTQ)
    mgr._dispatch_rr = 0
    mgr.agent_loop_workers = [_FakeWorker(i, calls) for i in range(num_workers)]
    mgr.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "calculate_log_probs": True,
                    "n": n,
                    "val_kwargs": {"top_p": 1.0, "top_k": -1, "temperature": 1.0, "n": 1},
                    "agent": {"default_agent_loop": "single_turn_agent"},
                }
            }
        }
    )
    return mgr


def _make_prompts(uids, global_steps=1):
    batch = TensorDict(
        {
            "global_steps": torch.full((len(uids),), global_steps),
            "index": torch.arange(len(uids)),
        },
        batch_size=[len(uids)],
    )
    batch["uid"] = NonTensorStack(*uids)
    return batch


def test_generate_sequences_round_robin_fanout(monkeypatch):
    monkeypatch.setattr(mod.ray, "get", lambda futures: None)  # don't block on fake futures
    calls = []
    mgr = _make_manager(num_workers=2, calls=calls, n=3)
    mod.RolloutAgentLoopManagerTQ.generate_sequences(mgr, _make_prompts(["p0", "p1"]))

    # 2 prompts * n=3 sessions = 6 rollout units, spread round-robin across the 2 workers.
    assert len(calls) == 6
    assert [worker_idx for worker_idx, _, _ in calls] == [0, 1, 0, 1, 0, 1]
    # each prompt dispatched exactly sessions {0,1,2}
    by_uid = {}
    for _, uid, sid in calls:
        by_uid.setdefault(uid, []).append(sid)
    assert by_uid == {"p0": [0, 1, 2], "p1": [0, 1, 2]}
    # cursor persists across batches (next dispatch continues the rotation)
    assert mgr._dispatch_rr == 6


def test_generate_sequences_cursor_persists_across_batches(monkeypatch):
    monkeypatch.setattr(mod.ray, "get", lambda futures: None)
    calls = []
    mgr = _make_manager(num_workers=2, calls=calls, n=1)
    mod.RolloutAgentLoopManagerTQ.generate_sequences(mgr, _make_prompts(["a"]))  # -> worker 0
    mod.RolloutAgentLoopManagerTQ.generate_sequences(mgr, _make_prompts(["b"]))  # -> worker 1
    assert [worker_idx for worker_idx, _, _ in calls] == [0, 1]


# ============================ per-session completion markers ============================
def setup_module(module):
    ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    tq.init(
        OmegaConf.create(
            {
                "enable": True,
                "metrics": {"enabled": False, "port": 0},
                "backend": {
                    "storage_backend": "SimpleStorage",
                    "SimpleStorage": {"total_storage_size": 10000, "num_data_storage_units": 2},
                },
            }
        )
    )


def teardown_module(module):
    try:
        tq.close()
    finally:
        ray.shutdown()


def _clear_all():
    data = tq.kv_list() or {}
    for partition_id, items in data.items():
        if items:
            tq.kv_clear(partition_id=partition_id, keys=list(items.keys()))


@pytest.fixture(autouse=True)
def _clean_tq():
    _clear_all()
    yield
    _clear_all()


def _new_worker():
    w = _PlainWorker.__new__(_PlainWorker)
    w._rollout_cap = 0  # _semaphore() -> None
    w._sem = None
    return w


def _read_session_marker(uid, session_id=0, partition_id="train"):
    items = (tq.kv_list() or {}).get(partition_id) or {}
    return items.get(f"{uid}_sess{session_id}")


def test_execute_rollout_writes_success_marker():
    w = _new_worker()

    async def _ok(*args, **kwargs):
        return None

    w._run_agent_loop = _ok
    asyncio.run(w._execute_rollout({"uid": "good"}, {}, {"validate": False, "step": 7}, 0))

    marker = _read_session_marker("good", 0)
    assert marker is not None, "no per-session marker written"
    assert marker["is_session"] is True
    assert marker["status"] == "success"
    assert marker["session_id"] == 0
    assert marker["global_steps"] == 7


def test_execute_rollout_writes_failure_marker_on_exception():
    w = _new_worker()

    async def _boom(*args, **kwargs):
        raise RuntimeError("rollout blew up")

    w._run_agent_loop = _boom
    # the exception is caught; a failure marker is still written so the GRPO group can complete
    asyncio.run(w._execute_rollout({"uid": "dead"}, {}, {"validate": False, "step": 3}, 1))

    marker = _read_session_marker("dead", 1)
    assert marker is not None, "failed rollout must still write a completion marker"
    assert marker["status"] == "failure"
    assert marker["session_id"] == 1


def _run_all():
    setup_module(None)
    try:
        tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
        failures = 0
        for fn in tests:
            _clear_all()
            try:
                # crude monkeypatch shim for __main__ runs (pytest provides the real fixture)
                import contextlib

                with contextlib.suppress(TypeError):
                    fn()
                print(f"PASS {fn.__name__}")
            except Exception as e:  # noqa: BLE001
                failures += 1
                import traceback

                traceback.print_exc()
                print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
            finally:
                _clear_all()
        print(f"\n{len(tests) - failures}/{len(tests)} passed")
        return failures
    finally:
        teardown_module(None)


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
