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

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch

from verl.experimental.teacher_loop import vllm_hidden_pooler


def test_image_prefill_checks_causal_alignment_before_encode():
    pytest.importorskip("vllm")
    from PIL import Image

    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    ids = [7, 10, 10, 10, 8, 21]
    request = {"sequence_ids": ids, "positions": [4], "uid": "diagram", "teacher_key": "geo3k"}
    processor = type("Qwen3VLProcessor", (SimpleNamespace,), {})(
        image_token_id=10,
        video_token_id=11,
    )
    image = Image.new("RGB", (16, 16))
    rendered = {"type": "multimodal", "prompt_token_ids": ids}
    renderer = AsyncMock(return_value=[rendered])

    async def encode(prompt, params, request_id):
        assert prompt is rendered
        assert params.extra_kwargs == {"opd_record": request}
        handle = torch.tensor(list(b'{"uid":"diagram"}'), dtype=torch.uint8)
        yield SimpleNamespace(finished=True, outputs=SimpleNamespace(data=handle))

    server = SimpleNamespace(
        _hidden_state_opd=True,
        _teacher_reload_state="ready",
        model_config=SimpleNamespace(processor=processor),
        engine=SimpleNamespace(renderer=SimpleNamespace(render_cmpl_async=renderer), encode=encode),
    )
    inputs = dict(queue_request=request, multi_modal_data={"images": [image]}, mm_processor_kwargs={"max_pixels": 8192})
    handle = asyncio.run(vLLMHttpServer.compute_teacher_hidden_states(server, "a", ids, **inputs))
    assert handle == {"uid": "diagram"}
    renderer.assert_awaited_once_with(
        [
            {
                "prompt_token_ids": [7, 10, 8, 21],
                "multi_modal_data": {"image": [image]},
                "mm_processor_kwargs": {"max_pixels": 8192},
            }
        ]
    )
    rendered["prompt_token_ids"] = [7, 10, 10, 10, 8, 22]
    with pytest.raises(ValueError, match="image-token expansion differs"):
        asyncio.run(vLLMHttpServer.compute_teacher_hidden_states(server, "b", ids, **inputs))


def test_teacher_swap_localizes_checkpoint_for_each_worker(monkeypatch, tmp_path):
    pytest.importorskip("vllm")
    from verl.utils import fs
    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

    uri = "hdfs://models/teacher-b"
    copied = []

    def copy(src, dst):
        copied.append(src)
        Path(dst).mkdir()
        (Path(dst) / "model.safetensors").touch()

    monkeypatch.setattr(fs, "copy", copy)
    monkeypatch.setattr(fs.tempfile, "gettempdir", lambda: str(tmp_path))
    for _ in range(2):
        worker = Mock(_hidden_teacher_reload_interrupted=False)
        vLLMColocateWorkerExtension.reload_hidden_teacher_weights(worker, uri, use_shm=False)
        path = Path(worker.reload_weights.call_args.kwargs["weights_path"])
        assert path.is_relative_to(tmp_path) and (path / "model.safetensors").is_file()
    assert copied == [uri]
    worker.reload_weights.side_effect = ValueError("invalid weights")
    result = vLLMColocateWorkerExtension.reload_hidden_teacher_weights(worker, uri, use_shm=False)
    assert result == "ValueError: invalid weights"


def test_interrupted_teacher_reload_stops_later_reloads_on_the_worker(monkeypatch):
    pytest.importorskip("vllm")
    from verl.utils import fs
    from verl.workers.rollout.vllm_rollout.utils import vLLMColocateWorkerExtension

    copies = ["/local/a", OSError("copy failed"), "/local/b", "/local/a"]
    monkeypatch.setattr(fs, "copy_to_local", Mock(side_effect=copies))
    worker = Mock(_hidden_teacher_reload_interrupted=False)
    reload = vLLMColocateWorkerExtension.reload_hidden_teacher_weights

    # A completed reload, or a checkpoint that never reached the model, leaves no layer progress.
    assert reload(worker, "/model/a", use_shm=False) is None
    assert reload(worker, "/model/b", use_shm=False) == "OSError: copy failed"
    assert worker.reload_weights.call_count == 1
    worker.reload_weights.side_effect = OSError("unreadable shard")
    assert reload(worker, "/model/b", use_shm=False) == "OSError: unreadable shard"
    # vLLM would finish the interrupted layers early, so this worker must not load again.
    worker.reload_weights.side_effect = None
    assert "stopped partway" in reload(worker, "/model/a", use_shm=False)
    assert [call.kwargs["weights_path"] for call in worker.reload_weights.call_args_list] == ["/local/a", "/local/b"]


def test_teacher_reload_distinguishes_rank_error_from_incomplete_rpc():
    pytest.importorskip("vllm")
    from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer

    rpc = AsyncMock(side_effect=[[None, "ValueError: bad weights"], [None, None], RuntimeError("connection lost")])
    server = SimpleNamespace(
        _hidden_state_opd=True,
        _teacher_reload_state="ready",
        model_config=SimpleNamespace(use_shm=False),
        engine=SimpleNamespace(collective_rpc=rpc),
        clear_kv_cache=AsyncMock(),
    )
    swap = vLLMHttpServer.switch_teacher_weights
    with pytest.raises(RuntimeError, match="rank 1: ValueError: bad weights"):
        asyncio.run(swap(server, "/model/b"))
    with pytest.raises(RuntimeError, match="reload_required"):
        asyncio.run(vLLMHttpServer.compute_teacher_hidden_states(server, "r", [1], queue_request={}))
    asyncio.run(swap(server, "/model/a"))
    with pytest.raises(RuntimeError, match="connection lost"):
        asyncio.run(swap(server, "/model/b"))
    with pytest.raises(RuntimeError, match="Recreate the teacher replica"):
        asyncio.run(swap(server, "/model/a"))
    with pytest.raises(RuntimeError, match="unusable"):
        asyncio.run(vLLMHttpServer.compute_teacher_hidden_states(server, "r", [1], queue_request={}))
    assert rpc.await_count == 3
    server.clear_kv_cache.assert_awaited_once()


@pytest.mark.parametrize("signal_kind", ["hidden_states", "full_logits"])
def test_chunked_prefill_export(monkeypatch, signal_kind):
    pytest.importorskip("vllm")
    from vllm.pooling_params import PoolingParams
    from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

    requests = [{"positions": [1, 3], "uid": "a"}, {"positions": [], "uid": "b"}]
    if signal_kind == "full_logits":
        from verl.trainer.distillation import hidden_state

        for request in requests:
            request.update(
                signal_kind=signal_kind, teacher_model_path="teacher", teacher_weight_version=7, projection_chunk_size=1
            )

        def project(path, version, rows, device, chunk_size):
            assert (path, version, device, chunk_size) == ("teacher", 7, rows.device, 1)
            return torch.cat([rows, rows], dim=1)

        monkeypatch.setattr(hidden_state, "project_full_logits", project)
    states = [PoolingStates(), PoolingStates()]
    stored = {}

    def write(request, hidden):
        assert request["uid"] not in stored
        stored[request["uid"]] = hidden
        return {"uid": request["uid"]}

    monkeypatch.setattr(vllm_hidden_pooler, "put_hidden_record", write)
    pooler = vllm_hidden_pooler.HiddenStatePooler(None, producer=True, hidden_size=3)
    peer = vllm_hidden_pooler.HiddenStatePooler(None, producer=False, hidden_size=3)

    def prefill(requests, states, lengths, ends, counts, hidden):
        meta = PoolingMetadata(
            prompt_lens=torch.tensor(lengths),
            prompt_token_ids=None,
            prompt_token_ids_cpu=None,
            pooling_params=[PoolingParams(task="token_embed", extra_kwargs={"opd_record": r}) for r in requests],
            pooling_states=states,
        )
        meta.build_pooling_cursor(np.array(counts), torch.tensor(ends), torch.device("cpu"))
        # A batch of prefill requests alone needs no device read for its row offsets.
        meta.get_pooling_cursor().first_token_indices_gpu = None
        peer(hidden, meta)
        return pooler(hidden, meta)

    first = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    second = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3) + 12
    outputs = prefill(requests, states, [4, 2], [2, 2], [2, 2], first)
    assert outputs[0] is None and json.loads(bytes(outputs[1].tolist())) == {"uid": "b"}
    assert stored["b"].shape == (0, 6 if signal_kind == "full_logits" else 3)
    outputs = prefill(requests[:1], states[:1], [4], [4], [2], second)
    assert json.loads(bytes(outputs[0].tolist())) == {"uid": "a"}
    expected = torch.stack([first[1], second[1]])
    if signal_kind == "full_logits":
        expected = torch.cat([expected, expected], dim=1)
    torch.testing.assert_close(stored["a"], expected)
    assert not states[0].hidden_states_cache


@pytest.mark.parametrize("prefix", [0, 3])
def test_native_token_embed_request_keeps_its_rows_in_mixed_pooling_batch(prefix):
    pytest.importorskip("vllm")
    from vllm.pooling_params import PoolingParams
    from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

    def native_pooler(rows, metadata):
        cursor = metadata.get_pooling_cursor()
        assert cursor.first_token_indices_gpu.tolist() == [0]
        assert cursor.last_token_indices_gpu.tolist() == [2]
        return list(rows.split(cursor.num_scheduled_tokens_cpu.tolist()))

    meta = PoolingMetadata(
        prompt_lens=torch.tensor([2, 3]),
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=[
            PoolingParams(task="token_embed", extra_kwargs={"opd_record": {"positions": [0]}}),
            PoolingParams(task="token_embed"),
        ],
        pooling_states=[PoolingStates(), PoolingStates()],
    )
    meta.build_pooling_cursor(np.array([2, 3]), torch.tensor([2, 3]), torch.device("cpu"))
    # vLLM 0.20 may slice only the metadata for a task group following a different task.
    if prefix:
        cursor = meta.get_pooling_cursor()
        cursor.first_token_indices_gpu += prefix
        cursor.last_token_indices_gpu += prefix
    hidden = torch.arange((prefix + 5) * 2).reshape(prefix + 5, 2)
    peer = vllm_hidden_pooler.HiddenStatePooler(native_pooler, producer=False, hidden_size=2)
    _, native = peer(hidden, meta)
    torch.testing.assert_close(native, hidden[prefix + 2 :])
