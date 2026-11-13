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
"""TeacherTopKDump must answer both engine call shapes with the right payload.

MegatronEngine calls the same object twice per micro-batch under different
signatures, and a mismatch there fails in two very different ways: as a processor
the engine *merges the returned dict into the model output* (so a wrong key name
means the dump is silently empty), and as a loss the engine expects
``(loss, metrics)`` (a bare tensor raises inside the pipeline schedule). Both are
pinned here, along with the flag check that turns a missing
``distillation_use_topk`` into a loud error instead of an empty dump.
"""

import pytest

torch = pytest.importorskip("torch")

from verl.workers.utils.teacher_dump import (  # noqa: E402
    DUMP_IDS_KEY,
    DUMP_LOGPROBS_KEY,
    TeacherTopKDump,
)


def test_processor_call_returns_topk_under_the_batch_key_names():
    torch.manual_seed(0)
    logits = torch.randn(2, 3, 11)
    dump = TeacherTopKDump(topk=4)
    out = dump(student_logits=logits)

    assert set(out) == {DUMP_IDS_KEY, DUMP_LOGPROBS_KEY}
    assert out[DUMP_IDS_KEY].shape == (2, 3, 4)
    assert out[DUMP_LOGPROBS_KEY].shape == (2, 3, 4)
    assert out[DUMP_IDS_KEY].dtype == torch.int64
    assert out[DUMP_LOGPROBS_KEY].dtype == torch.float32

    # the payload must be the real top-k, not the first k ids
    want = torch.log_softmax(logits.float(), dim=-1).topk(4, dim=-1)
    assert torch.equal(out[DUMP_IDS_KEY], want.indices)
    torch.testing.assert_close(out[DUMP_LOGPROBS_KEY], want.values, rtol=0, atol=1e-6)


def test_loss_call_returns_a_zero_scalar_and_metrics():
    logits = torch.randn(1, 2, 3)
    dump = TeacherTopKDump(topk=2)
    processed = dump(student_logits=logits)
    loss, metrics = dump(model_output=processed)

    assert loss.shape == (), "the engine expects a scalar loss"
    assert loss.item() == 0.0
    assert isinstance(metrics, dict)


def test_loss_call_names_the_missing_flag():
    """An empty dump must fail loudly: the batch forgot distillation_use_topk."""
    dump = TeacherTopKDump(topk=2)
    with pytest.raises(KeyError, match="distillation_use_topk"):
        dump(model_output={})


def test_unknown_signature_is_rejected():
    dump = TeacherTopKDump(topk=2)
    with pytest.raises(TypeError, match="logits processor"):
        dump(something_else=torch.zeros(1))


def test_rejects_non_positive_topk():
    with pytest.raises(ValueError, match="positive"):
        TeacherTopKDump(topk=0)


def test_consistency_stats_match_the_engines_own_log_probs():
    """The exported log-prob at the true token must equal log_probs exactly."""
    from verl.workers.utils.teacher_dump import dump_consistency_stats

    torch.manual_seed(0)
    b, s, v, k = 1, 5, 12, 4
    logits = torch.randn(b, s, v)
    ref = torch.log_softmax(logits.float(), dim=-1)
    topk = ref.topk(k, dim=-1)
    input_ids = torch.randint(1, v, (b, s))

    # verl's log_probs are POSITION-level: entry i is p(tokens[i + 1]), so the
    # fixture must score the rolled targets, not the current token.
    model_output = {
        DUMP_IDS_KEY: topk.indices,
        DUMP_LOGPROBS_KEY: topk.values,
        "log_probs": ref.gather(-1, torch.roll(input_ids, shifts=-1, dims=1).unsqueeze(-1)).squeeze(-1),
    }
    stats = dump_consistency_stats(model_output, input_ids)
    assert stats["checked_positions"] > 0
    assert stats["max_abs_diff"] < 1e-6, stats
    assert 0.0 < stats["coverage"] <= 1.0, stats


def test_consistency_stats_reports_missing_log_probs_instead_of_passing():
    from verl.workers.utils.teacher_dump import dump_consistency_stats

    model_output = {
        DUMP_IDS_KEY: torch.zeros(1, 3, 2, dtype=torch.int64),
        DUMP_LOGPROBS_KEY: torch.zeros(1, 3, 2),
    }
    stats = dump_consistency_stats(model_output, torch.zeros(1, 3, dtype=torch.int64))
    assert stats["coverage"] == 0.0
    assert "no log_probs" in stats["note"]


def test_consistency_stats_detects_a_wrong_distribution():
    """A dump shifted by one position must fail the comparison, not pass quietly."""
    from verl.workers.utils.teacher_dump import dump_consistency_stats

    torch.manual_seed(1)
    b, s, v, k = 1, 6, 10, 3
    logits = torch.randn(b, s, v)
    ref = torch.log_softmax(logits.float(), dim=-1)
    input_ids = torch.randint(1, v, (b, s))
    topk = ref.topk(k, dim=-1)
    # corrupt: roll the ids one position, keeping the log-probs, so ids and
    # log-probs no longer describe the same distribution
    model_output = {
        DUMP_IDS_KEY: torch.roll(topk.indices, shifts=1, dims=1),
        DUMP_LOGPROBS_KEY: topk.values,
        "log_probs": ref.gather(-1, torch.roll(input_ids, shifts=-1, dims=1).unsqueeze(-1)).squeeze(-1),
    }
    stats = dump_consistency_stats(model_output, input_ids)
    assert stats["checked_positions"] > 0
    assert stats["max_abs_diff"] > 1e-4, stats


def test_nested_consistency_excludes_every_sequence_end_including_the_last():
    from verl.workers.utils.teacher_dump import dump_consistency_stats

    input_ids = torch.nested.nested_tensor([torch.tensor([1, 2, 3]), torch.tensor([4, 5])], layout=torch.jagged)
    logps = torch.randn(1, 5, 8).log_softmax(-1)
    top = logps.topk(8, dim=-1)
    scored = logps.gather(-1, input_ids.values().roll(-1).reshape(1, 5, 1)).squeeze(-1)
    # There is no target after either final token; physical alignment may score
    # padding there. Neither phantom prediction belongs in the consistency check.
    scored[0, [2, 4]] = 1000.0
    stats = dump_consistency_stats(
        {DUMP_IDS_KEY: top.indices, DUMP_LOGPROBS_KEY: top.values, "log_probs": scored}, input_ids
    )
    assert stats["coverage"] == 1.0
    assert stats["checked_positions"] == 3
    assert stats["max_abs_diff"] < 1e-6


def test_capture_dump_tensors_copies_to_cpu_and_keeps_offsets():
    from verl.workers.utils.teacher_dump import capture_dump_tensors

    model_output = {
        DUMP_IDS_KEY: torch.zeros(2, 3, 4, dtype=torch.int64),
        DUMP_LOGPROBS_KEY: torch.zeros(2, 3, 4),
    }
    captured = capture_dump_tensors(model_output, torch.zeros(2, 3, dtype=torch.int64))
    assert captured["ids"].shape == (2, 3, 4)
    assert captured["nested"] is False
    assert captured["offsets"] is None
    assert captured["input_ids"].device.type == "cpu"


def test_capture_dump_tensors_keeps_nested_offsets():
    from torch.nested import nested_tensor

    from verl.workers.utils.teacher_dump import capture_dump_tensors

    # each element is one sequence of `n` tokens; the bounds below follow from these lengths
    seqs = [torch.zeros(3, dtype=torch.int64), torch.zeros(5, dtype=torch.int64)]
    nested_ids = nested_tensor(seqs, layout=torch.jagged)
    model_output = {
        DUMP_IDS_KEY: nested_ids.clone(),
        DUMP_LOGPROBS_KEY: nested_ids.to(torch.float32).clone(),
    }
    captured = capture_dump_tensors(model_output, nested_ids)
    assert captured["nested"] is True
    assert captured["offsets"].tolist() == [0, 3, 8], captured["offsets"]
    # stored flat, not as a nested object: torch.save cannot round-trip jagged
    assert not getattr(captured["input_ids"], "is_nested", False)
    assert captured["input_ids"].numel() == 8


def test_capture_dump_tensors_names_the_missing_flag():
    from verl.workers.utils.teacher_dump import capture_dump_tensors

    with pytest.raises(KeyError, match="distillation_use_topk"):
        capture_dump_tensors({}, torch.zeros(1, 2, dtype=torch.int64))


def test_capture_flattens_nested_ids_and_logps_too():
    """The engine wraps the dump itself in the batch layout, not only input_ids."""
    from torch.nested import nested_tensor

    from verl.workers.utils.teacher_dump import capture_dump_tensors

    seqs = [torch.zeros(3, 4, dtype=torch.int64), torch.zeros(5, 4, dtype=torch.int64)]
    nested_ids = nested_tensor(seqs, layout=torch.jagged)
    model_output = {DUMP_IDS_KEY: nested_ids.clone(), DUMP_LOGPROBS_KEY: nested_ids.float().clone()}
    captured = capture_dump_tensors(model_output, nested_ids)
    assert not getattr(captured["ids"], "is_nested", False), "ids must be stored flat"
    assert not getattr(captured["logps"], "is_nested", False), "logps must be stored flat"
    assert captured["ids"].shape == (8, 4), captured["ids"].shape


def test_trainer_dump_requests_logits_and_has_one_writer_across_dp(monkeypatch):
    """Both DP sources collect the corpus; only DP rank 0 writes the shared file."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    from tensordict.tensorclass import NonTensorData

    from verl.utils import tensordict_utils as tu

    # Exercise the actual trainer method without constructing a model/optimizer
    # or importing the trainer's unrelated distributed-loader dependencies.
    source = Path(__file__).parents[2] / "verl/trainer/sft_trainer.py"
    tree = ast.parse(source.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SFTTrainer")
    method = next(
        node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "dump_teacher_topk"
    )
    namespace = dict(torch=torch, tu=tu, NonTensorData=NonTensorData, logger=None, log_with_rank=lambda *a, **k: None)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)

    class Config(dict):
        __getattr__ = dict.__getitem__

    saves = []
    observed_flags = []
    monkeypatch.setattr(torch, "save", lambda payload, path: saves.append((payload, path)))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def gather(output, captured):
        output[:] = [captured, captured]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    input_ids = torch.nested.nested_tensor([torch.tensor([1, 2, 3])], layout=torch.jagged)

    def infer(data, loss_function):
        observed_flags.append(tu.get_non_tensor_data(data, "needs_vocab_logits", False))
        assert not tu.get_non_tensor_data(data, "distillation_use_topk", False)
        return {
            "model_output": {
                DUMP_IDS_KEY: torch.zeros(1, 3, 2, dtype=torch.int64),
                DUMP_LOGPROBS_KEY: torch.zeros(1, 3, 2),
            }
        }

    for dp_rank in (0, 1):
        engine = SimpleNamespace(
            is_mp_src_rank_with_outputs=lambda: True,
            get_data_parallel_rank=lambda dp_rank=dp_rank: dp_rank,
            infer_batch=infer,
        )
        instance = SimpleNamespace(
            config=SimpleNamespace(
                trainer=SimpleNamespace(teacher_dump=Config(path="teacher.pt", topk=2, save_tensors=True)),
                model=SimpleNamespace(use_remove_padding=True, path="test-model"),
                data=SimpleNamespace(
                    use_dynamic_bsz=False,
                    max_token_len_per_gpu=16,
                    micro_batch_size_per_gpu=1,
                    pad_mode="no_padding",
                ),
            ),
            engine=engine,
            engine_config=SimpleNamespace(tensor_model_parallel_size=1),
            global_batch_size=2,
            model_config=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0)),
            train_dataloader=[{"input_ids": input_ids}],
            _get_batch_seqlens=lambda data: [3],
            _dump_consistency_stats=lambda *args: {"coverage": 1.0, "max_abs_diff": 0.0},
        )
        namespace["dump_teacher_topk"](instance)

    assert observed_flags == [True, True]
    assert len(saves) == 1
    payload, path = saves[0]
    assert path == "teacher.pt"
    assert len(payload["tensors"]) == 2
