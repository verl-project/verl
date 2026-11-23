import math
from types import SimpleNamespace

import pytest
import torch

from examples.search_r1.sd_hindsight import build_steps, hindsight, queries
from examples.search_r1.sd_losses import sd_coefficient, topk_union_jsd, trajectory_mean
from examples.search_r1.sd_search import answer_f1


def test_jsd_union_reference_gradient():
    s = torch.randn(4, 11, dtype=torch.float64, requires_grad=True)
    t = torch.randn(4, 11, dtype=torch.float64, requires_grad=True)
    values = topk_union_jsd(s, t, 3)
    expected = []
    for a, b in zip(s, t, strict=False):
        ids = torch.cat((a.topk(3).indices, b.topk(3).indices)).unique()
        p = a[ids].softmax(-1)
        q = b.detach()[ids].softmax(-1)
        m = (p + q) / 2
        expected.append(0.5 * (p * (p.log() - m.log()) + q * (q.log() - m.log())).sum())
    ref = torch.stack(expected)
    assert torch.allclose(values, ref)
    g = torch.autograd.grad(values.sum(), s, retain_graph=True)[0]
    assert torch.allclose(g, torch.autograd.grad(ref.sum(), s)[0])
    values.sum().backward()
    assert t.grad is None
    assert (values >= 0).all() and (values <= math.log(2)).all()


def test_zero_queries_and_warmup():
    v = torch.tensor([[1.0, 3.0], [5.0, 8.0], [1.0, 2.0]])
    mask = torch.tensor([[1, 1], [1, 0], [0, 0]])
    assert trajectory_mean(v, mask) == pytest.approx(7 / 3)
    assert sd_coefficient(49) == 0 and sd_coefficient(50) == 0.001


def test_outcome_partial_f1_separate_from_em():
    assert answer_f1("<answer>New York</answer>", {"target": ["New York City"]}, "answer") == 0.8
    assert answer_f1("<answer>New York</answer>", {"target": ["New York City"]}, "action_budget") == 0


class ByteCodec:
    special_ids = {999}
    tokenizer = SimpleNamespace(pad_token_id=0, encode=lambda text, add_special_tokens=False: list(text.encode()))

    def token_bytes(self, x):
        return bytes([x])


def test_assistant_only_hindsight_and_alignment():
    codec = ByteCodec()
    chunks = [
        ("<think>plan</think><search>北京 father</search>", 1),
        ("<documents><search>SECRET</search>answer</documents>", 0),
        ("<think>next</think><search>death date</search>", 1),
    ]
    ids = []
    mask = []
    for text, generated in chunks:
        raw = list(text.encode())
        ids += raw + [999]
        mask += [generated] * len(raw) + [0]
    q = queries(codec, ids, mask)
    assert len(q) == 2 and q[0].text == "北京 father"
    group = [q, q, q[:1]]
    outcomes = [0, 0.2, 0.2]
    h = hindsight(group, outcomes, 0, 1)
    assert "SECRET" not in h and "<think>" not in h and "<documents>" not in h
    assert "[Outcome]: Incorrect" in h and "[Outcome]: Correct" in h
    steps = build_steps(codec, [7, 8], ids, group, outcomes, 0, 1024, 8192)
    for entry in steps:
        for a, b in zip(entry["student_positions"], entry["teacher_positions"], strict=False):
            assert entry["student_ids"][a] == entry["teacher_ids"][b]
    assert "<search>北京 father</search>" not in h.split("[Sibling Rollout]:")[-1].split("[Outcome]: Correct")[-1]
    with pytest.raises(ValueError):
        build_steps(codec, [7, 8], ids, group, outcomes, 0, 1024, 4)


def test_real_qwen_token_alignment():
    from pathlib import Path

    from transformers import AutoTokenizer

    from examples.search_r1.skill_teachers import SkillTokenCodec

    root = Path(__file__).resolve().parents[4]
    tokenizer = AutoTokenizer.from_pretrained(root / "models/Qwen2.5-3B-Instruct", local_files_only=True)
    codec = SkillTokenCodec(tokenizer)
    content = "<think>查找父亲</think><search>北京父亲 biography</search>"
    ids = tokenizer.encode(content, add_special_tokens=False)
    q = queries(codec, ids, [1] * len(ids))
    assert len(q) == 1
    steps = build_steps(codec, [tokenizer.eos_token_id], ids, [q, q], [0.8, 0], 0)
    x = steps[0]
    assert all(
        x["teacher_ids"][t] == x["student_ids"][s]
        for s, t in zip(x["student_positions"], x["teacher_positions"], strict=False)
    )
    assert b"biography" in b"".join(codec.token_bytes(ids[i]) for i in q[0].positions)


def test_auxiliary_backward_normalization_and_teacher_detach():
    from tensordict import TensorDict
    from torch import nn

    from examples.search_r1.sd_search import MASK, backward_sd
    from verl.utils import tensordict_utils as tu

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(7, 5)
            self.head = nn.Linear(5, 7)

        def forward(self, input_ids, **kwargs):
            # A causal cumulative state makes extra hindsight affect the output.
            return SimpleNamespace(logits=self.head(self.emb(input_ids).cumsum(1)))

    model = Tiny()
    engine = SimpleNamespace(module=model, scaler=None)

    def make(indices, denominator):
        # Row 1 has no queries and must remain in the trajectory denominator.
        fields = {}
        allrows = {
            "student_ids": [[1, 2, 3], [0, 0]],
            "teacher_ids": [[4, 1, 2, 3], [0, 0]],
            "student_positions": [[2], [1]],
            "teacher_positions": [[3], [1]],
            "weights": [[1.0], [0.0]],
        }
        for k, rows in allrows.items():
            dtype = torch.float32 if k == "weights" else torch.long
            fields["sd_0_" + k] = torch.nested.as_nested_tensor(
                [torch.tensor(rows[i], dtype=dtype) for i in indices], layout=torch.jagged
            )
        data = TensorDict(fields, batch_size=[len(indices)])
        tu.assign_non_tensor(
            data,
            sd_alpha=0.001,
            sd_slots=1,
            sd_top_k=3,
            sd_micro_batch=1,
            auxiliary_loss_mask_counts={MASK: denominator},
            dp_size=1,
        )
        return data

    full = make([0, 1], 2)
    backward_sd(engine, full)
    grad = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    backward_sd(engine, make([0], 2))
    backward_sd(engine, make([1], 2))
    for p, g in zip(model.parameters(), grad, strict=False):
        assert torch.allclose(p.grad, g) and torch.isfinite(g).all()
    assert sum(g.abs().sum() for g in grad) > 0


def test_driver_grouping_after_balance_and_padding(monkeypatch, tmp_path):
    import sys

    class NonTensorStack:
        @staticmethod
        def _from_list(items):
            return SimpleNamespace(tolist=lambda: items)

    from examples.search_r1.sd_search import MASK, HindsightSupervisor

    # Two groups interleaved as they may be after length balancing, plus padding.
    keys = ["a_session_0", "b_session_0", "a_session_1", "b_session_1", "pad_session_0"]
    ids = list(b"<think>x</think><search>query</search>")
    fields = {
        "prompts": torch.nested.as_nested_tensor([torch.tensor([1, 2]) for _ in keys], layout=torch.jagged),
        "responses": torch.nested.as_nested_tensor([torch.tensor(ids) for _ in keys], layout=torch.jagged),
        "response_mask": torch.nested.as_nested_tensor(
            [torch.ones(len(ids), dtype=torch.bool) for _ in keys], layout=torch.jagged
        ),
        "reward_model": NonTensorStack._from_list([{"ground_truth": {"target": ["New York City"]}} for _ in keys]),
        "extra_fields": NonTensorStack._from_list(
            [
                {"search_r1_final_assistant": "<answer>New York</answer>", "search_r1_termination": "answer"}
                for _ in keys
            ]
        ),
    }
    stored = {}
    monkeypatch.setitem(
        sys.modules,
        "transfer_queue",
        SimpleNamespace(kv_batch_get=lambda **kw: fields, kv_batch_put=lambda **kw: stored.update(kw)),
    )
    sup = object.__new__(HindsightSupervisor)
    sup.codec = ByteCodec()
    sup.group_size = 2
    sup.c = SimpleNamespace(
        warmup_steps=50,
        alpha=0.001,
        slots=4,
        top_k=50,
        teacher_micro_batch_size_per_gpu=4,
        max_hindsight_tokens=1024,
        max_context_tokens=8192,
    )
    batch = SimpleNamespace(
        keys=keys, tags=[{}] * 4 + [{"is_padding": True}], partition_id="train", extra_info={}, fields=[]
    )

    class Batch(SimpleNamespace):
        def __len__(self):
            return len(self.keys)

    batch = Batch(**vars(batch))
    metrics = {}
    sup.prepare(SimpleNamespace(global_steps=50), batch, metrics)
    assert not stored
    sup.prepare(SimpleNamespace(global_steps=51), batch, metrics)
    assert metrics["sd/groups"] == 2 and batch.fields is None
    assert stored["fields"][MASK].values().sum() == 4
    assert stored["fields"]["sd_0_weights"].unbind()[4].sum() == 0
    assert all(stored["fields"]["sd_0_weights"].unbind()[i].sum() == pytest.approx(1) for i in range(4))
