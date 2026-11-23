"""Gate lifetime, normalized forward KL, gradients and engine causal layout."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from examples.search_r1.skill_distributions import (
    forward_kl_topk_tail,
    overlap_prefix_gate,
    skill_model_outputs,
    topk_distribution,
)
from verl.utils import tensordict_utils as tu


def nt(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def test_gate_is_strict_and_permanent_across_tools_and_turns():
    student = torch.tensor([[0, 1, 2, 3]]).repeat(6, 1)
    teacher = torch.tensor([[0, 1, 8, 9], [8, 9, 10, 11], [0, 8, 9, 10], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]])
    # Exactly .5 is accepted. Tool divergence at index 1 is ignored. Thinking
    # divergence at index 2 excludes later query/answer even after full recovery.
    prefix, overlap, stop = overlap_prefix_gate(student, teacher, torch.tensor([1, 0, 1, 0, 1, 1]), 0.5)
    assert stop == 2 and overlap[0] == 0.5
    assert prefix.tolist() == [True, True, False, False, False, False]
    other, _, stop = overlap_prefix_gate(student, student, torch.ones(6), 0.5)
    assert other.all() and stop == -1  # one teacher cannot stop another
    padding, _, stop = overlap_prefix_gate(student, teacher, torch.zeros(6), 0.5)
    assert padding.all() and stop == -1


def test_forward_kl_matches_reset_distribution_and_exact_gradient():
    teacher_logits = torch.tensor([[3.0, 1.0, 0.0, -1.0, -2.0]], requires_grad=True)
    student_logits = torch.tensor([[-1.0, 2.0, 1.0, 0.0, -2.0]], requires_grad=True)
    ids, lp, tail = topk_distribution(teacher_logits, 2)
    assert not lp.requires_grad and not tail.requires_grad
    torch.testing.assert_close(lp.exp().sum(-1) + tail.exp(), torch.ones(1))
    actual = forward_kl_topk_tail(student_logits, ids, lp, tail)
    q = student_logits.softmax(-1)
    p = teacher_logits.detach().softmax(-1)
    p_bins = torch.cat([p[:, :2], p[:, 2:].sum(-1, keepdim=True)], -1)
    q_bins = torch.cat([q[:, :2], q[:, 2:].sum(-1, keepdim=True)], -1)
    expected = (p_bins * (p_bins.log() - q_bins.log())).sum(-1)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.sum(), student_logits, retain_graph=True)[0]
    ge = torch.autograd.grad(expected.sum(), student_logits)[0]
    torch.testing.assert_close(ga, ge)
    assert ga[0, 0] < 0  # cover teacher mode even though student assigned low mass
    assert ga[0, 2:].abs().sum() > 0  # tail receives a genuine gradient
    assert teacher_logits.grad is None
    # Reverse KL on the same bins is a different scalar.
    reverse = (q_bins.detach() * (q_bins.detach().log() - p_bins.log())).sum(-1)
    assert not torch.allclose(actual.detach(), reverse)


def test_tail_is_stable_for_saturated_distributions():
    p = torch.tensor([[100.0, 99.0, -100.0, -101.0]])
    q = torch.tensor([[99.0, 100.0, -101.0, -100.0]], requires_grad=True)
    ids, lp, tail = topk_distribution(p, 2)
    assert torch.isfinite(tail).all()  # 1-sum(topk) would round to zero
    value = forward_kl_topk_tail(q, ids, lp, tail)
    value.sum().backward()
    assert torch.isfinite(value).all() and torch.isfinite(q.grad).all()


def packed_fixture():
    torch.manual_seed(7)
    prompts = [torch.tensor([1, 2]), torch.tensor([3, 4, 5])]
    responses = [torch.tensor([3, 4, 5]), torch.tensor([6, 7])]
    logits = torch.randn(10, 9, requires_grad=True)
    data = TensorDict(
        {
            "input_ids": nt([torch.cat([p, r]) for p, r in zip(prompts, responses, strict=False)]),
            "prompts": nt(prompts),
            "responses": nt(responses),
            "skill_answer_mask": nt([torch.tensor([1, 0, 1]), torch.tensor([0, 1])]),
        },
        batch_size=[2],
    )
    teacher = torch.randn(5, 9)
    ids, lp, tail = topk_distribution(teacher, 3)
    data["skill_answer_ids"] = nt([ids[:3], ids[3:]])
    data["skill_answer_logps"] = nt([lp[:3], lp[3:]])
    data["skill_answer_tail"] = nt([tail[:3], tail[3:]])
    tu.assign_non_tensor(data, skill_forward_kl_names=("answer",))
    return logits, data


def test_packed_targets_use_predictor_positions_and_ignore_masked_tokens():
    logits, data = packed_fixture()
    value = skill_model_outputs(logits, data)["skill_answer_forward_kl"]
    # Response predictors: [1,2,3] and [7,8]; masked -> [1,3,8].
    expected = torch.tensor([1, 3, 8])
    assert value.nonzero().flatten().tolist() == expected.tolist()
    loss = value.sum()
    loss.backward()
    assert logits.grad.abs().sum(-1).nonzero().flatten().tolist() == expected.tolist()
    for key in ["skill_answer_logps", "skill_answer_tail"]:
        assert not data[key].requires_grad
    logits, data = packed_fixture()
    data["skill_answer_mask"] = nt([torch.zeros(3), torch.zeros(2)])
    skill_model_outputs(logits, data)["skill_answer_forward_kl"].sum().backward()
    assert not logits.grad.any()


def engine_method():
    # Load the actual method AST so CPU tests do not import unavailable GPU
    # FSDP/flash-attn dependencies in the local Torch 2.4 environment.
    path = Path(__file__).resolve().parents[3] / "verl/workers/engine/fsdp/transformer_impl.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FSDPEngineWithLMHead")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "prepare_model_outputs")
    method.decorator_list = []
    from verl.utils.dataset.dataset_utils import DatasetPadMode

    def logprobs(logits, labels, **kwargs):
        return logits.log_softmax(-1).gather(-1, labels[:, None]).squeeze(-1)

    ns = {
        "torch": torch,
        "tu": tu,
        "TensorDict": TensorDict,
        "DatasetPadMode": DatasetPadMode,
        "DTensor": type("NotDTensor", (), {}),
        "logprobs_from_logits": logprobs,
        "_scale_logits_by_temperature": lambda logits, *a, **kw: logits,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), ns)
    return ns["prepare_model_outputs"]


@pytest.mark.parametrize("remove_padding", [True, False])
@pytest.mark.parametrize("collect", [True, False])
def test_real_engine_hook_in_both_padding_paths(remove_padding, collect):
    from verl.utils.dataset.dataset_utils import DatasetPadMode

    logits, data = packed_fixture()
    kwargs = {
        "pad_size": 0,
        "temperature_is_one": True,
        "temperature_rmpad": torch.ones(10),
        "temperature": torch.ones(2),
        "input_ids_rmpad_rolled": torch.ones(10, dtype=torch.long),
    }
    output_logits = logits.unsqueeze(0) if remove_padding else logits.reshape(2, 5, 9)
    tu.assign_non_tensor(
        data,
        use_remove_padding=remove_padding,
        pad_mode=DatasetPadMode.NO_PADDING,
        use_fused_kernels=False,
        calculate_entropy=False,
        distillation_use_topk=False,
        skill_collect_topk=3 if collect else 0,
        skill_forward_kl_names=() if collect else ("answer",),
    )
    eng = SimpleNamespace(
        use_ulysses_sp=False,
        engine_config=SimpleNamespace(entropy_checkpointing=False),
        _gather_and_unpad_packed=lambda t, pad: t[:-pad] if pad else t,
    )
    output = engine_method()(eng, SimpleNamespace(logits=output_logits), kwargs, data, None)
    if collect:
        assert output["skill_topk_ids"].values().shape == (10, 3)
        assert output["skill_topk_logps"].values().shape == (10, 3)
    else:
        actual = output["skill_answer_forward_kl"].values()
        torch.testing.assert_close(actual, skill_model_outputs(logits, data)["skill_answer_forward_kl"])
        (actual.sum() + output["log_probs"].values().sum()).backward()
        assert torch.isfinite(logits.grad).all()
