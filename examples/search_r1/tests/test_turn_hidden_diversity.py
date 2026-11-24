"""Token provenance, differentiable capture, and DP/accumulation invariance."""

import ast
import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from tensordict import TensorDict
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from examples.search_r1.skill_teachers import SkillTokenCodec
from examples.search_r1.tests.test_skill_teachers import ROOT, encoded
from examples.search_r1.turn_hidden_diversity import (
    PAIR_MASK,
    PREVIOUS,
    capture_decisions,
    distance_sums,
    pair_positions,
    turn_ppo_loss,
    validate_settings,
)
from verl.utils import tensordict_utils as tu


@pytest.fixture(scope="module", name="tokenizer")
def qwen_tokenizer():
    default = ROOT.parent.parent / "sd-search/models/Qwen2.5-3B"
    return AutoTokenizer.from_pretrained(os.environ.get("SEARCH_R1_TOKENIZER", str(default)), local_files_only=True)


def test_positions_are_actual_unshifted_search_endpoints(tokenizer):
    codec = SkillTokenCodec(tokenizer)
    first = "<think>Find café 法国</think>\n<search>Paris</search>"
    second = "<think>Use new evidence</think>\n<search>Paris country</search>"
    third = "<think>Enough evidence</think>\n<answer>France</answer>"
    tool = (
        "<|im_end|>\n<|im_start|>user\n<documents><think>fake</think><search>fake</search>"
        "</documents><|im_end|>\n<|im_start|>assistant\n"
    )
    ids, mask = encoded(tokenizer, [(first, 1), (tool, 0), (second, 1), (tool, 0), (third, 1)])
    pairs, previous = pair_positions(codec, ids, mask)
    assert pairs.sum() == 1
    current = pairs.nonzero().item()
    prior = previous[current].item()
    text = lambda stop: b"".join(codec.token_bytes(i) for i in ids[: stop + 1].tolist()).rstrip()
    assert text(current).endswith(b"<search>") and text(prior).endswith(b"<search>")
    assert prior < current and mask[current] and mask[prior]
    assert not pairs[~mask.bool()].any()


@pytest.mark.parametrize(
    "middle",
    ["<think>unfinished", "<think>bad<search>nested</search></think>", "<think>ok</think><answer>done</answer>"],
)
def test_invalid_or_answer_turn_breaks_adjacency(tokenizer, middle):
    codec = SkillTokenCodec(tokenizer)
    search = "<think>need evidence</think>\n<search>query</search>"
    boundary = "<|im_end|>\n<|im_start|>assistant\n"
    ids, mask = encoded(tokenizer, [(search, 1), (boundary, 0), (middle, 1), (boundary, 0), (search, 1)])
    pairs, _ = pair_positions(codec, ids, mask)
    assert pairs.sum() == 0


def test_cosine_hinge_detach_and_margin():
    current = torch.tensor([[1.0, 0.1], [0.0, 1.0]], requires_grad=True)
    previous = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    values = distance_sums(current, previous, 0.1)
    assert values["active_sum"] == 1
    assert values["loss_sum"].item() == pytest.approx(0.1 - (1 - 1 / (1.01**0.5)), abs=1e-6)
    values["loss_sum"].backward()
    assert previous.grad is None
    assert current.grad[0].abs().sum() > 0 and current.grad[1].abs().sum() == 0
    assert distance_sums(current * 7, previous * 3, 0.1)["loss_sum"].item() == pytest.approx(
        values["loss_sum"].item(), abs=1e-6
    )


def batch():
    def nt(rows):
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    return tu.get_tensordict(
        tensor_dict={
            "prompts": nt([torch.tensor([1, 2]), torch.tensor([1])]),
            PAIR_MASK: nt([torch.tensor([0, 0, 1], dtype=torch.bool), torch.tensor([0, 0], dtype=torch.bool)]),
            PREVIOUS: nt([torch.tensor([-1, -1, 0]), torch.tensor([-1, -1])]),
        },
        non_tensor_dict={
            "turn_hidden_margin": 0.1,
            "turn_hidden_coefficient": 0.001,
            "auxiliary_loss_mask_counts": {PAIR_MASK: 1.0},
            "dp_size": 1,
        },
    )


def test_real_qwen_hook_gets_final_hidden_and_removes_it():
    torch.manual_seed(3)
    config = Qwen2Config(
        vocab_size=24,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = Qwen2ForCausalLM(config).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 0, 0]])
    data = batch()
    with capture_decisions(model, data) as values:
        output = model(ids, attention_mask=(ids != 0), output_hidden_states=True)
    expected = distance_sums(output.hidden_states[-1][0, 4:5], output.hidden_states[-1][0, 2:3], 0.1)
    torch.testing.assert_close(values["loss_sum"], expected["loss_sum"])
    assert values["pairs"] == 1 and not model.lm_head._forward_pre_hooks
    # A larger margin makes this randomly initialized model's loss active.
    tu.assign_non_tensor(data, turn_hidden_margin=2.0)
    with capture_decisions(model, data) as captured:
        model(ids, attention_mask=(ids != 0))
    captured["loss_sum"].backward()
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None
    assert torch.isfinite(model.model.layers[0].self_attn.q_proj.weight.grad).all()
    with pytest.raises(RuntimeError, match="test exception"):
        with capture_decisions(model, data):
            raise RuntimeError("test exception")
    assert not model.lm_head._forward_pre_hooks


def test_empty_pairs_are_finite_and_backward_safe():
    data = batch()
    data[PAIR_MASK].values().zero_()

    class Model:
        def __init__(self):
            self.head = torch.nn.Identity()

        def get_output_embeddings(self):
            return self.head

    model = Model()
    hidden = torch.randn(2, 5, 4, requires_grad=True)
    with capture_decisions(model, data) as values:
        model.head(hidden)
    assert values["pairs"] == 0 and values["loss_sum"].item() == 0
    values["loss_sum"].backward()
    assert hidden.grad.abs().sum() == 0
    with pytest.raises(RuntimeError, match="bypassed"):
        with capture_decisions(model, data):
            pass
    assert not model.head._forward_pre_hooks


def test_microbatch_and_dp_gradient_normalization():
    torch.manual_seed(5)
    current = torch.randn(7, 5, requires_grad=True)
    previous = torch.randn(7, 5)
    full = distance_sums(current, previous, 2.0)["loss_sum"] / 7
    expected = torch.autograd.grad(full, current)[0]
    split = sum(distance_sums(current[a:b], previous[a:b], 2.0)["loss_sum"] / 7 for a, b in [(0, 2), (2, 3), (3, 7)])
    torch.testing.assert_close(torch.autograd.grad(split, current)[0], expected)
    # Each DP rank multiplies by DP size; averaging its gradients recovers full.
    dp = sum(distance_sums(current[a:b], previous[a:b], 2.0)["loss_sum"] * 2 / 7 for a, b in [(0, 3), (3, 7)]) / 2
    torch.testing.assert_close(torch.autograd.grad(dp, current)[0], expected)


def test_wrapper_preserves_existing_loss_and_combines_skill(monkeypatch):
    import examples.search_r1.skill_distillation as skill
    import verl.workers.utils.losses as base

    data = batch()
    hidden = torch.tensor(0.04, requires_grad=True)
    model_output = {
        "turn_hidden_diversity": {
            "loss_sum": hidden,
            "cosine_sum": hidden * 0 + 1,
            "distance_sum": hidden * 0,
            "active_sum": hidden * 0 + 1,
            "pairs": 1,
        }
    }
    monkeypatch.setattr(base, "ppo_loss", lambda *args: (hidden * 0 + 3, {}))
    monkeypatch.setattr(skill, "skill_ppo_loss", lambda *args: (hidden * 0 + 5, {"skill/marker": 1}))
    for weights, expected in [(None, 3), ({"entity": 1.0}, 5)]:
        loss, metrics = turn_ppo_loss(None, weights, model_output, data)
        assert loss.item() == pytest.approx(expected + 0.001 * 0.04, abs=3e-7)
        assert torch.autograd.grad(loss, hidden, retain_graph=True)[0] == pytest.approx(0.001)
        assert metrics["turn_hidden/pairs"].aggregate() == 1
        assert ("skill/marker" in metrics) == (weights is not None)


def test_enabled_config_validation_and_disabled_default():
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "+turn_hidden_diversity=search_r1",
                "actor_rollout_ref.rollout.agent.default_agent_loop=search_r1",
                "actor_rollout_ref.model.use_remove_padding=False",
            ],
        )
    assert not config.turn_hidden_diversity.enabled
    assert config.turn_hidden_diversity.margin == 0.1 and config.turn_hidden_diversity.coefficient == 0.001
    validate_settings(config)
    config.turn_hidden_diversity.margin = float("nan")
    with pytest.raises(ValueError, match="margin"):
        validate_settings(config)


def test_engine_forward_step_uses_capture_only_for_actor_training(monkeypatch):
    # Execute the actual engine method without allocating a distributed GPU engine.
    # This checks its hook lifetime and loss handoff, not a duplicate forward stub.
    path = ROOT / "verl/workers/engine/fsdp/transformer_impl.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FSDPEngineWithLMHead")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_step")
    namespace = {
        "TensorDict": TensorDict,
        "get_device_name": lambda: "cpu",
        "get_device_id": lambda: "cpu",
        "tu": tu,
        "nullcontext": nullcontext,
        "torch": torch,
        "detach_tree": lambda x: x,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(12, 4)
            self.head = torch.nn.Linear(4, 12)

        def get_output_embeddings(self):
            return self.head

        def forward(self, input_ids, use_cache):
            return self.head(self.embed(input_ids))

    class Engine:
        _autocast_dtype = torch.float32

        def __init__(self):
            self.module = Model()

        def prepare_model_inputs(self, micro_batch):
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 0, 0]])}, {}

        def prepare_model_outputs(self, **kwargs):
            return {"base": kwargs["output"].sum()}

        def get_data_parallel_group(self):
            return None

    engine = Engine()
    data = batch()
    tu.assign_non_tensor(data, turn_hidden_enabled=True)

    def loss_function(model_output, **kwargs):
        assert "turn_hidden_diversity" in model_output
        return model_output["base"] + model_output["turn_hidden_diversity"]["loss_sum"], {}

    loss, output = namespace["forward_step"](engine, data, loss_function, False)
    loss.backward()
    assert not engine.module.head._forward_pre_hooks
    for forward_only, enabled in [(True, True), (False, False)]:
        tu.assign_non_tensor(data, turn_hidden_enabled=enabled)

        def base_loss(model_output, **kwargs):
            assert "turn_hidden_diversity" not in model_output
            return model_output["base"], {}

        namespace["forward_step"](engine, data, base_loss, forward_only)
        assert not engine.module.head._forward_pre_hooks


def test_supervisor_transfers_masks_excludes_padding_and_preserves_skill_metadata(tokenizer, monkeypatch):
    import sys

    from examples.search_r1.turn_hidden_diversity import TurnHiddenSupervisor

    text = "<think>first</think>\n<search>one</search>"
    boundary = "<|im_end|>\n<|im_start|>user\nresult<|im_end|>\n<|im_start|>assistant\n"
    ids, mask = encoded(tokenizer, [(text, 1), (boundary, 0), (text, 1)])
    nt = lambda xs: torch.nested.as_nested_tensor(xs, layout=torch.jagged)
    data = TensorDict({"responses": nt([ids, ids]), "response_mask": nt([mask, mask])}, batch_size=[2])
    stored = {}
    monkeypatch.setitem(
        sys.modules,
        "transfer_queue",
        SimpleNamespace(kv_batch_get=lambda **kw: data, kv_batch_put=lambda **kw: stored.update(kw)),
    )

    class Batch:
        keys = ["a", "b"]
        partition_id = "test"
        tags = [{}, {"is_padding": True}]
        fields = ["responses"]
        extra_info = {"auxiliary_loss_mask_names": ["skill_query_mask"]}

        def __len__(self):
            return 2

    batch_meta = Batch()
    supervisor = object.__new__(TurnHiddenSupervisor)
    supervisor.codec = SkillTokenCodec(tokenizer)
    supervisor.config = SimpleNamespace(margin=0.1, coefficient=0.001)
    metrics = {}
    supervisor.prepare(batch_meta, metrics)
    supervisor.update_metadata(batch_meta)
    masks = stored["fields"][PAIR_MASK].unbind()
    assert masks[0].sum() == 1 and masks[1].sum() == 0
    assert metrics["turn_hidden/prepared_pairs"] == 1
    assert batch_meta.fields is None
    assert batch_meta.extra_info["auxiliary_loss_mask_names"] == ["skill_query_mask", PAIR_MASK]
    supervisor.update_metadata(batch_meta)
    assert batch_meta.extra_info["auxiliary_loss_mask_names"].count(PAIR_MASK) == 1


@pytest.mark.parametrize('prefix', ['', 'Reason outside tags.\n', '<think></think>Reason outside.\n', '<think>reason</think>\n'])
def test_search_anchor_covers_missing_think(tokenizer, prefix):
    codec = SkillTokenCodec(tokenizer)
    boundary = '<|im_end|>\n<|im_start|>user\n<search>fake</search><|im_end|>\n<|im_start|>assistant\n'
    ids, mask = encoded(tokenizer, [(prefix+'<search>first</search>', 1), (boundary, 0),
                                    (prefix+'<search>second</search>', 1)])
    pairs, prior, stats = pair_positions(codec, ids, mask, return_stats=True)
    assert pairs.sum() == 1 and stats['valid_searches'] == stats['endpoints'] == 2
    assert stats['eligible_pairs'] == stats['captured_pairs'] == 1
    for index in [pairs.nonzero().item(), prior[pairs].item()]:
        text = b''.join(codec.token_bytes(i) for i in ids[:index+1].tolist())
        assert text.endswith(b'<search>')


def test_gradient_geometry_handles_conflict_and_zero():
    from examples.search_r1.hidden_gradient_diagnostics import gradient_geometry, geometry_metrics
    a=[torch.tensor([3.,4.]),None]
    b=[torch.tensor([-6.,-8.]), torch.tensor([0.])]
    m=geometry_metrics(*gradient_geometry(a,b))
    assert m['grpo_norm']==5 and m['weighted_hidden_norm']==10
    assert m['hidden_to_grpo_ratio']==2 and m['angle_degrees']==180
    z=geometry_metrics(1,0,0)
    assert z['angle_valid']==0 and z['weighted_hidden_norm']==0


def test_search_anchor_counts_merged_query_token():
    class Codec:
        special_ids=set()
        def token_bytes(self, token):
            return {0:b'<search>first',1:b'</search>'}[token]
    _,_,stats=pair_positions(Codec(),torch.tensor([0,1]),torch.ones(2),return_stats=True)
    assert stats['endpoints']==1 and stats['boundary_tokens_with_query']==1


def test_gradient_probe_clears_grads_and_preserves_parameters(monkeypatch):
    from examples.search_r1 import hidden_gradient_diagnostics as d
    from contextlib import nullcontext
    model=torch.nn.Linear(2,1,bias=False)
    before=model.weight.detach().clone()
    optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    class Engine:
        module=model
        scaler=None
        def optimizer_zero_grad(self): optimizer.zero_grad(set_to_none=True)
        def get_data_parallel_group(self): return None
        def forward_backward_batch(self,data,loss):
            assert self._turn_gradient_probe
            # Accumulate an entire miniature batch, including cancellation.
            for x in [torch.tensor([[1.,2.]]),torch.tensor([[3.,4.]])]:
                (model(x).sum()*(1 if loss.args[0]=='grpo' else -.001)).backward()
    monkeypatch.setattr(torch.random,'fork_rng',lambda **kw:nullcontext())
    monkeypatch.setattr(torch.cuda,'current_device',lambda:torch.device('cpu'))
    monkeypatch.setattr(torch.cuda,'get_rng_state',torch.get_rng_state)
    monkeypatch.setattr(torch.cuda,'set_rng_state',torch.set_rng_state)
    monkeypatch.setattr(torch.distributed,'all_reduce',lambda *a,**k:None)
    engine=Engine()
    metrics=d.measure_gradients(engine,None,SimpleNamespace(keywords={'config':None}))
    assert metrics['turn_hidden/gradient/grpo_norm']==pytest.approx(52**.5)
    assert metrics['turn_hidden/gradient/hidden_to_grpo_ratio']==pytest.approx(.001)
    assert metrics['turn_hidden/gradient/angle_degrees']==pytest.approx(180,abs=.001)
    assert model.weight.grad is None and not engine._turn_gradient_probe
    torch.testing.assert_close(model.weight,before)
