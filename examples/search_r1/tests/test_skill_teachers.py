"""Token alignment, causal rescoring and gradients for the skill teacher recipe."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tensordict import TensorDict
from transformers import AutoTokenizer

from examples.search_r1.skill_distillation import SkillTeacherSupervisor, skill_ppo_loss
from examples.search_r1.skill_teachers import (
    TEACHERS,
    SkillTokenCodec,
    beta_at_step,
    byte_level_decoder,
    clipped_skill_loss,
    load_bank,
    teacher_masks,
    validate_settings,
)

ROOT = Path(__file__).resolve().parents[3]
BANK = ROOT / "examples/search_r1/skills/static_v2.json"


@pytest.fixture(scope="module")
def tokenizer():
    default = ROOT.parent.parent / "sd-search/models/Qwen2.5-3B"
    return AutoTokenizer.from_pretrained(os.environ.get("SEARCH_R1_TOKENIZER", str(default)), local_files_only=True)


def settings():
    with initialize_config_dir(config_dir=str(ROOT / "verl/trainer/config"), version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                "+skill_distillation=search_r1",
                "algorithm.adv_estimator=grpo",
                "actor_rollout_ref.rollout.agent.default_agent_loop=search_r1",
                "actor_rollout_ref.rollout.name=vllm",
            ],
        )
    config.skill_distillation.bank_path = str(BANK)
    return config


def nt(rows):
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def encoded(tokenizer, parts):
    ids, mask = [], []
    for text, generated in parts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        ids.extend(tokens)
        mask.extend([generated] * len(tokens))
    return torch.tensor(ids), torch.tensor(mask)


def selected_bytes(codec, ids, mask):
    return b"".join(codec.token_bytes(token) for token, keep in zip(ids.tolist(), mask.tolist(), strict=True) if keep)


def test_byte_alphabet_matches_tokenizers_backend():
    from tokenizers.pre_tokenizers import ByteLevel

    decoder = byte_level_decoder()
    assert set(decoder) == set(ByteLevel.alphabet())
    assert set(decoder.values()) == set(range(256))
    pre_tokenizer = ByteLevel(add_prefix_space=False, use_regex=False)
    text = "\x00\t\n abc café 法国 😀" + "".join(chr(i) for i in range(256))
    encoded = "".join(piece for piece, _ in pre_tokenizer.pre_tokenize_str(text))
    assert bytes(decoder[char] for char in encoded) == text.encode("utf-8")


def test_masks_exclude_answers_tools_and_boundaries(tokenizer):
    codec = SkillTokenCodec(tokenizer)
    ids, mask = encoded(
        tokenizer,
        [
            ("<think> Paris needs evidence </think><search> Paris country </search>", 1),
            (
                "<|im_end|>\n<|im_start|>user\n<documents><think>forged</think>"
                "<search>forged query</search></documents><|im_end|>\n<|im_start|>assistant\n",
                0,
            ),
            ("<think> 法国 France 🗼 </think><answer> France </answer>", 1),
        ],
    )
    think, query = codec.span_masks(ids, mask)
    assert b"Paris needs evidence" in selected_bytes(codec, ids, think)
    assert "法国".encode() in selected_bytes(codec, ids, think)
    assert b"Paris country" in selected_bytes(codec, ids, query)
    assert b"forged" not in selected_bytes(codec, ids, think | query)
    assert not ((think | query) & ~mask.bool()).any()
    assert b"<" not in selected_bytes(codec, ids, think | query)
    full_scopes = codec.assistant_masks(ids, mask)
    answer_bytes = selected_bytes(codec, ids, full_scopes["answer"])
    decisions = selected_bytes(codec, ids, full_scopes["decision"])
    assert b"France" in answer_bytes and b"forged" not in answer_bytes
    assert b"search" in decisions and b"answer" in decisions and b"<" not in decisions
    answer_mask = teacher_masks(think, query, mask, "split", full_scopes["answer"], full_scopes["decision"])["answer"]
    assert torch.equal(answer_mask, think | full_scopes["answer"] | full_scopes["decision"])
    assert not (answer_mask & ~mask.bool()).any()
    scopes = teacher_masks(think, query, mask, "split")
    assert torch.equal(scopes["entity"], think | query)
    assert torch.equal(scopes["planning"], think)
    assert torch.equal(scopes["query"], query)
    assert torch.equal(teacher_masks(think, query, mask, "combined")["combined"], think | query)
    assert all(torch.equal(m, mask.bool()) for m in teacher_masks(think, query, mask, "all").values())


@pytest.mark.parametrize(
    "text,scope",
    [
        ("<think> unfinished reasoning", "think"),
        ("<search> unfinished query", "query"),
        ("<answer> unfinished answer", None),
        ("unstructured prose", None),
        ("<think> nested <search> unsafe </search></think>", None),
        ("<think> wrong close </search>", None),
    ],
)
def test_truncated_and_malformed_spans(tokenizer, text, scope):
    codec = SkillTokenCodec(tokenizer)
    ids = tokenizer.encode(text, add_special_tokens=False)
    think, query = codec.span_masks(ids, [1] * len(ids))
    assert bool(think.any()) == (scope == "think")
    assert bool(query.any()) == (scope == "query")


def test_noncanonical_generated_bpe_is_not_retokenized(tokenizer):
    codec = SkillTokenCodec(tokenizer)
    # Separately generated pieces need not be the tokenizer's preferred encoding.
    pieces = ["<think>", " New", " York", "</think>", "<search>", "a", "b", "c", "</search>"]
    ids = sum([tokenizer.encode(piece, add_special_tokens=False) for piece in pieces], [])
    assert ids != tokenizer.encode("".join(pieces), add_special_tokens=False)
    think, query = codec.span_masks(ids, [1] * len(ids))
    assert selected_bytes(codec, torch.tensor(ids), query) == b"abc"
    assert selected_bytes(codec, torch.tensor(ids), think) == b" New York"


@pytest.mark.parametrize("has_system", [True, False])
def test_teacher_preserves_original_prompt_and_response_tokens(tokenizer, has_system):
    codec = SkillTokenCodec(tokenizer)
    messages = [{"role": "system", "content": "Original exact instructions."}] if has_system else []
    messages += [{"role": "user", "content": "Who wrote this book?"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    original = prompt.copy()
    _, contexts, _ = load_bank(BANK)
    for name in TEACHERS:
        teacher = codec.teacher_prompt(prompt, contexts[name])
        added = len(teacher) - len(prompt)
        if prompt[: len(codec.system_header)] == codec.system_header:
            cut = len(codec.system_header)
            assert teacher[:cut] + teacher[cut + added :] == prompt
        else:
            assert teacher[added:] == prompt
        assert tokenizer.decode(teacher).count("<|im_start|>system\n") == 1
        assert "Do not write a critique" in tokenizer.decode(teacher)
    assert prompt == original


def test_skill_surrogate_sign_clipping_stop_gradient_and_empty_mask():
    old = torch.zeros(5, requires_grad=True)
    new = torch.tensor([0.0, 0.0, 0.8, -0.8, 50.0], requires_grad=True)
    advantage = torch.tensor([2.0, -3.0, 2.0, -3.0, float("nan")], requires_grad=True)
    mask = torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool)
    loss = clipped_skill_loss(new, old, advantage, mask, 4, 1, 0.2, 0.2)
    loss.backward()
    assert torch.allclose(new.grad, torch.tensor([-0.5, 0.75, 0, 0, 0]))
    assert old.grad is None and advantage.grad is None
    empty = clipped_skill_loss(new, old, advantage, mask & False, 0, 1, 0.2, 0.2)
    assert empty == 0 and torch.isfinite(empty)


def test_skill_normalization_invariant_to_microbatches_and_dp():
    torch.manual_seed(7)
    old = torch.randn(4, 7)
    delta = torch.randn(4, 7) * 0.3
    advantages = torch.randn(4, 7)
    mask = torch.rand(4, 7) > 0.6
    count = mask.sum().item()
    whole = (old + delta).requires_grad_()
    reference = clipped_skill_loss(whole, old, advantages, mask, count, 1, 0.2, 0.2)
    reference.backward()
    split = (old + delta).requires_grad_()
    # Two DP ranks with two accumulated micro-batches each; DDP averages ranks.
    accumulated = (
        sum(
            clipped_skill_loss(
                split[i : i + 1], old[i : i + 1], advantages[i : i + 1], mask[i : i + 1], count, 2, 0.2, 0.2
            )
            for i in range(4)
        )
        / 2
    )
    accumulated.backward()
    torch.testing.assert_close(accumulated, reference)
    torch.testing.assert_close(split.grad, whole.grad)


def test_worker_counts_each_scope_over_current_minibatch(monkeypatch):
    from verl.workers.utils.losses import auxiliary_mask_counts

    names = ["entity", "planning", "query"]
    data = {
        "entity": nt([torch.tensor([1, 1, 0]), torch.tensor([1, 0])]),
        "planning": nt([torch.tensor([1, 0, 0]), torch.tensor([0, 0])]),
        "query": nt([torch.tensor([0, 1, 0]), torch.tensor([1, 0])]),
    }
    group = object()
    calls = []

    def reduce(counts, group):
        calls.append(group)
        # Counts from another DP rank in the same optimizer mini-batch.
        counts += torch.tensor([4, 3, 1])

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    assert auxiliary_mask_counts(data, names, "cpu", group) == {"entity": 7, "planning": 4, "query": 3}
    assert calls == [group]
    # A later mini-batch must recount its masks rather than reuse rollout totals.
    empty = {name: torch.zeros(2, 3, dtype=torch.bool) for name in names}
    assert auxiliary_mask_counts(empty, names, "cpu", group) == {"entity": 4, "planning": 3, "query": 1}


def test_composed_config_and_bank():
    config = settings()
    validate_settings(config)
    bank, contexts, digest = load_bank(BANK)
    assert len(digest) == 64 and bank["version"] == "static-v2"
    assert set(contexts) == {*TEACHERS, "combined"}
    assert beta_at_step(0.01, 10, 1) == 0.001
    assert beta_at_step(0.01, 10, 100) == 0.01
    assert beta_at_step(0.01, 0, 1) == 0.01


def test_manifest_loss_installation_and_resume_schedule(tokenizer, tmp_path):
    config = settings()
    config.trainer.default_local_dir = str(tmp_path)
    config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
    installed = []
    trainer = SimpleNamespace(
        config=config,
        tokenizer=tokenizer,
        actor_rollout_wg=SimpleNamespace(set_loss_fn=installed.append),
    )
    supervisor = SkillTeacherSupervisor(config)
    supervisor.initialize(trainer)
    manifest = json.loads((tmp_path / "skill-teachers.json").read_text())
    assert manifest["bank_sha256"] == supervisor.bank_sha256
    assert installed[-1].func is skill_ppo_loss
    assert installed[-1].keywords["weights"] == {name: 1.0 for name in TEACHERS}
    # Restored training step sets the ramp; no additional teacher state is needed.
    resumed = SkillTeacherSupervisor(config)
    resumed.initialize(trainer)
    batch = Batch(["real"], "train", [{}])
    resumed.update_metadata(batch, 8)
    assert batch.extra_info["skill_beta"] == pytest.approx(0.008)
    config.skill_distillation.weights.entity = 2.0
    with pytest.raises(ValueError, match="different skill bank/objective"):
        SkillTeacherSupervisor(config).initialize(trainer)
    assert len(installed) == 2


@pytest.mark.parametrize(
    "path,value",
    [
        ("trainer.v1.trainer_mode", "colocate_async"),
        ("trainer.v1.sync.parameter_sync_step", 2),
        ("actor_rollout_ref.rollout.temperature", 0.6),
        ("distillation.enabled", True),
        ("algorithm.rollout_correction.bypass_mode", True),
        ("skill_distillation.beta", -0.1),
        ("skill_distillation.beta", float("nan")),
        ("skill_distillation.mode", "bad"),
        ("skill_distillation.top_k", 0),
        ("skill_distillation.min_topk_overlap", 1.1),
        ("skill_distillation.prefix_fraction", 0),
        ("skill_distillation.answer_component_weights.think", -1),
        ("sd_search.enabled", True),
        ("turn_hidden_diversity.enabled", True),
    ],
)
def test_unsupported_config_rejected(path, value):
    config = settings()
    OmegaConf.update(config, path, value, force_add=True)
    with pytest.raises(ValueError):
        validate_settings(config)


class Batch:
    def __init__(self, keys, partition_id, tags):
        self.keys, self.partition_id, self.tags = keys, partition_id, tags
        self.extra_info, self.fields = {}, None

    def __len__(self):
        return len(self.keys)


class MemoryQueue:
    def __init__(self, source):
        self.partitions = {"train": source}
        self.cleared = []

    def kv_batch_get(self, keys, partition_id, select_fields):
        return self.partitions[partition_id].select(*select_fields)

    def kv_batch_put(self, keys, partition_id, fields, tags=None):
        if partition_id in self.partitions:
            self.partitions[partition_id].update(fields)
        else:
            self.partitions[partition_id] = fields
        return Batch(keys, partition_id, tags or [{} for _ in keys])

    def kv_clear(self, keys, partition_id):
        self.cleared.append(partition_id)
        del self.partitions[partition_id]


def causal_scores(ids):
    """Tiny deterministic causal LM, predicting the next token modulo 11."""
    vocabulary = torch.arange(11).float()
    state = torch.sin((ids % 11).float().cumsum(0) / 7)
    logits = state[:, None] * vocabulary[None, :] / 11
    return logits.log_softmax(-1).gather(-1, ids.roll(-1).remainder(11)[:, None]).squeeze(-1)


def scoring_fixture(tokenizer, monkeypatch, mode="split", fail=False):
    config = settings()
    config.skill_distillation.mode = mode
    supervisor = SkillTeacherSupervisor(config)
    supervisor.codec = SkillTokenCodec(tokenizer)
    prompt = torch.tensor(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Which country is Paris in?"}], tokenize=True, add_generation_prompt=True
        )
    )
    response, mask = encoded(
        tokenizer,
        [
            ("<think> Need country evidence </think><search> Paris country </search>", 1),
            ("<|im_end|>\n<|im_start|>user\n<documents>France</documents><|im_end|>\n<|im_start|>assistant\n", 0),
            ("<think> Evidence found </think><answer> France </answer>", 1),
        ],
    )
    # A synthetic padding row must not be parsed as ChatML or contribute a loss.
    prompts, responses, masks = [prompt, torch.tensor([0, 0])], [response, torch.tensor([0])], [mask, torch.tensor([0])]
    config.skill_distillation.top_k = 4
    config.skill_distillation.min_topk_overlap = 0.0
    old_probs = [causal_scores(torch.cat((p, r)))[len(p) - 1 : -1] for p, r in zip(prompts, responses, strict=True)]
    source = TensorDict(
        {
            "prompts": nt(prompts),
            "responses": nt(responses),
            "response_mask": nt(masks),
            "old_log_probs": nt(old_probs),
            "advantages": nt([torch.ones_like(m).float() for m in masks]),
        },
        batch_size=[2],
    )
    queue = MemoryQueue(source)
    monkeypatch.setitem(sys.modules, "transfer_queue", queue)
    calls = []

    def score(batch):
        td = queue.partitions[batch.partition_id]
        calls.append(td.clone())
        assert batch.extra_info["compute_loss"] is False
        assert batch.extra_info["calculate_entropy"] is False
        assert td["responses"].unbind()[0].tolist() == response.tolist()
        full = td["input_ids"].unbind()[0]
        start = len(td["prompts"].unbind()[0])
        assert full[start:].tolist() == response.tolist()
        # Suffix changes cannot affect preceding logits; shift is verified below.
        altered = full.clone()
        altered[-3:] = 2
        torch.testing.assert_close(causal_scores(full)[:-4], causal_scores(altered)[:-4])
        if fail:
            raise RuntimeError("injected scoring failure")
        from examples.search_r1.skill_distributions import topk_distribution

        outputs = []
        for row in td["input_ids"].unbind():
            state = torch.sin((row % 11).float().cumsum(0) / 7)
            logits = state[:, None] * torch.arange(11)[None, :].float() / 11
            outputs.append(topk_distribution(logits, batch.extra_info["skill_collect_topk"]))
        for j, key in enumerate(["skill_topk_ids", "skill_topk_logps", "skill_tail_logp"]):
            td[key] = nt([row[j] for row in outputs])
        return batch

    trainer = SimpleNamespace(global_steps=3, actor_rollout_wg=SimpleNamespace(compute_log_prob=score))
    batch = Batch(["real", "padding"], "train", [{}, {"is_padding": True}])
    return supervisor, trainer, batch, queue, calls, source


@pytest.mark.parametrize("mode,teacher_count", [("split", 4), ("combined", 1), ("all", 4)])
def test_teacher_driver_alignment_masks_cleanup_and_frozen_targets(tokenizer, monkeypatch, mode, teacher_count):
    supervisor, trainer, batch, queue, calls, source = scoring_fixture(tokenizer, monkeypatch, mode)
    original = source.clone()
    metrics = {}
    assert supervisor.score(trainer, batch, metrics) is batch
    assert len(calls) == len(queue.cleared) == teacher_count + 1
    assert set(queue.partitions) == {"train"}
    for key in ("prompts", "responses", "old_log_probs", "advantages"):
        for before, after in zip(original[key].unbind(), source[key].unbind(), strict=True):
            torch.testing.assert_close(before, after)
    for name, scored in zip(supervisor.names, calls[1:], strict=True):
        mask = source[f"skill_{name}_mask"].unbind()[0]
        actual = source[f"skill_{name}_logps"].unbind()[0]
        ids = source[f"skill_{name}_ids"].unbind()[0]
        from examples.search_r1.skill_distributions import topk_distribution

        full = scored["input_ids"].unbind()[0]
        state = torch.sin((full % 11).float().cumsum(0) / 7)
        logits = state[:, None] * torch.arange(11)[None, :].float() / 11
        expected_ids, expected, _ = topk_distribution(logits, 4)
        plen = len(scored["prompts"].unbind()[0])
        torch.testing.assert_close(actual, expected[plen - 1 : -1])
        torch.testing.assert_close(ids, expected_ids[plen - 1 : -1])
        assert not actual.requires_grad
        assert not source[f"skill_{name}_mask"].unbind()[1].any()
        assert metrics[f"skill/{name}/tokens"] == int(mask.sum())
    supervisor.update_metadata(batch, 3)
    assert batch.extra_info["skill_beta"] == pytest.approx(0.003)
    assert len(batch.extra_info["auxiliary_loss_mask_names"]) == teacher_count + (3 if mode == "split" else 0)


def test_teacher_failure_cleanup_and_no_partial_student_write(tokenizer, monkeypatch):
    supervisor, trainer, batch, queue, calls, source = scoring_fixture(tokenizer, monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="injected"):
        supervisor.score(trainer, batch, {})
    assert len(queue.cleared) == 1 and set(queue.partitions) == {"train"}
    assert "skill_entity_logps" not in source


def test_teacher_context_overflow_fails_without_truncation(tokenizer, monkeypatch):
    supervisor, trainer, batch, queue, calls, source = scoring_fixture(tokenizer, monkeypatch)
    supervisor.config.max_teacher_context_length = 10
    with pytest.raises(ValueError, match="no truncation"):
        supervisor.score(trainer, batch, {})
    assert not calls and set(queue.partitions) == {"train"}


def test_thinking_divergence_stops_entity_before_query_and_answer_independently(tokenizer, monkeypatch):
    supervisor, trainer, batch, queue, calls, source = scoring_fixture(tokenizer, monkeypatch)
    supervisor.config.min_topk_overlap = 0.5
    original_score = trainer.actor_rollout_wg.compute_log_prob
    frozen = []

    def controlled_score(temporary):
        result = original_score(temporary)
        td = queue.partitions[temporary.partition_id]
        plen = len(td["prompts"].unbind()[0])
        rows = [row.clone() for row in td["skill_topk_ids"].unbind()]
        rlen = len(td["responses"].unbind()[0])
        if not frozen:
            frozen.append(rows[0][plen - 1 : plen + rlen - 1].clone())
        else:
            # All teachers agree with the frozen student's support, except one
            # entity token inside the first think. Later tokens agree again.
            rows[0][plen - 1 : plen + rlen - 1] = frozen[0]
            if len(calls) == 2:
                scope = supervisor.codec.assistant_masks(td["responses"].unbind()[0], td["response_mask"].unbind()[0])
                index = int(scope["think"].nonzero()[1])
                excluded = set(frozen[0][index].tolist())
                replacement = [i for i in range(11) if i not in excluded][:4]
                rows[0][plen - 1 + index] = torch.tensor(replacement)
        td["skill_topk_ids"] = nt(rows)
        return result

    trainer.actor_rollout_wg.compute_log_prob = controlled_score
    metrics = {}
    supervisor.score(trainer, batch, metrics)
    scope = supervisor.codec.assistant_masks(source["responses"].unbind()[0], source["response_mask"].unbind()[0])
    entity = source["skill_entity_mask"].unbind()[0]
    assert entity.any() and not entity[scope["query"]].any()
    stop = int(metrics["skill/entity/first_stop_response_index_mean"])
    assert not entity[stop:].any()
    assert source["skill_query_mask"].unbind()[0][scope["query"]].all()
    assert source["skill_answer_mask"].unbind()[0][scope["answer"]].all()
    assert metrics["skill/entity/stopped_trajectories"] == 1
    assert metrics["skill/answer/stopped_trajectories"] == 0


def test_actual_combined_loss_preserves_grpo_and_has_no_answer_tool_auxiliary_gradient():
    from verl.utils import tensordict_utils as tu
    from verl.workers.config import ActorConfig
    from verl.workers.utils.losses import ppo_loss

    config = ActorConfig(strategy="fsdp", rollout_n=2, use_dynamic_bsz=True, loss_agg_mode="token-mean")
    response_mask = torch.tensor([1, 1, 0, 1, 1])  # think, query, document, answer, end
    data = TensorDict(
        {
            "prompts": nt([torch.tensor([1, 2])]),
            "responses": nt([torch.tensor([3, 4, 5, 6, 7])]),
            # Already-padded response columns also exercise the real loss on
            # the project's Torch 2.4 CPU environment (jagged to_padded arrived
            # later). Model outputs/prompts still use the unpadded engine layout.
            "response_mask": response_mask.unsqueeze(0),
            "old_log_probs": torch.full((1, 5), -1.0),
            "advantages": torch.ones(1, 5),
            "skill_entity_mask": torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(
        data,
        dp_size=1,
        batch_num_tokens=4,
        global_batch_size=1,
        skill_beta=0.1,
        auxiliary_loss_mask_counts={"skill_entity_mask": 2},
    )
    logits = torch.full((7,), -1.0, requires_grad=True)
    kl_terms = torch.arange(7).float().requires_grad_()
    output = {"log_probs": nt([logits]), "skill_entity_forward_kl": nt([kl_terms])}
    base, _ = ppo_loss(config, output, data)
    combined, metrics = skill_ppo_loss(config, {"entity": 1.0}, output, data)
    base_gradient = torch.autograd.grad(base, logits, retain_graph=True)[0]
    full_gradient = torch.autograd.grad(combined, logits, retain_graph=True)[0]
    delta = full_gradient - base_gradient
    torch.testing.assert_close(delta, torch.zeros(7))
    torch.testing.assert_close(
        torch.autograd.grad(combined, kl_terms, retain_graph=True)[0],
        torch.tensor([0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0]),
    )
    assert "skill/entity/loss" in metrics
    tu.assign_non_tensor(data, skill_beta=0.0)
    disabled, _ = skill_ppo_loss(config, {"entity": 1.0}, output, data)
    torch.testing.assert_close(disabled, base)
    torch.testing.assert_close(torch.autograd.grad(disabled, logits)[0], base_gradient)


def test_answer_component_gradients_use_separate_global_mask_counts(monkeypatch):
    import verl.workers.utils.losses as base
    from verl.utils import tensordict_utils as tu

    values = torch.ones(5, requires_grad=True)
    monkeypatch.setattr(base, "ppo_loss", lambda *args: (values.sum() * 0, {}))
    data = TensorDict(
        {
            "prompts": nt([torch.tensor([1])]),
            "responses": nt([torch.arange(4)]),
            "skill_answer_think_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.bool),
            "skill_answer_decision_mask": torch.tensor([[0, 0, 1, 0]], dtype=torch.bool),
            "skill_answer_content_mask": torch.tensor([[0, 0, 0, 1]], dtype=torch.bool),
        },
        batch_size=[1],
    )
    tu.assign_non_tensor(
        data,
        dp_size=2,
        skill_beta=1.0,
        auxiliary_loss_mask_counts={
            "skill_answer_think_mask": 4,
            "skill_answer_decision_mask": 2,
            "skill_answer_content_mask": 2,
        },
    )
    loss, _ = skill_ppo_loss(
        None,
        {"answer": 1.0},
        {"skill_answer_forward_kl": nt([values])},
        data,
        answer_components={"think": 0.1, "decision": 1.0, "content": 1.0},
    )
    loss.backward()
    torch.testing.assert_close(values.grad, torch.tensor([0.05, 0.05, 1.0, 1.0, 0.0]))
