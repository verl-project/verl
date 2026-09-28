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
"""LOCAL policy loss through the FSDP engine's input/output preparation and ``ppo_loss`` on CPU.

The engine is a bare ``FSDPEngineWithLMHead`` instance and the model output is a random logits tensor, so
the test covers the packing, the static pad, the response alignment and the loss, but not Ulysses SP.
Both batch layouts are covered: padded prompts/responses with an attention mask (legacy trainer) and
nested prompts/responses (V1 trainer).
"""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.trainer.ppo.local_logit_regression import (
    CENTERED_LOGITS_KEY,
    CURRENT_FEATURES_FLAG,
    OLD_CENTERED_LOGITS_KEY,
    OLD_FEATURE_KEYS,
    OLD_FEATURES_FLAG,
    OLD_TOPK_IDS_KEY,
    compute_old_topk_features,
)
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.workers.config.actor import ActorConfig, PolicyLossConfig
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding, response_to_packed_sequence

VOCAB, TOPK, ETA = 13, 4, 2.0
PROMPT_LENS, RESPONSE_LENS = [2, 4, 3], [5, 1, 3]
MAX_PROMPT, MAX_RESPONSE = max(PROMPT_LENS), max(RESPONSE_LENS)


def _make_engine(pad_to_length=False):
    engine = object.__new__(FSDPEngineWithLMHead)
    engine.use_ulysses_sp = False
    engine.pad_to_length = pad_to_length
    engine.pad_to_length_bucket = 8
    engine.engine_config = SimpleNamespace(entropy_checkpointing=False, entropy_from_logits_with_chunking=False)
    return engine


def _make_batch(nested_layout: bool, seed: int = 0) -> TensorDict:
    generator = torch.Generator().manual_seed(seed)
    prompt_rows, response_rows, sequences = [], [], []
    for prompt_len, response_len in zip(PROMPT_LENS, RESPONSE_LENS, strict=True):
        prompt = torch.randint(0, VOCAB, (prompt_len,), generator=generator)
        response = torch.randint(0, VOCAB, (response_len,), generator=generator)
        prompt_rows.append(prompt)
        response_rows.append(response)
        sequences.append(torch.cat([prompt, response]))

    batch = {
        "input_ids": torch.nested.as_nested_tensor(sequences, layout=torch.jagged),
        "position_ids": torch.nested.as_nested_tensor([torch.arange(len(s)) for s in sequences], layout=torch.jagged),
    }
    advantages = [torch.randn(n, generator=generator) for n in RESPONSE_LENS]
    old_log_probs = [-torch.rand(n, generator=generator) for n in RESPONSE_LENS]
    if nested_layout:
        batch["prompts"] = torch.nested.as_nested_tensor(prompt_rows, layout=torch.jagged)
        batch["responses"] = torch.nested.as_nested_tensor(response_rows, layout=torch.jagged)
        batch["response_mask"] = torch.nested.as_nested_tensor(
            [torch.ones(n, dtype=torch.long) for n in RESPONSE_LENS], layout=torch.jagged
        )
        batch["advantages"] = torch.nested.as_nested_tensor(advantages, layout=torch.jagged)
        batch["old_log_probs"] = torch.nested.as_nested_tensor(old_log_probs, layout=torch.jagged)
    else:
        bsz = len(PROMPT_LENS)
        prompts = torch.zeros(bsz, MAX_PROMPT, dtype=torch.long)
        responses = torch.zeros(bsz, MAX_RESPONSE, dtype=torch.long)
        attention_mask = torch.zeros(bsz, MAX_PROMPT + MAX_RESPONSE, dtype=torch.long)
        response_mask = torch.zeros(bsz, MAX_RESPONSE, dtype=torch.long)
        padded_advantages = torch.zeros(bsz, MAX_RESPONSE)
        padded_old_log_probs = torch.zeros(bsz, MAX_RESPONSE)
        for i, (prompt, response) in enumerate(zip(prompt_rows, response_rows, strict=True)):
            prompts[i, MAX_PROMPT - len(prompt) :] = prompt  # left-padded prompt
            attention_mask[i, MAX_PROMPT - len(prompt) : MAX_PROMPT] = 1
            responses[i, : len(response)] = response  # right-padded response
            attention_mask[i, MAX_PROMPT : MAX_PROMPT + len(response)] = 1
            response_mask[i, : len(response)] = 1
            padded_advantages[i, : len(response)] = advantages[i]
            padded_old_log_probs[i, : len(response)] = old_log_probs[i]
        batch.update(
            prompts=prompts,
            responses=responses,
            attention_mask=attention_mask,
            response_mask=response_mask,
            advantages=padded_advantages,
            old_log_probs=padded_old_log_probs,
        )
    data = TensorDict(batch, batch_size=[len(PROMPT_LENS)])
    tu.assign_non_tensor(
        data,
        pad_mode=DatasetPadMode.NO_PADDING,
        use_fused_kernels=False,
        calculate_entropy=False,
        temperature=1.0,
    )
    return data


def _logits_output(engine, packed_logits, data, use_remove_padding):
    """Shape the packed logits as the model would return them."""
    if use_remove_padding:
        pad = engine._get_packed_pad_size(packed_logits.shape[0])
        return SimpleNamespace(logits=torch.nn.functional.pad(packed_logits, (0, 0, 0, pad)).unsqueeze(0))
    rows = packed_logits.split(data["input_ids"].offsets().diff().tolist())
    padded = torch.nn.utils.rnn.pad_sequence(list(rows), batch_first=True)
    return SimpleNamespace(logits=padded)


def _forward(engine, data, packed_logits, use_remove_padding):
    tu.assign_non_tensor(data, use_remove_padding=use_remove_padding)
    _, output_args = engine.prepare_model_inputs(data)
    output = _logits_output(engine, packed_logits, data, use_remove_padding)
    return engine.prepare_model_outputs(output, output_args, data, logits_processor_func=None)


def _response_level(tensor, data, nested_layout):
    """Store a response-level feature the way the matching trainer does."""
    padded = no_padding_2_padding(tensor, data)
    if not nested_layout:
        return padded
    return torch.nested.as_nested_tensor([padded[i, :n] for i, n in enumerate(RESPONSE_LENS)], layout=torch.jagged)


def test_response_to_packed_sequence_inverts_no_padding_2_padding():
    for nested_layout in (False, True):
        data = _make_batch(nested_layout)
        total = int(data["input_ids"].offsets()[-1])
        packed = torch.randn(total, 3)
        response = no_padding_2_padding(packed, data)
        stored = _response_level(packed, data, nested_layout)
        restored = response_to_packed_sequence(stored, data)
        torch.testing.assert_close(no_padding_2_padding(restored, data), response)


@pytest.mark.parametrize("nested_layout", [False, True], ids=["legacy", "v1"])
@pytest.mark.parametrize(
    "use_remove_padding,pad_to_length",
    [(True, False), (True, True), (False, False)],
    ids=["rmpad", "rmpad-static-pad", "padded"],
)
def test_local_features_and_loss_through_the_engine(nested_layout, use_remove_padding, pad_to_length):
    engine = _make_engine(pad_to_length=pad_to_length)
    data = _make_batch(nested_layout)
    total = int(data["input_ids"].offsets()[-1])
    old_logits = torch.randn(total, VOCAB, generator=torch.Generator().manual_seed(1)) * 3

    # Old-log-prob pass: the engine extracts the old-policy Top-K features for every packed row.
    tu.assign_non_tensor(data, **{OLD_FEATURES_FLAG: TOPK})
    old_output = _forward(engine, data, old_logits, use_remove_padding)
    labels = torch.roll(data["input_ids"].values(), shifts=-1)
    expected = compute_old_topk_features(old_logits, labels, TOPK)
    for key, value in zip(OLD_FEATURE_KEYS, expected, strict=True):
        torch.testing.assert_close(no_padding_2_padding(old_output[key], data), no_padding_2_padding(value, data))
        data[key] = _response_level(old_output[key], data, nested_layout)
    tu.assign_non_tensor(data, **{OLD_FEATURES_FLAG: 0, CURRENT_FEATURES_FLAG: True})

    # Actor update at the old parameters, up to one logit offset per state: the residual is -eta * A.
    current_logits = (old_logits + 5.0 * torch.randn(total, 1)).requires_grad_()
    model_output = _forward(engine, data, current_logits, use_remove_padding)
    torch.testing.assert_close(
        no_padding_2_padding(model_output[CENTERED_LOGITS_KEY], data),
        no_padding_2_padding(old_output[OLD_CENTERED_LOGITS_KEY], data),
        rtol=1e-5,
        atol=1e-4,
    )

    config = ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        ppo_micro_batch_size=2,
        policy_loss=PolicyLossConfig(loss_mode="local", local_eta=ETA, local_topk=TOPK),
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=None, global_batch_size=None)
    loss, metrics = ppo_loss(config, model_output, data)
    advantages = torch.cat([a[:n] for a, n in zip(data["advantages"].unbind(), RESPONSE_LENS, strict=True)])
    torch.testing.assert_close(loss, 0.5 * ETA**2 * advantages.square().mean(), rtol=1e-4, atol=1e-4)
    assert metrics["actor/local_feature_delta_abs"].aggregate() < 1e-3

    # The gradient reaches only response rows, at the label and the stored Top-K columns.
    loss.backward()
    grad_rows = current_logits.grad.abs().sum(dim=-1).nonzero().squeeze(-1).tolist()
    response_rows = no_padding_2_padding(torch.arange(total), data)
    expected_rows = {int(response_rows[i, t]) for i, n in enumerate(RESPONSE_LENS) for t in range(n)}
    assert grad_rows and set(grad_rows) <= expected_rows
    topk_ids = response_to_packed_sequence(data[OLD_TOPK_IDS_KEY], data)
    for row in grad_rows:
        allowed = set(topk_ids[row].tolist()) | {int(labels[row])}
        assert set(current_logits.grad[row].nonzero().squeeze(-1).tolist()) <= allowed
