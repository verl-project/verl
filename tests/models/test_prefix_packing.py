# Copyright 2025 Individual Contributor: Deepro Choudhury
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

"""GPU parity of the prefix-packed flash-attention forward against the rmpad forward.

Tolerances follow ``tests/models/test_transformer.py``.
"""

import pytest
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2ForTokenClassification,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils.device import get_device_name
from verl.utils.torch_functional import logprobs_from_logits
from verl.workers.utils.prefix_packing import build_prefix_packing_plan, find_prefix_groups

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device with flash-attn")

DEVICE = get_device_name()


@pytest.fixture(autouse=True)
def _restore_model_forward(monkeypatch):
    # apply_monkey_patch(use_fused_kernels=True) replaces forward on the class.
    for cls in (Qwen2ForCausalLM, Qwen3ForCausalLM, Qwen2ForTokenClassification):
        monkeypatch.setattr(cls, "forward", cls.forward)


def _configs():
    base = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=257,
    )
    return [
        pytest.param(Qwen2Config(**base, use_sliding_window=False), id="qwen2"),
        pytest.param(
            Qwen2Config(**base, use_sliding_window=True, sliding_window=6, max_window_layers=0), id="qwen2-sliding"
        ),
        pytest.param(Qwen3Config(**base), id="qwen3"),
    ]


def _make_batch(vocab_size: int, seed: int = 0):
    """Three prompt groups of 4, 3 and 1 samples."""
    g = torch.Generator().manual_seed(seed)
    rows, prompt_lens = [], []
    for n in (4, 3, 1):
        p_len = int(torch.randint(5, 10, (1,), generator=g))
        prompt = torch.randint(1, vocab_size, (p_len,), generator=g)
        for _ in range(n):
            r_len = int(torch.randint(2, 8, (1,), generator=g))
            rows.append(torch.cat([prompt, torch.randint(1, vocab_size, (r_len,), generator=g)]))
            prompt_lens.append(p_len)
    input_ids = torch.nested.as_nested_tensor([r.to(DEVICE) for r in rows], layout=torch.jagged)
    position_ids = torch.nested.as_nested_tensor(
        [torch.arange(len(r), device=DEVICE) for r in rows], layout=torch.jagged
    )
    return input_ids, position_ids, torch.tensor(prompt_lens, device=DEVICE)


def _read_rows(input_ids, prompt_lens):
    """Rows ``no_padding_2_padding`` reads: positions ``P-1 .. L-2`` of each sample."""
    offsets = input_ids.offsets().tolist()
    idx = []
    for i, (s, e) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        idx.extend(range(s + int(prompt_lens[i]) - 1, e - 1))
    return torch.tensor(idx, device=DEVICE)


def _forward(model, ids, pos, labels, use_fused_kernels, prefix_packing=None):
    kwargs = {"input_ids": ids, "attention_mask": None, "position_ids": pos}
    if prefix_packing is not None:
        kwargs["prefix_packing"] = prefix_packing
    if use_fused_kernels:
        out = model(**kwargs, temperature=1.0, return_dict=True, shift_labels=labels.unsqueeze(0))
        return out.log_probs.squeeze(0).float(), out.entropy.squeeze(0).float()
    logits = model(**kwargs).logits.squeeze(0).float()
    log_probs = logprobs_from_logits(logits, labels, inplace_backward=False)
    entropy = -(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(-1)
    return log_probs, entropy


@pytest.mark.parametrize("config", _configs())
@pytest.mark.parametrize("use_fused_kernels", [False, True], ids=["eager-head", "fused-head"])
@pytest.mark.parametrize("gradient_checkpointing", [False, True], ids=["no-ckpt", "ckpt"])
def test_actor_parity(config, use_fused_kernels, gradient_checkpointing):
    """Packed log-probs, entropy and gradients match the rmpad forward."""
    torch.manual_seed(0)
    with torch.device(DEVICE):
        model = AutoModelForCausalLM.from_config(
            config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
    model = model.to(DEVICE)
    apply_monkey_patch(
        model=model,
        use_remove_padding=True,
        ulysses_sp_size=1,
        use_fused_kernels=use_fused_kernels,
        fused_kernels_backend="torch",
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    input_ids, position_ids, prompt_lens = _make_batch(config.vocab_size)
    flat, flat_pos = input_ids.values(), position_ids.values()
    read = _read_rows(input_ids, prompt_lens)
    weights = torch.randn(read.numel(), device=DEVICE)

    ref_lp, ref_ent = _forward(model, flat.unsqueeze(0), flat_pos.unsqueeze(0), torch.roll(flat, -1), use_fused_kernels)
    (ref_lp[read] * weights).sum().backward()
    ref_grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)

    assert find_prefix_groups(input_ids, prompt_lens)[0] == [4, 3, 1]
    plan = build_prefix_packing_plan(input_ids, prompt_lens)
    assert plan.num_packed < plan.num_unpacked
    packed_ids, packed_pos = flat[plan.pack_index], flat_pos[plan.pack_index]
    meta = plan.attn
    lp, ent = _forward(
        model, packed_ids.unsqueeze(0), packed_pos.unsqueeze(0), torch.roll(packed_ids, -1), use_fused_kernels, meta
    )
    assert meta.layers_seen == config.num_hidden_layers
    lp, ent = lp[plan.unpack_index], ent[plan.unpack_index]
    (lp[read] * weights).sum().backward()

    torch.testing.assert_close(lp[read], ref_lp[read], atol=1e-2, rtol=1e-5)
    torch.testing.assert_close(ent[read], ref_ent[read], atol=1e-2, rtol=1e-5)
    for p, g_ref in zip(model.parameters(), ref_grads, strict=True):
        if g_ref is None:
            continue
        cos = F.cosine_similarity(p.grad.flatten().float(), g_ref.flatten().float(), dim=0).item()
        assert cos > 0.999, f"gradient mismatch: cosine {cos}"


def test_critic_parity():
    """Packed values match the rmpad forward."""
    torch.manual_seed(0)
    config = Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=257,
        num_labels=1,
    )
    with torch.device(DEVICE):
        model = AutoModelForTokenClassification.from_config(
            config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
    model = model.to(DEVICE)
    apply_monkey_patch(model=model, use_remove_padding=True, ulysses_sp_size=1)
    model.eval()  # the score head has dropout
    input_ids, position_ids, prompt_lens = _make_batch(config.vocab_size)
    flat, flat_pos = input_ids.values(), position_ids.values()
    read = _read_rows(input_ids, prompt_lens)

    ref = model(input_ids=flat.unsqueeze(0), attention_mask=None, position_ids=flat_pos.unsqueeze(0)).logits.squeeze()
    plan = build_prefix_packing_plan(input_ids, prompt_lens)
    packed = model(
        input_ids=flat[plan.pack_index].unsqueeze(0),
        attention_mask=None,
        position_ids=flat_pos[plan.pack_index].unsqueeze(0),
        prefix_packing=plan.attn,
    ).logits.squeeze()
    assert plan.attn.layers_seen == config.num_hidden_layers
    torch.testing.assert_close(packed[plan.unpack_index][read].float(), ref[read].float(), atol=1e-2, rtol=1e-5)


def test_sdpa_bypasses_wrapper_and_counter_stays_zero():
    """With sdpa no layer records the packed path, which the engine's layer guard relies on."""
    config = Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=257,
    )
    with torch.device(DEVICE):
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.to(DEVICE)
    apply_monkey_patch(model=model, use_remove_padding=True, ulysses_sp_size=1)
    input_ids, position_ids, prompt_lens = _make_batch(config.vocab_size)
    plan = build_prefix_packing_plan(input_ids, prompt_lens)
    model(
        input_ids=input_ids.values()[plan.pack_index].unsqueeze(0),
        attention_mask=None,
        position_ids=position_ids.values()[plan.pack_index].unsqueeze(0),
        prefix_packing=plan.attn,
    )
    assert plan.attn.layers_seen == 0
