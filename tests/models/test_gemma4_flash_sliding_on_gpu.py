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
"""GPU coverage for running Gemma4 under a flash ``attn_implementation``.

Builds a small but *real* Gemma4 text model exercising both attention flavours (head_dim-256
sliding with the theta=1e4 full-rotary RoPE, and head_dim-512 global with the theta=1e6
partial-rotary RoPE and ``attention_k_eq_v``) and checks that ``flash_attention_2`` reproduces
what ``sdpa`` computes, that padding never reaches a valid token, that ``sdpa`` is left alone,
and that the unsupported combinations fail loudly. Randomly initialised -- no checkpoint or
dataset required.
"""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("Requires CUDA", allow_module_level=True)
pytest.importorskip("flash_attn")
pytest.importorskip("transformers.models.gemma4.modeling_gemma4")

# 4 sliding : 2 global -- exercises both kernels and the fall-through
MIXED = ["sliding_attention", "sliding_attention", "full_attention"] * 2
GLOBAL_ONLY = ["full_attention", "full_attention"]

SEQ_LEN = 12
LENS = (12, 7)


def _config(layer_types, use_bidirectional_attention="vision"):
    """Tiny config that keeps the load-bearing shapes: head_dim 256/512 and the two RoPEs."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    return Gemma4TextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=len(layer_types),
        layer_types=list(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=256,  # sliding layers (flash-legal)
        global_head_dim=512,  # global layers (flash-illegal -> fall back to SDPA)
        num_global_key_value_heads=1,
        attention_k_eq_v=True,  # global layers: value_states = key_states, as in the real model
        sliding_window=8,  # small so the window actually bites within a short sequence
        rope_parameters={
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {"rope_theta": 10_000.0, "rope_type": "default"},
        },
        use_bidirectional_attention=use_bidirectional_attention,  # "vision" -> is_causal=True for text
        hidden_size_per_layer_input=0,
        enable_moe_block=False,  # dense MLP; MoE is per-token and orthogonal to attention
        num_kv_shared_layers=0,
        final_logit_softcapping=None,
        attention_dropout=0.0,
    )


def _batch(cfg, left_pad=False):
    """Right-padded (default) batch of two sequences of different real lengths."""
    input_ids = torch.randint(3, cfg.vocab_size, (len(LENS), SEQ_LEN), device="cuda")
    attention_mask = torch.zeros(len(LENS), SEQ_LEN, dtype=torch.long, device="cuda")
    for row, length in enumerate(LENS):
        if left_pad:
            attention_mask[row, SEQ_LEN - length :] = 1
            input_ids[row, : SEQ_LEN - length] = cfg.pad_token_id
        else:
            attention_mask[row, :length] = 1
            input_ids[row, length:] = cfg.pad_token_id
    position_ids = torch.arange(SEQ_LEN, device="cuda").unsqueeze(0).expand(len(LENS), SEQ_LEN)
    return input_ids, attention_mask, position_ids


def _build(layer_types, impl, dtype=torch.bfloat16):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    torch.manual_seed(0)  # same init for every implementation, so outputs are comparable
    cfg = _config(layer_types)
    return Gemma4TextModel._from_config(cfg, attn_implementation=impl).to(device="cuda", dtype=dtype).eval()


def _run(model, batch):
    input_ids, attention_mask, position_ids = batch
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,  # keep past_key_values None: the training forward
        )
    return out.last_hidden_state.float()


@pytest.fixture(autouse=True)
def _restore_forward():
    """Undo every patch between tests, so each starts from stock transformers.

    Covers the rebound mask builders as well as the class-level forward: leaving those in place
    would let one test's patch satisfy the next test's assertions.
    """
    from transformers.models.gemma4 import modeling_gemma4
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

    original_forward = Gemma4TextAttention.forward
    original_builders = (modeling_gemma4.create_causal_mask, modeling_gemma4.create_masks_for_generate)
    yield
    Gemma4TextAttention.forward = original_forward
    modeling_gemma4.create_causal_mask, modeling_gemma4.create_masks_for_generate = original_builders


@pytest.mark.parametrize("layer_types", [MIXED, GLOBAL_ONLY], ids=["mixed", "global_only"])
def test_flash_matches_sdpa_on_valid_tokens(layer_types):
    """flash_attention_2 must reproduce sdpa.

    ``global_only`` isolates the part this patch actually changes -- the head_dim-512 layers,
    where the mask has to be built in SDPA format while the rest of the model runs on flash.
    """
    from verl.models.transformers.gemma4 import apply_gemma4_flash_attention

    torch.manual_seed(0)
    batch = _batch(_config(layer_types))
    valid = batch[1].bool()

    ref = _run(_build(layer_types, "sdpa"), batch)  # reference: stock, unpatched

    model = _build(layer_types, "flash_attention_2")
    assert apply_gemma4_flash_attention(model) is True
    got = _run(model, batch)

    # Calibrate against bf16 itself rather than a hand-picked epsilon: run the *same* stock sdpa
    # model in fp32 and measure how far bf16 alone moves it. Two implementations that agree to
    # better than that are indistinguishable at the precision we train in, and the bar tracks the
    # model/tolerance of the machine instead of a constant that silently rots.
    noise_floor = (ref[valid] - _run(_build(layer_types, "sdpa", torch.float32), batch)[valid]).abs().max().item()

    # Compare only valid (non-pad) tokens -- pad-token outputs are discarded by the loss mask.
    max_abs = (got[valid] - ref[valid]).abs().max().item()
    assert max_abs < noise_floor, (
        f"flash diverges from sdpa on valid tokens by more than bf16 rounding does: "
        f"max|delta|={max_abs:.3e} vs bf16 noise floor {noise_floor:.3e}"
    )

    # Padding must not leak into valid tokens: corrupt the pad positions and re-run. The sliding
    # layers unpad before the kernel and the global layers get the padding in their mask, so
    # valid-token outputs must come back *bit identical* -- not merely close.
    corrupted = batch[0].clone()
    for row, length in enumerate(LENS):
        corrupted[row, length:] = 5
    got2 = _run(model, (corrupted, batch[1], batch[2]))
    assert torch.equal(got2[valid], got[valid]), "padding leaked into valid tokens"


def test_sdpa_implementation_is_untouched():
    """Under attn_implementation=sdpa the patch must be a no-op, bit for bit.

    This is the escape hatch: anything the flash path cannot serve (Ulysses SP, head_dim-512
    kernels) has to keep working exactly as it did before this patch existed.
    """
    from verl.models.transformers.gemma4 import apply_gemma4_flash_attention

    torch.manual_seed(0)
    batch = _batch(_config(MIXED))

    model = _build(MIXED, "sdpa")
    before = _run(model, batch)
    assert apply_gemma4_flash_attention(model) is True
    after = _run(model, batch)

    assert torch.equal(before, after), "patch perturbed the sdpa path"


def test_ulysses_sequence_parallel_with_flash_raises():
    """The global layers run on SDPA, bypassing the SP-aware flash forward Ulysses patches in.

    Refusing is a deliberate choice: the sharded sequence would otherwise be attended incorrectly
    with no crash to notice.
    """
    from verl.models.transformers.gemma4 import apply_gemma4_flash_attention

    model = _build(MIXED, "flash_attention_2")
    with pytest.raises(ValueError, match="Ulysses"):
        apply_gemma4_flash_attention(model, ulysses_sp_size=2)

    # ...but it is fine under sdpa, which is how Gemma4 ran before this patch.
    assert apply_gemma4_flash_attention(_build(MIXED, "sdpa"), ulysses_sp_size=2) is True


@pytest.mark.parametrize("layer_types", [MIXED, GLOBAL_ONLY], ids=["mixed", "global_only"])
def test_packed_sequences_do_not_attend_across_boundaries(layer_types):
    """veRL's remove-padding layout: one (1, total_nnz) row, attention_mask=None, boundaries in
    position_ids only.

    This is the regression test for the bug this patch exists to avoid. Perturb the *first* packed
    sequence and the second one's hidden states must not move at all: the sliding layers get
    cu_seqlens from position_ids, the global layers a block-diagonal mask. Applying plain causality
    to the flat row instead leaks the previous sequence into the first `sliding_window - 1` tokens
    of the next one.
    """
    from verl.models.transformers.gemma4 import apply_gemma4_flash_attention

    torch.manual_seed(0)
    cfg = _config(layer_types)
    l1, l2 = 9, 7
    ids = torch.randint(3, cfg.vocab_size, (1, l1 + l2), device="cuda")
    position_ids = torch.cat(
        [torch.arange(l1, device="cuda"), torch.arange(l2, device="cuda")]
    ).unsqueeze(0)

    model = _build(layer_types, "flash_attention_2")
    assert apply_gemma4_flash_attention(model) is True
    ref = _run(model, (ids, None, position_ids))

    perturbed = ids.clone()
    perturbed[0, :l1] = (perturbed[0, :l1] + 7) % cfg.vocab_size  # sequence 1 only
    got = _run(model, (perturbed, None, position_ids))

    assert torch.equal(got[:, l1:], ref[:, l1:]), (
        "packed sequence 2 changed when only sequence 1 was perturbed -- attention crossed the "
        "boundary"
    )
    assert not torch.equal(got[:, :l1], ref[:, :l1]), "sequence 1 unchanged: the test is vacuous"


def _composite_config(layer_types, use_bidirectional_attention="vision"):
    """Wrap the text config in the composite (multimodal) config, as the released checkpoints are."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config

    return Gemma4Config(text_config=_config(layer_types, use_bidirectional_attention).to_dict())


@pytest.mark.parametrize("layer_types", [MIXED, GLOBAL_ONLY], ids=["mixed", "global_only"])
def test_composite_model_packed_sequences_do_not_attend_across_boundaries(layer_types):
    """Same boundary guarantee, but through ``Gemma4ForConditionalGeneration``.

    This is the class ``AutoModel`` resolves the released checkpoints to (their config is
    ``model_type: gemma4`` with ``architectures: [Gemma4ForConditionalGeneration]``), so it is what
    veRL actually loads -- while the tests above exercise the bare text tower. The two differ in a
    way that matters: the composite forward builds the per-layer-type mask mapping *itself* and
    passes it down as a dict, so a patch that only intercepts the text tower's forward never runs
    and the global layers silently attend across packed boundaries.
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

    from verl.models.transformers.gemma4 import apply_gemma4_flash_attention

    torch.manual_seed(0)
    cfg = _composite_config(layer_types)
    vocab_size = cfg.get_text_config().vocab_size
    l1, l2 = 9, 7
    ids = torch.randint(3, vocab_size, (1, l1 + l2), device="cuda")
    position_ids = torch.cat([torch.arange(l1, device="cuda"), torch.arange(l2, device="cuda")]).unsqueeze(0)

    model = (
        Gemma4ForConditionalGeneration._from_config(cfg, attn_implementation="flash_attention_2")
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    assert apply_gemma4_flash_attention(model) is True

    def run(input_ids):
        with torch.no_grad():
            return model(
                input_ids=input_ids, attention_mask=None, position_ids=position_ids, use_cache=False
            ).logits.float()

    ref = run(ids)
    perturbed = ids.clone()
    perturbed[0, :l1] = (perturbed[0, :l1] + 7) % vocab_size
    got = run(perturbed)

    assert torch.equal(got[:, l1:], ref[:, l1:]), (
        "packed sequence 2 changed when only sequence 1 was perturbed -- the global layers "
        "attended across the boundary on the composite model path"
    )
    assert not torch.equal(got[:, :l1], ref[:, :l1]), "sequence 1 unchanged: the test is vacuous"


def test_non_vision_variant_with_flash_raises():
    """Variants that are not ``use_bidirectional_attention="vision"`` are refused under flash.

    Their forward builds the mask mapping through ``create_masks_for_generate`` instead of
    ``create_causal_mask_mapping``, and that builder resolves each layer type through
    masking_utils' pattern mapping -- so it never reaches the ``create_causal_mask`` rebinding.
    The kernel dispatch keys on ``self.is_sliding`` and would still send the head_dim-512 layers to
    SDPA, leaving them with a flash-format mask: under remove-padding that attends across packed
    sequence boundaries and returns a plausible answer rather than raising. Refusing is the
    conservative choice -- these variants still run under ``sdpa``, which is how they ran before.
    """
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

    from verl.models.transformers.gemma4 import apply_gemma4_flash_attention

    cfg = _composite_config(MIXED, use_bidirectional_attention=None)
    model = Gemma4ForConditionalGeneration._from_config(cfg, attn_implementation="flash_attention_2")
    with pytest.raises(ValueError, match="use_bidirectional_attention"):
        apply_gemma4_flash_attention(model)

    # ...but sdpa is fine, and stays a no-op for them.
    sdpa_model = Gemma4ForConditionalGeneration._from_config(
        _composite_config(MIXED, use_bidirectional_attention=None), attn_implementation="sdpa"
    )
    assert apply_gemma4_flash_attention(sdpa_model) is True


def test_monkey_patch_dispatch_reaches_gemma4():
    """``apply_monkey_patch`` must route a Gemma4 config to this patch.

    The tests above call ``apply_gemma4_flash_attention`` directly, so a wrong ``model_type`` string
    in the ``monkey_patch`` dispatch would leave every one of them green while the patch never runs
    in training.
    """
    from transformers.models.gemma4 import modeling_gemma4
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

    from verl.models.transformers.monkey_patch import apply_monkey_patch

    model = Gemma4ForConditionalGeneration._from_config(
        _composite_config(MIXED), attn_implementation="flash_attention_2"
    )
    apply_monkey_patch(model, use_remove_padding=True, ulysses_sp_size=1)

    assert getattr(modeling_gemma4.create_causal_mask, "_gemma4_sdpa_global_mask", False), (
        "apply_monkey_patch did not reach the gemma4 patch -- check the model_type dispatch"
    )
