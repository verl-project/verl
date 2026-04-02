# Copyright 2026 Amazon.com Inc and/or its affiliates
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
import random

import pytest
import torch
from tensordict import TensorDict

from verl.workers.utils.padding import (
    compress_batch_dtypes,
    extract_response,
    make_mask_nesting_specs,
    nest_batch_by_mask,
    prepare_response_slice,
    prepare_unnest,
    slice_response,
    unnest,
    unnest_batch_by_mask,
)

try:
    from flash_attn.bert_padding import unpad_input as _  # noqa: F401

    _has_flash_attn = True
except ImportError:
    _has_flash_attn = False

if _has_flash_attn:
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

requires_flash_attn = pytest.mark.skipif(not _has_flash_attn, reason="flash_attn not installed")

# Token-ID pad value (stand-in for tokenizer.pad_token_id in production).
_TEST_PAD_TOKEN_ID = 0


def _nest(data):
    """Test helper: build specs + nest in-place. Returns (data, specs).

    Exercises the "pre-built specs" entry point of ``nest_batch`` so the
    returned specs dict can be inspected by assertion-based tests. The
    simpler ``nest_batch(data, pad_token_id=...)`` call is tested
    separately via ``TestNestBatchSimple``.
    """
    specs = make_mask_nesting_specs(data, pad_token_id=_TEST_PAD_TOKEN_ID)
    nest_batch_by_mask(data, specs)
    return data, specs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(prompt_lens, response_lens, max_seq_len, max_response_len):
    """Helper to build a TensorDict with given prompt/response lengths."""
    batch_size = len(prompt_lens)
    input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_seq_len)
    response_mask = torch.zeros(batch_size, max_response_len)
    position_ids = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1).clone()

    for i in range(batch_size):
        total_len = prompt_lens[i] + response_lens[i]
        input_ids[i, :total_len] = torch.arange(1, total_len + 1)
        attention_mask[i, :total_len] = 1
        response_mask[i, : response_lens[i]] = 1

    prompt_list = [input_ids[i, : prompt_lens[i]] for i in range(batch_size)]
    response_list = [input_ids[i, prompt_lens[i] : prompt_lens[i] + response_lens[i]] for i in range(batch_size)]
    prompts_nested = torch.nested.as_nested_tensor(prompt_list, layout=torch.jagged)
    responses_nested = torch.nested.as_nested_tensor(response_list, layout=torch.jagged)

    return TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
            "prompts": prompts_nested,
            "responses": responses_nested,
        }
    )


# ---------------------------------------------------------------------------
# Tests requiring flash_attn (old path)
# ---------------------------------------------------------------------------


@requires_flash_attn
def test_padding_conversion_with_log_probs():
    """Test that log probability tensors remain in padded format after conversion (old path)."""
    batch_size = 4
    max_seq_len = 128
    max_response_len = 64

    input_ids = torch.randint(0, 1000, (batch_size, max_seq_len))
    attention_mask = torch.zeros(batch_size, max_seq_len)
    valid_lens = [100, 120, 90, 128]
    for i, vlen in enumerate(valid_lens):
        attention_mask[i, :vlen] = 1
    response_mask = torch.zeros(batch_size, max_response_len)
    response_lens = [50, 60, 45, 64]
    for i, rlen in enumerate(response_lens):
        response_mask[i, :rlen] = 1
    position_ids = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1)

    data = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
            "old_log_probs": torch.randn(batch_size, max_seq_len),
            "ref_log_prob": torch.randn(batch_size, max_seq_len),
            "advantages": torch.randn(batch_size, max_response_len),
            "rollout_log_probs": torch.randn(batch_size, max_seq_len),
        }
    )
    data_converted = left_right_2_no_padding(data)

    assert data_converted["input_ids"].is_nested
    assert data_converted["position_ids"].is_nested
    assert not data_converted["old_log_probs"].is_nested
    assert not data_converted["ref_log_prob"].is_nested
    assert not data_converted["advantages"].is_nested
    assert data_converted["old_log_probs"].shape == (batch_size, max_seq_len)
    for i, vlen in enumerate(valid_lens):
        assert data_converted["input_ids"][i].numel() == vlen


@requires_flash_attn
def test_padding_conversion_without_log_probs():
    """Test that padding conversion works without log prob tensors (old path)."""
    batch_size = 4
    max_seq_len = 128
    max_response_len = 64
    data = TensorDict(
        {
            "input_ids": torch.randint(0, 1000, (batch_size, max_seq_len)),
            "attention_mask": torch.ones(batch_size, max_seq_len),
            "response_mask": torch.ones(batch_size, max_response_len),
            "position_ids": torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1),
        }
    )
    data_converted = left_right_2_no_padding(data)
    assert data_converted["input_ids"].is_nested
    assert data_converted["position_ids"].is_nested


@requires_flash_attn
def test_padding_roundtrip():
    """Test roundtrip via old path preserves response values."""
    batch_size = 2
    max_seq_len = 64
    max_response_len = 32
    prompt_len = 32

    input_ids = torch.arange(1, max_seq_len + 1).unsqueeze(0).expand(batch_size, -1).clone()
    attention_mask = torch.ones(batch_size, max_seq_len)
    response_mask = torch.ones(batch_size, max_response_len)
    position_ids = torch.arange(max_seq_len).unsqueeze(0).expand(batch_size, -1)
    prompt_list = [input_ids[i, :prompt_len] for i in range(batch_size)]
    response_list = [input_ids[i, prompt_len:] for i in range(batch_size)]
    prompts_nested = torch.nested.as_nested_tensor(prompt_list, layout=torch.jagged)
    responses_nested = torch.nested.as_nested_tensor(response_list, layout=torch.jagged)

    data = TensorDict(
        {
            "input_ids": input_ids,
            "prompts": prompts_nested,
            "responses": responses_nested,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "position_ids": position_ids,
        }
    )
    data_nested = left_right_2_no_padding(data)
    recovered = no_padding_2_padding(data_nested["input_ids"], data_nested)
    assert recovered.shape == (batch_size, max_response_len)
    expected = torch.arange(prompt_len, max_seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    torch.testing.assert_close(recovered, expected)


@requires_flash_attn
def test_no_padding_2_padding_varying_lengths():
    """Test old roundtrip with varied prompt/response lengths."""
    prompt_lens = [10, 30, 5, 40]
    response_lens = [40, 20, 45, 10]
    data = _make_batch(prompt_lens, response_lens, max_seq_len=100, max_response_len=50)
    data_nested = left_right_2_no_padding(data)
    ids = data_nested["input_ids"]
    output = torch.nested.nested_tensor_from_jagged(ids.values().float(), offsets=ids.offsets())
    result = no_padding_2_padding(output, data_nested)
    for i in range(4):
        expected = torch.arange(prompt_lens[i], prompt_lens[i] + response_lens[i], dtype=torch.float)
        torch.testing.assert_close(result[i, : response_lens[i]], expected, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests for RLE path (no flash_attn dependency)
# ---------------------------------------------------------------------------


class TestCompressBatchDtypes:
    """Tests for ``compress_batch_dtypes`` — orthogonal dtype pre-step."""

    def test_routed_experts_compressed_when_in_range(self):
        data = _make_batch([10, 20], [15, 10], 30, 15)
        data["routed_experts"] = torch.randint(0, 8, (2, 30), dtype=torch.long)
        compress_batch_dtypes(data)
        assert data["routed_experts"].dtype == torch.uint8

    def test_routed_experts_skipped_when_out_of_range(self):
        """Values > 255 prevent the cast (would wrap around)."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        data["routed_experts"] = torch.full((2, 30), 300, dtype=torch.long)
        compress_batch_dtypes(data)
        # untouched: still int64
        assert data["routed_experts"].dtype == torch.long

    def test_routed_experts_skipped_on_negative_sentinel(self):
        """Negative sentinels prevent the cast (would wrap to 255)."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        t = torch.randint(0, 8, (2, 30), dtype=torch.long)
        t[0, 5] = -1
        data["routed_experts"] = t
        compress_batch_dtypes(data)
        assert data["routed_experts"].dtype == torch.long

    def test_already_compressed_is_noop(self):
        data = _make_batch([10, 20], [15, 10], 30, 15)
        data["routed_experts"] = torch.randint(0, 8, (2, 30), dtype=torch.uint8)
        compress_batch_dtypes(data)
        # unchanged (same tensor object not guaranteed, but dtype is)
        assert data["routed_experts"].dtype == torch.uint8

    def test_unknown_field_untouched(self):
        """Fields not in KNOWN_FIELD_DTYPE_COMPRESSIONS are left alone."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        orig_input_ids_dtype = data["input_ids"].dtype
        compress_batch_dtypes(data)
        assert data["input_ids"].dtype == orig_input_ids_dtype


class TestNestBatchFieldRouting:
    """Tests for ``nest_batch`` field routing — nests every field in the registry."""

    def test_all_seq_fields_nested(self):
        """All fields paired with attention_mask become nested tensors."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        data["old_log_probs"] = torch.randn(2, 30)
        data["ref_log_prob"] = torch.randn(2, 30)
        data["rollout_log_probs"] = torch.randn(2, 30)

        data, specs = _nest(data)

        for key in ("input_ids", "position_ids", "old_log_probs", "ref_log_prob", "rollout_log_probs"):
            assert data[key].is_nested, f"{key} should be nested"

        seq_spec = specs["attention_mask"]
        assert seq_spec.offsets_field in data
        assert seq_spec.lengths_field in data

    def test_resp_fields_nested(self):
        """Fields paired with response_mask are nested."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        data["advantages"] = torch.randn(2, 15)
        data["values"] = torch.randn(2, 15)

        data, specs = _nest(data)

        assert "response_mask" in specs
        assert data["advantages"].is_nested, "advantages should be nested"
        assert data["values"].is_nested, "values should be nested"

        resp_spec = specs["response_mask"]
        assert resp_spec.offsets_field in data
        assert resp_spec.lengths_field in data

    def test_already_nested_fields_untouched(self):
        """Fields that are already nested (prompts, responses) are not re-nested."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        assert data["prompts"].is_nested  # pre-condition

        data, specs = _nest(data)

        assert data["prompts"].is_nested
        assert data["responses"].is_nested

    def test_3d_position_ids_auto_permuted(self):
        """Raw multimodal 3-D ``(bs, heads, seq_len)`` position_ids is permuted to canonical form.

        ``KNOWN_FIELD_PERMUTATIONS["position_ids"] = (0, 2, 1)`` tells
        the library to swap dims 1 and 2 of any 3-D ``position_ids``
        it sees. After nesting, each sample's position_ids is shaped
        ``(valid_len, heads)`` — the expected canonical layout with
        ``heads`` as a trailing feature dim.
        """
        batch_size = 2
        max_seq_len = 30
        max_response_len = 15
        heads = 4

        input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_seq_len)
        response_mask = torch.zeros(batch_size, max_response_len)
        # Producer layout: (bs, heads, seq_len) — heads sits between
        # the batch and sample-mask dims and must be permuted out.
        position_ids = torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(batch_size, heads, -1).clone()

        prompt_lens, response_lens = [10, 20], [15, 10]
        for i in range(batch_size):
            total = prompt_lens[i] + response_lens[i]
            input_ids[i, :total] = torch.arange(1, total + 1)
            attention_mask[i, :total] = 1
            response_mask[i, : response_lens[i]] = 1

        prompt_list = [input_ids[i, : prompt_lens[i]] for i in range(batch_size)]
        response_list = [input_ids[i, prompt_lens[i] : prompt_lens[i] + response_lens[i]] for i in range(batch_size)]

        data = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
                "prompts": torch.nested.as_nested_tensor(prompt_list, layout=torch.jagged),
                "responses": torch.nested.as_nested_tensor(response_list, layout=torch.jagged),
            }
        )

        data, _ = _nest(data)

        assert data["position_ids"].is_nested
        # Each sample should have (valid_len, heads) shape — trailing dim preserved
        for i in range(batch_size):
            valid_len = prompt_lens[i] + response_lens[i]
            assert data["position_ids"][i].shape == (valid_len, heads)

    def test_routed_experts_preserves_precompressed_dtype(self):
        """nest_batch_by_mask preserves whatever dtype the caller passed in.

        The dtype compression is now an orthogonal pre-step
        (:func:`compress_batch_dtypes`) — this test asserts that
        :func:`nest_batch_by_mask` does not secretly do it anymore.
        """
        data = _make_batch([10, 20], [15, 10], 30, 15)
        # Caller chose to pre-compress to uint8.
        data["routed_experts"] = torch.randint(0, 8, (2, 30), dtype=torch.uint8)

        data, _ = _nest(data)

        assert data["routed_experts"].is_nested
        assert data["routed_experts"].dtype == torch.uint8

    def test_nest_batch_by_mask_does_not_compress_dtype(self):
        """int64 routed_experts stays int64 through nest_batch_by_mask alone."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        data["routed_experts"] = torch.randint(0, 8, (2, 30), dtype=torch.long)

        data, _ = _nest(data)

        assert data["routed_experts"].is_nested
        # nest_batch_by_mask is a pure nesting op; it does not touch dtypes
        assert data["routed_experts"].dtype == torch.long


def test_embeds_padding_2_no_padding_varying_lengths():
    """Test that padding tokens are stripped correctly when sequences have different valid lengths."""
    from verl.workers.utils.padding import embeds_padding_2_no_padding

    batch_size = 3
    max_seq_len = 20
    dim = 16
    num_steps = 8

    # Simulate different valid lengths: 20, 15, 10 (rest are padding zeros)
    valid_lens = [20, 15, 10]
    prompt_embeds = torch.randn(batch_size, max_seq_len, dim)
    prompt_embeds_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.int32)
    for i, vlen in enumerate(valid_lens):
        prompt_embeds_mask[i, :vlen] = 1
    response_mask = torch.ones(batch_size, num_steps)

    data = TensorDict(
        {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": prompt_embeds_mask,
            "response_mask": response_mask,
        },
        batch_size=batch_size,
    )

    result = embeds_padding_2_no_padding(data)

    assert result["prompt_embeds"].is_nested

    # Each sample's nested embedding should have the correct stripped length
    embeds_nested = result["prompt_embeds"]
    for i, vlen in enumerate(valid_lens):
        sample_embed = embeds_nested[i]
        assert sample_embed.shape[0] == vlen, f"Sample {i}: expected {vlen} tokens, got {sample_embed.shape[0]}"
        # Values should match the original (unpadded portion)
        torch.testing.assert_close(sample_embed, prompt_embeds[i, :vlen, :])


def test_response_from_nested():
    from verl.workers.utils.padding import response_from_nested

    batch_size = 10
    log_probs = [torch.rand(random.randint(2, 100)) for _ in range(batch_size)]
    log_probs_nt = torch.nested.as_nested_tensor(
        log_probs,
        layout=torch.jagged,
    )
    response_mask = [torch.ones(random.randint(1, log_probs[i].shape[0] - 1)) for i in range(batch_size)]
    response_mask_nt = torch.nested.as_nested_tensor(
        response_mask,
        layout=torch.jagged,
    )
    response_log_probs = response_from_nested(log_probs_nt, response_mask_nt)
    for i, tensor in enumerate(response_log_probs.unbind()):
        response_len = response_mask[i].shape[0]
        expected = log_probs[i][-response_len - 1 : -1]
        torch.testing.assert_close(tensor, expected)


def test_response_to_nested():
    from verl.workers.utils.padding import response_to_nested

    batch_size = 10
    log_probs = torch.rand(batch_size, 100)
    response_mask = [torch.ones(random.randint(1, log_probs[i].shape[0] - 1)) for i in range(batch_size)]
    response_mask_nt = torch.nested.as_nested_tensor(
        response_mask,
        layout=torch.jagged,
    )
    log_probs_nt = response_to_nested(log_probs, response_mask_nt)
    for i, tensor in enumerate(log_probs_nt.unbind()):
        response_len = response_mask[i].shape[0]
        expected = log_probs[i, :response_len]
        torch.testing.assert_close(tensor, expected)


if __name__ == "__main__":
    test_padding_conversion_with_log_probs()
    test_padding_conversion_without_log_probs()
    test_padding_roundtrip()
    test_no_padding_2_padding_varying_lengths()
    test_embeds_padding_2_no_padding_varying_lengths()
    test_response_from_nested()
    test_response_to_nested()
    print("All padding conversion tests passed!")


class TestNestUnnestRoundtrip:
    """Tests for ``nest_batch`` ↔ ``unnest_batch`` full round-trip."""

    def test_seq_fields_roundtrip(self):
        """seq-level fields survive nest → unnest at mask=True positions."""
        prompt_lens = [10, 30, 5, 40]
        response_lens = [40, 20, 45, 10]
        max_seq_len = 100
        max_response_len = 50

        data = _make_batch(prompt_lens, response_lens, max_seq_len, max_response_len)
        attn_mask = data["attention_mask"].bool()
        # Zero out padding positions so round-trip is exact
        data["position_ids"] = data["position_ids"] * attn_mask.long()
        orig_input_ids = data["input_ids"].clone()
        orig_position_ids = data["position_ids"].clone()

        data, specs = _nest(data)

        assert data["input_ids"].is_nested
        assert data["position_ids"].is_nested

        unnest_batch_by_mask(data, specs)

        assert not data["input_ids"].is_nested
        assert not data["position_ids"].is_nested
        assert data["attention_mask"].shape == (4, max_seq_len)

        torch.testing.assert_close(data["input_ids"], orig_input_ids)
        torch.testing.assert_close(data["position_ids"], orig_position_ids)

    def test_resp_fields_roundtrip(self):
        """resp-level fields survive nest → unnest at mask=True positions."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        resp_mask = data["response_mask"].bool()
        # Zero out padding positions so round-trip is exact
        data["advantages"] = torch.randn(2, 15) * resp_mask.float()
        data["values"] = torch.randn(2, 15) * resp_mask.float()
        orig_adv = data["advantages"].clone()
        orig_val = data["values"].clone()

        data, specs = _nest(data)
        assert data["advantages"].is_nested
        assert data["values"].is_nested

        unnest_batch_by_mask(data, specs)

        assert not data["advantages"].is_nested
        assert not data["values"].is_nested
        torch.testing.assert_close(data["advantages"], orig_adv)
        torch.testing.assert_close(data["values"], orig_val)
        # response_mask restored
        assert "response_mask" in data
        assert data["response_mask"].shape == (2, 15)

    def test_3d_position_ids_roundtrip(self):
        """Raw 3-D ``(bs, heads, seq_len)`` position_ids round-trips.

        The registered permutation in ``KNOWN_FIELD_PERMUTATIONS``
        fires on the way in (producing the canonical
        ``(bs, seq_len, heads)`` layout for nesting) and its inverse
        fires on the way out, restoring the original shape exactly.
        """
        batch_size = 2
        max_seq_len = 30
        max_response_len = 15
        heads = 4
        prompt_lens, response_lens = [10, 20], [15, 10]

        input_ids = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_seq_len)
        response_mask = torch.zeros(batch_size, max_response_len)
        # Producer layout: (bs, heads, seq_len).
        position_ids = torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(batch_size, heads, -1).clone()

        for i in range(batch_size):
            total = prompt_lens[i] + response_lens[i]
            input_ids[i, :total] = torch.arange(1, total + 1)
            attention_mask[i, :total] = 1
            response_mask[i, : response_lens[i]] = 1

        # Zero out padding positions so round-trip is exact (mask
        # broadcasts across the heads axis at position 1).
        position_ids = position_ids * attention_mask.unsqueeze(1).long()

        prompt_list = [input_ids[i, : prompt_lens[i]] for i in range(batch_size)]
        response_list = [input_ids[i, prompt_lens[i] : prompt_lens[i] + response_lens[i]] for i in range(batch_size)]

        data = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids.clone(),
                "prompts": torch.nested.as_nested_tensor(prompt_list, layout=torch.jagged),
                "responses": torch.nested.as_nested_tensor(response_list, layout=torch.jagged),
            }
        )
        orig_pos = position_ids.clone()

        data, specs = _nest(data)
        assert data["position_ids"].is_nested

        unnest_batch_by_mask(data, specs)
        assert not data["position_ids"].is_nested
        assert data["position_ids"].shape == (batch_size, heads, max_seq_len)
        torch.testing.assert_close(data["position_ids"], orig_pos)

    def test_mixed_fields_roundtrip(self):
        """A batch with seq-level + resp-level + already-nested fields round-trips."""
        data = _make_batch([10, 30], [15, 10], 40, 15)
        attn_mask = data["attention_mask"].bool()
        resp_mask = data["response_mask"].bool()
        # Zero out padding so round-trip is exact
        data["old_log_probs"] = torch.randn(2, 40) * attn_mask.float()
        data["advantages"] = torch.randn(2, 15) * resp_mask.float()
        orig = {k: data[k].clone() for k in ("input_ids", "old_log_probs", "advantages")}

        data, specs = _nest(data)
        unnest_batch_by_mask(data, specs)

        for key, orig_val in orig.items():
            assert not data[key].is_nested, f"{key} should be dense after unnest"
            torch.testing.assert_close(data[key], orig_val)


class TestUnnest:
    """Tests for the library layer: ``prepare_unnest`` + ``unnest``."""

    def test_unnest_preserves_values_at_mask_positions(self):
        """Unnested dense tensor holds nested values at the original mask=True positions."""
        prompt_lens = [10, 20, 5]
        response_lens = [15, 10, 25]
        max_seq_len = 50
        max_response_len = 30

        data = _make_batch(prompt_lens, response_lens, max_seq_len, max_response_len)
        # Capture attention_mask before nest_batch_by_mask pops it.
        attn_mask = data["attention_mask"].clone().bool()
        data, _ = _nest(data)

        ids = data["input_ids"]
        nested = torch.nested.nested_tensor_from_jagged(ids.values().float(), offsets=ids.offsets())

        ctx = prepare_unnest(data)
        dense = unnest(ctx, nested)

        # Dense shape uses RLE-inferred seq_len (max(offset+length)), which
        # for contiguous left-padding equals the largest total_len across rows.
        assert dense.ndim == 2
        assert dense.shape[0] == ctx.batch_size == len(prompt_lens)
        assert dense.shape[1] == ctx.seq_len

        # Every mask=True position carries its original input_id value (as float).
        for i in range(len(prompt_lens)):
            total = prompt_lens[i] + response_lens[i]
            expected = torch.arange(1, total + 1, dtype=torch.float)
            torch.testing.assert_close(dense[i, :total], expected)
            # Pad positions should be zero-filled.
            if total < dense.shape[1]:
                assert torch.all(dense[i, total:] == 0.0)
        # Sanity: dense width never exceeds the original attention_mask width.
        assert dense.shape[1] <= attn_mask.shape[1]

    def test_unnest_with_trailing_dims(self):
        """Trailing feature dims (e.g. top-k) flow through unchanged."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        data, _ = _nest(data)

        ids = data["input_ids"]
        flat_vals = ids.values().float().unsqueeze(-1).expand(-1, 4).contiguous()
        nested = torch.nested.nested_tensor_from_jagged(flat_vals, offsets=ids.offsets())

        dense = unnest(prepare_unnest(data), nested)
        assert dense.ndim == 3
        assert dense.shape[0] == 2
        assert dense.shape[-1] == 4

    def test_ctx_reused_across_tensors(self):
        """A single ``UnnestContext`` can scatter multiple nested tensors."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        data, _ = _nest(data)

        ids = data["input_ids"]
        a = torch.nested.nested_tensor_from_jagged(ids.values().float(), offsets=ids.offsets())
        b = torch.nested.nested_tensor_from_jagged(ids.values().float() * 2.0, offsets=ids.offsets())

        ctx = prepare_unnest(data)
        dense_a = unnest(ctx, a)
        dense_b = unnest(ctx, b)
        # dense_b == 2 * dense_a at every position (zeros at pads hold trivially).
        torch.testing.assert_close(dense_b, dense_a * 2.0)


class TestSliceResponse:
    """Tests for the PPO layer: ``prepare_response_slice`` + ``slice_response``."""

    def test_slice_bounds_match_reference_left_shift(self):
        """Slicer produces the PPO one-token left-shift of each response region."""
        prompt_lens = [10, 25, 5]
        response_lens = [20, 10, 30]
        max_seq_len = 60
        max_response_len = 35

        data = _make_batch(prompt_lens, response_lens, max_seq_len, max_response_len)
        data, _ = _nest(data)

        # Build a dense tensor with known values via the library layer.
        ids = data["input_ids"]
        nested = torch.nested.nested_tensor_from_jagged(ids.values().float(), offsets=ids.offsets())
        dense = unnest(prepare_unnest(data), nested)

        slice_ctx = prepare_response_slice(data)
        sliced = slice_response(slice_ctx, dense)
        assert sliced.shape == (len(prompt_lens), max_response_len)
        assert slice_ctx.max_response_len == max_response_len

        # _make_batch assigns input_ids[i, :total_len] = 1..total_len, so
        # position k carries value (k+1). The PPO left-shift means the
        # slice at response pos t holds the token at (prompt_len + t - 1)
        # zero-indexed → value (prompt_len + t).
        for i in range(len(prompt_lens)):
            r_len = response_lens[i]
            p_len = prompt_lens[i]
            expected = torch.arange(p_len, p_len + r_len, dtype=torch.float)
            torch.testing.assert_close(sliced[i, :r_len], expected)
            if r_len < max_response_len:
                assert torch.all(sliced[i, r_len:] == 0.0)

    def test_slice_preserves_trailing_dims(self):
        """Trailing dims on the dense tensor are preserved by the slicer."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        data, _ = _nest(data)

        ids = data["input_ids"]
        flat_vals = ids.values().float().unsqueeze(-1).expand(-1, 3).contiguous()
        nested = torch.nested.nested_tensor_from_jagged(flat_vals, offsets=ids.offsets())
        dense = unnest(prepare_unnest(data), nested)

        sliced = slice_response(prepare_response_slice(data), dense)
        assert sliced.shape == (2, 15, 3)


class TestResponseExtractionIntegration:
    """End-to-end tests for the two-layer pipeline (``unnest`` + ``slice_response``)."""

    def test_roundtrip_varying_lengths(self):
        """Roundtrip with varied prompt/response lengths produces correct response slices."""
        prompt_lens = [10, 30, 5, 40]
        response_lens = [40, 20, 45, 10]
        max_seq_len = 100
        max_response_len = 50

        data = _make_batch(prompt_lens, response_lens, max_seq_len, max_response_len)
        data, _ = _nest(data)

        ids = data["input_ids"]
        model_output = torch.nested.nested_tensor_from_jagged(ids.values().float(), offsets=ids.offsets())

        unnest_ctx = prepare_unnest(data)
        slice_ctx = prepare_response_slice(data)
        result = slice_response(slice_ctx, unnest(unnest_ctx, model_output))
        assert result.shape == (4, max_response_len)

        # Response values are left-shifted by 1 for log_prob alignment.
        for i in range(4):
            r_len = response_lens[i]
            expected = torch.arange(prompt_lens[i], prompt_lens[i] + r_len, dtype=torch.float)
            torch.testing.assert_close(result[i, :r_len], expected, rtol=1e-5, atol=1e-6)

    def test_roundtrip_uniform_lengths(self):
        """Roundtrip with uniform lengths (no padding needed)."""
        prompt_lens = [32] * 4
        response_lens = [32] * 4

        data = _make_batch(prompt_lens, response_lens, 64, 32)
        data, _ = _nest(data)

        ids = data["input_ids"]
        model_output = torch.nested.nested_tensor_from_jagged(ids.values().float(), offsets=ids.offsets())
        result = slice_response(prepare_response_slice(data), unnest(prepare_unnest(data), model_output))

        assert result.shape == (4, 32)
        for i in range(4):
            expected = torch.arange(32, 64, dtype=torch.float)
            torch.testing.assert_close(result[i], expected, rtol=1e-5, atol=1e-6)

    def test_roundtrip_with_trailing_dims(self):
        """Model output with trailing dimensions (e.g. top-k logits) works."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        data, _ = _nest(data)

        ids = data["input_ids"]
        flat_vals = ids.values().float().unsqueeze(-1).expand(-1, 5)
        model_output = torch.nested.nested_tensor_from_jagged(flat_vals, offsets=ids.offsets())

        result = slice_response(prepare_response_slice(data), unnest(prepare_unnest(data), model_output))
        assert result.shape == (2, 15, 5)


class TestExtractResponseDispatch:
    """Tests for ``extract_response`` — dispatches on batch nesting style.

    Verifies that the same ``extract_response(data, tensor)`` call
    handles new mask-nesting batches, and (with flash_attn) legacy
    batches, so worker loss code stays transparent to
    ``use_mask_nesting``.
    """

    def test_new_path_matches_manual_two_step(self):
        """extract_response on a new-path batch == manual unnest + slice_response."""
        data = _make_batch([10, 25, 5], [20, 10, 30], 60, 35)
        data, _ = _nest(data)

        ids = data["input_ids"]
        nested = torch.nested.nested_tensor_from_jagged(ids.values().float(), offsets=ids.offsets())

        via_dispatch = extract_response(data, nested)
        via_manual = slice_response(prepare_response_slice(data), unnest(prepare_unnest(data), nested))
        torch.testing.assert_close(via_dispatch, via_manual)

    def test_multiple_tensors_new_path(self):
        """Calling extract_response per tensor produces consistent results."""
        data = _make_batch([10, 20, 5], [15, 10, 25], 40, 25)
        data, _ = _nest(data)

        ids = data["input_ids"]
        a = torch.nested.nested_tensor_from_jagged(ids.values().float(), offsets=ids.offsets())
        b = torch.nested.nested_tensor_from_jagged(ids.values().float() * 2.0, offsets=ids.offsets())

        result_a = extract_response(data, a)
        result_b = extract_response(data, b)
        torch.testing.assert_close(result_b, result_a * 2.0)


@requires_flash_attn
class TestExtractResponseDispatchLegacy:
    """Legacy-path coverage for ``extract_response`` (requires flash_attn)."""

    def test_legacy_path_matches_no_padding_2_padding(self):
        """extract_response on a legacy batch == no_padding_2_padding."""
        prompt_lens = [10, 30, 5, 40]
        response_lens = [40, 20, 45, 10]

        data = _make_batch(prompt_lens, response_lens, 100, 50)
        data = left_right_2_no_padding(data)

        ids = data["input_ids"]
        model_output = torch.nested.nested_tensor_from_jagged(ids.values().float(), offsets=ids.offsets())

        via_dispatch = extract_response(data, model_output)
        via_direct = no_padding_2_padding(model_output, data)
        torch.testing.assert_close(via_dispatch, via_direct)

    def test_both_paths_produce_identical_output(self):
        """The same extract_response call on legacy vs new batch → same result."""
        prompt_lens = [10, 25, 5, 40]
        response_lens = [20, 15, 35, 10]

        data_legacy = _make_batch(prompt_lens, response_lens, 75, 35)
        data_new = _make_batch(prompt_lens, response_lens, 75, 35)

        data_legacy = left_right_2_no_padding(data_legacy)
        data_new, _ = _nest(data_new)

        legacy_ids = data_legacy["input_ids"]
        new_ids = data_new["input_ids"]
        legacy_output = torch.nested.nested_tensor_from_jagged(
            legacy_ids.values().float(), offsets=legacy_ids.offsets()
        )
        new_output = torch.nested.nested_tensor_from_jagged(new_ids.values().float(), offsets=new_ids.offsets())

        result_legacy = extract_response(data_legacy, legacy_output)
        result_new = extract_response(data_new, new_output)
        torch.testing.assert_close(result_legacy, result_new)


class TestNestBatchSimple:
    """Tests for the single-step ``nest_batch`` API (build-and-apply)."""

    def test_simple_kwarg_path(self):
        """nest_batch(data, pad_token_id=...) builds specs internally."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        nest_batch_by_mask(data, pad_token_id=_TEST_PAD_TOKEN_ID)
        assert data["input_ids"].is_nested
        assert data["position_ids"].is_nested

    def test_unnest_without_specs_uses_stash(self):
        """unnest_batch() with no specs retrieves them from the stash."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        attn_mask = data["attention_mask"].bool()
        data["position_ids"] = data["position_ids"] * attn_mask.long()
        orig_ids = data["input_ids"].clone()
        orig_pos = data["position_ids"].clone()

        nest_batch_by_mask(data, pad_token_id=_TEST_PAD_TOKEN_ID)
        assert data["input_ids"].is_nested

        unnest_batch_by_mask(data)  # no specs arg
        assert not data["input_ids"].is_nested
        torch.testing.assert_close(data["input_ids"], orig_ids)
        torch.testing.assert_close(data["position_ids"], orig_pos)

    def test_unnest_without_state_is_noop(self):
        """unnest_batch on a never-nested TensorDict is a no-op, not a crash."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        # Don't nest. Calling unnest should return data unchanged.
        result = unnest_batch_by_mask(data)
        assert result is data
        assert not data["input_ids"].is_nested

    def test_field_to_mask_and_pad_custom_field(self):
        """Declarative per-call custom field registration."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        # A fresh field the library has no knowledge of.
        data["my_custom_scores"] = torch.ones(2, 15)
        nest_batch_by_mask(
            data,
            pad_token_id=_TEST_PAD_TOKEN_ID,
            field_to_mask_and_pad={"my_custom_scores": ("response_mask", 0.0)},
        )
        assert data["my_custom_scores"].is_nested

    def test_dynamic_specs_mutation(self):
        """Pre-built specs can be mutated before being passed to nest_batch."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        data["routed_experts"] = torch.randint(0, 8, (2, 30))

        specs = make_mask_nesting_specs(data, pad_token_id=_TEST_PAD_TOKEN_ID)
        # Opt out of nesting routed_experts for this call.
        specs["attention_mask"].data_field_to_pad_value.pop("routed_experts", None)
        nest_batch_by_mask(data, specs)

        assert data["input_ids"].is_nested
        # routed_experts was skipped, so it remains dense.
        assert not data["routed_experts"].is_nested

    def test_conflicting_specs_and_kwargs_raise(self):
        """Passing both pre-built specs and build-from-scratch kwargs raises."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        specs = make_mask_nesting_specs(data, pad_token_id=_TEST_PAD_TOKEN_ID)
        with pytest.raises(TypeError, match="pre-built"):
            nest_batch_by_mask(data, specs, pad_token_id=1)


class TestNestBatchDispatchCompat:
    """Regression tests for the dispatch/collect path (chunk_tensordict → per-chunk consumer).

    When a nested TensorDict is chunked for RPC dispatch to Ray workers,
    each chunk inherits a reference to the stashed ``MaskNestingSpec``
    dict — which carries the **original full-batch** ``mask_shape``.
    Consumers that read ``mask_shape[-1]`` (seq dim) are fine; consumers
    that read or allocate with ``mask_shape[0]`` would see the stale
    full-batch dim instead of the per-chunk dim.
    """

    def _make_nested_batch(self, bs: int):
        max_seq_len, max_response_len = 12, 6
        input_ids = torch.zeros(bs, max_seq_len, dtype=torch.long)
        attention_mask = torch.zeros(bs, max_seq_len)
        response_mask = torch.zeros(bs, max_response_len)
        advantages = torch.randn(bs, max_response_len)
        for i in range(bs):
            total = 8 + (i % 3)
            input_ids[i, :total] = torch.arange(1, total + 1)
            attention_mask[i, :total] = 1
            response_mask[i, : 3 + i % 2] = 1
        advantages = advantages * response_mask
        prompt_list = [input_ids[i, :5] for i in range(bs)]
        response_list = [input_ids[i, 5 : 5 + (3 + i % 2)] for i in range(bs)]
        return TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "advantages": advantages,
                "prompts": torch.nested.as_nested_tensor(prompt_list, layout=torch.jagged),
                "responses": torch.nested.as_nested_tensor(response_list, layout=torch.jagged),
            },
            batch_size=[bs],
        )

    def test_chunked_per_chunk_unnest(self):
        """unnest_batch_by_mask on a per-chunk TensorDict uses local batch dim.

        Guards against a regression where ``MaskNestingSpec.unnest_in_td``
        allocated the mask with the full-batch ``mask_shape``, leaving
        extra rows of zeros on each chunk.
        """
        from verl.utils.tensordict_utils import chunk_tensordict

        data = self._make_nested_batch(bs=4)
        orig_ids = data["input_ids"].clone()
        orig_adv = data["advantages"].clone()

        nest_batch_by_mask(data, pad_token_id=_TEST_PAD_TOKEN_ID)
        chunks = chunk_tensordict(data, chunks=2)

        for i, chunk in enumerate(chunks):
            # Each chunk has its own RLE offsets (2 rows) but the stashed
            # specs still reference the full (4, seq_len) mask_shape.
            unnest_batch_by_mask(chunk)
            assert chunk["input_ids"].shape[0] == 2
            assert chunk["attention_mask"].shape[0] == 2
            assert chunk["advantages"].shape[0] == 2
            expected_ids = orig_ids[i * 2 : (i + 1) * 2]
            expected_adv = orig_adv[i * 2 : (i + 1) * 2]
            torch.testing.assert_close(chunk["input_ids"], expected_ids)
            torch.testing.assert_close(chunk["advantages"], expected_adv)

    def test_chunked_per_chunk_response_extraction(self):
        """unnest + slice_response work on a per-chunk TensorDict.

        ``prepare_response_slice`` reads ``specs[...].mask_shape[-1]``
        for ``max_response_len``; this guards the stale full-batch dim
        in ``mask_shape`` from leaking into shape computations.
        """
        from verl.utils.tensordict_utils import chunk_tensordict

        data = self._make_nested_batch(bs=4)
        nest_batch_by_mask(data, pad_token_id=_TEST_PAD_TOKEN_ID)
        chunks = chunk_tensordict(data, chunks=2)

        for chunk in chunks:
            ids = chunk["input_ids"]
            model_out = torch.nested.nested_tensor_from_jagged(ids.values().float(), offsets=ids.offsets())
            result = slice_response(prepare_response_slice(chunk), unnest(prepare_unnest(chunk), model_out))
            # Shape must be (per-chunk bs, response_len) — NOT (full bs, ...)
            assert result.shape[0] == 2
            assert result.shape[1] == 6  # max_response_len from spec


# ---------------------------------------------------------------------------
# Trainer dispatch helper simulation
# ---------------------------------------------------------------------------

# These tests replicate the logic of
# ``RayPPOTrainer._compress_batch`` / ``_decompress_batch`` inline —
# avoiding the heavy Ray / vLLM imports required to instantiate a real
# trainer. Since ``nest_batch_by_mask`` / ``unnest_batch_by_mask`` now
# handle shape normalisation (and its reversal) internally, the
# trainer helpers are reduced to thin wrappers that branch on
# ``use_mask_nesting``. The sim helpers mirror that simplicity so
# test failures surface drift from the real trainer implementation.


def _sim_compress_batch(data: TensorDict, pad_token_id: int) -> TensorDict:
    """Mirror of ``RayPPOTrainer._compress_batch`` (use_mask_nesting=True)."""
    compress_batch_dtypes(data)
    nest_batch_by_mask(data, pad_token_id=pad_token_id)
    # Legacy worker-side ``loss_mask`` alias — see trainer-side TODO.
    if "response_mask" in data:
        data["loss_mask"] = data["response_mask"]
    return data


def _sim_decompress_batch(data: TensorDict) -> TensorDict:
    """Mirror of ``RayPPOTrainer._decompress_batch`` (use_mask_nesting=True)."""
    unnest_batch_by_mask(data)  # auto-retrieve specs + reverse permutations
    data.pop("loss_mask", None)
    return data


class TestTrainerCompressBatch:
    """Tests for trainer compress/decompress helpers — mostly verifying the library does the heavy lifting."""

    def test_3d_position_ids_full_roundtrip(self):
        """(bs, heads, seq_len) position_ids survive nest → unnest at mask=True positions."""
        bs, heads, max_seq_len, max_response_len = 2, 4, 30, 15
        prompt_lens, response_lens = [10, 20], [15, 10]

        input_ids = torch.zeros(bs, max_seq_len, dtype=torch.long)
        attention_mask = torch.zeros(bs, max_seq_len)
        response_mask = torch.zeros(bs, max_response_len)
        # Original multimodal layout: heads in the middle
        position_ids = torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(bs, heads, -1).clone()

        for i in range(bs):
            total = prompt_lens[i] + response_lens[i]
            input_ids[i, :total] = torch.arange(1, total + 1)
            attention_mask[i, :total] = 1
            response_mask[i, : response_lens[i]] = 1
        # Zero out padding positions so round-trip is exact
        position_ids = position_ids * attention_mask.unsqueeze(1).long()

        prompt_list = [input_ids[i, : prompt_lens[i]] for i in range(bs)]
        response_list = [input_ids[i, prompt_lens[i] : prompt_lens[i] + response_lens[i]] for i in range(bs)]

        data = TensorDict(
            {
                "input_ids": input_ids.clone(),
                "attention_mask": attention_mask.clone(),
                "response_mask": response_mask.clone(),
                "position_ids": position_ids.clone(),
                "prompts": torch.nested.as_nested_tensor(prompt_list, layout=torch.jagged),
                "responses": torch.nested.as_nested_tensor(response_list, layout=torch.jagged),
            }
        )
        orig_input_ids = input_ids.clone()
        orig_pos = position_ids.clone()

        _sim_compress_batch(data, pad_token_id=_TEST_PAD_TOKEN_ID)

        # Mid-state: everything nested, transpose flag stashed
        assert data["input_ids"].is_nested
        assert data["position_ids"].is_nested
        # position_ids nested layout is (valid_len, heads) per sample
        for i in range(bs):
            valid_len = prompt_lens[i] + response_lens[i]
            assert data["position_ids"][i].shape == (valid_len, heads)

        _sim_decompress_batch(data)

        # After unnest: dense tensors restored, position_ids back to (bs, heads, seq_len)
        assert not data["input_ids"].is_nested
        assert not data["position_ids"].is_nested
        assert data["position_ids"].shape == (bs, heads, max_seq_len)
        torch.testing.assert_close(data["input_ids"], orig_input_ids)
        torch.testing.assert_close(data["position_ids"], orig_pos)

    def test_2d_position_ids_full_roundtrip(self):
        """2-D position_ids (single-head) flow: no transpose, regular round-trip."""
        bs, max_seq_len, max_response_len = 2, 30, 15
        data = _make_batch([10, 20], [15, 10], max_seq_len, max_response_len)
        attn_mask = data["attention_mask"].bool()
        data["position_ids"] = data["position_ids"] * attn_mask.long()

        orig_input_ids = data["input_ids"].clone()
        orig_pos = data["position_ids"].clone()

        _sim_compress_batch(data, pad_token_id=_TEST_PAD_TOKEN_ID)
        assert data["input_ids"].is_nested
        assert data["position_ids"].is_nested

        _sim_decompress_batch(data)
        assert not data["input_ids"].is_nested
        assert not data["position_ids"].is_nested
        assert data["position_ids"].shape == (bs, max_seq_len)
        torch.testing.assert_close(data["input_ids"], orig_input_ids)
        torch.testing.assert_close(data["position_ids"], orig_pos)

    def test_decompress_without_compress_is_noop(self):
        """Calling _decompress_batch on a never-nested TensorDict is a no-op."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        orig_ids = data["input_ids"].clone()

        # Skip nest; jump straight to unnest (no stashed state, no flag).
        _sim_decompress_batch(data)

        assert not data["input_ids"].is_nested
        torch.testing.assert_close(data["input_ids"], orig_ids)

    def test_loss_mask_alias_inference_path(self):
        """_compress_batch re-exposes ``loss_mask`` when response_mask isn't popped.

        Legacy convention: some worker loss code reads
        ``data["loss_mask"]`` instead of ``data["response_mask"]``.
        In inference-style flows (``_compute_values`` etc.) the batch
        has no resp-level data fields, so ``response_mask`` is never
        popped during nesting — the trainer then aliases it under
        ``loss_mask`` post-nest.
        """
        data = _make_batch([10, 20], [15, 10], 30, 15)
        # No resp-level data fields → response_mask stays in batch_td.
        _sim_compress_batch(data, pad_token_id=_TEST_PAD_TOKEN_ID)
        assert "loss_mask" in data
        assert "response_mask" in data
        # Both point to the same tensor (alias by reference).
        assert data["loss_mask"] is data["response_mask"]

    def test_loss_mask_alias_dropped_on_decompress(self):
        """_decompress_batch cleans up the ``loss_mask`` alias."""
        data = _make_batch([10, 20], [15, 10], 30, 15)
        _sim_compress_batch(data, pad_token_id=_TEST_PAD_TOKEN_ID)
        assert "loss_mask" in data
        _sim_decompress_batch(data)
        assert "loss_mask" not in data
        assert "response_mask" in data


@requires_flash_attn
class TestNewVsLegacyEquivalence:
    """Cross-validate ``nest_batch`` against the legacy flash_attn path."""

    def test_roundtrip_equivalence(self):
        """Both paths produce identical response extractions."""
        prompt_lens = [10, 30, 5, 40]
        response_lens = [40, 20, 45, 10]
        data_old = _make_batch(prompt_lens, response_lens, 100, 50)
        data_rle = _make_batch(prompt_lens, response_lens, 100, 50)

        data_old = left_right_2_no_padding(data_old)
        data_rle, rle_specs = _nest(data_rle)

        old_ids = data_old["input_ids"]
        old_output = torch.nested.nested_tensor_from_jagged(old_ids.values().float(), offsets=old_ids.offsets())
        rle_ids = data_rle["input_ids"]
        rle_output = torch.nested.nested_tensor_from_jagged(rle_ids.values().float(), offsets=rle_ids.offsets())

        result_old = no_padding_2_padding(old_output, data_old)
        result_rle = slice_response(prepare_response_slice(data_rle), unnest(prepare_unnest(data_rle), rle_output))

        torch.testing.assert_close(result_old, result_rle)
