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
"""CPU tests for ``verl.workers.utils.prefix_packing``.

The model tests run a tiny fp32 Qwen2 on the packed row through an SDPA attention with a dense tree mask.
"""

import random

import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.workers.utils.padding import get_prompt_lens
from verl.workers.utils.prefix_packing import (
    PrefixPackingAttnMeta,
    build_prefix_packing_plan,
    find_prefix_groups,
    packed_group_lens,
    split_groups_to_budget,
)


def _nested(rows: list[list[int]]) -> torch.Tensor:
    return torch.nested.as_nested_tensor([torch.tensor(r, dtype=torch.int64) for r in rows], layout=torch.jagged)


def _empty_nested() -> torch.Tensor:
    return torch.nested.nested_tensor_from_jagged(
        torch.empty(0, dtype=torch.int64), offsets=torch.zeros(1, dtype=torch.int64)
    )


def _make_batch(spec, vocab: int = 97, seed: int = 0):
    """``spec`` is a list of ``(n, prompt_len, resp_lens)``, one entry per shared prompt."""
    g = torch.Generator().manual_seed(seed)
    rows, prompt_lens, resp_lens = [], [], []
    for n, p, resps in spec:
        assert len(resps) == n
        prompt = torch.randint(0, vocab, (p,), generator=g).tolist()
        for r in resps:
            rows.append(prompt + torch.randint(0, vocab, (r,), generator=g).tolist())
            prompt_lens.append(p)
            resp_lens.append(r)
    return rows, torch.tensor(prompt_lens, dtype=torch.int64), resp_lens


def _positions(rows) -> torch.Tensor:
    return torch.cat([torch.arange(len(r)) for r in rows])


def _offsets(rows) -> list[int]:
    out = [0]
    for r in rows:
        out.append(out[-1] + len(r))
    return out


def _random_spec(rng: random.Random):
    spec = []
    for _ in range(rng.randint(1, 4)):
        n = rng.randint(1, 4)
        spec.append((n, rng.randint(2, 6), [rng.randint(1, 5) for _ in range(n)]))
    spec.append((1, rng.randint(1, 6), [rng.randint(1, 5)]))
    return spec


def _dense_tree_mask(meta: PrefixPackingAttnMeta) -> torch.Tensor:
    """Boolean ``(num_rows, num_rows)`` mask with bottom-right causal alignment per segment."""
    mask = torch.zeros(meta.num_rows, meta.num_rows, dtype=torch.bool)
    cu_q, cu_k = meta.cu_seqlens_q.tolist(), meta.cu_seqlens_k.tolist()
    for s in range(len(cu_q) - 1):
        keys = meta.kv_index[cu_k[s] : cu_k[s + 1]]
        num_q, num_k = cu_q[s + 1] - cu_q[s], cu_k[s + 1] - cu_k[s]
        for i in range(num_q):
            mask[cu_q[s] + i, keys[: num_k - num_q + i + 1]] = True
    return mask


class TestFindPrefixGroups:
    def test_equal_adjacent_prompts_group(self):
        rows = [[1, 2, 3, 10], [1, 2, 3, 11, 12], [1, 2, 3, 13]]
        assert find_prefix_groups(_nested(rows), torch.tensor([3, 3, 3])) == ([3], [4, 5, 4], [3, 3, 3])

    def test_different_tokens_split(self):
        rows = [[1, 2, 3, 10], [1, 2, 4, 10]]
        assert find_prefix_groups(_nested(rows), torch.tensor([3, 3]))[0] == [1, 1]

    def test_different_prompt_len_split(self):
        rows = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        assert find_prefix_groups(_nested(rows), torch.tensor([3, 4]))[0] == [1, 1]

    def test_prompt_len_one_never_grouped(self):
        rows = [[7, 1], [7, 2], [7, 3]]
        assert find_prefix_groups(_nested(rows), torch.tensor([1, 1, 1]))[0] == [1, 1, 1]

    def test_only_adjacent_samples_group(self):
        a, b = [1, 2, 3], [4, 5, 6]
        rows = [a + [10], a + [11], b + [12], a + [13], a + [14, 15]]
        assert find_prefix_groups(_nested(rows), torch.tensor([3] * 5))[0] == [2, 1, 2]

    def test_trailing_duplicates_join_last_group(self):
        # Auto-padding repeats the last sample to fill the batch.
        a, b = [1, 2, 3], [4, 5, 6]
        rows = [a + [10], b + [11], b + [12], b + [12], b + [12]]
        assert find_prefix_groups(_nested(rows), torch.tensor([3] * 5))[0] == [1, 4]

    def test_empty(self):
        assert find_prefix_groups(_empty_nested(), torch.empty(0, dtype=torch.int64)) == ([], [], [])


class TestGetPromptLens:
    def test_padded_prompts(self):
        prompts = torch.tensor([[0, 0, 5, 6], [7, 8, 9, 1]])
        attention_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0, 0]])
        td = TensorDict({"prompts": prompts, "attention_mask": attention_mask}, batch_size=[2])
        assert get_prompt_lens(td).tolist() == [2, 4]

    def test_nested_prompts(self):
        td = TensorDict({"prompts": _nested([[1, 2, 3], [4], [5, 6]])}, batch_size=[3])
        assert get_prompt_lens(td).tolist() == [3, 1, 2]


class TestBuildPlan:
    SPEC = [(3, 4, [2, 3, 1]), (1, 5, [2]), (2, 3, [4, 2]), (1, 1, [3])]

    def _plan(self):
        rows, prompt_lens, _ = _make_batch(self.SPEC)
        ids = _nested(rows)
        return rows, prompt_lens, ids, build_prefix_packing_plan(ids, prompt_lens)

    def test_round_trip(self):
        rows, _, ids, plan = self._plan()
        flat, pos = ids.values(), _positions(rows)
        assert torch.equal(flat[plan.pack_index][plan.unpack_index], flat)
        assert torch.equal(pos[plan.pack_index][plan.unpack_index], pos)

    def test_layout(self):
        rows, prompt_lens, ids, plan = self._plan()
        group_sizes = find_prefix_groups(ids, prompt_lens)[0]
        assert group_sizes == [3, 1, 2, 1]
        expected, i = [], 0
        for size in group_sizes:
            trunk_len = int(prompt_lens[i]) - 1 if size > 1 else 0
            expected += rows[i][:trunk_len]
            for m in range(i, i + size):
                expected += rows[m][trunk_len:]
            i += size
        assert ids.values()[plan.pack_index].tolist() == expected
        assert plan.num_packed == plan.attn.num_rows == len(expected)
        assert plan.num_unpacked == sum(len(r) for r in rows)

    def test_attn_metadata(self):
        _, _, _, plan = self._plan()
        meta = plan.attn
        assert meta.cu_seqlens_q.dtype == meta.cu_seqlens_k.dtype == torch.int32
        assert meta.kv_index.numel() == meta.cu_seqlens_k[-1]
        assert meta.max_seqlen_q == meta.cu_seqlens_q.diff().max()
        assert meta.max_seqlen_k == meta.cu_seqlens_k.diff().max()

    def test_empty_batch(self):
        plan = build_prefix_packing_plan(_empty_nested(), torch.empty(0, dtype=torch.int64))
        assert plan.num_packed == plan.num_unpacked == plan.attn.num_rows == 0
        assert plan.pack_index.numel() == plan.unpack_index.numel() == 0
        assert plan.attn.cu_seqlens_q.tolist() == [0]

    def test_with_tail_pad(self):
        _, _, _, plan = self._plan()
        meta = plan.attn
        meta.counter.value = 5
        assert meta.with_tail_pad(0) is meta
        padded = meta.with_tail_pad(3)
        n = plan.num_packed
        assert padded.num_rows == n + 3 and padded.layers_seen == 0
        assert padded.cu_seqlens_q.tolist()[-2:] == [n, n + 3]
        assert padded.kv_index[-3:].tolist() == [n, n + 1, n + 2]
        assert meta.num_rows == n and meta.layers_seen == 5


@pytest.mark.parametrize("seed", range(4))
def test_rolled_labels_on_read_rows(seed):
    rows, prompt_lens, resp_lens = _make_batch(_random_spec(random.Random(seed)), seed=seed)
    ids = _nested(rows)
    plan = build_prefix_packing_plan(ids, prompt_lens)
    rolled = torch.roll(ids.values()[plan.pack_index], -1)
    offs = _offsets(rows)
    for s, row in enumerate(rows):
        p = int(prompt_lens[s])
        for j in range(resp_lens[s]):
            assert rolled[plan.unpack_index[offs[s] + p - 1 + j]] == row[p + j], (s, j)


def _brute_force_mask(rows, prompt_lens, group_sizes, pad: int) -> torch.Tensor:
    allowed, cursor, i = [], 0, 0
    for size in group_sizes:
        trunk_len = int(prompt_lens[i]) - 1 if size > 1 else 0
        trunk = list(range(cursor, cursor + trunk_len))
        allowed += [trunk[: k + 1] for k in range(trunk_len)]
        cursor += trunk_len
        for m in range(i, i + size):
            branch_len = len(rows[m]) - trunk_len
            allowed += [trunk + list(range(cursor, cursor + k + 1)) for k in range(branch_len)]
            cursor += branch_len
        i += size
    allowed += [list(range(cursor, cursor + k + 1)) for k in range(pad)]
    mask = torch.zeros(len(allowed), len(allowed), dtype=torch.bool)
    for q, keys in enumerate(allowed):
        mask[q, keys] = True
    return mask


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("pad", [0, 3])
def test_dense_mask_matches_brute_force(seed, pad):
    rows, prompt_lens, _ = _make_batch(_random_spec(random.Random(100 + seed)), seed=seed)
    ids = _nested(rows)
    meta = build_prefix_packing_plan(ids, prompt_lens).attn.with_tail_pad(pad)
    expected = _brute_force_mask(rows, prompt_lens, find_prefix_groups(ids, prompt_lens)[0], pad)
    assert torch.equal(_dense_tree_mask(meta), expected)


class TestBudget:
    def test_packed_group_lens(self):
        seq_lens = [6, 7, 5, 9, 4, 6, 3, 3]
        prompt_lens = [4, 4, 4, 5, 3, 3, 1, 1]
        assert packed_group_lens([3, 1, 2, 2], seq_lens, prompt_lens) == [12, 9, 8, 6]

    def test_packed_group_lens_matches_plan(self):
        rows, prompt_lens, _ = _make_batch(_random_spec(random.Random(7)), seed=7)
        ids = _nested(rows)
        out = packed_group_lens(*find_prefix_groups(ids, prompt_lens))
        assert sum(out) == build_prefix_packing_plan(ids, prompt_lens).num_packed

    def test_split_over_budget(self):
        seq_lens, prompt_lens = [8, 9, 7, 10, 8], [5] * 5
        out = split_groups_to_budget([5], seq_lens, prompt_lens, 13)
        assert out == [2, 2, 1]
        assert all(x <= 13 for x in packed_group_lens(out, seq_lens, prompt_lens))

    def test_over_long_sample_is_alone(self):
        assert split_groups_to_budget([3], [20, 5, 5], [3, 3, 3], 10) == [1, 2]

    def test_singletons_and_short_prompts(self):
        assert split_groups_to_budget([1, 1, 3], [50, 4, 3, 3, 3], [2, 2, 1, 1, 1], 5) == [1, 1, 1, 1, 1]


_ATTN_NAME = "verl_test_tree_sdpa"
ORACLE_SPEC = [(3, 5, [2, 5, 3]), (1, 4, [4]), (2, 6, [3, 2])]


def _tree_sdpa(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
    meta: PrefixPackingAttnMeta = kwargs["prefix_packing"]
    groups = query.shape[1] // key.shape[1]
    out = F.scaled_dot_product_attention(
        query,
        key.repeat_interleave(groups, dim=1),
        value.repeat_interleave(groups, dim=1),
        attn_mask=_dense_tree_mask(meta).to(query.device),
        dropout_p=dropout,
        scale=scaling,
    )
    meta.record_layer()
    return out.transpose(1, 2).contiguous(), None


def _tiny_model(cls, **extra):
    from transformers import AttentionInterface, Qwen2Config

    AttentionInterface.register(_ATTN_NAME, _tree_sdpa)
    config = Qwen2Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        vocab_size=97,
        max_position_embeddings=256,
        attn_implementation="sdpa",
        **extra,
    )
    torch.manual_seed(0)
    return cls(config).to(torch.float32).eval()


def _read_rows(rows, prompt_lens, resp_lens) -> torch.Tensor:
    """Rows ``no_padding_2_padding`` reads: positions ``P-1 .. L-2`` of each sample."""
    offs = _offsets(rows)
    out = []
    for s in range(len(rows)):
        start = offs[s] + int(prompt_lens[s]) - 1
        out += range(start, start + resp_lens[s])
    return torch.tensor(out, dtype=torch.int64)


def _logp_entropy(logits, labels):
    logp_all = F.log_softmax(logits, dim=-1)
    logp = logp_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return logp, -(logp_all.exp() * logp_all).sum(-1)


class TestModelOracle:
    def _setup(self):
        rows, prompt_lens, resp_lens = _make_batch(ORACLE_SPEC, seed=42)
        ids = _nested(rows)
        plan = build_prefix_packing_plan(ids, prompt_lens)
        assert plan.num_packed < plan.num_unpacked
        return rows, ids, plan, _read_rows(rows, prompt_lens, resp_lens)

    @staticmethod
    def _unpacked(model, rows, temps=None):
        model.config._attn_implementation = "sdpa"
        outs = []
        for s, row in enumerate(rows):
            x = torch.tensor(row).unsqueeze(0)
            outs.append(model(input_ids=x, position_ids=torch.arange(len(row)).unsqueeze(0)).logits[0])
            if temps is not None:
                outs[-1] = outs[-1] / temps[s]
        return torch.cat(outs)

    @staticmethod
    def _packed(model, rows, ids, plan):
        model.config._attn_implementation = _ATTN_NAME
        plan.attn.counter.value = 0
        # A dict attention_mask is taken as precomputed, so transformers skips mask creation.
        out = model(
            input_ids=ids.values()[plan.pack_index].unsqueeze(0),
            position_ids=_positions(rows)[plan.pack_index].unsqueeze(0),
            attention_mask={"full_attention": None},
            prefix_packing=plan.attn,
        ).logits[0]
        assert plan.attn.layers_seen == model.config.num_hidden_layers
        return out

    def test_logprobs_entropy_and_grads(self):
        from transformers import Qwen2ForCausalLM

        rows, ids, plan, read = self._setup()
        model = _tiny_model(Qwen2ForCausalLM)
        flat = ids.values()
        weights = torch.randn(read.numel(), generator=torch.Generator().manual_seed(1))

        ref_logp, ref_ent = _logp_entropy(self._unpacked(model, rows), torch.roll(flat, -1))
        (weights * ref_logp[read]).sum().backward()
        ref_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
        model.zero_grad()

        packed_ids = flat[plan.pack_index]
        logp, ent = _logp_entropy(self._packed(model, rows, ids, plan), torch.roll(packed_ids, -1))
        logp, ent = logp[plan.unpack_index][read], ent[plan.unpack_index][read]
        torch.testing.assert_close(logp, ref_logp[read].detach(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ent, ref_ent[read].detach(), atol=1e-5, rtol=1e-5)
        (weights * logp).sum().backward()
        got_grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
        assert got_grads.keys() == ref_grads.keys() and ref_grads
        for name, grad in ref_grads.items():
            torch.testing.assert_close(got_grads[name], grad, atol=1e-5, rtol=1e-4, msg=name)

    @torch.no_grad()
    def test_logprobs_per_sample_temperature(self):
        from transformers import Qwen2ForCausalLM

        rows, ids, plan, read = self._setup()
        model = _tiny_model(Qwen2ForCausalLM)
        flat = ids.values()
        temps = [0.5, 1.0, 2.0, 0.7, 1.5, 3.0]
        temps_flat = torch.cat([torch.full((len(r),), t) for r, t in zip(rows, temps, strict=True)])

        ref_logp, _ = _logp_entropy(self._unpacked(model, rows, temps), torch.roll(flat, -1))
        logits = self._packed(model, rows, ids, plan) / temps_flat[plan.pack_index].unsqueeze(-1)
        logp, _ = _logp_entropy(logits, torch.roll(flat[plan.pack_index], -1))
        torch.testing.assert_close(logp[plan.unpack_index][read], ref_logp[read], atol=1e-5, rtol=1e-5)

    @torch.no_grad()
    def test_values(self):
        from transformers import Qwen2ForTokenClassification

        rows, ids, plan, read = self._setup()
        model = _tiny_model(Qwen2ForTokenClassification, num_labels=1)
        ref = self._unpacked(model, rows).squeeze(-1)
        got = self._packed(model, rows, ids, plan).squeeze(-1)
        torch.testing.assert_close(got[plan.unpack_index][read], ref[read], atol=1e-5, rtol=1e-5)


def test_layer_counter_shared_with_fsdp_kwarg_copy():
    """FSDP copies dataclass kwargs with ``dataclasses.replace``; the copy must update the same counter."""
    from torch.distributed.utils import _apply_to_tensors

    meta = build_prefix_packing_plan(_nested([[1, 2, 3, 4], [1, 2, 3, 5, 6]]), torch.tensor([3, 3])).attn
    copied = _apply_to_tensors(lambda t: t, meta)
    assert copied is not meta
    copied.record_layer()
    copied.record_layer()
    assert meta.layers_seen == 2
    assert meta.with_tail_pad(4).layers_seen == 0
