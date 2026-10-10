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

"""Cross-step prefix KV cache for the reference (ref) model.

Exploits two facts together:

1. The ref model's weights are frozen for the whole training run, so a prompt's
   KV cache is valid across training steps (it never goes stale).
2. HuggingFace models natively support ``use_cache=True`` / ``past_key_values``
   (the prefill mechanism used by generation), which works with **every**
   attention backend (eager, sdpa, flash_attention_2). No attention surgery.

So instead of the design doc's Phase-2 "KV-injection attention hook" (which
binds to a specific attention impl and breaks flash-attn), we cache the prompt's
``past_key_values`` produced by a single prefill forward, and reuse it for every
response that shares that prompt. On a cache hit the prompt's transformer layers
are not recomputed at all.

Caveats:
    - ``DynamicCache`` is mutated in-place by a forward; each use clones the
      cache via :func:`_clone_cache` — a *shallow* clone (shares KV tensors,
      appends to new layer objects, O(num_layers), no KV copy) — so the prompt
      cache stays pristine for reuse. Essential for long prompts on large
      models where a deepcopy would be O(prompt_len) memory per response.
    - Cross-step reuse applies to a prompt that recurs across steps. For GRPO
      this means the *constant* portion of the prompt (e.g. a long system
      prompt); per-sample user input is not reused. This module caches by full
      (de-padded) prompt hash; prefix-only caching is a natural extension.
    - FSDP + ``use_cache=True`` interaction must be validated on GPU/NPU before
      the forward_step integration is used in real training.
"""

from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict
from typing import Any, Optional

import torch

from verl.utils.torch_functional import logprobs_from_logits


def _clone_cache(cache: Any) -> Any:
    """Return a forwardable clone of ``cache`` that shares the underlying KV
    tensors (read-only in attention) but appends to *new* layer objects, so the
    original prompt cache stays pristine for reuse across responses/steps.

    For transformers 5.x ``DynamicCache`` (``layers`` list of ``DynamicLayer``
    with ``keys``/``values``/``is_initialized``), this is O(num_layers) — no KV
    tensor copy — which is essential for long prompts on large models. Falls
    back to ``copy.deepcopy`` on other cache shapes.
    """
    layers = getattr(cache, "layers", None)
    can_shallow = bool(layers) and all(hasattr(lyr, "is_initialized") for lyr in layers)
    if can_shallow:
        fresh = type(cache)()
        for lyr in layers:
            nl = type(lyr)()
            nl.keys = lyr.keys
            nl.values = lyr.values
            nl.is_initialized = True  # so update() concatenates instead of resetting
            fresh.layers.append(nl)
        return fresh
    return copy.deepcopy(cache)


class RefPrefixKVCache:
    """LRU cache of prompt ``past_key_values`` keyed by de-padded prompt hash.

    Each entry stores a pristine ``past_key_values`` (never forwarded directly,
    only deep-copied per use), the prompt's last-position logits (predicts the
    first response token), and the prompt length. Because the ref model is
    frozen, entries never go stale and can persist for the whole run.
    """

    def __init__(self, max_entries: int = 64, max_total_tokens: int = 1_000_000):
        self._cache: OrderedDict[str, tuple[Any, torch.Tensor, int]] = OrderedDict()
        self._max_entries = max_entries
        self._max_total_tokens = max_total_tokens
        self._total_cached_tokens = 0
        self._hit_count = 0
        self._miss_count = 0

    @staticmethod
    def compute_hash(prefix_ids: torch.Tensor) -> str:
        return hashlib.md5(prefix_ids.cpu().numpy().tobytes()).hexdigest()

    def get(self, key: str) -> Optional[tuple[Any, torch.Tensor, int]]:
        if key in self._cache:
            self._hit_count += 1
            value = self._cache.pop(key)
            self._cache[key] = value  # LRU: move to most-recent
            return value
        self._miss_count += 1
        return None

    def put(self, key: str, past_kv: Any, last_logit: torch.Tensor, prompt_len: int) -> None:
        while (
            len(self._cache) >= self._max_entries or self._total_cached_tokens + prompt_len > self._max_total_tokens
        ) and self._cache:
            _, (_, _, evicted_len) = self._cache.popitem(last=False)
            self._total_cached_tokens -= evicted_len
        self._cache[key] = (past_kv, last_logit, prompt_len)
        self._total_cached_tokens += prompt_len

    def stats(self) -> dict:
        total = self._hit_count + self._miss_count
        return {
            "entries": len(self._cache),
            "total_cached_tokens": self._total_cached_tokens,
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": self._hit_count / total if total > 0 else 0.0,
        }

    def clear(self) -> None:
        self._cache.clear()
        self._total_cached_tokens = 0
        self._hit_count = 0
        self._miss_count = 0


def forward_ref_with_prefix_cache(
    model: Any,
    prompts: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    uids: Any,
    pad_token_id: int,
    cache: RefPrefixKVCache,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Ref forward that reuses a cached prompt ``past_key_values`` across steps.

    For each uid group (same prompt), look up the prompt's cached prefill KV. On
    a hit, skip the prompt forward entirely and only forward the response tokens
    with the cloned cache. On a miss, prefill the prompt with ``use_cache=True``,
    store the cache, then forward the responses.

    Args:
        model: HF ``CausalLM`` (any attention backend that supports use_cache).
        prompts: ``(bsz, max_prompt_len)`` padded prompt ids.
        responses: ``(bsz, max_response_len)`` padded response ids.
        response_mask: ``(bsz, max_response_len)`` valid-response mask.
        uids: length-``bsz`` group ids; equal uids must be consecutive.
        pad_token_id: padding id used to de-pad the prompt.
        cache: the persistent :class:`RefPrefixKVCache`.
        temperature: logits are divided by this before log-softmax.

    Returns:
        Jagged nested tensor of per-sample log-probs (one ``(resp_len_i,)`` per
        sample), matching the standard ref forward output layout.
    """
    bsz = responses.size(0)
    device = responses.device

    starts = [0]
    for i in range(1, bsz):
        if uids[i] != uids[i - 1]:
            starts.append(i)
    starts.append(bsz)

    all_log_probs: list[torch.Tensor] = []

    with torch.no_grad():
        for g in range(len(starts) - 1):
            s, e = starts[g], starts[g + 1]
            prefix_full = prompts[s]
            prefix_ids = prefix_full[prefix_full.ne(pad_token_id)]
            prefix_len = prefix_ids.size(0)
            key = RefPrefixKVCache.compute_hash(prefix_ids)

            cached = cache.get(key)
            if cached is not None:
                past_kv, last_logit, _ = cached
            else:
                out = model(input_ids=prefix_ids.unsqueeze(0), use_cache=True)
                # Store the freshly-built cache; it is only deep-copied per use,
                # never forwarded directly, so it stays pristine.
                past_kv = out.past_key_values
                last_logit = out.logits[0, -1]
                cache.put(key, past_kv, last_logit, prefix_len)

            suffixes = [responses[i, : int(response_mask[i].sum())] for i in range(s, e)]
            for suf in suffixes:
                sl = suf.size(0)
                pos = torch.arange(prefix_len, prefix_len + sl, device=device).unsqueeze(0)
                resp_out = model(
                    input_ids=suf.unsqueeze(0),
                    past_key_values=_clone_cache(past_kv),
                    position_ids=pos,
                    use_cache=True,
                )
                resp_logits = resp_out.logits[0]
                if temperature != 1.0:
                    last_logit_t = last_logit / temperature
                    resp_logits = resp_logits / temperature
                else:
                    last_logit_t = last_logit
                first = logprobs_from_logits(last_logit_t.unsqueeze(0), suf[:1])
                rest = logprobs_from_logits(resp_logits[: sl - 1], suf[1:]) if sl > 1 else torch.empty(0, device=device)
                all_log_probs.append(torch.cat([first, rest]))

    return torch.nested.as_nested_tensor(all_log_probs, layout=torch.jagged)


def forward_ref_with_prefix_len_cache(
    model: Any,
    prompts: torch.Tensor,
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    pad_token_id: int,
    cache: RefPrefixKVCache,
    prefix_len: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Ref forward caching a *shared prefix* (e.g. a long system prompt) across
    samples that each have a different suffix (user input) + response.

    Unlike :func:`forward_ref_with_prefix_cache` (which caches the full prompt,
    reused only when the exact prompt recurs), this caches the first
    ``prefix_len`` tokens of the prompt — the constant prefix shared by every
    sample. On a cache hit the prefix's transformer layers are skipped, and only
    each sample's [suffix + response] is forwarded with the cached prefix KV.

    The response log-probs come entirely from the suffix forward: response[0] is
    predicted by the suffix's last token (the user-input's last token), and
    response[j>=1] by response[j-1] — no need for the prefix's last logit.

    Args:
        prefix_len: number of leading prompt tokens to cache as the shared prefix.
    """
    bsz = responses.size(0)
    device = responses.device

    # The shared prefix is the first prefix_len (de-padded) tokens of any prompt;
    # all prompts are assumed to share it.
    first_prompt = prompts[0][prompts[0].ne(pad_token_id)]
    eff_prefix_len = min(prefix_len, first_prompt.size(0))
    prefix_ids = first_prompt[:eff_prefix_len]

    with torch.no_grad():
        key = RefPrefixKVCache.compute_hash(prefix_ids)
        cached = cache.get(key)
        if cached is not None:
            prefix_kv, _, _ = cached
        else:
            out = model(input_ids=prefix_ids.unsqueeze(0), use_cache=True)
            prefix_kv = out.past_key_values
            cache.put(key, prefix_kv, torch.zeros(1, device=device), eff_prefix_len)

        all_log_probs: list[torch.Tensor] = []
        for i in range(bsz):
            full = prompts[i][prompts[i].ne(pad_token_id)]
            suffix = full[eff_prefix_len:]
            resp = responses[i][: int(response_mask[i].sum())]
            sl = suffix.size(0)
            rl = resp.size(0)
            seq = torch.cat([suffix, resp]).unsqueeze(0)
            pos = torch.arange(eff_prefix_len, eff_prefix_len + sl + rl, device=device).unsqueeze(0)
            o = model(
                input_ids=seq,
                past_key_values=_clone_cache(prefix_kv),
                position_ids=pos,
                use_cache=True,
            )
            logits = o.logits[0]
            if temperature != 1.0:
                logits = logits / temperature
            # response[0..rl-1] predicted by logits at suffix positions [sl-1 .. sl+rl-2]
            resp_logits = logits[sl - 1 : sl + rl - 1] if rl > 0 else logits[:0]
            all_log_probs.append(logprobs_from_logits(resp_logits, resp))

    return torch.nested.as_nested_tensor(all_log_probs, layout=torch.jagged)
