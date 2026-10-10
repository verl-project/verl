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
"""Engine-wide token ban at SGLang's sampler: the SGLang counterpart of the vLLM
``monkey_patch_compute_logits`` mask.

``Sampler._preprocess_logits`` runs on the next-token logits before sampling, before the
sampled-token logprobs, and in ``compute_logprobs_only``. It runs after CUDA-graph replay, and
its logits are already TP-gathered and sliced to ``config.vocab_size`` by SGLang's
LogitsProcessor, so every rank masks the same columns. Prompt logprobs are not touched.

SGLang spawns its scheduler processes, so a patch applied in the rollout server process is not
inherited; ``run_scheduler_process_with_token_ban`` installs it inside each scheduler.
"""

import logging
from collections.abc import Iterable
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_PATCH_ATTR = "_verl_token_ban"


def install_sampler_token_ban(banned_token_ids: Iterable[int], vocab_size: Optional[int] = None) -> None:
    """Mask ``banned_token_ids`` and every column ``>= vocab_size`` on all requests this process samples.

    Re-installing replaces the previous ban rather than stacking on it.
    """
    from sglang.srt.layers.sampler import Sampler

    installed = getattr(Sampler, _PATCH_ATTR, None)
    original = installed["original"] if installed else getattr(Sampler, "_preprocess_logits", None)
    if original is None:
        # The ban guards correctness: a silently missing hook would let banned ids be sampled.
        raise RuntimeError(
            "sglang.srt.layers.sampler.Sampler._preprocess_logits not found; "
            "cannot install the rollout token ban on this SGLang version."
        )

    banned = sorted({int(token_id) for token_id in banned_token_ids})
    # One index per (device, logits width), built on the first step and reused by every later one.
    index_cache: dict[tuple, torch.Tensor] = {}

    def _preprocess_logits(self, logits, sampling_info):
        if banned:
            key = (logits.device, logits.shape[-1])
            index = index_cache.get(key)
            if index is None:
                index = torch.tensor(
                    [i for i in banned if i < logits.shape[-1]], dtype=torch.long, device=logits.device
                )
                index_cache[key] = index
            if index.numel():
                logits.index_fill_(-1, index, float("-inf"))
        if vocab_size is not None:
            logits[..., vocab_size:] = float("-inf")
        return original(self, logits, sampling_info)

    Sampler._preprocess_logits = _preprocess_logits
    setattr(
        Sampler,
        _PATCH_ATTR,
        {"original": original, "banned_token_ids": banned, "vocab_size": vocab_size, "index_cache": index_cache},
    )
    logger.info("SGLang sampler token ban installed: %d ids, vocab_size=%s", len(banned), vocab_size)


def run_scheduler_process_with_token_ban(
    *args, verl_banned_token_ids: Iterable[int] = (), verl_vocab_size: Optional[int] = None, **kwargs
):
    """SGLang ``run_scheduler_process`` with the token ban installed first.

    Bind the ``verl_*`` arguments with ``functools.partial``: SGLang starts this target with
    ``multiprocessing`` spawn, so it has to stay a picklable module-level function.
    """
    import sglang.srt.entrypoints.engine

    install_sampler_token_ban(verl_banned_token_ids, verl_vocab_size)
    return sglang.srt.entrypoints.engine.run_scheduler_process(*args, **kwargs)
