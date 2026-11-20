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
"""Store the dataset fields of a prompt once in TransferQueue and link its trajectories to them.

A GRPO prompt is rolled out ``n`` times, and every trajectory ``{uid}_{session_id}_{index}`` would
otherwise store its own copy of the dataset fields (``raw_prompt``, ``reward_model``, ...). The rollout
worker puts those fields once under key ``uid`` in ``tq.PUBLIC_PARTITION`` before running the prompt, and
each trajectory links them instead. The first finished session of a multimodal prompt likewise puts its
``multi_modal_inputs`` (e.g. ``pixel_values``) under ``{uid}-mm``, and the sessions whose inputs are equal
link them. A public key is freed once no key links to it any more, which
``release_public_prompts`` sweeps for; it runs where the prompts of the next batch are published,
the one point that is reached after a step cleared the trajectories of the previous one.

The trainer knows none of this: everything here is called from the TransferQueue rollout adapter.
"""

from typing import Any, Iterable

import torch
import transfer_queue as tq
from omegaconf import DictConfig
from tensordict import NonTensorData, TensorDict

PUBLIC_PARTITION = tq.PUBLIC_PARTITION

# Fields of a prompt batch that trajectories do not store, or that only async recovery reads from the prompt key.
_PRIVATE_PROMPT_FIELDS = frozenset({"agent_name", "multi_modal_data", "__rollout_n__", "__do_sample__"})


def prompt_sharing_enabled(config: DictConfig) -> bool:
    """Whether ``rollout.agent.share_prompts`` is on, which needs TransferQueue links on SimpleStorage."""
    storage_backend = config.transfer_queue.get("backend", {}).get("storage_backend", "SimpleStorage")
    share_prompts = config.actor_rollout_ref.rollout.agent.get("share_prompts", True)
    return bool(share_prompts) and storage_backend == "SimpleStorage"


def shared_prompt_fields(batch: TensorDict) -> list[str]:
    """Per-sample fields of a prompt batch that all trajectories of a prompt read unchanged."""
    return [
        key
        for key, value in batch.items()
        if not isinstance(value, NonTensorData) and key not in _PRIVATE_PROMPT_FIELDS
    ]


def public_multi_modal_key(uid: str) -> str:
    """Public key of the ``multi_modal_inputs`` the trajectories of prompt ``uid`` share."""
    return f"{uid}-mm"


def _values_equal(a: Any, b: Any) -> bool:
    if a is b:
        return True
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return False
    return a == b


def multi_modal_inputs_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(_values_equal(value, b[key]) for key, value in a.items())


async def publish_public_prompts(batch: TensorDict) -> list[str]:
    """Put the shared fields of each prompt under its uid in ``PUBLIC_PARTITION``; return the fields.

    Only the prompts that have no public key yet are written: recovering an async checkpoint
    re-dispatches prompts whose keys already exist, and TransferQueue refuses to write a key twice.
    """
    fields = shared_prompt_fields(batch)
    if not fields:
        return fields
    uids = list(batch["uid"])
    present = await tq.async_kv_exists(keys=uids, partition_id=PUBLIC_PARTITION)
    missing = [i for i, uid in enumerate(uids) if uid not in present]
    if missing:
        await tq.async_kv_batch_put(
            keys=[uids[i] for i in missing],
            partition_id=PUBLIC_PARTITION,
            fields=batch.select(*fields)[missing],
        )
    return fields


def release_public_prompts(uids: Iterable[str] = ()) -> None:
    """Free the public prompts no key links to any more.

    ``uids`` are prompt groups that are given up on, so their public prompts are freed even if no
    trajectory ever linked them, e.g. a group whose sessions all failed.
    """
    tq.kv_clear_unlinked_public(stale_keys=_stale_public_keys(uids))


async def async_release_public_prompts(uids: Iterable[str] = ()) -> None:
    """``release_public_prompts`` for a caller that must not block its event loop."""
    await tq.async_kv_clear_unlinked_public(stale_keys=_stale_public_keys(uids))


def _stale_public_keys(uids: Iterable[str]) -> list[str] | None:
    return [key for uid in uids for key in (uid, public_multi_modal_key(uid))] or None
