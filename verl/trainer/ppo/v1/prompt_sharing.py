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
otherwise store its own copy of the dataset fields (``raw_prompt``, ``reward_model``, ...). The trainer
puts those fields once under key ``uid`` in ``tq.PUBLIC_PARTITION`` before dispatching the prompt, and
each trajectory links them instead. The first finished session of a multimodal prompt likewise puts its
``multi_modal_inputs`` (e.g. ``pixel_values``) under ``{uid}-mm``, and the sessions whose inputs are equal
link them. A public key is freed once no key links to it any more, so the
trainer only has to call ``release_public_prompts`` after clearing trajectories or prompt groups.
"""

from typing import Iterable

import torch
import transfer_queue as tq
from omegaconf import DictConfig
from tensordict import NonTensorData, TensorDict

PUBLIC_PARTITION = getattr(tq, "PUBLIC_PARTITION", "__public__")

# Fields of a prompt batch that trajectories do not store, or that only async recovery reads from the prompt key.
_PRIVATE_PROMPT_FIELDS = frozenset({"agent_name", "multi_modal_data", "__rollout_n__", "__do_sample__"})


def prompt_sharing_requested(tq_config: DictConfig) -> bool:
    """Whether ``transfer_queue.share_prompts`` asks the trajectories of a prompt to share preprocessing and storage."""
    return bool(tq_config.get("share_prompts", True))


def prompt_sharing_enabled(tq_config: DictConfig) -> bool:
    """Whether to store prompts once, which needs links (TransferQueue > 0.1.11) on the configured storage backend."""
    storage_backend = tq_config.get("backend", {}).get("storage_backend", "SimpleStorage")
    return prompt_sharing_requested(tq_config) and hasattr(tq, "kv_link") and storage_backend == "SimpleStorage"


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


def multi_modal_inputs_equal(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    for key, value in a.items():
        other = b[key]
        if value is other:
            continue
        if isinstance(value, torch.Tensor) or isinstance(other, torch.Tensor):
            if not (
                isinstance(value, torch.Tensor)
                and isinstance(other, torch.Tensor)
                and value.dtype == other.dtype
                and torch.equal(value, other)
            ):
                return False
        elif value != other:
            return False
    return True


def put_public_prompts(batch: TensorDict) -> list[str]:
    """Put the shared fields of each prompt under its uid in ``PUBLIC_PARTITION``; return the fields."""
    fields = shared_prompt_fields(batch)
    if fields:
        tq.kv_batch_put(keys=list(batch["uid"]), partition_id=PUBLIC_PARTITION, fields=batch.select(*fields))
    return fields


def release_public_prompts(uids: Iterable[str] = ()) -> None:
    """Free the public prompts no key links to any more.

    ``uids`` are prompt groups that are given up on, so their public prompts are freed even if no
    trajectory ever linked them, e.g. a group whose sessions all failed.
    """
    if hasattr(tq, "kv_clear_unlinked_public"):
        stale_keys = [key for uid in uids for key in (uid, public_multi_modal_key(uid))]
        tq.kv_clear_unlinked_public(stale_keys=stale_keys or None)
