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
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from verl.utils.skip.base_skip import BaseSkip, SkipAction, register_skip

logger = logging.getLogger(__name__)


class _CacheLayoutMismatch(ValueError):
    """The selected dump cannot satisfy the current rollout request."""


def _config_trajectories(
    old_keys: list[str], data: TensorDict, expected_sessions: set[str]
) -> Iterator[tuple[str, str, str]]:
    """Read keys and identity columns together from their original inner batch."""
    saved_uids = data.get("uid").tolist() if "uid" in data.keys() else None
    saved_sessions = data.get("session_id").tolist() if "session_id" in data.keys() else None

    for row, key in enumerate(old_keys):
        parts = key.rsplit("_", 2) if isinstance(key, str) else []
        if len(parts) != 3:
            raise _CacheLayoutMismatch(f"invalid trajectory key {key!r}")
        uid, session, index = parts
        if not uid or session not in expected_sessions or not index.isascii() or not index.isdecimal():
            raise _CacheLayoutMismatch(f"invalid trajectory key {key!r} for rollout.n={len(expected_sessions)}")
        if saved_uids is not None and saved_uids[row] != uid:
            raise _CacheLayoutMismatch(f"row uid disagrees with trajectory key {key!r}")
        if saved_sessions is not None and saved_sessions[row] != int(session):
            raise _CacheLayoutMismatch(f"row session_id disagrees with trajectory key {key!r}")
        yield uid, session, index


def _config_replay_batches(
    key_batches: list[list[str]], data_list: list[TensorDict], new_prompt_uids: list[str], n: int
) -> list[list[str]]:
    """Validate every saved row, then remap each inner batch in its original layout."""
    from tensordict.tensorclass import NonTensorData

    from verl.utils.tensordict_utils import assign_non_tensor_stack

    expected_sessions = {str(i) for i in range(n)}
    sessions_by_uid: dict[str, set[str]] = {}
    seen = set()
    parsed_batches = []
    for keys, data in zip(key_batches, data_list, strict=True):
        rows = list(_config_trajectories(keys, data, expected_sessions))
        for uid, session, index in rows:
            # Advantage grouping compares the numeric output index.
            identity = (uid, session, index.lstrip("0") or "0")
            if identity in seen:
                raise _CacheLayoutMismatch("duplicate trajectory keys")
            seen.add(identity)
            sessions_by_uid.setdefault(uid, set()).add(session)
        parsed_batches.append(rows)

    if len(sessions_by_uid) != len(new_prompt_uids):
        raise _CacheLayoutMismatch(f"expected {len(new_prompt_uids)} prompt groups, found {len(sessions_by_uid)}")
    for uid, sessions in sessions_by_uid.items():
        if sessions != expected_sessions:
            raise _CacheLayoutMismatch(f"expected {n} sessions for prompt {uid!r}, found {len(sessions)}")

    schemas = {
        frozenset(data.keys(include_nested=True, leaves_only=True)) - {"uid", "session_id"}
        for data in data_list
        if len(data)
    }
    if len(schemas) > 1:
        raise _CacheLayoutMismatch("inner batches have different trajectory fields")

    uid_map = dict(zip(sessions_by_uid, new_prompt_uids, strict=True))
    session_template = next((data.get("session_id") for data in data_list if "session_id" in data.keys()), None)
    mapped_batches = []
    for data, rows in zip(data_list, parsed_batches, strict=True):
        mapped_batches.append([f"{uid_map[uid]}_{session}_{index}" for uid, session, index in rows])
        if not rows:
            continue
        assign_non_tensor_stack(data, "uid", [uid_map[uid] for uid, _, _ in rows])
        if session_template is not None and "session_id" not in data.keys():
            session_ids = [int(session) for _, session, _ in rows]
            if isinstance(session_template, torch.Tensor):
                data["session_id"] = session_template.new_tensor(session_ids)
            else:
                assign_non_tensor_stack(data, "session_id", session_ids)
        # The storage writer selects individual positions from each field.
        for name, value in list(data.items()):
            if isinstance(value, NonTensorData):
                assign_non_tensor_stack(data, name, value.tolist())

    return mapped_batches


@register_skip("rollout_tq")
class RolloutTqSkip(BaseSkip):
    """Rollout skip for V1 TransferQueue-based trainer (``skip.rollout_tq``).

    V1's split architecture (submit prompts to TQ,
    then sample trajectories) requires direct method calls from the trainer:
    - Phase one (no cache): trainer calls ``prepare_data`` after ``replay_buffer.sample``
    - Phase two (cache exists): trainer calls ``load_dump_data`` in ``_add_batch_to_generate``

    When ``parameter_sync_step > 1`` (separate async), one global step performs multiple
    ``sample`` calls.  Each mini-batch is saved to a separate inner sub-directory
    ``{step}/{inner_idx}/`` so the full step is captured; on load all inner dirs are
    merged.  For ``parameter_sync_step == 1`` (sync / colocate async) the directory
    structure is (``{step}/{0}/tq_batch.pt``).
    """

    support_actions = [SkipAction.CACHE, SkipAction.REPEAT]
    meta_name = "meta.json"

    print_mark = "[RolloutTqSkip()] "
    tq_batch_name = "tq_batch.pt"

    def __init__(self, local_config, global_config):
        super().__init__(local_config, global_config)
        self.load_mode = getattr(local_config, "load_mode", "dump")
        # prepare experiment info
        self.exp_name = global_config.trainer.get("experiment_name", "default_experiment_name")
        self.project_name = global_config.trainer.get("project_name", "default_project_name")
        self.n = int(OmegaConf.select(global_config, "actor_rollout_ref.rollout.n", default=0))
        gen_batch_size = OmegaConf.select(global_config, "data.gen_batch_size")
        if gen_batch_size is None:
            gen_batch_size = OmegaConf.select(global_config, "data.train_batch_size", default=0)
        self.gbs = int(gen_batch_size)
        self.response_length = OmegaConf.select(global_config, "data.max_response_length", default=0)
        self.prompt_length = OmegaConf.select(global_config, "data.max_prompt_length", default=0)

        self.parameter_sync_step = int(
            OmegaConf.select(
                global_config,
                f"trainer.v1.{global_config.trainer.v1.trainer_mode}.parameter_sync_step",
                default=1,
            )
        )

    def _get_project_dump_dir(self) -> Path:
        dumped_dir = Path(self.dump_dir).expanduser().resolve()
        sub_dir = (
            f"{self.exp_name}_{self.project_name}"
            + f"/GBS{self.gbs}_N{self.n}_in{self.prompt_length}_out{self.response_length}"
        )
        dumped_dir = dumped_dir.joinpath(sub_dir).absolute()
        return dumped_dir

    def _get_step_dump_dir(self, step) -> Path:
        return self._get_project_dump_dir().joinpath(f"{step}").absolute()

    def _get_v1_inner_dir(self, step: int, inner_idx: int) -> Path:
        """Return the dump directory for one mini-batch within a step."""
        return self._get_step_dump_dir(step) / str(inner_idx)

    def _check_valid_v1_step(self, step: int) -> bool:
        """Check whether ALL inner dirs for *step* exist (complete cache)."""
        return all(
            self._check_valid_v1_step_path(self._get_v1_inner_dir(step, i)) for i in range(self.parameter_sync_step)
        )

    def _find_first_missing_inner(self, step: int) -> int:
        """Return the first inner index whose dir does not yet exist."""
        for i in range(self.parameter_sync_step):
            if not self._check_valid_v1_step_path(self._get_v1_inner_dir(step, i)):
                return i
        return -1  # all present

    def _check_valid_v1_step_path(self, path: Path) -> bool:
        """Check whether a V1-format cached batch (``tq_batch.pt``) exists at *path*."""
        if not path.is_dir():
            return False
        return (path / self.tq_batch_name).is_file() and (path / self.meta_name).is_file()

    def _get_available_steps_v1(self) -> list[int]:
        """Return sorted list of steps that have V1-format cached data."""
        result: list[int] = []
        project_dir = self._get_project_dump_dir()
        if not project_dir.is_dir():
            return result
        for child in project_dir.iterdir():
            if not child.is_dir():
                continue
            try:
                step = int(child.name)
            except ValueError:
                continue
            if not self._check_valid_v1_step(step):
                continue
            result.append(step)
        return sorted(result)

    def _resolve_load_step_v1(self, step: int) -> int:
        """Return the actual step to load from for V1 path, or -1 if none available.

        - ``cache``: exact step match
        - ``repeat``: closest available step (prefer smaller, then larger)
        """
        if self._check_valid_v1_step(step):
            return step
        if self.action == SkipAction.REPEAT:
            available = self._get_available_steps_v1()
            if not available:
                return -1
            smaller = [s for s in available if s < step]
            if smaller:
                return smaller[-1]
            larger = [s for s in available if s > step]
            if larger:
                return larger[0]
        return -1

    def has_v1_cache(self, step: int) -> bool:
        """V1 phase check: whether cached TQ batch data exists for *step*.

        Returns True -> phase two (load from disk).
        Returns False -> phase one (normal rollout + save).
        """
        return self._resolve_load_step_v1(step) != -1

    def meet_precondition(self, step: int, *args, **kwargs) -> bool:
        """Phase two: whether cached data exists and should be loaded/injected."""
        return self.has_v1_cache(step)

    def should_save(self, step: int, partition_id: str = "train") -> bool:
        """Phase one: whether the sampled batch should be saved to disk."""
        return partition_id == "train" and not self.has_v1_cache(step)

    def maybe_load_and_inject(self, step: int, new_prompt_uids: list[str], partition_id: str = "train") -> bool:
        """Phase two: load cached data and inject into TransferQueue if cache exists.

        Convenience wrapper that combines ``meet_precondition`` + ``load_dump_data``.
        Returns ``True`` if cached data was injected (caller should skip real rollout),
        ``False`` otherwise.

        Args:
            step: Current training step.
            new_prompt_uids: Freshly generated uids for the current batch.
            partition_id: TQ partition (``"train"`` or ``"val"``).
        """
        if not self.meet_precondition(step):
            print(
                f"{self.print_mark}\033[33mNo cached data found for step {step}. "
                f"The trainer will generate and dump the data.\033[0m",
                flush=True,
            )
            return False
        try:
            self.load_dump_data(
                step=step,
                new_prompt_uids=new_prompt_uids,
                n=self.n,
                global_steps=step,
                partition_id=partition_id,
            )
        except _CacheLayoutMismatch as exc:
            logger.warning("Incompatible rollout cache at step %d: %s. Collecting fresh rollouts.", step, exc)
            return False
        return True

    def prepare_data(self, step: int, batch, global_steps: int) -> None:
        """Phase one: read batch from TransferQueue and save to disk.

        When ``parameter_sync_step > 1``, each ``sample`` call saves its mini-batch
        to ``{step}/{inner_idx}/``.  The first missing inner index is used so that
        repeated calls within the same step fill successive sub-directories.

        Args:
            step: Current training step (used as dump directory name).
            batch: :class:`transfer_queue.KVBatchMeta` from ``ReplayBuffer.sample()``.
            global_steps: Current ``global_steps`` for metadata.
        """
        import transfer_queue as tq

        # Determine which inner index to save to (-1 means all present, shouldn't happen)
        inner_idx = self._find_first_missing_inner(step)
        if inner_idx == -1:
            return
        save_dir = self._get_v1_inner_dir(step, inner_idx)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Read all trajectory fields from TQ
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id)

        save_payload = {
            "tensordict": data,
            "tags": batch.tags,
            "keys": list(batch.keys),
            "global_steps": global_steps,
        }
        torch.save(save_payload, save_dir / self.tq_batch_name)

        meta_path = save_dir / self.meta_name
        meta_path.write_text(json.dumps({"global_steps": global_steps, "num_trajectories": len(batch.keys)}))
        print(
            f"{self.print_mark}\033[33mDump TQ batch at step {step} "
            f"({len(batch.keys)} trajectories) to {save_dir}\033[0m",
            flush=True,
        )

    def load_dump_data(
        self,
        step: int,
        new_prompt_uids: list[str],
        n: int,
        global_steps: int,
        partition_id: str = "train",
    ) -> None:
        """Phase two: load cached data from disk and inject into TransferQueue.

        Maps saved groups to new prompt UIDs using the configured replay policy.

        Args:
            step: Current training step.
            new_prompt_uids: Freshly generated uids for the current batch.
            n: Number of rollout sessions per prompt (``rollout.n``).
            global_steps: Current ``global_steps``, used to override staleness tags.
            partition_id: TQ partition (``"train"`` or ``"val"``).
        """
        import transfer_queue as tq

        load_step = self._resolve_load_step_v1(step)
        if load_step == -1:
            raise FileNotFoundError(
                f"{self.print_mark}No dump data found for step {step} under {self._get_project_dump_dir()}"
            )

        # Load all inner dirs and merge (parameter_sync_step == 1 → single dir)
        key_batches: list[list[str]] = []
        old_tags: list = []
        data_list: list[TensorDict] = []
        for inner_idx in range(self.parameter_sync_step):
            inner_dir = self._get_v1_inner_dir(load_step, inner_idx)
            payload = torch.load(inner_dir / self.tq_batch_name, weights_only=False)
            if self.load_mode == "config" and not (
                len(payload["tensordict"]) == len(payload["keys"]) == len(payload["tags"])
            ):
                raise _CacheLayoutMismatch(f"field, key and tag counts disagree in {inner_dir}")
            data_list.append(payload["tensordict"])
            key_batches.append(payload["keys"])
            old_tags.extend(payload["tags"])

        if self.load_mode == "config":
            # Keep saved tensor layouts and per-inner metadata intact during injection.
            key_batches = _config_replay_batches(key_batches, data_list, new_prompt_uids, n)
        else:
            from verl.utils.tensordict_utils import concat_tensordict, index_select_tensor_dict

            old_keys = [key for keys in key_batches for key in keys]
            data = concat_tensordict(data_list) if len(data_list) > 1 else data_list[0]
            new_keys, traj_indices = self._dump_replay_rows(old_keys, new_prompt_uids, n, load_step)
            # NestedTensor (jagged prompts/responses) cannot be indexed directly on dim=0.
            data_list = [index_select_tensor_dict(data, traj_indices)]
            key_batches = [new_keys]
            old_tags = [old_tags[i] for i in traj_indices]

        new_tags = []
        for old_tag in old_tags:
            tag = dict(old_tag)
            tag.update(global_steps=global_steps, min_global_steps=global_steps, max_global_steps=global_steps)
            tag.pop("is_prompt", None)
            new_tags.append(tag)

        offset = 0
        for keys, fields in zip(key_batches, data_list, strict=True):
            end = offset + len(keys)
            if keys:
                tq.kv_batch_put(keys=keys, fields=fields, tags=new_tags[offset:end], partition_id=partition_id)
            offset = end
        prompt_tags = [{"is_prompt": True, "status": "finished", "global_steps": global_steps}] * len(new_prompt_uids)
        tq.kv_batch_put(keys=new_prompt_uids, partition_id=partition_id, tags=prompt_tags)

        print(
            f"{self.print_mark}\033[33mInjected {len(new_tags)} cached trajectories "
            f"from step {load_step} ({len(new_prompt_uids)} prompts x {n} sessions) "
            f"into TQ partition '{partition_id}' at current step {step}\033[0m",
            flush=True,
        )

    def _dump_replay_rows(
        self, old_keys: list[str], new_prompt_uids: list[str], n: int, load_step: int
    ) -> tuple[list[str], list[int]]:
        # Group saved trajectories by parent uid.
        # Key format: {uid}_{session_id}_{index}, uid is UUID4 (no underscores).
        # ReplayBuffer.sample() does NOT guarantee keys are sorted by uid prefix,
        # so we use a dict to group regardless of key ordering.
        uid_to_indices: dict[str, list[int]] = {}
        for idx, key in enumerate(old_keys):
            parent_uid = key.split("_")[0]
            uid_to_indices.setdefault(parent_uid, []).append(idx)
        groups = list(uid_to_indices.values())

        if not groups:
            raise RuntimeError(
                f"{self.print_mark}No trajectory groups found in cached data ({len(old_keys)} keys) at step {load_step}"
            )

        num_cached_groups = len(groups)
        num_prompts = len(new_prompt_uids)
        if num_cached_groups < num_prompts:
            print(
                f"{self.print_mark}\033[33mCached {num_cached_groups} prompt groups but need {num_prompts} "
                f"prompts; will cycle through available groups to fill\033[0m",
                flush=True,
            )
        elif num_cached_groups > num_prompts:
            print(
                f"{self.print_mark}\033[33mCached {num_cached_groups} prompt groups but only need {num_prompts} "
                f"prompts; using first {num_prompts} groups\033[0m",
                flush=True,
            )

        # Build new keys/tags: map each new uid to a cached group.
        # If group has fewer than *n* trajectories (some sessions failed),
        # cycle within the group to fill *n* trajectories.
        # If cached groups < new prompts, cycle through groups (modulo).
        new_keys = []
        traj_indices: list[int] = []
        for prompt_idx, new_uid in enumerate(new_prompt_uids):
            group = groups[prompt_idx % num_cached_groups]
            for session_id in range(n):
                traj_idx = group[session_id % len(group)]
                traj_indices.append(traj_idx)
                new_keys.append(f"{new_uid}_{session_id}_0")
        return new_keys, traj_indices
