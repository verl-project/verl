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
"""Score-aware retention of ``global_step_{N}`` checkpoints for the v1 PPO trainer.

Checkpoints are still written every ``trainer.save_freq`` steps. After the
validation of each step the driver keeps only:

* ``latest``: the ``keep_last`` most recent checkpoints and the one named in
  ``latest_checkpointed_iteration.txt``, to resume training;
* ``best``: the highest-scoring checkpoint so far, to restart from when the
  training dynamics need fixing;
* ``converged``: the earliest checkpoint whose score is within
  ``converge_tolerance`` of the best, to study the dynamics before convergence;
* ``milestone``: a sparse trail for coarse rollback: a checkpoint that sets a new
  best at least ``milestone_interval`` steps after the previous milestone (or
  the start of the run). Milestones are never deleted.

``best``, ``converged`` and ``milestone`` are always *records*: checkpoints whose score
beats every earlier checkpoint (the earliest checkpoint reaching any level is
necessarily a record). Since the best score only rises, a record that falls more
than ``converge_tolerance`` below it can never become ``converged`` again and is
deleted; records still inside the band are kept as future ``converged``
candidates. The early fast rise therefore leaves nothing behind once the score
has moved on.

Scores may be smoothed over the last ``window`` validation points so a single
lucky evaluation does not define the best (and thus the convergence band).
``save_freq`` should be a multiple of ``test_freq``: a checkpoint without a
validation at its own step is unscored and only kept while it is among the
latest ones.

State persists in ``<default_local_dir>/checkpoint_retention.json``. Only
checkpoints registered there are ever deleted; directories written by other
runs or before the policy was enabled are left alone. Deletion runs on the
driver, so ``default_local_dir`` must be visible to it (shared filesystem);
copies under ``default_hdfs_dir`` are not pruned.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from verl.utils.checkpoint.checkpoint_manager import get_checkpoint_tracker_filename

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

RETENTION_REGISTRY_FILENAME = "checkpoint_retention.json"
_REGISTRY_VERSION = 3
_METRIC_PREFIX = "checkpoint_retention"


@dataclass
class CheckpointRetentionConfig:
    enable: bool = False
    # Validation metric key(s); the score is their mean, e.g.
    # ["val-core/aime24/acc/mean@32", "val-core/aime25/acc/mean@32"].
    metric: list[str] = field(default_factory=list)
    # "max" when higher scores are better, "min" otherwise.
    mode: str = "max"
    keep_last: int = 1
    # Number of most recent validation points averaged into a checkpoint's score.
    window: int = 1
    # Width of the convergence band below the best score; None disables ``converged``.
    converge_tolerance: Optional[float] = None
    # Minimum steps between milestones; None disables ``milestone``.
    milestone_interval: Optional[int] = None

    @classmethod
    def from_config(cls, cfg: Any) -> CheckpointRetentionConfig:
        if cfg is None:
            return cls()
        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
        cfg = dict(cfg)
        metric = cfg.get("metric") or []
        if isinstance(metric, str):
            metric = [metric]
        tolerance = cfg.get("converge_tolerance")
        interval = cfg.get("milestone_interval")
        return cls(
            enable=bool(cfg.get("enable", False)),
            metric=[str(m) for m in metric],
            mode=str(cfg.get("mode", "max")),
            keep_last=int(cfg.get("keep_last", 1)),
            window=int(cfg.get("window", 1)),
            converge_tolerance=None if tolerance is None else float(tolerance),
            milestone_interval=None if interval is None else int(interval),
        )


@dataclass
class CheckpointRecord:
    step: int
    score: Optional[float] = None
    # Score averaged over the last ``window`` validation points; what ranking uses.
    smoothed_score: Optional[float] = None
    # Whether this checkpoint beat every earlier checkpoint when it was scored.
    is_record: bool = False
    # A record kept permanently as a coarse rollback point.
    milestone: bool = False


def validate_checkpoint_retention_config(config: DictConfig) -> CheckpointRetentionConfig:
    """Parse ``trainer.checkpoint_retention`` and reject settings that would silently misbehave."""
    trainer_cfg = config.trainer
    cfg = CheckpointRetentionConfig.from_config(trainer_cfg.get("checkpoint_retention", None))
    if not cfg.enable:
        return cfg

    if not cfg.metric:
        raise ValueError("trainer.checkpoint_retention.metric is required")
    if cfg.mode not in ("max", "min"):
        raise ValueError(f"trainer.checkpoint_retention.mode must be 'max' or 'min', got {cfg.mode!r}")
    if cfg.keep_last < 1:
        raise ValueError("trainer.checkpoint_retention.keep_last must be >= 1 so training can always resume")
    if cfg.window < 1:
        raise ValueError("trainer.checkpoint_retention.window must be >= 1")
    if cfg.converge_tolerance is not None and cfg.converge_tolerance < 0:
        raise ValueError("trainer.checkpoint_retention.converge_tolerance must be >= 0 or null")
    if cfg.milestone_interval is not None and cfg.milestone_interval < 1:
        raise ValueError("trainer.checkpoint_retention.milestone_interval must be >= 1 or null")
    if trainer_cfg.get("test_freq", -1) <= 0:
        raise ValueError("trainer.checkpoint_retention needs trainer.test_freq > 0 to score checkpoints")

    # Worker-side rotation deletes only ``actor/``/``critic/`` by save order and
    # would remove shards of checkpoints this policy decided to keep.
    for key in ("max_actor_ckpt_to_keep", "max_critic_ckpt_to_keep"):
        if trainer_cfg.get(key, None) is not None:
            raise ValueError(f"trainer.{key} must be null when trainer.checkpoint_retention.enable=True")
    if trainer_cfg.get("remove_previous_ckpt_in_save", False):
        raise ValueError("trainer.remove_previous_ckpt_in_save conflicts with trainer.checkpoint_retention")

    critic_cfg = config.get("critic", {}) or {}
    critic_ckpt_cfg = critic_cfg.get("checkpoint", {}) or {}
    if critic_ckpt_cfg.get("async_save", False):
        from verl.trainer.ppo.utils import need_critic

        if need_critic(config):
            # Actor and critic finalize independently and update the same tracker.
            # An actor durability marker therefore cannot confirm critic completion.
            raise ValueError("trainer.checkpoint_retention requires synchronous critic checkpoint saving")

    # With async saves the newest checkpoint may still be in flight, so the
    # previous one has to survive until it is durable.
    actor_ckpt_cfg = config.actor_rollout_ref.actor.get("checkpoint", {}) or {}
    if actor_ckpt_cfg.get("async_save", False) and cfg.keep_last < 2:
        raise ValueError("trainer.checkpoint_retention.keep_last must be >= 2 with async checkpoint saving")

    save_freq = trainer_cfg.get("save_freq", -1)
    test_freq = trainer_cfg.get("test_freq", -1)
    if save_freq > 0 and save_freq % test_freq != 0:
        logger.warning(
            f"save_freq={save_freq} is not a multiple of test_freq={test_freq}: checkpoints saved without a "
            "validation at the same step stay unscored and are kept only while among the latest ones"
        )
    return cfg


class CheckpointRetention:
    """Driver-side bookkeeping and pruning of scored checkpoints.

    Usage from the trainer: :meth:`load` after resuming, :meth:`observe_validation`
    for a validation without a checkpoint (e.g. before training), :meth:`on_save`
    after every save, and :meth:`on_step_end` once per step that saved or validated.
    """

    def __init__(self, cfg: CheckpointRetentionConfig, root_dir: str, *, async_save: bool = False):
        self.cfg = cfg
        self.root_dir = root_dir
        self.async_save = async_save
        self.records: dict[int, CheckpointRecord] = {}
        # (step, score) of every validation seen, used for smoothing.
        self.history: list[tuple[int, float]] = []
        # Step training (re)started from in this run directory; the first milestone interval counts from it.
        self.start_step: int = 0

    @property
    def enabled(self) -> bool:
        return self.cfg.enable

    @property
    def registry_path(self) -> str:
        return os.path.join(self.root_dir, RETENTION_REGISTRY_FILENAME)

    # ------------------------------------------------------------------ state

    def load(self, current_step: int) -> None:
        """Restore persisted state; entries after ``current_step`` are dropped because those steps will re-run."""
        if not self.enabled:
            self.start_step = current_step
            return
        tracker_step = self._tracker_step()
        if self.enabled and self.async_save and tracker_step is not None and tracker_step > current_step:
            raise ValueError(
                "Asynchronous checkpoint retention cannot resume behind the durability tracker; "
                "use synchronous saving or a fresh checkpoint directory for rollback"
            )
        if not os.path.exists(self.registry_path):
            self.start_step = current_step
            return
        with open(self.registry_path) as f:
            state = json.load(f)
        if state.get("version") in (1, 2) or "score_policy" not in state:
            raise ValueError(
                "Checkpoint retention registry has no score-policy provenance; "
                "disable retention for this directory or use a fresh checkpoint directory. "
                "Leave the existing registry and checkpoints intact."
            )
        if state.get("version") != _REGISTRY_VERSION:
            raise ValueError(f"Unsupported checkpoint retention registry version in {self.registry_path}")
        if state["score_policy"] != self._score_policy():
            raise ValueError(
                "Checkpoint retention score policy changed (metric, mode or window); "
                "disable retention for this directory or use a fresh checkpoint directory. "
                "Leave the existing registry and checkpoints intact."
            )
        self.records = {
            int(r["step"]): CheckpointRecord(**r) for r in state.get("records", []) if int(r["step"]) <= current_step
        }
        self.history = [(int(s), float(v)) for s, v in state.get("history", []) if int(s) <= current_step]
        self.start_step = min(int(state.get("start_step", current_step)), current_step)
        logger.info(f"Loaded checkpoint retention registry: {len(self.records)} checkpoints, best={self.best_step()}")

    def _score_policy(self) -> dict:
        """Identify the policy that produced the persisted scores and record flags."""
        return {"metric": list(self.cfg.metric), "mode": self.cfg.mode, "window": self.cfg.window}

    def _persist(self) -> None:
        os.makedirs(self.root_dir, exist_ok=True)
        state = {
            "version": _REGISTRY_VERSION,
            "score_policy": self._score_policy(),
            "start_step": self.start_step,
            "history": [[s, v] for s, v in self.history],
            "records": [asdict(self.records[s]) for s in sorted(self.records)],
        }
        tmp_path = self.registry_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, self.registry_path)

    # ---------------------------------------------------------------- scoring

    def score(self, val_metrics: dict) -> float:
        missing = [k for k in self.cfg.metric if k not in val_metrics]
        if missing:
            available = sorted(k for k in val_metrics if k.startswith("val-core/"))
            raise KeyError(f"checkpoint_retention metric(s) {missing} not in validation metrics; val-core: {available}")
        values = [float(val_metrics[k]) for k in self.cfg.metric]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"checkpoint_retention metric(s) {self.cfg.metric} are not finite: {values}")
        return sum(values) / len(values)

    def _better(self, candidate: float, reference: float) -> bool:
        return candidate > reference if self.cfg.mode == "max" else candidate < reference

    def _record_validation(self, step: int, score: float) -> float:
        """Append to history (replacing a same-step entry) and return the smoothed score."""
        self.history = [(s, v) for s, v in self.history if s != step]
        self.history.append((step, score))
        self.history.sort()
        window = [v for _, v in self.history[-self.cfg.window :]]
        return sum(window) / len(window)

    def _records_chain(self) -> list[CheckpointRecord]:
        return [self.records[s] for s in sorted(self.records) if self.records[s].is_record]

    def best_step(self) -> Optional[int]:
        """Records improve monotonically, so the best checkpoint is the last record."""
        chain = self._records_chain()
        return chain[-1].step if chain else None

    def _in_band(self, record: CheckpointRecord, best: CheckpointRecord) -> bool:
        tol = self.cfg.converge_tolerance
        if self.cfg.mode == "max":
            return record.smoothed_score >= best.smoothed_score - tol
        return record.smoothed_score <= best.smoothed_score + tol

    def _milestone_anchor(self) -> int:
        milestones = [r.step for r in self.records.values() if r.milestone]
        return max(milestones) if milestones else self.start_step

    def converged_step(self) -> Optional[int]:
        """Earliest record within ``converge_tolerance`` of the best."""
        if self.cfg.converge_tolerance is None:
            return None
        chain = self._records_chain()
        if not chain:
            return None
        return next(r.step for r in chain if self._in_band(r, chain[-1]))

    # ------------------------------------------------------------ trainer API

    def observe_validation(self, step: int, val_metrics: dict) -> dict[str, float]:
        """Record a validation that has no checkpoint of its own; it only feeds smoothing."""
        if not val_metrics:
            return {}
        score = self.score(val_metrics)
        self._record_validation(step, score)
        self._persist()
        return {f"{_METRIC_PREFIX}/score": score}

    def on_save(self, step: int) -> None:
        # A re-save of the same step (e.g. after resuming from an older one) starts over.
        self.records[step] = CheckpointRecord(step=step)
        self._persist()

    def on_step_end(self, step: int, val_metrics: Optional[dict]) -> dict[str, float]:
        """Score the step's checkpoint if validated, then prune. Returns metrics to log."""
        metrics: dict[str, float] = {}
        if val_metrics:
            metrics.update(self._score_step(step, val_metrics))
        kept, deleted = self.prune()
        metrics[f"{_METRIC_PREFIX}/num_kept"] = float(len(kept))
        metrics[f"{_METRIC_PREFIX}/num_deleted"] = float(len(deleted))
        for name, value in (("best_step", self.best_step()), ("converged_step", self.converged_step())):
            if value is not None:
                metrics[f"{_METRIC_PREFIX}/{name}"] = float(value)
        return metrics

    def _score_step(self, step: int, val_metrics: dict) -> dict[str, float]:
        score = self.score(val_metrics)
        smoothed = self._record_validation(step, score)
        metrics = {f"{_METRIC_PREFIX}/score": score, f"{_METRIC_PREFIX}/smoothed_score": smoothed}
        record = self.records.get(step)
        if record is not None:
            # Compare against the best before this checkpoint is scored.
            record.is_record = False
            record.milestone = False
            best_step = self.best_step()
            best = self.records[best_step] if best_step is not None else None
            record.score = score
            record.smoothed_score = smoothed
            record.is_record = best is None or self._better(smoothed, best.smoothed_score)
            if record.is_record and self.cfg.milestone_interval is not None:
                record.milestone = step - self._milestone_anchor() >= self.cfg.milestone_interval
            metrics[f"{_METRIC_PREFIX}/is_new_best"] = float(record.is_record)
            metrics[f"{_METRIC_PREFIX}/is_milestone"] = float(record.milestone)
            logger.info(
                f"Checkpoint global_step_{step}: score={score:.6g} smoothed={smoothed:.6g} "
                f"previous_best={None if best is None else best.smoothed_score} new_best={record.is_record} "
                f"milestone={record.milestone}"
            )
        self._persist()
        return metrics

    # ---------------------------------------------------------------- pruning

    def _tracker_step(self) -> Optional[int]:
        tracker = get_checkpoint_tracker_filename(self.root_dir)
        if not os.path.exists(tracker):
            return None
        try:
            with open(tracker) as f:
                return int(f.read().strip())
        except ValueError:
            return None

    def keep_reasons(self) -> dict[int, list[str]]:
        """Why each registered checkpoint is kept; steps missing from the result are to be deleted."""
        reasons: dict[int, list[str]] = {}

        def add(step: int, reason: str) -> None:
            reasons.setdefault(step, []).append(reason)

        for step in sorted(self.records)[-self.cfg.keep_last :]:
            add(step, "latest")
        tracker_step = self._tracker_step()
        if tracker_step in self.records:
            add(tracker_step, "resume")
        if self.async_save:
            # Megatron queues asynchronous saves in FIFO order; multiple checkpoints
            # may still be writing. Only the durability tracker confirms completion.
            for step in self.records:
                if tracker_step is None or step > tracker_step:
                    add(step, "pending_save")

        chain = self._records_chain()
        if chain:
            best = chain[-1]
            add(best.step, "best")
            converged = self.converged_step()
            if converged is not None:
                add(converged, "converged")
                # Later records still inside the band become ``converged`` if the best rises further.
                for r in chain:
                    if r.step > converged and r.step != best.step and self._in_band(r, best):
                        add(r.step, "converge_candidate")
            for r in chain:
                if r.milestone:
                    add(r.step, "milestone")
        return reasons

    def prune(self) -> tuple[list[int], list[int]]:
        reasons = self.keep_reasons()
        deleted = [s for s in sorted(self.records) if s not in reasons]
        for step in deleted:
            path = os.path.join(self.root_dir, f"global_step_{step}")
            logger.info(f"checkpoint_retention: deleting {path} (score={self.records[step].score})")
            shutil.rmtree(path, ignore_errors=True)
            del self.records[step]
        if deleted:
            self._persist()
        kept = sorted(reasons)
        logger.info(f"checkpoint_retention: kept {{{', '.join(f'{s}: {reasons[s]}' for s in kept)}}}")
        return kept, deleted


def build_checkpoint_retention(config: DictConfig) -> CheckpointRetention:
    """Build the retention policy from ``trainer.checkpoint_retention``; disabled when unset."""
    cfg = validate_checkpoint_retention_config(config)
    root_dir = config.trainer.default_local_dir
    if not os.path.isabs(root_dir):
        root_dir = os.path.join(os.getcwd(), root_dir)
    actor_checkpoint = config.actor_rollout_ref.actor.get("checkpoint", {}) or {}
    return CheckpointRetention(cfg, root_dir, async_save=bool(actor_checkpoint.get("async_save", False)))
