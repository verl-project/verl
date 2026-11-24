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
"""Rollout-side independent restart: engine-agnostic lifecycle core.

Importable without vllm/sglang/torch, so its semantics stay CPU-testable:
the engine-death taxonomy shared by the supervisor probes and the
request-path replay, the frozen evidence/weight-source dataclasses, owned
engine-process accounting with its cleanup ladder, the checkpoint
cold-start locate, the restart and replay budgets, the replica runtime
state machine, and the polling fault supervisor. Nothing here runs unless
``rollout.restart.enabled`` is true.
"""

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import psutil

logger = logging.getLogger(__name__)


# ── Internal tunables (exposed knobs live in RolloutFaultToleranceConfig) ──────────

# check_health probe timeout (detection probes, VERIFYING gate).
PROBE_TIMEOUT_S = 10.0

# Bound for the update_weights-entry detection round (see
# ReplicaFaultSupervisor.probe_membership): a round that overruns falls
# back to the last snapshot — the commit watchdog remains the backstop.
_ENTRY_ROUND_TIMEOUT_S = 30.0

# Upper bound for the checkpoint cold-start locate (a hung shared FS must not stall the cold start).
_LOCATE_TIMEOUT_S = 30.0

# Request-path bounded-replay budgets (internal constants). Replay is inherent to
# restart.enabled=true — there is deliberately no disable switch.
ENGINE_RECOVERY_TIMEOUT_S = 300.0
MAX_ENGINE_RETRIES = 3


# ── Error taxonomy ───────────────────────────────────────────────────────────


class ReplicaRestartError(RuntimeError):
    """Restart primitive failed (cleanup/fence/probe). Fail-closed: keep ISOLATED."""


# Exception classes that mean "the engine/actor died" (as opposed to a bad
# request): matched by class name anywhere on the cause chain.
ENGINE_FAILURE_CLASSES = {"EngineDeadError", "ActorDiedError", "RayActorError", "WorkerCrashedError"}


def _iter_exception_causes(exc: BaseException):
    """Yield ``exc`` and each successive ``__cause__``/``cause`` node, stopping at
    the first node without an inner cause (cycle-safe)."""
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        inner = getattr(current, "__cause__", None)
        if inner is None:
            inner = getattr(current, "cause", None)
        if inner is None:
            return
        current = inner


def _unwrap_ray_exception(exc: BaseException) -> BaseException:
    """Ray wraps task exceptions; classify on the innermost cause."""
    last = exc
    for node in _iter_exception_causes(exc):
        last = node
    if getattr(last, "__cause__", None) is not None or getattr(last, "cause", None) is not None:
        # A cycle means no natural end: classify on the outermost exception.
        return exc
    return last


def is_engine_failure(exc: BaseException) -> bool:
    """Engine/actor death classes only; request-parameter errors propagate untouched."""
    for node in _iter_exception_causes(exc):
        if type(node).__name__ in ENGINE_FAILURE_CLASSES:
            return True
    return False


# ── Frozen data structures ────────────────────────────────────────────────────


@dataclass(frozen=True)
class FaultEvidence:
    """One observation of a (suspected) fault; frozen so it is safe across retries."""

    source: str
    """Where the evidence came from: "detect_faults" | "actor_state"."""

    error_class: str
    """Normalized failure class, e.g. "EngineDeadError" | "ActorDiedError" | "check_health_failed"."""


@dataclass(frozen=True)
class WeightSourceOverride:
    """Checkpoint cold start: the replacement engine cold-loads this checkpoint HF export at
    startup; its version stamp is set to ``global_steps``.

    Rides the restart chain explicitly (``replica.restart(weight_source=...)``
    -> ``launch_servers(weight_source=...)`` -> server constructor).
    """

    model_path: str
    global_steps: Optional[int] = None


# ── Process identity probe (witness of the teardown) ──────────────────────────


def verify_process_dead(pid: int, create_time: float) -> bool:
    """Whether the process identified by ``(pid, create_time)`` is dead.

    ``create_time`` pins the identity so a recycled PID cannot impersonate the
    original; a zombie counts as dead.
    """
    try:
        proc = psutil.Process(int(pid))
    except (psutil.NoSuchProcess, ValueError):
        return True
    try:
        if abs(proc.create_time() - float(create_time)) > 1e-6:
            return True  # PID was recycled; the original process is gone.
        return proc.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


# ── Checkpoint cold-start locate (trainer tracker convention) ─────────────────


class CheckpointUnavailable(FileNotFoundError):
    """No checkpoint usable for a cold start under the given root."""


def _hf_export_ready(hf_dir: str) -> bool:
    """An engine-cold-startable HF export: config + tokenizer + safetensors shards."""
    if not os.path.isdir(hf_dir):
        return False
    names = set(os.listdir(hf_dir))
    if "config.json" not in names:
        return False
    if not any(name.endswith(".safetensors") for name in names):
        return False
    return any(name.startswith("tokenizer") for name in names)


def _locate_cold_start_checkpoint(root: str) -> tuple[str, int]:
    """Locate the tracker-pointed checkpoint's HF export directory.

    ``latest_checkpointed_iteration.txt`` is the trainer's commit pointer
    (written last, after the save completes), so the tracked step is the
    cold-start candidate; the only additional gate is engine applicability
    — the step must carry an HF export an engine can boot from. Both trainer
    layouts resolve: Megatron (v2) nests it under ``model/huggingface``,
    FSDP writes ``huggingface`` directly. Torn trackers and half-written
    step directories get no scan or fallback; they raise here and the
    caller defers the replica to the next weight sync.

    Returns:
        ``(model_path, global_step)`` — the HF export directory suitable
        for an engine cold start, and its step number.

    Raises:
        CheckpointUnavailable: no usable cold-start source under ``root``.
    """
    if not root or not os.path.isdir(root):
        raise CheckpointUnavailable(f"checkpoint root does not exist or is not a directory: {root!r}")

    tracker = os.path.join(root, "latest_checkpointed_iteration.txt")
    tracked_step = None
    if os.path.isfile(tracker):
        try:
            with open(tracker) as f:
                tracked_step = int(f.read().strip())
        except (OSError, ValueError):
            logger.warning("unreadable tracker file %s; no checkpoint to cold start from", tracker)
    if tracked_step is None:
        raise CheckpointUnavailable(f"no readable tracker under {root!r}")

    step_dir = os.path.join(root, f"global_step_{tracked_step}", "actor")
    for hf_dir in (os.path.join(step_dir, "model", "huggingface"), os.path.join(step_dir, "huggingface")):
        if _hf_export_ready(hf_dir):
            return hf_dir, tracked_step
    raise CheckpointUnavailable(f"step {tracked_step} carries no engine-cold-startable HF export")


# ── Owned engine-process accounting ───────────────────────────────────────────


class EngineProcessAccounting:
    """Track — and verified-cleanup — the engine-owned child processes of one
    server actor (direct children minus the construction-time baseline, so
    pre-existing children are never touched).

    :meth:`shutdown_owned` is the verified cleanup ladder: bounded wait ->
    SIGTERM -> SIGKILL -> any survivor raises (a surviving engine would fight
    its replacement for the same devices).
    """

    def __init__(self):
        self._pid = os.getpid()
        self._baseline = {child.pid for child in self._snapshot_children()}
        self._owned: list[psutil.Process] = []

    def _snapshot_children(self) -> list[psutil.Process]:
        try:
            return psutil.Process(self._pid).children(recursive=False)
        except psutil.Error:
            return []

    def record(self) -> list[psutil.Process]:
        """Refresh the owned set: current direct children minus the baseline."""
        current = {child.pid: child for child in self._snapshot_children()}
        self._owned = [child for pid, child in current.items() if pid not in self._baseline]
        return list(self._owned)

    @property
    def owned_pids(self) -> list[int]:
        return [proc.pid for proc in self._owned]

    @staticmethod
    def _is_running_alive(proc: psutil.Process) -> bool:
        try:
            return proc.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    def shutdown_owned(self, wait_s: float = 5.0, term_s: float = 2.0, kill_s: float = 2.0) -> None:
        """Bounded wait -> SIGTERM -> SIGKILL over the owned children; any
        survivor raises (the engine's own graceful shutdown is the caller's
        business and may fail without blocking this ladder)."""
        owned = [proc for proc in self._owned if self._is_running_alive(proc)]
        if not owned:
            return

        _, alive = psutil.wait_procs(owned, timeout=wait_s)
        if alive:
            for proc in alive:
                try:
                    proc.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(alive, timeout=term_s)
        if alive:
            for proc in alive:
                try:
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(alive, timeout=kill_s)

        survivors = [proc for proc in alive if self._is_running_alive(proc)]
        if survivors:
            raise ReplicaRestartError(
                f"{len(survivors)} owned engine process(es) survived the cleanup ladder "
                f"(pids={[p.pid for p in survivors]}); refusing to restart — a surviving "
                "engine would contend with its replacement for the same devices"
            )


# ── Restart budget (anti-flapping) ────────────────────────────────────────────


class RestartBudget:
    """Per-replica anti-flapping budget: sliding-window count plus a forced
    cooldown so a crash-looping engine cannot burn it in one burst."""

    def __init__(self, max_restarts: int = 3, window_s: float = 3600.0, cooldown_s: float = 30.0):
        self.max_restarts = max_restarts
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self._events: list[float] = []

    def try_acquire(self) -> bool:
        return self.try_acquire_state() == "acquired"

    def try_acquire_state(self) -> str:
        """Return "acquired", "cooldown" or "exhausted".

        A cooldown rejection is pacing, not exhaustion: the caller keeps the
        replica ISOLATED and retries on the next polling round.
        """
        now = time.monotonic()
        self._events = [ts for ts in self._events if now - ts <= self.window_s]
        if self._events and now - self._events[-1] < self.cooldown_s:
            return "cooldown"
        if len(self._events) >= self.max_restarts:
            return "exhausted"
        self._events.append(now)
        return "acquired"


# ── Request-path replay budget ────────────────────────────────────────────────


@dataclass
class EngineRecovery:
    """Bounded-replay budget for one ``generate()`` call. The deadline starts
    lazily at the first engine failure — the window exists to cover the rebuild."""

    timeout: float
    max_retries: int
    retries: int = 0
    deadline: float | None = None

    def failed(self) -> None:
        """Count one engine failure; raises RuntimeError past the retry limit (stop-loss)."""
        if self.deadline is None:
            self.deadline = time.monotonic() + self.timeout
        self.retries += 1
        if self.retries > self.max_retries:
            raise RuntimeError(f"engine recovery gave up after {self.retries - 1} failed replays")

    def remaining(self) -> float:
        """Time left in the recovery window; TimeoutError once exhausted."""
        if self.deadline is None:
            return self.timeout
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("engine recovery window exhausted")
        return remaining


def make_engine_recovery() -> EngineRecovery:
    """Request-path replay budget wired to the module's internal tunables."""
    return EngineRecovery(timeout=ENGINE_RECOVERY_TIMEOUT_S, max_retries=MAX_ENGINE_RETRIES)


# ── Replica runtime state machine ─────────────────────────────────────────────


class ReplicaRuntimeState(str, Enum):
    SERVING = "serving"
    ISOLATED = "isolated"
    RESTARTING = "restarting"
    COLD_START = "cold_start"
    VERIFYING = "verifying"
    DEAD = "dead"


def _get_actor_state_dict(handle) -> Optional[dict]:
    """Actor state via the Ray GCS state API (the ``_is_worker_alive`` pattern)."""
    try:
        from ray.experimental.state.api import get_actor

        return get_actor(handle._actor_id.hex())
    except Exception:
        return None


# ── Membership snapshot (pull-reconcile payload) ──────────────────────────────


@dataclass(frozen=True)
class MembershipSnapshot:
    """Authoritative membership snapshot pulled by the driver-side reconcile at
    every ``update_weights`` entry: the manager is the single source of truth
    and the checkpoint manager's bookkeeping is a derived copy.
    """

    replicas: list
    """Syncable replicas (SERVING / COLD_START) — the weight-sync group members;
    everything else is fenced out of the group."""

    group_generation: int
    """Fencing generation: bumped at every conviction; the next sync builds
    under the derived name."""

    fatal_error: Optional[str]
    """Not None when every replica is DEAD — the sync pipeline fails closed."""

    awaiting_weight_sync: bool = False
    """True when no replica is serving while a rebuilt one (COLD_START) waits
    for a weight sync to promote it — the trainer's out-of-band sync trigger."""


# ── Supervisor ─────────────────────────────────────────────────────────────────


class ReplicaFaultSupervisor:
    """Detection + orchestration state machine, owned by the actorized
    ``LLMServerManager``.

    ``start()`` spawns the periodic polling task (interval
    ``probe_interval_s``); each round probes every registered replica.
    Recovery runs as on-demand asyncio tasks on the manager's event loop, so
    a restart never blocks the manager's other RPCs. The poll is the single
    detection path — there is no client-to-manager report channel.
    """

    def __init__(
        self,
        manager: Any,
        config: Any,
    ):
        self._ensure_logger_handler()
        self._manager = manager
        self._config = config
        self._states: dict[str, ReplicaRuntimeState] = {}
        self._budgets: dict[str, RestartBudget] = {}
        self._recovering: set[str] = set()
        self._recovery_tasks: set[asyncio.Task] = set()  # strong refs so rebuilds are not GC-ed
        self._poll_task: asyncio.Task | None = None
        self._last_issued_syncable_keys: set[str] | None = None

    @staticmethod
    def _ensure_logger_handler() -> None:
        """Emit detection/recovery markers regardless of root-logger state:
        a dedicated stdout handler with propagation disabled, immune to
        later root-handler swaps."""
        if logger.handlers:
            return
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s:%(lineno)d] => %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Validate applicability, then start the periodic polling task on the
        manager actor's event loop (must be called from a coroutine context)."""
        self._validate_applicability()
        self._ensure_live_logging()
        self._poll_task = asyncio.create_task(self._poll_loop())

    @staticmethod
    def _ensure_live_logging() -> None:
        """Re-bind the root logger to the *current* stdout: a handler bound
        to a stale std stream (captured before Ray redirected the actor's
        std streams) silently swallows early records. Idempotent — a later
        full reconfigure simply replaces it."""
        root = logging.getLogger()
        live = sys.stdout if sys.stdout is not None else sys.stderr
        for handler in root.handlers:
            try:
                if getattr(handler, "stream", None) is live:
                    return
            except Exception:
                continue
        handler = logging.StreamHandler(live)
        handler.setLevel(logging.WARNING)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(asctime)s:%(name)s:%(message)s"))
        root.addHandler(handler)

    async def _poll_loop(self):
        interval = float(getattr(self._config, "probe_interval_s", 10.0))
        while True:
            await asyncio.sleep(interval)
            try:
                await self.detect_faults()
            except asyncio.CancelledError:
                raise
            except Exception:
                # Same discipline as every detection round: a broken round must
                # never take the supervisor (or the manager actor) down.
                logger.warning("fault detection round failed (non-fatal)", exc_info=True)

    # ── detection ───────────────────────────────────────────────────────

    async def detect_faults(self) -> None:
        """One polling round: probe every registered replica concurrently and
        reconcile runtime states. Never raises: a broken detection channel
        must not break the sync or the actor."""
        replicas = list(getattr(self._manager, "rollout_replicas", []) or [])
        await asyncio.gather(*(self._probe_and_classify(replica, self._replica_key(replica)) for replica in replicas))

    async def _probe_replica(self, replica) -> Optional[FaultEvidence]:
        """One health pass: GCS actor states (authoritative) + engine probe."""
        actors = list(getattr(replica, "servers", []) or []) + list(getattr(replica, "workers", []) or [])
        for actor in actors:
            # Blocking GCS query — keep it off the manager actor's event loop.
            state_dict = await asyncio.to_thread(_get_actor_state_dict, actor)
            if state_dict is not None and state_dict.get("state") != "ALIVE":
                return FaultEvidence(source="actor_state", error_class="ActorGCSDead")
        try:
            await asyncio.wait_for(replica.check_health(), timeout=PROBE_TIMEOUT_S)
        except asyncio.TimeoutError:
            return FaultEvidence(source="detect_faults", error_class="probe_timeout")
        except Exception as exc:
            inner = _unwrap_ray_exception(exc)
            return FaultEvidence(source="detect_faults", error_class=type(inner).__name__)
        return None

    async def _probe_and_classify(self, replica, key: str) -> None:
        state = self._states.get(key, ReplicaRuntimeState.SERVING)
        if state in (ReplicaRuntimeState.RESTARTING, ReplicaRuntimeState.VERIFYING, ReplicaRuntimeState.DEAD):
            # RESTARTING / VERIFYING are driven by their own recovery task; DEAD is terminal.
            return
        if state == ReplicaRuntimeState.ISOLATED:
            # A failed restart left "retry on next evidence" — a polling
            # round is that evidence.
            evidence = await self._probe_replica(replica)
            if evidence is None:
                logger.info("replica %s: ISOLATED but probe now healthy; leaving state unchanged", key)
                return
            self._on_confirmed(replica, key, evidence)
            return
        # SERVING, or COLD_START awaiting promotion — if it died meanwhile,
        # the poll is the backstop that convicts it.
        evidence = await self._probe_replica(replica)
        if evidence is None:
            return
        if evidence.error_class == "ActorGCSDead":
            # Ray GCS death is authoritative — no re-verification.
            self._on_confirmed(replica, key, evidence)
            return
        reprobe = await self._probe_replica(replica)
        if reprobe is None:
            logger.info("replica %s: first probe failure %s was a transient false alarm", key, evidence.error_class)
            return
        self._on_confirmed(replica, key, reprobe)

    # ── orchestration ───────────────────────────────────────────────────

    def _on_confirmed(self, replica, key: str, evidence: FaultEvidence) -> None:
        logger.warning(
            "replica %s confirmed dead (source=%s, error=%s) — scheduling recovery",
            key,
            evidence.source,
            evidence.error_class,
        )
        self._states[key] = ReplicaRuntimeState.ISOLATED
        self._schedule_recovery(replica, key, evidence)

    def _schedule_recovery(self, replica, key: str, evidence: FaultEvidence) -> None:
        if key in self._recovering:
            logger.info("recovery already in flight for replica %s", key)
            return
        self._recovering.add(key)
        task = asyncio.create_task(self._recover_replica(replica, key, evidence))
        self._recovery_tasks.add(task)

        def _done(finished: asyncio.Task) -> None:
            self._recovering.discard(key)
            self._recovery_tasks.discard(finished)

        task.add_done_callback(_done)

    async def _recover_replica(self, replica, key: str, evidence: FaultEvidence) -> None:
        try:
            await self._handle_confirmed_fault(replica, key, evidence)
        except Exception:
            logger.exception("recovery pipeline failed for replica %s; staying ISOLATED", key)
            self._states[key] = ReplicaRuntimeState.ISOLATED

    async def _handle_confirmed_fault(self, replica, key: str, evidence: FaultEvidence) -> None:
        """ISOLATED -> (budget) -> restart -> weight-source pick -> VERIFYING -> SERVING/DEAD.

        With a checkpoint weight source the rebuilt replica cold-loads it,
        stamps the version, verifies, and re-registers — resuming serving as
        soon as possible, trailing the trainer by at most the save interval
        until the next weight sync. Without one it lands on COLD_START: the
        next weight sync installs its weights and the sync-succeeded
        notification finishes recovery."""
        # Fences before any budget decision: the dead replica leaves the LB and
        # the next sync rebuilds under a new group name (also holds on DEAD).
        await self._isolate(replica, key)
        self._manager.invalidate_group()

        budget = self._budgets.get(key)
        if budget is None:
            budget = RestartBudget(max_restarts=int(getattr(self._config, "max_restarts", 3)))
            self._budgets[key] = budget
        verdict = budget.try_acquire_state()
        if verdict == "exhausted":
            logger.error("restart budget exhausted for replica %s — going DEAD", key)
            self._states[key] = ReplicaRuntimeState.DEAD
            self._check_all_dead()
            return
        if verdict == "cooldown":
            # Pin the state: a direct caller may reach this branch without a
            # prior ISOLATED conviction.
            self._states[key] = ReplicaRuntimeState.ISOLATED
            logger.info(
                "restart for replica %s deferred by the anti-flap cooldown; staying ISOLATED (retry on next poll)",
                key,
            )
            return

        self._states[key] = ReplicaRuntimeState.RESTARTING
        override = await self._plan_weight_source()
        try:
            await replica.restart(weight_source=override)
        except Exception:
            logger.exception("restart failed for replica %s; staying ISOLATED (retry on next polling round)", key)
            self._states[key] = ReplicaRuntimeState.ISOLATED
            return

        self._states[key] = ReplicaRuntimeState.COLD_START

        if override is None:
            # Deferred weight sync: wait for the next weight sync; its
            # completion notification (weight_sync_succeeded) finishes recovery.
            logger.warning(
                "replica %s relaunched without a checkpoint weight source (deferred to the next weight sync)", key
            )
            return

        # A failure here falls back to the deferred-weight-sync semantics: the
        # rebuild is up, so stranding it ISOLATED would fence a healthy
        # replica forever; as COLD_START the next sync finishes recovery.
        try:
            new_server = replica.servers[0]
            await new_server.set_global_steps.remote(override.global_steps)
            self._states[key] = ReplicaRuntimeState.VERIFYING
            await self._verify(replica, expected_version=override.global_steps)
            await self._register(replica, key)
        except Exception:
            logger.exception("post-restart verification failed for replica %s; deferring to the next weight sync", key)
            self._states[key] = ReplicaRuntimeState.COLD_START
            return
        logger.info("replica %s recovered via checkpoint cold start at version %s", key, override.global_steps)

    async def weight_sync_succeeded(self) -> None:
        """The weight sync completed (all-or-nothing): every COLD_START
        replica's weights were installed — verify (health-only; the sync
        stamped the version) and re-register each. A failure returns the
        replica to COLD_START; the next sync's notification retries."""
        for replica in getattr(self._manager, "rollout_replicas", []) or []:
            key = self._replica_key(replica)
            if self._states.get(key) != ReplicaRuntimeState.COLD_START:
                continue
            self._states[key] = ReplicaRuntimeState.VERIFYING
            try:
                await self._verify(replica, expected_version=None)
            except Exception:
                logger.exception("post-sync verification failed for replica %s; retrying on the next sync", key)
                self._states[key] = ReplicaRuntimeState.COLD_START
                continue
            await self._register(replica, key)
            logger.info("replica %s recovered via weight sync (installed, verified, registered)", key)

    async def _isolate(self, replica, key: str) -> None:
        """LB CAS fence + best-effort abort of the dead replica.

        The balancer is embedded in the manager actor (the supervisor's host),
        so these are in-process calls — no cross-actor RPC."""
        lb = self._manager.global_load_balancer
        address = getattr(replica, "_server_address", None)
        handle = getattr(replica, "_server_handle", None)
        if address is not None:
            try:
                lb.remove_server_if_current(address, handle)
            except Exception:
                logger.warning("CAS removal unavailable; falling back to remove_servers", exc_info=True)
                try:
                    lb.remove_servers([address])
                except Exception:
                    logger.warning("remove_servers fallback failed (non-fatal)", exc_info=True)
        try:
            await asyncio.wait_for(replica.abort_all_requests(reject_request=True), timeout=PROBE_TIMEOUT_S)
        except Exception:
            # Expected when the actor is already gone — best-effort by design.
            logger.debug("best-effort abort on dead replica %s failed (expected)", key, exc_info=True)

    async def _plan_weight_source(self) -> Optional[WeightSourceOverride]:
        """Locate a checkpoint for a cold start; on any miss defer the
        replica to the next weight sync. The result rides the restart chain
        explicitly (``replica.restart(weight_source=...)``)."""
        if str(getattr(self._config, "weight_source", "checkpoint")) != "checkpoint":
            return None
        trainer_cfg = getattr(getattr(self._manager, "config", None), "trainer", None)
        checkpoint_dir = getattr(trainer_cfg, "default_local_dir", None)
        if not checkpoint_dir:
            logger.warning(
                "no checkpoint root for the cold start (trainer.default_local_dir unset) "
                "— deferring to the next weight sync"
            )
            return None
        checkpoint_dir = str(checkpoint_dir)

        try:
            model_path, global_steps = await asyncio.wait_for(
                asyncio.to_thread(_locate_cold_start_checkpoint, checkpoint_dir),
                timeout=_LOCATE_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "checkpoint locate timed out after %ss — deferring to the next weight sync", _LOCATE_TIMEOUT_S
            )
            return None
        except Exception as exc:
            logger.warning(
                "checkpoint cold start unavailable (%s: %s) — deferring to the next weight sync",
                type(exc).__name__,
                exc,
            )
            return None
        logger.info("checkpoint cold start from %s (version %s)", model_path, global_steps)
        return WeightSourceOverride(model_path=model_path, global_steps=global_steps)

    async def _verify(self, replica, expected_version: Optional[int]) -> None:
        """VERIFYING: health probe + version stamp check."""
        await asyncio.wait_for(replica.check_health(), timeout=PROBE_TIMEOUT_S)
        if expected_version is not None:
            server = replica.servers[0]
            actual = await server.get_global_steps.remote()
            if actual != expected_version:
                raise ReplicaRestartError(
                    f"version stamp mismatch after restart: expected {expected_version}, got {actual}"
                )

    async def _register(self, replica, key: str) -> None:
        """Re-register with the LB — strictly the last step of recovery.
        In-process calls into the embedded balancer (same actor)."""
        lb = self._manager.global_load_balancer
        lb.add_servers({replica.server_address: replica.server_handle})
        lb.clear_sticky_cache()
        self._states[key] = ReplicaRuntimeState.SERVING

    def _check_all_dead(self) -> None:
        replica_keys = {self._replica_key(r) for r in getattr(self._manager, "rollout_replicas", []) or []}
        dead_keys = {k for k, s in self._states.items() if s == ReplicaRuntimeState.DEAD}
        if replica_keys and dead_keys >= replica_keys:
            message = "all rollout replicas are DEAD"
            logger.critical(message)
            # Fatal lives on the manager (the membership authority); the
            # pulled snapshot propagates it and the next update_weights
            # fails closed.
            self._manager.mark_fatal(message)

    # ── membership authority ────────────────────────────────────────────

    def get_membership(self) -> MembershipSnapshot:
        """Authoritative membership snapshot for the pull reconcile: syncable
        replicas (SERVING / COLD_START — everything else is fenced out of the
        weight-sync group), the group generation, the fatal flag, and the
        awaiting-weight-sync flag.

        Membership growth re-fences the group: a rebuilt replica re-entering
        the syncable set changes the next sync's world_size/ranks, and the
        survivors' cached fenced group (same fence value, stale layout)
        would trip the engines' idempotent-reuse asserts and stall the fresh
        worker's group rendezvous. Convictions already fence shrinks; growth
        is fenced here, at the only point a snapshot is issued."""
        replicas = []
        syncable_keys = set()
        any_serving = False
        any_cold_start = False
        syncable = (ReplicaRuntimeState.SERVING, ReplicaRuntimeState.COLD_START)
        for replica in getattr(self._manager, "rollout_replicas", []) or []:
            key = self._replica_key(replica)
            state = self._states.get(key, ReplicaRuntimeState.SERVING)
            if state in syncable:
                replicas.append(replica)
                syncable_keys.add(key)
            if state is ReplicaRuntimeState.SERVING:
                any_serving = True
            elif state is ReplicaRuntimeState.COLD_START:
                any_cold_start = True
        if self._last_issued_syncable_keys is not None and syncable_keys - self._last_issued_syncable_keys:
            generation = self._manager.invalidate_group()
            logger.info("membership growth re-fenced: weight-sync group generation -> %s", generation)
        self._last_issued_syncable_keys = syncable_keys
        return MembershipSnapshot(
            replicas=replicas,
            group_generation=getattr(self._manager, "_group_generation", 0),
            fatal_error=getattr(self._manager, "_fatal_error", None),
            awaiting_weight_sync=any_cold_start and not any_serving,
        )

    async def probe_membership(self) -> MembershipSnapshot:
        """Fresh membership for the ``update_weights`` entry: run one
        detection round now (bounded), then build the snapshot.

        The periodic poll's view can be up to ``probe_interval_s`` stale: a
        replica that died inside that window would still enter the sync
        group as SERVING and stall the transfer until the commit watchdog
        burns it into a failed attempt. One fresh round convicts it here —
        the entry snapshot fences it out before the group is built, and the
        survivors sync without waiting for the rebuild (which rejoins at
        the following sync). A round that overruns the bound or raises
        falls back to the last snapshot (the commit watchdog remains the
        backstop); driven states (RESTARTING / VERIFYING / DEAD) are
        skipped by :meth:`detect_faults` itself.
        """
        try:
            await asyncio.wait_for(self.detect_faults(), timeout=_ENTRY_ROUND_TIMEOUT_S)
        except asyncio.TimeoutError:
            logger.warning(
                "update_weights-entry detection round did not finish within %.0fs; using the last membership snapshot",
                _ENTRY_ROUND_TIMEOUT_S,
            )
        except Exception:
            logger.warning("update_weights-entry detection round failed (non-fatal)", exc_info=True)
        return self.get_membership()

    # ── helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _replica_key(replica) -> str:
        return f"{replica.replica_rank}{getattr(replica, 'name_suffix', '') or ''}"

    def _validate_applicability(self) -> None:
        """Fail-loud applicability gate."""
        rollout_cfg = self._manager.rollout_config
        disaggregation = getattr(rollout_cfg, "disaggregation", None)
        if disaggregation is not None and getattr(disaggregation, "enabled", False):
            raise NotImplementedError("rollout.restart is not supported with PD disaggregation")
        rollout_name = str(getattr(rollout_cfg, "name", ""))
        if rollout_name == "sglang":
            # The SGLang server-side restart contract (liveness probe / verified
            # shutdown / receipt-aware generate) is not implemented yet.
            raise NotImplementedError("rollout.restart does not support rollout.name='sglang' yet")
        if rollout_name != "vllm":
            # restart needs the server-side FT contract (check_health /
            # shutdown / receipt-aware generate) — only vllm ships it.
            raise NotImplementedError(f"rollout.restart does not support rollout.name={rollout_name!r} yet")
        if int(getattr(rollout_cfg, "nnodes", 1)) != 1:
            raise NotImplementedError("rollout.restart requires a single-node replica (nnodes == 1)")
        model_cfg = getattr(self._manager, "model_config", None)
        lora_rank = int(getattr(model_cfg, "lora_rank", 0) or 0)
        lora_merged = bool(getattr(getattr(model_cfg, "lora", None), "get", lambda *_: False)("merge", False))
        if lora_rank > 0 and not lora_merged:
            raise NotImplementedError("rollout.restart requires merged full weights (unmerged LoRA is not supported)")
        engine_cfg = getattr(rollout_cfg, "checkpoint_engine", None)
        if engine_cfg is not None and str(getattr(engine_cfg, "backend", "")) == "delta_sharded":
            raise NotImplementedError("rollout.restart is not supported with checkpoint_engine.backend=delta_sharded")
        quantization = getattr(rollout_cfg, "quantization", None)
        if quantization is not None and getattr(quantization, "enabled", False):
            raise NotImplementedError(
                "rollout.restart does not support quantized rollout (refit × HF cold start unverified)"
            )

    # ── observability ───────────────────────────────────────────────────

    def get_state(self, replica_key: str) -> ReplicaRuntimeState:
        return self._states.get(replica_key, ReplicaRuntimeState.SERVING)


# ── Driver-side reconcile helpers (wired by the trainer) ──────────────────────


def make_pre_sync_reconcile(checkpoint_manager, manager_handle):
    """The checkpoint manager's pre-sync hook: run one detection round on
    the manager, then pull the resulting authoritative membership snapshot
    into the checkpoint manager (``sync_membership``) at the only boundary
    that reads it — the entry view must be fresh, or a replica that died
    inside the detection window would enter the sync group as SERVING."""

    async def _reconcile() -> None:
        snapshot = await manager_handle.probe_and_get_membership.remote()
        checkpoint_manager.sync_membership(snapshot)

    return _reconcile


def make_sync_succeeded_callback(manager_handle):
    """The checkpoint manager's sync-succeeded callback: fire-and-forget RPC
    into the manager actor so rebuilt replicas walk verification ->
    re-registration. A lost notification is retried by the next sync."""

    def _succeeded() -> None:
        manager_handle.weight_sync_succeeded.remote()

    return _succeeded
