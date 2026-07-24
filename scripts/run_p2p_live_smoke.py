#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Run P2P control-plane live smoke against a running Verl + SGLang rollout job.

Requires:
  - Ray cluster with named actors ``sglang_server_{replica_rank}_{node_rank}``
  - Rollout launched with checkpoint_engine.backend=p2p, e.g.:
      actor_rollout_ref.rollout.checkpoint_engine.backend=p2p

Example:
  python scripts/run_p2p_live_smoke.py --replica-rank 0 --num-ranks 1
  python scripts/run_p2p_live_smoke.py --try-all-replicas --num-ranks 1
  python scripts/run_p2p_live_smoke.py --check-bootstrap-only --replica-rank 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import requests  # noqa: E402

from scripts.p2p_control_smoke import run_p2p_control_plane_smoke  # noqa: E402

logger = logging.getLogger(__name__)


def _sglang_actor_name(
    replica_rank: int,
    *,
    node_rank: int = 0,
    name_suffix: str = "",
    reward: bool = False,
    teacher: bool = False,
) -> str:
    if reward:
        return f"sglang_server_reward_{replica_rank}_{node_rank}{name_suffix}"
    if teacher:
        return f"sglang_server_teacher_{replica_rank}_{node_rank}{name_suffix}"
    return f"sglang_server_{replica_rank}_{node_rank}{name_suffix}"


def _format_ray_error(exc: BaseException) -> str:
    try:
        from ray.exceptions import RayTaskError
    except ImportError:
        return str(exc)
    if isinstance(exc, RayTaskError):
        if exc.cause is not None:
            return f"{exc.cause} (ray task: {exc})"
        return str(exc)
    return str(exc)


def _ray_call(server: Any, method: str, **kwargs: Any) -> Any:
    import ray
    from ray.exceptions import RayTaskError

    try:
        return ray.get(getattr(server, method).remote(**kwargs))
    except RayTaskError as exc:
        raise RuntimeError(f"{method} failed: {_format_ray_error(exc)}") from exc


class RaySGLangReplica:
    """SGLangReplica protocol implemented over a named Ray actor."""

    def __init__(self, server: Any) -> None:
        self._server = server

    async def _call(self, method: str, **kwargs: Any) -> Any:
        return await asyncio.to_thread(_ray_call, self._server, method, **kwargs)

    async def abort_all_requests(self) -> None:
        await self._call("abort_all_requests")

    async def get_remote_instance_transfer_engine_info(self, rank: int) -> Any:
        return await self._call("get_remote_instance_transfer_engine_info", rank=rank)

    async def get_parallelism_info(self, rank: int) -> Any:
        return await self._call("get_parallelism_info", rank=rank)

    async def begin_weight_update(self, selector: str = "all") -> dict[str, Any]:
        return await self._call("begin_weight_update", selector=selector)

    async def update_weight_version(self, weight_version: str, abort_all_requests: bool = True) -> dict[str, Any]:
        return await self._call(
            "update_weight_version",
            weight_version=weight_version,
            abort_all_requests=abort_all_requests,
        )

    async def end_weight_update(self) -> dict[str, Any]:
        return await self._call("end_weight_update")

    async def resume_generation(self) -> None:
        await self._call("resume_generation")


def _actor_lookup_kwargs(actor_meta: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"name": actor_meta["name"]}
    namespace = actor_meta.get("namespace")
    if namespace:
        kwargs["namespace"] = namespace
    return kwargs


def _resolve_named_actor(actor_name: str) -> Any:
    import ray
    from ray.util import list_named_actors

    try:
        return ray.get_actor(actor_name)
    except ValueError:
        pass

    matches = [actor for actor in list_named_actors(all_namespaces=True) if actor.get("name") == actor_name]
    if len(matches) == 1:
        return ray.get_actor(**_actor_lookup_kwargs(matches[0]))
    if len(matches) > 1:
        namespaces = sorted({actor.get("namespace", "default") for actor in matches})
        raise RuntimeError(
            f"Multiple actors named {actor_name!r} in namespaces {namespaces}. "
            "Re-run with --ray-namespace <job-namespace>."
        )
    raise ValueError(
        f"Actor {actor_name!r} not found in any Ray namespace. "
        "Run with --list-sglang-actors to inspect live rollout actors."
    )


async def _get_ray_server(actor_name: str) -> Any:
    return await asyncio.to_thread(_resolve_named_actor, actor_name)


def _list_sglang_actors() -> None:
    import ray
    from ray.util import list_named_actors

    ray.init(address="auto", ignore_reinit_error=True)
    actors = list_named_actors(all_namespaces=True)
    matches = sorted(
        (actor for actor in actors if "sglang_server" in actor.get("name", "")),
        key=lambda actor: (actor.get("namespace", ""), actor.get("name", "")),
    )
    if not matches:
        print("No sglang_server* named actors found.")
        return
    namespaces = sorted({actor.get("namespace", "default") for actor in matches})
    for actor in matches:
        print(
            f"{actor.get('name')}\tnamespace={actor.get('namespace', 'default')}\tstate={actor.get('state', 'unknown')}"
        )
    if len(namespaces) == 1:
        print(
            "\nExample:\n"
            f"  python scripts/run_p2p_live_smoke.py "
            f"--ray-namespace {namespaces[0]} --replica-rank 0 --num-ranks 1 --check-bootstrap-only"
        )


async def _check_bootstrap_via_ray(replica: RaySGLangReplica, num_ranks: int) -> dict[str, Any]:
    server_info = await replica._call("get_server_info")
    metadata: dict[int, dict[str, Any]] = {}
    for rank in range(num_ranks):
        transfer_info = await replica.get_remote_instance_transfer_engine_info(rank)
        parallelism_info = await replica.get_parallelism_info(rank)
        if transfer_info is None:
            raise RuntimeError(f"missing transfer engine info for rank {rank}")
        if parallelism_info is None:
            raise RuntimeError(f"missing parallelism config for rank {rank}")
        metadata[rank] = {
            "transfer_info": transfer_info,
            "parallelism_info": parallelism_info,
        }
    return {"server_info": server_info, "metadata": metadata}


def _check_bootstrap_http(host: str, port: int, num_ranks: int, timeout: float) -> dict[str, Any]:
    base = f"http://{host}:{port}"
    health = requests.get(f"{base}/health", timeout=timeout)
    health.raise_for_status()

    metadata: dict[int, dict[str, Any]] = {}
    for rank in range(num_ranks):
        transfer = requests.get(
            f"{base}/get_transfer_engine_info",
            params={"rank": rank},
            timeout=timeout,
        )
        transfer.raise_for_status()
        parallelism = requests.get(
            f"{base}/get_parallelism_config",
            params={"rank": rank},
            timeout=timeout,
        )
        parallelism.raise_for_status()
        metadata[rank] = {
            "transfer_info": transfer.json()["remote_instance_transfer_engine_info"],
            "parallelism_info": parallelism.json(),
        }

    return {"health": health.text, "metadata": metadata}


async def _run_smoke_for_replica(
    *,
    actor_name: str,
    num_ranks: int,
    weight_version: str,
    selector: str,
    check_bootstrap_only: bool,
    bootstrap_host: str,
    bootstrap_port: int | None,
    http_timeout: float,
) -> dict[str, Any]:
    if bootstrap_port is not None:
        logger.info(
            "Checking bootstrap HTTP directly at %s:%s (no Ray actor lookup)",
            bootstrap_host,
            bootstrap_port,
        )
        return _check_bootstrap_http(bootstrap_host, bootstrap_port, num_ranks, http_timeout)

    server = await _get_ray_server(actor_name)
    replica = RaySGLangReplica(server)
    if check_bootstrap_only:
        logger.info("Checking bootstrap metadata via Ray actor %s", actor_name)
        return await _check_bootstrap_via_ray(replica, num_ranks)

    logger.info("Running P2P control-plane smoke via Ray actor %s", actor_name)
    return await run_p2p_control_plane_smoke(
        replica,
        num_ranks=num_ranks,
        weight_version=weight_version,
        selector=selector,
    )


async def _async_main(args: argparse.Namespace) -> int:
    if args.list_sglang_actors:
        _list_sglang_actors()
        return 0

    import ray

    ray.init(
        address=args.ray_address,
        namespace=args.ray_namespace,
        ignore_reinit_error=True,
    )

    replica_ranks: list[int]
    if args.try_all_replicas:
        replica_ranks = list(range(args.max_replicas))
    elif args.replica_rank is not None:
        replica_ranks = args.replica_rank
    else:
        replica_ranks = [0]

    failures = 0
    successes = 0
    for replica_rank in replica_ranks:
        actor_name = _sglang_actor_name(
            replica_rank,
            node_rank=args.node_rank,
            name_suffix=args.name_suffix,
            reward=args.reward_model,
            teacher=args.teacher_model,
        )
        try:
            result = await _run_smoke_for_replica(
                actor_name=actor_name,
                num_ranks=args.num_ranks,
                weight_version=args.weight_version,
                selector=args.selector,
                check_bootstrap_only=args.check_bootstrap_only,
                bootstrap_host=args.bootstrap_host,
                bootstrap_port=args.bootstrap_port,
                http_timeout=args.http_timeout,
            )
        except ValueError as exc:
            if args.try_all_replicas:
                logger.info("Skip replica_rank=%s (%s)", replica_rank, exc)
                continue
            logger.error("Failed for %s: %s", actor_name, exc)
            failures += 1
            continue
        except Exception as exc:
            err = str(exc).lower()
            if args.try_all_replicas and ("does not exist" in err or "not found" in err):
                logger.info("Stop at replica_rank=%s: actor not found", replica_rank)
                break
            if args.verbose:
                logger.exception("Failed for %s", actor_name)
            else:
                logger.error("Failed for %s: %s", actor_name, exc)
            failures += 1
            continue

        successes += 1
        label = actor_name if args.bootstrap_port is None else f"{args.bootstrap_host}:{args.bootstrap_port}"
        print(f"OK {label}")
        print(json.dumps(result, indent=2, default=str))

    if successes == 0:
        logger.error("No replicas passed smoke")
        return 1
    if failures:
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run P2P control-plane live smoke on a running SGLang rollout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--replica-rank",
        type=int,
        action="append",
        dest="replica_rank",
        help="Replica rank to smoke. Repeat for multiple replicas.",
    )
    parser.add_argument(
        "--try-all-replicas",
        action="store_true",
        help="Try replica ranks 0..max-replicas-1 until an actor is missing.",
    )
    parser.add_argument(
        "--max-replicas",
        type=int,
        default=8,
        help="Upper bound when --try-all-replicas is set.",
    )
    parser.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="SGLang node rank within the replica (control RPCs use node_rank=0).",
    )
    parser.add_argument(
        "--name-suffix",
        default="",
        help="Actor name suffix, if rollout was launched with one.",
    )
    parser.add_argument("--reward-model", action="store_true", help="Target reward-model actors.")
    parser.add_argument("--teacher-model", action="store_true", help="Target teacher-model actors.")
    parser.add_argument(
        "--num-ranks",
        type=int,
        default=None,
        help="Number of rollout ranks (tp * dp * pp).",
    )
    parser.add_argument("--weight-version", default="live-smoke-1")
    parser.add_argument("--selector", default="all")
    parser.add_argument(
        "--check-bootstrap-only",
        action="store_true",
        help="Only query bootstrap metadata via Ray actor RPCs; skip begin/end lifecycle.",
    )
    parser.add_argument(
        "--bootstrap-host",
        default="127.0.0.1",
        help="Host for direct HTTP bootstrap checks (--bootstrap-port).",
    )
    parser.add_argument(
        "--bootstrap-port",
        type=int,
        default=None,
        help="Direct HTTP bootstrap check without Ray actor lookup (use host/port from job log).",
    )
    parser.add_argument("--http-timeout", type=float, default=5.0)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument(
        "--ray-namespace",
        default=None,
        help="Ray job namespace. Initializes the driver in that namespace before actor lookup.",
    )
    parser.add_argument(
        "--list-sglang-actors",
        action="store_true",
        help="List live sglang_server* named actors and exit.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    if args.try_all_replicas and args.replica_rank:
        parser.error("Use either --replica-rank or --try-all-replicas, not both.")
    if not args.list_sglang_actors and args.num_ranks is None:
        parser.error("--num-ranks is required unless --list-sglang-actors is set.")
    exit_code = asyncio.run(_async_main(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

#########
# Example:
# Посмотреть живые акторы:
# python scripts/run_p2p_live_smoke.py --list-sglang-actors
#
# Повторить smoke (должно заработать):
# python scripts/run_p2p_live_smoke.py \
#   --replica-rank 0 \
#   --num-ranks 1 \
#   --check-bootstrap-only
#
# Если namespace явный (из шага 1):
# python scripts/run_p2p_live_smoke.py \
#   --ray-namespace <namespace> \
#   --replica-rank 0 \
#   --num-ranks 1 \
#   --check-bootstrap-only
#
# Полный control-plane smoke (без передачи весов):
# python scripts/run_p2p_live_smoke.py \
#   --ray-namespace <namespace>  \
#   --replica-rank 0 \
#   --num-ranks 1
#
# Прямой HTTP (если ты на worker-ноде 10.224.194.231; порт из лога):
# python scripts/run_p2p_live_smoke.py \
#   --bootstrap-host 10.224.194.231 \
#   --bootstrap-port 62736 \
#   --num-ranks 1 \
#   --check-bootstrap-only
