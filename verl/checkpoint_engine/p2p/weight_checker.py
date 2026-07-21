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

"""SGLang ``/weights_checker`` helpers (Miles ``--check-weight-update-equal`` parity)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class P2PWeightChecker:
    """Stateful P2P corruption-test controller (Miles ``--check-weight-update-equal``)."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        allow_quant_error: bool = False,
    ) -> None:
        self._enabled = enabled
        self._allow_quant_error = allow_quant_error
        self._setup_done = False
        self._compared = False

    async def setup(self, replicas: list[Any]) -> None:
        """Snapshot rollout weights and corrupt them before the first P2P sync."""
        if not self._enabled or self._setup_done:
            return
        await setup_weight_checker_snapshots(replicas)
        self._setup_done = True

    async def compare(self, replicas: list[Any]) -> None:
        """Compare rollout weights against the startup snapshot (once, after first sync)."""
        if not self._enabled or self._compared or not self._setup_done:
            return
        await compare_weight_updates(replicas, allow_quant_error=self._allow_quant_error)
        self._compared = True


def _replica_label(replica: Any) -> str:
    return f"replica={getattr(replica, 'replica_rank', '?')}"


def _raise_for_check_weights_result(replica: Any, result: Any, *, action: str) -> None:
    if isinstance(result, BaseException):
        raise RuntimeError(f"weights_checker action={action!r} failed on rollout {_replica_label(replica)}") from result
    if not isinstance(result, dict):
        raise RuntimeError(
            f"weights_checker action={action!r} returned unexpected result on "
            f"rollout {_replica_label(replica)}: {result!r}"
        )
    if result.get("success") is False:
        message = result.get("message", "unknown error")
        raise RuntimeError(f"weights_checker action={action!r} failed on rollout {_replica_label(replica)}: {message}")


async def check_weights_on_replicas(
    replicas: list[Any],
    *,
    action: str,
    allow_quant_error: bool = False,
) -> list[dict[str, Any]]:
    """Fan out ``weights_checker`` to every rollout replica (SGLang only)."""
    if not replicas:
        return []

    missing = [r for r in replicas if not hasattr(r, "check_weights")]
    if missing:
        raise RuntimeError(
            "check_weight_update_equal requires SGLang rollout replicas with check_weights(); "
            f"missing on {len(missing)} replica(s)"
        )

    results = await asyncio.gather(
        *[replica.check_weights(action=action, allow_quant_error=allow_quant_error) for replica in replicas],
        return_exceptions=True,
    )
    outputs: list[dict[str, Any]] = []
    for replica, result in zip(replicas, results, strict=True):
        _raise_for_check_weights_result(replica, result, action=action)
        outputs.append(result)
    return outputs


async def setup_weight_checker_snapshots(replicas: list[Any]) -> None:
    """Miles startup sequence: snapshot reference weights, then corrupt with random."""
    logger.info("Weight checker: snapshot reference tensors on %s rollout replica(s)", len(replicas))
    await check_weights_on_replicas(replicas, action="snapshot")
    logger.info("Weight checker: reset rollout tensors to random values")
    await check_weights_on_replicas(replicas, action="reset_tensors")
    print(
        f"[WeightChecker] snapshot + reset_tensors completed on {len(replicas)} rollout replica(s)",
        flush=True,
    )


async def compare_weight_updates(
    replicas: list[Any],
    *,
    allow_quant_error: bool = False,
) -> list[dict[str, Any]]:
    """Verify post-sync rollout weights match the snapshot taken at setup."""
    logger.info("Weight checker: compare rollout weights (allow_quant_error=%s)", allow_quant_error)
    results = await check_weights_on_replicas(
        replicas,
        action="compare",
        allow_quant_error=allow_quant_error,
    )
    print(
        f"[WeightChecker] compare PASSED on {len(replicas)} rollout replica(s) (allow_quant_error={allow_quant_error})",
        flush=True,
    )
    return results
