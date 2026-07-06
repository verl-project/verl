"""Shared routing value types consumed by strategies and the Balancer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReplicaInfo:
    """Descriptor of a routable replica.

    Carries only the replica id — the actor handle never leaves the Balancer's
    ``_servers`` dict (detailed_balancer.md §2.3).
    """

    replica_id: str
