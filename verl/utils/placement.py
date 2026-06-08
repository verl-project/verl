# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
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
"""
Placement search and parallelism degree utilities for verl.

Enable rack-aware placement search by setting config.trainer.enable_placement = True.
See docs/msrl_integration/Placement_Explanation.md for detailed documentation.
"""

import copy
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class ParallelismDegrees:
    """Container for parallelism degrees.
    
    Formula: world_size = TP × PP × CP × DP
    Note: EP operates INSIDE DP, not as a separate dimension.
    """
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    expert_tensor_parallel_size: int = 1
    sequence_parallel: bool = False
    data_parallel_size: int = 1
    model_parallel_size: int = 1
    
    @classmethod
    def from_megatron_config(cls, megatron_cfg, n_gpus: int) -> "ParallelismDegrees":
        """Extract parallelism degrees from Megatron config."""
        tp = int(getattr(megatron_cfg, "tensor_model_parallel_size", 1))
        pp = int(getattr(megatron_cfg, "pipeline_model_parallel_size", 1))
        cp = int(getattr(megatron_cfg, "context_parallel_size", 1))
        ep = int(getattr(megatron_cfg, "expert_model_parallel_size", 1))
        etp = int(getattr(megatron_cfg, "expert_tensor_parallel_size", 1) or 1)
        sp = bool(getattr(megatron_cfg, "sequence_parallel", False))
        
        model_parallel_size = tp * pp * cp
        
        if model_parallel_size > 0 and n_gpus % model_parallel_size == 0:
            dp = n_gpus // model_parallel_size
        else:
            dp = max(1, n_gpus // model_parallel_size) if model_parallel_size > 0 else 1
        
        return cls(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            context_parallel_size=cp,
            expert_model_parallel_size=ep,
            expert_tensor_parallel_size=etp,
            sequence_parallel=sp,
            data_parallel_size=dp,
            model_parallel_size=model_parallel_size,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for pool_degrees."""
        return {
            "tensor_model_parallel_size": self.tensor_model_parallel_size,
            "pipeline_model_parallel_size": self.pipeline_model_parallel_size,
            "context_parallel_size": self.context_parallel_size,
            "expert_model_parallel_size": self.expert_model_parallel_size,
            "expert_tensor_parallel_size": self.expert_tensor_parallel_size,
            "sequence_parallel": self.sequence_parallel,
            "data_parallel_size": self.data_parallel_size,
            "model_parallel_size": self.model_parallel_size,
        }
    
    def validate(self, n_gpus: int, n_gpus_per_node: int, nnodes: int) -> List[str]:
        """Validate degrees against cluster configuration. Returns list of messages."""
        messages = []
        
        if n_gpus % self.model_parallel_size != 0:
            messages.append(
                f"ERROR: Total GPUs ({n_gpus}) must be divisible by model_parallel_size "
                f"(TP={self.tensor_model_parallel_size} × PP={self.pipeline_model_parallel_size} "
                f"× CP={self.context_parallel_size} = {self.model_parallel_size})."
            )
        
        if self.expert_model_parallel_size > 1 and self.data_parallel_size % self.expert_model_parallel_size != 0:
            messages.append(
                f"ERROR: EP ({self.expert_model_parallel_size}) must divide DP ({self.data_parallel_size})."
            )
        
        if self.tensor_model_parallel_size > n_gpus_per_node:
            messages.append(
                f"WARNING: TP ({self.tensor_model_parallel_size}) > n_gpus_per_node ({n_gpus_per_node}). "
                f"Cross-node TP may reduce performance."
            )
        
        if self.pipeline_model_parallel_size > 1 and self.pipeline_model_parallel_size > nnodes:
            messages.append(
                f"WARNING: PP ({self.pipeline_model_parallel_size}) > nnodes ({nnodes})."
            )
        
        return messages
    
    def get_summary(self) -> str:
        """Return human-readable summary."""
        return (
            f"TP={self.tensor_model_parallel_size}, PP={self.pipeline_model_parallel_size}, "
            f"CP={self.context_parallel_size}, EP={self.expert_model_parallel_size}, "
            f"DP={self.data_parallel_size}"
        )


def extract_degrees_from_config(config, n_gpus: int) -> Optional[ParallelismDegrees]:
    """Extract parallelism degrees from verl config."""
    actor_cfg = getattr(config.actor_rollout_ref, "actor", None)
    megatron_cfg = getattr(actor_cfg, "megatron", None) if actor_cfg is not None else None
    
    if megatron_cfg is None:
        return None
    
    return ParallelismDegrees.from_megatron_config(megatron_cfg, n_gpus)


def get_cluster_placement_racks() -> Dict[str, List[str]]:
    """Discover rack-to-node placement from Ray nodes (looks for RACK_* resources)."""
    try:
        import ray
        nodes = ray.nodes()
    except Exception as e:
        logger.warning(f"Failed to query Ray nodes: {e}")
        return {}
    
    rack_map = defaultdict(list)
    
    for node in nodes:
        if not node.get("Alive", False):
            continue

        resources = node.get("Resources", {})
        # Skip CPU-only nodes (e.g. head) that may still declare RACK_* resources.
        if resources.get("GPU", 0) == 0 and resources.get("NPU", 0) == 0:
            continue

        node_id = node.get("NodeID", "")

        # Deterministic pick when a node declares multiple RACK_* resources (same string Ray uses).
        rack_keys = sorted(
            (k for k in resources if k.upper().startswith("RACK_")),
            key=lambda k: k.upper(),
        )
        if rack_keys:
            rack_map[rack_keys[0]].append(node_id)

    return dict(rack_map)


def get_ray_declared_rack_resource_keys() -> set[str]:
    """Exact RACK_* resource name strings on all alive Ray nodes (case-sensitive, as in ``ray start``)."""
    try:
        import ray
        nodes = ray.nodes()
    except Exception as e:
        logger.warning("Failed to query Ray nodes for rack resource discovery: %s", e)
        return set()

    keys: set[str] = set()
    for node in nodes:
        if not node.get("Alive", False):
            continue
        for key in node.get("Resources", {}):
            if key.upper().startswith("RACK_"):
                keys.add(key)
    return keys


def apply_rack_bundle_resource_map(
    rack_ids: List[str],
    resource_map: Optional[Dict[str, str]],
) -> List[str]:
    """Remap placement rack ids to PG bundle resource names when they must differ from ``selected_nodes[i][0]``."""
    if not resource_map:
        return list(rack_ids)
    return [resource_map.get(rid, rid) for rid in rack_ids]


def get_ray_alive_node_ids() -> set[str]:
    """Ray ``NodeID`` strings for all alive nodes (same as ``selected_nodes[i][1]`` from placement search)."""
    try:
        import ray
        nodes = ray.nodes()
    except Exception as e:
        logger.warning("Failed to query Ray nodes for node id validation: %s", e)
        return set()

    return {n.get("NodeID", "") for n in nodes if n.get("Alive", False) and n.get("NodeID")}


def validate_ray_node_ids_exist(node_ids: List[str]) -> None:
    """Ensure each id is an alive Ray node's ``NodeID`` (used for placement-group node pinning)."""
    if not node_ids:
        return
    empty_slots = [i for i, nid in enumerate(node_ids) if not nid]
    if empty_slots:
        raise ValueError(
            f"placement_pin_node_affinity: empty Ray NodeID at slot(s) {empty_slots} in selected_nodes"
        )
    try:
        import ray  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "placement_pin_node_affinity: cannot validate NodeIDs because Ray is not importable"
        ) from e

    alive = get_ray_alive_node_ids()
    if not alive:
        raise ValueError(
            "placement_pin_node_affinity: ray.nodes() reported no alive NodeIDs; cannot validate placement targets"
        )

    missing = [nid for nid in node_ids if nid and nid not in alive]
    if not missing:
        return
    raise ValueError(
        "placement_pin_node_affinity: Ray NodeID(s) "
        f"{missing} are not among alive nodes. "
        f"Placement search must use ids from ray.nodes()[*]['NodeID']. "
        f"Alive NodeIDs (sample): {sorted(x for x in alive if x)[:32]}"
    )


def validate_placement_targets_distinct_ray_nodes(node_ids: List[str]) -> None:
    """Each PG maps to one machine; duplicate Ray NodeIDs would pack multiple STRICT_PACK groups onto the same node."""
    nonempty = [nid for nid in node_ids if nid]
    if len(nonempty) <= 1:
        return
    if len(set(nonempty)) != len(nonempty):
        raise ValueError(
            "placement_pin_node_affinity: duplicate Ray NodeID among placement targets "
            f"{node_ids}; each nnodes slot must be a distinct node for multi-node training"
        )


def validate_rack_bundle_resource_labels(rack_labels: List[str]) -> None:
    """Ensure PG bundle requests use resource names actually declared on the cluster (exact match)."""
    if not rack_labels:
        return
    available = get_ray_declared_rack_resource_keys()
    missing = [r for r in rack_labels if r and r not in available]
    if not missing:
        return
    raise ValueError(
        "placement_pin_rack_resources: placement-group bundle rack resource(s) "
        f"{missing} are not declared on any alive Ray node. Names must match exactly (case-sensitive) "
        'what `ray start --resources` registers, e.g. \'{"RACK_A": 1}\'. '
        f"Available RACK_* keys on this cluster: {sorted(available)}"
    )


def _can_place_rack_strict_pack_pg(
    res: Dict[str, float],
    n_accel: int,
    rack_key: str,
    max_colocate_count: int,
    rack_fraction: float,
) -> bool:
    """One STRICT_PACK PG on a node: n bundles, each with CPU, accelerator, and rack_fraction of RACK_*."""
    if res.get("CPU", 0) < n_accel * max_colocate_count:
        return False
    if rack_key and res.get(rack_key, 0) < n_accel * rack_fraction:
        return False
    if res.get("GPU", 0) >= n_accel:
        return True
    if res.get("NPU", 0) >= n_accel:
        return True
    return False


def _consume_rack_strict_pack_pg(
    res: Dict[str, float],
    n_accel: int,
    rack_key: str,
    max_colocate_count: int,
    rack_fraction: float,
) -> None:
    res["CPU"] = float(res.get("CPU", 0)) - n_accel * max_colocate_count
    if res.get("GPU", 0) >= n_accel:
        res["GPU"] = float(res.get("GPU", 0)) - n_accel
    else:
        res["NPU"] = float(res.get("NPU", 0)) - n_accel
    if rack_key:
        res[rack_key] = float(res.get(rack_key, 0)) - n_accel * rack_fraction


def validate_rack_pinned_pools_feasible(
    resource_pool_spec: Dict[str, List[int]],
    pool_rack_labels: Optional[Dict[str, List[str]]],
    node_available_resources: Dict[str, Dict[str, float]],
    max_colocate_count: int = 3,
    rack_fraction: float = 1e-4,
) -> None:
    """Fail fast before ``pg.ready()`` if RACK-tagged PGs cannot fit per-node (CPU/GPU/NPU + RACK).

    Mirrors ``RayResourcePool.get_placement_groups`` bundle shape: each bundle requests ``rack_fraction``
    of the rack custom resource (same as accelerator_type). Greedy placement: largest PGs first.
    """
    if not pool_rack_labels:
        return

    slots: List[Tuple[int, str, str]] = []
    for pool_name, process_on_nodes in resource_pool_spec.items():
        rack_list = pool_rack_labels.get(pool_name)
        if not rack_list or len(rack_list) != len(process_on_nodes):
            continue
        for idx, n in enumerate(process_on_nodes):
            rkey = rack_list[idx] if idx < len(rack_list) else ""
            if not rkey:
                continue
            slots.append((n, rkey, f"{pool_name}[{idx}]"))

    if not slots:
        return

    state: Dict[str, Dict[str, float]] = copy.deepcopy(node_available_resources)
    for nid in state:
        state[nid] = {k: float(v) for k, v in state[nid].items()}

    slots.sort(key=lambda x: -x[0])

    for n_accel, rack_key, label in slots:
        placed = False
        for node_id, res in state.items():
            if _can_place_rack_strict_pack_pg(res, n_accel, rack_key, max_colocate_count, rack_fraction):
                _consume_rack_strict_pack_pg(res, n_accel, rack_key, max_colocate_count, rack_fraction)
                placed = True
                break
        if not placed:
            sample_nodes = list(state.keys())[:12]
            raise ValueError(
                "placement_pin_rack_resources: cannot place a STRICT_PACK placement group on any single node "
                f"for slot {label} (need {n_accel} GPU or NPU on one node, "
                f"CPU>={n_accel * max_colocate_count}, and rack {rack_key!r} >= {n_accel * rack_fraction}). "
                "Total cluster accelerators can be enough while no one node satisfies CPU/GPU/NPU+RACK together. "
                f"Node ids in this check (sample): {sample_nodes}"
            )


def get_rack_capacities(rack_map: Dict[str, List[str]], npus_per_node: int) -> Dict[str, int]:
    """Get capacity of each rack in NPUs."""
    return {rack_id: len(nodes) * npus_per_node for rack_id, nodes in rack_map.items()}


def _select_nodes_group_within_rack(
    total_npus: int,
    npus_per_node: int,
    rack_map: Dict[str, List[str]],
    group_size: int,
    description: str,
) -> Optional[List[Tuple[str, str]]]:
    """Allocate nodes keeping groups of group_size NPUs within the same rack."""
    if group_size <= 1 or group_size > total_npus:
        return None

    if total_npus % group_size != 0:
        logger.debug(f"  ✗ {description}: total NPUs {total_npus} not divisible by group size {group_size}")
        return None

    num_nodes_needed = (total_npus + npus_per_node - 1) // npus_per_node
    num_groups = total_npus // group_size

    rack_state: Dict[str, List[Dict[str, Union[str, int]]]] = {
        rack_id: [{"id": node_id, "remaining": npus_per_node, "used": 0} for node_id in rack_map[rack_id]]
        for rack_id in rack_map
    }
    rack_remaining_capacity = {rack_id: len(nodes) * npus_per_node for rack_id, nodes in rack_map.items()}
    used_racks = set()

    for group_idx in range(num_groups):
        candidate_racks = [rid for rid in rack_state if rack_remaining_capacity[rid] >= group_size]
        if not candidate_racks:
            logger.debug(f"  ✗ {description}: insufficient rack capacity for group {group_idx}")
            return None

        candidate_racks.sort(key=lambda rid: (rid not in used_racks, -rack_remaining_capacity[rid], rid))
        selected_rack = candidate_racks[0]
        group_remaining = group_size

        for node_state in rack_state[selected_rack]:
            if group_remaining <= 0:
                break
            remaining = int(node_state["remaining"])
            if remaining <= 0:
                continue
            allocation = min(remaining, group_remaining)
            node_state["remaining"] = remaining - allocation
            node_state["used"] = int(node_state["used"]) + allocation
            group_remaining -= allocation

        if group_remaining > 0:
            logger.debug(f"  ✗ {description}: rack {selected_rack} cannot accommodate group size {group_size}")
            return None

        rack_remaining_capacity[selected_rack] -= group_size
        used_racks.add(selected_rack)

    selected_nodes: List[Tuple[str, str]] = []
    per_rack_usage: Dict[str, Dict[str, int]] = {}

    for rack_id in sorted(rack_state.keys()):
        rack_usage: Dict[str, int] = {}
        for node_state in rack_state[rack_id]:
            used = int(node_state["used"])
            if used > 0:
                node_id = str(node_state["id"])
                selected_nodes.append((rack_id, node_id))
                rack_usage[node_id] = used
        if rack_usage:
            per_rack_usage[rack_id] = rack_usage

    if len(selected_nodes) != num_nodes_needed:
        logger.debug(f"  ✗ {description}: node count mismatch (expected {num_nodes_needed}, got {len(selected_nodes)})")
        return None

    logger.info(f"  ✓ Strategy: {description}")
    for rack_id, usage in per_rack_usage.items():
        logger.info(f"    {rack_id}: {len(usage)} nodes, {sum(usage.values())} NPUs")

    return selected_nodes


def select_balanced_nodes(
    total_npus: int, 
    npus_per_node: int, 
    rack_map: Dict[str, List[str]],
    degrees: Optional[ParallelismDegrees] = None
) -> List[Tuple[str, str]]:
    """Select nodes with rack-aware placement search.
    
    Hierarchical approach - try progressively larger groups within rack:
        1. Single-rack (if all NPUs fit in one rack)
        2. TP within rack (mandatory, warns if fails)
        3. TP×CP within rack (if fits, use it; otherwise CP spans racks)
        4. TP×CP×PP within rack (if fits, use it; otherwise PP spans racks)
        5. DP replicas within rack (each replica = TP×CP×PP NPUs)
        6. Sequential fallback (fill largest racks first)
        7. Symmetric fallback (even distribution)
    
    The largest successful grouping is used. Higher communication dimensions
    (TP, CP) are prioritized to stay within rack for bandwidth.
    
    Note: EP is NOT considered for rack placement as it operates within DP dimension.
    """
    num_nodes_needed = (total_npus + npus_per_node - 1) // npus_per_node
    sorted_racks = sorted(rack_map.keys())

    if len(sorted_racks) == 0:
        raise ValueError("No available rack in placement map")

    rack_capacities = get_rack_capacities(rack_map, npus_per_node)

    logger.info(f"Resource requirements: {total_npus} NPUs ({num_nodes_needed} nodes)")
    for rack_id in sorted_racks:
        logger.info(f"  {rack_id}: {rack_capacities[rack_id]} NPUs ({len(rack_map[rack_id])} nodes)")

    # Try single rack first
    suitable_racks = [(rid, cap) for rid, cap in rack_capacities.items() if cap >= total_npus]
    if suitable_racks:
        best_rack = min(suitable_racks, key=lambda x: x[1])
        rack_id = best_rack[0]
        logger.info(f"✓ Strategy: single rack allocation ({rack_id})")
        
        selected_nodes = []
        available_nodes = rack_map[rack_id]
        for i in range(num_nodes_needed):
            if i < len(available_nodes):
                selected_nodes.append((rack_id, available_nodes[i]))
            else:
                raise ValueError(f"Rack {rack_id} has insufficient nodes")
        return selected_nodes

    def attempt_group(group_size: int, description: str) -> Optional[List[Tuple[str, str]]]:
        if group_size <= 1:
            return None
        return _select_nodes_group_within_rack(total_npus, npus_per_node, rack_map, group_size, description)

    selection: Optional[List[Tuple[str, str]]] = None

    if degrees:
        tp_size = degrees.tensor_model_parallel_size
        cp_size = degrees.context_parallel_size
        pp_size = degrees.pipeline_model_parallel_size
        dp_size = degrees.data_parallel_size
        # Note: EP not used for rack placement as it operates within DP

        logger.info(f"Parallelization: {degrees.get_summary()}")

        # Hierarchical approach: try progressively larger groups within rack
        # Each level includes all previous levels
        
        # Level 1: TP within rack (mandatory - warn if fails)
        current_group_size = tp_size
        if tp_size > 1:
            tp_selection = attempt_group(current_group_size, "TP within rack")
            if not tp_selection:
                logger.warning("⚠️ Unable to place TP groups within a single rack.")
            else:
                selection = tp_selection

        # Level 2: TP×CP within rack
        if cp_size > 1:
            current_group_size = tp_size * cp_size
            cp_selection = attempt_group(current_group_size, "TP×CP within rack")
            if cp_selection:
                selection = cp_selection
                logger.info(f"  → CP groups fit within rack (group_size={current_group_size})")

        # Level 3: TP×CP×PP within rack
        if pp_size > 1:
            current_group_size = tp_size * cp_size * pp_size
            pp_selection = attempt_group(current_group_size, "TP×CP×PP within rack")
            if pp_selection:
                selection = pp_selection
                logger.info(f"  → PP stages fit within rack (group_size={current_group_size})")
            else:
                logger.info(f"  → PP will span racks (TP×CP×PP={current_group_size} > rack capacity)")

        # Level 4: TP×CP×PP×DP within rack (= all NPUs in one rack, already handled above)
        # If we reach here and DP > 1, DP will span racks
        if dp_size > 1:
            # model_parallel_size = TP×CP×PP, this is what stays within each DP replica
            model_parallel_size = tp_size * cp_size * pp_size
            full_group_size = model_parallel_size * dp_size  # = total_npus
            if full_group_size <= total_npus:
                # Try to fit each DP replica (model_parallel_size) within a rack
                dp_selection = attempt_group(model_parallel_size, "DP replicas within rack")
                if dp_selection:
                    selection = dp_selection
                    logger.info(f"  → Each DP replica ({model_parallel_size} NPUs) fits within rack")
                else:
                    logger.info(f"  → DP will span racks")

    if selection:
        return selection

    # Sequential fallback (fill rack A, then B, etc.)
    sequential_selection = _sequential_allocation(total_npus, npus_per_node, num_nodes_needed, rack_map, rack_capacities, sorted_racks)
    if sequential_selection:
        return sequential_selection

    # Symmetric fallback (even distribution)
    return _symmetric_allocation(total_npus, npus_per_node, num_nodes_needed, rack_map, rack_capacities, sorted_racks)


def _sequential_allocation(
    total_npus: int,
    npus_per_node: int,
    num_nodes_needed: int,
    rack_map: Dict[str, List[str]],
    rack_capacities: Dict[str, int],
    sorted_racks: List[str],
) -> Optional[List[Tuple[str, str]]]:
    """Sequential allocation: fill rack A completely, then B, then C, etc.
    
    This minimizes cross-rack communication by keeping most work in fewer racks.
    """
    total_capacity = sum(rack_capacities.values())
    if total_capacity < total_npus:
        return None

    selected_nodes: List[Tuple[str, str]] = []
    remaining_nodes = num_nodes_needed
    per_rack_usage: Dict[str, int] = {}

    # Sort racks by capacity (largest first) for better packing
    racks_by_capacity = sorted(sorted_racks, key=lambda r: -rack_capacities[r])

    for rack_id in racks_by_capacity:
        if remaining_nodes <= 0:
            break
        
        available_nodes = rack_map[rack_id]
        nodes_from_rack = min(len(available_nodes), remaining_nodes)
        
        for i in range(nodes_from_rack):
            selected_nodes.append((rack_id, available_nodes[i]))
        
        if nodes_from_rack > 0:
            per_rack_usage[rack_id] = nodes_from_rack
        remaining_nodes -= nodes_from_rack

    if remaining_nodes > 0:
        return None

    logger.info("✓ Strategy: Sequential allocation (fill largest racks first)")
    for rack_id, count in per_rack_usage.items():
        logger.info(f"    {rack_id}: {count} nodes, {count * npus_per_node} NPUs")

    return selected_nodes


def _symmetric_allocation(
    total_npus: int,
    npus_per_node: int,
    num_nodes_needed: int,
    rack_map: Dict[str, List[str]],
    rack_capacities: Dict[str, int],
    sorted_racks: List[str],
) -> List[Tuple[str, str]]:
    """Symmetric fallback allocation across racks."""
    logger.info("✓ Strategy: Symmetric allocation across racks")
    # Prefer racks with more nodes (avoid arbitrary alphabetical ordering).
    sorted_racks = sorted(sorted_racks, key=lambda r: -len(rack_map[r]))

    total_capacity = sum(rack_capacities.values())
    if total_capacity < total_npus:
        raise ValueError(f"Insufficient capacity: need {total_npus} NPUs, have {total_capacity}")

    num_racks_available = len(sorted_racks)
    min_racks_needed = None
    nodes_per_rack = None
    remaining_nodes = 0

    # Try perfect symmetry
    for num_racks_to_use in range(1, num_racks_available + 1):
        if num_nodes_needed % num_racks_to_use == 0:
            candidate_nodes_per_rack = num_nodes_needed // num_racks_to_use
            suitable_count = sum(
                1 for rack_id in sorted_racks[:num_racks_to_use]
                if rack_capacities[rack_id] >= candidate_nodes_per_rack * npus_per_node
                and len(rack_map[rack_id]) >= candidate_nodes_per_rack
            )
            if suitable_count == num_racks_to_use:
                min_racks_needed = num_racks_to_use
                nodes_per_rack = candidate_nodes_per_rack
                break

    # Near-symmetric fallback
    if min_racks_needed is None:
        for num_racks_to_use in range(1, num_racks_available + 1):
            base_nodes = num_nodes_needed // num_racks_to_use
            extra_nodes = num_nodes_needed % num_racks_to_use
            
            can_fit = True
            for i in range(num_racks_to_use):
                rack_id = sorted_racks[i]
                nodes_for_this_rack = base_nodes + (1 if i < extra_nodes else 0)
                if (rack_capacities[rack_id] < nodes_for_this_rack * npus_per_node or
                    len(rack_map[rack_id]) < nodes_for_this_rack):
                    can_fit = False
                    break

            if can_fit:
                min_racks_needed = num_racks_to_use
                nodes_per_rack = base_nodes
                remaining_nodes = extra_nodes
                break

    if min_racks_needed is None:
        min_racks_needed = num_racks_available
        nodes_per_rack = num_nodes_needed // num_racks_available
        remaining_nodes = num_nodes_needed % num_racks_available

    logger.info(f"  {num_nodes_needed} nodes → {min_racks_needed} racks")

    selected_nodes = []
    for rack_idx, rack_id in enumerate(sorted_racks[:min_racks_needed]):
        available_nodes = rack_map[rack_id]
        nodes_to_take = nodes_per_rack + (1 if rack_idx < remaining_nodes else 0)

        if nodes_to_take > len(available_nodes):
            raise ValueError(f"Rack {rack_id} has insufficient nodes")

        logger.info(f"  {rack_id}: {nodes_to_take} nodes")

        for node_id in available_nodes[:nodes_to_take]:
            selected_nodes.append((rack_id, node_id))
            if len(selected_nodes) >= num_nodes_needed:
                break
        if len(selected_nodes) >= num_nodes_needed:
            break

    return selected_nodes


def compute_process_on_nodes(
    n_gpus_per_node: int,
    nnodes: int,
    degrees: Optional[ParallelismDegrees] = None,
    enable_placement: bool = False,
) -> Tuple[List[int], Optional[List[Tuple[str, str]]]]:
    """Compute process_on_nodes with optional placement search."""
    n_gpus = n_gpus_per_node * nnodes
    process_on_nodes = [n_gpus_per_node] * nnodes
    selected_nodes = None
    
    if not enable_placement:
        return process_on_nodes, selected_nodes
    
    rack_map = get_cluster_placement_racks()
    
    if not rack_map:
        logger.info("No rack information available, using flat placement")
        return process_on_nodes, selected_nodes
    
    logger.info(f"Placement search enabled. Found {len(rack_map)} racks.")
    
    try:
        selected_nodes = select_balanced_nodes(
            total_npus=n_gpus,
            npus_per_node=n_gpus_per_node,
            rack_map=rack_map,
            degrees=degrees,
        )
        logger.info(f"Placement selection: {len(selected_nodes)} nodes")
    except Exception as e:
        logger.warning(f"Placement search failed: {e}. Using flat placement.")
        selected_nodes = None
    
    return process_on_nodes, selected_nodes


def init_pool_degrees_and_spec(
    config,
    global_pool_id: str = "global_pool",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[int]], Optional[List[Tuple[str, str]]]]:
    """Initialize pool_degrees and resource_pool_spec from config.

    Set config.trainer.enable_placement = True to enable placement search.
    With config.trainer.placement_pin_rack_resources, main_ppo passes rack ids from
    selected_nodes into Ray placement-group bundles as custom resources; names are
    validated against the live Ray cluster (or remapped via placement_rack_bundle_resource_map).
    With config.trainer.placement_pin_node_affinity, placement groups use Ray
    ``_soft_target_node_id`` with ``selected_nodes[i][1]`` (Ray NodeID).
    """
    n_gpus_per_node = config.trainer.n_gpus_per_node
    nnodes = config.trainer.nnodes
    n_gpus = n_gpus_per_node * nnodes
    
    # Default: placement search disabled for backward compatibility
    from omegaconf import OmegaConf

    enable_placement = OmegaConf.select(config, "trainer.enable_placement", default=False)
    
    degrees = extract_degrees_from_config(config, n_gpus)
    
    pool_degrees: Dict[str, Dict[str, Any]] = {}
    
    if degrees is not None:
        messages = degrees.validate(n_gpus, n_gpus_per_node, nnodes)
        for msg in messages:
            if msg.startswith("ERROR"):
                raise ValueError(msg)
            else:
                logger.warning(msg)
        
        pool_degrees[global_pool_id] = degrees.to_dict()
        logger.info(f"Parallelism degrees: {degrees.get_summary()}")
    
    process_on_nodes, selected_nodes = compute_process_on_nodes(
        n_gpus_per_node=n_gpus_per_node,
        nnodes=nnodes,
        degrees=degrees,
        enable_placement=enable_placement,
    )
    
    resource_pool_spec = {global_pool_id: process_on_nodes}
    
    return pool_degrees, resource_pool_spec, selected_nodes
