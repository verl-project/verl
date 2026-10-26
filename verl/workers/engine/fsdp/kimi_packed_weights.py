"""Export packed Kimi experts to the colocated rollout expert layout."""

from functools import lru_cache
from itertools import product

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

KIMI_PACKED_LOCAL_MARKER = ".__verl_packed_local__."
_RESHARD_BUFFER_BYTES = 64 * 1024 * 1024


@lru_cache(maxsize=32)
def _reshard_plan(shape, mesh_shape, ranks, shard_dims, target_ep, rank, element_size, buffer_bytes):
    """Plan disjoint expert/matrix tiles; no weight or metadata gather is needed."""
    owners = {}
    for source_rank, coordinate in zip(ranks, product(*(range(size) for size in mesh_shape)), strict=True):
        sizes, offsets = list(shape), [0, 0, 0]
        for dim, size, index in zip(shard_dims, mesh_shape, coordinate, strict=True):
            if dim < 0:
                continue
            chunk = (sizes[dim] + size - 1) // size
            offset = min(index * chunk, sizes[dim])
            offsets[dim] += offset
            sizes[dim] = min(chunk, sizes[dim] - offset)
        if all(sizes):
            owners.setdefault((*offsets, *sizes), []).append(source_rank)

    world_size, local_experts = len(ranks), shape[0] // target_ep
    sends, receives = {}, {}
    send_degree, recv_degree = [0] * world_size, [0] * world_size
    largest = 0
    target_start = rank % target_ep * local_experts
    for (expert_start, row_start, column_start, expert_count, row_count, column_count), replicas in owners.items():
        first_ep = expert_start // local_experts
        last_ep = (expert_start + expert_count - 1) // local_experts
        for replica_start in range(0, world_size, target_ep):
            for target in range(first_ep, last_ep + 1):
                destination = replica_start + target
                # Spread replicated source shards over destinations; prefer a
                # local copy when the destination already owns this exact tile.
                source = (
                    destination
                    if destination in replicas
                    else replicas[(replica_start // target_ep + target) % len(replicas)]
                )
                begin = max(expert_start, target * local_experts)
                end = min(expert_start + expert_count, (target + 1) * local_experts)
                count = (end - begin) * row_count * column_count
                send_degree[source] += 1
                recv_degree[destination] += 1
                largest = max(largest, count)
                if source == rank:
                    sends[destination] = ((begin - expert_start) * row_count * column_count, count)
                if destination == rank:
                    receives[source] = (begin - target_start, row_start, column_start, row_count, column_count, count)
    degree = max(*send_degree, *recv_degree)
    chunk = max(1, buffer_bytes // (element_size * degree))
    return tuple(sorted(sends.items())), tuple(sorted(receives.items())), chunk, (largest + chunk - 1) // chunk


def _exchange_expert_shards(param, device, target_ep):
    """Send source eFSDP shards directly to their target expert owners."""
    mesh = param.device_mesh
    shard_dims = tuple(placement.dim if isinstance(placement, Shard) else -1 for placement in param.placements)
    if any(dim not in (-1, 0, 1, 2) for dim in shard_dims) or any(
        not isinstance(placement, (Shard, Replicate)) for placement in param.placements
    ):
        raise ValueError("Kimi reshard requires expert/matrix Shard(0/1/2) or Replicate placements")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    sends, receives, chunk, rounds = _reshard_plan(
        tuple(param.shape),
        tuple(mesh.shape),
        tuple(mesh.mesh.flatten().tolist()),
        shard_dims,
        target_ep,
        rank,
        param.element_size(),
        _RESHARD_BUFFER_BYTES,
    )
    local = param.to(device, non_blocking=True).to_local().detach().contiguous().view(-1)
    target = local.new_empty((param.shape[0] // target_ep, *param.shape[1:]))
    for step in range(rounds):
        begin = step * chunk
        send_sizes, recv_sizes = [0] * world_size, [0] * world_size
        pieces = []
        for peer, (offset, count) in sends:
            size = max(0, min(chunk, count - begin))
            send_sizes[peer] = size
            pieces.append(local.narrow(0, offset + min(begin, count), size))
        for peer, (_, _, _, _, _, count) in receives:
            recv_sizes[peer] = max(0, min(chunk, count - begin))
        send_buffer = torch.cat(pieces) if pieces else local.new_empty(0)
        recv_buffer = local.new_empty(sum(recv_sizes))
        dist.all_to_all_single(recv_buffer, send_buffer, recv_sizes, send_sizes)
        offset = 0
        for peer, (expert, row, column, rows, columns, _) in receives:
            size = recv_sizes[peer]
            position, stop = begin, begin + size
            expert_elements = rows * columns
            while position < stop:
                index, within = divmod(position, expert_elements)
                complete = (stop - position) // expert_elements if within == 0 else 0
                if complete:
                    length = complete * expert_elements
                    destination = target[
                        expert + index : expert + index + complete, row : row + rows, column : column + columns
                    ]
                    destination.copy_(recv_buffer.narrow(0, offset, length).view(complete, rows, columns))
                else:
                    local_row, local_column = divmod(within, columns)
                    complete_rows = min(rows - local_row, (stop - position) // columns) if local_column == 0 else 0
                    if complete_rows:
                        length = complete_rows * columns
                        target[
                            expert + index, row + local_row : row + local_row + complete_rows, column : column + columns
                        ].copy_(recv_buffer.narrow(0, offset, length).view(complete_rows, columns))
                    else:
                        length = min(stop - position, columns - local_column)
                        target[
                            expert + index, row + local_row, column + local_column : column + local_column + length
                        ].copy_(recv_buffer.narrow(0, offset, length))
                position += length
                offset += length
    return target


def export_kimi_packed_local_param(name: str, param: DTensor, device, rollout_layout: dict):
    """Keep the direct export for matching layouts; otherwise exchange shards."""
    if not isinstance(param, DTensor) or param.ndim != 3:
        raise TypeError(f"Kimi packed export requires a 3-D DTensor: {name}")
    if rollout_layout is None:
        raise ValueError("Kimi packed export requires the rollout layout")
    ep = int(rollout_layout["expert_parallel_size"])
    tp = int(rollout_layout["tensor_parallel_size"])
    dp = int(rollout_layout["data_parallel_size"])
    pp = int(rollout_layout["pipeline_model_parallel_size"])
    if min(ep, tp, dp) <= 0 or ep != tp * dp or pp != 1:
        raise ValueError("Kimi packed export requires rollout EP=TP*DP>0 and PP=1")

    placements = param.placements
    expert_dims = [
        index for index, placement in enumerate(placements) if isinstance(placement, Shard) and placement.dim == 0
    ]
    if len(expert_dims) != 1:
        raise ValueError(f"Kimi packed export requires one expert Shard(0): {placements}")
    expert_dim = expert_dims[0]
    mesh = param.device_mesh.mesh.cpu()
    world_size = dist.get_world_size()
    if mesh.numel() != world_size or world_size % ep:
        raise ValueError("Kimi packed export requires rollout EP to divide the full actor mesh")
    if param.shape[0] % ep:
        raise ValueError(f"Kimi experts must be divisible by EP: experts={param.shape[0]}, EP={ep}")
    if mesh.shape[expert_dim] != ep:
        local = _exchange_expert_shards(param, device, ep)
        return f"{name}{KIMI_PACKED_LOCAL_MARKER}{(dist.get_rank() % ep) * local.shape[0]}", local
    coordinate_shape = [1] * mesh.ndim
    coordinate_shape[expert_dim] = ep
    expected_ranks = torch.arange(ep).reshape(coordinate_shape).expand(mesh.shape)
    if not torch.equal(mesh.remainder(ep), expected_ranks):
        local = _exchange_expert_shards(param, device, ep)
        return f"{name}{KIMI_PACKED_LOCAL_MARKER}{(dist.get_rank() % ep) * local.shape[0]}", local

    export_placements = tuple(
        placement if index == expert_dim else Replicate() for index, placement in enumerate(placements)
    )
    exported = param.to(device, non_blocking=True)
    if export_placements != placements:
        exported = exported.redistribute(param.device_mesh, export_placements)
    _, offset = compute_local_shape_and_global_offset(param.shape, param.device_mesh, export_placements)
    local = exported.to_local().detach().contiguous()
    expected_start = (dist.get_rank() % ep) * (param.shape[0] // ep)
    if offset[0] != expected_start or local.shape[0] != param.shape[0] // ep:
        raise RuntimeError(f"Unexpected Kimi local expert interval: {name}, offset={offset}, shape={local.shape}")
    return f"{name}{KIMI_PACKED_LOCAL_MARKER}{expected_start}", local
