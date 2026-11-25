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

"""Step-scoped teacher records. Only their handles enter the training batch."""

import hashlib
import json
import logging
import uuid
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)
_PARTITION = "opd_hidden_states"
_IDENTITY = ("hidden_partition", "hidden_key", "uid", "batch_index", "sequence_hash")
# Grace period for peer actor calls to stop reading a failed step's records.
_READER_DRAIN_TIMEOUT_S = 30.0


def hidden_state_enabled(config):
    return bool(config and config.get("enabled", False)) and (
        config.get("teacher_execution", {}).get("mode") in ("weight_swapped_hidden_state", "dedicated_hidden_state")
    )


def distillation_target_mask(response_mask: torch.Tensor, loss_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Select response targets still eligible for loss after rollout filtering, for one trajectory."""
    if response_mask.ndim != 1 or (loss_mask is not None and loss_mask.shape != response_mask.shape):
        raise ValueError("Expected one response_mask and loss_mask entry per response token")
    eligible = response_mask.bool()
    return eligible if loss_mask is None else eligible & loss_mask.bool()


def _sequence_hash(ids):
    return hashlib.sha256(ids.detach().to("cpu", torch.int64).numpy().astype("<i8", copy=False).tobytes()).hexdigest()


def put_hidden_record(request: dict, hidden: torch.Tensor) -> dict:
    """Called by the teacher after selecting the causal prediction positions."""
    import transfer_queue as tq

    positions = torch.tensor(request["positions"], dtype=torch.int64)
    if hidden.ndim != 2 or hidden.size(0) != len(positions):
        raise ValueError(f"Unexpected hidden-state shape {tuple(hidden.shape)} for {len(positions)} tokens")
    handle = {key: request[key] for key in (*_IDENTITY, "teacher_key", "teacher_index", "teacher_weight_version")}
    handle.update(
        seq_len=request["seq_len"],
        response_len=request["response_len"],
        hidden_shape=list(hidden.shape),
        hidden_dtype=str(hidden.dtype),
        signal_kind=request.get("signal_kind", "hidden_states"),
        signal_nbytes=hidden.numel() * hidden.element_size(),
    )
    # kv_put attaches lazily to the running controller in the producer process.
    tq.kv_put(
        key=handle["hidden_key"],
        partition_id=handle["hidden_partition"],
        fields={
            handle["signal_kind"]: hidden.detach().cpu().contiguous(),
            "positions": positions,
            "record_header": torch.tensor(list(json.dumps(handle, sort_keys=True).encode()), dtype=torch.uint8),
        },
    )
    return handle


def read_hidden_records(data):
    """Validate each record against this actor microbatch before reconstructing logits."""
    import transfer_queue as tq

    handles = [json.loads(s) for s in data["teacher_hidden_handle"]]
    # TransferQueue collates jagged tensors along their token dimension. Width and
    # dtype must agree within each read, even when different teachers share a step.
    groups = {}
    for i, handle in enumerate(handles):
        groups.setdefault((handle["hidden_shape"][-1], handle["hidden_dtype"]), []).append(i)
    rows = {}
    for indices in groups.values():
        batch = tq.kv_batch_get(keys=[handles[i]["hidden_key"] for i in indices], partition_id=_PARTITION)
        for offset, index in enumerate(indices):
            rows[index] = batch[offset]

    loss_mask = data.get("loss_mask")
    result = []
    for i, handle in enumerate(handles):
        record = rows[i]
        header = json.loads(bytes(record["record_header"].tolist()))
        ids = data["input_ids"][i]
        if (
            header != handle
            or handle["hidden_partition"] != _PARTITION
            or str(data["uid"][i]) != handle["uid"]
            or ids.numel() != handle["seq_len"]
            or _sequence_hash(ids) != handle["sequence_hash"]
        ):
            raise ValueError(f"Stale or mismatched hidden-state record for {handle['hidden_key']}")
        hidden = record[header.get("signal_kind", "hidden_states")]
        positions = record["positions"]
        response_len = handle["response_len"]
        mask = distillation_target_mask(data["response_mask"][i], loss_mask[i] if loss_mask is not None else None)
        mask = mask[:response_len]
        expected_positions = mask.nonzero().flatten().cpu() + ids.numel() - response_len - 1
        if (
            not torch.equal(positions, expected_positions)
            or list(hidden.shape) != header["hidden_shape"]
            or str(hidden.dtype) != header["hidden_dtype"]
            or positions.numel() != hidden.size(0)
        ):
            raise ValueError(f"Malformed hidden-state record for {handle['hidden_key']}")
        result.append(({**header, "positions": positions.tolist()}, hidden))
    return result


@contextmanager
def step_hidden_handles(manager, rows, teacher_key, *, padded=False, padding_indices=(), reader_refs=None):
    """Schedule teachers in groups, verify the scatter, then release after all actor reads."""
    import transfer_queue as tq

    step_id = uuid.uuid4().hex
    requests = []
    for i, row in enumerate(rows):
        ids = row["input_ids"]
        target_mask = distillation_target_mask(row["response_mask"], row.get("loss_mask"))
        if padded:
            attention = row["attention_mask"].bool()
            if target_mask.numel():
                target_mask = target_mask[attention[-target_mask.numel() :]]
            ids = ids[attention]
        prompt_len = ids.numel() - target_mask.numel()
        if prompt_len < 1:
            raise ValueError("Hidden-state distillation requires a nonempty prompt.")
        # The hidden row immediately before a target predicts that token.
        positions = target_mask.nonzero().flatten() + prompt_len - 1
        prefill_ids = ids
        if i in padding_indices:
            if positions.numel():
                raise ValueError("Synthetic padding has active distillation targets.")
            # The pooler only needs one EOS to produce a typed empty record.
            # Keep the full actor sequence in the record identity below.
            prefill_ids = ids[:1]
        requests.append(
            {
                **(row["teacher_prefill"] if "teacher_prefill" in row.keys() else {}),
                "uid": str(row["uid"]),
                "routing_key": row[teacher_key] if teacher_key in row.keys() else None,
                "batch_index": i,
                "sequence_ids": prefill_ids.tolist(),
                "seq_len": ids.numel(),
                "response_len": target_mask.numel(),
                "positions": positions.tolist(),
                "sequence_hash": _sequence_hash(ids),
                "hidden_key": f"{step_id}_{i}",
                "hidden_partition": _PARTITION,
            }
        )
    keys = [r["hidden_key"] for r in requests]
    all_written = False
    failed = True
    try:
        handles = manager.compute_hidden_states(requests)
        for handle, request in zip(handles, requests, strict=True):
            if any(handle[k] != request[k] for k in _IDENTITY):
                raise ValueError(f"Teacher returned a mismatched handle at batch index {request['batch_index']}")
        all_written = True
        trajectories = sorted(
            (h["teacher_key"], h["sequence_hash"], h["response_len"], r["positions"])
            for h, r in zip(handles, requests, strict=True)
        )
        digest = hashlib.sha256(json.dumps(trajectories, separators=(",", ":")).encode()).hexdigest()
        selected_rows = sum(h["hidden_shape"][0] for h in handles)
        response_rows = sum(r["response_len"] for r in requests)
        logger.info(
            "Hidden-state OPD stored %d records for %d teacher(s) signal=%s dtype=%s rows=%d "
            "signal_bytes=%d response_rows=%d skipped_rows=%d trajectory_digest=%s",
            len(handles),
            len({h["teacher_index"] for h in handles}),
            ",".join(sorted({h["signal_kind"] for h in handles})),
            ",".join(sorted({h["hidden_dtype"] for h in handles})),
            selected_rows,
            sum(h["signal_nbytes"] for h in handles),
            response_rows,
            response_rows - selected_rows,
            digest,
        )
        yield [json.dumps(h, sort_keys=True) for h in handles]
        failed = False
    finally:
        try:
            if failed and reader_refs:
                import ray

                # A peer of the failed rank can block in a collective and never return.
                _, running = ray.wait(
                    reader_refs, num_returns=len(reader_refs), timeout=_READER_DRAIN_TIMEOUT_S, fetch_local=False
                )
                if running:
                    logger.warning(
                        "Hidden-state OPD clears step %s with %d actor call(s) still running", step_id, len(running)
                    )
            if not all_written:
                existing = (tq.kv_list(partition_id=_PARTITION) or {}).get(_PARTITION, {})
                keys = [key for key in keys if key in existing]
            if keys:
                tq.kv_clear(keys=keys, partition_id=_PARTITION)
                logger.info("Hidden-state OPD cleared %d records", len(keys))
        except Exception:
            if not failed:
                raise
            logger.exception("Hidden-state OPD cleanup failed for step %s", step_id)
