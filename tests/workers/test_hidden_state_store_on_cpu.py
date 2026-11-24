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
import sys
from types import SimpleNamespace

import pytest
import torch

from verl.experimental.teacher_loop.hidden_state_store import (
    put_hidden_record,
    read_hidden_records,
    step_hidden_handles,
)
from verl.utils.tensordict_utils import list_of_dict_to_tensordict


@pytest.fixture
def storage(monkeypatch):
    entries = {}

    def put(key, partition_id, fields):
        entries[partition_id, key] = fields

    def get(keys, partition_id):
        return list_of_dict_to_tensordict([entries[partition_id, key] for key in keys])

    def clear(keys, partition_id):
        for key in keys:
            del entries[partition_id, key]

    def list_keys(partition_id):
        return {partition_id: {key: {} for partition, key in entries if partition == partition_id}}

    monkeypatch.setitem(
        sys.modules,
        "transfer_queue",
        SimpleNamespace(
            kv_put=put,
            kv_batch_get=get,
            kv_clear=clear,
            kv_list=list_keys,
        ),
    )
    return entries


def produce(request):
    request = dict(request, teacher_key="a", teacher_index=0, teacher_weight_version=3)
    hidden = torch.ones((len(request["positions"]), 4), dtype=torch.bfloat16)
    return put_hidden_record(request, hidden)


def test_masked_record_identity_and_cleanup(storage):
    rows = list_of_dict_to_tensordict(
        [
            {"uid": "a", "input_ids": torch.arange(6), "response_mask": torch.tensor([0, 1, 1])},
            {"uid": "b", "input_ids": torch.arange(4), "response_mask": torch.tensor([0])},
        ]
    )
    manager = SimpleNamespace(compute_hidden_states=lambda rs: list(map(produce, rs)))
    with step_hidden_handles(manager, rows, "route") as handles:
        rows.update(list_of_dict_to_tensordict([{"teacher_hidden_handle": h} for h in handles]))
        records = read_hidden_records(rows)
        assert [m["positions"] for m, _ in records] == [[3, 4], []]
        assert [tuple(h.shape) for _, h in records] == [(2, 4), (0, 4)]
        rows[0]["response_mask"][0] = 1
        with pytest.raises(ValueError, match="Malformed hidden-state"):
            read_hidden_records(rows)
        rows[0]["input_ids"][0] = 9
        with pytest.raises(ValueError, match="Stale or mismatched"):
            read_hidden_records(rows)
    assert not storage


@pytest.mark.parametrize("padded", [False, True], ids=["packed", "padded"])
def test_multiturn_record_transfers_only_target_predictions(storage, padded):
    from verl.experimental.teacher_loop.vllm_hidden_pooler import HiddenStatePooler

    # Two prompt tokens, assistant targets, four tool-call/result tokens, then
    # two more assistant targets. The final environment token is context only.
    ids = [10, 11, 20, 21, 30, 31, 40, 41, 22, 23, 42]
    mask = [1, 1, 0, 0, 0, 0, 1, 1, 0]
    # Token 21 is ordinary assistant output excluded by a custom loss mask.
    # A stale one for an environment token must not re-enable that target.
    loss_mask = [1, 0, 0, 0, 1, 0, 1, 1, 0]
    row = {
        "uid": "multiturn",
        "input_ids": torch.tensor(ids),
        "response_mask": torch.tensor(mask),
        "loss_mask": torch.tensor(loss_mask),
    }
    actor_rows = list_of_dict_to_tensordict([row])
    if padded:
        row = {
            **row,
            "input_ids": torch.tensor([0, *ids, 0, 0]),
            "attention_mask": torch.tensor([0, *([1] * len(ids)), 0, 0]),
            "response_mask": torch.tensor([*mask, 0, 0]),
            "loss_mask": torch.tensor([*loss_mask, 1, 1]),
        }
    rows = list_of_dict_to_tensordict([row])

    def compute(requests):
        (request,) = requests
        assert request["sequence_ids"] == ids
        request = dict(request, teacher_key="a", teacher_index=0, teacher_weight_version=3)
        seq_len = len(ids)
        cursor = SimpleNamespace(
            first_token_indices_gpu=torch.tensor([0]),
            seq_lens_cpu=torch.tensor([seq_len]),
            num_scheduled_tokens_cpu=torch.tensor([seq_len]),
            is_finished=lambda: torch.tensor([True]),
        )
        chunks = []
        metadata = SimpleNamespace(
            get_pooling_cursor=lambda: cursor,
            pooling_states=[SimpleNamespace(hidden_states_cache=chunks, clean=chunks.clear)],
            pooling_params=[SimpleNamespace(extra_kwargs={"opd_record": request})],
        )
        # Distinct prefix sums expose both mistaken row offsets and missing context.
        hidden = torch.tensor(request["sequence_ids"]).cumsum(0).to(torch.bfloat16).unsqueeze(1)
        (result,) = HiddenStatePooler(None, producer=True, hidden_size=1)(hidden, metadata)
        return [json.loads(bytes(result.tolist()))]

    with step_hidden_handles(SimpleNamespace(compute_hidden_states=compute), rows, "route", padded=padded) as handles:
        actor_rows.update(list_of_dict_to_tensordict([{"teacher_hidden_handle": handles[0]}]))
        [(header, payload)] = read_hidden_records(actor_rows)
        assert header["positions"] == [1, 7, 8]
        assert header["signal_nbytes"] == 3 * torch.bfloat16.itemsize
        assert torch.equal(payload[:, 0], torch.tensor([21, 204, 226], dtype=torch.bfloat16))
        actor_rows["loss_mask"][0, 1] = 1
        with pytest.raises(ValueError, match="Malformed hidden-state"):
            read_hidden_records(actor_rows)
    assert not storage


def test_partial_write_cleanup_preserves_other_steps(storage):
    rows = list_of_dict_to_tensordict(
        [{"uid": "a", "input_ids": torch.arange(4), "response_mask": torch.tensor([1, 1])}]
    )

    def fail_after_put(requests):
        handle = produce(requests[0])
        storage[handle["hidden_partition"], "other_step"] = {}
        raise RuntimeError("prefill failed")

    with pytest.raises(RuntimeError, match="prefill failed"):
        with step_hidden_handles(SimpleNamespace(compute_hidden_states=fail_after_put), rows, "route"):
            pytest.fail("Incomplete teacher batch reached student")
    assert [k for _, k in storage] == ["other_step"]


def test_mixed_width_records_keep_actor_row_order(storage):
    widths = [2560, 4096, 2560]
    rows = list_of_dict_to_tensordict(
        [
            {"uid": str(i), "input_ids": torch.arange(4 + i), "response_mask": torch.ones(i + 1)}
            for i in range(len(widths))
        ]
    )

    def produce_group(requests):
        handles = []
        for i, request in enumerate(requests):
            teacher = i % 2
            hidden = torch.full((len(request["positions"]), widths[i]), i, dtype=torch.bfloat16)
            handles.append(put_hidden_record(
                dict(request, teacher_key=str(teacher), teacher_index=teacher, teacher_weight_version=1),
                hidden,
            ))
        return handles

    with step_hidden_handles(SimpleNamespace(compute_hidden_states=produce_group), rows, "route") as handles:
        rows.update(list_of_dict_to_tensordict([{"teacher_hidden_handle": h} for h in handles]))
        for i, (header, hidden) in enumerate(read_hidden_records(rows)):
            assert header["uid"] == str(i)
            assert tuple(hidden.shape) == (i + 1, widths[i])
            assert torch.all(hidden == i)
    assert not storage


def test_signal_byte_counts_and_digests_match_transferred_records(storage, caplog):
    rows = list_of_dict_to_tensordict(
        [{"uid": "a", "input_ids": torch.arange(5), "response_mask": torch.tensor([1, 0, 1])}]
    )

    def measure(signal_kind, width):
        def compute(requests):
            request = dict(
                requests[0], teacher_key="a", teacher_index=0, teacher_weight_version=1, signal_kind=signal_kind
            )
            payload = torch.ones((len(request["positions"]), width), dtype=torch.bfloat16)
            return [put_hidden_record(request, payload)]

        manager = SimpleNamespace(compute_hidden_states=compute)
        with caplog.at_level("INFO", logger="verl.experimental.teacher_loop.hidden_state_store"):
            with step_hidden_handles(manager, rows, "route") as handles:
                rows.update(list_of_dict_to_tensordict([{"teacher_hidden_handle": handles[0]}]))
                [(header, payload)] = read_hidden_records(rows)
                assert header["signal_kind"] == signal_kind
                assert header["signal_nbytes"] == payload.numel() * payload.element_size()
                message = caplog.records[-1].getMessage()
                assert f"signal={signal_kind} dtype=torch.bfloat16 rows=2 signal_bytes={4 * width} " in message
                return message.split("trajectory_digest=")[1]

    hidden_digest = measure("hidden_states", 3)
    assert measure("full_logits", 7) == hidden_digest
    rows[0]["response_mask"][:] = torch.tensor([0, 1, 1])
    assert measure("full_logits", 7) != hidden_digest
    assert not storage
