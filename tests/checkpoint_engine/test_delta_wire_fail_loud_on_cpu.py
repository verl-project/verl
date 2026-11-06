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
"""Fail-loud and wiring unit tests for the delta HCCL receiver path.

The checksum-corruption cases pin the spec's fail-loud requirement: a damaged
flush must raise, never silently apply. The rendezvous/rank cases pin the
control contract between the rollout engine and the vLLM worker engines.
Requires vLLM (the decode path builds CheckpointWeightPatch objects); skips
cleanly where vLLM is absent.
"""

import json

import pytest
import torch

from verl.checkpoint_engine.delta_sync.encode import DeltaParam, checksum

pytest.importorskip("vllm")


def _spec_tensor(spec: dict) -> torch.Tensor:
    return torch.frombuffer(bytearray(json.dumps(spec).encode()), dtype=torch.uint8)


def _dense_flush_payload(name: str = "model.layers.0.weight"):
    values = torch.randn(64).to(torch.bfloat16)
    spec = {
        "encoding": "dense",
        "params": [
            vars(
                DeltaParam(
                    name=name,
                    dtype="bfloat16",
                    shape=[8, 8],
                    pos_start=0,
                    pos_end=0,
                    pos_width=4,
                    val_start=0,
                    val_end=64,
                )
            )
        ],
        "checksum": int(checksum(torch.empty(0, dtype=torch.uint8), values)),
    }
    return [("__delta_spec__", _spec_tensor(spec)), ("__values__", values)]


def test_decode_delta_payload_applies_clean_flush():
    from verl.workers.rollout.vllm_rollout.delta_weight_transfer import decode_delta_payload

    encoding, patches = decode_delta_payload(_dense_flush_payload())
    assert encoding == "dense"
    assert len(patches) == 1 and patches[0].values.numel() == 64


def test_decode_delta_payload_fails_loud_on_value_corruption():
    from verl.workers.rollout.vllm_rollout.delta_weight_transfer import decode_delta_payload

    payload = _dense_flush_payload()
    raw = payload[1][1].view(torch.uint8)
    raw[3] ^= 0x40
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        decode_delta_payload(payload)


def test_decode_delta_payload_fails_loud_on_spec_corruption():
    from verl.workers.rollout.vllm_rollout.delta_weight_transfer import decode_delta_payload

    payload = _dense_flush_payload()
    spec = json.loads(bytes(payload[0][1].numpy().tobytes()).decode())
    spec["checksum"] = (spec["checksum"] + 1) & ((1 << 128) - 1)
    payload[0] = ("__delta_spec__", _spec_tensor(spec))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        decode_delta_payload(payload)


def test_decode_delta_rendezvous_roundtrip():
    from verl.workers.rollout.vllm_rollout.vllm_rollout import _decode_delta_rendezvous

    info = {
        "mode": "verl_delta_hccl",
        "zmq_ip": "10.0.0.1",
        "zmq_port": 1234,
        "dist_ip": "10.0.0.1",
        "dist_port": 5678,
        "topic": "bucket_metadata",
        "world_size": 7,
        "engine_rank": 3,
    }
    decoded = _decode_delta_rendezvous([("__delta_rendezvous__", _spec_tensor(info))])
    assert decoded == info
    assert _decode_delta_rendezvous([("__delta_spec__", torch.empty(0, dtype=torch.uint8))]) is None


def test_hccl_group_update_infos_number_replica_major_tp_minor():
    from verl.workers.rollout.vllm_rollout.vllm_rollout import _hccl_group_update_infos

    assert _hccl_group_update_infos(replica_rank=0, tp_size=1) == [{"group_rank": 1}]
    assert _hccl_group_update_infos(replica_rank=0, tp_size=2) == [{"group_rank": 1}, {"group_rank": 2}]
    assert _hccl_group_update_infos(replica_rank=2, tp_size=2) == [{"group_rank": 5}, {"group_rank": 6}]
    ranks = [u["group_rank"] for u in _hccl_group_update_infos(replica_rank=1, tp_size=3)]
    assert ranks == [4, 5, 6]


if __name__ == "__main__":
    pytest.main([__file__])
