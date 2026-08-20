# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Tests for the V0 data-plane dispatch and worker adapter."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto as PublicDataProto
from verl import LegacyDataProto
from verl.experimental.neoproto import (
    DataProto,
    DefaultStorageEngine,
    InMemoryStorageEngine,
    set_default_storage_engine,
)
from verl.experimental.neoproto.worker_bridge import finalize_engine_output, prepare_engine_input
from verl.single_controller.base.decorator import _split_args_kwargs_data_proto
from verl.workers.utils.batch_adapter import EngineBatchSpec, run_engine_batch, set_batch_control_fields

_BASE_KEYS = ("input_ids", "attention_mask", "position_ids", "response_mask", "prompts", "responses")
_INFER_SPEC = EngineBatchSpec(
    required_keys=_BASE_KEYS,
    optional_keys=("compute_loss", "temperature"),
    restore_padding_keys=("values", "log_probs"),
)


class CountingStorageEngine(InMemoryStorageEngine):
    backend = "counting"

    def __init__(self):
        super().__init__()
        self.get_count = 0
        self.get_keys: list[str] = []

    def get(self, ref):
        self.get_count += 1
        return super().get(ref)


@pytest.fixture
def storage():
    engine = InMemoryStorageEngine()
    set_default_storage_engine(engine)
    yield engine
    set_default_storage_engine(None)


@pytest.fixture
def counting_storage():
    engine = CountingStorageEngine()
    set_default_storage_engine(engine)
    yield engine
    set_default_storage_engine(None)


@pytest.fixture
def identity_padding_bridge(monkeypatch):
    """Isolate data-plane tests from the GPU-only FlashAttention unpad kernel."""
    monkeypatch.setattr(
        "verl.experimental.neoproto.worker_bridge.left_right_2_no_padding",
        lambda data: data,
    )


def _bridge_batch(storage, *, extra_tensors=None):
    tensors = {
        "input_ids": torch.ones(4, 4, dtype=torch.long),
        "attention_mask": torch.ones(4, 4, dtype=torch.long),
        "position_ids": torch.arange(4).unsqueeze(0).expand(4, -1),
        "prompts": torch.ones(4, 2, dtype=torch.long),
        "responses": torch.ones(4, 2, dtype=torch.long),
        "response_mask": torch.ones(4, 2, dtype=torch.long),
    }
    if extra_tensors:
        tensors.update(extra_tensors)
    data = DataProto.from_dict(tensors=tensors, storage=storage)
    data.meta_info["compute_loss"] = False
    data.meta_info["temperature"] = 1.0
    return data


def test_neoproto_implements_common_data_proto_contract(storage):
    data = DataProto.from_dict(tensors={"input_ids": torch.arange(8).view(4, 2)}, storage=storage)
    assert PublicDataProto is DataProto
    assert isinstance(data, LegacyDataProto)
    assert callable(data.prepare_dispatch)
    assert data.new_like(batch=TensorDict({}, batch_size=[0])).__class__ is DataProto


def test_neoproto_request_controls_do_not_mutate_source(storage):
    data = _bridge_batch(storage)
    request = data.prepare_worker_request(no_lora_adapter=True)

    assert "no_lora_adapter" not in data.meta_info
    assert request.meta_info["no_lora_adapter"] is True


def test_neoproto_collects_worker_output_by_ref(storage):
    output = DataProto.from_dict(
        tensors={"log_probs": torch.zeros(2, 1), "entropy": torch.ones(2, 1)},
        meta_info={"metrics": {"mfu": 0.5}},
        storage=storage,
    )

    collected = DataProto.collect_worker_output(output, {"log_probs": "old_log_probs"})

    assert set(collected.batch.keys()) == {"old_log_probs"}
    assert collected.ref_table["old_log_probs"] is output.ref_table["log_probs"]


def test_neoproto_rejects_non_neo_worker_output():
    output = TensorDict({"log_probs": torch.zeros(2, 1)}, batch_size=[2])

    with pytest.raises(RuntimeError, match="non-Neo worker payload"):
        DataProto.collect_worker_output(output, {"log_probs": "old_log_probs"})


def test_split_args_attaches_obj_ref_local_ref(storage):
    # Force object_store-shaped refs via DefaultStorageEngine when Ray is up;
    # with InMemoryStorageEngine, attach is still a no-op-safe path for local refs.
    data = DataProto.from_dict(
        tensors={"input_ids": torch.arange(16).view(8, 2)},
        non_tensors={"uid": np.array([f"u{i}" for i in range(8)], dtype=object)},
        storage=storage,
    )
    args, kwargs = _split_args_kwargs_data_proto(4, data)
    chunks = args[0]
    assert len(chunks) == 4
    # Local-backend columns may leave OBJ_REF unset; attaching must not crash.
    for c in chunks:
        assert isinstance(c, DataProto)
        assert len(c) == 2


def test_attach_preserialized_ref_tables_with_ray_object_store():
    try:
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=2)
    except Exception:
        pytest.skip("Ray not available")

    engine = DefaultStorageEngine()
    set_default_storage_engine(engine)
    try:
        raw_prompts = np.asarray([{"prompt": f"sample-{i}"} for i in range(8)], dtype=object)
        reward_models = np.asarray([{"ground_truth": str(i)} for i in range(8)], dtype=object)
        data = DataProto.from_dict(
            tensors={"input_ids": torch.arange(16).view(8, 2)},
            non_tensors={"raw_prompt": raw_prompts, "reward_model": reward_models},
        )
        assert data.ref_table["raw_prompt"].backend == "local"
        assert data.ref_table["reward_model"].backend == "local"
        np.testing.assert_array_equal(data.non_tensor_batch["raw_prompt"], raw_prompts)
        # Runtime assignments use the same split: tensors and media payloads
        # stay in Ray's object store, while prompt/control columns stay local.
        data.batch["token_level_scores"] = torch.zeros(8, 2)
        data.non_tensor_batch["raw_prompt"] = raw_prompts.copy()
        data.non_tensor_batch["reward_model"] = reward_models.copy()
        assert data.ref_table["token_level_scores"].backend == "object_store"
        assert data.ref_table["raw_prompt"].backend == "local"
        assert data.ref_table["reward_model"].backend == "local"
        small_batch = DataProto(
            batch=TensorDict(
                {"responses": torch.zeros(1, 16, dtype=torch.long)},
                batch_size=[1],
            )
        )
        large_batch = DataProto(
            batch=TensorDict(
                {"responses": torch.zeros(1, 8193, dtype=torch.long)},
                batch_size=[1],
            )
        )
        assert small_batch.ref_table["responses"].backend == "local"
        assert large_batch.ref_table["responses"].backend == "object_store"
        chunks = _split_args_kwargs_data_proto(4, data)[0][0]
        for c in chunks:
            assert hasattr(c, "OBJ_REF")
            assert hasattr(c, "LOCAL_REF")
            # Round-trip pickle path uses OBJ_REF/LOCAL_REF
            state = c.__getstate__()
            assert isinstance(state[1], tuple)
    finally:
        set_default_storage_engine(None)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_default_storage_wire_round_trip_copies_read_only_numpy(dtype):
    engine = DefaultStorageEngine()
    source = torch.arange(8, dtype=dtype).view(2, 4)
    wire = engine.to_wire(source)
    wire.data.setflags(write=False)

    restored = engine.from_wire(wire)

    torch.testing.assert_close(restored, source)
    restored.add_(1)
    torch.testing.assert_close(source, torch.arange(8, dtype=dtype).view(2, 4))


def test_worker_bridge_materialize_and_wrap(storage, identity_padding_bridge):
    data = _bridge_batch(storage)
    # required_keys=None → legacy full to_tensordict path
    engine_td, ctx = prepare_engine_input(data, restore_padding_keys=())
    assert isinstance(engine_td, TensorDict)
    assert ctx is not None
    assert engine_td["compute_loss"] is False or getattr(engine_td["compute_loss"], "data", None) is False

    fake_out = TensorDict(
        {
            "log_probs": torch.zeros(4, 2, dtype=torch.bfloat16),
            "values": torch.ones(4, 2, dtype=torch.bfloat16),
        },
        batch_size=[4],
    )
    wrapped = finalize_engine_output(fake_out, ctx)
    assert isinstance(wrapped, DataProto)
    assert "log_probs" in wrapped.batch.keys()
    assert wrapped.batch["log_probs"].dtype == torch.float32
    assert wrapped.batch["values"].dtype == torch.float32


def test_worker_bridge_subset_skips_unused_columns(counting_storage, identity_padding_bridge):
    dummy = torch.arange(4 * 128, dtype=torch.float32).view(4, 128)
    data = _bridge_batch(counting_storage, extra_tensors={"dummy_unused": dummy})
    counting_storage.get_count = 0

    engine_td, ctx = prepare_engine_input(
        data,
        restore_padding_keys=(),
        required_keys=_INFER_SPEC.required_keys,
        optional_keys=_INFER_SPEC.optional_keys,
    )
    assert ctx is not None
    assert "input_ids" in engine_td.keys()
    # Unused column must not appear on the engine TensorDict.
    assert "dummy_unused" not in set(engine_td.keys())

    # Touching dummy after subset prepare would fetch it; ensure bridge did not.
    gets_after_bridge = counting_storage.get_count
    _ = data.batch["dummy_unused"]
    assert counting_storage.get_count > gets_after_bridge


def test_worker_bridge_subset_fetches_fewer_than_full(counting_storage, identity_padding_bridge):
    dummy = torch.arange(4 * 64, dtype=torch.float32).view(4, 64)
    data_full = _bridge_batch(counting_storage, extra_tensors={"dummy_unused": dummy})
    counting_storage.get_count = 0
    prepare_engine_input(data_full, restore_padding_keys=())
    full_gets = counting_storage.get_count

    data_sub = _bridge_batch(counting_storage, extra_tensors={"dummy_unused": dummy})
    counting_storage.get_count = 0
    prepare_engine_input(
        data_sub,
        restore_padding_keys=(),
        required_keys=_INFER_SPEC.required_keys,
        optional_keys=_INFER_SPEC.optional_keys,
    )
    subset_gets = counting_storage.get_count
    assert subset_gets < full_gets


def test_worker_bridge_missing_required_key_raises(counting_storage):
    data = DataProto.from_dict(
        tensors={"input_ids": torch.ones(2, 2, dtype=torch.long)},
        storage=counting_storage,
    )
    with pytest.raises(KeyError, match="missing required keys"):
        prepare_engine_input(
            data,
            restore_padding_keys=(),
            required_keys=_INFER_SPEC.required_keys,
            optional_keys=_INFER_SPEC.optional_keys,
        )


def test_generic_engine_adapter_passes_tensordict_through():
    data = TensorDict({"x": torch.ones(2, 1)}, batch_size=[2])

    output = run_engine_batch(data, lambda td: td + 1, EngineBatchSpec())

    torch.testing.assert_close(output["x"], torch.full((2, 1), 2.0))


def test_generic_control_fields_support_both_batch_types(storage):
    classic = TensorDict({"x": torch.ones(2, 1)}, batch_size=[2])
    neo = DataProto.from_dict(tensors={"x": torch.ones(2, 1)}, storage=storage)

    set_batch_control_fields(classic, compute_loss=False)
    set_batch_control_fields(neo, compute_loss=False)

    classic_value = classic["compute_loss"]
    assert classic_value is False or getattr(classic_value, "data", None) is False
    assert neo.meta_info["compute_loss"] is False


def test_generic_engine_adapter_materializes_neoproto(storage, identity_padding_bridge):
    data = _bridge_batch(storage)
    spec = EngineBatchSpec(required_keys=_BASE_KEYS, optional_keys=("compute_loss", "temperature"))

    output = run_engine_batch(
        data,
        lambda td: TensorDict({"log_probs": torch.zeros(4, 2)}, batch_size=[4]),
        spec,
    )

    assert isinstance(output, DataProto)
    assert "log_probs" in output.batch
