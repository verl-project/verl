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

import torch

from verl.utils.fsdp_utils import (
    load_fsdp_model_from_disk,
    load_fsdp_optimizer_from_disk,
    offload_fsdp_model_to_disk,
    offload_fsdp_optimizer_to_disk,
)
from verl.utils.offload import DiskOffloadStore, read_storage_refs, release_storage_refs, write_storage_refs


def _store(tmp_path):
    return DiskOffloadStore(
        str(tmp_path),
        rank=0,
        chunk_size_mb=1,
        cleanup_on_exit=False,
        job_id="storage-test",
    )


def test_aliasing_storage_round_trip_preserves_views(tmp_path):
    store = _store(tmp_path)
    base = torch.arange(32, dtype=torch.float32)
    left = base[:16]
    right = base[16:]
    expected = base.clone()
    identities = (id(base), id(left), id(right))

    refs = write_storage_refs(store, "param", [("left", left), ("right", right)])

    assert len(refs) == 1
    assert base.untyped_storage().nbytes() == 0
    assert (id(base), id(left), id(right)) == identities

    read_storage_refs(store, "param", refs)

    torch.testing.assert_close(base, expected, rtol=0, atol=0)
    assert left.data_ptr() == base.data_ptr()
    assert right.data_ptr() == base.data_ptr() + left.numel() * left.element_size()
    assert (id(base), id(left), id(right)) == identities


def test_storage_refs_can_reuse_a_committed_snapshot(tmp_path):
    store = _store(tmp_path)
    tensor = torch.arange(300_000, dtype=torch.float32)
    expected = tensor.clone()
    refs = write_storage_refs(store, "param", [("weight", tensor)])
    store.pop_io_stats()

    read_storage_refs(store, "param", refs)
    release_storage_refs(store, "param", refs)

    stats = store.pop_io_stats()
    assert set(stats) == {("onload", "param")}
    assert tensor.untyped_storage().nbytes() == 0

    read_storage_refs(store, "param", refs)
    torch.testing.assert_close(tensor, expected, rtol=0, atol=0)


def test_noncontiguous_view_round_trip_persists_full_storage(tmp_path):
    store = _store(tmp_path)
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    transposed = base.transpose(0, 1)
    expected = transposed.clone()

    refs = write_storage_refs(store, "param", [("transposed", transposed)])

    assert len(refs) == 1
    assert base.untyped_storage().nbytes() == 0
    read_storage_refs(store, "param", refs)
    torch.testing.assert_close(transposed, expected, rtol=0, atol=0)
    assert not transposed.is_contiguous()


def test_fsdp_style_model_param_and_grad_round_trip(tmp_path):
    store = _store(tmp_path)
    model = torch.nn.Linear(4, 3)
    model.register_buffer("scale", torch.arange(3, dtype=torch.float32))
    model(torch.ones(2, 4)).sum().backward()
    expected_params = [param.detach().clone() for param in model.parameters()]
    expected_grads = [param.grad.clone() for param in model.parameters()]
    expected_buffer = model.scale.clone()
    param_ids = [id(param) for param in model.parameters()]
    grad_ids = [id(param.grad) for param in model.parameters()]

    refs = offload_fsdp_model_to_disk(
        model,
        store,
        offload_param=True,
        offload_grad=True,
        preserve_grad=True,
    )
    assert all(param.untyped_storage().nbytes() == 0 for param in model.parameters())
    assert all(param.grad.untyped_storage().nbytes() == 0 for param in model.parameters())
    assert model.scale.untyped_storage().nbytes() == 0

    load_fsdp_model_from_disk(store, refs, load_param=True, load_grad=True)

    assert [id(param) for param in model.parameters()] == param_ids
    assert [id(param.grad) for param in model.parameters()] == grad_ids
    for param, expected in zip(model.parameters(), expected_params, strict=True):
        torch.testing.assert_close(param, expected, rtol=0, atol=0)
    for param, expected in zip(model.parameters(), expected_grads, strict=True):
        torch.testing.assert_close(param.grad, expected, rtol=0, atol=0)
    torch.testing.assert_close(model.scale, expected_buffer, rtol=0, atol=0)


def test_optimizer_state_round_trip(tmp_path):
    store = _store(tmp_path)
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(2, 4)).sum().backward()
    optimizer.step()
    expected = {
        (param_index, key): value.clone()
        for param_index, param in enumerate(model.parameters())
        for key, value in optimizer.state[param].items()
        if isinstance(value, torch.Tensor)
    }

    refs = offload_fsdp_optimizer_to_disk(optimizer, store)
    assert refs
    assert all(ref.storage.nbytes() == 0 for ref in refs)

    load_fsdp_optimizer_from_disk(store, refs)

    for param_index, param in enumerate(model.parameters()):
        for key, value in optimizer.state[param].items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, expected[(param_index, key)], rtol=0, atol=0)
