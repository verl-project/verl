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

"""Temporary compatibility patch for Monarch's CPU UntypedStorage codec.

Remove when the supported Monarch picklers round-trip bare CPU UntypedStorage
with the supported PyTorch version while preserving buffer ownership."""

from __future__ import annotations

import pickle
from collections.abc import Callable
from typing import Any

_StorageReducer = Callable[[Any], tuple[Callable[..., Any], tuple[Any, ...]]]
_UNTYPED_STORAGE_FALLBACK: _StorageReducer | None = None


def install_monarch_storage_codec() -> None:
    """Install the CPU UntypedStorage reducer before actor RPC handling.

    Monarch's legacy storage codec cannot load bare ``UntypedStorage`` with
    PyTorch 2.9 because the legacy loader expects every storage class to expose
    ``dtype``. Both Monarch picklers cache their dispatch tables, so the
    replacement must be installed at process bootstrap before any actor
    payload can be serialized.
    """
    import torch
    from monarch._src.actor import pickle as monarch_pickle

    global _UNTYPED_STORAGE_FALLBACK
    storage_type = torch.storage.UntypedStorage

    monarch_pickle._ensure_torch_pickle()
    monarch_pickle._Pickler._init_torch_dispatch()
    if _UNTYPED_STORAGE_FALLBACK is None:
        _UNTYPED_STORAGE_FALLBACK = monarch_pickle._Pickler._dispatch_table[storage_type]

    monarch_pickle._Pickler._dispatch_table[storage_type] = _reduce_untyped_storage
    monarch_pickle._TorchPickler.dispatch_table[storage_type] = _reduce_untyped_storage


def _reduce_untyped_storage(storage: Any) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """Serialize exact CPU UntypedStorage values without the legacy codec."""
    import torch

    if type(storage) is torch.storage.UntypedStorage and storage.device.type == "cpu":
        view = torch.tensor([], dtype=torch.uint8).set_(storage).numpy()
        return (
            _untyped_storage_from_bytes,
            (storage.nbytes(), pickle.PickleBuffer(memoryview(view))),
        )

    assert _UNTYPED_STORAGE_FALLBACK is not None
    return _UNTYPED_STORAGE_FALLBACK(storage)


def _untyped_storage_from_bytes(nbytes: int, data: Any) -> Any:
    """Rebuild an owning storage independent of the pickle input buffer."""
    import torch

    if isinstance(data, pickle.PickleBuffer):
        data = data.raw()
    tensor = torch.empty(nbytes, dtype=torch.uint8)
    if nbytes:
        memoryview(tensor.numpy())[:] = memoryview(data).cast("B")
    return tensor.untyped_storage()


__all__ = ["install_monarch_storage_codec"]
