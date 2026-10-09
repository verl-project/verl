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

from __future__ import annotations

import pytest


def test_monarch_codec_round_trips_cpu_untyped_storage():
    torch = pytest.importorskip("torch")
    pytest.importorskip("monarch")

    from monarch._rust_bindings.monarch_hyperactor import pickle as monarch_pickle

    from verl.single_controller.monarch.patches.codec import install_monarch_storage_codec

    install_monarch_storage_codec()
    source = torch.arange(8192, dtype=torch.uint8).untyped_storage()
    restored = monarch_pickle.pickle(source).unpickle()

    assert isinstance(restored, torch.storage.UntypedStorage)
    assert bytes(restored) == bytes(source)
