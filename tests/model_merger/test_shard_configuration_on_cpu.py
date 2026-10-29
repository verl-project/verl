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

import numpy as np
import pytest

from verl.model_merger.fsdp_model_merger import FSDPModelMerger


def _shard_configuration(mesh_dim_names):
    # _calculate_shard_configuration is a pure function of (mesh, mesh_dim_names);
    # bypass __init__ so no checkpoint directory is needed.
    merger = FSDPModelMerger.__new__(FSDPModelMerger)
    return merger._calculate_shard_configuration(np.arange(8, dtype=np.int64), mesh_dim_names)


@pytest.mark.parametrize("mesh_dim_names", [("fsdp",), ("dp_shard",)])
def test_one_dimensional_full_shard_mesh_is_accepted(mesh_dim_names):
    """VeOmni's FSDP2 engine names its single shard dim ``dp_shard``; it is the
    same 1-D full-shard layout the merger already accepts as ``fsdp``."""
    total_shards, mesh_shape = _shard_configuration(mesh_dim_names)
    assert total_shards == 8
    assert mesh_shape == (8,)


def test_unknown_mesh_dim_names_are_still_rejected():
    with pytest.raises(AssertionError, match="Unsupported mesh_dim_names"):
        _shard_configuration(("dp_shard", "tp"))
