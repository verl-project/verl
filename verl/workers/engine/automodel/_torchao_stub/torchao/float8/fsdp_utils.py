# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

# Stub torchao.float8.fsdp_utils (see torchao/__init__.py).
#
# nemo_automodel's init_empty_weights (components/utils/model_utils.py) probes
# the torchao package with safe_import and, when the probe succeeds, runs
# isinstance(param, torchao.float8.fsdp_utils.WeightWithDynamicFloat8CastTensor)
# to special-case Float8 FSDP parameters. This placeholder makes the attribute
# exist; the isinstance check is then simply False for ordinary nn.Parameter
# objects and the standard path is taken. Never instantiate this class.


class WeightWithDynamicFloat8CastTensor:
    """Placeholder: real torchao float8 FSDP parameter class."""
