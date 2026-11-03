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

# Stub torchao.float8 (see torchao/__init__.py).
#
# Intentionally does NOT provide Float8LinearConfig / convert_to_float8_training
# so nemo_automodel's HAVE_TORCHAO probe stays False. The only import verl's
# automodel path needs at import time is precompute_float8_dynamic_scale_for_fsdp
# (recipes/llm/train_ft.py), which is invoked only under
# fp8.precompute_float8_dynamic_scale_for_fsdp=True.

# Submodules are NOT auto-exposed as package attributes in Python; import them
# explicitly so attribute chains like `torchao.float8.fsdp_utils.WeightWith...
# ` (nemo_automodel model_utils.register_empty_parameter) resolve the same way
# they do against the real torchao.
from torchao.float8 import fsdp_utils  # noqa: F401


def precompute_float8_dynamic_scale_for_fsdp(*args, **kwargs):
    raise NotImplementedError(
        "torchao is not installed in this environment; "
        "precompute_float8_dynamic_scale_for_fsdp is a verl stub. Install torchao "
        "and enable fp8 to use FSDP float8 all-gather."
    )
