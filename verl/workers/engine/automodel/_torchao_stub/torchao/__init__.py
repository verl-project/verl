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

# Minimal STUB of torchao, used only when the environment does not provide
# torchao (see verl/workers/engine/automodel/utils.py::ensure_vendored_nemo_automodel).
#
# The vendored nemo_automodel has two top-level imports that make torchao a
# hard requirement even when QAT / Float8 training are unused:
#   - nemo_automodel/components/quantization/qat.py:
#       from torchao.quantization.qat import (Int4WeightOnlyQATQuantizer,
#                                             Int8DynActInt4WeightQATQuantizer)
#       from torchao.quantization.qat.linear import (disable_4w_fake_quant, ...)
#   - nemo_automodel/recipes/llm/train_ft.py:
#       from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
#
# This package satisfies those imports; every symbol raises on use so a
# misconfigured run fails loudly instead of silently faking quantization.
# It is intentionally NOT a torchao.float8.Float8LinearConfig provider, so
# fp8.py's `HAVE_TORCHAO` probe stays False and apply_fp8_to_model reports
# "torchao is required" rather than half-working.

# Import the submodules eagerly so any attribute chain nemo_automodel walks
# (e.g. torchao.float8.fsdp_utils.WeightWithDynamicFloat8CastTensor in
# model_utils.register_empty_parameter) resolves without depending on some
# other module having imported the submodule first.
from torchao import float8, quantization  # noqa: E402,F401

# Valid semver so libraries reading the module attribute (rather than
# importlib.metadata) never see a missing/invalid version.  Keep it below
# every real torchao release so version-gated features stay off.
__version__ = "0.0.0"
