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

"""Real FP8 (torch.float8_e4m3fn) training support for verl's FSDP engine.

Not QAT (see verl.utils.qat for fake-quantization aimed at a future low-bit
export) and not the rollout-engine quantizer (see verl.utils.fp8_utils).

Usage:
    from verl.utils.fp8_training import apply_fp8_training, FP8TrainingConfig

    config = FP8TrainingConfig(enable=True, mode="rowwise")
    model = apply_fp8_training(model, config)  # after model build, before FSDP wrap
"""

from verl.utils.fp8_training.core import FP8TrainingConfig, apply_fp8_training
from verl.utils.fp8_training.linear import FP8Linear, FP8Mode

__all__ = [
    "FP8TrainingConfig",
    "apply_fp8_training",
    "FP8Linear",
    "FP8Mode",
]
