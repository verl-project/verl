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

"""Real NVFP4 training/rollout integration helpers."""

from .bf16_transport import attest_real_nvfp4_bf16_transport
from .config import (
    REAL_NVFP4_TE_COMMIT,
    REAL_NVFP4_TE_VERSION,
    real_nvfp4_expected_counts,
    validate_real_nvfp4_model_contract,
    validate_real_nvfp4_te_recipe,
)
from .vllm_runtime import (
    NVFP4_PER_TOKEN_METHOD,
    REAL_NVFP4_MOE_BACKEND,
    attest_vllm_native_nvfp4_runtime,
    require_vllm_native_nvfp4_per_token,
    require_vllm_native_reload_contract,
    vllm_native_nvfp4_fingerprint,
)

__all__ = [
    "NVFP4_PER_TOKEN_METHOD",
    "REAL_NVFP4_MOE_BACKEND",
    "REAL_NVFP4_TE_COMMIT",
    "REAL_NVFP4_TE_VERSION",
    "attest_real_nvfp4_bf16_transport",
    "attest_vllm_native_nvfp4_runtime",
    "real_nvfp4_expected_counts",
    "require_vllm_native_nvfp4_per_token",
    "require_vllm_native_reload_contract",
    "validate_real_nvfp4_model_contract",
    "validate_real_nvfp4_te_recipe",
    "vllm_native_nvfp4_fingerprint",
]
