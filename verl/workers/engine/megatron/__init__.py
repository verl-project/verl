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


# HACK Avoid cpu worker trigger cuda jit error
import os

from verl.utils.device import is_cuda_available

if not is_cuda_available and "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"

from .transformer_impl import MegatronEngine, MegatronEngineWithLMHead  # noqa: E402

if not is_cuda_available:
    del os.environ["TORCH_CUDA_ARCH_LIST"]

__all__ = ["MegatronEngine", "MegatronEngineWithLMHead"]
