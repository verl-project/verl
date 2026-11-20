# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


from verl.utils.device import get_torch_device, is_device_available, is_npu_available


def set_random_seed(seed):
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # model_parallel_cuda_manual_seed() seeds Megatron-Core's CUDA RNG tracker. This is
    # required for Megatron model construction: layers such as VocabParallelEmbedding call
    # get_cuda_rng_tracker().fork() and raise "cuda rng state model-parallel-rng is not
    # added" if the tracker was never seeded. On Ascend/NPU torch has no CUDA backend, but
    # torch_npu/MindSpeed patch Megatron's tracker so this call works there as well. Only
    # skip it when no real accelerator (CUDA or NPU) is present, which avoids "Torch not
    # compiled with CUDA enabled" on CPU-only environments.
    if is_device_available and get_torch_device().device_count() > 0:
        from megatron.core import tensor_parallel

        tensor_parallel.model_parallel_cuda_manual_seed(seed)


def enable_megatron_full_determinism(seed: int):
    """Enable full determinism settings specific to the Megatron engine.

    Avoids resetting raw device seeds after model_parallel_cuda_manual_seed has seeded
    Megatron's RNG tracker, and applies determinism controls across PyTorch, cuBLAS,
    cuDNN, FlashAttention, and Transformer Engine.
    """
    import os

    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ.setdefault("FLASH_ATTENTION_DETERMINISTIC", "1")
    os.environ.setdefault("VERL_DISABLE_FLASH_ATTN_CE", "1")
    os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    os.environ.setdefault("NCCL_DETERMINISTIC", "1")
    os.environ.setdefault("NCCL_ALGO", "Ring")
    os.environ.setdefault("NCCL_PROTO", "Simple")
    if is_npu_available:
        os.environ.setdefault("HCCL_DETERMINISTIC", "true")
        os.environ.setdefault("CLOSE_MATMUL_K_SHIFT", "1")

    # warn_only=True because colocated vLLM sampler uses torch.cumsum which has no deterministic kernel
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
