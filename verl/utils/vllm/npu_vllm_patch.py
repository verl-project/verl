# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team
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


import logging
from functools import wraps
import torch
from verl.utils.device import is_torch_npu_available

logger = logging.getLogger(__name__)


def refresh_ascend_moe_comm_state_after_l2_wake() -> None:
    """Restore All2AllV ``expert_ids_per_ep_rank`` after vLLM L2 (discard) wake + IPC reload.

    L2 discards this persistent NPU tensor; IPC reload and buffer restore do not recreate it,
    so after wake it can hold garbage and route tokens to wrong local experts under EP>1.
    Uses process-global ``_MoECommMethods`` (populated at MoE layer init), not the ``nn.Module``.

    TODO: remove when vllm-ascend restores expert_ids in ``NPUWorker.wake_up`` after L2.
    """
    try:
        from vllm_ascend.ops.fused_moe.moe_comm_method import _MoECommMethods

        for comm_method in list(_MoECommMethods.values()):
            dispatcher = getattr(comm_method, "token_dispatcher", None)
            if dispatcher is None:
                continue
            ids = getattr(dispatcher, "expert_ids_per_ep_rank", None)
            num_local = getattr(dispatcher, "num_local_experts", 0)
            num_experts = getattr(dispatcher, "num_experts", 0)
            if ids is None or num_local <= 1 or num_experts <= 0:
                continue
            correct = torch.arange(num_experts, dtype=ids.dtype, device=ids.device) % num_local
            if ids.shape == correct.shape:
                ids.copy_(correct)
            else:
                dispatcher.expert_ids_per_ep_rank = correct
    except Exception as exc:
        logger.warning("Failed to restore expert_ids_per_ep_rank after L2 wake: %s", exc)


def vllm_v013_weight_loader_method_wrapper(fn):
    @wraps(fn)
    def wrapper(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
        if (shard_id in ("w1", "w3") and param.shape[1] == self.hidden_size) or (
            shard_id == "w2" and param.shape[2] == self.hidden_size
        ):
            param.data = param.data.transpose(1, 2)
        return fn(self, param, loaded_weight, weight_name, shard_id, expert_id, return_success)

    return wrapper


def patch_vllm013_rotary_emb():
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    def vllm013_npu_rotary_embedding_init_impl(
        self,
        enforce_enable: bool = False,
        is_neox_style: bool = True,
        enable_fp32_compute: bool = False,
    ) -> None:
        super(ApplyRotaryEmb, self).__init__()
        self.is_neox_style = is_neox_style
        self.enable_fp32_compute = enable_fp32_compute
        self.apply_rotary_emb_flash_attn = None

    ApplyRotaryEmb.__init__ = vllm013_npu_rotary_embedding_init_impl


def apply_npu_vllm_patches() -> None:
    """Apply NPU-specific vLLM patches for weight loading and rotary embedding.

    Must be called before the vLLM engine is created.
    """
    if not is_torch_npu_available(check_device=False):
        return

    import vllm
    from packaging import version

    _VLLM_VERSION = version.parse(vllm.__version__)
    if _VLLM_VERSION >= version.parse("0.13.0") and _VLLM_VERSION <= version.parse("0.14.0"):
        # Disable flash_attn in RotaryEmbedding (NPU) when VLLM >= 0.13
        from vllm.model_executor.layers.fused_moe import FusedMoE

        patch_vllm013_rotary_emb()
        FusedMoE.weight_loader = vllm_v013_weight_loader_method_wrapper(FusedMoE.weight_loader)
    elif _VLLM_VERSION >= version.parse("0.18.0"):
        # Disable flash_attn in RotaryEmbedding (NPU) when VLLM >= 0.18
        from vllm.model_executor.layers.fused_moe import FusedMoE

        patch_vllm013_rotary_emb()
        FusedMoE.weight_loader = vllm_v013_weight_loader_method_wrapper(FusedMoE.weight_loader)
