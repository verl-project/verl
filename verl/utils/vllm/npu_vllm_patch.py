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


import os
from functools import wraps

from verl.utils.device import is_torch_npu_available


def vllm_ascend_v011_select_moe_comm_method_wrapper(fn):
    @wraps(fn)
    def wrapper(self, num_tokens, with_prefill):
        moe_comm_method = fn(self, num_tokens, with_prefill)
        from vllm_ascend.ascend_forward_context import MoECommType
        from vllm_ascend.utils import AscendSocVersion, enable_sp, get_ascend_soc_version

        soc_version = get_ascend_soc_version()

        # AscendSocVersion.A2 is not support MC2 in Single-card multi-process scenario now.
        if soc_version in {AscendSocVersion.A2} and moe_comm_method == MoECommType.MC2:
            quant_type = getattr(self.vllm_config.model_config.hf_config, "moe_quantize", None)
            # Currently, w4a8_dynamic does not support allgatherep
            if quant_type == "w4a8_dynamic":
                moe_comm_method = MoECommType.ALLTOALL
            else:
                moe_comm_method = MoECommType.ALLGATHER

        if with_prefill:
            if enable_sp():
                moe_comm_method = MoECommType.ALLGATHER
            else:
                moe_comm_method = MoECommType.NAIVE_MULTICAST

        return moe_comm_method

    return wrapper


def vllm_ascend_v011_matmul_and_reduce_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        from vllm_ascend.utils import AscendSocVersion, get_ascend_soc_version

        soc_version = get_ascend_soc_version()
        # AscendSocVersion.A2 is not support MC2 in Single-card multi-process scenario now.
        if soc_version in {AscendSocVersion.A2}:
            from vllm.forward_context import get_forward_context

            try:
                forward_context = get_forward_context()
                forward_context.mmrs_fusion = False
            except AssertionError:
                # forward_context.mmrs_fusion will be false in matmul_and_reduce func.
                pass
        return fn(self, *args, **kwargs)

    return wrapper


def check_vllm_ascend_before_server_launch():
    import torch_npu
    import vllm

    def _is_ascend_soc_version_A2_v011_local():
        from vllm_ascend.utils import AscendSocVersion

        soc_version = torch_npu.npu.get_soc_version()
        if 220 <= soc_version <= 225:
            _ascend_soc_version = AscendSocVersion.A2
        elif 250 <= soc_version <= 255:
            _ascend_soc_version = AscendSocVersion.A3
        else:
            _ascend_soc_version = AscendSocVersion.UNDEFINED

        return _ascend_soc_version == AscendSocVersion.A2

    def _is_ascend_soc_version_A2_v013_local():
        from vllm_ascend.utils import AscendDeviceType

        soc_version = torch_npu.npu.get_soc_version()
        if 220 <= soc_version <= 225:
            cur_device_type = AscendDeviceType.A2
        elif 250 <= soc_version <= 255:
            cur_device_type = AscendDeviceType.A3
        elif 200 <= soc_version <= 205:
            cur_device_type = AscendDeviceType._310P
        elif soc_version == 260:
            cur_device_type = AscendDeviceType.A5
        else:
            raise RuntimeError(f"Can not support soc_version: {soc_version}.")

        return cur_device_type == AscendDeviceType.A2

    if vllm.__version__ == "0.11.0":
        is_A2 = _is_ascend_soc_version_A2_v011_local()
    elif vllm.__version__ == "0.13.0":
        is_A2 = _is_ascend_soc_version_A2_v013_local()
    else:
        is_A2 = False

    if is_A2:
        VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE = bool(int(os.getenv("VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", "0")))
        if VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE:
            raise AssertionError(
                "AscendSocVersion.A2 is not support VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE \
                in Single-card multi-process scenario now. "
            )


def vllm_ascend_v013_select_moe_comm_method_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        moe_comm_method = fn(*args, **kwargs)
        from vllm_ascend.ascend_forward_context import MoECommType
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        ascend_device_type = get_ascend_device_type()

        # AscendSocVersion.A2 is not support MC2 in Single-card multi-process scenario now.
        if ascend_device_type in {AscendDeviceType.A2} and moe_comm_method == MoECommType.MC2:
            moe_comm_method = MoECommType.ALLGATHER

        return moe_comm_method

    return wrapper


def vllm_ascend_v013_matmul_and_reduce_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        ascend_device_type = get_ascend_device_type()
        # AscendSocVersion.A2 is not support MC2 in Single-card multi-process scenario now.
        if ascend_device_type in {AscendDeviceType.A2}:
            from vllm.forward_context import get_forward_context

            try:
                forward_context = get_forward_context()
                forward_context.mmrs_fusion = False
            except AssertionError:
                # forward_context.mmrs_fusion will be false in matmul_and_reduce func.
                pass
        return fn(self, *args, **kwargs)

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


if is_torch_npu_available(check_device=False):
    import vllm
    from packaging import version

    _VLLM_VERSION = version.parse(vllm.__version__)
    if _VLLM_VERSION >= version.parse("0.13.0"):
        # Disable flash_attn in RotaryEmbedding (NPU) when VLLM >= 0.13
        patch_vllm013_rotary_emb()

    VERL_NPU_ENABLE_A2_PATCH_VLLM_ASCEND_MC2 = bool(int(os.getenv("VERL_NPU_ENABLE_A2_PATCH_VLLM_ASCEND_MC2", "1")))
    if VERL_NPU_ENABLE_A2_PATCH_VLLM_ASCEND_MC2:
        # only support vllm 0.13 and 0.11 now.
        if _VLLM_VERSION >= version.parse("0.13.0"):
            from vllm_ascend import ascend_forward_context
            from vllm_ascend.ops.linear_op import SequenceRowParallelOp

            ascend_forward_context.select_moe_comm_method = vllm_ascend_v013_select_moe_comm_method_wrapper(
                ascend_forward_context.select_moe_comm_method
            )
            SequenceRowParallelOp.matmul_and_reduce = vllm_ascend_v013_matmul_and_reduce_wrapper(
                SequenceRowParallelOp.matmul_and_reduce
            )
        elif _VLLM_VERSION >= version.parse("0.11.0"):
            from vllm_ascend.ops.linear_op import SequenceRowParallelOp
            from vllm_ascend.worker.model_runner_v1 import NPUModelRunner

            NPUModelRunner._select_moe_comm_method = vllm_ascend_v011_select_moe_comm_method_wrapper(
                NPUModelRunner._select_moe_comm_method
            )
            SequenceRowParallelOp.matmul_and_reduce = vllm_ascend_v011_matmul_and_reduce_wrapper(
                SequenceRowParallelOp.matmul_and_reduce
            )
