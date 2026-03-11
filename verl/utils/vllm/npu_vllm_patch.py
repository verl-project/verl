import os
from functools import wraps
from verl.utils.device import is_torch_npu_available


def vllm_ascend_select_moe_comm_method_wrapper(fn):
    @wraps(fn)
    def wrapper(self, num_tokens, with_prefill):
        moe_comm_method = fn(self, num_tokens, with_prefill)
        from vllm_ascend.ascend_forward_context import MoECommType
        from vllm_ascend.utils import get_ascend_soc_version, AscendSocVersion
        soc_version = get_ascend_soc_version()

        # AscendSocVersion.A2 is not support MC2 in Single-card multi-process scenario now.
        if soc_version in {AscendSocVersion.A2} and moe_comm_method == MoECommType.MC2:
            quant_type = getattr(self.vllm_config.model_config.hf_config,
                                'moe_quantize', None)
            # Currently, w4a8_dynamic does not support allgatherep
            if quant_type == "w4a8_dynamic":
                moe_comm_method = MoECommType.ALLTOALL
            else:
                moe_comm_method = MoECommType.ALLGATHER

        if with_prefill:
            from vllm_ascend.utils import enable_sp
            if enable_sp():
                moe_comm_method = MoECommType.ALLGATHER
            else:
                moe_comm_method = MoECommType.NAIVE_MULTICAST

        return moe_comm_method

    return wrapper

def vllm_ascend_matmul_and_reduce_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        from vllm_ascend.utils import get_ascend_soc_version, AscendSocVersion
        soc_version = get_ascend_soc_version()
        #AscendSocVersion.A2 is not support MC2 in Single-card multi-process scenario now.
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
    from vllm_ascend.utils import AscendSocVersion
    def get_ascend_soc_version_local():
        soc_version = torch_npu.npu.get_soc_version()
        if 220 <= soc_version <= 225:
            _ascend_soc_version = AscendSocVersion.A2
        elif 250 <= soc_version <= 255:
            _ascend_soc_version = AscendSocVersion.A3
        else:
            _ascend_soc_version = AscendSocVersion.UNDEFINED
        return _ascend_soc_version
    soc_version = get_ascend_soc_version_local()
    if soc_version in {AscendSocVersion.A2}:
        VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE = bool(int(os.getenv("VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE", '0')))
        if VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE:
            raise AssertionError(
                "AscendSocVersion.A2 is not support VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE in Single-card multi-process scenario now. "
            )


if is_torch_npu_available(check_device=False):
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner
    from vllm_ascend.ops.linear_op import SequenceRowParallelOp
    NPUModelRunner._select_moe_comm_method = vllm_ascend_select_moe_comm_method_wrapper(NPUModelRunner._select_moe_comm_method)
    SequenceRowParallelOp.matmul_and_reduce = vllm_ascend_matmul_and_reduce_wrapper(SequenceRowParallelOp.matmul_and_reduce)