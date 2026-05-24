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
import base64
import inspect
from typing import Optional

import torch
from verl.workers.rollout.trtllm_rollout import _w4a4_compat  # noqa: F401  (registers float8 storage stubs at import)

# Defer tensorrt_llm imports to avoid FlashInfer's check_cuda_arch() crash
# when this module is loaded on CPU-only Ray actors. The module is normally
# loaded only on GPU workers via string path in trtllm_async_server.py, but
# guard defensively in case of transitive imports.
try:
    from tensorrt_llm import serialization
    from tensorrt_llm._ray_utils import control_action_decorator
    from tensorrt_llm._torch.modules.fused_moe.moe_load_balancer import MoeLoadBalancer
    from tensorrt_llm._torch.utils import get_device_uuid
    from tensorrt_llm.llmapi.rlhf_utils import WorkerExtension as TrtllmWorkerExtension
    from tensorrt_llm.logger import logger
except (ImportError, RuntimeError):
    # On CPU actors without CUDA, these imports may fail.
    # The class below won't be usable, but the module can be imported safely.
    serialization = None
    control_action_decorator = lambda f: f  # noqa: E731 — identity fallback
    MoeLoadBalancer = None
    get_device_uuid = None
    TrtllmWorkerExtension = object
    logger = None


# Monkey-patch tensorrt_llm.serialization.loads to permit torch.storage._load_from_bytes
# and float8 storage stubs. The parent rlhf_utils.WorkerExtension.update_weights calls
# serialization.loads with a hard-coded approved_imports that omits these — needed for
# W4A4 weight_scale (float8_e4m3fn) IPC round-trip.
if serialization is not None and not getattr(serialization, "_verl_w4a4_patched", False):
    _orig_loads = serialization.loads

    def _patched_loads(data, approved_imports=None, **kwargs):
        extra = {
            "torch.storage": ["_load_from_bytes", "UntypedStorage", "TypedStorage", "_TypedStorage"],
            "torch._utils": ["_rebuild_tensor_v2"],
            "torch.multiprocessing.reductions": ["rebuild_cuda_tensor", "rebuild_tensor"],
            "torch": ["Tensor", "Size", "dtype", "device", "Float8_e4m3fnStorage", "Float8_e5m2Storage",
                      "float8_e4m3fn", "float8_e5m2"],
        }
        if approved_imports is None:
            approved_imports = {}
        approved_imports = dict(approved_imports)
        for _mod, _names in extra.items():
            merged = list(approved_imports.get(_mod, []))
            for _n in _names:
                if _n not in merged:
                    merged.append(_n)
            approved_imports[_mod] = merged
        return _orig_loads(data, approved_imports=approved_imports, **kwargs)

    serialization.loads = _patched_loads
    serialization._verl_w4a4_patched = True


class WorkerExtension(TrtllmWorkerExtension):
    def __init__(self):
        pass

    @control_action_decorator
    def supports_partial_loading(self) -> bool:
        """Check if the model supports partial weight loading."""
        try:
            model = self.engine.model_engine.model
            load_weights_args = inspect.getfullargspec(model.load_weights).args
            return "allow_partial_loading" in load_weights_args
        except Exception as e:
            logger.warning(f"Failed to check partial loading support: {e}")
            return False

    @control_action_decorator
    def update_weights(self, ipc_handles: Optional[dict] = None):
        try:
            if not hasattr(self.engine.model_engine.model, "first_pre_reload_weights"):
                for module in self.engine.model_engine.model.modules():
                    if hasattr(module, "pre_reload_weights") and not getattr(module, "_weights_removed", False):
                        module.pre_reload_weights()
                self.engine.model_engine.model.first_pre_reload_weights = True

            if ipc_handles is not None:
                logger.info("Update weights from IPC handles")
                device_uuid = get_device_uuid(self.device_id)

                if device_uuid not in ipc_handles:
                    raise ValueError(f"Device UUID {device_uuid} not found in ipc_handles")

                weights = {}

                serialized_handles = ipc_handles[device_uuid]
                if isinstance(serialized_handles, str):
                    # Data is base64-encoded pickled bytes - deserialize it
                    # using restricted unpickler from tensorrt_llm.serialization
                    logger.info("Deserializing base64-encoded weight handles")
                    decoded_data = base64.b64decode(serialized_handles)
                    # Allow basic builtins and torch tensor reconstruction classes
                    approved_imports = {
                        "builtins": [
                            "list",
                            "tuple",
                            "str",
                            "int",
                            "float",
                            "bool",
                            "bytes",
                            "dict",
                            "NoneType",
                            "type",
                        ],
                        "torch": [
                            "Tensor",
                            "FloatTensor",
                            "DoubleTensor",
                            "HalfTensor",
                            "BFloat16Tensor",
                            "IntTensor",
                            "LongTensor",
                            "ShortTensor",
                            "CharTensor",
                            "ByteTensor",
                            "BoolTensor",
                            "Size",
                            "dtype",
                            "device",
                            "float32",
                            "float16",
                            "bfloat16",
                            "int32",
                            "int64",
                            "int16",
                            "int8",
                            "uint8",
                            "bool",
                        ],
                        "torch.multiprocessing.reductions": [
                            "rebuild_cuda_tensor",
                            "rebuild_tensor",
                        ],
                        "torch._utils": [
                            "_rebuild_tensor_v2",
                        ],
                        "torch.storage": [
                            "_load_from_bytes",
                            "_TypedStorage",
                            "UntypedStorage",
                            "TypedStorage",
                        ],
                    }
                    all_handles = serialization.loads(
                        decoded_data,
                        approved_imports=approved_imports,
                    )

                    # Verify the result is a list as expected
                    if not isinstance(all_handles, list):
                        raise ValueError(f"Deserialized data must be a list, got {type(all_handles).__name__} instead")
                else:
                    # Data is already in the correct format (backward compatibility)
                    all_handles = serialized_handles

                for entry in all_handles:
                    # Support both 2-tuple (legacy) and 3-tuple (with orig_dtype_str
                    # for float8 transport-as-uint8 round-trip).
                    if len(entry) == 3:
                        param_name, tensor_handle, orig_dtype_str = entry
                    else:
                        param_name, tensor_handle = entry
                        orig_dtype_str = None
                    func, args = tensor_handle
                    list_args = list(args)
                    list_args[6] = self.device_id
                    tensor = func(*list_args)
                    if orig_dtype_str is not None:
                        tensor = tensor.view(getattr(torch, orig_dtype_str))
                    weights[param_name] = tensor

                logger.info(f"weights key size: {len(weights.keys())}")

                # Check if model supports partial loading and use appropriate strategy
                model = self.engine.model_engine.model
                load_weights_args = inspect.getfullargspec(model.load_weights).args
                supports_partial_loading = "allow_partial_loading" in load_weights_args

                if supports_partial_loading:
                    self.engine.model_engine.model_loader.reload(model, weights, allow_partial_loading=True)
                else:
                    self.engine.model_engine.model_loader.reload(model, weights, allow_partial_loading=False)
            else:
                logger.info("Finalize update weights")
                for module in self.engine.model_engine.model.modules():
                    if hasattr(module, "process_weights_after_loading") and not getattr(
                        module, "_weights_removed", False
                    ):
                        module.process_weights_after_loading()
                    if hasattr(module, "post_load_weights") and not getattr(module, "_weights_removed", False):
                        module.post_load_weights()
                moe_load_balancer = getattr(self.engine.model_engine, "moe_load_balancer", None)
                if isinstance(moe_load_balancer, MoeLoadBalancer):
                    moe_load_balancer.register_weight_slots_after_to_cuda()
                    logger.info("moe_load_balancer finalizing model...")
                    moe_load_balancer.finalize_model()
                    logger.info("moe_load_balancer finalize model done")
                self.engine.reset_prefix_cache()
                delattr(self.engine.model_engine.model, "first_pre_reload_weights")

        except Exception as e:
            logger.error("Encountered an error in update_weights")
            raise e
