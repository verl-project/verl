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
from tensorrt_llm import serialization
from tensorrt_llm._ray_utils import control_action_decorator
from tensorrt_llm._torch.modules.fused_moe.moe_load_balancer import MoeLoadBalancer
from tensorrt_llm._torch.utils import get_device_uuid
from tensorrt_llm.logger import logger


class WorkerExtension:
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
                    # Allow basic builtins and all torch modules
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
                    }
                    all_handles = serialization.loads(
                        decoded_data,
                        approved_imports=approved_imports,
                        approved_module_patterns=[r"^torch.*"],
                    )

                    # Verify the result is a list as expected
                    if not isinstance(all_handles, list):
                        raise ValueError(f"Deserialized data must be a list, got {type(all_handles).__name__} instead")
                else:
                    # Data is already in the correct format (backward compatibility)
                    all_handles = serialized_handles

                for param_name, tensor_handle in all_handles:
                    func, args = tensor_handle
                    list_args = list(args)
                    list_args[6] = self.device_id
                    tensor = func(*list_args)
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

    def check_weights_updated(self) -> bool:
        """Check if the weights are updated to 0."""
        weights_updated = True
        for name, p in self.engine.model_engine.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(p, torch.zeros_like(p))
        return weights_updated
