import base64
import inspect
import pickle
from typing import Optional

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
                    if hasattr(module, "pre_reload_weights") and not getattr(
                        module, "_weights_removed", False
                    ):
                        module.pre_reload_weights()
                setattr(self.engine.model_engine.model, "first_pre_reload_weights", True)

            if ipc_handles is not None:
                device_uuid = get_device_uuid()
                handles = ipc_handles.get(device_uuid, None)
                if handles is not None:
                    weights = pickle.loads(base64.b64decode(handles))
                    model = self.engine.model_engine.model
                    load_weights_args = inspect.getfullargspec(model.load_weights).args
                    supports_partial_loading = "allow_partial_loading" in load_weights_args

                    if supports_partial_loading:
                        self.engine.model_engine.model_loader.reload(
                            model, weights, allow_partial_loading=True
                        )
                    else:
                        self.engine.model_engine.model_loader.reload(
                            model, weights, allow_partial_loading=False
                        )
            else:
                for module in self.engine.model_engine.model.modules():
                    if hasattr(module, "process_weights_after_loading") and not getattr(
                        module, "_weights_removed", False
                    ):
                        module.process_weights_after_loading()
                    if hasattr(module, "post_load_weights") and not getattr(
                        module, "_weights_removed", False
                    ):
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
