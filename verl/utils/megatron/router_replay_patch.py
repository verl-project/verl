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
import inspect
import logging
import types
import warnings
from enum import Enum
from functools import wraps

import torch

try:
    from megatron.core.transformer.moe.moe_utils import (
        apply_router_token_dropping,
        compute_routing_scores_for_aux_loss,
        group_limited_topk,
    )
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
except ImportError:
    warnings.warn("NPU not support router replay for now.", stacklevel=2)
    MoEAlltoAllTokenDispatcher = None
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig

# https://github.com/THUDM/slime/blob/main/slime/utils/routing_replay.py

logger = logging.getLogger(__name__)


class RouterReplayAction(Enum):
    RECORD = "record"
    REPLAY_FORWARD = "replay_forward"
    REPLAY_BACKWARD = "replay_backward"


class RouterReplay:
    """
    A class to manage the recording and replaying of MoE routing decisions.
    It holds all router instances and provides static methods to globally
    control recording and replaying.
    """

    # Static variable to hold all router instances, one per MoE layer.
    router_instances = []

    @staticmethod
    def set_replay_data(all_layers_topk_indices: list):
        """
        Distributes the topk indices for all layers to their respective RouterReplay instances.
        :param all_layers_topk_indices: A list of tensors, where each tensor contains the
                                        topk indices for a specific layer. The order
                                        must match the instantiation order of the routers.
        """
        if len(all_layers_topk_indices) != len(RouterReplay.router_instances):
            raise ValueError(
                f"The number of replay tensors ({len(all_layers_topk_indices)}) "
                f"does not match the number of router instances ({len(RouterReplay.router_instances)})."
            )
        for i, router_instance in enumerate(RouterReplay.router_instances):
            router_instance.set_target_indices(all_layers_topk_indices[i])

    @staticmethod
    def get_recorded_data() -> list:
        """
        Collects the recorded topk indices from all RouterReplay instances.
        :return: A list of tensors, each containing the recorded topk indices for a layer.
        """
        return [router.get_recorded_indices() for router in RouterReplay.router_instances]

    @staticmethod
    def clear_global_indices():
        """Clears the recorded and target topk indices in all instances."""
        for router in RouterReplay.router_instances:
            router.clear_indices()

    def __init__(self):
        """Initializes a RouterReplay instance for a specific layer."""
        self.target_topk_idx = None  # For replay
        self.target_replay_mask = None  # Optional per-token replay gate
        self.recorded_topk_idx = None  # For recording
        self.router_replay_action = None  # Router replay action for this layer
        self.replay_backward_list = []  # List of tensors for backward pass replay
        self.replay_backward_mask_list = []  # List of optional replay gates
        RouterReplay.router_instances.append(self)

    def set_target_indices(
        self,
        topk_indices: torch.Tensor,
        replay_mask: torch.Tensor | None = None,
    ):
        """Sets the target topk indices for replay."""
        self.target_topk_idx = topk_indices
        self.target_replay_mask = replay_mask
        self.replay_backward_list.append(topk_indices)
        self.replay_backward_mask_list.append(replay_mask)

    def get_recorded_indices(self):
        """Returns the recorded topk indices."""
        return self.recorded_topk_idx

    def record_indices(self, topk_indices: torch.Tensor):
        """Records the topk indices."""
        self.recorded_topk_idx = topk_indices

    def clear_indices(self):
        """Clears the recorded and target topk indices."""
        self.recorded_topk_idx = None
        self.target_topk_idx = None
        self.target_replay_mask = None
        self.replay_backward_list = []
        self.replay_backward_mask_list = []

    def set_router_replay_action(self, router_replay_action: RouterReplayAction):
        """Sets the router replay action for this layer."""
        self.router_replay_action = router_replay_action

    def clear_router_replay_action(self):
        """Clears the router replay action for this layer."""
        self.router_replay_action = None

    @staticmethod
    def set_global_router_replay_action(router_replay_action: RouterReplayAction):
        """Sets the router replay action for all router instances."""
        for router in RouterReplay.router_instances:
            router.set_router_replay_action(router_replay_action)

    @staticmethod
    def clear_global_router_replay_action():
        """Clears the router replay action for all router instances."""
        for router in RouterReplay.router_instances:
            router.clear_router_replay_action()


def _apply_router_replay_indices(
    native_top_indices: torch.Tensor,
    target_topk_idx: torch.Tensor,
    replay_mask: torch.Tensor | None,
    error_prefix: str,
) -> torch.Tensor:
    target_topk_idx = target_topk_idx.to(native_top_indices.device)
    if target_topk_idx.shape != native_top_indices.shape:
        raise RuntimeError(
            f"{error_prefix}: target_topk_idx shape {tuple(target_topk_idx.shape)} "
            f"does not match native top-k shape {tuple(native_top_indices.shape)}."
        )
    if replay_mask is None:
        return target_topk_idx

    replay_mask = replay_mask.to(native_top_indices.device).bool()
    if replay_mask.shape[0] != target_topk_idx.shape[0]:
        raise RuntimeError(
            f"{error_prefix}: replay_mask has {replay_mask.shape[0]} rows "
            f"but target_topk_idx has {target_topk_idx.shape[0]} rows."
        )
    return torch.where(replay_mask.unsqueeze(-1), target_topk_idx, native_top_indices)


def _patched_topk_routing_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool,
    num_groups: int,
    group_topk: int,
    score_function: str,
    expert_bias: torch.Tensor,
    router_replay: RouterReplay,
    scaling_factor: float,
):
    """
    Patched version of topk_routing_with_score_function that supports router replay.
    """
    num_tokens, num_experts = logits.shape

    def _compute_topk(scores, topk, num_groups=None, group_topk=None):
        if group_topk:
            return group_limited_topk(
                scores=scores,
                topk=topk,
                num_tokens=num_tokens,
                num_experts=num_experts,
                num_groups=num_groups,
                group_topk=group_topk,
            )
        else:
            return torch.topk(scores, k=topk, dim=1)

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        # Default behavior if no replay is active

        routing_action = router_replay.router_replay_action if router_replay is not None else None

        if routing_action is None:
            return _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)

        if routing_action == RouterReplayAction.RECORD:
            probs, top_indices = _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            if router_replay is not None:
                router_replay.record_indices(top_indices)
            return probs, top_indices

        def replay_topk(target_topk_idx, replay_mask):
            _, native_top_indices = _compute_topk(scores, topk, num_groups=num_groups, group_topk=group_topk)
            top_indices = _apply_router_replay_indices(
                native_top_indices, target_topk_idx, replay_mask, "router_replay REPLAY"
            )
            probs = scores.gather(1, top_indices)
            return probs, top_indices

        if routing_action == RouterReplayAction.REPLAY_FORWARD:
            if router_replay.target_topk_idx is None:
                raise RuntimeError("router_replay REPLAY_FORWARD requires target top-k indices.")

            return replay_topk(
                router_replay.target_topk_idx,
                router_replay.target_replay_mask,
            )
        elif routing_action == RouterReplayAction.REPLAY_BACKWARD:
            if not router_replay.replay_backward_list:
                raise RuntimeError("router_replay REPLAY_BACKWARD requires queued top-k indices.")

            # Use the last recorded indices for backward replay
            top_indices = router_replay.replay_backward_list.pop(0)
            replay_mask = router_replay.replay_backward_mask_list.pop(0)
            return replay_topk(top_indices, replay_mask)
        else:
            raise RuntimeError(f"Unknown router replay action: {routing_action}.")

    if score_function == "softmax":
        if use_pre_softmax:
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
    elif score_function in ("sigmoid", "sqrtsoftplus"):
        if score_function == "sigmoid":
            scores = torch.sigmoid(logits.float())
        else:
            scores = torch.nn.functional.softplus(logits.float()).sqrt()
        if expert_bias is not None:
            scores_for_routing = scores + expert_bias.float()
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
            scores = torch.gather(scores, dim=1, index=top_indices)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores
    else:
        raise ValueError(f"Invalid score_function: {score_function}")

    if scaling_factor:
        probs = probs * scaling_factor

    probs = probs.type_as(logits)

    if torch.are_deterministic_algorithms_enabled():
        # build [num_tokens, num_experts] from [num_tokens, topk]
        routing_probs = torch.zeros_like(logits)
        rows = torch.arange(num_tokens, device=logits.device).unsqueeze(1)
        routing_probs.index_put_((rows, top_indices), probs, accumulate=False)

        routing_map = torch.zeros_like(logits, dtype=logits.dtype)
        routing_map.index_put_((rows, top_indices), torch.ones_like(probs, dtype=routing_map.dtype), accumulate=False)
        routing_map = routing_map.bool()
    else:
        # TODO Try using element-wise operations instead of scatter?
        routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()

    return routing_probs, routing_map


def _get_aux_loss_coeff(_self, aux_loss_type: str) -> float:
    """Return the aux loss coeff for the given auxiliary loss type.
    If the auxiliary loss type is not found, return 0.0.
    """
    if isinstance(_self.routing_type, str):
        if _self.routing_type == aux_loss_type:
            return _self.config.moe_aux_loss_coeff
    if isinstance(_self.routing_type, list):
        try:
            idx = _self.routing_type.index(aux_loss_type)
            return _self.config.moe_aux_loss_coeff[idx]
        except (ValueError, IndexError):
            return 0.0
    return 0.0


def _is_aux_loss_enabled(_self) -> bool:
    """Check if the auxiliary loss is enabled."""
    for aux_loss_type in ["aux_loss", "seq_aux_loss", "global_aux_loss"]:
        if _get_aux_loss_coeff(_self, aux_loss_type) > 0:
            return True
    return False


def _hash_routing_with_replay(self, logits: torch.Tensor, input_ids: torch.Tensor):
    """Hash-based DSv4 routing with optional router replay substitution."""
    if self.score_function == "softmax":
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32)
    elif self.score_function == "sigmoid":
        scores = torch.sigmoid(logits.float())
    elif self.score_function == "sqrtsoftplus":
        scores = torch.nn.functional.softplus(logits.float()).sqrt()
    else:
        raise ValueError(f"Invalid score_function: {self.score_function}")

    flat_ids = input_ids.T.reshape(-1)
    native_top_indices = self.tid2eid[flat_ids].long()
    top_indices = native_top_indices

    router_replay = getattr(self, "router_replay", None)
    routing_action = router_replay.router_replay_action if router_replay is not None else None
    if routing_action == RouterReplayAction.RECORD:
        router_replay.record_indices(native_top_indices)
    elif routing_action == RouterReplayAction.REPLAY_FORWARD:
        if router_replay.target_topk_idx is None:
            raise RuntimeError("router_replay hash REPLAY_FORWARD requires target top-k indices.")
        top_indices = _apply_router_replay_indices(
            native_top_indices,
            router_replay.target_topk_idx,
            router_replay.target_replay_mask,
            "router_replay hash REPLAY",
        )
    elif routing_action == RouterReplayAction.REPLAY_BACKWARD:
        if not router_replay.replay_backward_list:
            raise RuntimeError("router_replay hash REPLAY_BACKWARD requires queued top-k indices.")
        top_indices = _apply_router_replay_indices(
            native_top_indices,
            router_replay.replay_backward_list.pop(0),
            router_replay.replay_backward_mask_list.pop(0),
            "router_replay hash REPLAY",
        )
    elif routing_action is not None:
        raise RuntimeError(f"Unknown router replay action: {routing_action}.")

    probs = scores.gather(1, top_indices)
    if self.score_function != "softmax":
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-20)
    if self.config.moe_router_topk_scaling_factor:
        probs = probs * self.config.moe_router_topk_scaling_factor
    probs = probs.type_as(logits)

    routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
    routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()

    return routing_probs, routing_map


def patched_routing(self, logits: torch.Tensor, *args, **kwargs):
    """Top-k routing function

    Args:
        logits (torch.Tensor): Logits tensor after gating.

    Returns:
        probs (torch.Tensor): The probabilities of token to experts assignment.
        routing_map (torch.Tensor): The mapping of token to experts assignment,
            with shape [num_tokens, num_experts].
    """
    padding_mask = kwargs.pop("padding_mask", None)
    input_ids = kwargs.pop("input_ids", None)
    if len(args) > 0:
        padding_mask = args[0]
    if len(args) > 1:
        input_ids = args[1]

    seq_length, bsz = logits.shape[:2]
    logits = logits.view(-1, self.config.num_moe_experts)
    if padding_mask is not None:
        padding_mask = padding_mask.reshape(-1)

    # Apply Z-Loss
    logits = self.apply_z_loss(logits, padding_mask=padding_mask)

    # Megatron versions before 0.14.0 do not have 'moe_router_fusion' in TransformerConfig.
    # We use getattr with a default value of False to ensure compatibility across different
    # versions of Megatron-LM and MindSpeed.
    moe_router_fusion = getattr(self.config, "moe_router_fusion", False)

    # Calculate probs and routing_map for token dispatching
    if getattr(self, "is_hash_layer", False):
        if input_ids is None:
            raise RuntimeError("input_ids is required for hash-based router replay.")
        probs, routing_map = _hash_routing_with_replay(self, logits, input_ids)
    elif self.routing_type == "sinkhorn":
        probs, routing_map = self.sinkhorn_load_balancing(logits)
    else:
        probs, routing_map = _patched_topk_routing_with_score_function(
            logits=logits,
            topk=self.topk,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
            router_replay=getattr(self, "router_replay", None),
        )

    # Apply token dropping to probs and routing_map.
    if self.config.moe_expert_capacity_factor is not None:
        probs, routing_map = apply_router_token_dropping(
            probs,
            routing_map,
            router_topk=self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            drop_policy=self.config.moe_token_drop_policy,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
        )

    if not hasattr(self, "is_aux_loss_enabled"):
        self.is_aux_loss_enabled = types.MethodType(_is_aux_loss_enabled, self)
    # Apply each aux loss type and attach aux loss autograd function to probs
    if self.training and torch.is_grad_enabled() and self.is_aux_loss_enabled():
        # Calculate scores and routing_map for aux loss
        routing_map_for_aux_loss, scores_for_aux_loss = compute_routing_scores_for_aux_loss(
            logits,
            self.topk,
            self.score_function,
            fused=moe_router_fusion,
            padding_mask=padding_mask,
        )
        probs = self._apply_aux_loss(
            probs, scores_for_aux_loss, routing_map_for_aux_loss, with_padding_mask=padding_mask is not None
        )
        probs = self._apply_seq_aux_loss(
            probs,
            scores_for_aux_loss,
            routing_map_for_aux_loss,
            seq_length,
            bsz,
            with_padding_mask=padding_mask is not None,
        )
        probs = self._apply_global_aux_loss(
            probs, scores_for_aux_loss, routing_map_for_aux_loss, with_padding_mask=padding_mask is not None
        )

    # Update expert bias and tokens_per_expert
    # Prevent extra local tokens accumulation on evaluation or activation recomputation
    self._apply_expert_bias(routing_map, padding_mask=padding_mask)

    return probs, routing_map


def apply_router_replay_patch():
    """
    Applies the monkey patch for MoE Router Replay functionality.
    This patch dynamically adds the fallback 'enable_routing_replay' attribute to TransformerConfig
    and modifies the TopKRouter to support recording and replaying of routing decisions.
    """
    logger.info("Applying Router Replay Patch...")
    # Clear router instances to avoid state leakage between model initializations.
    RouterReplay.router_instances.clear()
    # Step 1: Patch TransformerConfig to include the feature flag

    try:
        sig = inspect.signature(TransformerConfig.__init__)
        native_params = sig.parameters
        params = list(sig.parameters.values())
    except Exception:
        sig = None
        native_params = {}
        params = []

    ext_attrs = ["enable_routing_replay"]

    # Update __signature__ to prevent NPU/MindSpeed wrappers from filtering out or blocking custom parameters.
    for attr in ext_attrs:
        if attr not in native_params:
            if sig:
                new_param = inspect.Parameter(attr, inspect.Parameter.KEYWORD_ONLY, default=False)
                if params and params[-1].kind == inspect.Parameter.VAR_KEYWORD:
                    params.insert(-1, new_param)
                else:
                    params.append(new_param)

    if sig:
        try:
            TransformerConfig.__init__.__signature__ = sig.replace(parameters=params)
        except Exception as e:
            logger.warning("Failed to update signature metadata: %s", e)

    if not hasattr(TransformerConfig, "_verl_router_patched"):
        # Store original __init__ method
        original_tf_config_init = TransformerConfig.__init__

        # Define new __init__ method that safely handles enable_routing_replay parameter
        @wraps(original_tf_config_init)
        def patched_tf_config_init(self, *args, **kwargs):
            # Simple solution: remove the unknown parameter before calling original constructor
            enable_routing_replay = kwargs.get("enable_routing_replay", False)
            if "enable_routing_replay" not in native_params:
                enable_routing_replay = kwargs.pop("enable_routing_replay", False)

            # Call original constructor with remaining kwargs
            original_tf_config_init(self, *args, **kwargs)

            # Set the instance attribute
            self.enable_routing_replay = enable_routing_replay

        # Apply the patch
        TransformerConfig.__init__ = patched_tf_config_init
        TransformerConfig._verl_router_patched = True

    # Step 2: Patch TopKRouter only once to ensure idempotency.
    if hasattr(TopKRouter, "_router_replay_patched"):
        return

    original_init = TopKRouter.__init__

    def _router_replay_enabled(config):
        return getattr(config, "enable_routing_replay", False) or getattr(config, "moe_enable_routing_replay", False)

    # Step 3: Define the new __init__ method
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.router_replay = None
        if _router_replay_enabled(self.config):
            self.router_replay = RouterReplay()

    # Step 4: Patch MoEAlltoAllTokenDispatcher.preprocess to handle router replay
    # When router replay is enabled, duplicate indices in top_indices can cause
    # routing_map.sum() < num_tokens * topk, leading to split size mismatch in alltoall.
    if MoEAlltoAllTokenDispatcher is not None and not hasattr(MoEAlltoAllTokenDispatcher, "_preprocess_patched"):
        original_preprocess = MoEAlltoAllTokenDispatcher.preprocess

        def patched_preprocess(self, routing_map):
            """Patched preprocess that handles router replay correctly for alltoall dispatcher."""
            # Call original preprocess
            result = original_preprocess(self, routing_map)

            # Fix num_out_tokens when router replay is enabled
            if (
                _router_replay_enabled(self.config)
                and not self.drop_and_pad
                and self.config.moe_expert_capacity_factor is None
                and not (
                    getattr(self.config, "moe_router_padding_for_quantization", None)
                    or getattr(self.config, "moe_router_padding_for_fp8", None)
                )
            ):
                # With router replay, duplicate indices can reduce the actual routed
                # token count, so derive it from the routing map instead.
                self.num_out_tokens = int(routing_map.sum().item())

            return result

        MoEAlltoAllTokenDispatcher.preprocess = patched_preprocess
        MoEAlltoAllTokenDispatcher._preprocess_patched = True

    # Step 5: Apply the patches
    TopKRouter.__init__ = patched_init
    TopKRouter.routing = patched_routing
    TopKRouter._router_replay_patched = True
