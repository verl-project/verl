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

"""Make vLLM's monolithic fused-MoE path observable to router replay.

vLLM 0.26 and 0.27.1 call ``router.select_experts`` on its modular MoE path.  The
FlashInfer TRT-LLM NVFP4 kernel is monolithic and performs routing internally,
so the call that drives ``RoutedExpertsCapturer`` is otherwise skipped.  R3
would then receive an all-zero route tensor.

This is deliberately a narrow, fail-closed compatibility patch for the vLLM
0.26 / 0.27.1 private API.  It should be removed when the behavior is available in
upstream vLLM.
"""

import functools
import inspect
import logging
from importlib.metadata import version

import numpy as np
from packaging.version import Version

logger = logging.getLogger(__name__)

_PATCH_MARKER = "_verl_r3_monolithic_capture_patch"
_EXPECTED_PARAMETERS = (
    "self",
    "hidden_states",
    "router_logits",
    "shared_experts_input",
    "input_ids",
)
_route_marker_printed = False


def attest_r3_rollout_routes(routed_experts):
    """Reject the all-zero route payload produced by a missed capture hook."""

    if routed_experts is None:
        raise RuntimeError("R3 rollout requested routed experts, but vLLM returned None")
    route_array = np.asarray(routed_experts)
    if route_array.ndim < 2:
        raise RuntimeError(f"R3 rollout returned malformed routed experts with shape {route_array.shape}")

    # vLLM 0.26 returns only tokens that have actually gone through a forward
    # pass (normally prompt + sampled tokens - 1). It does not append a filler
    # row for the final sampled token, so validate the complete returned array.
    if route_array.shape[0] == 0:
        raise RuntimeError("R3 rollout returned an empty routed-experts array")
    if not np.any(route_array):
        raise RuntimeError("R3 rollout routes are all zero; the monolithic fused-MoE capture hook did not fire")

    global _route_marker_printed
    if not _route_marker_printed:
        _route_marker_printed = True
        logger.warning(
            "VERL_R3_ROLLOUT_ROUTES PASS shape=%s nonzero=%d",
            route_array.shape,
            int(np.count_nonzero(route_array)),
        )
    return routed_experts


def _monolithic_branch(method) -> str:
    """Return the audited vLLM monolithic branch or fail closed."""

    try:
        source = inspect.getsource(method)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "cannot inspect vLLM MoERunner._apply_quant_method; refusing an unverified R3 monolithic-MoE patch"
        ) from exc

    branch_start = source.find("if self.routed_experts.quant_method.is_monolithic:")
    branch_end = source.find("\n        else:", branch_start)
    if branch_start < 0 or branch_end < 0:
        raise RuntimeError(
            "vLLM MoERunner._apply_quant_method no longer has the expected "
            "monolithic branch; refusing an unverified R3 patch"
        )
    branch = source[branch_start:branch_end]
    if "self.routed_experts.forward_monolithic(" not in branch:
        raise RuntimeError(
            "vLLM monolithic MoE branch no longer calls forward_monolithic; refusing an unverified R3 patch"
        )
    return branch


def _patch_moe_runner_class(moe_runner_cls) -> str:
    """Patch one vLLM-like ``MoERunner`` class; split out for unit tests."""

    current_method = moe_runner_cls._apply_quant_method
    if getattr(current_method, _PATCH_MARKER, False):
        return "already_patched"

    parameters = tuple(inspect.signature(current_method).parameters)
    if parameters != _EXPECTED_PARAMETERS:
        raise RuntimeError(
            f"unexpected vLLM MoERunner._apply_quant_method signature {parameters}; expected {_EXPECTED_PARAMETERS}"
        )

    branch = _monolithic_branch(current_method)
    if "capture_fn" in branch and "router.select_experts(" in branch:
        logger.warning("VERL_R3_MONOLITHIC_CAPTURE_PATCH PASS implementation=upstream_native")
        return "upstream_native"

    original = current_method

    @functools.wraps(original)
    def _apply_quant_method_with_capture(
        self,
        hidden_states,
        router_logits,
        shared_experts_input,
        input_ids=None,
    ):
        quant_method = self.routed_experts.quant_method
        capture_fn = getattr(self.router, "capture_fn", None)
        if quant_method.is_monolithic and capture_fn is not None:
            self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=router_logits,
                topk_indices_dtype=self._quant_method.topk_indices_dtype,
                input_ids=input_ids,
            )
        return original(
            self,
            hidden_states,
            router_logits,
            shared_experts_input,
            input_ids,
        )

    setattr(_apply_quant_method_with_capture, _PATCH_MARKER, True)
    moe_runner_cls._apply_quant_method = _apply_quant_method_with_capture
    logger.warning("VERL_R3_MONOLITHIC_CAPTURE_PATCH PASS implementation=verl_wrapper")
    return "verl_wrapper"


def patch_vllm_monolithic_moe_r3_capture() -> str:
    """Install the audited vLLM R3 capture fix before engine construction."""

    current = Version(version("vllm"))
    # _apply_quant_method is AST-identical in the two audited release tags.
    # Keep the signature and branch checks below; do not accept arbitrary newer APIs.
    if current not in (Version("0.26.0"), Version("0.27.1")):
        raise RuntimeError(
            f"the R3 monolithic-MoE compatibility patch supports only the audited vLLM 0.26.0 and 0.27.1 builds, got {current}"
        )

    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    return _patch_moe_runner_class(MoERunner)
