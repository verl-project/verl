# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
import os
import re
from collections.abc import Iterable
from typing import Any

from verl.utils.fp8_utils import FP8QuantizerHelper

logger = logging.getLogger(__name__)


def _get_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    get_value = getattr(config, "get", None)
    if callable(get_value):
        return get_value(key, default)
    return getattr(config, key, default)


def _normalize_ignored_layers(ignored_layers: Any) -> list[str]:
    if ignored_layers is None:
        return []
    if isinstance(ignored_layers, str):
        ignored_layers = ignored_layers.split(",")
    elif not isinstance(ignored_layers, Iterable):
        ignored_layers = [ignored_layers]

    normalized = []
    for layer in ignored_layers:
        layer_name = str(layer).strip()
        if layer_name:
            normalized.append(layer_name)
    return normalized


def _dedupe_layers(ignored_layers: Iterable[str]) -> list[str]:
    seen = set()
    deduped = []
    for layer in ignored_layers:
        layer_lower = layer.lower()
        if layer_lower in seen:
            continue
        seen.add(layer_lower)
        deduped.append(layer)
    return deduped


def _get_ignored_layers_from_env() -> list[str]:
    return _normalize_ignored_layers(os.getenv("SGLANG_FP8_IGNORED_LAYERS"))


def get_sglang_fp8_ignored_layers(quant_config: Any = None) -> list[str]:
    ignored_layers = []
    ignored_layers.extend(_normalize_ignored_layers(_get_config_value(quant_config, "ignored_layers")))
    ignored_layers.extend(_normalize_ignored_layers(_get_config_value(quant_config, "modules_to_not_convert")))
    ignored_layers.extend(_get_ignored_layers_from_env())
    return _dedupe_layers(ignored_layers)


def _matches_ignored_layer(param_name: str, ignored_layer: str) -> bool:
    ignored_layer = ignored_layer.strip()
    if not ignored_layer:
        return False

    name = param_name.strip(".")
    module_name = name[: -len(".weight")] if name.lower().endswith(".weight") else name
    if ignored_layer.startswith("re:"):
        pattern = ignored_layer[3:]
        return any(re.match(pattern, candidate) for candidate in (name, module_name))

    ignored_layer = ignored_layer.lower().strip(".")
    name = name.lower()
    module_name = module_name.lower()
    for candidate in (name, module_name):
        if candidate == ignored_layer:
            return True
        if candidate.startswith(f"{ignored_layer}."):
            return True
        if candidate.endswith(f".{ignored_layer}"):
            return True
        if f".{ignored_layer}." in f".{candidate}.":
            return True
    return False


def build_sglang_fp8_quant_config(hf_config: Any = None, ignored_layers: Any = None) -> dict[str, Any]:
    """Build SGLang block-wise FP8 config shared by server init and weight sync."""
    fp8_quant_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }

    hf_quant_config = _get_config_value(hf_config, "quantization_config")
    # Carry the checkpoint's scale dialect through. Dropping it here is what
    # made the delta engine's "no ue8m0" guard toothless for DSv4: the flag
    # never survived to be checked, and the plain-fp32 seed formula shipped.
    scale_fmt = None
    if hf_quant_config is not None:
        scale_fmt = _get_config_value(hf_quant_config, "scale_fmt")
    if scale_fmt is not None:
        fp8_quant_config["scale_fmt"] = scale_fmt
    merged_ignored_layers = get_sglang_fp8_ignored_layers(hf_quant_config)
    merged_ignored_layers.extend(_normalize_ignored_layers(ignored_layers))
    merged_ignored_layers = _dedupe_layers(merged_ignored_layers)
    if merged_ignored_layers:
        fp8_quant_config["ignored_layers"] = merged_ignored_layers

    return fp8_quant_config


class SGLangFP8QuantizerHelper(FP8QuantizerHelper):
    def __init__(self, quant_config):
        super().__init__(quant_config)
        self.ignored_layers = get_sglang_fp8_ignored_layers(quant_config)

    def should_quantize_param(self, param_name):
        for ignored_layer in self.ignored_layers:
            if _matches_ignored_layer(param_name, ignored_layer):
                return False
        return super().should_quantize_param(param_name)


class DeepseekV4FP8QuantizerHelper(SGLangFP8QuantizerHelper):
    """DSv4-native fp8 conversion for the nccl full-sync path.

    The serialized rollout ckpt is the single source of truth: a weight is
    quantized iff its ``<stem>.scale`` companion exists in the ckpt index
    (wo_a stays bf16, norms/sinks/router never appear), and scales follow the
    ckpt's ue8m0 dialect (power-of-two, ``.scale`` suffix) so sglang's
    deepseek_v4 loader consumes them exactly like a checkpoint load.
    """

    def __init__(self, quant_config, ckpt_path: str):
        super().__init__(quant_config)
        from verl.utils.fp8_ckpt_dtypes import build_ckpt_fp8_predicate

        self._is_quantized = build_ckpt_fp8_predicate(ckpt_path)
        if self._is_quantized is None:
            raise ValueError(f"cannot determine FP8 tensor dtypes from checkpoint at {ckpt_path}")
        self._ckpt_path = ckpt_path

    def should_quantize_param(self, param_name):
        return self._is_quantized(param_name)

    async def quant_weights_by_name(self, weights, dtype=None):
        import torch

        from verl.utils.fp8_sharded import (
            load_ckpt_scales,
            local_blockwise_absmax,
            quantize_shard_with_descale,
            sticky_ue8m0_descale,
        )
        from verl.workers.rollout.utils import ensure_async_iterator

        bm_bn = self.quant_config.get("weight_block_size") if isinstance(self.quant_config, dict) else None
        bm_bn = tuple(bm_bn or (128, 128))
        ckpt_scales = load_ckpt_scales(self._ckpt_path)
        n_fp8_passthrough = 0
        async for k, v in ensure_async_iterator(weights):
            if not self.should_quantize_param(k):
                yield (k, v)
                continue
            if v.element_size() == 1:
                # The stream is ALREADY quantized upstream (the bridge's
                # auto-fp8 export ships codes + scale companions). Quantizing
                # fp8 codes garbles them (their float view is not the master),
                # and the freshly emitted scale joins the original in the same
                # push -- SGLang's fused wqkv_a loader rejects the duplicate
                # shard. Pass codes through untouched; their scale
                # companions are not in the quantize manifest and pass through
                # on their own. Fidelity of a pre-quantized stream is the
                # upstream producer's business, not this converter's.
                n_fp8_passthrough += 1
                yield (k, v)
                continue
            x = v.to(torch.float32)
            amax = local_blockwise_absmax(x, bm_bn, row_offset=0, full_shape=tuple(x.shape))
            # ue8m0 dialect (power-of-two, exact in fp32), preferring the ckpt's
            # own scale wherever it still covers -- the ckpt carries per-block
            # headroom that amax alone cannot reconstruct.
            descale = sticky_ue8m0_descale(amax, ckpt_scales.get(k))
            codes = quantize_shard_with_descale(x, descale, bm_bn, row_offset=0)
            yield (k, codes)
            yield (k[: -len(".weight")] + ".scale", descale)
            del x, amax, descale, codes
            # DSv4 is large enough that the per-parameter transients accumulate
            # faster than the caching allocator releases them, and the conversion
            # OOMs partway through. Reclaim every 32 params: frequent enough to
            # bound the high-water mark, rare enough that the sync cost is noise.
            _n_q = getattr(self, "_n_q", 0) + 1
            self._n_q = _n_q
            if _n_q % 32 == 0:
                torch.cuda.empty_cache()
        if n_fp8_passthrough:
            logger.warning(
                "DSv4 named_tensors converter: %d params arrived already fp8-quantized and passed "
                "through untouched -- their scales keep the UPSTREAM dialect (not sticky/ue8m0). "
                "Fix the exporter if checkpoint-exact scales are required on this path.",
                n_fp8_passthrough,
            )


def named_tensors_quant_mode(quantization, hf_config) -> str | None:
    """Which verl-side quantizer the named_tensors full-sync path must use.

    Returns ``"dsv4"``, ``"generic"``, or None (send raw).

    The DSv4 answer deliberately does NOT depend on the rollout's
    ``quantization`` flag: that flag doubles as an init-time switch that
    rewrites ``hf_config.quantization_config``, so delta runs keep it unset --
    and with it unset the hybrid ServerAdapter would push raw bf16, which
    SGLang then requantizes with its own plain ``amax/448`` formula. An
    fp8-SERIALIZED checkpoint is itself the instruction to ship codes+scales:
    quantize verl-side with ue8m0 and checkpoint-sticky scales whenever the
    checkpoint is fp8, flag or no flag.
    """
    import os

    model_type = getattr(hf_config, "model_type", None)
    if model_type == "deepseek_v4":
        qc = getattr(hf_config, "quantization_config", None)
        if qc is not None:
            get = qc.get if isinstance(qc, dict) else lambda k, d=None: getattr(qc, k, d)
            if get("quant_method") == "fp8":
                return "dsv4"
    if quantization == "fp8":
        return "dsv4" if model_type == "deepseek_v4" else "generic"
    return None
