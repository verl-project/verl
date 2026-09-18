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

"""Opt-in, instance-local exclusion matching for verl-generated ModelOpt MXFP8 configs.

vLLM 0.24's ModelOpt matcher retains a legacy substring fallback. A generated
router exclusion ``mlp.gate`` consequently excludes the SwiGLU ``mlp.gate_up_proj``
too. This helper removes that fallback only on an explicitly opted-in config
instance; checkpoint configs and other quantization methods retain their behavior.

This is standalone integration infrastructure, not CUDA MXFP8 rollout support.
Call it in each model-building process before constructing layers. See
``docs/low_precision/mxfp8_exclusion_patch.md`` for the integration contract.
"""

from collections.abc import Mapping
from fnmatch import fnmatchcase
from types import MethodType
from typing import Any

# Keep provenance in the HF config passed between processes, not a global env var.
VERL_EXACT_MXFP8_EXCLUSIONS = "_verl_exact_mxfp8_exclusions"


def _is_layer_excluded(self, prefix: str) -> bool:
    from vllm.model_executor.layers.quantization.utils.quant_utils import is_layer_skipped

    excluded = self.exclude_modules
    mapping = self.packed_modules_mapping
    if not excluded:
        return False

    # Resolve glob patterns against both the engine's fused name and its HF
    # shards. Feed literal matches into vLLM's helper so its all-or-none fusion
    # check remains authoritative, including mixed exact/glob exclusions.
    parent, separator, leaf = prefix.rpartition(".")
    candidates = [prefix] + [parent + separator + shard for shard in mapping.get(leaf, ())]
    matched = [name for name in candidates if any(fnmatchcase(name, pattern) for pattern in excluded)]
    # Retain original entries as well: vLLM also understands per-expert names
    # when deciding whether a fused expert group is excluded.
    return is_layer_skipped(prefix, [*excluded, *matched], mapping)


def apply_mxfp8_exclusion_patch(quant_config: Any, *, hf_quant_config: Mapping[str, Any]) -> bool:
    """Opt a single, explicitly marked ModelOpt MXFP8 config into strict matching.

    The caller must mark ONLY configs it generates for online MXFP8 rollout with
    ``VERL_EXACT_MXFP8_EXCLUSIONS: True`` and pass that same raw HF quantization
    config here, alongside its parsed vLLM config. Do not mark user/checkpoint
    configs or infer provenance from the quantization class alone.

    Install after configuration parsing/name mapping but BEFORE model layers are
    built, separately in every model-building worker (also after deserialization).
    ``get_quant_method`` and the global ModelOpt classes are left untouched.

    Returns True when installed, False for an unmarked or already-patched config.
    Invalid explicit opt-ins raise rather than silently leaving the bug active.
    """
    if hf_quant_config.get(VERL_EXACT_MXFP8_EXCLUSIONS) is not True:
        return False
    if hf_quant_config.get("quant_method") != "mxfp8":
        raise ValueError("Exact MXFP8 exclusions require a verl-generated quant_method='mxfp8' config")

    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config

    if not isinstance(quant_config, ModelOptMxFp8Config):
        raise TypeError("Exact MXFP8 exclusions require a parsed ModelOptMxFp8Config")
    if getattr(getattr(quant_config, "is_layer_excluded", None), "__func__", None) is _is_layer_excluded:
        return False
    if "is_layer_excluded" in vars(quant_config):
        raise RuntimeError("Refusing to replace an existing instance-specific exclusion matcher")
    if not callable(getattr(quant_config, "is_layer_excluded", None)):
        raise TypeError("Unsupported ModelOpt MXFP8 config: is_layer_excluded is unavailable")
    if not isinstance(getattr(quant_config, "packed_modules_mapping", None), Mapping):
        raise TypeError("Unsupported ModelOpt MXFP8 config: packed_modules_mapping is unavailable")
    excluded = getattr(quant_config, "exclude_modules", None)
    if not isinstance(excluded, list) or not all(isinstance(name, str) for name in excluded):
        raise TypeError("Unsupported ModelOpt MXFP8 config: exclude_modules must be a list of strings")

    quant_config.is_layer_excluded = MethodType(_is_layer_excluded, quant_config)
    return True
