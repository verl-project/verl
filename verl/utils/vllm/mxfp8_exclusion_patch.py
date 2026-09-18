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

"""Opt-in exclusion matching for verl-generated ModelOpt MXFP8 configs.

vLLM 0.24's ModelOpt matcher retains a legacy substring fallback. A generated
router exclusion ``mlp.gate`` consequently excludes the SwiGLU ``mlp.gate_up_proj``
too. This helper removes that fallback only on an explicitly opted-in config
instance; checkpoint configs and other quantization methods retain their behavior.

The patchers are registered by ``build_fp8_method_patchers``. The MXFP8 feature
branch must still mark its generated configs explicitly. See
``docs/low_precision/mxfp8_exclusion_patch.md`` for the integration contract.
"""

from collections.abc import Mapping
from fnmatch import fnmatchcase
from functools import wraps
from typing import Any
from unittest.mock import patch

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


def _mark_generated_config(quant_config: Any, hf_quant_config: Mapping[str, Any]) -> bool:
    """Validate and carry the raw HF opt-in onto its parsed config instance.

    The caller must mark ONLY configs it generates for online MXFP8 rollout with
    ``VERL_EXACT_MXFP8_EXCLUSIONS: True`` and pass that same raw HF quantization
    config here, alongside its parsed vLLM config. Do not mark user/checkpoint
    configs or infer provenance from the quantization class alone.

    Only plain metadata is attached; no bound methods cross process boundaries.
    Returns True when marked, False for an unmarked or already-marked config.
    Invalid explicit opt-ins raise rather than silently leaving the bug active.
    """
    if hf_quant_config.get(VERL_EXACT_MXFP8_EXCLUSIONS) is not True:
        return False
    if hf_quant_config.get("quant_method") != "mxfp8":
        raise ValueError("Exact MXFP8 exclusions require a verl-generated quant_method='mxfp8' config")

    from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config

    if not isinstance(quant_config, ModelOptMxFp8Config):
        raise TypeError("Exact MXFP8 exclusions require a parsed ModelOptMxFp8Config")
    if getattr(quant_config, VERL_EXACT_MXFP8_EXCLUSIONS, False) is True:
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

    setattr(quant_config, VERL_EXACT_MXFP8_EXCLUSIONS, True)
    return True


def build_mxfp8_exclusion_patchers():
    """Return unstarted, opt-in MXFP8 parser/matcher patches for the common registry.

    Install before parsing HF configs in the driver, and before model construction
    in every worker. An already-parsed config carries its boolean opt-in through
    serialization; the worker only needs to install the class wrapper. Unmarked
    configs delegate to the original matcher, including legacy substring rules.
    """
    try:
        from vllm.model_executor.layers.quantization.modelopt import ModelOptMxFp8Config
    except ImportError:
        return []  # Older vLLM without ModelOpt MXFP8: leave its FP8 patches alone.

    original_from_config = ModelOptMxFp8Config.from_config.__func__
    original_is_layer_excluded = ModelOptMxFp8Config.is_layer_excluded

    @wraps(original_from_config)
    def from_config(cls, config):
        # Read the marker before upstream normalizes the MiniMax-style dictionary.
        marked = config.get(VERL_EXACT_MXFP8_EXCLUSIONS) is True
        method = config.get("quant_method")
        if marked and method != "mxfp8":
            raise ValueError("Exact MXFP8 exclusions require a verl-generated quant_method='mxfp8' config")
        parsed = original_from_config(cls, config)
        if marked:
            _mark_generated_config(parsed, {"quant_method": method, VERL_EXACT_MXFP8_EXCLUSIONS: True})
        return parsed

    @wraps(original_is_layer_excluded)
    def is_layer_excluded(self, prefix):
        if getattr(self, VERL_EXACT_MXFP8_EXCLUSIONS, False) is True:
            return _is_layer_excluded(self, prefix)
        return original_is_layer_excluded(self, prefix)

    # Target the MXFP8 subclass, not ModelOptQuantConfigBase: NVFP4/FP8 are unchanged.
    return [
        patch.object(ModelOptMxFp8Config, "from_config", classmethod(from_config)),
        patch.object(ModelOptMxFp8Config, "is_layer_excluded", is_layer_excluded),
    ]
