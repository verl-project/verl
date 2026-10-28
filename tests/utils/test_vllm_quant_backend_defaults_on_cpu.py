# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""``set_default_vllm_quantization_backends`` on Hopper and Blackwell (verl-project/verl#8381).

The module is loaded by path with its vLLM-facing siblings stubbed, so these run without vLLM.
"""

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Native DeepSeek-V4 checkpoint: FP8 block config at the top level, experts declared on the model config.
_DSV4_QUANT = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128], "activation_scheme": "dynamic"}
_DSV4_HF = SimpleNamespace(model_type="deepseek_v4", expert_dtype="fp4")
# DeepSeek-V4.1 nests the expert format inside quantization_config; the -Base checkpoints have FP8 experts.
_DSV41_QUANT = {"quant_method": "fp8", "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"}
_DSV4_BASE_HF = SimpleNamespace(model_type="deepseek_v4", expert_dtype="fp8")
# compressed-tensors MXFP4 experts name the scheme inside the quantization config.
_CT_MXFP4_QUANT = {
    "quant_method": "compressed-tensors",
    "format": "mxfp4-pack-quantized",
    "config_groups": {"group_0": {"weights": {"type": "float", "num_bits": 4, "strategy": "group"}}},
}
_FP8_QUANT = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128], "scale_fmt": "ue8m0"}
_NVFP4_QUANT = {"quant_method": "modelopt", "quant_algo": "NVFP4"}


def _stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


@pytest.fixture(scope="module")
def quant_utils():
    noop = lambda *args, **kwargs: None  # noqa: E731
    stubs = {
        "verl.utils.vllm.vllm_fp4_utils": _stub(
            "verl.utils.vllm.vllm_fp4_utils",
            install_mxfp4_mxfp8_compressed_tensors_patch=noop,
            install_nvfp4_cutlass_compressed_tensors_patch=noop,
            is_deepseek_v4_model=lambda model: False,
            process_mxfp4_moe_weights_after_loading=noop,
            process_nvfp4_moe_weights_after_loading=noop,
            stage_mxfp4_moe_params_for_loading=noop,
            stage_nvfp4_moe_params_for_loading=noop,
        ),
        "verl.utils.vllm.vllm_fp8_utils": _stub(
            "verl.utils.vllm.vllm_fp8_utils",
            build_fp8_method_patchers=noop,
            process_fp8_weights_after_loading=noop,
            stage_fp8_params_for_loading=noop,
        ),
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            "vllm_quant_utils_under_test", _REPO_ROOT / "verl/utils/vllm/vllm_quant_utils.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


@pytest.fixture
def defaults(quant_utils, monkeypatch):
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM_E8M0", raising=False)
    monkeypatch.delenv("VERL_ROLLOUT_USE_DEEP_GEMM", raising=False)

    def run(quantization_config, hf_config=None, hopper=False, **engine_kwargs):
        monkeypatch.setattr(quant_utils, "_is_hopper", lambda: hopper)
        quant_utils.set_default_vllm_quantization_backends(quantization_config, engine_kwargs, hf_config=hf_config)
        return engine_kwargs

    return run


@pytest.mark.parametrize(
    "quantization_config, hf_config", [(_DSV4_QUANT, _DSV4_HF), (_DSV41_QUANT, None), (_CT_MXFP4_QUANT, None)]
)
def test_mxfp4_experts_keep_vllm_moe_selection_on_hopper(defaults, quantization_config, hf_config, caplog):
    with caplog.at_level("INFO"):
        kwargs = defaults(quantization_config, hf_config, hopper=True)
    assert kwargs == {"linear_backend": "deep_gemm"}  # no moe_backend: vLLM picks Marlin on SM90
    assert any("left to vLLM" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("quantization_config, hf_config", [(_DSV4_QUANT, _DSV4_HF), (_CT_MXFP4_QUANT, None)])
def test_mxfp4_experts_default_to_deep_gemm_on_blackwell(defaults, quantization_config, hf_config):
    kwargs = defaults(quantization_config, hf_config, hopper=False)
    assert kwargs == {"linear_backend": "deep_gemm", "moe_backend": "deep_gemm"}


@pytest.mark.parametrize("hopper", [True, False])
@pytest.mark.parametrize("hf_config", [SimpleNamespace(model_type="deepseek_v3"), _DSV4_BASE_HF])
def test_fp8_moe_still_defaults_to_deep_gemm(defaults, hopper, hf_config):
    kwargs = defaults(_FP8_QUANT, hf_config, hopper=hopper)
    assert kwargs == {"linear_backend": "deep_gemm", "moe_backend": "deep_gemm"}


@pytest.mark.parametrize("hopper", [True, False])
def test_explicit_backends_are_kept(defaults, hopper):
    kwargs = defaults(_DSV4_QUANT, _DSV4_HF, hopper=hopper, moe_backend="marlin", linear_backend="auto")
    assert kwargs == {"linear_backend": "auto", "moe_backend": "marlin"}


def test_nvfp4_and_unquantized_are_untouched(defaults):
    assert defaults(_NVFP4_QUANT, None, hopper=False) == {}
    assert defaults(None, _DSV4_HF, hopper=True) == {}


def test_opt_out_env_leaves_everything_to_vllm(defaults, monkeypatch):
    import os

    monkeypatch.setenv("VERL_ROLLOUT_USE_DEEP_GEMM", "0")
    assert defaults(_FP8_QUANT, None, hopper=False) == {}
    assert os.environ.get("VLLM_USE_DEEP_GEMM_E8M0") is None
    assert defaults(_DSV4_QUANT, _DSV4_HF, hopper=True, moe_backend="marlin") == {"moe_backend": "marlin"}
    # vLLM may still pick DeepGEMM for the FP8 linear layers, so Hopper FP32 block scales keep the E8M0 override.
    assert os.environ.get("VLLM_USE_DEEP_GEMM_E8M0") == "0"


def test_rollout_quantization_override_ignores_checkpoint_expert_dtype(quant_utils, defaults):
    # A rollout quantization override requantizes the experts too, so launch_server must not forward the
    # checkpoint's expert_dtype; the effective FP8 block config then keeps DeepGEMM for the MoE on Hopper.
    pick = quant_utils.checkpoint_hf_config_for_backend_defaults
    assert pick({"quantization_config": _FP8_QUANT}, _DSV4_HF) is None
    assert pick({}, _DSV4_HF) is _DSV4_HF
    assert pick(None, _DSV4_HF) is _DSV4_HF
    assert defaults(_FP8_QUANT, pick({"quantization_config": _FP8_QUANT}, _DSV4_HF), hopper=True) == {
        "linear_backend": "deep_gemm",
        "moe_backend": "deep_gemm",
    }


def test_explicit_auto_linear_backend_keeps_hopper_e8m0_override(defaults):
    import os

    kwargs = defaults(_DSV4_QUANT, _DSV4_HF, hopper=True, linear_backend="auto")
    assert kwargs == {"linear_backend": "auto"}  # vLLM may still pick DeepGEMM for the FP8 linear layers
    assert os.environ.get("VLLM_USE_DEEP_GEMM_E8M0") == "0"


def test_explicit_non_deepgemm_backends_skip_the_e8m0_override(defaults):
    import os

    defaults(_DSV4_QUANT, _DSV4_HF, hopper=True, linear_backend="flashinfer", moe_backend="marlin")
    assert os.environ.get("VLLM_USE_DEEP_GEMM_E8M0") is None


def test_hopper_fp32_block_scales_disable_e8m0(defaults, monkeypatch):
    import os

    defaults(_DSV4_QUANT, _DSV4_HF, hopper=True)
    assert os.environ.get("VLLM_USE_DEEP_GEMM_E8M0") == "0"  # linear_backend=deep_gemm still triggers it
