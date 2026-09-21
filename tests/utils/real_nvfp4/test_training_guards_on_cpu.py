# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Configuration rejection tests, not TE/vLLM numerical-equivalence tests.

The two engine methods are extracted without importing Megatron providers. This
tests their actual guard wiring without constructing a model or initializing
distributed workers; real execution remains covered by scheduled GPU jobs.
"""

import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from verl.utils.real_nvfp4.config import (
    real_nvfp4_expected_counts,
    validate_real_nvfp4_model_contract,
    validate_real_nvfp4_parallelism,
    validate_real_nvfp4_precision_configs,
)

ROOT = Path(__file__).resolve().parents[3]
ENGINE_PATH = ROOT / "verl/workers/engine/megatron/transformer_impl.py"
RECIPE_PATHS = [
    ROOT / "examples/real_nvfp4/config/attn_bf16_mlp_nvfp4.yaml",
    ROOT / "examples/real_nvfp4/config/attn_bf16_mlp_nvfp4_first2_last4.yaml",
]


def _engine_method(name):
    tree = ast.parse(ENGINE_PATH.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronEngine")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"os": os, "OmegaConf": OmegaConf}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ENGINE_PATH), "exec"), namespace)
    return namespace[name]


def _model_config(**kwargs):
    values = {"num_hidden_layers": 48, "num_experts": 128, "decoder_sparse_step": 1, "mlp_only_layers": []}
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_current_all_moe_model_still_has_the_same_export_counts():
    model = _model_config()
    validate_real_nvfp4_model_contract(model)
    assert real_nvfp4_expected_counts(model) == (48 * 128 * 3, 48 * 128 * 2)


@pytest.mark.parametrize("overrides", [{"decoder_sparse_step": 2}, {"mlp_only_layers": [0, 1]}])
def test_mixed_dense_moe_fails_before_unattested_dense_quantization(overrides):
    with pytest.raises(ValueError, match="mixed dense/MoE"):
        validate_real_nvfp4_model_contract(_model_config(**overrides))


@pytest.mark.parametrize("virtual_pipeline_size", [None, 1])
def test_unsplit_pipeline_is_supported(virtual_pipeline_size):
    validate_real_nvfp4_parallelism(pipeline_size=1, virtual_pipeline_size=virtual_pipeline_size)


def test_current_engine_carveout_is_unchanged():
    engine = SimpleNamespace(
        engine_config=SimpleNamespace(pipeline_model_parallel_size=1, virtual_pipeline_model_parallel_size=None),
        _real_nvfp4_config=SimpleNamespace(num_layers_at_start_in_bf16=2, num_layers_at_end_in_bf16=4),
        model_config=SimpleNamespace(hf_config=_model_config()),
    )
    overrides = {
        "first_last_layers_bf16": True,
        "num_layers_at_start_in_bf16": 2,
        "num_layers_at_end_in_bf16": 4,
    }
    assert _engine_method("_resolve_real_nvfp4_bf16_layers")(engine, overrides) == (True, 2, 4)


@pytest.mark.parametrize("pp,vpp", [(2, None), (1, 2), (2, 2)])
@pytest.mark.parametrize("carve_start,carve_end", [(0, 0), (2, 4)])
def test_engine_rejects_pipeline_splitting_with_and_without_carveout(pp, vpp, carve_start, carve_end):
    engine = SimpleNamespace(
        engine_config=SimpleNamespace(pipeline_model_parallel_size=pp, virtual_pipeline_model_parallel_size=vpp),
        _real_nvfp4_config=SimpleNamespace(
            num_layers_at_start_in_bf16=carve_start, num_layers_at_end_in_bf16=carve_end
        ),
        model_config=SimpleNamespace(hf_config=_model_config()),
    )
    overrides = {
        "first_last_layers_bf16": carve_start + carve_end > 0,
        "num_layers_at_start_in_bf16": carve_start,
        "num_layers_at_end_in_bf16": carve_end,
    }
    with pytest.raises(ValueError, match="pipeline_model_parallel_size=1"):
        _engine_method("_resolve_real_nvfp4_bf16_layers")(engine, overrides)


@pytest.mark.parametrize("recipe_path", RECIPE_PATHS)
@pytest.mark.parametrize("explicit_eval", [False, True])
def test_formal_recipe_and_identical_explicit_eval_are_accepted(recipe_path, explicit_eval):
    configs = OmegaConf.to_container(OmegaConf.load(recipe_path), resolve=True)["configs"]
    if explicit_eval:
        for payload in configs.values():
            payload["evaluation_recipe"] = copy.deepcopy(payload["training_recipe"])
    validate_real_nvfp4_precision_configs(configs)


@pytest.mark.parametrize(
    "config_name,evaluation_recipe",
    [
        ("nvfp4", {}),
        ("nvfp4", None),
        ("nvfp4", {"fp8_quantization_recipe": "tensorwise"}),
        ("bf16", {"fp4_quantization_recipe": "nvfp4"}),
        ("bf16", None),
    ],
)
def test_effective_eval_precision_cannot_differ(config_name, evaluation_recipe):
    configs = OmegaConf.to_container(OmegaConf.load(RECIPE_PATHS[0]), resolve=True)["configs"]
    configs[config_name]["evaluation_recipe"] = evaluation_recipe
    with pytest.raises(ValueError, match="evaluation_recipe must match training_recipe"):
        validate_real_nvfp4_precision_configs(configs)


def test_engine_loader_rejects_eval_drift_before_loading_megatron_recipe(monkeypatch):
    raw = OmegaConf.to_container(OmegaConf.load(RECIPE_PATHS[0]), resolve=True)
    raw["configs"]["nvfp4"]["evaluation_recipe"] = {}
    monkeypatch.setattr(OmegaConf, "load", lambda _path: OmegaConf.create(raw))
    engine = SimpleNamespace(
        _real_nvfp4_config=SimpleNamespace(te_precision_config_file=str(RECIPE_PATHS[0])),
        _real_nvfp4_bf16_layers=(False, 0, 0),
        model_config=SimpleNamespace(hf_config=_model_config()),
    )
    with pytest.raises(ValueError, match="evaluation_recipe must match training_recipe"):
        _engine_method("_load_real_nvfp4_precision_recipe")(engine)
