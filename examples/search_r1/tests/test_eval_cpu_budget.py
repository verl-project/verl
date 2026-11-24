"""Run the eval entrypoint with a CPU-only Ray stub to check resource wiring."""

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf


@pytest.mark.parametrize("fail", [False, True])
def test_eval_honors_cpu_budget_and_shuts_down(tmp_path, monkeypatch, fail):
    path = Path(__file__).resolve().parents[1] / "evaluate.py"
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    function.decorator_list = []
    calls = []
    ray = SimpleNamespace(init=lambda **kw: calls.append(("init", kw)), shutdown=lambda: calls.append(("shutdown", {})))

    async def evaluate(config):
        assert config.ray_kwargs.ray_init.num_cpus == 16
        if fail:
            raise RuntimeError("evaluation failed")

    import os

    monkeypatch.setenv("RUN_DIR", str(tmp_path))
    namespace = {
        "OmegaConf": OmegaConf,
        "Path": Path,
        "json": json,
        "os": os,
        "ray": ray,
        "get_ppo_ray_runtime_env": lambda cfg: {},
        "asyncio": asyncio,
        "evaluate": evaluate,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    config = OmegaConf.create({"ray_kwargs": {"ray_init": {"num_cpus": 16}}})
    if fail:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            namespace["main"](config)
    else:
        namespace["main"](config)
    assert calls == [("init", {"num_cpus": 16, "runtime_env": {}}), ("shutdown", {})]
    assert json.loads((tmp_path / "resolved-eval-config.json").read_text())["ray_kwargs"]["ray_init"]["num_cpus"] == 16
