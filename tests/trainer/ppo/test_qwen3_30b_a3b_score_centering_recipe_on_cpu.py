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

"""CPU validation for the Qwen3-30B-A3B 32K Score Centering recipe script and EOSNaiveRewardManager."""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.config import validate_config

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPE_SCRIPT = REPO_ROOT / "examples" / "rollout_correction" / "run_qwen3_30b_a3b_score_centering.sh"
REWARD_MANAGER_REL = "examples/rollout_correction/eos_reward_manager.py"


def _capture_script_overrides(env_overrides: dict[str, str], tmp_path: Path):
    """Run the recipe shell script against a stub python3 to capture emitted Hydra overrides."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    out_file = tmp_path / "args.txt"
    fake_python = bin_dir / "python3"
    fake_python.write_text(
        "#!/usr/bin/env bash\n" + 'printf "%s\\n" "$@" > "' + str(out_file) + '"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    env = dict(os.environ)
    env.update(
        {
            "VERL_USE_UV": "0",
            "PATH": f"{bin_dir}:{env.get('PATH', '')}",
            **env_overrides,
        }
    )
    proc = subprocess.run(["bash", str(RECIPE_SCRIPT)], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    if proc.returncode == 0:
        lines = out_file.read_text(encoding="utf-8").splitlines()
        assert lines[:2] == ["-m", "verl.trainer.main_ppo"]
        proc.overrides = lines[2:]
    return proc


@pytest.mark.parametrize(
    "sc_mode, expected_is, expected_threshold",
    [
        ("sc", None, 2.0),
        ("sc_tis", "token", 2.0),
        ("sc_mis", "token", "0.5_5.0"),
    ],
)
def test_qwen3_30b_a3b_score_centering_default_backend_tilelang_and_fp8_rollout(
    sc_mode, expected_is, expected_threshold, tmp_path
):
    """Verify default Megatron + TileLang fused head + FP8_BLOCK rollout composes and validates across SC modes."""
    proc = _capture_script_overrides({"SC_MODE": sc_mode}, tmp_path)
    assert proc.returncode == 0, proc.stderr

    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=proc.overrides)

    validate_config(cfg, use_reference_policy=False, use_critic=False)
    assert cfg.actor_rollout_ref.actor.strategy == "megatron"
    assert cfg.actor_rollout_ref.model.use_fused_kernels is True
    assert cfg.actor_rollout_ref.actor.use_fused_kernels is True
    assert cfg.actor_rollout_ref.model.fused_kernel_options.impl_backend == "tilelang"
    assert cfg.actor_rollout_ref.model.use_remove_padding is True
    assert cfg.actor_rollout_ref.actor.loss_agg_mode == "seq-mean-token-sum-norm"
    assert cfg.actor_rollout_ref.actor.loss_scale_factor == 16384
    assert cfg.actor_rollout_ref.rollout.topk_log_probs == 128
    assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size == 4
    assert cfg.actor_rollout_ref.rollout.expert_parallel_size == 4
    assert cfg.actor_rollout_ref.rollout.quantization.linear is None
    assert cfg.actor_rollout_ref.rollout.quantization.moe == "FP8_BLOCK"
    assert cfg.actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype == "fp8"
    assert cfg.reward.preset == "math_dapo"
    assert cfg.reward.reward_manager.source == "importlib"
    assert cfg.reward.reward_manager.name == "EOSNaiveRewardManager"
    assert cfg.reward.reward_manager.module.path == REWARD_MANAGER_REL
    assert cfg.reward.reward_kwargs.missing_eos_reward == -1.0
    assert cfg.data.val_files.endswith("aime2024-2026.parquet")
    assert cfg.data.max_prompt_length == 1024
    assert cfg.data.max_response_length == 31743
    assert cfg.actor_rollout_ref.rollout.max_model_len == 32768
    assert cfg.trainer.use_v1 is True
    assert cfg.trainer.v1.trainer_mode == "colocate_async"
    assert cfg.algorithm.adv_estimator == "rloo"

    for rc in (cfg.algorithm.rollout_correction, cfg.actor_rollout_ref.actor.policy_loss.rollout_correction):
        assert rc.score_centering is True
        assert rc.bypass_mode is True
        assert rc.loss_type == "reinforce"
        assert rc.rollout_is == expected_is
        assert rc.rollout_is_threshold == expected_threshold


@pytest.mark.parametrize(
    "train_backend, expected_strategy, expected_max_tokens, expected_quant_moe",
    [
        ("veomni", "veomni", 32768, "FP8_BLOCK"),
        ("fsdp2", "fsdp2", 16384, None),
    ],
)
def test_qwen3_30b_a3b_score_centering_veomni_and_fsdp2_backend_toggles(
    train_backend, expected_strategy, expected_max_tokens, expected_quant_moe, tmp_path
):
    """Verify TRAIN_BACKEND=veomni and TRAIN_BACKEND=fsdp2 compose and validate with TileLang fused kernels."""
    proc = _capture_script_overrides({"SC_MODE": "sc_tis", "TRAIN_BACKEND": train_backend}, tmp_path)
    assert proc.returncode == 0, proc.stderr

    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=proc.overrides)

    validate_config(cfg, use_reference_policy=False, use_critic=False)
    assert cfg.actor_rollout_ref.actor.strategy == expected_strategy
    assert cfg.actor_rollout_ref.model.use_fused_kernels is True
    assert cfg.actor_rollout_ref.model.fused_kernel_options.impl_backend == "tilelang"
    assert cfg.actor_rollout_ref.actor.ppo_max_token_len_per_gpu == expected_max_tokens
    assert cfg.actor_rollout_ref.rollout.quantization.moe == expected_quant_moe
    if train_backend == "veomni":
        assert cfg.actor_rollout_ref.actor.veomni.ulysses_parallel_size == 4
        assert cfg.actor_rollout_ref.actor.veomni.expert_parallel_size == 8
    else:
        assert cfg.actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size == 8


def _make_sample(response_tokens: list[int], *, data_source: str = "math_dapo", extra_info: dict | None = None):
    """Construct a single-sample DataProto matching AgentLoopWorker._compute_score layout."""
    r_len = len(response_tokens)
    return DataProto.from_dict(
        tensors={
            "prompts": torch.tensor([[1, 2]], dtype=torch.long),
            "responses": torch.tensor([response_tokens], dtype=torch.long)
            if r_len
            else torch.zeros((1, 0), dtype=torch.long),
            "attention_mask": torch.ones((1, 2 + r_len), dtype=torch.long),
        },
        non_tensors={
            "data_source": np.array([data_source], dtype=object),
            "reward_model": np.array([{"ground_truth": "42"}], dtype=object),
            "extra_info": np.array([extra_info], dtype=object),
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_eos_env, expected_unfinished_score",
    [
        ("-1.0", -1.0),
        ("0.0", 0.0),
        ("null", 1.0),
    ],
)
async def test_eos_naive_reward_manager_via_importlib(
    missing_eos_env, expected_unfinished_score, tmp_path, monkeypatch
):
    """Verify importlib EOSNaiveRewardManager + reward.preset=math_dapo across dapo_filter and AIME."""
    model_dir = tmp_path / "qwen3_stub"
    model_dir.mkdir()
    # Primary tokenizer EOS is 151643 (<|endoftext|>); generation_config.json adds 151645 (<|im_end|>).
    (model_dir / "generation_config.json").write_text(json.dumps({"eos_token_id": [151645, 151643]}), encoding="utf-8")

    proc = _capture_script_overrides(
        {"MODEL_PATH": str(model_dir), "MISSING_EOS_REWARD": missing_eos_env},
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr

    with initialize_config_module(config_module="verl.trainer.config", version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=proc.overrides)

    monkeypatch.chdir(REPO_ROOT)
    decode_calls = 0

    def fake_decode(token_ids, skip_special_tokens=True):
        nonlocal decode_calls
        decode_calls += 1
        return r"Preliminary reasoning \boxed{42}"

    tokenizer = SimpleNamespace(eos_token_id=151643, name_or_path=str(model_dir), decode=fake_decode)
    manager = load_reward_manager(cfg, tokenizer)

    # 1. Primary EOS (151643) + unmapped "dapo_filter" data_source -> routed via reward.preset=math_dapo (+1.0).
    shared_extra = {"split": "train"}
    out_primary = await manager.run_single(
        _make_sample([10, 20, 151643], data_source="dapo_filter", extra_info=shared_extra)
    )
    assert out_primary["reward_score"] == 1.0
    assert out_primary["reward_extra_info"]["acc"] is True
    assert shared_extra == {"split": "train"}

    # 2. Secondary generation_config EOS (151645) + AIME session data_source -> scored normally (+1.0).
    aime_sample = _make_sample([10, 20, 151645], data_source="aime2026-II")
    out_secondary = await manager.run_single(aime_sample)
    assert out_secondary["reward_score"] == 1.0
    assert out_secondary["reward_extra_info"]["acc"] is True

    # 3. Unfinished response (no trailing EOS despite early \boxed{42}) -> assigned missing_eos_reward.
    before_calls = decode_calls
    out_unfinished = await manager.run_single(_make_sample([10, 20, 999], data_source="dapo_filter"))
    assert out_unfinished["reward_score"] == expected_unfinished_score
    assert out_unfinished["reward_extra_info"]["acc"] is (missing_eos_env == "null")
    if missing_eos_env != "null":
        assert out_unfinished["reward_extra_info"]["pred"] == "[INVALID]"
        # Truncated responses short-circuit in O(1) without running tokenizer.decode.
        assert decode_calls == before_calls

    # 4. Empty response -> handled cleanly without IndexError or prompt-mask miscount.
    if missing_eos_env != "null":
        out_empty = await manager.run_single(_make_sample([]))
        assert out_empty["reward_score"] == expected_unfinished_score


def test_eos_naive_reward_manager_rejects_incompatible_config(monkeypatch):
    """Verify EOSNaiveRewardManager fails fast when ignore_eos=True or no EOS IDs exist."""
    monkeypatch.chdir(REPO_ROOT)
    base_cfg = {
        "actor_rollout_ref": {"model": {"path": None}, "rollout": {"ignore_eos": True}},
        "reward": {
            "preset": "math_dapo",
            "reward_manager": {
                "source": "importlib",
                "name": "EOSNaiveRewardManager",
                "module": {"path": REWARD_MANAGER_REL},
            },
            "reward_kwargs": {"missing_eos_reward": -1.0},
            "custom_reward_function": {"path": None, "name": "compute_score"},
        },
    }
    tok = SimpleNamespace(eos_token_id=151643, name_or_path=None, decode=lambda ids, skip_special_tokens=True: "")
    with pytest.raises(ValueError, match="ignore_eos=False"):
        load_reward_manager(OmegaConf.create(base_cfg), tok)

    base_cfg["actor_rollout_ref"]["rollout"]["ignore_eos"] = False
    tok_no_eos = SimpleNamespace(eos_token_id=None, name_or_path=None, decode=lambda ids, skip_special_tokens=True: "")
    with pytest.raises(ValueError, match="eos_token_id"):
        load_reward_manager(OmegaConf.create(base_cfg), tok_no_eos)


def test_qwen3_30b_a3b_score_centering_rejects_unknown_mode_or_backend(tmp_path):
    """Verify invalid SC_MODE or TRAIN_BACKEND values fail fast with clear error messages."""
    proc_mode = _capture_script_overrides({"SC_MODE": "invalid_mode"}, tmp_path)
    assert proc_mode.returncode != 0
    assert "Unsupported SC_MODE" in proc_mode.stderr

    proc_backend = _capture_script_overrides({"TRAIN_BACKEND": "invalid_backend"}, tmp_path)
    assert proc_backend.returncode != 0
    assert "Unsupported TRAIN_BACKEND" in proc_backend.stderr
