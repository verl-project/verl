# Copyright 2026 Tilde Research
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

"""CPU tests for nitrobrew driver-side SVD + safetensors lm_head loading."""

import json
import os

import pytest
import torch
from safetensors.torch import save_file

from verl.experimental.teacher_loop.nitrobrew_teacher import (
    _compute_svd,
    _load_lm_head_weight,
)

# ---- _compute_svd ---------------------------------------------------------


def test_compute_svd_full_rank_round_trips():
    torch.manual_seed(0)
    v, d = 16, 8
    w_t = torch.randn(v, d)
    w_up, p_down = _compute_svd(w_t, d_comp=d, dtype=torch.float32)

    assert w_up.shape == (v, d)
    assert p_down.shape == (d, d)
    recon = w_up @ p_down.T
    assert torch.allclose(recon, w_t, atol=1e-5, rtol=1e-4)


def test_compute_svd_truncates_to_d_comp():
    torch.manual_seed(1)
    v, d, d_comp = 16, 8, 3
    w_t = torch.randn(v, d)
    w_up, p_down = _compute_svd(w_t, d_comp=d_comp, dtype=torch.float32)

    assert w_up.shape == (v, d_comp)
    assert p_down.shape == (d, d_comp)
    # Truncated reconstruction error must equal sum of dropped singular values squared.
    recon = w_up @ p_down.T
    expected_err = (torch.linalg.svdvals(w_t)[d_comp:] ** 2).sum().sqrt()
    actual_err = torch.linalg.norm(w_t - recon)
    assert torch.allclose(actual_err, expected_err, atol=1e-5)


def test_compute_svd_respects_dtype():
    w_t = torch.randn(4, 4)
    w_up, p_down = _compute_svd(w_t, d_comp=2, dtype=torch.bfloat16)
    assert w_up.dtype == torch.bfloat16
    assert p_down.dtype == torch.bfloat16


# ---- _load_lm_head_weight -------------------------------------------------


def _write_minimal_hf_repo(
    tmp_path,
    *,
    tied: bool,
    sharded: bool,
) -> tuple[torch.Tensor, str]:
    """Create a minimal HF-style repo on disk; return (lm_head, repo_dir)."""
    repo = tmp_path / "model"
    repo.mkdir()
    config = {
        "model_type": "qwen2",
        "tie_word_embeddings": tied,
        "hidden_size": 8,
        "vocab_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "intermediate_size": 16,
    }
    (repo / "config.json").write_text(json.dumps(config))

    torch.manual_seed(42)
    lm_head = torch.randn(16, 8)
    embed = torch.randn(16, 8)

    if sharded:
        # Put lm_head in shard A, embed in shard B; indexed via safetensors.index.json.
        save_file({"lm_head.weight": lm_head}, str(repo / "model-00001.safetensors"))
        save_file(
            {"model.embed_tokens.weight": embed},
            str(repo / "model-00002.safetensors"),
        )
        index = {
            "metadata": {"total_size": 0},
            "weight_map": {
                "lm_head.weight": "model-00001.safetensors",
                "model.embed_tokens.weight": "model-00002.safetensors",
            },
        }
        (repo / "model.safetensors.index.json").write_text(json.dumps(index))
    else:
        # Single safetensors file with both tensors.
        save_file(
            {"lm_head.weight": lm_head, "model.embed_tokens.weight": embed},
            str(repo / "model.safetensors"),
        )

    return (embed if tied else lm_head), str(repo)


def test_load_lm_head_weight_single_file_untied(tmp_path):
    expected, repo = _write_minimal_hf_repo(tmp_path, tied=False, sharded=False)
    got = _load_lm_head_weight(repo)
    assert got.shape == expected.shape
    assert got.dtype == torch.float32
    assert torch.allclose(got, expected.float())


def test_load_lm_head_weight_single_file_tied(tmp_path):
    expected, repo = _write_minimal_hf_repo(tmp_path, tied=True, sharded=False)
    got = _load_lm_head_weight(repo)
    assert torch.allclose(got, expected.float())


def test_load_lm_head_weight_sharded_untied(tmp_path):
    expected, repo = _write_minimal_hf_repo(tmp_path, tied=False, sharded=True)
    got = _load_lm_head_weight(repo)
    assert torch.allclose(got, expected.float())


def test_load_lm_head_weight_missing_raises(tmp_path):
    repo = tmp_path / "broken"
    repo.mkdir()
    (repo / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen2",
                "tie_word_embeddings": False,
                "hidden_size": 8,
                "vocab_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "intermediate_size": 16,
            }
        )
    )
    save_file({"unrelated.weight": torch.zeros(4, 4)}, str(repo / "model.safetensors"))
    with pytest.raises(ValueError, match="lm_head"):
        _load_lm_head_weight(str(repo))


# ---- end-to-end: load + SVD -----------------------------------------------


def test_lm_head_to_svd_round_trip(tmp_path):
    expected, repo = _write_minimal_hf_repo(tmp_path, tied=False, sharded=False)
    w_t = _load_lm_head_weight(repo)
    w_up, p_down = _compute_svd(w_t, d_comp=4, dtype=torch.float32)

    assert w_up.shape == (expected.shape[0], 4)
    assert p_down.shape == (expected.shape[1], 4)
    # Truncated reconstruction must beat the trivial zero reconstruction.
    err_trunc = torch.linalg.norm(w_t - w_up @ p_down.T)
    err_zero = torch.linalg.norm(w_t)
    assert err_trunc < err_zero


# Skip noisy huggingface_hub network smoke if no token / offline.
@pytest.mark.skipif(
    not os.environ.get("HF_HUB_OFFLINE_TEST_OK"),
    reason="requires opt-in env to hit huggingface_hub",
)
def test_load_from_hub_smoke():
    # Tiny config-only repo; this is a smoke test for the snapshot_download path.
    _ = _load_lm_head_weight("hf-internal-testing/tiny-random-Qwen2ForCausalLM")
