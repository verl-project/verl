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

"""Unit tests for VeOmni SP (Sequence Parallelism) utilities.

These tests validate:
1. _prepare_veomni_flash_attention_kwargs: various position_ids shapes (requires veomni)
2. VL_TYPE2INDEX: Qwen3.5 model support
3. compute_topk_loss: veomni strategy match
"""

import os

import pytest
import torch

veomni_available = True
try:
    from verl.workers.engine.veomni.transformer_impl import (
        _prepare_veomni_flash_attention_kwargs,
    )
except ImportError:
    veomni_available = False


@pytest.mark.skipif(not veomni_available, reason="veomni or ray not installed")
class TestPrepareVeomniFlashAttentionKwargs:
    """Test _prepare_veomni_flash_attention_kwargs with various position_ids layouts."""

    def test_1d_flat_packed(self):
        """1D: (total_nnz,) - flat packed format."""
        position_ids = torch.tensor([0, 1, 2, 0, 1])
        result = _prepare_veomni_flash_attention_kwargs(position_ids)
        assert "cu_seq_lens_q" in result
        assert "cu_seq_lens_k" in result
        assert "max_length_q" in result
        assert "max_length_k" in result

    def test_2d_standard_packed(self):
        """2D: (1, total_nnz) - standard packed format."""
        position_ids = torch.tensor([[0, 1, 2, 0, 1]])
        result = _prepare_veomni_flash_attention_kwargs(position_ids)
        assert "cu_seq_lens_q" in result

    def test_3d_rope_dim_1_nnz(self):
        """3D: (rope_dim, 1, total_nnz) - VeRL mRoPE packed format."""
        position_ids = torch.tensor([
            [[0, 1, 2, 0, 1]],
            [[0, 1, 2, 0, 1]],
            [[0, 1, 2, 0, 1]],
        ])
        assert position_ids.shape == (3, 1, 5)
        result = _prepare_veomni_flash_attention_kwargs(position_ids)
        assert "cu_seq_lens_q" in result

    def test_3d_1_rope_dim_nnz(self):
        """3D: (1, rope_dim, total_nnz) - alternative layout."""
        position_ids = torch.tensor([
            [
                [0, 1, 2, 0, 1],
                [0, 1, 2, 0, 1],
                [0, 1, 2, 0, 1],
            ]
        ])
        assert position_ids.shape == (1, 3, 5)
        result = _prepare_veomni_flash_attention_kwargs(position_ids)
        assert "cu_seq_lens_q" in result

    def test_3d_rope_dim_batch_nnz(self):
        """3D: (rope_dim, batch, total_nnz) - take first rope dim."""
        position_ids = torch.tensor([
            [[0, 1, 2, 0, 1], [0, 1, 0, 1, 2]],
            [[0, 1, 2, 0, 1], [0, 1, 0, 1, 2]],
            [[0, 1, 2, 0, 1], [0, 1, 0, 1, 2]],
        ])
        assert position_ids.shape == (3, 2, 5)
        result = _prepare_veomni_flash_attention_kwargs(position_ids)
        assert "cu_seq_lens_q" in result

    def test_3d_unsupported_shape_raises(self):
        """3D with ambiguous shape should raise ValueError."""
        position_ids = torch.zeros(2, 2, 5, dtype=torch.long)
        with pytest.raises(ValueError, match="Unsupported 3D position_ids shape"):
            _prepare_veomni_flash_attention_kwargs(position_ids)

    def test_4d_raises(self):
        """4D tensor should raise ValueError."""
        position_ids = torch.zeros(1, 1, 1, 5, dtype=torch.long)
        with pytest.raises(ValueError, match="Unsupported position_ids rank"):
            _prepare_veomni_flash_attention_kwargs(position_ids)


# ---- Source-level tests that don't require runtime imports ----
# These parse the source AST directly to verify structural correctness.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVLTypeIndex:
    """Test VL_TYPE2INDEX includes Qwen3.5 models (source-level parsing)."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        utils_path = os.path.join(REPO_ROOT, "verl", "workers", "engine", "veomni", "utils.py")
        with open(utils_path) as f:
            self.source = f.read()

    def test_qwen3_5_present(self):
        assert '"qwen3_5"' in self.source, "qwen3_5 should be in VL_TYPE2INDEX"
        assert "248056" in self.source, "Qwen3.5 IMAGE_INPUT_INDEX should be 248056"
        assert "248057" in self.source, "Qwen3.5 VIDEO_INPUT_INDEX should be 248057"

    def test_qwen3_5_moe_present(self):
        assert '"qwen3_5_moe"' in self.source, "qwen3_5_moe should be in VL_TYPE2INDEX"

    def test_existing_models_untouched(self):
        """Ensure existing model entries are not accidentally removed."""
        assert '"qwen2_5_vl"' in self.source
        assert '"qwen3_vl"' in self.source
        assert '"deepseek_v3"' in self.source, "deepseek_v3 in MOE_PARAM_HANDERS should not be removed"


class TestDistillationLossVeomniStrategy:
    """Test that compute_topk_loss supports veomni strategy (source-level parsing)."""

    def test_veomni_strategy_in_source(self):
        """Verify 'veomni' is matched alongside 'fsdp' in compute_topk_loss."""
        losses_path = os.path.join(REPO_ROOT, "verl", "trainer", "distillation", "losses.py")
        with open(losses_path) as f:
            source = f.read()
        assert '"fsdp" | "veomni"' in source, (
            "compute_topk_loss should match 'veomni' alongside 'fsdp'"
        )


class TestTransformerImplSPLogic:
    """Source-level verification of SP-related changes in transformer_impl.py."""

    @pytest.fixture(autouse=True)
    def _load_source(self):
        impl_path = os.path.join(REPO_ROOT, "verl", "workers", "engine", "veomni", "transformer_impl.py")
        with open(impl_path) as f:
            self.source = f.read()

    def test_pixel_values_videos_in_padding_features(self):
        """pixel_values_videos should be in padding_features for video SP support."""
        assert '"pixel_values_videos"' in self.source

    def test_dict_keys_iteration_safe(self):
        """__call__ should iterate over list(batch.keys()) to avoid dict mutation."""
        assert "list(batch.keys())" in self.source

    def test_sp_structure_follows_upstream(self):
        """prepare_model_inputs should follow upstream structure: collator before cu_seq_lens."""
        prep_pos = self.source.find("def prepare_model_inputs")
        assert prep_pos > 0
        section = self.source[prep_pos:]
        # sp_shard_collator(model_inputs) should appear before _prepare_veomni_flash_attention_kwargs
        collator_call_pos = section.find("sp_shard_collator(model_inputs)")
        fa_pos = section.find("_prepare_veomni_flash_attention_kwargs(model_inputs")
        assert collator_call_pos > 0, "sp_shard_collator(model_inputs) call should exist"
        assert fa_pos > 0, "_prepare_veomni_flash_attention_kwargs call should exist"
        assert collator_call_pos < fa_pos, (
            "SP collator should run before cu_seq_lens computation (upstream structure)"
        )
        # position_ids SP slice should happen after cu_seq_lens
        sp_slice_pos = section.find("sp_shard_collator.sp_slice(model_inputs")
        assert sp_slice_pos > fa_pos, (
            "position_ids SP slice should happen after cu_seq_lens computation"
        )

    def test_1d_position_ids_support(self):
        """_prepare_veomni_flash_attention_kwargs should handle 1D position_ids."""
        assert "position_ids.dim() == 1" in self.source

    def test_no_chinese_comments(self):
        """No Chinese comments should be present in the modified file."""
        # Check for common Chinese characters
        import re
        chinese_pattern = re.compile(r"[\u4e00-\u9fff]")
        for i, line in enumerate(self.source.split("\n"), 1):
            if line.lstrip().startswith("#") and chinese_pattern.search(line):
                pytest.fail(f"Chinese comment found at line {i}: {line.strip()}")
