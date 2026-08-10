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

"""Unit tests for the --fuse-lora flag in model_merger.

Tests the config plumbing, the branching logic in save_hf_model_and_tokenizer,
and the _fuse_lora_into_model method. All tests run on CPU without GPU,
ray, or distributed setup.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from collections import OrderedDict
from unittest.mock import MagicMock, patch, call

import pytest
import torch

from verl.model_merger.base_model_merger import ModelMergerConfig, generate_config_from_args, parse_args


# ---------------------------------------------------------------------------
# Config and CLI tests
# ---------------------------------------------------------------------------


class TestFuseLoraConfig:
    """Test that --fuse-lora flag is wired through CLI to config."""

    def test_config_default_is_false(self):
        config = ModelMergerConfig(operation="merge", backend="fsdp")
        assert config.fuse_lora is False

    def test_config_accepts_true(self):
        config = ModelMergerConfig(operation="merge", backend="fsdp", fuse_lora=True)
        assert config.fuse_lora is True

    def test_cli_flag_parsed(self):
        """--fuse-lora flag is recognized by argparse."""
        with patch("sys.argv", ["prog", "merge", "--backend", "fsdp", "--local_dir", "/tmp/x", "--fuse-lora"]):
            args = parse_args()
        assert args.fuse_lora is True

    def test_cli_flag_absent(self):
        """Without --fuse-lora, fuse_lora defaults to False."""
        with patch("sys.argv", ["prog", "merge", "--backend", "fsdp", "--local_dir", "/tmp/x"]):
            args = parse_args()
        assert args.fuse_lora is False

    def test_generate_config_passes_fuse_lora(self):
        """generate_config_from_args passes fuse_lora to ModelMergerConfig."""
        args = argparse.Namespace(
            operation="merge",
            backend="fsdp",
            local_dir="/tmp/x",
            target_dir="/tmp/y",
            hf_upload_path=None,
            private=False,
            fuse_lora=True,
            tie_word_embedding=False,
            trust_remote_code=False,
            is_value_model=False,
            use_cpu_initialization=False,
        )
        with patch("os.makedirs"):
            config = generate_config_from_args(args)
        assert config.fuse_lora is True

    def test_test_operation_no_fuse_lora(self):
        """Test operation does not have fuse_lora, getattr fallback works."""
        args = argparse.Namespace(
            operation="test",
            backend="fsdp",
            local_dir="/tmp/x",
            test_hf_dir="/tmp/z",
            tie_word_embedding=False,
            trust_remote_code=False,
            is_value_model=False,
            use_cpu_initialization=False,
        )
        config = generate_config_from_args(args)
        assert config.fuse_lora is False


# ---------------------------------------------------------------------------
# save_lora_adapter splits LoRA keys from state_dict (existing behavior)
# ---------------------------------------------------------------------------


class TestSaveLoraAdapterSplitsKeys:
    """Verify that save_lora_adapter pops LoRA keys and leaves base keys."""

    def test_lora_keys_removed_from_state_dict(self):
        """After save_lora_adapter, state_dict should not contain lora_ keys."""
        state_dict = OrderedDict(
            {
                "base_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.randn(4, 4),
                "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(2, 4),
                "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(4, 2),
            }
        )
        merger = MagicMock()
        merger.config = MagicMock()
        merger.config.target_dir = tempfile.mkdtemp()
        merger._load_lora_train_meta = MagicMock(return_value=None)

        try:
            from verl.model_merger.base_model_merger import BaseModelMerger

            lora_path = BaseModelMerger.save_lora_adapter(merger, state_dict)

            assert lora_path is not None
            for key in state_dict:
                assert "lora_" not in key
            assert "layers.0.self_attn.q_proj.weight" in state_dict
        finally:
            shutil.rmtree(merger.config.target_dir)

    def test_no_lora_keys_returns_none(self):
        """If no LoRA keys present, save_lora_adapter returns None."""
        state_dict = OrderedDict({"model.weight": torch.randn(4, 4)})
        merger = MagicMock()

        from verl.model_merger.base_model_merger import BaseModelMerger

        result = BaseModelMerger.save_lora_adapter(merger, state_dict)
        assert result is None


# ---------------------------------------------------------------------------
# _fuse_lora_into_model tests
# ---------------------------------------------------------------------------


class TestFuseLoraIntoModel:
    """Test the _fuse_lora_into_model method."""

    def test_no_lora_keys_returns_none(self):
        """If state_dict has no lora_ keys, return None and leave dict unchanged."""
        from verl.model_merger.base_model_merger import BaseModelMerger

        state_dict = OrderedDict({"model.weight": torch.randn(4, 4)})
        original_keys = list(state_dict.keys())
        model = MagicMock()

        merger = MagicMock(spec=BaseModelMerger)
        merger._fuse_lora_into_model = BaseModelMerger._fuse_lora_into_model.__get__(merger)

        result = merger._fuse_lora_into_model(state_dict, model)
        assert result is None
        assert list(state_dict.keys()) == original_keys

    def test_fuse_calls_merge_and_unload(self):
        """Verify that _fuse_lora_into_model calls PeftModel.from_pretrained
        and merge_and_unload, then replaces state_dict contents."""
        from verl.model_merger.base_model_merger import BaseModelMerger

        base_weight = torch.randn(4, 4)
        lora_a = torch.randn(2, 4)
        lora_b = torch.randn(4, 2)

        state_dict = OrderedDict({
            "base_model.model.layers.0.self_attn.q_proj.base_layer.weight": base_weight.clone(),
            "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": lora_a.clone(),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": lora_b.clone(),
        })

        model = MagicMock()
        fused_state = {"layers.0.self_attn.q_proj.weight": torch.randn(4, 4)}
        fused_model = MagicMock()
        fused_model.state_dict.return_value = fused_state

        peft_model = MagicMock()
        peft_model.merge_and_unload.return_value = fused_model

        merger = MagicMock(spec=BaseModelMerger)
        merger.config = MagicMock()
        merger.config.target_dir = tempfile.mkdtemp()
        merger._load_lora_train_meta = MagicMock(return_value=None)
        # Wire real save_lora_adapter so it actually processes the keys
        merger.save_lora_adapter = BaseModelMerger.save_lora_adapter.__get__(merger)
        merger._fuse_lora_into_model = BaseModelMerger._fuse_lora_into_model.__get__(merger)

        try:
            with patch("peft.PeftModel.from_pretrained", return_value=peft_model) as mock_from_pretrained:
                result = merger._fuse_lora_into_model(state_dict, model)

            assert result is None
            # merge_and_unload must have been called
            peft_model.merge_and_unload.assert_called_once()
            # state_dict should now contain fused weights, not original keys
            assert "layers.0.self_attn.q_proj.weight" in state_dict
            assert len([k for k in state_dict if "lora_" in k]) == 0
        finally:
            # cleanup target_dir if it still exists
            if os.path.exists(merger.config.target_dir):
                shutil.rmtree(merger.config.target_dir)

    def test_fuse_cleans_up_temp_adapter_dir(self):
        """After fusing, the temporary lora_adapter directory should be removed."""
        from verl.model_merger.base_model_merger import BaseModelMerger

        state_dict = OrderedDict({
            "base_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.randn(4, 4),
            "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(2, 4),
            "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(4, 2),
        })

        model = MagicMock()
        fused_model = MagicMock()
        fused_model.state_dict.return_value = {"q_proj.weight": torch.randn(4, 4)}

        peft_model = MagicMock()
        peft_model.merge_and_unload.return_value = fused_model

        merger = MagicMock(spec=BaseModelMerger)
        merger.config = MagicMock()
        merger.config.target_dir = tempfile.mkdtemp()
        merger._load_lora_train_meta = MagicMock(return_value=None)
        merger.save_lora_adapter = BaseModelMerger.save_lora_adapter.__get__(merger)
        merger._fuse_lora_into_model = BaseModelMerger._fuse_lora_into_model.__get__(merger)

        try:
            with patch("peft.PeftModel.from_pretrained", return_value=peft_model):
                merger._fuse_lora_into_model(state_dict, model)

            lora_adapter_path = os.path.join(merger.config.target_dir, "lora_adapter")
            assert not os.path.exists(lora_adapter_path), "temp lora_adapter dir should be cleaned up"
        finally:
            if os.path.exists(merger.config.target_dir):
                shutil.rmtree(merger.config.target_dir)


# ---------------------------------------------------------------------------
# Branching logic in save_hf_model_and_tokenizer
# ---------------------------------------------------------------------------


class TestSaveHfModelBranching:
    """Test that save_hf_model_and_tokenizer calls the right method based on fuse_lora."""

    def _make_merger(self, fuse_lora: bool):
        """Create a mock merger with the branching logic wired up."""
        from verl.model_merger.base_model_merger import BaseModelMerger

        merger = MagicMock(spec=BaseModelMerger)
        merger.config = ModelMergerConfig(
            operation="merge", backend="fsdp", fuse_lora=fuse_lora, target_dir="/tmp/test-target"
        )
        merger.save_hf_model_and_tokenizer = BaseModelMerger.save_hf_model_and_tokenizer.__get__(merger)
        return merger

    @patch("verl.utils.hf_processor", return_value=None)
    @patch("verl.utils.hf_tokenizer", return_value=None)
    @patch("verl.model_merger.output_validation.validate_hf_model_output")
    def test_fuse_lora_true_calls_fuse_method(self, _val, _tok, _proc):
        """When fuse_lora=True, _fuse_lora_into_model is called instead of save_lora_adapter."""
        merger = self._make_merger(fuse_lora=True)
        merger._fuse_lora_into_model.return_value = None
        merger.hf_model_config_path = "/tmp/fake"

        mock_model = MagicMock()
        mock_model.can_generate.return_value = False

        with patch("verl.model_merger.base_model_merger.init_empty_weights"):
            merger.get_transformers_auto_model_class.return_value.from_config.return_value = mock_model
            mock_model.to_empty.return_value = mock_model
            merger.patch_model_generation_config.return_value = mock_model

            state_dict = {"model.weight": torch.randn(4, 4)}
            merger.save_hf_model_and_tokenizer(state_dict)

        merger._fuse_lora_into_model.assert_called_once()
        merger.save_lora_adapter.assert_not_called()

    @patch("verl.utils.hf_processor", return_value=None)
    @patch("verl.utils.hf_tokenizer", return_value=None)
    @patch("verl.model_merger.output_validation.validate_hf_model_output")
    def test_fuse_lora_false_calls_save_lora_adapter(self, _val, _tok, _proc):
        """When fuse_lora=False, save_lora_adapter is called (default behavior)."""
        merger = self._make_merger(fuse_lora=False)
        merger.save_lora_adapter.return_value = None
        merger.hf_model_config_path = "/tmp/fake"

        mock_model = MagicMock()
        mock_model.can_generate.return_value = False

        with patch("verl.model_merger.base_model_merger.init_empty_weights"):
            merger.get_transformers_auto_model_class.return_value.from_config.return_value = mock_model
            mock_model.to_empty.return_value = mock_model
            merger.patch_model_generation_config.return_value = mock_model

            state_dict = {"model.weight": torch.randn(4, 4)}
            merger.save_hf_model_and_tokenizer(state_dict)

        merger.save_lora_adapter.assert_called_once()
        merger._fuse_lora_into_model.assert_not_called()
