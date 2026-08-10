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
from unittest.mock import MagicMock, patch

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
            base_model_path=None,
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
            base_model_path=None,
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

        state_dict = OrderedDict(
            {
                "base_model.model.layers.0.self_attn.q_proj.base_layer.weight": base_weight.clone(),
                "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": lora_a.clone(),
                "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": lora_b.clone(),
            }
        )

        model = MagicMock()
        fused_state = {"layers.0.self_attn.q_proj.weight": torch.randn(4, 4)}
        fused_model = MagicMock()
        fused_model.state_dict.return_value = fused_state

        peft_model = MagicMock()
        peft_model.merge_and_unload.return_value = fused_model

        merger = MagicMock(spec=BaseModelMerger)
        merger.config = MagicMock()
        merger.config.target_dir = tempfile.mkdtemp()
        merger._load_lora_train_meta = MagicMock(return_value={"r": 2, "lora_alpha": 4, "use_rslora": False})
        # Wire real save_lora_adapter so it actually processes the keys
        merger.save_lora_adapter = BaseModelMerger.save_lora_adapter.__get__(merger)
        merger._fuse_lora_into_model = BaseModelMerger._fuse_lora_into_model.__get__(merger)

        try:
            with patch("peft.PeftModel.from_pretrained", return_value=peft_model):
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

        state_dict = OrderedDict(
            {
                "base_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.randn(4, 4),
                "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(2, 4),
                "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(4, 2),
            }
        )

        model = MagicMock()
        fused_model = MagicMock()
        fused_model.state_dict.return_value = {"q_proj.weight": torch.randn(4, 4)}

        peft_model = MagicMock()
        peft_model.merge_and_unload.return_value = fused_model

        merger = MagicMock(spec=BaseModelMerger)
        merger.config = MagicMock()
        merger.config.target_dir = tempfile.mkdtemp()
        merger._load_lora_train_meta = MagicMock(return_value={"r": 2, "lora_alpha": 4, "use_rslora": False})
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

    @pytest.mark.parametrize("meta", [None, {}, {"r": 2, "lora_alpha": 0}])
    def test_fuse_raises_when_lora_alpha_missing_or_zero(self, meta):
        """Fusing when lora_train_meta.json is missing or has lora_alpha=0 must raise ValueError
        before mutating state_dict, rather than silently scaling the adapter by 0."""
        from verl.model_merger.base_model_merger import BaseModelMerger

        state_dict = OrderedDict(
            {
                "base_model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.randn(4, 4),
                "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(2, 4),
                "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(4, 2),
            }
        )
        original_keys = list(state_dict.keys())
        model = MagicMock()

        merger = MagicMock(spec=BaseModelMerger)
        merger.config = MagicMock()
        merger.config.local_dir = "/tmp/checkpoint"
        merger._load_lora_train_meta = MagicMock(return_value=meta)
        merger._fuse_lora_into_model = BaseModelMerger._fuse_lora_into_model.__get__(merger)

        with pytest.raises(ValueError, match="lora_train_meta.json must record a nonzero lora_alpha"):
            merger._fuse_lora_into_model(state_dict, model)

        assert list(state_dict.keys()) == original_keys
        merger.save_lora_adapter.assert_not_called()

    def test_fuse_raises_when_adapter_only_checkpoint(self):
        """Fusing an adapter-only checkpoint (no base weights) must raise ValueError before mutating state_dict."""
        from verl.model_merger.base_model_merger import BaseModelMerger

        state_dict = OrderedDict(
            {
                "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.randn(2, 4),
                "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.randn(4, 2),
            }
        )
        original_keys = list(state_dict.keys())
        model = MagicMock()

        merger = MagicMock(spec=BaseModelMerger)
        merger.config = MagicMock()
        merger.config.local_dir = "/tmp/checkpoint"
        merger._load_lora_train_meta = MagicMock(return_value={"r": 2, "lora_alpha": 4, "use_rslora": False})
        merger._fuse_lora_into_model = BaseModelMerger._fuse_lora_into_model.__get__(merger)

        with pytest.raises(ValueError, match="no base model weights"):
            merger._fuse_lora_into_model(state_dict, model)

        assert list(state_dict.keys()) == original_keys
        merger.save_lora_adapter.assert_not_called()

    def test_fuse_raises_on_missing_base_parameters_and_allows_tied_aliases(self):
        """Missing non-tied base parameters must raise ValueError, while tied-weight aliases
        satisfied by their canonical parameter are allowed."""
        from verl.model_merger.base_model_merger import BaseModelMerger

        class ToyTiedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Linear(4, 4, bias=False)
                self.q_proj = torch.nn.Linear(4, 4, bias=False)
                self.lm_head = torch.nn.Linear(4, 4, bias=False)

            def tie_weights(self):
                self.lm_head.weight = self.embed.weight

        merger = MagicMock(spec=BaseModelMerger)
        merger.config = MagicMock()
        merger.config.local_dir = "/tmp/checkpoint"
        merger.config.target_dir = tempfile.mkdtemp()
        merger._load_lora_train_meta = MagicMock(return_value={"r": 2, "lora_alpha": 4, "use_rslora": False})
        merger.save_lora_adapter = BaseModelMerger.save_lora_adapter.__get__(merger)
        merger._fuse_lora_into_model = BaseModelMerger._fuse_lora_into_model.__get__(merger)

        try:
            # 1) Missing embed.weight (and lm_head.weight) should raise ValueError and clean up temp adapter dir
            incomplete_sd = OrderedDict(
                {
                    "base_model.model.q_proj.base_layer.weight": torch.randn(4, 4),
                    "base_model.model.q_proj.lora_A.default.weight": torch.randn(2, 4),
                    "base_model.model.q_proj.lora_B.default.weight": torch.randn(4, 2),
                }
            )
            model_incomplete = ToyTiedModel()
            with pytest.raises(ValueError, match="missing base model parameters"):
                merger._fuse_lora_into_model(incomplete_sd, model_incomplete)
            assert not os.path.exists(os.path.join(merger.config.target_dir, "lora_adapter"))

            # 2) Providing embed.weight + q_proj.weight (with lm_head.weight omitted because it is tied) succeeds
            complete_sd = OrderedDict(
                {
                    "base_model.model.embed.weight": torch.randn(4, 4),
                    "base_model.model.q_proj.base_layer.weight": torch.randn(4, 4),
                    "base_model.model.q_proj.lora_A.default.weight": torch.randn(2, 4),
                    "base_model.model.q_proj.lora_B.default.weight": torch.randn(4, 2),
                }
            )
            model_complete = ToyTiedModel()
            fused_model = MagicMock()
            fused_model.state_dict.return_value = {
                "embed.weight": torch.randn(4, 4),
                "q_proj.weight": torch.randn(4, 4),
            }
            peft_model = MagicMock()
            peft_model.merge_and_unload.return_value = fused_model
            with patch("peft.PeftModel.from_pretrained", return_value=peft_model):
                merger._fuse_lora_into_model(complete_sd, model_complete)
            peft_model.merge_and_unload.assert_called_once()
        finally:
            if os.path.exists(merger.config.target_dir):
                shutil.rmtree(merger.config.target_dir)

    def test_fuse_raises_on_target_parameters_adapter(self):
        """Adapters configured with target_modules=[] and target_parameters (or expert base_layer keys)
        must be rejected before extraction without mutating state_dict."""
        from verl.model_merger.base_model_merger import BaseModelMerger

        state_dict = OrderedDict(
            {
                "base_model.model.layers.0.feed_forward.experts.base_layer.gate_up_proj": torch.randn(4, 4),
                "base_model.model.layers.0.feed_forward.experts.lora_A.default.weight": torch.randn(2, 4),
                "base_model.model.layers.0.feed_forward.experts.lora_B.default.weight": torch.randn(4, 2),
            }
        )
        original_keys = list(state_dict.keys())
        model = MagicMock()

        merger = MagicMock(spec=BaseModelMerger)
        merger.config = MagicMock()
        merger.config.local_dir = "/tmp/checkpoint"
        merger._load_lora_train_meta = MagicMock(
            return_value={
                "r": 2,
                "lora_alpha": 4,
                "use_rslora": False,
                "target_parameters": ["feed_forward.experts.gate_up_proj"],
            }
        )
        merger._fuse_lora_into_model = BaseModelMerger._fuse_lora_into_model.__get__(merger)

        with pytest.raises(ValueError, match="target_parameters"):
            merger._fuse_lora_into_model(state_dict, model)

        assert list(state_dict.keys()) == original_keys
        merger.save_lora_adapter.assert_not_called()

    def test_fuse_raises_when_use_rslora_missing_in_metadata(self):
        """Older checkpoints whose lora_train_meta.json omits use_rslora must be rejected before extraction."""
        from verl.model_merger.base_model_merger import BaseModelMerger

        state_dict = OrderedDict(
            {
                "base_model.model.q_proj.base_layer.weight": torch.randn(4, 4),
                "base_model.model.q_proj.lora_A.default.weight": torch.randn(2, 4),
                "base_model.model.q_proj.lora_B.default.weight": torch.randn(4, 2),
            }
        )
        original_keys = list(state_dict.keys())
        model = MagicMock()

        merger = MagicMock(spec=BaseModelMerger)
        merger.config = MagicMock()
        merger.config.local_dir = "/tmp/checkpoint"
        merger._load_lora_train_meta = MagicMock(return_value={"r": 2, "lora_alpha": 4})
        merger._fuse_lora_into_model = BaseModelMerger._fuse_lora_into_model.__get__(merger)

        with pytest.raises(ValueError, match="must explicitly record boolean use_rslora"):
            merger._fuse_lora_into_model(state_dict, model)

        assert list(state_dict.keys()) == original_keys
        merger.save_lora_adapter.assert_not_called()

    def test_fuse_cleans_up_temp_adapter_dir_when_load_state_dict_raises(self):
        """If model.load_state_dict raises (e.g., shape mismatch), the temp lora_adapter directory is still removed."""
        from verl.model_merger.base_model_merger import BaseModelMerger

        state_dict = OrderedDict(
            {
                "base_model.model.q_proj.base_layer.weight": torch.randn(8, 8),
                "base_model.model.q_proj.lora_A.default.weight": torch.randn(2, 4),
                "base_model.model.q_proj.lora_B.default.weight": torch.randn(4, 2),
            }
        )
        model = MagicMock()
        model.load_state_dict.side_effect = RuntimeError("size mismatch for q_proj.weight")

        merger = MagicMock(spec=BaseModelMerger)
        merger.config = MagicMock()
        merger.config.local_dir = "/tmp/checkpoint"
        merger.config.target_dir = tempfile.mkdtemp()
        merger._load_lora_train_meta = MagicMock(return_value={"r": 2, "lora_alpha": 4, "use_rslora": False})
        merger.save_lora_adapter = BaseModelMerger.save_lora_adapter.__get__(merger)
        merger._fuse_lora_into_model = BaseModelMerger._fuse_lora_into_model.__get__(merger)

        try:
            with pytest.raises(RuntimeError, match="size mismatch"):
                merger._fuse_lora_into_model(state_dict, model)
            assert not os.path.exists(os.path.join(merger.config.target_dir, "lora_adapter"))
        finally:
            shutil.rmtree(merger.config.target_dir, ignore_errors=True)

    def test_save_lora_adapter_strips_default_from_embedding_keys(self):
        """Embedding LoRA keys (.lora_embedding_A.default / .lora_embedding_B.default) must have
        the trailing .default suffix stripped before saving to adapter_model.safetensors."""
        from safetensors.torch import load_file

        from verl.model_merger.base_model_merger import BaseModelMerger

        merger = MagicMock(spec=BaseModelMerger)
        merger.config = MagicMock()
        merger.config.local_dir = None
        merger.config.target_dir = tempfile.mkdtemp()
        merger._load_lora_train_meta = MagicMock(return_value={"r": 2, "lora_alpha": 4, "use_rslora": False})
        merger.save_lora_adapter = BaseModelMerger.save_lora_adapter.__get__(merger)

        try:
            state_dict = OrderedDict(
                {
                    "base_model.model.embed_tokens.base_layer.weight": torch.randn(8, 4),
                    "base_model.model.embed_tokens.lora_embedding_A.default": torch.randn(2, 8),
                    "base_model.model.embed_tokens.lora_embedding_B.default": torch.randn(4, 2),
                }
            )
            lora_path = merger.save_lora_adapter(state_dict)
            saved_tensors = load_file(os.path.join(lora_path, "adapter_model.safetensors"))
            assert "base_model.model.embed_tokens.lora_embedding_A" in saved_tensors
            assert "base_model.model.embed_tokens.lora_embedding_B" in saved_tensors
            assert "embed_tokens.weight" in state_dict
        finally:
            shutil.rmtree(merger.config.target_dir, ignore_errors=True)

    def test_fuse_preserves_use_rslora_through_adapter_config(self):
        """When lora_train_meta.json records use_rslora=True, save_lora_adapter must persist
        use_rslora=True in adapter_config.json so PeftModel.from_pretrained uses alpha/sqrt(r) scaling."""
        import json

        from verl.model_merger.base_model_merger import BaseModelMerger

        merger = MagicMock(spec=BaseModelMerger)
        merger.config = MagicMock()
        merger.config.local_dir = tempfile.mkdtemp()
        merger.config.target_dir = tempfile.mkdtemp()
        merger._load_lora_train_meta = BaseModelMerger._load_lora_train_meta.__get__(merger)
        merger.save_lora_adapter = BaseModelMerger.save_lora_adapter.__get__(merger)

        try:
            with open(os.path.join(merger.config.local_dir, "lora_train_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"r": 4, "lora_alpha": 8, "task_type": "CAUSAL_LM", "use_rslora": True}, f)

            state_dict = OrderedDict(
                {
                    "base_model.model.q_proj.base_layer.weight": torch.randn(4, 4),
                    "base_model.model.q_proj.lora_A.default.weight": torch.randn(4, 4),
                    "base_model.model.q_proj.lora_B.default.weight": torch.randn(4, 4),
                }
            )
            lora_path = merger.save_lora_adapter(state_dict)
            with open(os.path.join(lora_path, "adapter_config.json"), encoding="utf-8") as f:
                adapter_cfg = json.load(f)
            assert adapter_cfg["use_rslora"] is True
        finally:
            shutil.rmtree(merger.config.local_dir, ignore_errors=True)
            shutil.rmtree(merger.config.target_dir, ignore_errors=True)

    def test_fuse_embedding_lora_does_not_alter_tied_lm_head(self):
        """Real PEFT merge of an embedding LoRA on a tied-weights model must update embed_tokens.weight
        while keeping the untargeted lm_head.weight unchanged across save_pretrained and from_pretrained."""
        from transformers import LlamaConfig, LlamaForCausalLM

        from verl.model_merger.base_model_merger import BaseModelMerger, drop_tied_target_keys

        cfg = LlamaConfig(
            vocab_size=8,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            tie_word_embeddings=True,
        )
        model = LlamaForCausalLM(cfg)
        base_sd = model.state_dict()
        base_emb = base_sd["model.embed_tokens.weight"].clone()
        lora_emb_a = torch.ones(2, 8)
        lora_emb_b = torch.ones(4, 2)

        state_dict = OrderedDict()
        for k, v in base_sd.items():
            if k == "lm_head.weight":
                continue
            if k == "model.embed_tokens.weight":
                state_dict["base_model.model.model.embed_tokens.base_layer.weight"] = v.clone()
            else:
                state_dict[f"base_model.model.{k}"] = v.clone()
        state_dict["base_model.model.model.embed_tokens.lora_embedding_A.default"] = lora_emb_a.clone()
        state_dict["base_model.model.model.embed_tokens.lora_embedding_B.default"] = lora_emb_b.clone()

        merger = MagicMock(spec=BaseModelMerger)
        merger.config = MagicMock()
        merger.config.local_dir = "/tmp/checkpoint"
        merger.config.target_dir = tempfile.mkdtemp()
        merger.model_config = cfg
        merger._load_lora_train_meta = MagicMock(return_value={"r": 2, "lora_alpha": 4, "use_rslora": False})
        merger.save_lora_adapter = BaseModelMerger.save_lora_adapter.__get__(merger)
        merger._fuse_lora_into_model = BaseModelMerger._fuse_lora_into_model.__get__(merger)

        save_dir = tempfile.mkdtemp()
        try:
            merger._fuse_lora_into_model(state_dict, model)
            expected_emb = base_emb + (lora_emb_a.T @ lora_emb_b.T) * (4.0 / 2.0)
            assert torch.allclose(state_dict["model.embed_tokens.weight"], expected_emb)
            assert torch.equal(state_dict["lm_head.weight"], base_emb)
            assert merger.model_config.tie_word_embeddings is False

            drop_tied_target_keys(state_dict, model, merger.model_config)
            model.save_pretrained(save_dir, state_dict=state_dict)
            reloaded = LlamaForCausalLM.from_pretrained(save_dir)
            assert torch.allclose(reloaded.model.embed_tokens.weight, expected_emb)
            assert torch.equal(reloaded.lm_head.weight, base_emb)
        finally:
            shutil.rmtree(merger.config.target_dir, ignore_errors=True)
            shutil.rmtree(save_dir, ignore_errors=True)

    def test_fuse_preserves_alpha_pattern_in_real_peft_merge(self):
        """Real PEFT merge must honor per-layer alpha_pattern overrides from lora_train_meta.json."""
        import json

        from verl.model_merger.base_model_merger import BaseModelMerger

        class SimpleLinearModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(4, 4, bias=False)

        base_w = torch.zeros(4, 4)
        lora_a = torch.eye(2, 4)
        lora_b = torch.eye(4, 2)

        state_dict = OrderedDict(
            {
                "base_model.model.q_proj.base_layer.weight": base_w.clone(),
                "base_model.model.q_proj.lora_A.default.weight": lora_a.clone(),
                "base_model.model.q_proj.lora_B.default.weight": lora_b.clone(),
            }
        )

        merger = MagicMock(spec=BaseModelMerger)
        merger.config = MagicMock()
        merger.config.local_dir = tempfile.mkdtemp()
        merger.config.target_dir = tempfile.mkdtemp()
        merger._load_lora_train_meta = BaseModelMerger._load_lora_train_meta.__get__(merger)
        merger.save_lora_adapter = BaseModelMerger.save_lora_adapter.__get__(merger)
        merger._fuse_lora_into_model = BaseModelMerger._fuse_lora_into_model.__get__(merger)

        try:
            with open(os.path.join(merger.config.local_dir, "lora_train_meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "r": 2,
                        "lora_alpha": 4,
                        "use_rslora": False,
                        "alpha_pattern": {"q_proj": 8},
                    },
                    f,
                )

            model = SimpleLinearModel()
            merger._fuse_lora_into_model(state_dict, model)
            expected = (lora_b @ lora_a) * (8.0 / 2.0)
            assert torch.allclose(state_dict["q_proj.weight"], expected)
        finally:
            shutil.rmtree(merger.config.local_dir, ignore_errors=True)
            shutil.rmtree(merger.config.target_dir, ignore_errors=True)


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
        merger.model_config = MagicMock()
        merger.save_hf_model_and_tokenizer = BaseModelMerger.save_hf_model_and_tokenizer.__get__(merger)
        return merger

    @patch("verl.model_merger.base_model_merger.drop_tied_target_keys")
    @patch("verl.model_merger.base_model_merger.hf_processor", return_value=None)
    @patch("verl.model_merger.base_model_merger.hf_tokenizer", return_value=None)
    @patch("verl.model_merger.base_model_merger.validate_hf_model_output")
    def test_fuse_lora_true_calls_fuse_method(self, _val, _tok, _proc, _drop):
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

    @patch("verl.model_merger.base_model_merger.drop_tied_target_keys")
    @patch("verl.model_merger.base_model_merger.hf_processor", return_value=None)
    @patch("verl.model_merger.base_model_merger.hf_tokenizer", return_value=None)
    @patch("verl.model_merger.base_model_merger.validate_hf_model_output")
    def test_fuse_lora_false_calls_save_lora_adapter(self, _val, _tok, _proc, _drop):
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
