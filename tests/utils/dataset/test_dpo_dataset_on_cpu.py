# Copyright 2026 Amazon.com Inc and/or its affiliates
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
"""
Unit tests for the DPO dataset, collator, and related utilities.

Tests cover:
- Data loading: single file, multi-file, directory, JSONL, error handling
- Packed-tensor output format: input_ids, loss_mask, boundary_offset
- Truncation behavior
- Invalid sample filtering
- Collator: NestedTensor batching, boundary_offset preservation
- YAML config resolution
- Collator subclass defaults
- Shuffle determinism (property-based)
- File loading roundtrip (property-based)
- Code quality: constants, try-parse, formatting standard
"""

import importlib.util
import json
import os
import re
import sys
import tempfile
import types

import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from omegaconf import OmegaConf

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


# ─── Helpers ─────────────────────────────────────────────────────────────


# Check torch availability once at module level
_TORCH_AVAILABLE = False
try:
    import torch as _torch_check  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    pass


def _get_dpo_dataset_source() -> str:
    """Read the raw source of dpo_dataset.py for inspection tests."""
    path = os.path.join(_REPO_ROOT, "verl/utils/dataset/dpo_dataset.py")
    with open(path) as f:
        return f.read()


def _ensure_mock_dependencies():
    """Install minimal mock modules for torch, ray, etc. if not available."""
    if "torch" not in sys.modules:
        mock_torch = types.ModuleType("torch")
        mock_torch.Tensor = type("Tensor", (), {})
        mock_torch.utils = types.ModuleType("torch.utils")
        mock_torch.utils.data = types.ModuleType("torch.utils.data")
        mock_torch.utils.data.Dataset = type("Dataset", (), {})
        mock_torch.nested = types.ModuleType("torch.nested")
        mock_torch.jagged = "jagged"
        sys.modules["torch"] = mock_torch
        sys.modules["torch.utils"] = mock_torch.utils
        sys.modules["torch.utils.data"] = mock_torch.utils.data
        sys.modules["torch.nested"] = mock_torch.nested

    if "tensordict" not in sys.modules:
        mock_td = types.ModuleType("tensordict")
        mock_td.tensorclass = types.ModuleType("tensordict.tensorclass")
        mock_td.tensorclass.NonTensorData = type("NonTensorData", (), {})
        sys.modules["tensordict"] = mock_td
        sys.modules["tensordict.tensorclass"] = mock_td.tensorclass

    if "transformers" not in sys.modules:
        mock_tf = types.ModuleType("transformers")
        mock_tf.PreTrainedTokenizer = type("PreTrainedTokenizer", (), {})
        sys.modules["transformers"] = mock_tf

    if "ray" not in sys.modules:
        sys.modules["ray"] = types.ModuleType("ray")

    if "verl" not in sys.modules:
        mock_verl = types.ModuleType("verl")
        sys.modules["verl"] = mock_verl
    else:
        mock_verl = sys.modules["verl"]

    if not hasattr(mock_verl, "utils"):
        mock_verl.utils = types.ModuleType("verl.utils")
        mock_verl.utils.fs = types.ModuleType("verl.utils.fs")
        mock_verl.utils.fs.copy_to_local = lambda src, **kwargs: src
        mock_verl.utils.hf_tokenizer = lambda x: x
        sys.modules.setdefault("verl.utils", mock_verl.utils)
        sys.modules.setdefault("verl.utils.fs", mock_verl.utils.fs)
        sys.modules.setdefault(
            "verl.utils.hf_tokenizer", types.ModuleType("verl.utils.hf_tokenizer")
        )

    if "verl.utils.tensordict_utils" not in sys.modules:
        mock_td_utils = types.ModuleType("verl.utils.tensordict_utils")
        mock_td_utils.nested_tensor_from_tensor_list = lambda tensors: tensors
        sys.modules["verl.utils.tensordict_utils"] = mock_td_utils

    if "verl.base_config" not in sys.modules:
        base_config_path = os.path.join(_REPO_ROOT, "verl", "base_config.py")
        base_spec = importlib.util.spec_from_file_location("verl.base_config", base_config_path)
        base_mod = importlib.util.module_from_spec(base_spec)
        base_spec.loader.exec_module(base_mod)
        sys.modules["verl.base_config"] = base_mod
        if "verl" in sys.modules:
            sys.modules["verl"].base_config = base_mod


def _load_dataset_utils():
    """Load verl/utils/dataset/dataset_utils.py directly (no heavy deps)."""
    _ensure_mock_dependencies()
    path = os.path.join(_REPO_ROOT, "verl", "utils", "dataset", "dataset_utils.py")
    spec = importlib.util.spec_from_file_location("dataset_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_dpo_dataset():
    """Load verl/utils/dataset/dpo_dataset.py directly (no heavy deps)."""
    _ensure_mock_dependencies()
    path = os.path.join(_REPO_ROOT, "verl", "utils", "dataset", "dpo_dataset.py")
    spec = importlib.util.spec_from_file_location("dpo_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_tokenizer():
    """Get a small tokenizer for testing (requires torch + transformers)."""
    from verl.utils import hf_tokenizer
    return hf_tokenizer("Qwen/Qwen2.5-0.5B-Instruct")


def _create_parquet(tmp_dir: str, name: str = "dpo_data.parquet", num_samples: int = 5, **col_overrides) -> str:
    """Create a temporary parquet file with single-turn DPO data."""
    data = {
        "prompt": [f"Question {i}: What is {i}+{i}?" for i in range(num_samples)],
        "chosen": [f"The answer is {i*2}." for i in range(num_samples)],
        "rejected": [f"I think it's {i*2 + 1}." for i in range(num_samples)],
    }
    data.update(col_overrides)
    df = pd.DataFrame(data)
    path = os.path.join(tmp_dir, name)
    df.to_parquet(path)
    return path


def _create_jsonl(tmp_dir: str, name: str = "dpo_data.jsonl", num_samples: int = 5) -> str:
    """Create a temporary JSONL file with DPO data."""
    path = os.path.join(tmp_dir, name)
    with open(path, "w") as f:
        for i in range(num_samples):
            f.write(json.dumps({
                "prompt": f"Question {i}?",
                "chosen": f"Good answer {i}.",
                "rejected": f"Bad answer {i}.",
            }) + "\n")
    return path


def _default_config(**overrides):
    """Create a default DPODataConfig for testing."""
    from verl.utils.dataset.dpo_dataset import DPODataConfig
    defaults = {"max_length": 256, "truncation": "right"}
    defaults.update(overrides)
    return DPODataConfig(**defaults)


@pytest.fixture(autouse=True)
def _isolate_sys_modules():
    """Snapshot sys.modules before mock injection and restore after."""
    snapshot = dict(sys.modules)
    yield
    sys.modules.clear()
    sys.modules.update(snapshot)


# ═════════════════════════════════════════════════════════════════════════
# SECTION 1: Data Loading Tests
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
class TestDPODatasetLoading:
    """Tests for data loading, validation, and multi-file support."""

    def test_loads_parquet(self):
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _create_parquet(tmp_dir, num_samples=7)
            dataset = DPODataset(data_files=path, tokenizer=tokenizer, config=_default_config())
            assert len(dataset) == 7

    def test_loads_jsonl(self):
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _create_jsonl(tmp_dir, num_samples=5)
            dataset = DPODataset(data_files=path, tokenizer=tokenizer, config=_default_config())
            assert len(dataset) == 5

    def test_max_samples(self):
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _create_parquet(tmp_dir, num_samples=10)
            dataset = DPODataset(data_files=path, tokenizer=tokenizer, config=_default_config(), max_samples=3)
            assert len(dataset) == 3

    def test_shuffle_changes_order(self):
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _create_parquet(tmp_dir, num_samples=10)
            ds_no = DPODataset(data_files=path, tokenizer=tokenizer, config=_default_config(shuffle=False))
            ds_yes = DPODataset(data_files=path, tokenizer=tokenizer, config=_default_config(shuffle=True, seed=42))
            assert ds_no.prompts != ds_yes.prompts

    def test_missing_columns_raises(self):
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            df = pd.DataFrame({"wrong_col": ["data"]})
            path = os.path.join(tmp_dir, "bad.parquet")
            df.to_parquet(path)
            with pytest.raises(ValueError, match="missing required columns"):
                DPODataset(data_files=path, tokenizer=tokenizer, config=_default_config())

    def test_unsupported_format_raises(self):
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "data.csv")
            with open(path, "w") as f:
                f.write("prompt,chosen,rejected\nhi,good,bad\n")
            with pytest.raises(ValueError, match="Could not parse"):
                DPODataset(data_files=path, tokenizer=tokenizer, config=_default_config())

    def test_custom_column_names(self):
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            df = pd.DataFrame({"q": ["Hi?"], "good": ["Hello!"], "bad": ["Bye"]})
            path = os.path.join(tmp_dir, "custom.parquet")
            df.to_parquet(path)
            config = _default_config(prompt_key="q", chosen_key="good", rejected_key="bad")
            dataset = DPODataset(data_files=path, tokenizer=tokenizer, config=config)
            assert len(dataset) == 1

    def test_multi_file_concatenation(self):
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            p1 = _create_parquet(tmp_dir, "part1.parquet", num_samples=5)
            p2 = _create_parquet(tmp_dir, "part2.parquet", num_samples=8)
            dataset = DPODataset(data_files=[p1, p2], tokenizer=tokenizer, config=_default_config())
            assert len(dataset) == 13

    def test_directory_loading(self):
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            _create_parquet(tmp_dir, "a.parquet", num_samples=3)
            _create_parquet(tmp_dir, "b.parquet", num_samples=4)
            dataset = DPODataset(data_files=tmp_dir, tokenizer=tokenizer, config=_default_config())
            assert len(dataset) == 7

    def test_empty_directory_raises(self):
        from verl.utils.dataset.dpo_dataset import resolve_data_files
        with tempfile.TemporaryDirectory() as tmp_dir:
            empty = os.path.join(tmp_dir, "empty")
            os.makedirs(empty)
            with pytest.raises(ValueError, match="No .parquet or .jsonl"):
                resolve_data_files(empty)


# ═════════════════════════════════════════════════════════════════════════
# SECTION 2: Packed-Tensor Output Tests
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
class TestDPODatasetPackedOutput:
    """Tests for the packed-tensor output format."""

    @pytest.fixture
    def dataset(self, tmp_path):
        from verl.utils.dataset.dpo_dataset import DPODataset
        path = _create_parquet(str(tmp_path), num_samples=3)
        return DPODataset(data_files=path, tokenizer=_get_tokenizer(), config=_default_config())

    def test_output_keys(self, dataset):
        sample = dataset[0]
        assert set(sample.keys()) == {"input_ids", "loss_mask", "boundary_offset"}

    def test_no_old_format_keys(self, dataset):
        sample = dataset[0]
        for old_key in ["chosen_input_ids", "rejected_input_ids", "chosen_labels",
                        "rejected_labels", "chosen_attention_mask", "rejected_attention_mask"]:
            assert old_key not in sample

    def test_boundary_offset_is_int(self, dataset):
        assert isinstance(dataset[0]["boundary_offset"], int)

    def test_boundary_offset_in_range(self, dataset):
        sample = dataset[0]
        offset = sample["boundary_offset"]
        total = sample["input_ids"].shape[0]
        assert 0 < offset < total

    def test_loss_mask_is_binary(self, dataset):
        mask = dataset[0]["loss_mask"]
        assert set(mask.unique().tolist()).issubset({0, 1})

    def test_loss_mask_shape_matches_input_ids(self, dataset):
        sample = dataset[0]
        assert sample["loss_mask"].shape == sample["input_ids"].shape

    def test_prompt_masked_in_both_halves(self, dataset):
        sample = dataset[0]
        offset = sample["boundary_offset"]
        assert sample["loss_mask"][0] == 0, "Chosen half: first token should be prompt (mask=0)"
        assert sample["loss_mask"][offset] == 0, "Rejected half: first token should be prompt (mask=0)"

    def test_response_tokens_have_mask_1(self, dataset):
        sample = dataset[0]
        mask = sample["loss_mask"]
        assert mask.sum() > 0, "Should have some response tokens (mask=1)"

    def test_all_samples_accessible(self, dataset):
        for i in range(len(dataset)):
            sample = dataset[i]
            assert "input_ids" in sample


# ═════════════════════════════════════════════════════════════════════════
# SECTION 3: Truncation Tests
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
class TestDPODatasetTruncation:
    """Tests for truncation behavior."""

    def test_right_truncation(self):
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _create_parquet(tmp_dir)
            config = _default_config(max_length=10, truncation="right")
            dataset = DPODataset(data_files=path, tokenizer=tokenizer, config=config)
            sample = dataset[0]
            assert sample["input_ids"].shape[0] <= 20

    def test_left_truncation(self):
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _create_parquet(tmp_dir)
            config = _default_config(max_length=10, truncation="left")
            dataset = DPODataset(data_files=path, tokenizer=tokenizer, config=config)
            sample = dataset[0]
            assert sample["input_ids"].shape[0] <= 20

    def test_invalid_truncation_raises(self):
        with pytest.raises(AssertionError):
            _default_config(max_length=10, truncation="invalid")


# ═════════════════════════════════════════════════════════════════════════
# SECTION 4: Invalid Sample Filtering Tests
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
class TestDPODatasetFiltering:
    """Tests for inspect_and_filter_invalid_samples behavior."""

    def _create_parquet_with_invalids(self, tmp_dir, num_valid=5):
        data = {
            "prompt": (
                [f"Valid prompt {i}" for i in range(num_valid)]
                + [None, "", "  ", "Good prompt", "Identical prompt"]
            ),
            "chosen": (
                [f"Good answer {i}" for i in range(num_valid)]
                + ["Some answer", "Some answer", "Some answer", None, "Same response"]
            ),
            "rejected": (
                [f"Bad answer {i}" for i in range(num_valid)]
                + ["Some bad", "Some bad", "Some bad", "Some bad", "Same response"]
            ),
        }
        df = pd.DataFrame(data)
        path = os.path.join(tmp_dir, "mixed.parquet")
        df.to_parquet(path)
        return path

    def test_remove_invalid_true_drops_samples(self):
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = self._create_parquet_with_invalids(tmp_dir, num_valid=5)
            config = _default_config(remove_invalid_samples=True)
            dataset = DPODataset(data_files=path, tokenizer=tokenizer, config=config)
            assert len(dataset) == 5

    def test_remove_invalid_false_keeps_all_samples(self):
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = self._create_parquet_with_invalids(tmp_dir, num_valid=5)
            config = _default_config(remove_invalid_samples=False)
            dataset = DPODataset(data_files=path, tokenizer=tokenizer, config=config)
            assert len(dataset) == 10

    def test_remove_invalid_false_logs_warning(self, caplog):
        import logging
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = self._create_parquet_with_invalids(tmp_dir, num_valid=5)
            config = _default_config(remove_invalid_samples=False)
            with caplog.at_level(logging.WARNING):
                DPODataset(data_files=path, tokenizer=tokenizer, config=config)
            assert "invalid samples" in caplog.text

    def test_all_valid_samples_no_warning(self, caplog):
        import logging
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _create_parquet(tmp_dir, num_samples=5)
            config = _default_config(remove_invalid_samples=False)
            with caplog.at_level(logging.WARNING):
                dataset = DPODataset(data_files=path, tokenizer=tokenizer, config=config)
            assert "invalid samples" not in caplog.text
            assert len(dataset) == 5

    def test_null_rejected_filtered(self):
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            data = {"prompt": ["Q1", "Q2", "Q3"], "chosen": ["A1", "A2", "A3"], "rejected": ["R1", None, "R3"]}
            df = pd.DataFrame(data)
            path = os.path.join(tmp_dir, "null_rej.parquet")
            df.to_parquet(path)
            config = _default_config(remove_invalid_samples=True)
            dataset = DPODataset(data_files=path, tokenizer=tokenizer, config=config)
            assert len(dataset) == 2

    def test_identical_pair_filtered(self):
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            data = {"prompt": ["Q1", "Q2"], "chosen": ["Same answer", "Different"], "rejected": ["Same answer", "Also different"]}
            df = pd.DataFrame(data)
            path = os.path.join(tmp_dir, "identical.parquet")
            df.to_parquet(path)
            config = _default_config(remove_invalid_samples=True)
            dataset = DPODataset(data_files=path, tokenizer=tokenizer, config=config)
            assert len(dataset) == 1

    def test_default_config_does_not_remove(self):
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            data = {"prompt": ["Q1", "Q2"], "chosen": ["A1", "A2"], "rejected": [None, "R2"]}
            df = pd.DataFrame(data)
            path = os.path.join(tmp_dir, "default.parquet")
            df.to_parquet(path)
            config = _default_config()
            dataset = DPODataset(data_files=path, tokenizer=tokenizer, config=config)
            assert len(dataset) == 2

    def test_none_values_with_chat_template_do_not_crash(self):
        """Samples with None fields should not crash when chat_template_key is set."""
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            data = {"prompt": ["Q1", None], "chosen": ["A1", "A2"], "rejected": ["R1", None]}
            df = pd.DataFrame(data)
            path = os.path.join(tmp_dir, "nulls.parquet")
            df.to_parquet(path)
            config = _default_config(chat_template_key="default", remove_invalid_samples=False)
            dataset = DPODataset(data_files=path, tokenizer=tokenizer, config=config)
            # Should not raise — None values are guarded before apply_chat_template
            for i in range(len(dataset)):
                sample = dataset[i]
                assert "input_ids" in sample


# ═════════════════════════════════════════════════════════════════════════
# SECTION 5: Collator Tests
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch not available")
class TestDPOCollator:
    """Tests for DPOTensorCollator with packed sequences."""

    def test_no_padding_produces_nested_tensors(self):
        import torch
        from verl.utils.dataset.dataset_utils import DPOTensorCollator
        collator = DPOTensorCollator("no_padding")
        batch = [
            {"input_ids": torch.arange(10), "loss_mask": torch.zeros(10), "boundary_offset": 5},
            {"input_ids": torch.arange(15), "loss_mask": torch.zeros(15), "boundary_offset": 8},
        ]
        result = collator(batch)
        assert result["input_ids"].is_nested
        assert result["loss_mask"].is_nested

    def test_batch_size_correct(self):
        import torch
        from verl.utils.dataset.dataset_utils import DPOTensorCollator
        collator = DPOTensorCollator("no_padding")
        batch = [
            {"input_ids": torch.arange(10), "loss_mask": torch.zeros(10), "boundary_offset": 5},
            {"input_ids": torch.arange(15), "loss_mask": torch.zeros(15), "boundary_offset": 8},
            {"input_ids": torch.arange(12), "loss_mask": torch.zeros(12), "boundary_offset": 6},
        ]
        result = collator(batch)
        assert len(result["input_ids"].offsets()) - 1 == 3

    def test_variable_lengths_preserved(self):
        import torch
        from verl.utils.dataset.dataset_utils import DPOTensorCollator
        collator = DPOTensorCollator("no_padding")
        batch = [
            {"input_ids": torch.ones(20), "loss_mask": torch.zeros(20), "boundary_offset": 10},
            {"input_ids": torch.ones(50), "loss_mask": torch.zeros(50), "boundary_offset": 25},
        ]
        result = collator(batch)
        seqlens = result["input_ids"].offsets().diff()
        assert seqlens[0].item() == 20
        assert seqlens[1].item() == 50

    def test_boundary_offsets_preserved(self):
        import torch
        from verl.utils.dataset.dataset_utils import DPOTensorCollator
        collator = DPOTensorCollator("no_padding")
        batch = [
            {"input_ids": torch.arange(10), "loss_mask": torch.zeros(10), "boundary_offset": 5},
            {"input_ids": torch.arange(15), "loss_mask": torch.zeros(15), "boundary_offset": 8},
        ]
        result = collator(batch)
        offsets = result["boundary_offset"]
        assert offsets is not None

    def test_dataloader_integration(self):
        import torch
        from torch.utils.data import DataLoader
        from verl.utils.dataset.dataset_utils import DPOTensorCollator
        from verl.utils.dataset.dpo_dataset import DPODataset
        tokenizer = _get_tokenizer()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _create_parquet(tmp_dir, num_samples=6)
            dataset = DPODataset(data_files=path, tokenizer=tokenizer, config=_default_config())
            collator = DPOTensorCollator("no_padding")
            dataloader = DataLoader(dataset, batch_size=3, collate_fn=collator)
            batch = next(iter(dataloader))
            assert batch["input_ids"].is_nested
            assert len(batch["input_ids"].offsets()) - 1 == 3


# ═════════════════════════════════════════════════════════════════════════
# SECTION 6: YAML Config Resolution Tests
# ═════════════════════════════════════════════════════════════════════════


class TestYamlResolutionPreservation:
    """Verify dpo_trainer.yaml resolves to expected configuration values."""

    @pytest.fixture(autouse=True)
    def load_yaml(self):
        yaml_path = os.path.join(_REPO_ROOT, "verl", "trainer", "config", "dpo_trainer.yaml")
        self.cfg = OmegaConf.load(yaml_path)

    def test_data_section_keys_preserved(self):
        expected_keys = {
            "train_batch_size", "micro_batch_size_per_gpu", "max_token_len_per_gpu",
            "use_dynamic_bsz", "dataloader_num_workers", "train_files", "val_files", "train_max_samples",
            "val_max_samples", "prompt_key", "chosen_key", "rejected_key",
            "pad_mode", "max_length", "truncation", "shuffle",
            "seed", "remove_invalid_samples", "custom_cls", "use_shm",
            "apply_chat_template_kwargs", "chat_template_key",
        }
        actual_keys = set(self.cfg.data.keys())
        assert expected_keys.issubset(actual_keys), (
            f"Missing data keys after load: {expected_keys - actual_keys}"
        )

    def test_data_scalar_values_preserved(self):
        data = self.cfg.data
        assert data.train_batch_size == 256
        assert data.micro_batch_size_per_gpu == 4
        assert data.max_token_len_per_gpu == 8192
        assert data.use_dynamic_bsz is True
        assert data.train_files is None
        assert data.val_files is None
        assert data.train_max_samples == -1
        assert data.val_max_samples == -1
        assert data.prompt_key == "prompt"
        assert data.chosen_key == "chosen"
        assert data.rejected_key == "rejected"
        assert data.pad_mode == "no_padding"
        assert data.max_length == 1024
        assert data.truncation == "error"
        assert data.shuffle is True
        assert data.seed == 42
        assert data.remove_invalid_samples is False
        assert data.use_shm is False
        assert data.chat_template_key is None

    def test_trainer_section_values_preserved(self):
        trainer = self.cfg.trainer
        assert trainer.beta == 0.01
        assert trainer.loss_type == "sigmoid"
        assert trainer.label_smoothing == 0.0
        assert trainer.reference_free is False
        assert trainer.project_name == "dpo-training"
        assert trainer.experiment_name == "test_run"
        assert trainer.total_epochs == 3
        assert trainer.save_freq == -1
        assert trainer.test_freq == -1
        assert trainer.nnodes == 1
        assert trainer.n_gpus_per_node == 1

    def test_checkpoint_section_values_preserved(self):
        ckpt = self.cfg.checkpoint
        assert ckpt.save_only_adapter_if_lora is True
        assert list(ckpt.save_contents) == ["model", "optimizer", "extra", "hf_model"]

    def test_engine_strategy_preserved(self):
        assert self.cfg.engine.strategy == "fsdp2"

    @given(st.just(True))
    @settings(max_examples=1)
    def test_yaml_reload_produces_identical_values(self, _):
        yaml_path = os.path.join(_REPO_ROOT, "verl", "trainer", "config", "dpo_trainer.yaml")
        cfg1 = OmegaConf.load(yaml_path)
        cfg2 = OmegaConf.load(yaml_path)
        assert OmegaConf.to_container(cfg1) == OmegaConf.to_container(cfg2)


# ═════════════════════════════════════════════════════════════════════════
# SECTION 7: Collator Subclass Defaults
# ═════════════════════════════════════════════════════════════════════════


class TestCollatorSubclassDefaults:
    """Verify DPOTensorCollator and SFTTensorCollator default pad_mode values."""

    @pytest.fixture(autouse=True)
    def load_module(self):
        self.mod = _load_dataset_utils()

    def test_dpo_tensor_collator_default_is_no_padding(self):
        collator = self.mod.DPOTensorCollator()
        assert collator.pad_mode == self.mod.DatasetPadMode.NO_PADDING

    def test_sft_tensor_collator_default_is_left_right(self):
        collator = self.mod.SFTTensorCollator()
        assert collator.pad_mode == self.mod.DatasetPadMode.LEFT_RIGHT

    def test_custom_tensor_collator_requires_pad_mode(self):
        """CustomTensorCollator has no default — pad_mode is required."""
        with pytest.raises(TypeError):
            self.mod.CustomTensorCollator()

    @given(st.just(True))
    @settings(max_examples=1)
    def test_collator_defaults_are_stable(self, _):
        for _ in range(5):
            assert self.mod.DPOTensorCollator().pad_mode == self.mod.DatasetPadMode.NO_PADDING
            assert self.mod.SFTTensorCollator().pad_mode == self.mod.DatasetPadMode.LEFT_RIGHT


# ═════════════════════════════════════════════════════════════════════════
# SECTION 8: Shuffle Determinism (property-based)
# ═════════════════════════════════════════════════════════════════════════


class TestShuffleDeterminism:
    """Verify shuffle_and_limit determinism and max_samples behavior."""

    @pytest.fixture(autouse=True)
    def load_module(self):
        self.mod = _load_dpo_dataset()

    def _make_dataframe(self, n_rows: int) -> pd.DataFrame:
        return pd.DataFrame({
            "prompt": [f"prompt_{i}" for i in range(n_rows)],
            "chosen": [f"chosen_{i}" for i in range(n_rows)],
            "rejected": [f"rejected_{i}" for i in range(n_rows)],
        })

    def test_same_seed_same_output(self):
        df = self._make_dataframe(20)
        r1 = self.mod.shuffle_and_limit(df.copy(), shuffle=True, seed=42)
        r2 = self.mod.shuffle_and_limit(df.copy(), shuffle=True, seed=42)
        pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))

    def test_different_seed_different_output(self):
        df = self._make_dataframe(20)
        r1 = self.mod.shuffle_and_limit(df.copy(), shuffle=True, seed=42)
        r2 = self.mod.shuffle_and_limit(df.copy(), shuffle=True, seed=99)
        assert not r1.reset_index(drop=True).equals(r2.reset_index(drop=True))

    def test_no_shuffle_preserves_order(self):
        df = self._make_dataframe(10)
        result = self.mod.shuffle_and_limit(df.copy(), shuffle=False, seed=42)
        pd.testing.assert_frame_equal(result.reset_index(drop=True), df.reset_index(drop=True))

    def test_max_samples_limits_rows(self):
        df = self._make_dataframe(20)
        result = self.mod.shuffle_and_limit(df.copy(), max_samples=5, shuffle=False)
        assert len(result) == 5

    @given(
        n_rows=st.integers(min_value=2, max_value=100),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    @settings(max_examples=20)
    def test_shuffle_determinism_property(self, n_rows, seed):
        df = self._make_dataframe(n_rows)
        r1 = self.mod.shuffle_and_limit(df.copy(), shuffle=True, seed=seed)
        r2 = self.mod.shuffle_and_limit(df.copy(), shuffle=True, seed=seed)
        pd.testing.assert_frame_equal(r1.reset_index(drop=True), r2.reset_index(drop=True))

    @given(
        n_rows=st.integers(min_value=5, max_value=100),
        max_samples=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=10)
    def test_max_samples_property(self, n_rows, max_samples):
        assume(max_samples < n_rows)
        df = self._make_dataframe(n_rows)
        result = self.mod.shuffle_and_limit(df.copy(), max_samples=max_samples, shuffle=False)
        assert len(result) == max_samples


# ═════════════════════════════════════════════════════════════════════════
# SECTION 9: File Loading Roundtrip (property-based)
# ═════════════════════════════════════════════════════════════════════════


class TestFileLoadingPreservation:
    """Verify load_data_file correctly roundtrips parquet and jsonl files."""

    @pytest.fixture(autouse=True)
    def load_module(self):
        self.mod = _load_dpo_dataset()

    def _sample_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "prompt": ["What is 2+2?", "Tell me a joke"],
            "chosen": ["4", "Why did the chicken cross the road?"],
            "rejected": ["5", "I don't know any jokes"],
        })

    def test_load_parquet_file(self):
        df = self._sample_dataframe()
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name)
            tmp_path = f.name
        try:
            loaded = self.mod.load_data_file(tmp_path)
            pd.testing.assert_frame_equal(loaded, df)
        finally:
            os.unlink(tmp_path)

    def test_load_jsonl_file(self):
        df = self._sample_dataframe()
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            df.to_json(f.name, orient="records", lines=True)
            tmp_path = f.name
        try:
            loaded = self.mod.load_data_file(tmp_path)
            pd.testing.assert_frame_equal(loaded, df)
        finally:
            os.unlink(tmp_path)

    @given(
        n_rows=st.integers(min_value=1, max_value=20),
        n_cols=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=10)
    def test_parquet_roundtrip_property(self, n_rows, n_cols):
        data = {f"col_{i}": [f"val_{i}_{j}" for j in range(n_rows)] for i in range(n_cols)}
        df = pd.DataFrame(data)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name)
            tmp_path = f.name
        try:
            loaded = self.mod.load_data_file(tmp_path)
            pd.testing.assert_frame_equal(loaded, df)
        finally:
            os.unlink(tmp_path)

    @given(
        n_rows=st.integers(min_value=1, max_value=20),
        n_cols=st.integers(min_value=1, max_value=5),
    )
    @settings(max_examples=10)
    def test_jsonl_roundtrip_property(self, n_rows, n_cols):
        data = {f"col_{i}": [f"val_{i}_{j}" for j in range(n_rows)] for i in range(n_cols)}
        df = pd.DataFrame(data)
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            df.to_json(f.name, orient="records", lines=True)
            tmp_path = f.name
        try:
            loaded = self.mod.load_data_file(tmp_path)
            pd.testing.assert_frame_equal(loaded, df)
        finally:
            os.unlink(tmp_path)

    def test_file_not_found_raises(self):
        """load_data_file should raise FileNotFoundError for nonexistent files."""
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            self.mod.load_data_file("/nonexistent/path/to/data.parquet")

    def test_unsupported_content_raises_value_error(self):
        """load_data_file should raise ValueError for files that aren't parquet or jsonl."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("this is plain text, not parquet or jsonl")
            tmp_path = f.name
        try:
            with pytest.raises(ValueError, match="Could not parse"):
                self.mod.load_data_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_unsupported_content_chains_parquet_error(self):
        """ValueError from unsupported content should chain the parquet error and include the JSONL error."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            f.write("this is plain text, not parquet or jsonl")
            tmp_path = f.name
        try:
            with pytest.raises(ValueError) as exc_info:
                self.mod.load_data_file(tmp_path)
            # Parquet error chained via __cause__
            assert exc_info.value.__cause__ is not None, (
                "ValueError should chain the original parquet error via 'from parquet_err'"
            )
            # JSONL error included in the message text
            assert "JSON Lines error:" in str(exc_info.value), (
                "ValueError message should include the JSON Lines parse error"
            )
        finally:
            os.unlink(tmp_path)

    def test_empty_file_raises(self):
        """load_data_file should raise on an empty file (not silently return empty DataFrame)."""
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            tmp_path = f.name
        try:
            with pytest.raises(Exception):
                self.mod.load_data_file(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_parquet_with_non_standard_extension(self):
        """A valid parquet file with a .dat extension should load via try-parse."""
        df = self._sample_dataframe()
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            df.to_parquet(f.name)
            tmp_path = f.name
        try:
            loaded = self.mod.load_data_file(tmp_path)
            pd.testing.assert_frame_equal(loaded, df)
        finally:
            os.unlink(tmp_path)


# ═════════════════════════════════════════════════════════════════════════
# SECTION 10: YAML & Code Quality Checks
# ═════════════════════════════════════════════════════════════════════════


class TestDpoYamlFormattingStandard:
    """Assert dpo_trainer.yaml follows the PPO formatting standard."""

    @pytest.fixture(autouse=True)
    def load_yaml(self):
        yaml_path = os.path.join(_REPO_ROOT, "verl", "trainer", "config", "dpo_trainer.yaml")
        with open(yaml_path) as f:
            self.yaml_content = f.read()
        self.yaml_lines = self.yaml_content.splitlines()

    def test_has_format_check_header(self):
        assert "# Format checks enforced on CI:" in self.yaml_content

    def test_no_inline_comments(self):
        inline_comment_pattern = re.compile(r"^[^#\n]*:\s*[^#\n]+\s+#")
        violations = []
        for i, line in enumerate(self.yaml_lines, 1):
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if inline_comment_pattern.match(line):
                violations.append(f"  Line {i}: {line.rstrip()}")
        assert not violations, (
            f"dpo_trainer.yaml has inline comments:\n" + "\n".join(violations)
        )

    def test_blank_lines_between_fields(self):
        field_lines = []
        for i, line in enumerate(self.yaml_lines):
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                field_lines.append((i, line))
        violations = []
        for idx in range(1, len(field_lines)):
            prev_lineno = field_lines[idx - 1][0]
            curr_lineno = field_lines[idx][0]
            if curr_lineno == prev_lineno + 1:
                prev_stripped = field_lines[idx - 1][1].strip()
                curr_stripped = field_lines[idx][1].strip()
                if prev_stripped.startswith("-") and curr_stripped.startswith("-"):
                    continue
                violations.append(
                    f"  Lines {prev_lineno + 1}-{curr_lineno + 1}: "
                    f"{field_lines[idx - 1][1].rstrip()} / {field_lines[idx][1].rstrip()}"
                )
        assert not violations, (
            f"dpo_trainer.yaml missing blank lines between fields:\n" + "\n".join(violations[:5])
        )


class TestCodeQualityConstants:
    """Assert named constants and idiomatic patterns in dpo_dataset.py."""

    @pytest.fixture(autouse=True)
    def load_source(self):
        self.source = _get_dpo_dataset_source()

    def test_dpo_supported_truncation_constant_exists(self):
        assert "DPO_SUPPORTED_TRUNCATION" in self.source

    def test_dpo_supported_file_types_constant_exists(self):
        assert "DPO_SUPPORTED_FILE_TYPES" in self.source

    def test_dpo_data_config_class_defined(self):
        assert "class DPODataConfig" in self.source

    def test_dpo_data_config_inherits_base_config(self):
        assert re.search(r"class DPODataConfig\([^)]*BaseConfig[^)]*\)", self.source)

    def test_shuffle_uses_dataframe_sample(self):
        match = re.search(
            r"(def shuffle_and_limit\b.*?)(?=\ndef |\nclass |\Z)", self.source, re.DOTALL
        )
        assert match is not None
        func_source = match.group(1)
        assert ".sample(" in func_source
        assert "np.random" not in func_source
        assert "rng.permutation" not in func_source

    def test_load_data_file_uses_try_parse(self):
        match = re.search(
            r"(def load_data_file\b.*?)(?=\ndef |\nclass |\Z)", self.source, re.DOTALL
        )
        assert match is not None
        func_source = match.group(1)
        assert ".endswith(" not in func_source
        assert "try:" in func_source and "read_parquet" in func_source

    def test_resolve_data_files_references_constant(self):
        match = re.search(
            r"(def resolve_data_files\b.*?)(?=\ndef |\nclass |\Z)", self.source, re.DOTALL
        )
        assert match is not None
        assert "DPO_SUPPORTED_FILE_TYPES" in match.group(1)
