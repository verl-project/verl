# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import pytest

from verl.model_merger.base_model_merger import _default_hf_model_config_path


@pytest.mark.parametrize("root_metadata", [False, True])
def test_megatron_config_always_uses_v2_layout(tmp_path, root_metadata):
    if root_metadata:
        (tmp_path / ".metadata").touch()
    expected = tmp_path / "model" / "huggingface"
    assert _default_hf_model_config_path("megatron", str(tmp_path)) == str(expected)
    assert not expected.exists(), "Resolving a checkpoint must not create directories"


def test_fsdp_config_layout_unchanged(tmp_path):
    assert _default_hf_model_config_path("fsdp", str(tmp_path)) == str(tmp_path / "huggingface")
