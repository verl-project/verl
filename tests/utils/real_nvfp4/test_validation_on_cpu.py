# Copyright 2026 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
from types import SimpleNamespace

import pytest
import torch

from verl.utils.real_nvfp4.validation import dump_paired_log_probs


def _batch():
    return SimpleNamespace(
        batch={
            "response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
            "responses": torch.tensor([[10, 11, 0], [12, 0, 0]]),
            "prompts": torch.tensor([[1, 2], [3, 4]]),
            "old_log_probs": torch.tensor([[-1.0, -2.0, float("nan")], [-3.0, 0.0, 0.0]]),
            "rollout_log_probs": torch.tensor([[-1.1, -2.1, float("nan")], [-3.1, 0.0, 0.0]]),
        },
        meta_info={"temperature": 1.0},
    )


def test_paired_dump_masks_padding_and_does_not_mutate(tmp_path):
    batch = _batch()
    before = batch.batch["old_log_probs"].clone()
    summary = dump_paired_log_probs(batch, str(tmp_path), 1, "/fixed/model")
    assert summary["valid_tokens"] == 3
    assert summary["abs_mean"] == pytest.approx(0.1)
    assert summary["k3_mean"] == pytest.approx(0.0051709, abs=1e-6)
    assert summary["checkpoint_position"] == "initial_before_first_update"
    saved = torch.load(tmp_path / "step_1.pt", weights_only=True)
    assert saved["response_tokens"].tolist() == [10, 11, 12]
    torch.testing.assert_close(before, batch.batch["old_log_probs"], equal_nan=True)
    with pytest.raises(FileExistsError):
        dump_paired_log_probs(batch, str(tmp_path), 1, "/fixed/model")


@pytest.mark.parametrize("bad", ["nan", "empty", "shape"])
def test_paired_dump_refuses_invalid_evidence(tmp_path, bad):
    batch = _batch()
    if bad == "nan":
        batch.batch["old_log_probs"][0, 0] = float("nan")
    elif bad == "empty":
        batch.batch["response_mask"].zero_()
    else:
        batch.batch["rollout_log_probs"] = torch.zeros(1, 1)
    with pytest.raises(ValueError):
        dump_paired_log_probs(batch, str(tmp_path), 1, "/fixed/model")
    assert not list(tmp_path.iterdir())
