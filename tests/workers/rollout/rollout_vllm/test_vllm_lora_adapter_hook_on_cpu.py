# Copyright 2026 Individual Contributor: Shen Ao
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from verl.utils.vllm.utils import TensorLoRARequest, VLLMHijack


def test_tensor_adapter_uses_model_converter(monkeypatch):
    from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

    captured = {}
    converter = lambda tensors, helper: (tensors, helper)

    class StubLoRAModel:
        @staticmethod
        def from_lora_tensors(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(rank=8, extra_vocab_size=0)

    monkeypatch.setattr(
        LRUCacheWorkerLoRAManager,
        "_load_adapter",
        LRUCacheWorkerLoRAManager._load_adapter,
    )
    VLLMHijack.hijack()
    manager = SimpleNamespace(
        _adapter_manager=SimpleNamespace(
            supported_lora_modules=["q_proj"],
            packed_modules_mapping={},
            model=SimpleNamespace(convert_lora_adapter=converter),
        ),
        lora_config=SimpleNamespace(
            max_lora_rank=8,
            lora_dtype=torch.float16,
            lora_extra_vocab_size=0,
        ),
        _lora_model_cls=StubLoRAModel,
        vocab_size=32,
    )
    request = TensorLoRARequest(
        "adapter",
        1,
        "unused",
        peft_config={"r": 4, "lora_alpha": 4, "target_modules": ["q_proj"]},
        lora_tensors={"q_proj.lora_A.weight": torch.ones(4, 4)},
    )

    result = LRUCacheWorkerLoRAManager._load_adapter(manager, request)

    assert result.rank == 8
    assert captured["adapter_converter"] is converter

    manager.lora_config.max_lora_rank = 4
    with pytest.raises(ValueError, match="Converted LoRA rank 8 exceeds"):
        LRUCacheWorkerLoRAManager._load_adapter(manager, request)
