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
"""VeOmni engine: the trainer-side rollout weight quantizer wraps the exported weight stream.

VeOmni is optional in CPU CI, so its import surfaces are stubbed as in the other VeOmni engine tests.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

for _mod in (
    "veomni",
    "veomni.arguments",
    "veomni.distributed",
    "veomni.distributed.torch_parallelize",
    "veomni.models",
    "veomni.models.auto",
    "veomni.models.checkpoint_tensor_loading",
    "veomni.optim",
    "veomni.utils",
    "veomni.utils.seqlen_pos_transform_utils",
):
    sys.modules.setdefault(_mod, MagicMock())

from verl.workers.engine.veomni import transformer_impl  # noqa: E402


def test_rollout_weight_quantizer_wraps_the_exported_stream():
    # Only the attribute the methods under test read: __init__ needs a distributed mesh.
    engine = object.__new__(transformer_impl.VeOmniEngine)
    engine._rollout_weight_quantizer = None
    stream = iter([("w", 1)])
    assert engine._quantize_for_rollout(stream) is stream

    engine.set_rollout_weight_quantizer(lambda weights: ((f"{name}.q", value) for name, value in weights))
    assert list(engine._quantize_for_rollout(iter([("w", 1), ("b", 2)]))) == [("w.q", 1), ("b.q", 2)]


def test_checkpoint_export_skips_the_rollout_weight_quantizer(monkeypatch):
    engine = object.__new__(transformer_impl.VeOmniEngine)
    engine._rollout_weight_quantizer = None
    engine._uses_fsdp2_cpu_offload_policy = True  # CPUOffloadPolicy owns placement: no model move
    engine.module = object()
    converter = SimpleNamespace(export_weights=lambda module: iter([("w", 1)]))
    monkeypatch.setattr(transformer_impl, "get_checkpoint_tensor_converter", lambda module: converter)
    engine.set_rollout_weight_quantizer(lambda weights: ((f"{name}.q", value) for name, value in weights))

    # The checkpoint manager's 'hf_model' export keeps the trainer's weights; only the sync is quantized.
    exported, _ = engine._get_hf_export_per_tensor_param()
    assert list(exported) == [("w", 1)]
    synced, _ = engine.get_per_tensor_param()
    assert list(synced) == [("w.q", 1)]


def test_initialize_wires_the_checkpoint_manager_to_the_unquantized_export(monkeypatch):
    engine = object.__new__(transformer_impl.VeOmniEngine)
    engine.__dict__.update(
        _rollout_weight_quantizer=None,
        enable_routing_replay=False,
        engine_config=SimpleNamespace(mixed_precision=False),
        model_config=SimpleNamespace(get_processor=lambda: None, trust_remote_code=False),
        checkpoint_config=None,
        module=None,
        optimizer=None,
        lr_scheduler=None,
        _is_offload_param=False,
        _is_offload_optimizer=False,
    )
    monkeypatch.setattr(transformer_impl.VeOmniEngine, "_build_model_optimizer", lambda self: None)
    monkeypatch.setattr(transformer_impl.VeOmniEngine, "to", lambda self, **kwargs: None)
    monkeypatch.setattr(transformer_impl, "log_gpu_memory_usage", lambda *args, **kwargs: None)
    monkeypatch.setattr(transformer_impl, "FSDPCheckpointManager", lambda **kwargs: SimpleNamespace(**kwargs))

    engine.initialize()

    assert engine.checkpoint_manager.per_tensor_param_fn == engine._get_hf_export_per_tensor_param
