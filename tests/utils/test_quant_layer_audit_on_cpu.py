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
"""CPU tests for the train/rollout quantized-layer audit (Megatron and vLLM modules faked)."""

import logging
from types import SimpleNamespace

import pytest
import torch

from verl.utils.quant_layer_audit import (
    FP8_WORKSPACE_SEEN_ATTR,
    TrainExportAudit,
    audit_train_vs_export,
    collect_train_fp8_report,
    exported_weight,
    fp8_recipe_of,
    remember_fp8_workspaces,
    trace,
    unjudged_weights,
)

N_LAYERS = 4
_AUDIT_LOGGER = "verl.utils.quant_layer_audit"


@pytest.fixture(autouse=True)
def _audit_logs_reach_caplog(caplog):
    # The audit logger does not propagate (it writes to stderr on its own); attach caplog's handler to it.
    log = logging.getLogger(_AUDIT_LOGGER)
    log.addHandler(caplog.handler)
    level = log.level
    log.setLevel(logging.INFO)
    yield
    log.setLevel(level)
    log.removeHandler(caplog.handler)


class _TELinear(torch.nn.Module):
    """A TE linear as it looks after the first FP8 forward: it holds a weight workspace."""

    def __init__(self, quantized: bool):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(4, 64), requires_grad=False)
        if quantized:
            self._fp8_workspaces = {"weight": object()}


class TransformerLayer(torch.nn.Module):  # name matters: the audit looks for *TransformerLayer*
    def __init__(self, layer_number: int, quantized: bool):
        super().__init__()
        self.layer_number = layer_number  # mcore: 1-based, global across PP
        self.self_attention = torch.nn.Module()
        self.self_attention.linear_qkv = _TELinear(quantized)
        self.self_attention.linear_proj = _TELinear(quantized)
        self.mlp = torch.nn.Module()
        self.mlp.linear_fc1 = _TELinear(quantized)
        self.mlp.linear_fc2 = _TELinear(quantized)


class _Chunk(torch.nn.Module):
    def __init__(self, quantized_layers, n=N_LAYERS):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 4)
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList([TransformerLayer(i + 1, i in quantized_layers) for i in range(n)])
        self.output_layer = torch.nn.Linear(4, 10)  # mcore ColumnParallelLinear: never TE


def _hf_names(n=N_LAYERS, router=None, experts=None):
    """Qwen/Llama-style dense names; ``router`` adds a router weight per layer, ``experts`` a
    (prefix, [leaf names]) pair adding two experts per layer, e.g. ("block_sparse_moe.experts", ["w1", "w2", "w3"])."""
    names = ["model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"]
    for i in range(n):
        p = f"model.layers.{i}."
        names += [
            p + "self_attn.q_proj.weight",
            p + "self_attn.k_proj.weight",
            p + "self_attn.v_proj.weight",
            p + "self_attn.o_proj.weight",
            p + "self_attn.q_norm.weight",
            p + "mlp.gate_proj.weight",
            p + "mlp.up_proj.weight",
            p + "mlp.down_proj.weight",
            p + "input_layernorm.weight",
            p + "post_attention_layernorm.weight",
        ]
        if router:
            names.append(p + router)
        if experts:
            prefix, leaves = experts
            for e in range(2):
                names += [f"{p}{prefix}.{e}.{leaf}.weight" for leaf in leaves]
    return names


def _preset_quantizes(name):
    """What the MXFP8 preset with no extra ``ignore`` sends quantized: the projections and the experts."""
    return name.endswith("_proj.weight") or ".experts." in name


def _export(names, quantizes=_preset_quantizes):
    """The weight stream after the rollout quantizer: an FP8 weight plus its scale, or the BF16 weight."""
    stream = []
    for name in names:
        if quantizes(name):
            stream.append((name, torch.zeros(4, 64, dtype=torch.float8_e4m3fn)))
            stream.append((name + "_scale", torch.zeros(4, 2, dtype=torch.uint8)))
        else:
            stream.append((name, torch.zeros(4, 64, dtype=torch.bfloat16)))
    return stream


def _recorded(names, quantizes=_preset_quantizes):
    auditor = TrainExportAudit("warn")
    list(auditor.wrap_export(iter(_export(names, quantizes))))
    return auditor.exported


def _problem_names(problems):
    return {p.split(":")[0] for p in problems}


def test_fp8_recipe_of_reads_the_megatron_overrides():
    def engine_config(**overrides):
        return SimpleNamespace(override_transformer_config=overrides)

    assert fp8_recipe_of(None) is None
    assert fp8_recipe_of(SimpleNamespace()) is None
    assert fp8_recipe_of(engine_config()) is None
    assert fp8_recipe_of(engine_config(fp8=None, fp8_recipe="mxfp8")) is None
    assert fp8_recipe_of(engine_config(fp8="e4m3", fp8_recipe="mxfp8")) == "mxfp8"
    assert fp8_recipe_of(engine_config(fp8="hybrid", fp8_recipe="Blockwise")) == "blockwise"
    # Megatron's TransformerConfig defaults to the delayed recipe.
    assert fp8_recipe_of(engine_config(fp8="e4m3")) == "delayed"


def test_report_reads_layer_numbers_and_workspaces():
    r = collect_train_fp8_report([_Chunk(quantized_layers={1, 2})])
    assert r.seen_layers == {0, 1, 2, 3} and r.quantized_layers == {1, 2}
    assert r.outside_layers_quantized == [] and r.active


def test_exported_weight_reads_the_format_and_skips_scales():
    assert exported_weight("a.weight", torch.zeros(1, dtype=torch.float8_e4m3fn)) == ("a.weight", True, "float8_e4m3fn")
    assert exported_weight("a.weight", torch.zeros(1, dtype=torch.bfloat16)) == ("a.weight", False, "bfloat16")
    assert exported_weight("a.weight_packed", torch.zeros(1, dtype=torch.uint8)) == ("a.weight", True, "packed FP4")
    assert exported_weight("a.weight_scale", torch.zeros(1, dtype=torch.uint8)) is None
    assert exported_weight("a.weight_scale_inv", torch.zeros(1, dtype=torch.uint8)) is None
    fused = exported_weight("model.layers.0.mlp.experts.gate_up_proj", torch.zeros(1, dtype=torch.bfloat16))
    assert fused == ("model.layers.0.mlp.experts.gate_up_proj", False, "bfloat16")


def test_export_matching_training_is_consistent():
    r = collect_train_fp8_report([_Chunk(quantized_layers=set(range(N_LAYERS)))])
    names = _hf_names(router="mlp.gate.weight", experts=("mlp.experts", ["gate_proj", "up_proj", "down_proj"]))
    problems, by_design = audit_train_vs_export(r, _recorded(names))
    assert problems == []
    assert by_design == 2 + N_LAYERS  # embed_tokens and lm_head, and one router per layer


def test_first_last_layers_bf16_without_a_rollout_ignore_is_reported():
    # training kept layers 0 and 3 in bf16; the export quantizes every decoder layer
    r = collect_train_fp8_report([_Chunk(quantized_layers={1, 2})])
    problems, _ = audit_train_vs_export(r, _recorded(_hf_names()))
    assert _problem_names(problems) == {
        n for n in _hf_names() if n.startswith(("model.layers.0.", "model.layers.3.")) and n.endswith("_proj.weight")
    }
    assert all("the export sends it in float8_e4m3fn, training ran it in high precision" in p for p in problems)

    # ... and ignoring those layers on the rollout makes it consistent again
    def ignoring_first_last(name):
        return _preset_quantizes(name) and not name.startswith(("model.layers.0.", "model.layers.3."))

    assert audit_train_vs_export(r, _recorded(_hf_names(), ignoring_first_last))[0] == []


def test_training_fp8_but_exported_in_bf16_is_reported():
    r = collect_train_fp8_report([_Chunk(quantized_layers=set(range(N_LAYERS)))])

    def ignoring_one(name):
        return _preset_quantizes(name) and name != "model.layers.2.mlp.down_proj.weight"

    problems, _ = audit_train_vs_export(r, _recorded(_hf_names(), ignoring_one))
    assert problems == ["model.layers.2.mlp.down_proj.weight: training ran it in FP8, the export sends it in bfloat16"]


def test_a_router_the_export_quantizes_is_reported():
    # Training never runs a router in FP8, whatever the checkpoint calls it. Mixtral's block_sparse_moe.gate is
    # outside the exporter's default ignore list, so an export that quantizes it is a reported disagreement.
    r = collect_train_fp8_report([_Chunk(quantized_layers=set(range(N_LAYERS)))])
    names = _hf_names(router="block_sparse_moe.gate.weight")

    def with_router(name):
        return _preset_quantizes(name) or name.endswith("block_sparse_moe.gate.weight")

    problems, _ = audit_train_vs_export(r, _recorded(names, with_router))
    assert _problem_names(problems) == {f"model.layers.{i}.block_sparse_moe.gate.weight" for i in range(N_LAYERS)}


def test_designed_bf16_modules_are_not_reported():
    # MTP layers and vision towers stay BF16 on the rollout by design; an MTP name also carries "layers.N",
    # which without the exporter's ignore list would be judged against decoder layer N.
    r = collect_train_fp8_report([_Chunk(quantized_layers=set(range(N_LAYERS)))])
    names = _hf_names() + ["mtp.layers.0.eh_proj.weight", "model.visual.blocks.0.attn.qkv.weight"]

    def preset(name):
        return _preset_quantizes(name) and not name.startswith(("mtp.", "model.visual."))

    problems, by_design = audit_train_vs_export(r, _recorded(names, preset))
    assert problems == [] and by_design == 4


def test_packed_fp4_weights_count_as_quantized():
    r = collect_train_fp8_report([_Chunk(quantized_layers=set(range(N_LAYERS)))])
    stream = [
        (n + "_packed", torch.zeros(4, 32, dtype=torch.uint8)) if _preset_quantizes(n) else (n, torch.zeros(4))
        for n in _hf_names()
    ]
    auditor = TrainExportAudit("warn")
    list(auditor.wrap_export(iter(stream)))
    assert auditor.exported["model.layers.1.mlp.down_proj.weight"] == (True, "packed FP4")
    assert audit_train_vs_export(r, auditor.exported)[0] == []


def test_pipeline_rank_only_judges_its_own_layers():
    # this rank owns layers 2-3 only (PP stage 2): layers 0-1 are not judged, whatever the export sends
    chunk = _Chunk(quantized_layers=set(), n=0)
    chunk.decoder.layers = torch.nn.ModuleList([TransformerLayer(3, True), TransformerLayer(4, True)])
    r = collect_train_fp8_report([chunk])
    assert r.seen_layers == {2, 3}

    def broken(name):
        return _preset_quantizes(name) and not name.startswith(("model.layers.0.", "model.layers.3.mlp."))

    problems, _ = audit_train_vs_export(r, _recorded(_hf_names(), broken))
    assert _problem_names(problems) == {f"model.layers.3.mlp.{p}.weight" for p in ("gate_proj", "up_proj", "down_proj")}


def test_auditor_waits_for_the_first_training_step_then_runs_once(caplog):
    auditor = TrainExportAudit("warn")
    weights = _export(_hf_names())

    # sync 0: no training step yet -> no workspaces -> not run, keeps waiting
    assert list(auditor.wrap_export(iter(weights))) == weights
    assert auditor.compare_with_training([_Chunk(quantized_layers=set())]) is None and not auditor.finished
    # sync 1: workspaces present (layers 0 and 3 kept bf16 in training) -> runs, warns, marks done
    list(auditor.wrap_export(iter(weights)))
    problems = auditor.compare_with_training([_Chunk(quantized_layers={1, 2})])
    assert len(problems) == 2 * 7 and auditor.finished  # 7 projection weights per bf16 layer
    assert "training and the exported weights disagree on 14 weight(s)" in caplog.text
    # later syncs: pass-through, no re-run
    assert list(auditor.wrap_export(iter(weights))) == weights and auditor.compare_with_training([]) is None


def test_auditor_raise_mode_and_disabled_mode():
    auditor = TrainExportAudit("raise")
    list(auditor.wrap_export(iter(_export(_hf_names()))))
    with pytest.raises(RuntimeError, match="disagree"):
        auditor.compare_with_training([_Chunk(quantized_layers={1, 2})])
    off = TrainExportAudit("0")
    assert not off.enabled and off.compare_with_training([_Chunk(quantized_layers={0})]) is None
    assert not TrainExportAudit("warn", enabled=False).enabled


def test_create_arms_only_for_a_te_fp8_trainer_with_a_quantized_rollout(monkeypatch, caplog):
    monkeypatch.delenv("VERL_QUANT_LAYER_AUDIT", raising=False)
    armed = TrainExportAudit.create(rollout_quantized=True, trainer_fp8_recipe="mxfp8", rollout_name="vllm")
    assert armed.enabled and armed.engine == "vllm" and armed.mode == "warn"
    assert "quantized-layer audit, training vs export: armed" in caplog.text
    caplog.clear()
    for quantized, recipe in ((True, None), (False, "mxfp8")):
        assert not TrainExportAudit.create(
            rollout_quantized=quantized, trainer_fp8_recipe=recipe, rollout_name="vllm"
        ).enabled
    assert "not armed" in caplog.text and not any(r.levelno >= logging.WARNING for r in caplog.records)
    monkeypatch.setenv("VERL_QUANT_LAYER_AUDIT", "0")
    assert not TrainExportAudit.create(rollout_quantized=True, trainer_fp8_recipe="mxfp8", rollout_name="vllm").enabled


def test_record_runs_the_comparison_when_the_weight_stream_ends():
    # Wrapping the producer's stream is enough: the verdict does not depend on which sync route consumes it.
    aud = TrainExportAudit("warn", engine="sglang")
    weights = _export(_hf_names())
    chunk = _Chunk(quantized_layers={1, 2})  # layers 0 and 3 stayed bf16 in training
    stream = aud.wrap_export(iter(weights), training_modules=[chunk])
    assert not aud.finished  # nothing happens until the stream is consumed
    assert list(stream) == weights and aud.finished  # passed through untouched, and the verdict was reached

    # raise mode propagates out of the consumer, which is what stops a mismatched run
    strict = TrainExportAudit("raise", engine="sglang")
    with pytest.raises(RuntimeError, match="disagree"):
        list(strict.wrap_export(iter(weights), training_modules=[chunk]))

    # disabled auditor: pure pass-through, no comparison
    off = TrainExportAudit("warn", enabled=False)
    assert list(off.wrap_export(iter(weights), training_modules=[chunk])) == weights and not off.finished


def test_init_sync_then_training_sync_produces_a_verdict():
    # The pre-training-step sync has no FP8 workspace yet; it must not mark the audit done, or the first
    # real training-step sync would be skipped.
    aud = TrainExportAudit("warn", engine="sglang")
    weights = _export(_hf_names())
    empty = [_Chunk(quantized_layers=set())]
    list(aud.wrap_export(iter(weights), training_modules=empty))
    assert not aud.finished and aud.syncs_seen == 1
    assert (
        aud.compare_with_training(empty) is None and not aud.finished
    )  # a second run() on the same sync: still waiting
    list(aud.wrap_export(iter(weights), training_modules=[_Chunk(quantized_layers={1, 2})]))
    assert aud.finished


def test_two_inactive_syncs_give_up_and_warn(caplog):
    aud = TrainExportAudit("warn", engine="sglang")
    weights = _export(_hf_names())
    empty = [_Chunk(quantized_layers=set())]
    list(aud.wrap_export(iter(weights), training_modules=empty))
    assert not aud.finished and "produced no verdict" not in caplog.text
    list(aud.wrap_export(iter(weights), training_modules=empty))
    assert aud.finished and "produced no verdict" in caplog.text and "0 of 4 decoder layers" in caplog.text


def test_auditor_names_disable_parameter_transpose_cache_as_the_cause(caplog):
    chunk = _Chunk(quantized_layers=set())
    for m in chunk.modules():
        if isinstance(m, _TELinear):
            m.disable_parameter_transpose_cache = True
    auditor = TrainExportAudit("warn")
    list(auditor.wrap_export(iter(_export(_hf_names()))))
    assert auditor.compare_with_training([chunk]) is None and auditor.finished
    assert "disable_parameter_transpose_cache=True" in caplog.text and "produced no verdict" in caplog.text


def _offload_like_verl(chunk):
    """What offload_megatron_model_to_cpu does to TE caches: mark, then drop."""
    for sub in chunk.modules():
        ws = getattr(sub, "_fp8_workspaces", None)
        if isinstance(ws, dict) and ws:
            assert remember_fp8_workspaces(sub)
            ws.clear()


def test_audit_survives_param_offload_clearing_the_te_workspaces():
    chunk = _Chunk(quantized_layers={1, 2})
    _offload_like_verl(chunk)
    assert all(not getattr(m, "_fp8_workspaces", {}) for m in chunk.modules())  # caches really gone
    r = collect_train_fp8_report([chunk])
    assert r.quantized_layers == {1, 2} and r.active
    assert not remember_fp8_workspaces(chunk.decoder.layers[0].mlp.linear_fc1)  # bf16 layer: nothing to mark
    assert not hasattr(chunk.decoder.layers[0].mlp.linear_fc1, FP8_WORKSPACE_SEEN_ATTR)


def test_trace_is_silent_unless_enabled(monkeypatch, caplog):
    monkeypatch.setenv("VERL_QUANT_LAYER_AUDIT_TRACE", "0")
    trace("should-not-appear")
    assert "should-not-appear" not in caplog.text
    monkeypatch.setenv("VERL_QUANT_LAYER_AUDIT_TRACE", "1")
    aud = TrainExportAudit("warn", engine="sglang")
    list(aud.wrap_export(iter(_export(_hf_names())), training_modules=[_Chunk(quantized_layers={1, 2})]))
    for where in ("trace: wrap_export", "wrap_export.stream_end", "trace: compare_with_training"):
        assert where in caplog.text, where


def test_a_non_gemm_weight_in_an_fp8_layer_is_left_unjudged_not_reported(caplog):
    # Qwen3-Next's linear attention: the exporter keeps the convolution in BF16, as training does, inside a
    # layer whose projections ran FP8 GEMMs. The layer-level trace cannot tell, so the audit does not judge it.
    r = collect_train_fp8_report([_Chunk(quantized_layers=set(range(N_LAYERS)))])
    conv = "model.layers.0.linear_attn.conv1d.weight"
    names = _hf_names() + [conv]
    exported = _recorded(names)
    assert exported[conv] == (False, "bfloat16")
    assert audit_train_vs_export(r, exported)[0] == [] and unjudged_weights(exported) == [conv]
    auditor = TrainExportAudit("warn")
    list(auditor.wrap_export(iter(_export(names))))
    assert auditor.compare_with_training([_Chunk(quantized_layers=set(range(N_LAYERS)))]) == []
    assert "1 decoder-layer weights of a kind the audit does not judge, e.g. " + conv in caplog.text


@pytest.mark.parametrize(
    "leaf",
    [
        "self_attn.qkv_proj.weight",
        "self_attn.q_a_proj.weight",  # MLA
        "self_attn.kv_b_proj.weight",
        "mlp.gate_up_proj.weight",
        "linear_attn.in_proj_qkvz.weight",  # Qwen3-Next linear attention
        "linear_attn.out_proj.weight",
        "mixer.in_proj.weight",  # SSM
        "block_sparse_moe.experts.0.w1.weight",
        "mlp.experts.gate_up_proj",
    ],
)
def test_the_known_gemm_projections_are_judged(leaf):
    r = collect_train_fp8_report([_Chunk(quantized_layers={1})])
    name = "model.layers.0." + leaf
    problems, _ = audit_train_vs_export(r, {name: (True, "float8_e4m3fn")})  # layer 0 ran in BF16
    assert _problem_names(problems) == {name}
    assert unjudged_weights({name: (True, "float8_e4m3fn")}) == []
