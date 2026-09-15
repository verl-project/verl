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
"""CPU tests for the train/rollout quantized-layer audit (Megatron modules faked)."""

import pytest
import torch

from verl.utils.quant_layer_audit import (
    FP8_WORKSPACE_SEEN_ATTR,
    ROLLOUT_LABELS,
    QuantLayerAuditor,
    audit_layer_sets,
    collect_train_fp8_report,
    remember_fp8_workspaces,
    vllm_rollout_predicate,
)
from verl.utils.sglang.sglang_mxfp8_utils import SGLangMXFP8QuantizerHelper, build_sglang_mxfp8_quant_config

N_LAYERS = 4


class _TELinear(torch.nn.Module):
    """A TE linear as it looks after the first fp8 forward: it holds a weight workspace."""

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


def _sglang_pred(hf_config=None):
    return SGLangMXFP8QuantizerHelper(build_sglang_mxfp8_quant_config(hf_config)).should_quantize_param


def test_report_reads_layer_numbers_and_workspaces():
    r = collect_train_fp8_report([_Chunk(quantized_layers={1, 2})])
    assert r.seen_layers == {0, 1, 2, 3} and r.quantized_layers == {1, 2}
    assert r.outside_layers_quantized == [] and r.active


def test_dense_default_config_is_consistent_on_both_engines():
    r = collect_train_fp8_report([_Chunk(quantized_layers=set(range(N_LAYERS)))])
    names = _hf_names()
    assert audit_layer_sets(r, names, _sglang_pred()) == []
    vllm = vllm_rollout_predicate(N_LAYERS, ("lm_head", "model.embed_tokens"))
    assert audit_layer_sets(r, names, vllm) == []


def test_first_last_layers_bf16_without_rollout_regex_is_reported():
    # training kept layers 0 and 3 in bf16; rollout quantizes every decoder layer
    r = collect_train_fp8_report([_Chunk(quantized_layers={1, 2})])
    problems = audit_layer_sets(r, _hf_names(), _sglang_pred())
    flagged = {p.split(":")[0] for p in problems}
    assert flagged == {
        n for n in _hf_names() if n.startswith(("model.layers.0.", "model.layers.3.")) and "norm" not in n
    }
    assert all("rule quantizes it" in p for p in problems)
    # ... and the matching rollout regex makes it consistent again
    pred = _sglang_pred({"quantization_config": {"ignored_layers": ["re:model\\.layers\\.(0|3)\\..*"]}})
    assert audit_layer_sets(r, _hf_names(), pred) == []


def test_routers_are_excluded_on_both_sides_whatever_the_name():
    # The SGLang sync rule is an include-whitelist (q_proj/.../mlp) plus excludes: neither a Qwen-style
    # nor a Mixtral-style router name passes it, and training never quantizes a router either.
    r = collect_train_fp8_report([_Chunk(quantized_layers=set(range(N_LAYERS)))])
    assert audit_layer_sets(r, _hf_names(router="mlp.gate.weight"), _sglang_pred()) == []
    assert audit_layer_sets(r, _hf_names(router="block_sparse_moe.gate.weight"), _sglang_pred()) == []


def test_mixtral_expert_names_miss_the_sglang_sync_whitelist():
    # Mixtral experts are w1/w2/w3 under block_sparse_moe: the sync-side whitelist does not match them,
    # so the sync would ship them in bf16 although training ran them in fp8 (and the SGLang engine,
    # which built fused fp8 experts, would expect fp8 + scales). The audit reports every expert weight.
    r = collect_train_fp8_report([_Chunk(quantized_layers=set(range(N_LAYERS)))])
    names = _hf_names(router="block_sparse_moe.gate.weight", experts=("block_sparse_moe.experts", ["w1", "w2", "w3"]))
    problems = audit_layer_sets(r, names, _sglang_pred())
    assert len(problems) == N_LAYERS * 2 * 3
    assert all("block_sparse_moe.experts" in p and "rule does not quantize it" in p for p in problems)
    # Qwen-MoE expert names (gate_proj/up_proj/down_proj under mlp.experts) pass the whitelist: consistent
    qwen = _hf_names(router="mlp.gate.weight", experts=("mlp.experts", ["gate_proj", "up_proj", "down_proj"]))
    assert audit_layer_sets(r, qwen, _sglang_pred()) == []


def test_shared_expert_gate_is_excluded_by_sync_and_would_be_reported_otherwise():
    # Qwen2-MoE's shared_expert_gate is a plain linear on both the training side (mcore, torch) and the
    # SGLang engine (ReplicatedLinear(quant_config=None)). The real sync rule now excludes it ...
    r = collect_train_fp8_report([_Chunk(quantized_layers=set(range(N_LAYERS)))])
    names = _hf_names(router="mlp.shared_expert_gate.weight")
    assert audit_layer_sets(r, names, _sglang_pred()) == []

    # ... and a rule that quantized it (the pre-fix "mlp" whitelist) is what the audit is there to catch.
    def naive(n):  # every linear-looking weight, "mlp" catching the shared-expert gate as well
        return n.endswith(".weight") and ("proj" in n or "mlp" in n) and "norm" not in n

    problems = audit_layer_sets(r, names, naive)
    assert [p.split(":")[0] for p in problems] == [
        f"model.layers.{i}.mlp.shared_expert_gate.weight" for i in range(N_LAYERS)
    ]
    assert all("rule quantizes it" in p for p in problems)


def test_lm_head_left_quantized_on_rollout_is_reported():
    r = collect_train_fp8_report([_Chunk(quantized_layers=set(range(N_LAYERS)))])
    vllm_without_keep = vllm_rollout_predicate(N_LAYERS, ())  # the pre-cf69c645 blacklist
    problems = audit_layer_sets(r, _hf_names(), vllm_without_keep)
    assert [p.split(":")[0] for p in problems] == ["lm_head.weight"]


def test_training_fp8_but_rollout_unquantized_layer_is_reported():
    r = collect_train_fp8_report([_Chunk(quantized_layers=set(range(N_LAYERS)))])
    pred = _sglang_pred({"quantization_config": {"ignored_layers": ["model.layers.2.mlp.down_proj"]}})
    problems = audit_layer_sets(r, _hf_names(), pred)
    assert problems == [
        "model.layers.2.mlp.down_proj.weight: training ran it in fp8, the rollout-side rule does not quantize it"
    ]


def test_pipeline_rank_only_judges_its_own_layers():
    # this rank owns layers 2-3 only (PP stage 2); layers 0-1 are not judged, lm_head still is
    chunk = _Chunk(quantized_layers=set(), n=0)
    chunk.decoder.layers = torch.nn.ModuleList([TransformerLayer(3, True), TransformerLayer(4, True)])
    r = collect_train_fp8_report([chunk])
    assert r.seen_layers == {2, 3}
    problems = audit_layer_sets(r, _hf_names(), vllm_rollout_predicate(N_LAYERS, ()))
    assert [p.split(":")[0] for p in problems] == ["lm_head.weight"]


def test_auditor_waits_for_the_first_training_step_then_runs_once(monkeypatch, caplog):
    monkeypatch.setenv("VERL_QUANT_LAYER_AUDIT", "warn")
    auditor = QuantLayerAuditor(_sglang_pred(), "warn")
    names = _hf_names()
    weights = [(n, torch.zeros(1)) for n in names]

    # sync 0: no training step yet -> no workspaces -> not run, keeps waiting
    assert list(auditor.record(iter(weights))) == weights
    assert auditor.run([_Chunk(quantized_layers=set())]) is None and not auditor.done
    # sync 1: workspaces present (layers 0 and 3 kept bf16 on training) -> runs, warns, marks done
    list(auditor.record(iter(weights)))
    problems = auditor.run([_Chunk(quantized_layers={1, 2})])
    assert len(problems) == 2 * 7 and auditor.done  # 7 linear weights per bf16 layer
    assert "training and the rollout-side rule disagree on 14 parameter(s)" in caplog.text
    # later syncs: pass-through, no re-run
    assert list(auditor.record(iter(weights))) == weights and auditor.run([]) is None


def test_auditor_raise_mode_and_disabled_mode():
    auditor = QuantLayerAuditor(_sglang_pred(), "raise")
    list(auditor.record((n, torch.zeros(1)) for n in _hf_names()))
    with pytest.raises(RuntimeError, match="disagree"):
        auditor.run([_Chunk(quantized_layers={1, 2})])
    off = QuantLayerAuditor(_sglang_pred(), "0")
    assert not off.enabled and off.run([_Chunk(quantized_layers={0})]) is None
    assert not QuantLayerAuditor(None, "warn").enabled  # no predicate: bf16 rollout or bf16 trainer


def test_fused_expert_layout_without_weight_suffix_is_still_judged():
    # transformers >= 5 writes experts as mlp.experts.gate_up_proj / down_proj (no ".weight"); if such a
    # checkpoint's names ever reach the sync, the name rule (which requires ".weight") ships them in bf16
    # while training ran the experts in fp8 -> the audit must report it rather than skip the names.
    r = collect_train_fp8_report([_Chunk(quantized_layers=set(range(N_LAYERS)))])
    names = _hf_names() + [
        f"model.layers.{i}.mlp.experts.{leaf}" for i in range(N_LAYERS) for leaf in ("gate_up_proj", "down_proj")
    ]
    problems = audit_layer_sets(r, names, _sglang_pred())
    assert len(problems) == N_LAYERS * 2 and all("experts" in p and "rule does not quantize it" in p for p in problems)


def _offload_like_verl(chunk):
    """What offload_megatron_model_to_cpu does to TE caches: mark, then drop."""
    for sub in chunk.modules():
        ws = getattr(sub, "_fp8_workspaces", None)
        if isinstance(ws, dict) and ws:
            assert remember_fp8_workspaces(sub)
            ws.clear()


def test_audit_survives_param_offload_clearing_the_te_workspaces():
    # With param_offload=True verl clears module._fp8_workspaces at the end of every train_mode(); the
    # sync that follows must still see which layers ran fp8, through the marker set before clearing.
    chunk = _Chunk(quantized_layers={1, 2})
    _offload_like_verl(chunk)
    assert all(not getattr(m, "_fp8_workspaces", {}) for m in chunk.modules())  # caches really gone
    r = collect_train_fp8_report([chunk])
    assert r.quantized_layers == {1, 2} and r.active
    assert not remember_fp8_workspaces(chunk.decoder.layers[0].mlp.linear_fc1)  # bf16 layer: nothing to mark
    assert not hasattr(chunk.decoder.layers[0].mlp.linear_fc1, FP8_WORKSPACE_SEEN_ATTR)


def test_auditor_warns_once_when_there_is_no_training_side_signal(caplog):
    # sync 0 before any step is expected to find nothing; a second sync without any trace means the
    # signal is missing (not "too early") and the auditor must say so, then stop trying.
    auditor = QuantLayerAuditor(_sglang_pred(), "warn")
    weights = [(n, torch.zeros(1)) for n in _hf_names()]
    list(auditor.record(iter(weights)))
    assert auditor.run([_Chunk(quantized_layers=set())]) is None and not auditor.done
    assert "no training-side signal" not in caplog.text
    list(auditor.record(iter(weights)))
    assert auditor.run([_Chunk(quantized_layers=set())]) is None and auditor.done
    assert "no training-side signal" in caplog.text and "_fp8_workspaces" in caplog.text


def test_auditor_names_disable_parameter_transpose_cache_as_the_cause(caplog):
    # mcore's TE wrappers expose config.disable_parameter_transpose_cache on the module; True means TE
    # never caches the fp8 weight, so the audit reports the cause at the very first sync it sees it.
    chunk = _Chunk(quantized_layers=set())
    for m in chunk.modules():
        if isinstance(m, _TELinear):
            m.disable_parameter_transpose_cache = True
    auditor = QuantLayerAuditor(_sglang_pred(), "warn")
    list(auditor.record((n, torch.zeros(1)) for n in _hf_names()))
    assert auditor.run([chunk]) is None and auditor.done
    assert "disable_parameter_transpose_cache=True" in caplog.text


def test_messages_say_what_the_rollout_side_of_the_comparison_was(caplog):
    names = _hf_names()
    weights = [(n, torch.zeros(1)) for n in names]
    # SGLang without engine truth: the rule compared is the weight-sync rule, and the message says so
    sgl = QuantLayerAuditor(_sglang_pred(), "warn", engine="sglang")
    list(sgl.record(iter(weights)))
    problems = sgl.run([_Chunk(quantized_layers={1, 2})])
    assert problems and all("the weight-sync rule" in p for p in problems)
    assert "NOT against the engine's live parameters" in caplog.text and "sync-vs-engine dtype check" in caplog.text
    caplog.clear()
    # vLLM without engine truth: the engine blacklist as configured
    vl = QuantLayerAuditor(vllm_rollout_predicate(N_LAYERS, ("lm_head", "model.embed_tokens")), "warn", engine="vllm")
    list(vl.record(iter(weights)))
    assert all("the engine blacklist as configured" in p for p in vl.run([_Chunk(quantized_layers={1, 2})]))
    assert "the engine could not be asked" in caplog.text


def test_engine_truth_replaces_the_configured_rule(caplog):
    names = _hf_names()
    weights = [(n, torch.zeros(1)) for n in names]
    # The configured rule (blacklist) says every decoder linear is fp8; the engine, asked directly, says
    # it kept layer 3's MLP in bf16 (e.g. model code passed quant_config=None). Truth wins over the rule.
    truth = {n: n.endswith("_proj.weight") and not n.startswith("model.layers.3.mlp.") for n in names}
    aud = QuantLayerAuditor(vllm_rollout_predicate(N_LAYERS, ("lm_head", "model.embed_tokens")), "warn", engine="vllm")
    list(aud.record(iter(weights)))
    chunk = _Chunk(quantized_layers=set(range(N_LAYERS)))
    assert aud.wants_engine_truth(chunk)  # a training step happened: the caller should ask the engine now
    problems = aud.run(chunk, engine_truth=truth)
    assert sorted(problems) == sorted(
        f"model.layers.3.mlp.{leaf}.weight: training ran it in fp8, {ROLLOUT_LABELS['engine']} does not quantize it"
        for leaf in ("gate_proj", "up_proj", "down_proj")
    )
    assert "real train/rollout precision mismatch" in caplog.text
    # before any training step the caller must not bother the engine
    fresh = QuantLayerAuditor(_sglang_pred(), "warn", engine="sglang")
    list(fresh.record(iter(weights)))
    assert not fresh.wants_engine_truth(_Chunk(quantized_layers=set()))
