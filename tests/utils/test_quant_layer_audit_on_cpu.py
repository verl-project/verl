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

import logging

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
    assert "produced no verdict" not in caplog.text
    list(auditor.record(iter(weights)))
    assert auditor.run([_Chunk(quantized_layers=set())]) is None and auditor.done
    assert "produced no verdict" in caplog.text and "_fp8_workspaces" in caplog.text
    assert "0 of 4 decoder layers" in caplog.text


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
    assert "disable_parameter_transpose_cache=True" in caplog.text and "produced no verdict" in caplog.text


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


def test_engine_recheck_catches_what_the_configured_rule_missed(caplog):
    names = _hf_names()
    weights = [(n, torch.zeros(1)) for n in names]
    # The configured rule (blacklist) says every decoder linear is fp8, and the training side agrees, so
    # the rule-based verdict is clean. The engine, asked directly after the sync, says it kept layer 3's
    # MLP in bf16 (e.g. model code passed quant_config=None): the recheck catches what the rule could not.
    aud = QuantLayerAuditor(vllm_rollout_predicate(N_LAYERS, ("lm_head", "model.embed_tokens")), "warn", engine="vllm")
    list(aud.record(iter(weights)))
    assert aud.run(_Chunk(quantized_layers=set(range(N_LAYERS)))) == []  # rule-based verdict stands, clean
    assert aud.wants_engine_recheck()
    truth = {n: n.endswith("_proj.weight") and not n.startswith("model.layers.3.mlp.") for n in names}
    problems = aud.recheck_against_engine(truth)
    assert sorted(problems) == sorted(
        f"model.layers.3.mlp.{leaf}.weight: training ran it in fp8, {ROLLOUT_LABELS['engine']} does not quantize it"
        for leaf in ("gate_proj", "up_proj", "down_proj")
    )
    assert "the configured rule looked consistent" in caplog.text
    assert not aud.wants_engine_recheck()  # a recheck runs at most once
    # before a verdict exists there is nothing to re-check against the engine
    fresh = QuantLayerAuditor(_sglang_pred(), "warn", engine="sglang")
    list(fresh.record(iter(weights)))
    assert not fresh.wants_engine_recheck()


def test_from_worker_arms_on_the_real_config_objects(caplog):
    """The arming path reads two nested config values; a CPU test on the real dataclasses is the only
    thing that catches a wrong access path before a GPU hour is spent (it did not, on 2026-09-16)."""
    import types

    from omegaconf import OmegaConf

    from verl.workers.config.engine import McoreEngineConfig

    def _worker(quantization, override):
        engine = types.SimpleNamespace(
            engine_config=McoreEngineConfig(override_transformer_config=override),
            model_config=types.SimpleNamespace(
                hf_config=types.SimpleNamespace(num_hidden_layers=N_LAYERS, quantization_config=None)
            ),
        )
        return types.SimpleNamespace(
            config=OmegaConf.create({"rollout": {"quantization": quantization, "name": "vllm"}}),
            actor=types.SimpleNamespace(engine=engine),
        )

    fp8 = {"fp8": "e4m3", "fp8_recipe": "mxfp8"}
    armed = QuantLayerAuditor.from_worker(_worker("mxfp8", fp8))
    assert armed.enabled and armed.engine == "vllm" and armed.rollout_quantizes is not None
    assert armed.rollout_quantizes("model.layers.1.mlp.down_proj.weight") is True
    assert armed.rollout_quantizes("model.layers.1.mlp.gate.weight") is False
    # bf16 trainer + quantized rollout: not armed, and loud about it (this is the silent-guard case)
    caplog.clear()
    off = QuantLayerAuditor.from_worker(_worker("mxfp8", {}))
    assert not off.enabled and "not armed" in caplog.text
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("runs unaudited" in r.getMessage() for r in warnings)
    # bf16 rollout: not armed either, but this is an ordinary config, so it is noted at INFO, never warned
    caplog.clear()
    assert not QuantLayerAuditor.from_worker(_worker(None, fp8)).enabled
    assert not any(r.levelno >= logging.WARNING for r in caplog.records)


def test_record_runs_the_comparison_when_the_weight_stream_ends():
    """The audit must not depend on any single sync route: wrapping the producer's stream is enough.

    This is the regression test for the 2026-09-16 hardware finding - the hook used to sit in the
    colocated worker's update_weights, which the v1 trainer with server-mode replicas does not drive,
    so the audit never ran and said nothing.
    """
    aud = QuantLayerAuditor(_sglang_pred(), "warn", engine="sglang")
    weights = [(n, torch.zeros(1)) for n in _hf_names()]
    chunk = _Chunk(quantized_layers={1, 2})  # layers 0 and 3 stayed bf16 on the training side
    stream = aud.record(iter(weights), modules=[chunk])
    assert not aud.done  # nothing happens until the stream is consumed
    out = list(stream)
    assert out == weights and aud.done  # passed through untouched, and the verdict was reached
    assert aud._verdict is not None and len(aud._verdict[1]) == len(weights)

    # raise mode propagates out of the consumer, which is what stops a mismatched run
    strict = QuantLayerAuditor(_sglang_pred(), "raise", engine="sglang")
    with pytest.raises(RuntimeError, match="disagree"):
        list(strict.record(iter(weights), modules=[chunk]))

    # disabled auditor: pure pass-through, no comparison
    off = QuantLayerAuditor(None, "warn")
    assert list(off.record(iter(weights), modules=[chunk])) == weights and not off.done


def test_engine_recheck_reports_only_when_the_engine_disagrees(caplog):
    names = _hf_names()
    weights = [(n, torch.zeros(1)) for n in names]
    aud = QuantLayerAuditor(vllm_rollout_predicate(N_LAYERS, ("lm_head", "model.embed_tokens")), "warn", engine="vllm")
    chunk = _Chunk(quantized_layers=set(range(N_LAYERS)))
    list(aud.record(iter(weights), modules=[chunk]))  # rule-based verdict: consistent
    assert aud.done and aud.wants_engine_recheck()
    caplog.clear()
    # the engine agrees with the rule. The "everything is fine" line is INFO, and the module logger sits
    # at VERL_LOGGING_LEVEL (WARN by default) - the very filter that hid this guard on hardware - so the
    # test has to raise the level explicitly to see it.
    agree = {n: bool(aud.rollout_quantizes(n)) for n in names}
    with caplog.at_level(logging.INFO, logger="verl.utils.quant_layer_audit"):
        assert aud.recheck_against_engine(agree) == []
    assert "agree with the training side" in caplog.text
    assert not aud.wants_engine_recheck()  # once only

    # a second auditor whose engine kept layer 2's MLP in bf16 although training ran it in fp8
    aud2 = QuantLayerAuditor(vllm_rollout_predicate(N_LAYERS, ("lm_head", "model.embed_tokens")), "warn", engine="vllm")
    list(aud2.record(iter(weights), modules=[chunk]))
    caplog.clear()
    truth = {n: bool(aud2.rollout_quantizes(n)) and not n.startswith("model.layers.2.mlp.") for n in names}
    problems = aud2.recheck_against_engine(truth)
    assert len(problems) == 3 and "even though the configured rule looked consistent" in caplog.text
    assert aud2.recheck_against_engine(truth) is None  # not repeated


def test_trace_makes_every_step_visible_in_one_run(monkeypatch, caplog):
    """VERL_QUANT_LAYER_AUDIT_TRACE=1 must show install, wrap, stream end and verdict at WARNING.

    Three hardware rounds were spent telling "never installed" from "installed but never reached" from
    "reached but silent", each needing the source patched by hand on the pod. One env var now answers it.
    """
    from verl.utils.quant_layer_audit import trace

    monkeypatch.setenv("VERL_QUANT_LAYER_AUDIT_TRACE", "0")
    caplog.clear()
    trace("should-not-appear")
    assert caplog.text == ""

    monkeypatch.setenv("VERL_QUANT_LAYER_AUDIT_TRACE", "1")
    caplog.clear()
    aud = QuantLayerAuditor(_sglang_pred(), "warn", engine="sglang")
    list(aud.record(iter([(n, torch.zeros(1)) for n in _hf_names()]), modules=[_Chunk(quantized_layers={1, 2})]))
    for where in ("trace: record", "record.stream_end", "trace: run"):
        assert where in caplog.text, where
    assert "names=" in caplog.text and "will_run=True" in caplog.text


def test_record_can_tee_names_for_a_later_explicit_run():
    """The worker consumes the stream (possibly off-process) and then calls run() itself.

    record() must populate names during pass-through even without modules, so a later run() with the
    modules produces the verdict. This is the deployment-robust path: it does not need the export
    generator's tail to fire in-process. run() is idempotent, so a tail-triggered run does not double.
    """
    aud = QuantLayerAuditor(_sglang_pred(), "warn", engine="sglang")
    weights = [(n, torch.zeros(1)) for n in _hf_names()]
    # engine wraps without driving the tail's comparison (modules=None): only tees names
    out = list(aud.record(iter(weights), modules=None))
    assert out == weights and not aud.done and len(aud.names) == len(weights)
    # worker runs later with the modules; verdict now produced
    problems = aud.run([_Chunk(quantized_layers={1, 2})])
    assert aud.done and len(problems) == 2 * 7
    # a second run (e.g. a stray tail) is a no-op
    assert aud.run([_Chunk(quantized_layers={1, 2})]) is None


def test_init_sync_then_training_sync_produces_verdict_not_premature_done():
    """Regression for the double-run premature-done bug (found by review, 2026-09-17).

    The engine wraps every sync with record(modules=...) whose tail runs the comparison. The
    pre-training-step ("init") sync has no fp8 workspace yet; that must NOT mark the audit done, or the
    first real training-step sync (which does have workspaces) is skipped and no verdict is ever produced.
    Give-up is judged per distinct sync, so even a spurious second run() on the same sync cannot trip it.
    """
    aud = QuantLayerAuditor(_sglang_pred(), "warn", engine="sglang")
    weights = [(n, torch.zeros(1)) for n in _hf_names()]
    empty = [_Chunk(quantized_layers=set())]  # init sync: no layer has run fp8 yet

    list(aud.record(iter(weights), modules=empty))  # sync 0 (init): record tail runs the comparison
    assert not aud.done and aud.syncs_seen == 1  # inactive but NOT given up after one sync
    assert aud.run(empty) is None and not aud.done  # a spurious extra run() on the same sync: still waiting

    trained = [_Chunk(quantized_layers={1, 2})]  # sync 1: training step produced fp8 workspaces
    problems = list(aud.record(iter(weights), modules=trained))  # returns the passed-through weights
    assert problems == weights and aud.done and aud._verdict is not None  # verdict produced, not skipped


def test_two_inactive_syncs_do_give_up_and_warn(caplog):
    """The give-up path still fires - but only after two *distinct* syncs with no fp8 trace."""
    aud = QuantLayerAuditor(_sglang_pred(), "warn", engine="sglang")
    weights = [(n, torch.zeros(1)) for n in _hf_names()]
    empty = [_Chunk(quantized_layers=set())]
    list(aud.record(iter(weights), modules=empty))  # sync 0: inactive, not done
    assert not aud.done
    caplog.clear()
    list(aud.record(iter(weights), modules=empty))  # sync 1: still inactive -> give up
    assert aud.done and "produced no verdict" in caplog.text
