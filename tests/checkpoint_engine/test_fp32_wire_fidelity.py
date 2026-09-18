# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""fp32 wire fidelity: the checkpoint's fp32 families must survive the sync.

DSv4 stores a handful of sensitive params in FP32 (hyper-connection
coefficients, ape compressor position embeddings, attention sinks, router gate
bias -- 406 tensors on the serving side, ~68 MB). Three sites used to fold
them to bf16, costing the last 16 mantissa bits (measured rel err p50 1.35e-3,
max 3.9e-3) *invisibly to the verify sweep*, because the replay folded
identically:

  1. transformer_impl's hook-disarming ``weight_dtype=bf16`` stamp on every
     conversion task (the legacy seed's actual fold site);
  2. ``_send_full_seed``'s fold of non-fp8 floats to the rollout dtype;
  3. ``quant_shard_stream``'s catch-all bf16 group on the steady path.

All three now route by ``QuantSpec.fp32_predicate``, derived from the
checkpoint's safetensors headers -- never from the local tensor's presence or
dtype, because the steady wire's group slot layouts must be identical on every
rank and non-owner ranks see no tensor at all.
"""

import json
import os
import struct

import pytest
import torch

CKPT = os.environ.get("VERL_TEST_DSV4_CKPT", "")


def _write_safetensors(path, tensors: dict):
    """Minimal safetensors writer (header + raw bytes) so the test does not
    depend on the safetensors package supporting every dtype we place."""
    _DT = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float8_e4m3fn: "F8_E4M3"}
    header = {}
    blobs = []
    off = 0
    for name, t in tensors.items():
        b = t.contiguous().view(torch.uint8).numpy().tobytes() if t.dtype != torch.float32 else t.numpy().tobytes()
        header[name] = {"dtype": _DT[t.dtype], "shape": list(t.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hdr = json.dumps(header).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(hdr)))
        fh.write(hdr)
        for b in blobs:
            fh.write(b)


@pytest.fixture
def synth_ckpt(tmp_path):
    _write_safetensors(
        tmp_path / "model-00001-of-00001.safetensors",
        {
            "model.layers.0.self_attn.hc_attn_scale": torch.randn(4, dtype=torch.float32),
            "model.layers.0.self_attn.wkv.weight": (torch.randn(8, 8) * 0.01).to(torch.float8_e4m3fn),
            "model.layers.0.self_attn.wkv.weight.scale": torch.ones(1, 1, dtype=torch.float32),
            "model.layers.0.mlp.w1.weight": torch.randn(8, 8, dtype=torch.bfloat16),
            "model.layers.0.ffn.gate.bias": torch.randn(8, dtype=torch.float32),
        },
    )
    return str(tmp_path)


def test_fp32_predicate_selects_fp32_and_only_fp32(synth_ckpt):
    from verl.utils.fp8_ckpt_dtypes import build_ckpt_fp32_predicate

    pred = build_ckpt_fp32_predicate(synth_ckpt)
    assert pred is not None
    assert pred("model.layers.0.self_attn.hc_attn_scale") is True
    assert pred("model.layers.0.ffn.gate.bias") is True
    assert pred("model.layers.0.mlp.w1.weight") is False, "bf16 storage must not be promoted"
    assert pred("model.layers.0.self_attn.wkv.weight") is False, "fp8 codes are not this predicate's business"


def test_fp32_predicate_excludes_scale_grids(synth_ckpt):
    """The F32 scale grids ride the wire's dedicated scale group; routing them
    into the fp32 group would double-ship every scale."""
    from verl.utils.fp8_ckpt_dtypes import build_ckpt_fp32_predicate

    pred = build_ckpt_fp32_predicate(synth_ckpt)
    assert pred("model.layers.0.self_attn.wkv.weight.scale") is False
    assert pred("model.layers.0.self_attn.wkv.weight_scale_inv") is False


def test_fp32_predicate_matches_across_the_bridge_rename(synth_ckpt):
    """Megatron-Bridge spells the block ``attn`` where the checkpoint says
    ``self_attn``; the export asks with its own spelling."""
    from verl.utils.fp8_ckpt_dtypes import build_ckpt_fp32_predicate

    pred = build_ckpt_fp32_predicate(synth_ckpt)
    assert pred("model.layers.0.attn.hc_attn_scale") is True


def test_fp32_predicate_missing_ckpt_returns_none(tmp_path):
    """"could not read" must not masquerade as "nothing is fp32"."""
    from verl.utils.fp8_ckpt_dtypes import build_ckpt_fp32_predicate

    assert build_ckpt_fp32_predicate(str(tmp_path / "nonexistent")) is None


def test_fp32_predicate_all_bf16_ckpt_returns_none(tmp_path):
    from verl.utils.fp8_ckpt_dtypes import build_ckpt_fp32_predicate

    _write_safetensors(
        tmp_path / "model.safetensors",
        {"model.embed.weight": torch.randn(4, 4, dtype=torch.bfloat16)},
    )
    assert build_ckpt_fp32_predicate(str(tmp_path)) is None


def test_quant_spec_carries_fp32_predicate():
    from verl.utils.fp8_sharded import QuantSpec

    spec = QuantSpec(weight_block_size=(128, 128), should_quantize=lambda n: False, fp32_predicate=lambda n: True)
    assert spec.fp32_predicate("x")
    # default stays None: legacy fold-to-rollout-dtype for specs built elsewhere
    assert QuantSpec(weight_block_size=(128, 128), should_quantize=lambda n: False).fp32_predicate is None


@pytest.mark.skipif(not CKPT or not os.path.isdir(CKPT), reason="set VERL_TEST_DSV4_CKPT to a DSv4 checkpoint")
def test_dsv4_fp32_families_are_exactly_the_406_plus_mtp():
    """Pin the census to the checkpoint: 417 F32 non-scale tensors, of which 11
    are mtp.* (not served by the rollout) -- the remaining 406 are exactly the
    tensors that would otherwise lose precision in a bf16 fold."""
    from verl.utils.fp8_ckpt_dtypes import build_ckpt_fp32_predicate, read_checkpoint_dtypes

    dtypes = read_checkpoint_dtypes(CKPT)
    fp32 = [
        n for n, d in dtypes.items() if d == "F32" and not (n.endswith(".scale") or n.endswith("_scale_inv"))
    ]
    assert len(fp32) == 417
    assert sum(1 for n in fp32 if not n.startswith("mtp.")) == 406

    pred = build_ckpt_fp32_predicate(CKPT)
    assert pred is not None
    for n in fp32:
        assert pred(n) is True, n
    # spot-check the five families by their serving-side names
    assert pred("model.layers.7.self_attn.attn_sink") is True
    assert pred("model.layers.7.self_attn.compressor.ape") is True
    assert pred("model.layers.7.hc_attn_scale") is True
    assert pred("model.layers.7.ffn.gate.bias") is True
    assert pred("hc_head_scale") is True
    # and the quantized/bf16 bulk stays out
    assert pred("model.layers.7.self_attn.wkv.weight") is False
    assert pred("model.layers.7.self_attn.wkv.weight.scale") is False


class _HFCfg:
    def __init__(self, model_type=None, quantization_config=None):
        self.model_type = model_type
        if quantization_config is not None:
            self.quantization_config = quantization_config


DSV4_QC = {"quant_method": "fp8", "scale_fmt": "ue8m0", "weight_block_size": [128, 128]}


def test_named_tensors_quant_mode_fp8_ckpt_needs_no_flag():
    """With the quantization flag unset, hybrid servers would be fed
    raw bf16 and SGLang requantized with the plain formula. An fp8-serialized
    DSv4 checkpoint alone must select the verl-side converter."""
    from verl.utils.sglang.sglang_fp8_utils import named_tensors_quant_mode

    assert named_tensors_quant_mode(None, _HFCfg("deepseek_v4", DSV4_QC)) == "dsv4"
    assert named_tensors_quant_mode("fp8", _HFCfg("deepseek_v4", DSV4_QC)) == "dsv4"


def test_named_tensors_quant_mode_legacy_paths_unchanged():
    from verl.utils.sglang.sglang_fp8_utils import named_tensors_quant_mode

    # flag-driven generic path for non-DSv4 models
    assert named_tensors_quant_mode("fp8", _HFCfg("llama", None)) == "generic"
    # no flag, no fp8 ckpt -> raw, exactly as before
    assert named_tensors_quant_mode(None, _HFCfg("llama", None)) is None
    assert named_tensors_quant_mode(None, _HFCfg("deepseek_v4", None)) is None
    # DSv4 with a NON-fp8 quant config does not auto-select
    assert named_tensors_quant_mode(None, _HFCfg("deepseek_v4", {"quant_method": "awq"})) is None

def test_dsv4_converter_passes_pre_quantized_streams_through(tmp_path):
    """The hybrid full sync can already carry the
    bridge's fp8 codes + scale companions; quantizing the codes garbles them
    and the second scale kills SGLang's fused loader with 'duplicate shard kv'.
    Pre-quantized input must pass through byte-identical, with no extra scale."""
    import asyncio
    import json

    from verl.utils.sglang.sglang_fp8_utils import DeepseekV4FP8QuantizerHelper

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _write_safetensors(
        ckpt / "model-00001-of-00001.safetensors",
        {
            "model.layers.0.self_attn.wkv.weight": (torch.randn(256, 256) * 0.01).to(torch.float8_e4m3fn),
            "model.layers.0.self_attn.wkv.scale": torch.ones(2, 2, dtype=torch.float32),
        },
    )
    (ckpt / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.self_attn.wkv.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.self_attn.wkv.scale": "model-00001-of-00001.safetensors",
                }
            }
        )
    )
    helper = DeepseekV4FP8QuantizerHelper({"weight_block_size": [128, 128]}, str(ckpt))

    codes = (torch.randn(256, 256) * 0.01).to(torch.float8_e4m3fn)
    scale = torch.full((2, 2), 2.0**-7, dtype=torch.float32)
    stream = [
        ("model.layers.0.self_attn.wkv.weight", codes),
        ("model.layers.0.self_attn.wkv.scale", scale),
    ]

    async def collect():
        return [(k, v) async for k, v in helper.quant_weights_by_name(iter(stream))]

    out = asyncio.run(collect())
    names = [k for k, _ in out]
    assert names == [n for n, _ in stream], f"stream shape changed: {names}"
    assert torch.equal(out[0][1].view(torch.uint8), codes.view(torch.uint8)), "codes must pass through untouched"
    assert torch.equal(out[1][1], scale), "the upstream scale must pass through untouched"


def test_dsv4_converter_still_quantizes_bf16_input(tmp_path):
    """The passthrough must not disable the converter for genuine bf16 pushes."""
    import asyncio
    import json

    from verl.utils.sglang.sglang_fp8_utils import DeepseekV4FP8QuantizerHelper

    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    _write_safetensors(
        ckpt / "model-00001-of-00001.safetensors",
        {
            "model.layers.0.self_attn.wkv.weight": (torch.randn(256, 256) * 0.01).to(torch.float8_e4m3fn),
            "model.layers.0.self_attn.wkv.scale": torch.exp2(
                torch.randint(-10, -5, (2, 2)).float()
            ),
        },
    )
    (ckpt / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.self_attn.wkv.weight": "model-00001-of-00001.safetensors",
                    "model.layers.0.self_attn.wkv.scale": "model-00001-of-00001.safetensors",
                }
            }
        )
    )
    helper = DeepseekV4FP8QuantizerHelper({"weight_block_size": [128, 128]}, str(ckpt))

    async def collect():
        stream = [("model.layers.0.self_attn.wkv.weight", torch.randn(256, 256, dtype=torch.bfloat16) * 0.01)]
        return [(k, v) async for k, v in helper.quant_weights_by_name(iter(stream))]

    out = asyncio.run(collect())
    assert [k for k, _ in out] == [
        "model.layers.0.self_attn.wkv.weight",
        "model.layers.0.self_attn.wkv.scale",
    ]
    assert out[0][1].element_size() == 1, "bf16 input must come out as fp8 codes"
    log = torch.log2(out[1][1])
    assert torch.equal(log, log.round()), "emitted scales must stay ue8m0 (power of two)"


def test_dsv4_converter_ignores_stale_scale_index_entry(tmp_path):
    """An index-only scale entry must not turn a BF16 weight into FP8."""
    import asyncio
    import json

    from verl.utils.sglang.sglang_fp8_utils import DeepseekV4FP8QuantizerHelper

    shard = "model-00001-of-00001.safetensors"
    weight = torch.randn(8, 8, dtype=torch.bfloat16)
    _write_safetensors(
        tmp_path / shard,
        {
            "model.layers.0.self_attn.wo_a.weight": weight,
            "model.layers.0.self_attn.wkv.weight": torch.randn(8, 8).to(torch.float8_e4m3fn),
            "model.layers.0.self_attn.wkv.scale": torch.ones(1, 1),
        },
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.self_attn.wo_a.weight": shard,
                    "model.layers.0.self_attn.wo_a.scale": shard,
                    "model.layers.0.self_attn.wkv.weight": shard,
                    "model.layers.0.self_attn.wkv.scale": shard,
                }
            }
        )
    )
    helper = DeepseekV4FP8QuantizerHelper(DSV4_QC, str(tmp_path))

    async def collect():
        stream = iter([("model.layers.0.self_attn.wo_a.weight", weight)])
        return [(name, tensor) async for name, tensor in helper.quant_weights_by_name(stream)]

    out = asyncio.run(collect())
    assert len(out) == 1
    assert out[0][0].endswith("wo_a.weight")
    assert out[0][1].dtype == torch.bfloat16
