"""The seed must reproduce a ue8m0 checkpoint's bytes, not merely approximate them.

Under the plain amax/FP8_MAX scale, quantized codes and scale_inv can differ
from the checkpoint state. The mechanism is the round trip:
a power-of-two scale shifts exponents losslessly, an arbitrary real rewrites
mantissas. These tests pin the dialect switch end to end at the pure-function
level: formula, stream output, config plumbing, and the round-trip invariant
that is the whole point.
"""

import torch

from verl.utils.fp8_sharded import FP8_MAX, QuantSpec, quantize_hf_stream, ue8m0_descale
from verl.utils.sglang.sglang_fp8_utils import build_sglang_fp8_quant_config


def _spec(scale_fmt=None):
    return QuantSpec(
        weight_block_size=(128, 128),
        should_quantize=lambda n: n.endswith(".weight"),
        scale_fmt=scale_fmt,
    )


def _stream(spec, t):
    return dict(quantize_hf_stream(iter([("x.weight", t)]), spec))


def test_ue8m0_descale_is_power_of_two_and_matches_the_converter():
    amax = torch.rand(64, 64) * 100 + 1e-6
    d = ue8m0_descale(amax)
    log = torch.log2(d)
    assert torch.equal(log, log.round()), "ue8m0 descale must be a power of two"
    # the DSv4 nccl converter's literal expression (sglang_fp8_utils.py) --
    # the two paths must be byte-identical
    ref = torch.exp2(torch.ceil(torch.log2(amax.clamp_min(1e-10) / FP8_MAX)))
    assert torch.equal(d, ref)


def test_stream_emits_ue8m0_scales_when_asked():
    t = torch.randn(256, 256, dtype=torch.bfloat16)
    out = _stream(_spec("ue8m0"), t)
    log = torch.log2(out["x.weight_scale_inv"])
    assert torch.equal(log, log.round())


def test_stream_default_formula_is_unchanged():
    """Non-ue8m0 checkpoints (every model validated before DSv4) keep their bytes."""
    t = torch.randn(256, 256, dtype=torch.bfloat16)
    legacy = _stream(QuantSpec(weight_block_size=(128, 128), should_quantize=lambda n: True), t)
    explicit = _stream(_spec(None), t)
    assert torch.equal(
        legacy["x.weight_scale_inv"], explicit["x.weight_scale_inv"]
    ) and torch.equal(
        legacy["x.weight"].view(torch.uint8), explicit["x.weight"].view(torch.uint8)
    )


def test_ue8m0_round_trip_is_bit_exact():
    """quantize(dequant(codes, scales)) must reproduce codes AND scales bitwise.

    This is the property that makes seed == disk possible at all: the trainer
    holds dequant(ckpt) in bf16 and the seed re-quantizes it. Deliberately NOT
    asserted for the plain formula -- it does not hold there, which is the bug.
    """
    spec = _spec("ue8m0")
    t = torch.randn(256, 256, dtype=torch.bfloat16)
    first = _stream(spec, t)
    codes, descale = first["x.weight"], first["x.weight_scale_inv"]
    # dequantize per 128x128 block, as the trainer-side bridge does
    up = codes.to(torch.float32).reshape(2, 128, 2, 128) * descale.reshape(2, 1, 2, 1)
    second = _stream(spec, up.reshape(256, 256).to(torch.bfloat16))
    assert torch.equal(second["x.weight_scale_inv"], descale), "scales must survive the round trip"
    assert torch.equal(
        second["x.weight"].view(torch.uint8), codes.view(torch.uint8)
    ), "fp8 codes must survive the round trip"


def test_sticky_descale_keeps_ckpt_headroom_and_bumps_overflow():
    from verl.utils.fp8_sharded import sticky_ue8m0_descale

    amax = torch.tensor([[100.0, 100.0], [1000.0, 100.0]])
    # ckpt scale 1.0 covers amax<=448; block (1,0) outgrew it
    ck = torch.ones(2, 2)
    d = sticky_ue8m0_descale(amax, ck)
    assert d[0, 0] == 1.0 and d[0, 1] == 1.0 and d[1, 1] == 1.0, "covered blocks must keep the ckpt scale"
    assert d[1, 0] == 4.0, f"outgrown block must bump to the tightest covering power, got {d[1,0]}"


def test_sticky_descale_shape_mismatch_fails_loud():
    import pytest

    from verl.utils.fp8_sharded import sticky_ue8m0_descale

    with pytest.raises(AssertionError):
        sticky_ue8m0_descale(torch.ones(2, 2), torch.ones(2, 3))


def test_headroom_round_trip_is_bit_exact_with_ckpt_scales():
    """A checkpoint whose scale carries
    headroom (max code <= FP8_MAX/2). Recomputing from amax tightens the scale
    and rewrites the bytes; with the ckpt's scales in the spec they must come
    back identical."""
    from verl.utils.fp8_sharded import quantize_shard_with_descale

    torch.manual_seed(7)
    # build a "checkpoint": quantize with a DELIBERATELY loose power-of-two scale
    w = torch.randn(128, 128, dtype=torch.bfloat16) * 10
    loose = torch.tensor([[1.0]])  # amax ~60 << 448 -> tight would be 0.25
    ck_codes = quantize_shard_with_descale(w, loose, [128, 128], 0)
    # trainer sees the dequantized master
    master = (ck_codes.float() * loose).to(torch.bfloat16)

    spec_no_ck = _spec("ue8m0")
    out = _stream(spec_no_ck, master)
    assert not torch.equal(out["x.weight_scale_inv"], loose), "sanity: without ckpt scales the scale tightens"

    spec_ck = QuantSpec(
        weight_block_size=(128, 128),
        should_quantize=lambda n: n.endswith(".weight"),
        scale_fmt="ue8m0",
        ckpt_scales={"x.weight": loose},
    )
    out2 = _stream(spec_ck, master)
    assert torch.equal(out2["x.weight_scale_inv"], loose), "scale must come back identical"
    assert torch.equal(
        out2["x.weight"].view(torch.uint8), ck_codes.view(torch.uint8)
    ), "codes must come back identical"


def test_load_ckpt_scales_keys_by_weight_name(tmp_path):
    import json

    from safetensors.torch import save_file

    from verl.utils.fp8_sharded import load_ckpt_scales

    save_file(
        {"a.weight": torch.zeros(4, 4, dtype=torch.bfloat16), "a.scale": torch.full((1, 1), 2.0)},
        tmp_path / "model-00001-of-00001.safetensors",
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a.weight": "model-00001-of-00001.safetensors",
                                   "a.scale": "model-00001-of-00001.safetensors"}})
    )
    scales = load_ckpt_scales(str(tmp_path))
    assert set(scales) == {"a.weight"} and scales["a.weight"].item() == 2.0


def test_load_ckpt_scales_ignores_stale_index_entries(tmp_path):
    """The published DSv4 index lists BF16 wo_a scales absent from the shard."""
    import json

    from safetensors.torch import save_file

    from verl.utils.fp8_sharded import load_ckpt_scales

    shard = "model-00001-of-00001.safetensors"
    save_file(
        {
            "layers.0.attn.wo_a.weight": torch.zeros(4, 4, dtype=torch.bfloat16),
            "layers.0.attn.wo_b.scale": torch.full((1, 1), 2.0),
        },
        tmp_path / shard,
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layers.0.attn.wo_a.weight": shard,
                    "layers.0.attn.wo_a.scale": shard,
                    "layers.0.attn.wo_b.scale": shard,
                }
            }
        )
    )

    scales = load_ckpt_scales(str(tmp_path))
    assert set(scales) == {"layers.0.attn.wo_b.weight"}
    assert scales["layers.0.attn.wo_b.weight"].item() == 2.0


def test_stream_refuses_already_quantized_input():
    """fp8 codes in means a second quantizer exists upstream; refuse it."""
    import pytest

    codes = torch.arange(64, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(8, 8)
    with pytest.raises(AssertionError, match="already"):
        list(quantize_hf_stream(iter([("x.weight", codes)]), _spec("ue8m0")))


def test_build_config_preserves_scale_fmt():
    cfg = build_sglang_fp8_quant_config(
        {"quantization_config": {"weight_block_size": [128, 128], "scale_fmt": "ue8m0"}}
    )
    assert cfg.get("scale_fmt") == "ue8m0", f"scale_fmt dropped again: {cfg}"
    cfg2 = build_sglang_fp8_quant_config({"quantization_config": {"weight_block_size": [128, 128]}})
    assert "scale_fmt" not in cfg2
