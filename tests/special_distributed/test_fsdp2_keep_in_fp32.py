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

"""Regression for verl#7092: FSDP2 must honour HF ``_keep_in_fp32_modules`` /
``_keep_in_fp32_modules_strict``.

Two independent degradations are covered.

``ISOLATION`` (runnable unmodified against the pre-fix tree, where it fails):
    verl builds the actor in fp32 and relies on ``MixedPrecisionPolicy`` for
    bf16 compute. ``fully_shard`` otherwise all-gathers every parameter in
    ``mp_policy.param_dtype``, including parameters that Hugging Face kept in
    fp32. The sharded values look perfect while the materialized parameter dtype
    is wrong, which is exactly why the bug is silent.

``BUILD_CAST``:
    ``FSDPEngine._build_module`` calls ``module.to(torch_dtype)`` after
    ``from_pretrained``. HF had already materialised the fp32-keep modules in
    fp32; the blanket cast destroys them (values are rounded through bf16 and
    cannot be recovered). The probe compares the legacy blanket cast against the
    keep-fp32-aware cast.

Everything runs on a locally constructed toy ``PreTrainedModel``; no weights are
downloaded.

Launch:
    torchrun --nproc-per-node=1 --standalone \\
        tests/special_distributed/test_fsdp2_keep_in_fp32.py
    torchrun --nproc-per-node=2 --standalone \\
        tests/special_distributed/test_fsdp2_keep_in_fp32.py
"""

import itertools
import os
import tempfile
from collections import OrderedDict
from contextlib import nullcontext

import torch
import torch.distributed
import torch.nn as nn
from torch.distributed import init_device_mesh
from transformers import PretrainedConfig, PreTrainedModel

from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.distributed import initialize_global_process_group
from verl.utils.fsdp_utils import (
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    get_init_weight_context_manager,
)

SEED = 7092

# ---------------------------------------------------------------------------
# Toy PreTrainedModel
# ---------------------------------------------------------------------------


class ToyKeepFp32Config(PretrainedConfig):
    model_type = "verl_toy_keep_fp32"

    def __init__(self, hidden_size=32, num_hidden_layers=2, vocab_size=64, **kwargs):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)


class ToySensitive(nn.Module):
    """Leaf that owns a float parameter *and* a float buffer."""

    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.register_buffer("running_scale", torch.ones(hidden_size))

    def forward(self, x):
        return self.proj(x) * self.running_scale.to(x.dtype)


class ToyGate(nn.Module):
    """Container without direct parameters -- exercises nested name matching."""

    def __init__(self, hidden_size):
        super().__init__()
        self.inner = ToySensitive(hidden_size)

    def forward(self, x):
        return self.inner(x)


class ToyBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # control module: its parameters follow the low-precision FSDP policy
        self.dense = nn.Linear(hidden_size, hidden_size, bias=False)
        # matched by ``_keep_in_fp32_modules`` (fp16 only, per HF semantics)
        self.fp32_regular = ToySensitive(hidden_size)
        # matched by ``_keep_in_fp32_modules_strict`` (fp16 and bf16)
        self.fp32_strict = ToyGate(hidden_size)

    def forward(self, x):
        # An fp32-keep sub-module returns fp32 activations (this is also what
        # `from_pretrained` produces); real models cast back at the residual, so
        # the toy model does the same.
        dtype = x.dtype
        x = x + self.dense(x)
        x = x + self.fp32_regular(x).to(dtype)
        x = x + self.fp32_strict(x).to(dtype)
        return x


class ToyKeepFp32Model(PreTrainedModel):
    config_class = ToyKeepFp32Config
    base_model_prefix = "toy"
    _no_split_modules = ["ToyBlock"]
    _keep_in_fp32_modules = ["fp32_regular"]
    _keep_in_fp32_modules_strict = ["fp32_strict"]
    _supports_sdpa = False

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([ToyBlock(config.hidden_size) for _ in range(config.num_hidden_layers)])
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear | nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.05)

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class SelfCastingSensitive(ToySensitive):
    """Computes in its weight dtype while preserving the activation dtype.

    This mirrors modules such as Inkling's short convolution: the FSDP boundary
    must not force its incoming activation to fp32.
    """

    def forward(self, x):
        self.activation_input_dtype = x.dtype
        operator_input = x.to(self.proj.weight.dtype)
        self.operator_input_dtype = operator_input.dtype
        projected = self.proj(operator_input)
        self.operator_output_dtype = projected.dtype
        computed = projected * self.running_scale.to(operator_input.dtype)
        self.computation_result_dtype = computed.dtype
        output = computed.to(self.activation_input_dtype)
        self.activation_output_dtype = output.dtype
        return output


class ToySelfCastingModel(ToyKeepFp32Model):
    def __init__(self, config):
        super().__init__(config)
        for layer in self.layers:
            for owner, attr in ((layer, "fp32_regular"), (layer.fp32_strict, "inner")):
                old = getattr(owner, attr)
                new = SelfCastingSensitive(config.hidden_size)
                new.load_state_dict(old.state_dict())
                setattr(owner, attr, new)


class ToyStack(nn.Module):
    """A wrappable container holding the standard transformer-layer targets."""

    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([ToyBlock(config.hidden_size) for _ in range(config.num_hidden_layers)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ToyKeepAncestorModel(ToyKeepFp32Model):
    """The keep declaration hits the container that holds the standard targets.

    ``stack`` owns both ``ToyBlock`` wrap targets, so the keep unit is an
    *ancestor* of them and they must not be wrapped again underneath it.
    """

    # Declared per instance below: the base __init__ validates the keep list
    # against the modules that exist at that point, and `stack` is not one yet.
    _keep_in_fp32_modules = None
    _keep_in_fp32_modules_strict = None

    def __init__(self, config):
        super().__init__(config)
        self.stack = ToyStack(config)
        self.layers = nn.ModuleList()  # the stack replaces them
        self._keep_in_fp32_modules_strict = ["stack"]
        self.post_init()

    def forward(self, input_ids):
        return self.lm_head(self.stack(self.embed_tokens(input_ids)))


class ToyNoForwardContainerModel(ToyKeepFp32Model):
    """A fp32-keep leaf behind a container that has no ``forward`` of its own.

    ``holder`` is a plain ``nn.Module`` used purely for namespacing, and the
    model reaches straight past it to ``holder.child``. FSDP2 unshards a unit's
    parameters from that unit's *own* forward hook, so making ``holder`` the
    unit would leave the hook unreachable and the parameters sharded during
    compute. The keep declaration names the container, so the selector has to
    descend to ``holder.child``.
    """

    _keep_in_fp32_modules = None
    _keep_in_fp32_modules_strict = None

    def __init__(self, config):
        super().__init__(config)
        self.holder = nn.Module()
        self.holder.child = ToySensitive(config.hidden_size)
        self._keep_in_fp32_modules_strict = ["holder"]
        self.post_init()

    def forward(self, input_ids):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        # deliberately *not* self.holder(x): the container is never called
        x = x + self.holder.child(x).to(x.dtype)
        return self.lm_head(x)


class ToyTiedAliasModel(ToyKeepFp32Model):
    """A kept weight exposed under two FQNs inside a single keep unit.

    ``state_dict()`` lists both aliases, so the loader's dtype map has to see
    both -- ``named_parameters()`` alone de-duplicates them away.
    """

    _keep_in_fp32_modules = None
    _keep_in_fp32_modules_strict = ["fp32_strict"]

    def __init__(self, config):
        super().__init__(config)
        for layer in self.layers:
            inner = layer.fp32_strict.inner
            inner.proj_alias = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            inner.proj_alias.weight = inner.proj.weight


class ToyOverlapModel(ToyKeepFp32Model):
    """Parent/child overlap plus a duplicate entry in the keep list.

    ``fp32_strict`` is a parent of ``inner``/``proj``; ``proj`` additionally
    matches inside ``fp32_regular``. The resulting FSDP2 units must be the
    topmost fully-matched modules, each wrapped exactly once.
    """

    _keep_in_fp32_modules = None
    _keep_in_fp32_modules_strict = ["fp32_strict", "inner", "proj", "fp32_strict"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_checkpoint(tmpdir, model_cls, rank, safe_serialization=True):
    """Rank 0 materialises a deterministic fp32 checkpoint on local disk.

    ``safe_serialization=False`` is needed for models that deliberately tie two
    parameters: safetensors refuses to store shared storage.
    """
    path = os.path.join(tmpdir, model_cls.__name__)
    if rank == 0:
        torch.manual_seed(SEED)
        config = ToyKeepFp32Config()
        model = model_cls(config)
        model.save_pretrained(path, safe_serialization=safe_serialization)
    torch.distributed.barrier()
    return path


def _build_module(path, model_cls, torch_dtype, mesh, keep_fp32_aware):
    """Mirror ``FSDPEngine._build_module``'s meta-init + dtype cast."""
    init_context = get_init_weight_context_manager(use_meta_tensor=True, mesh=mesh)
    with init_context():
        module = model_cls.from_pretrained(path, torch_dtype=torch_dtype)
        if keep_fp32_aware:
            from verl.utils.fsdp_utils import cast_module_to_dtype_keeping_fp32_modules

            cast_module_to_dtype_keeping_fp32_modules(module, torch_dtype)
        else:
            # legacy behaviour: blanket cast, wipes HF's fp32-keep modules
            module.to(torch_dtype)
    return module


def _fsdp_kwargs(param_dtype, mesh):
    return {
        "mesh": mesh,
        "mp_policy": MixedPrecisionPolicy(
            param_dtype=param_dtype, reduce_dtype=torch.float32, cast_forward_inputs=True
        ),
        "offload_policy": None,
        "reshard_after_forward": True,
    }


def _wrap_fsdp2(module, param_dtype, mesh):
    """Mirror ``FSDPEngine._build_fsdp_module``'s fsdp2 branch."""
    fsdp_kwargs = _fsdp_kwargs(param_dtype, mesh)
    full_state = module.state_dict()
    apply_fsdp2(module, fsdp_kwargs, {})
    fsdp2_load_full_state_dict(module, full_state, mesh, None)
    return module


def _forward_context(autocast_dtype):
    """Match FSDPEngine.forward_step: autocast for low precision, direct for fp32."""
    if autocast_dtype is None or autocast_dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type=get_device_name(), dtype=autocast_dtype)


def _forward_parameter_dtypes(module, input_ids, autocast_dtype=torch.bfloat16):
    """Record each leaf weight after materialization, separately from activations."""
    seen = {}
    handles = []

    def make_hook(name):
        def hook(mod, args):
            seen[name] = mod.weight.dtype

        return hook

    for name, sub in module.named_modules():
        if isinstance(sub, nn.Linear):
            handles.append(sub.register_forward_pre_hook(make_hook(name)))
    with _forward_context(autocast_dtype):
        out = module(input_ids)
    for h in handles:
        h.remove()
    return seen, out


def _self_casting_observations(module):
    """Keep activation-boundary and internal tensor dtypes explicitly separate."""
    return {
        name: {
            "activation": (submodule.activation_input_dtype, submodule.activation_output_dtype),
            "internal": (
                submodule.operator_input_dtype,
                submodule.operator_output_dtype,
                submodule.computation_result_dtype,
            ),
        }
        for name, submodule in module.named_modules()
        if isinstance(submodule, SelfCastingSensitive)
    }


def _matches(name, prefixes):
    """HF matches whole dot-separated path segments, anywhere in the FQN."""
    segments = name.split(".")
    return any(prefix in segments for prefix in prefixes)


def _reference_model(path, model_cls, torch_dtype):
    """What ``from_pretrained`` alone produces -- the behaviour verl must match."""
    return model_cls.from_pretrained(path, torch_dtype=torch_dtype)


def _local_dtypes(module):
    out = {}
    for name, param in module.named_parameters():
        out[name] = param.dtype
    for name, buf in module.named_buffers():
        out[name] = buf.dtype
    return out


def _full_tensor(t):
    return t.full_tensor() if hasattr(t, "full_tensor") else t


def _gather_state(module):
    """All-gather every sharded state. Collective: must run on every rank, in a
    deterministic (sorted) order, otherwise ranks mismatch on the collective."""
    return {name: _full_tensor(t).cpu().float() for name, t in sorted(module.state_dict().items())}


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def case_isolation(path, mesh, rank, model_cls, expected_fp32_prefixes, label, compute_dtype=torch.bfloat16):
    """fp32 storage + low-precision policy: kept parameters all-gather in fp32.

    This case never downcasts the parameters, so it runs unmodified against the
    pre-fix tree -- where it fails purely on the all-gather dtype.
    """
    module = _build_module(path, model_cls, torch.float32, mesh, keep_fp32_aware=False)
    reference = {k: v.clone() for k, v in module.state_dict().items()} if rank == 0 else None
    module = _wrap_fsdp2(module, compute_dtype, mesh)

    # Checked first on purpose: the stored values are bit-identical, so a
    # value-only assertion passes both before and after the fix. The all-gather
    # dtype below exposes the degradation -- that is what makes the bug silent.
    for name, param in module.named_parameters():
        assert param.dtype == torch.float32, f"[{label}] {name}: sharded dtype {param.dtype}"
    gathered = _gather_state(module)
    if rank == 0:
        for name, ref in reference.items():
            assert torch.equal(gathered[name], ref.cpu().float()), f"[{label}] {name} value mismatch"
        print(f"[{label}] parameter values are bit-identical to the checkpoint")

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, 64, (2, 8), device=get_device_id())
    dtypes, out = _forward_parameter_dtypes(module, input_ids, compute_dtype)

    bad = []
    for name, dtype in sorted(dtypes.items()):
        want = torch.float32 if _matches(name, expected_fp32_prefixes) else compute_dtype
        if dtype != want:
            bad.append(f"{name}: all-gather dtype {dtype}, expected {want}")
    assert not bad, f"[{label}] fp32-keep modules degraded during forward:\n  " + "\n  ".join(bad)

    loss = out.float().pow(2).mean()
    loss.backward()
    for name, param in sorted(module.named_parameters()):
        assert param.grad is not None, f"[{label}] {name} has no grad"
        assert param.grad.dtype == param.dtype, f"[{label}] {name}: grad dtype {param.grad.dtype} != {param.dtype}"
        assert torch.isfinite(_full_tensor(param.grad).float()).all(), f"[{label}] {name} grad not finite"

    if rank == 0:
        print(f"[{label}] isolation case OK (loss={loss.item():.6f})")
    return module


def case_build_cast(path, mesh, rank, torch_dtype, expected_fp32_prefixes, label):
    """The blanket ``module.to(torch_dtype)`` must not wipe HF's fp32 modules."""
    reference = _reference_model(path, ToyKeepFp32Model, torch_dtype) if rank == 0 else None
    module = _build_module(path, ToyKeepFp32Model, torch_dtype, mesh, keep_fp32_aware=True)

    if rank == 0:
        ref_dtypes = _local_dtypes(reference)
        got_dtypes = _local_dtypes(module)
        bad = [
            f"{n}: {got_dtypes[n]} != {d}" for n, d in ref_dtypes.items() if got_dtypes[n] != d and d.is_floating_point
        ]
        assert not bad, f"[{label}] dtype drift vs from_pretrained:\n  " + "\n  ".join(bad)

        ref_state = reference.state_dict()
        for name, tensor in module.state_dict().items():
            if not tensor.dtype.is_floating_point:
                continue
            assert torch.equal(tensor.cpu(), ref_state[name].cpu()), f"[{label}] {name} value drift vs from_pretrained"

        kept = [n for n, d in got_dtypes.items() if d == torch.float32]
        for prefix in expected_fp32_prefixes:
            assert any(_matches(n, [prefix]) for n in kept), f"[{label}] nothing kept in fp32 for '{prefix}'"
        print(f"[{label}] build cast OK ({len(kept)} fp32 states kept)")

    module = _wrap_fsdp2(module, torch_dtype, mesh)
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, 64, (2, 8), device=get_device_id())
    dtypes, out = _forward_parameter_dtypes(module, input_ids, torch_dtype)

    bad = [
        f"{n}: {d}"
        for n, d in sorted(dtypes.items())
        if d != (torch.float32 if _matches(n, expected_fp32_prefixes) else torch_dtype)
    ]
    assert not bad, f"[{label}] all-gather dtype wrong after fsdp2:\n  " + "\n  ".join(bad)
    out.float().pow(2).mean().backward()
    if rank == 0:
        print(f"[{label}] build cast + fsdp2 forward/backward OK")


def probe_legacy_build_cast(path, mesh, rank, torch_dtype, label):
    """Show the legacy blanket cast still degrades (informational)."""
    module = _build_module(path, ToyKeepFp32Model, torch_dtype, mesh, keep_fp32_aware=False)
    if rank == 0:
        degraded = [n for n, d in _local_dtypes(module).items() if "fp32_strict" in n and d != torch.float32]
        print(f"[{label}] legacy module.to({torch_dtype}) degraded {len(degraded)} fp32-keep states: {degraded}")


def case_no_keep_list(path, mesh, rank):
    """Models without keep lists must be completely unaffected."""

    class PlainModel(ToyKeepFp32Model):
        _keep_in_fp32_modules = None
        _keep_in_fp32_modules_strict = None

    module = _build_module(path, PlainModel, torch.float32, mesh, keep_fp32_aware=True)
    for name, param in module.named_parameters():
        assert param.dtype == torch.float32, f"[no-keep] {name}: {param.dtype}"
    module = _wrap_fsdp2(module, torch.bfloat16, mesh)

    from torch.distributed.fsdp import FSDPModule

    units = [n for n, m in module.named_modules() if isinstance(m, FSDPModule)]
    expected = {"", "embed_tokens", "lm_head", "layers.0", "layers.1"}
    assert set(units) == expected, f"[no-keep] fsdp2 units changed: {sorted(units)}"

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, 64, (2, 8), device=get_device_id())
    dtypes, out = _forward_parameter_dtypes(module, input_ids)
    assert set(dtypes.values()) == {torch.bfloat16}, f"[no-keep] parameter dtypes {set(dtypes.values())}"
    out.float().pow(2).mean().backward()
    if rank == 0:
        print("[no-keep] baseline behaviour unchanged")


def case_single_wrapping(path, mesh, rank):
    """Overlapping / duplicated keep entries must yield one unit per module."""
    from torch.distributed.fsdp import FSDPModule

    module = _build_module(path, ToyOverlapModel, torch.float32, mesh, keep_fp32_aware=True)
    module = _wrap_fsdp2(module, torch.bfloat16, mesh)

    units = sorted(n for n, m in module.named_modules() if isinstance(m, FSDPModule))
    assert len(units) == len(set(units)), f"[overlap] duplicate fsdp2 units: {units}"
    for layer in range(2):
        # 'fp32_strict' (parent), 'inner' and 'proj' (descendants) all match, and
        # 'fp32_strict' is listed twice -- the topmost match wins, exactly once.
        assert f"layers.{layer}.fp32_strict" in units, units
        assert f"layers.{layer}.fp32_strict.inner" not in units, units
        assert f"layers.{layer}.fp32_strict.inner.proj" not in units, units
        # 'fp32_regular' is not named in the list, but its only parameter is
        # 'fp32_regular.proj.weight', so the whole module is fully matched and
        # becomes the unit (its unmatched float *buffer* is not FSDP-managed).
        assert f"layers.{layer}.fp32_regular" in units, units
        assert f"layers.{layer}.fp32_regular.proj" not in units, units

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, 64, (2, 8), device=get_device_id())
    dtypes, out = _forward_parameter_dtypes(module, input_ids)
    for name, dtype in sorted(dtypes.items()):
        want = torch.float32 if _matches(name, ["fp32_strict", "inner", "proj"]) else torch.bfloat16
        assert dtype == want, f"[overlap] {name}: {dtype} != {want}"
    out.float().pow(2).mean().backward()
    if rank == 0:
        print(f"[overlap] single wrapping OK, units={units}")


def case_matches_unsharded_reference(path, mesh, rank, activation_dtype, use_autocast):
    """FSDP2 must reproduce plain ``from_pretrained`` in the same context.

    The self-casting child preserves its low-precision activation boundary, so
    this pins ``param_dtype=None``: forcing fp32 inputs changes its output
    contract even though its parameters are correctly all-gathered in fp32. The
    engine-style autocast and direct/no-autocast contexts are both covered.
    """
    context_dtype = activation_dtype if use_autocast else None
    context_label = "engine-autocast" if use_autocast else "direct"
    dtype_label = str(activation_dtype).removeprefix("torch.")
    label = f"reference/{dtype_label}/{context_label}"

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, 64, (2, 8), device=get_device_id())

    reference = ToySelfCastingModel.from_pretrained(path, torch_dtype=activation_dtype).to(get_device_id())
    ref_parameter_dtypes, ref_logits = _forward_parameter_dtypes(reference, input_ids, context_dtype)
    ref_observations = _self_casting_observations(reference)
    assert {obs["activation"] for obs in ref_observations.values()} == {(activation_dtype, activation_dtype)}, (
        f"[{label}] unexpected reference activation boundary: {ref_observations}"
    )
    ref_loss = ref_logits.float().pow(2).mean()
    ref_loss.backward()
    ref_grads = {name: param.grad.detach().clone() for name, param in sorted(reference.named_parameters())}
    ref_dtypes = {name: param.dtype for name, param in reference.named_parameters()}

    module = _build_module(path, ToySelfCastingModel, activation_dtype, mesh, keep_fp32_aware=True)
    module = _wrap_fsdp2(module, activation_dtype, mesh)
    parameter_dtypes, logits = _forward_parameter_dtypes(module, input_ids, context_dtype)
    observations = _self_casting_observations(module)
    assert observations == ref_observations, (
        f"[{label}] activation/internal dtypes differ: {observations} != {ref_observations}"
    )
    assert parameter_dtypes == ref_parameter_dtypes, (
        f"[{label}] materialized parameter dtypes differ: {parameter_dtypes} != {ref_parameter_dtypes}"
    )
    kept_parameter_dtypes = {name: dtype for name, dtype in parameter_dtypes.items() if dtype == torch.float32}
    assert kept_parameter_dtypes, f"[{label}] no fp32 parameter was observed during forward"
    assert all(dtype == torch.float32 for dtype in kept_parameter_dtypes.values())

    loss = logits.float().pow(2).mean()
    loss.backward()

    assert logits.dtype == ref_logits.dtype, f"[{label}] logits dtype {logits.dtype} != {ref_logits.dtype}"
    assert torch.equal(logits, ref_logits), f"[{label}] logits differ from the unsharded model"
    assert loss.item() == ref_loss.item(), f"[{label}] loss {loss.item()!r} != {ref_loss.item()!r}"

    for name, param in sorted(module.named_parameters()):
        assert param.dtype == ref_dtypes[name], f"[{label}] {name}: dtype {param.dtype} != {ref_dtypes[name]}"
        grad = _full_tensor(param.grad)
        assert grad.dtype == ref_grads[name].dtype, f"[{label}] {name}: grad dtype {grad.dtype}"
        assert torch.equal(grad, ref_grads[name]), f"[{label}] {name}: gradient differs from the unsharded model"
    if rank == 0:
        activation_observations = {name: obs["activation"] for name, obs in observations.items()}
        internal_observations = {name: obs["internal"] for name, obs in observations.items()}
        print(
            f"[{label}] activation={activation_observations}; internal={internal_observations}; "
            f"all-gathered-fp32={sorted(kept_parameter_dtypes)}"
        )
        print(f"[{label}] fsdp2 matches unsharded from_pretrained bit-for-bit (loss={loss.item():.10e})")


def case_loader_does_not_mutate_full_state(path, mesh, rank):
    """`fsdp2_load_full_state_dict` documents that it modifies the *model*.

    The dtype alignment must therefore land in a private copy: callers keep and
    reuse the state dict they passed in.
    """
    module = _build_module(path, ToyKeepFp32Model, torch.bfloat16, mesh, keep_fp32_aware=True)
    full_state = module.state_dict()

    # Feed a low-precision source for every floating state, as a bf16 checkpoint
    # would, so the alignment has real work to do on the fp32-keep entries.
    full_state = {
        name: (tensor.to(torch.bfloat16) if tensor.dtype.is_floating_point else tensor)
        for name, tensor in full_state.items()
    }
    before_keys = set(full_state)
    before_ids = {name: id(tensor) for name, tensor in full_state.items()}
    before_dtypes = {name: tensor.dtype for name, tensor in full_state.items()}
    # Ranks other than 0 build on meta, exactly as the engine does, and meta
    # tensors have no values to compare.
    before_values = {name: tensor.detach().clone() for name, tensor in full_state.items() if not tensor.is_meta}
    floating = [name for name, tensor in full_state.items() if tensor.dtype.is_floating_point]
    assert floating, "expected floating entries in the source state"

    apply_fsdp2(module, _fsdp_kwargs(torch.bfloat16, mesh), {})
    fsdp2_load_full_state_dict(module, full_state, mesh, None)

    assert set(full_state) == before_keys, "[no-mutate] key set changed"
    for name, tensor in full_state.items():
        assert id(tensor) == before_ids[name], f"[no-mutate] {name}: tensor object was replaced"
        assert tensor.dtype == before_dtypes[name], f"[no-mutate] {name}: dtype changed to {tensor.dtype}"
        if name in before_values:
            assert torch.equal(tensor, before_values[name]), f"[no-mutate] {name}: value changed"

    # ...and the model still took the target dtype, for parameters and buffers.
    targets = dict(itertools.chain(module.named_parameters(), module.named_buffers()))
    for name in ("layers.0.fp32_strict.inner.proj.weight", "layers.0.fp32_strict.inner.running_scale"):
        assert targets[name].dtype == torch.float32, f"[no-mutate] {name} loaded as {targets[name].dtype}"
    assert targets["layers.0.dense.weight"].dtype == torch.bfloat16
    if rank == 0:
        print(f"[no-mutate] loader left all {len(before_keys)} source entries untouched")


def case_keep_ancestor_absorbs_standard_targets(path, mesh, rank):
    """A keep unit above the standard targets must swallow them, not race them.

    ``layers`` is wrapped first; re-wrapping ``layers.0`` / ``layers.1`` beneath
    it would double-wrap them and invert FSDP2's children-before-parents order.
    """
    from torch.distributed.fsdp import FSDPModule

    module = _build_module(path, ToyKeepAncestorModel, torch.float32, mesh, keep_fp32_aware=True)
    module = _wrap_fsdp2(module, torch.bfloat16, mesh)

    units = sorted(name for name, mod in module.named_modules() if isinstance(mod, FSDPModule))
    assert units == ["", "embed_tokens", "lm_head", "stack"], units
    assert not [u for u in units if u.startswith("stack.")], f"[keep-ancestor] descendant re-wrapped: {units}"
    assert len(units) == len(set(units)), f"[keep-ancestor] a module was wrapped twice: {units}"

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, 64, (2, 8), device=get_device_id())
    dtypes, out = _forward_parameter_dtypes(module, input_ids)
    for name, dtype in sorted(dtypes.items()):
        want = torch.float32 if _matches(name, ["stack"]) else torch.bfloat16
        assert dtype == want, f"[keep-ancestor] {name} all-gathered in {dtype}, expected {want}"
    out.float().pow(2).mean().backward()
    for name, param in sorted(module.named_parameters()):
        assert param.grad is not None, f"[keep-ancestor] {name} has no grad"
    if rank == 0:
        print(f"[keep-ancestor] units={units} (standard descendants absorbed)")


def case_no_forward_container_is_not_a_unit(path, mesh, rank):
    """A unit must have a forward hook that actually fires.

    ``holder`` inherits ``nn.Module.forward`` and is never called, so wrapping
    it would leave its parameters sharded during compute. The unit has to be
    ``holder.child``.
    """
    from torch.distributed.fsdp import FSDPModule

    module = _build_module(path, ToyNoForwardContainerModel, torch.float32, mesh, keep_fp32_aware=True)
    module = _wrap_fsdp2(module, torch.bfloat16, mesh)

    units = sorted(name for name, mod in module.named_modules() if isinstance(mod, FSDPModule))
    assert "holder" not in units, f"[no-forward] container without forward became a unit: {units}"
    assert "holder.child" in units, f"[no-forward] callable child was not made a unit: {units}"
    assert len(units) == len(set(units)), f"[no-forward] a module was wrapped twice: {units}"

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, 64, (2, 8), device=get_device_id())
    dtypes, out = _forward_parameter_dtypes(module, input_ids)
    assert dtypes["holder.child.proj"] == torch.float32, (
        f"[no-forward] fp32-keep leaf all-gathered in {dtypes['holder.child.proj']}"
    )
    for name, dtype in sorted(dtypes.items()):
        if not _matches(name, ["holder", "fp32_strict", "fp32_regular"]):
            assert dtype == torch.bfloat16, f"[no-forward] control {name} all-gathered in {dtype}"

    loss = out.float().pow(2).mean()
    loss.backward()
    for name, param in sorted(module.named_parameters()):
        assert param.grad is not None, f"[no-forward] {name} has no grad"
        assert torch.isfinite(_full_tensor(param.grad).float()).all(), f"[no-forward] {name} grad not finite"
    if rank == 0:
        print(f"[no-forward] units={units} (loss={loss.item():.6f})")


def case_keep_child_under_standard_parent(path, mesh, rank):
    """Reverse control: a standard target that is an *ancestor* of a keep unit.

    It must survive, and be wrapped after the keep child.
    """
    from torch.distributed.fsdp import FSDPModule

    module = _build_module(path, ToyKeepFp32Model, torch.float32, mesh, keep_fp32_aware=True)
    module = _wrap_fsdp2(module, torch.bfloat16, mesh)

    units = sorted(name for name, mod in module.named_modules() if isinstance(mod, FSDPModule))
    for layer in range(2):
        assert f"layers.{layer}.fp32_strict" in units, units  # keep child
        assert f"layers.{layer}" in units, units  # standard parent survives
    assert "" in units, units  # root
    assert len(units) == len(set(units)), units

    torch.manual_seed(SEED)
    input_ids = torch.randint(0, 64, (2, 8), device=get_device_id())
    dtypes, out = _forward_parameter_dtypes(module, input_ids)
    for name, dtype in sorted(dtypes.items()):
        want = torch.float32 if _matches(name, ["fp32_strict"]) else torch.bfloat16
        assert dtype == want, f"[keep-child] {name} all-gathered in {dtype}, expected {want}"
    out.float().pow(2).mean().backward()
    if rank == 0:
        print(f"[keep-child] keep child + standard parent + root all present: {units}")


def case_loader_aligns_every_tied_alias(path, mesh, rank):
    """Both state-dict aliases of a tied keep weight must reach the fp32 target.

    ``named_parameters()`` de-duplicates tied tensors, so a de-duplicated dtype
    map would leave the second alias to define the parameter's dtype.
    """
    import torch.distributed.checkpoint.state_dict as dcp_state_dict

    module = _build_module(path, ToyTiedAliasModel, torch.bfloat16, mesh, keep_fp32_aware=True)

    alias_a = "layers.0.fp32_strict.inner.proj.weight"
    alias_b = "layers.0.fp32_strict.inner.proj_alias.weight"
    # Captured before wrapping, exactly as FSDPEngine._build_fsdp_module does:
    # after `fully_shard` the entries are DTensors and cannot be broadcast.
    full_state = module.state_dict()
    assert alias_a in full_state and alias_b in full_state, sorted(full_state)

    # a low-precision source for every floating entry, as a bf16 checkpoint gives
    source = OrderedDict(
        (name, tensor.to(torch.bfloat16) if tensor.dtype.is_floating_point else tensor)
        for name, tensor in full_state.items()
    )
    source._metadata = OrderedDict({"": {"version": 1}})
    before = {name: (id(t), t.dtype) for name, t in source.items()}

    apply_fsdp2(module, _fsdp_kwargs(torch.bfloat16, mesh), {})

    captured = {}
    original = dcp_state_dict.set_model_state_dict

    def _capture(model, state, options=None):
        captured["state"] = state
        return original(model, state, options=options)

    dcp_state_dict.set_model_state_dict = _capture
    try:
        fsdp2_load_full_state_dict(module, source, mesh, None)
    finally:
        dcp_state_dict.set_model_state_dict = original

    working = captured["state"]
    assert working is not source, "[tied-alias] loader must not pass the caller's dict through"
    assert isinstance(working, OrderedDict), f"[tied-alias] mapping type became {type(working).__name__}"
    assert getattr(working, "_metadata", None) == {"": {"version": 1}}, "[tied-alias] _metadata was dropped"
    for alias in (alias_a, alias_b):
        assert working[alias].dtype == torch.float32, f"[tied-alias] {alias} not aligned: {working[alias].dtype}"

    # the caller's dict is untouched
    for name, tensor in source.items():
        assert (id(tensor), tensor.dtype) == before[name], f"[tied-alias] source entry {name} was modified"
    assert source._metadata == {"": {"version": 1}}

    loaded = dict(module.named_parameters())
    for alias in (alias_a, alias_b):
        assert loaded[alias].dtype == torch.float32, f"[tied-alias] {alias} loaded as {loaded[alias].dtype}"
    assert loaded[alias_a] is loaded[alias_b] or torch.equal(
        _full_tensor(loaded[alias_a].detach()), _full_tensor(loaded[alias_b].detach())
    ), "[tied-alias] aliases diverged"

    # a plain dict source must still work
    plain = {name: tensor for name, tensor in source.items()}
    fsdp2_load_full_state_dict(module, plain, mesh, None)
    assert dict(module.named_parameters())[alias_a].dtype == torch.float32
    if rank == 0:
        print(f"[tied-alias] both aliases aligned to fp32; _metadata preserved ({len(working)} entries)")


def case_state_dict_round_trip(module, mesh, rank):
    """Full state dict save/load round trip preserves dtypes and values."""
    from verl.utils.fsdp_utils import get_fsdp_full_state_dict

    saved = get_fsdp_full_state_dict(module, offload_to_cpu=True, rank0_only=True)
    before = {n: _full_tensor(p.detach()).cpu().float().clone() for n, p in sorted(module.named_parameters())}
    dtypes = {n: p.dtype for n, p in module.named_parameters()}

    fsdp2_load_full_state_dict(module, saved, mesh, None)

    for name, param in sorted(module.named_parameters()):
        assert param.dtype == dtypes[name], f"[round-trip] {name}: dtype {param.dtype} != {dtypes[name]}"
        after = _full_tensor(param.detach()).cpu().float()
        assert torch.equal(after, before[name]), f"[round-trip] {name} value drift"
    if rank == 0:
        print("[round-trip] full state dict save/load OK")


def main():
    assert get_torch_device().device_count() >= 1, "need at least 1 gpu for test"
    _, rank, world_size = initialize_global_process_group()
    mesh = init_device_mesh(get_device_name(), mesh_shape=(world_size,), mesh_dim_names=("fsdp",))

    tmpdir = os.environ.get("VERL_TEST_TMPDIR")
    ctx = tempfile.TemporaryDirectory() if tmpdir is None else None
    if ctx is not None:
        tmpdir = ctx.name
    # all ranks must agree on the checkpoint path
    obj = [tmpdir]
    torch.distributed.broadcast_object_list(obj, src=0)
    tmpdir = obj[0]
    os.makedirs(tmpdir, exist_ok=True)

    path = _make_checkpoint(tmpdir, ToyKeepFp32Model, rank)
    overlap_path = _make_checkpoint(tmpdir, ToyOverlapModel, rank)
    self_cast_path = _make_checkpoint(tmpdir, ToySelfCastingModel, rank)
    tied_path = _make_checkpoint(tmpdir, ToyTiedAliasModel, rank, safe_serialization=False)
    ancestor_path = _make_checkpoint(tmpdir, ToyKeepAncestorModel, rank)
    no_forward_path = _make_checkpoint(tmpdir, ToyNoForwardContainerModel, rank)

    # 1. strict list under bf16 compute (regular list must NOT apply -- HF only
    #    honours it for fp16 / quantized loads).
    module = case_isolation(path, mesh, rank, ToyKeepFp32Model, ["fp32_strict"], "strict/bf16")
    case_state_dict_round_trip(module, mesh, rank)
    del module
    get_torch_device().empty_cache()

    # 2. union of both lists under fp16 compute.
    case_isolation(
        path,
        mesh,
        rank,
        ToyKeepFp32Model,
        ["fp32_strict", "fp32_regular"],
        "union/fp16-compute",
        compute_dtype=torch.float16,
    )
    get_torch_device().empty_cache()

    # 3. build-time cast, bf16 target dtype (strict list only).
    probe_legacy_build_cast(path, mesh, rank, torch.bfloat16, "strict/bf16")
    case_build_cast(path, mesh, rank, torch.bfloat16, ["fp32_strict"], "build/bf16")
    get_torch_device().empty_cache()

    # 4. build-time cast, fp16 target dtype (union of both lists).
    case_build_cast(path, mesh, rank, torch.float16, ["fp32_strict", "fp32_regular"], "build/fp16")
    get_torch_device().empty_cache()

    # 5. BF16/FP16 parity with the unsharded `from_pretrained` graph, both in
    #    FSDPEngine's autocast context and through a direct/no-autocast call.
    for activation_dtype in (torch.bfloat16, torch.float16):
        for use_autocast in (True, False):
            case_matches_unsharded_reference(self_cast_path, mesh, rank, activation_dtype, use_autocast)
            get_torch_device().empty_cache()

    # 5b. the loader must not touch the caller's state dict.
    case_loader_does_not_mutate_full_state(path, mesh, rank)
    get_torch_device().empty_cache()

    # 5c. tied state-dict aliases and mapping metadata.
    case_loader_aligns_every_tied_alias(tied_path, mesh, rank)
    get_torch_device().empty_cache()

    # 5d. keep unit above / below the standard wrap targets.
    case_keep_ancestor_absorbs_standard_targets(ancestor_path, mesh, rank)
    get_torch_device().empty_cache()
    case_keep_child_under_standard_parent(path, mesh, rank)
    get_torch_device().empty_cache()

    # 5e. a keep container whose forward hook would never fire.
    case_no_forward_container_is_not_a_unit(no_forward_path, mesh, rank)
    get_torch_device().empty_cache()

    # 6. overlap / duplicate handling, and the untouched baseline.
    case_single_wrapping(overlap_path, mesh, rank)
    get_torch_device().empty_cache()
    case_no_keep_list(path, mesh, rank)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    if ctx is not None:
        ctx.cleanup()
    if rank == 0:
        print("test_fsdp2_keep_in_fp32 passed")


if __name__ == "__main__":
    main()
