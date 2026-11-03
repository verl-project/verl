# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Utility functions for the Automodel engine integration."""

import logging
from pathlib import Path

import torch
import torch.distributed

from verl.utils.device import get_device_id, get_torch_device


def ensure_vendored_nemo_automodel() -> bool:
    """Make the vendored ``third_party/automodel`` checkout importable.

    The training environment may not install ``nemo_automodel`` (NeMo AutoModel).
    When the job ships the vendored checkout, put it on ``sys.path`` BEFORE the
    automodel engine module runs its top-level ``nemo_automodel`` imports.

    This is a no-op when ``nemo_automodel`` is already importable
    (site-packages) or when the vendored checkout is absent (pip-installed
    verl without the repo tree). Only the package itself comes from the
    checkout; its Python dependencies (transformers, megatron-fsdp, torchao,
    torchdata, ...) must already be installed.

    Returns:
        True if ``nemo_automodel`` is importable after this call.
    """
    import importlib.util
    import sys

    # Must run BEFORE the early-return below: nemo_automodel may already be
    # importable via PYTHONPATH (the runtime_env sets it, and with a zip
    # working_dir the relative entry resolves correctly), in which case the
    # vendored-path insert is skipped -- but the torchao stub is still needed.
    _ensure_torchao_stub(Path(__file__).resolve().parent)

    try:
        if importlib.util.find_spec("nemo_automodel") is not None:
            _apply_compat_patches()
            return True
    except (ImportError, ValueError):
        # Broken installation: fall through and try the vendored checkout.
        pass

    # <repo>/verl/workers/engine/automodel/utils.py -> <repo> (4 levels up).
    # With the Ray working_dir layout this resolves to the uploaded package
    # root regardless of the worker process cwd, unlike a relative PYTHONPATH.
    vendored = Path(__file__).resolve().parents[4] / "third_party" / "automodel"
    if (vendored / "nemo_automodel" / "__init__.py").is_file():
        sys.path.insert(0, str(vendored))
        _apply_compat_patches()
        return True
    return False


def _apply_compat_patches() -> None:
    """Apply verl-side runtime compat patches to nemo_automodel (best-effort).

    The vendored automodel checkout stays upstream-clean; compatibility fixes
    that upstream has not merged yet live in compat_patches.py (same pattern as
    the sglang weight_sync_patches). Runs on the driver and every worker that
    imports the automodel engine, before nemo_automodel submodules are used.
    """
    try:
        from .compat_patches import apply_automodel_compat_patches

        apply_automodel_compat_patches()
    except Exception:  # pragma: no cover - never block engine startup
        import traceback

        logging.getLogger(__name__).warning("automodel compat patches failed to apply:\n%s", traceback.format_exc())


def _ensure_torchao_stub(engine_dir: Path) -> None:
    """Satisfy nemo_automodel's top-level torchao imports when torchao is absent.

    The vendored nemo_automodel imports torchao unconditionally from
    components/quantization/qat.py and recipes/llm/train_ft.py even when
    QAT / Float8 training are unused. When the environment does not provide
    torchao, fall back to the stub package shipped next to this file
    (_torchao_stub/torchao): imports succeed, every stubbed symbol raises on
    use, and HAVE_TORCHAO-style probes stay False because the stub never
    provides Float8LinearConfig / convert_to_float8_training.
    """
    import importlib.util
    import sys

    try:
        if importlib.util.find_spec("torchao") is not None:
            return
    except (ImportError, ValueError):
        pass
    stub_dir = engine_dir / "_torchao_stub"
    if (stub_dir / "torchao" / "__init__.py").is_file():
        sys.path.insert(0, str(stub_dir))


def get_dp_rank(device_mesh, include_cp=False):
    """Get data-parallel rank from device mesh."""
    if device_mesh is None:
        return 0
    if include_cp and "cp" in device_mesh.mesh_dim_names and device_mesh["cp"].size() > 1:
        return device_mesh.get_local_rank("dp_cp")
    return device_mesh.get_local_rank("dp")


def get_tp_rank(device_mesh):
    """Get tensor-parallel rank from device mesh."""
    if device_mesh is None or "tp" not in device_mesh.mesh_dim_names or device_mesh["tp"].size() == 1:
        return 0
    return device_mesh.get_local_rank("tp")


def get_pp_rank(device_mesh):
    """Get pipeline-parallel rank from device mesh."""
    if device_mesh is None or "pp" not in device_mesh.mesh_dim_names or device_mesh["pp"].size() == 1:
        return 0
    return device_mesh.get_local_rank("pp")


def get_dp_group_size(device_mesh, include_cp=False):
    """Get data-parallel group size from device mesh."""
    if device_mesh is None:
        return torch.distributed.get_world_size()
    if include_cp and "cp" in device_mesh.mesh_dim_names and device_mesh["cp"].size() > 1:
        return device_mesh["dp_cp"].size()
    if "dp" in device_mesh.mesh_dim_names:
        return device_mesh["dp"].size()
    return torch.distributed.get_world_size()


def maybe_fully_shard_optimizer(model, optimizer, distributed_config):
    """Call fully_shard_optimizer for MegatronFSDP strategy."""
    from nemo_automodel.components.distributed.config import MegatronFSDPConfig

    if isinstance(distributed_config, MegatronFSDPConfig) and torch.distributed.get_world_size() > 1:
        from megatron_fsdp.fully_shard import fully_shard_optimizer

        fully_shard_optimizer(model, optimizer)


def build_distributed_config_from_engine_config(engine_config, world_size):
    """Build the automodel DistributedSetup (new sidecar API) from engine config.

    Replaces the removed public ``create_device_mesh`` API (upstream #4032
    moved parallelization into model sidecars consumed via
    ``DistributedSetup``/``MeshContext``). Returns the setup alongside the
    legacy triple so existing call sites can keep reading
    distributed_config / device_mesh / moe_mesh directly.

    Args:
        engine_config: AutomodelEngineConfig instance.
        world_size: Total number of processes in the job.

    Returns:
        Tuple of (distributed_setup, distributed_config, device_mesh, moe_mesh).
    """
    from nemo_automodel.components.distributed.config import (
        DDPConfig,
        DistributedSetup,
        FSDP2Config,
        MegatronFSDPConfig,
        MoEParallelizerConfig,
    )
    from nemo_automodel.components.distributed.mesh import ParallelismSizes

    strategy = engine_config.distributed_strategy
    moe_parallel_config = None

    if strategy == "fsdp2":
        from torch.distributed.fsdp import MixedPrecisionPolicy

        from verl.utils.torch_dtypes import PrecisionType

        mp_policy = MixedPrecisionPolicy(
            param_dtype=PrecisionType.to_dtype(engine_config.mp_param_dtype),
            reduce_dtype=PrecisionType.to_dtype(engine_config.mp_reduce_dtype),
            output_dtype=PrecisionType.to_dtype(engine_config.mp_output_dtype),
            cast_forward_inputs=True,
        )

        distributed_config = FSDP2Config(
            sequence_parallel=engine_config.sequence_parallel,
            mp_policy=mp_policy,
            activation_checkpointing=engine_config.activation_checkpointing,
            defer_fsdp_grad_sync=engine_config.defer_fsdp_grad_sync,
        )

    elif strategy == "megatron_fsdp":
        distributed_config = MegatronFSDPConfig(
            activation_checkpointing=engine_config.activation_checkpointing,
        )

    elif strategy == "ddp":
        distributed_config = DDPConfig(
            activation_checkpointing=engine_config.activation_checkpointing,
        )

    else:
        raise ValueError(f"Unsupported distributed_strategy: {strategy}")

    # MoE parallelizer config lives on the DistributedSetup (upstream now
    # rejects a separate ``moe_config=`` kwarg on from_pretrained).
    if engine_config.ep_size > 1:
        moe_kwargs = dict(engine_config.moe_config) if engine_config.moe_config else {}
        if hasattr(distributed_config, "mp_policy"):
            moe_kwargs.setdefault("mp_policy", distributed_config.mp_policy)

        moe_parallel_config = MoEParallelizerConfig(**moe_kwargs)

    distributed_setup = DistributedSetup.build(
        strategy=distributed_config,
        parallelism_sizes=ParallelismSizes(
            tp_size=engine_config.tp_size,
            pp_size=engine_config.pp_size,
            cp_size=engine_config.cp_size,
            ep_size=engine_config.ep_size,
            dp_replicate_size=engine_config.dp_replicate_size,
        ),
        moe_parallel_config=moe_parallel_config,
        activation_checkpointing=engine_config.activation_checkpointing,
        world_size=world_size,
    )

    mesh_context = distributed_setup.mesh_context
    return (
        distributed_setup,
        distributed_setup.strategy_config,
        mesh_context.device_mesh,
        mesh_context.moe_mesh,
    )


def build_automodel_model(model_config, engine_config, distributed_setup):
    """Build a model using NeMoAutoModelForCausalLM.from_pretrained().

    Args:
        model_config: HFModelConfig with model path and settings.
        engine_config: AutomodelEngineConfig with distributed settings.
        distributed_setup: Automodel DistributedSetup (mesh + strategy + AC
            policy). Upstream removed the separate device_mesh / moe_mesh /
            distributed_config / activation_checkpointing kwargs (#4032) and
            now rejects them with a TypeError.

    Returns:
        A HuggingFace model with Automodel's distributed infrastructure applied.
    """
    from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM

    kwargs = {}

    if engine_config.enable_fp8:
        from nemo_automodel.components.quantization.fp8 import FP8Config

        # NOTE: FP8Config.enabled defaults to False and apply_fp8_to_model
        # early-returns on it -- pass enabled=True or the flag is a no-op.
        kwargs["fp8_config"] = FP8Config(enabled=True)

    if engine_config.enable_compile:
        from nemo_automodel.components.utils.compile_utils import CompileConfig

        kwargs["compile_config"] = CompileConfig()

    # Qwen/Llama with ep_size<=1: use HF implementation.
    from transformers import AutoConfig

    _cfg = AutoConfig.from_pretrained(model_config.path, trust_remote_code=model_config.trust_remote_code)
    _arch = (getattr(_cfg, "architectures", None) or [""])[0].lower()
    if engine_config.ep_size <= 1 and ("qwen" in _arch or "llama" in _arch):
        kwargs["force_hf"] = True

    if engine_config.backend_config and not kwargs.get("force_hf", False):
        from nemo_automodel.components.models.common.utils import BackendConfig

        backend_kwargs = dict(engine_config.backend_config)
        kwargs["backend"] = BackendConfig(**backend_kwargs)

    kwargs["attn_implementation"] = engine_config.attn_implementation

    from verl.utils.torch_dtypes import PrecisionType

    kwargs["torch_dtype"] = PrecisionType.to_dtype(engine_config.model_dtype)

    # Apply user config overrides (e.g. num_nextn_predict_layers=0 to skip
    # building MTP modules that do not participate in RL training).  They
    # flow through get_hf_config -> AutoConfig.from_pretrained, which applies
    # unknown kwargs as config attributes.
    for _k, _v in (getattr(model_config, "override_config", None) or {}).items():
        kwargs[_k] = _v

    model = NeMoAutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_config.path,
        distributed_setup=distributed_setup,
        trust_remote_code=model_config.trust_remote_code,
        **kwargs,
    )

    return model


@torch.no_grad()
def offload_automodel_model_to_cpu(model, empty_cache=True):
    """Offload an FSDP2-wrapped model to CPU (reshard, move to CPU, optional cache clear)."""
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
    from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state

    for module in model.modules():
        state = _get_module_fsdp_state(module)
        if state is None:
            continue
        fsdp_param_group = state._fsdp_param_group

        if fsdp_param_group is None:
            continue

        fsdp_param_group._training_state = TrainingState.IDLE

    model.reshard()
    model.cpu()
    if empty_cache:
        get_torch_device().empty_cache()


@torch.no_grad()
def load_automodel_model_to_gpu(model):
    """Load model back to GPU."""
    device = get_device_id()
    model.to(device, non_blocking=True)


@torch.no_grad()
def offload_automodel_optimizer(optimizer):
    """Offload optimizer state to CPU."""
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to("cpu", non_blocking=True)


@torch.no_grad()
def load_automodel_optimizer(optimizer, device_id):
    """Load optimizer state back to GPU."""
    if not optimizer.state:
        return
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            state = optimizer.state[param]
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device_id, non_blocking=True)
