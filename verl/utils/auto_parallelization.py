"""
Analytical (formula-based) auto-parallelization utilities for verl.

When `trainer.auto_parallel.enabled=true`, this module may override Megatron and
rollout parallelism degrees before `validate_config()` runs.

Selection policy (configurable):
- ``objective=max_mp`` (default): prefer higher TP×PP×CP (more sharding). This matches
  common Megatron + colocated vLLM scripts (e.g. TP=world_size, DP=1) and avoids the
  legacy ``max_dp`` behavior that could replace an explicit CLI ``tensor_model_parallel_size=8``
  with TP=1 whenever the analytical memory estimate passed.
- ``objective=max_dp``: maximize data parallelism first (documented in ANALYTICAL_AUTO_PARALLEL_DOC).

Safety:
- If the analytical memory gate rejects every candidate, the fallback search **without**
  memory uses ``max_mp`` only, never ``max_dp``, so we do not pick TP=1 purely because
  the estimator was wrong.
- ``respect_config_parallelism=true``: do not search; validate + optional memory check only,
  then sync rollout fields — use when you want auto_parallel for validation only.
- ``allow_context_parallel=false`` (default): only ``context_parallel_size=1`` is considered.
  Megatron Core ``DotProductAttention`` asserts CP==1 unless ``TEDotProductAttention`` is used
  (common MindSpeed / mbridge stacks use DotProductAttention → CP>1 from auto-parallel breaks ref/actor init).
- When HF ``config.json`` is readable, candidates must satisfy ``num_attention_heads % TP == 0``
  (Megatron Core) and, if ``PP > 1``, ``num_hidden_layers % PP == 0`` — e.g. Qwen2.5-7B has 28 heads,
  so TP=8 is invalid on that model.

Memory/CI-based acceptance can be extended later; analytical mode uses coarse formulas.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _iter_divisors(n: int) -> Iterable[int]:
    """Yield positive integer divisors of `n` (unordered)."""
    if n <= 0:
        return
    i = 1
    while i * i <= n:
        if n % i == 0:
            yield i
            other = n // i
            if other != i:
                yield other
        i += 1


def _get_megatron_degrees_from_actor_config(config, n_gpus: int):
    """
    Read TP/PP/CP/EP/SP from `actor_rollout_ref.actor.megatron` and derive DP.

    Returns:
      - degrees_obj: `verl.utils.placement.ParallelismDegrees` instance
      - etp: expert_tensor_parallel_size (int, default 1)
    """
    from verl.utils.placement import extract_degrees_from_config

    degrees_obj = extract_degrees_from_config(config=config, n_gpus=n_gpus)
    if degrees_obj is None:
        # For non-megatron actors (e.g. dp/fsdp), we can not safely override degrees.
        logger.warning(
            "trainer.auto_parallel.enabled=true but actor engine does not expose `actor_rollout_ref.actor.megatron` "
            "(skipping auto-parallelization)."
        )
        return None
    return degrees_obj


def validate_degrees(degrees, n_gpus: int, n_gpus_per_node: int, nnodes: int) -> None:
    """
    Validate candidate degrees for feasibility.

    Raises:
      ValueError if any hard constraint fails.
    """
    messages = degrees.validate(n_gpus=n_gpus, n_gpus_per_node=n_gpus_per_node, nnodes=nnodes)
    errors = [m for m in messages if m.startswith("ERROR")]
    warnings = [m for m in messages if m.startswith("WARNING")]

    for w in warnings:
        logger.warning(w)

    if errors:
        raise ValueError("\n".join(errors))


def _dtype_bytes(dtype: str) -> int:
    """Best-effort mapping from dtype string to element byte size."""
    d = (dtype or "").lower()
    if d in {"bfloat16", "bf16", "float16", "fp16", "half"}:
        return 2
    if d in {"float32", "fp32"}:
        return 4
    # Conservative default
    return 2


def _get_memory_budget_gb(*, target_utilization: float) -> Optional[float]:
    """
    Compute per-device memory budget as: total_device_mem_gb * target_utilization.

    Returns None if device memory can't be queried.
    """
    try:
        import torch

        # 1) Manual override (matches luncher_msrl.py behavior)
        npu_memory_env = os.environ.get("NPU_MEMORY_GB") or os.environ.get("ASCEND_RT_MEMORY_GB")
        if npu_memory_env:
            try:
                total_gb = float(npu_memory_env)
                return float(total_gb) * float(target_utilization)
            except Exception:
                logger.warning("Invalid NPU_MEMORY_GB/ASCEND_RT_MEMORY_GB=%r. Ignoring.", npu_memory_env)

        # 2) CUDA memory (common on GPU)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total_gb = props.total_memory / 1024**3
            return float(total_gb) * float(target_utilization)

        # 3) NPU memory (Ascend)
        if hasattr(torch, "npu"):
            # torch_npu exposes either get_device_properties or memory_info
            try:
                if hasattr(torch.npu, "get_device_properties"):
                    props = torch.npu.get_device_properties(0)
                    # Some versions expose total_memory in bytes
                    total_mem_bytes = getattr(props, "total_memory", None)
                    if total_mem_bytes is not None:
                        total_gb = total_mem_bytes / 1024**3
                        return float(total_gb) * float(target_utilization)
            except Exception:
                pass

            try:
                if hasattr(torch.npu, "memory_info"):
                    mem_info = torch.npu.memory_info(0)
                    # memory_info may return allocated/available; be conservative with "available"
                    # We'll prefer allocated+reserved isn't stable, so use total if possible.
                    total_bytes = None
                    if isinstance(mem_info, dict):
                        total_bytes = mem_info.get("total") or mem_info.get("total_memory")
                    if total_bytes is not None:
                        total_gb = float(total_bytes) / 1024**3
                        return float(total_gb) * float(target_utilization)
            except Exception:
                pass

        # 4) Fallback (keep memory gating active even on NPU environments without torch.npu)
        logger.warning("Could not auto-detect device total memory. Using default 16.0GB. Set NPU_MEMORY_GB for accuracy.")
        return 16.0 * float(target_utilization)
    except Exception as e:
        logger.warning("Could not query device total memory: %s. Using default 16.0GB.", e)
        return 16.0 * float(target_utilization)


def _try_load_hf_config_json(model_path: str, trust_remote_code: bool) -> Optional[dict[str, Any]]:
    """
    Load a model's small HF config (config.json) without downloading weights.
    """
    try:
        from verl.utils.fs import copy_to_local, is_non_local

        src = model_path.rstrip("/")
        config_json_path = None

        if is_non_local(src):
            # Copy only config.json to keep this lightweight.
            config_json_path = copy_to_local(f"{src}/config.json")
        else:
            # Local dir or file.
            if os.path.isdir(src) and os.path.exists(os.path.join(src, "config.json")):
                config_json_path = os.path.join(src, "config.json")
            elif os.path.isfile(src) and os.path.basename(src) == "config.json":
                config_json_path = src

        if config_json_path and os.path.exists(config_json_path):
            with open(config_json_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Fallback: try HF AutoConfig (downloads only config.json typically).
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        return hf_cfg.to_dict()
    except Exception as e:
        logger.warning("Failed to load HF config for analytical memory estimation: %s", e)
        return None


def _try_get_num_attention_heads(config) -> Optional[int]:
    """Read ``num_attention_heads`` from HF ``config.json`` (Megatron requires TP | heads)."""
    try:
        from omegaconf import OmegaConf as _OC

        model_path = _OC.select(config, "actor_rollout_ref.model.path", default=None)
        if not model_path:
            return None
        trust_remote_code = bool(
            _OC.select(config, "data.trust_remote_code", default=False)
            or _OC.select(config, "actor_rollout_ref.model.trust_remote_code", default=False)
        )
        hf_cfg = _try_load_hf_config_json(str(model_path), trust_remote_code)
        if not hf_cfg:
            return None
        h = hf_cfg.get("num_attention_heads", hf_cfg.get("n_head"))
        return int(h) if h is not None else None
    except Exception as e:
        logger.warning("Could not read num_attention_heads for auto-parallel: %s", e)
        return None


def _try_get_num_hidden_layers(config) -> Optional[int]:
    """Read layer count for PP divisibility (optional)."""
    try:
        from omegaconf import OmegaConf as _OC

        model_path = _OC.select(config, "actor_rollout_ref.model.path", default=None)
        if not model_path:
            return None
        trust_remote_code = bool(
            _OC.select(config, "data.trust_remote_code", default=False)
            or _OC.select(config, "actor_rollout_ref.model.trust_remote_code", default=False)
        )
        hf_cfg = _try_load_hf_config_json(str(model_path), trust_remote_code)
        if not hf_cfg:
            return None
        n = hf_cfg.get("num_hidden_layers", hf_cfg.get("n_layer"))
        return int(n) if n is not None else None
    except Exception as e:
        logger.warning("Could not read num_hidden_layers for auto-parallel: %s", e)
        return None


def _try_build_memory_estimator_inputs(config) -> Optional[dict[str, Any]]:
    """
    Build lightweight analytical memory estimator inputs (no weights).
    """
    try:
        from omegaconf import OmegaConf as _OC

        rollout_cfg = config.actor_rollout_ref.rollout
        actor_cfg = config.actor_rollout_ref.actor
        model_cfg = config.actor_rollout_ref.model

        # Dtype + KV cache shapes
        dtype_bytes = _dtype_bytes(getattr(rollout_cfg, "dtype", "bfloat16"))
        max_num_seqs = int(getattr(rollout_cfg, "max_num_seqs", 1))

        # For vLLM KV cache, use rollout.max_model_len if set; otherwise fall back
        # to actor ppo_max_token_len_per_gpu.
        kv_seq_len = getattr(rollout_cfg, "max_model_len", None)
        if kv_seq_len is None:
            kv_seq_len = getattr(actor_cfg, "ppo_max_token_len_per_gpu", None)
        kv_seq_len = int(kv_seq_len) if kv_seq_len is not None else 16384

        # Training activation memory must not assume the full Megatron PPO token cap when
        # rollout / dataset use a much shorter context (common in RL). Using 16384 here while
        # KV uses max_model_len=256 rejects every degree under the analytical memory gate.
        ppo_max_raw = getattr(actor_cfg, "ppo_max_token_len_per_gpu", None)
        ppo_max = int(ppo_max_raw) if ppo_max_raw is not None else kv_seq_len
        data_seq_cap: Optional[int] = None
        try:
            pl = _OC.select(config, "data.max_prompt_length", default=None)
            rl = _OC.select(config, "data.max_response_length", default=None)
            if pl is not None and rl is not None:
                data_seq_cap = int(pl) + int(rl)
        except Exception:
            data_seq_cap = None
        train_seq_len = min(ppo_max, kv_seq_len)
        if data_seq_cap is not None:
            train_seq_len = min(train_seq_len, max(1, data_seq_cap))
        train_seq_len = max(1, int(train_seq_len))

        # micro-batch proxy (activation memory is sensitive; be conservative if missing)
        mbs_per_gpu = getattr(actor_cfg, "ppo_micro_batch_size_per_gpu", None)
        if mbs_per_gpu is None:
            mbs_per_gpu = getattr(actor_cfg, "ppo_micro_batch_size", None)
        if mbs_per_gpu is None:
            mbs_per_gpu = 1
        mbs_per_gpu = int(mbs_per_gpu)

        # Load HF config.json (small); trust flag may live on data.* or model.*
        trust_remote_code = bool(
            _OC.select(config, "data.trust_remote_code", default=False)
            or _OC.select(config, "actor_rollout_ref.model.trust_remote_code", default=False)
        )
        hf_cfg = _try_load_hf_config_json(model_cfg.path, trust_remote_code)
        if not hf_cfg:
            return None

        hidden_size = hf_cfg.get("hidden_size", hf_cfg.get("d_model", None))
        num_layers = hf_cfg.get("num_hidden_layers", hf_cfg.get("n_layer", None))
        num_heads = hf_cfg.get("num_attention_heads", hf_cfg.get("n_head", None))
        intermediate_size = hf_cfg.get("intermediate_size", hf_cfg.get("ffn_dim", None))
        vocab_size = hf_cfg.get("vocab_size", None)

        if hidden_size is None or num_layers is None or num_heads is None:
            return None

        # MoE fields (optional)
        num_experts = hf_cfg.get("num_experts", hf_cfg.get("n_experts", 1))
        moe_intermediate_size = hf_cfg.get("moe_intermediate_size", intermediate_size)
        if moe_intermediate_size is None:
            moe_intermediate_size = intermediate_size if intermediate_size is not None else hidden_size

        return {
            "hidden_size": int(hidden_size),
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
            "intermediate_size": int(intermediate_size) if intermediate_size is not None else int(hidden_size),
            "vocab_size": int(vocab_size) if vocab_size is not None else 0,
            "num_experts": int(num_experts) if num_experts is not None else 1,
            "moe_intermediate_size": int(moe_intermediate_size),
            "dtype_bytes": dtype_bytes,
            "kv_seq_len": int(kv_seq_len),
            "train_seq_len": int(train_seq_len),
            "max_num_seqs": int(max_num_seqs),
            "micro_batch_per_gpu": int(mbs_per_gpu),
        }
    except Exception as e:
        logger.warning("Failed to build memory estimator inputs: %s", e)
        return None


def _estimate_kv_cache_gb(*, degrees, estimator: dict[str, Any], dtype_bytes: int) -> float:
    """
    KV cache estimate (vLLM) per rank.

    Approximations:
    - PP is forced to 1 for rollout in this module.
    - KV cache memory is sharded across TP by head dimension partitioning => /TP.
    """
    hs = int(estimator["hidden_size"])
    layers = int(estimator["num_layers"])
    seq_len = int(estimator["kv_seq_len"])
    max_num_seqs = int(estimator["max_num_seqs"])
    tp = int(degrees.tensor_model_parallel_size)

    # k and v: 2 * hidden_size per token, per layer
    kv_bytes = layers * seq_len * max_num_seqs * 2 * hs * dtype_bytes / max(1, tp)
    return float(kv_bytes) / 1024**3


def _estimate_training_memory_gb(*, degrees, estimator: dict[str, Any]) -> float:
    """
    Very rough training memory estimate per rank (weights+optimizer+activations).

    Approximations (conservative defaults):
    - parameters:
      * attention params per layer: ~4 * hs^2
      * MLP params per layer: ~3 * hs * intermediate (gated MLP)
    - if MoE (num_experts > 1):
      * expert params per layer: ~num_experts * 3 * hs * moe_intermediate
      * expert weights are sharded additionally by EP (/EP)
    - optimizer+grads: 4x weights (params + grads + Adam moments)
    - activations: scales with (layers * micro_batch * seq_len * hs) and is sharded by (TP*PP*CP)
    """
    hs = int(estimator["hidden_size"])
    layers = int(estimator["num_layers"])
    intermediate = int(estimator["intermediate_size"])
    moe_intermediate = int(estimator["moe_intermediate_size"])
    vocab_size = int(estimator["vocab_size"])
    num_experts = int(estimator["num_experts"])

    dtype_bytes = int(estimator["dtype_bytes"])
    train_seq_len = int(estimator["train_seq_len"])
    micro_batch = int(estimator["micro_batch_per_gpu"])

    tp = int(degrees.tensor_model_parallel_size)
    pp = int(degrees.pipeline_model_parallel_size)
    cp = int(degrees.context_parallel_size)
    ep = int(degrees.expert_model_parallel_size)

    # Embeddings + LM head params (approx); if vocab_size unknown, skip.
    if vocab_size > 0:
        embed_params = 2 * vocab_size * hs
    else:
        embed_params = 0

    attn_params_per_layer = 4 * hs * hs
    mlp_params_per_layer = 3 * hs * intermediate
    dense_params_per_layer = attn_params_per_layer + mlp_params_per_layer

    # Sharding: per-rank weights divide by (TP*PP). MoE expert weights additionally divide by EP.
    if num_experts > 1:
        # Expert weights dominate the MLP part.
        expert_params_per_layer = num_experts * 3 * hs * moe_intermediate
        dense_params_total = embed_params + layers * attn_params_per_layer
        expert_params_total = layers * expert_params_per_layer

        weights_dense_gb = (dense_params_total * dtype_bytes) / (1024**3 * max(1, tp * pp))
        weights_expert_gb = (expert_params_total * dtype_bytes) / (1024**3 * max(1, tp * pp * ep))
        weights_gb = float(weights_dense_gb + weights_expert_gb)
    else:
        total_params = embed_params + layers * dense_params_per_layer
        weights_gb = (total_params * dtype_bytes) / (1024**3 * max(1, tp * pp))

    # Weights + grads + optimizer moments (roughly 4x params).
    weights_plus_opt_gb = weights_gb * 4.0

    # Activations estimate: conservative multiplier.
    activation_multiplier = 2.0
    # Activations per rank roughly proportional to layers/PP and seq length and sharded by TP/CP.
    activations_bytes = layers * micro_batch * train_seq_len * hs * dtype_bytes * activation_multiplier
    activations_gb = activations_bytes / (1024**3 * max(1, tp * pp * cp))

    return float(weights_plus_opt_gb + activations_gb)


def _select_degrees_maximize_dp(
    *,
    n_gpus_total: int,
    n_gpus_per_node: int,
    nnodes: int,
    base_degrees,
    seq_len: Optional[int],
    memory_budget_gb: Optional[float],
    safety_margin: float,
    memory_estimator: Optional[dict[str, Any]],
    gate_kv_cache: bool = True,
    objective: str = "max_mp",
    allow_context_parallel: bool = False,
    num_attention_heads: Optional[int] = None,
    num_hidden_layers: Optional[int] = None,
    _memory_fallback: bool = False,
):
    """
    Minimal analytical selector:
      - Enumerate TP/PP/CP candidates by divisibility.
      - Derive DP from world size: DP = n_gpus_total / (TP * PP * CP)
      - Enumerate feasible EP candidates (MoE) and validate EP/DP divisibility.
      - Choose the candidate by `objective`:
          * ``max_dp`` (legacy): maximize DP first (throughput-oriented; can override a user TP=8
            down to TP=1 if the analytical memory gate incorrectly passes).
          * ``max_mp`` (default): maximize model-parallel product TP×PP×CP first (prefer sharding),
            which matches typical Megatron + colocated vLLM setups and is safer when the
            analytical estimate is imperfect.

      Tie-breakers (within objective):
          1) Prefer tp <= n_gpus_per_node (within-node, when possible)
          2) Prefer larger CP when seq_len is long (heuristic)
          3) Prefer smaller EP to reduce expert-parallel all-to-all risk
    """
    tp_candidates = sorted(_iter_divisors(n_gpus_total))

    # To limit search, restrict PP and CP to divisors of the remaining factor
    best = None
    best_score = None
    best_kv_gb: Optional[float] = None
    best_train_gb: Optional[float] = None

    tried = 0
    feasible = 0
    mem_pass = 0

    etp = int(base_degrees.expert_tensor_parallel_size or 1)
    existing_sequence_parallel = bool(base_degrees.sequence_parallel)

    long_seq = seq_len is not None and seq_len > 16000
    objective = (objective or "max_mp").strip().lower()
    if objective not in ("max_dp", "max_mp"):
        logger.warning("Unknown trainer.auto_parallel.objective=%r; using 'max_mp'.", objective)
        objective = "max_mp"

    if not allow_context_parallel:
        logger.info(
            "Auto-parallel: allow_context_parallel=false — search restricted to context_parallel_size=1 "
            "(Megatron DotProductAttention requires CP==1 unless TEDotProductAttention is enabled)."
        )

    if num_attention_heads is not None:
        logger.info(
            "Auto-parallel: filtering TP/PP by architecture — num_attention_heads=%d "
            "(Megatron: heads %% TP == 0; PP>1 also requires layers %% PP == 0 when num_hidden_layers known).",
            int(num_attention_heads),
        )

    # EP (MoE) policy:
    # - TP+EP requires sequence parallelism (Megatron Core guidance).
    # - EP is used for expert-parallel token dispatch (all-to-all style).
    #   As a practical heuristic (comm/latency risk reduction), we prefer smaller EP.
    # - We also keep feasibility by enumerating EP as divisors of the user-configured
    #   base EP. If base EP divides the model's num_experts (true by construction in
    #   a valid training config), then any divisor of base EP will also divide num_experts.
    base_ep = int(base_degrees.expert_model_parallel_size)
    if base_ep > 1:
        ep_candidates = sorted(set(_iter_divisors(base_ep)))
        ep_candidates = [e for e in ep_candidates if e > 0]
    else:
        ep_candidates = [1]

    for tp in tp_candidates:
        # Quick EP/DP compatibility check requires dp, which depends on PP and CP.
        # We still keep the search small by ensuring tp divides n_gpus_total.
        if n_gpus_total % tp != 0:
            continue
        if num_attention_heads is not None and int(num_attention_heads) % int(tp) != 0:
            continue

        remaining_after_tp = n_gpus_total // tp
        pp_candidates = sorted(_iter_divisors(remaining_after_tp))

        for pp in pp_candidates:
            if num_hidden_layers is not None and int(pp) > 1 and int(num_hidden_layers) % int(pp) != 0:
                continue
            remaining_after_tp_pp = remaining_after_tp // pp
            cp_candidates = sorted(_iter_divisors(remaining_after_tp_pp))
            if not allow_context_parallel:
                cp_candidates = [c for c in cp_candidates if c == 1]

            for cp in cp_candidates:
                if n_gpus_total % (tp * pp * cp) != 0:
                    continue

                dp = n_gpus_total // (tp * pp * cp)

                for ep in ep_candidates:
                    tried += 1
                    # EP operates INSIDE DP: enforce dp % ep == 0 for EP>1.
                    if ep > 1 and dp % ep != 0:
                        continue

                    # Preserve existing sequence_parallel unless EP+TP requires it.
                    sp_required = (ep > 1 and tp > 1)
                    sequence_parallel = existing_sequence_parallel or sp_required

                    candidate = type(base_degrees)(
                        tensor_model_parallel_size=int(tp),
                        pipeline_model_parallel_size=int(pp),
                        context_parallel_size=int(cp),
                        expert_model_parallel_size=int(ep),
                        expert_tensor_parallel_size=int(etp),
                        sequence_parallel=bool(sequence_parallel),
                        data_parallel_size=int(dp),
                        model_parallel_size=int(tp * pp * cp),
                    )

                    # Validate feasibility (may warn/raise depending on constraints)
                    try:
                        validate_degrees(candidate, n_gpus_total, n_gpus_per_node, nnodes)
                    except ValueError:
                        continue
                    feasible += 1

                    kv_gb: Optional[float] = None
                    train_gb: Optional[float] = None

                    # Analytical memory acceptance (simple + conservative so we don't OOM).
                    if memory_budget_gb is not None and memory_estimator is not None:
                        train_gb = _estimate_training_memory_gb(degrees=candidate, estimator=memory_estimator)
                        kv_gb = None
                        if gate_kv_cache:
                            kv_gb = _estimate_kv_cache_gb(
                                degrees=candidate,
                                estimator=memory_estimator,
                                dtype_bytes=memory_estimator["dtype_bytes"],
                            )
                            # CI_upper surrogate: CI_upper = estimate * safety_margin
                            if kv_gb * safety_margin > memory_budget_gb:
                                continue
                        if train_gb * safety_margin > memory_budget_gb:
                            continue
                    mem_pass += 1

                    # Score depends on objective (see docstring).
                    tp_within_node = int(tp <= n_gpus_per_node)
                    mp = int(tp * pp * cp)
                    if objective == "max_dp":
                        if long_seq:
                            score = (dp, tp_within_node, cp, -int(ep), -tp, -pp, tp, pp, cp, ep)
                        else:
                            score = (dp, tp_within_node, -tp, -int(ep), -pp, pp, cp, ep)
                    else:
                        # max_mp: prefer larger TP×PP×CP, then higher TP (typical LLM preference), lower DP.
                        if long_seq:
                            score = (mp, tp_within_node, cp, -int(ep), tp, -pp, -dp, pp, cp, ep)
                        else:
                            score = (mp, tp_within_node, tp, -int(ep), -pp, -dp, pp, cp, ep)

                    if best is None or score > best_score:
                        best = candidate
                        best_score = score
                        best_kv_gb = kv_gb
                        best_train_gb = train_gb

    if best is None:
        if feasible == 0:
            extra = ""
            if num_attention_heads is not None:
                extra = (
                    f" Check tensor_model_parallel_size divides num_attention_heads={num_attention_heads} "
                    f"(Megatron Core) and PP divides num_hidden_layers when PP>1."
                )
            raise ValueError("No feasible auto-parallel degree configuration found." + extra)
        if _memory_fallback:
            raise ValueError(
                "No feasible auto-parallel degree configuration found after memory fallback "
                f"(feasible={feasible})."
            )
        logger.warning(
            "Auto-parallel: analytical memory gate rejected all %d feasible candidates "
            "(budget_gb=%s). Retrying feasibility-only selection without memory gate using "
            "objective=max_mp (prefer sharding — avoids blindly picking TP=1).",
            feasible,
            f"{memory_budget_gb:.2f}" if memory_budget_gb is not None else "n/a",
        )
        return _select_degrees_maximize_dp(
            n_gpus_total=n_gpus_total,
            n_gpus_per_node=n_gpus_per_node,
            nnodes=nnodes,
            base_degrees=base_degrees,
            seq_len=seq_len,
            memory_budget_gb=None,
            safety_margin=safety_margin,
            memory_estimator=memory_estimator,
            gate_kv_cache=gate_kv_cache,
            objective="max_mp",
            allow_context_parallel=allow_context_parallel,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            _memory_fallback=True,
        )

    logger.info(
        "Auto-parallel selected degrees (objective=%s): %s",
        objective,
        best.get_summary() if hasattr(best, "get_summary") else str(best),
    )
    if memory_budget_gb is not None:
        logger.info(
            "Auto-parallel memory gate: budget_gb=%.2f, kv_est_gb=%s, train_est_gb=%s, safety_margin=%.4f",
            float(memory_budget_gb),
            f"{best_kv_gb:.2f}" if best_kv_gb is not None else "n/a",
            f"{best_train_gb:.2f}" if best_train_gb is not None else "n/a",
            float(safety_margin),
        )
    logger.info(
        "Auto-parallel search stats: tried=%d feasible=%d mem_pass=%d",
        tried,
        feasible,
        mem_pass,
    )
    return best


def _apply_degrees_to_config(config, *, degrees) -> None:
    """
    Override config degree fields in-place.

    Training (Megatron-side):
      - actor_rollout_ref.actor.megatron.*
      - actor_rollout_ref.ref.megatron.* (if present)
      - config.critic.megatron.* (if present)

    Rollout (vLLM-side):
      - actor_rollout_ref.rollout.*

    Note: vLLM/sglang/trtllm rollout currently do not support PP>1; we force PP=1.
    """
    # Training-side overrides (actor)
    config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size = degrees.tensor_model_parallel_size
    config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size = degrees.pipeline_model_parallel_size
    config.actor_rollout_ref.actor.megatron.context_parallel_size = degrees.context_parallel_size
    config.actor_rollout_ref.actor.megatron.expert_model_parallel_size = degrees.expert_model_parallel_size
    config.actor_rollout_ref.actor.megatron.sequence_parallel = bool(degrees.sequence_parallel)

    # Training-side overrides (reference policy), only if present in config
    if "ref" in config.actor_rollout_ref and hasattr(config.actor_rollout_ref.ref, "megatron"):
        config.actor_rollout_ref.ref.megatron.tensor_model_parallel_size = degrees.tensor_model_parallel_size
        config.actor_rollout_ref.ref.megatron.pipeline_model_parallel_size = degrees.pipeline_model_parallel_size
        config.actor_rollout_ref.ref.megatron.context_parallel_size = degrees.context_parallel_size
        config.actor_rollout_ref.ref.megatron.expert_model_parallel_size = degrees.expert_model_parallel_size
        config.actor_rollout_ref.ref.megatron.sequence_parallel = bool(degrees.sequence_parallel)

    # Training-side overrides (critic), only if critic uses megatron
    if hasattr(config, "critic") and getattr(config.critic, "strategy", None) == "megatron" and hasattr(
        config.critic, "megatron"
    ):
        config.critic.megatron.tensor_model_parallel_size = degrees.tensor_model_parallel_size
        config.critic.megatron.pipeline_model_parallel_size = degrees.pipeline_model_parallel_size
        config.critic.megatron.context_parallel_size = degrees.context_parallel_size
        config.critic.megatron.expert_model_parallel_size = degrees.expert_model_parallel_size
        config.critic.megatron.sequence_parallel = bool(degrees.sequence_parallel)

    # Rollout-side overrides (force PP=1)
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = degrees.tensor_model_parallel_size
    config.actor_rollout_ref.rollout.pipeline_model_parallel_size = 1
    config.actor_rollout_ref.rollout.data_parallel_size = degrees.data_parallel_size
    # RolloutConfig has an internal constraint:
    #   if expert_parallel_size > 1:
    #       expert_parallel_size == tensor_model_parallel_size * data_parallel_size
    # So map MoE EP from training degrees into the rollout-compatible setting.
    rollout_ep = int(degrees.expert_model_parallel_size)
    if rollout_ep > 1:
        rollout_ep = int(degrees.tensor_model_parallel_size * degrees.data_parallel_size)
    config.actor_rollout_ref.rollout.expert_parallel_size = rollout_ep


def apply_analytical_auto_parallel_degrees(config) -> None:
    """
    Apply analytical auto-parallelism degree overrides (in-place).

    This is invoked from `verl/trainer/main_ppo.py` and only mutates config
    when `trainer.auto_parallel.enabled=true`.
    """
    from omegaconf import OmegaConf

    enabled = OmegaConf.select(config, "trainer.auto_parallel.enabled", default=False)
    if not enabled:
        return

    n_gpus_per_node = int(config.trainer.n_gpus_per_node)
    nnodes = int(config.trainer.nnodes)
    n_gpus_total = n_gpus_per_node * nnodes

    base_degrees = _get_megatron_degrees_from_actor_config(config=config, n_gpus=n_gpus_total)
    if base_degrees is None:
        return

    num_attention_heads = _try_get_num_attention_heads(config)
    num_hidden_layers = _try_get_num_hidden_layers(config)

    # Heuristic long-sequence threshold: try to allocate more degrees to CP.
    from omegaconf import OmegaConf as _OmegaConf

    actor_seq = _OmegaConf.select(
        config, "actor_rollout_ref.actor.ppo_max_token_len_per_gpu", default=None
    )
    rollout_seq = _OmegaConf.select(config, "actor_rollout_ref.rollout.max_model_len", default=None)
    seq_len = None
    for cand in (actor_seq, rollout_seq):
        try:
            if cand is not None:
                cand_i = int(cand)
                seq_len = cand_i if seq_len is None else max(seq_len, cand_i)
        except Exception:
            pass

    # Select candidate degrees (feasibility-first; objective from config).
    auto_parallel_cfg = _OmegaConf.select(config, "trainer.auto_parallel", default=None)
    if auto_parallel_cfg is None:
        return

    target_utilization = float(auto_parallel_cfg.get("target_utilization", 0.9))
    safety_margin = float(auto_parallel_cfg.get("safety_margin", 1.10))
    memory_check_mode = str(auto_parallel_cfg.get("memory_check_mode", "analytical"))
    objective = str(auto_parallel_cfg.get("objective", "max_mp"))
    respect_config = bool(auto_parallel_cfg.get("respect_config_parallelism", False))
    allow_context_parallel = bool(auto_parallel_cfg.get("allow_context_parallel", False))

    rollout_name = str(
        _OmegaConf.select(config, "actor_rollout_ref.rollout.name", default="vllm") or "vllm"
    ).lower()
    gate_kv_cache = rollout_name == "vllm"

    # Keep user-provided TP/PP/CP/EP/DP: only validate + optional analytical memory check.
    if respect_config:
        if num_attention_heads is not None:
            tp0 = int(base_degrees.tensor_model_parallel_size)
            if int(num_attention_heads) % tp0 != 0:
                raise ValueError(
                    "respect_config_parallelism: Megatron requires num_attention_heads divisible by "
                    f"tensor_model_parallel_size, got heads={num_attention_heads}, TP={tp0} "
                    f"(e.g. Qwen2.5-7B has 28 heads → TP must be in {{1,2,4,7,14,28}}, not 8)."
                )
        if num_hidden_layers is not None and int(base_degrees.pipeline_model_parallel_size) > 1:
            pp0 = int(base_degrees.pipeline_model_parallel_size)
            if int(num_hidden_layers) % pp0 != 0:
                raise ValueError(
                    "respect_config_parallelism: num_hidden_layers must be divisible by "
                    f"pipeline_model_parallel_size when PP>1, got layers={num_hidden_layers}, PP={pp0}."
                )
        if (
            not allow_context_parallel
            and int(getattr(base_degrees, "context_parallel_size", 1) or 1) > 1
        ):
            logger.warning(
                "respect_config_parallelism: context_parallel_size=%d with allow_context_parallel=false "
                "may fail Megatron DotProductAttention (needs TEDotProductAttention for CP>1).",
                int(base_degrees.context_parallel_size),
            )
        validate_degrees(base_degrees, n_gpus_total, n_gpus_per_node, nnodes)
        if memory_check_mode == "analytical":
            memory_budget_gb = _get_memory_budget_gb(target_utilization=target_utilization)
            memory_estimator = _try_build_memory_estimator_inputs(config=config)
            if memory_budget_gb is not None and memory_estimator is not None:
                train_gb = _estimate_training_memory_gb(degrees=base_degrees, estimator=memory_estimator)
                if gate_kv_cache:
                    kv_gb = _estimate_kv_cache_gb(
                        degrees=base_degrees,
                        estimator=memory_estimator,
                        dtype_bytes=memory_estimator["dtype_bytes"],
                    )
                    if kv_gb * safety_margin > memory_budget_gb:
                        raise ValueError(
                            "respect_config_parallelism: configured degrees fail KV memory gate "
                            f"(kv_est_gb={kv_gb:.2f}, budget_gb={memory_budget_gb:.2f}, "
                            f"safety_margin={safety_margin})."
                        )
                if train_gb * safety_margin > memory_budget_gb:
                    raise ValueError(
                        "respect_config_parallelism: configured degrees fail training memory gate "
                        f"(train_est_gb={train_gb:.2f}, budget_gb={memory_budget_gb:.2f}, "
                        f"safety_margin={safety_margin})."
                    )
        _apply_degrees_to_config(config, degrees=base_degrees)
        logger.info(
            "Auto-parallel: respect_config_parallelism=true — kept configured degrees (%s).",
            base_degrees.get_summary() if hasattr(base_degrees, "get_summary") else base_degrees,
        )
        return

    memory_budget_gb: Optional[float] = None
    memory_estimator: Optional[dict[str, Any]] = None

    if memory_check_mode == "analytical":
        memory_budget_gb = _get_memory_budget_gb(target_utilization=target_utilization)
        memory_estimator = _try_build_memory_estimator_inputs(config=config)

        logger.info(
            "Auto-parallel memory inputs: target_utilization=%.3f safety_margin=%.3f budget_gb=%s",
            target_utilization,
            safety_margin,
            f"{memory_budget_gb:.2f}" if memory_budget_gb is not None else "n/a",
        )
        if memory_estimator is None:
            logger.warning(
                "Auto-parallel memory estimator inputs missing; will skip analytical memory checks."
            )
        else:
            # Log only key estimator values (avoid spamming).
            logger.info(
                "Auto-parallel estimator: hidden_size=%d num_layers=%d heads=%d hidden_intermediate=%d dtype_bytes=%d "
                "kv_seq_len=%d train_seq_len=%d max_num_seqs=%d micro_batch_per_gpu=%d num_experts=%d moe_intermediate=%d",
                memory_estimator["hidden_size"],
                memory_estimator["num_layers"],
                memory_estimator["num_heads"],
                memory_estimator["intermediate_size"],
                memory_estimator["dtype_bytes"],
                memory_estimator["kv_seq_len"],
                memory_estimator["train_seq_len"],
                memory_estimator["max_num_seqs"],
                memory_estimator["micro_batch_per_gpu"],
                memory_estimator["num_experts"],
                memory_estimator["moe_intermediate_size"],
            )

    selected = _select_degrees_maximize_dp(
        n_gpus_total=n_gpus_total,
        n_gpus_per_node=n_gpus_per_node,
        nnodes=nnodes,
        base_degrees=base_degrees,
        seq_len=seq_len,
        memory_budget_gb=memory_budget_gb,
        safety_margin=safety_margin,
        memory_estimator=memory_estimator,
        gate_kv_cache=gate_kv_cache,
        objective=objective,
        allow_context_parallel=allow_context_parallel,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
    )

    # Override config degrees (ignore user-provided degrees on the fields we control).
    _apply_degrees_to_config(config, degrees=selected)

    # Hard print so you can see the result even if VERL_LOGGING_LEVEL is not INFO.
    # This is the "ground truth" degrees used for placement-aware scheduling.
    try:
        summary = selected.get_summary() if hasattr(selected, "get_summary") else str(selected)
    except Exception:
        summary = str(selected)
    print(
        f"[AutoParallel] selected degrees={summary} "
        f"(target_utilization={target_utilization}, safety_margin={safety_margin}, "
        f"memory_check_mode={memory_check_mode}, objective={objective})",
        flush=True,
    )
    if memory_budget_gb is not None:
        print(f"[AutoParallel] memory_budget_gb={memory_budget_gb:.4f}GB", flush=True)
    if memory_estimator is not None:
        # Print a compact estimator signature (avoid huge dicts / tensors).
        est_sig = {
            "hidden_size": memory_estimator.get("hidden_size"),
            "num_layers": memory_estimator.get("num_layers"),
            "num_heads": memory_estimator.get("num_heads"),
            "kv_seq_len": memory_estimator.get("kv_seq_len"),
            "train_seq_len": memory_estimator.get("train_seq_len"),
            "micro_batch_per_gpu": memory_estimator.get("micro_batch_per_gpu"),
            "max_num_seqs": memory_estimator.get("max_num_seqs"),
        }
        print(f"[AutoParallel] estimator_signature={est_sig}", flush=True)

