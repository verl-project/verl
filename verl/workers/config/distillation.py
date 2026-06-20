# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from verl.base_config import BaseConfig
from verl.utils.config import omega_conf_to_dataclass

from .rollout import RolloutConfig

__all__ = [
    "DistillationLossConfig",
    "DistillationTeacherModelConfig",
    "SelfDistillationConfig",
    "DistillationConfig",
]

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class DistillationLossConfig(BaseConfig):
    """Configuration for distillation loss settings.

    loss_mode (str):
        Distillation loss function to use.
    topk (int, optional):
        Number of top tokens to consider for top-k distillation losses.
    use_task_rewards (bool):
        Whether to include task rewards alongside distillation loss.
    distillation_loss_coef (float):
        Coefficient for distillation loss when combined with task rewards.
    loss_max_clamp (float, optional):
        Maximum value to clamp distillation loss. If None, no clamping is applied.
    log_prob_min_clamp (float, optional):
        Minimum value to clamp log probabilities for stability, e.g., log q - log p where p or q are
        very close to zero. If None, no clamping is applied.
    use_policy_gradient (bool):
        Whether to incorporate distillation loss as a reward, as done
        by https://thinkingmachines.ai/blog/on-policy-distillation/. Recommended to use loss_mode=k1.
        Otherwise, distillation loss is directly backpropagated as a supervised loss,
        as in https://arxiv.org/abs/2306.13649. Recommended to use loss_mode=k3 or forward_kl_topk.
    policy_loss_mode (str):
        Name of the policy loss to use when use_policy_gradient is true.
    clip_ratio (float):
        PPO clipping ratio for policy loss.
    clip_ratio_low (float):
        Lower bound for PPO clipping ratio.
    clip_ratio_high (float):
        Upper bound for PPO clipping ratio.
    loss_settings (DistillationLossSettings, optional):
        Runtime-populated settings based on loss_mode. Not set by user.
    """

    loss_mode: str = "k3"
    topk: Optional[int] = 128
    use_task_rewards: bool = True
    distillation_loss_coef: float = 1.0
    loss_max_clamp: Optional[float] = 10.0
    log_prob_min_clamp: Optional[float] = -10.0

    # OPSD per-vocab pointwise KL clip τ (arXiv:2601.18734 §4.3.3). When set, each
    # per-vocab divergence contribution is clamped to <= clip_tau BEFORE the vocab
    # sum, so a few stylistic tokens cannot dominate the signal. None == off.
    # Distinct from loss_max_clamp (whole-position, post-sum). FSDP forward_kl_topk
    # only in cut 1; Megatron raises NotImplementedError if set.
    clip_tau: Optional[float] = None

    # OPSD reference parity (siyan-zhao/OPSD) knobs.
    #
    # clamp_negative_to_zero: when True (default, original verl behavior) the per-token
    #   top-k divergence is floored at 0 (``distillation_losses.clamp_min(0.0)``). The
    #   reference OPSD does NOT floor it: with per-vocab ``clip_tau`` the per-position sum
    #   is allowed to go negative, which keeps gradient flowing on tokens where the student
    #   already over-covers the teacher. Set False to reproduce the reference.
    # loss_temperature: softmax temperature applied to BOTH student and teacher logits before
    #   the top-k KL, matching the reference (``student_logits/T``, ``teacher_logits/T``).
    #   None (default) == no scaling (T=1). The reference 1.7B run used 1.1. Only consumed by
    #   ``loss_mode='forward_kl_topk'`` on the FSDP backend.
    clamp_negative_to_zero: bool = True
    loss_temperature: Optional[float] = None

    # Chunked top-K log-probs (opt-in, avoids [B, T, V] log_softmax buffer
    # at long context). Only consumed by ``loss_mode='forward_kl_topk'``.
    # Default ``False`` to preserve short-context performance (chunked path
    # has ~6x time overhead at N=14K, V=152K). Set ``True`` when hitting OOM
    # at long context (>=64K tokens, V=152K) where the baseline path OOMs.
    use_chunked_topk: bool = False
    # Tokens per chunk along (B*T) when ``use_chunked_topk=True``. Larger
    # chunks reduce kernel-launch overhead but increase per-chunk memory;
    # smaller chunks reduce per-chunk memory but increase kernel-launch
    # overhead (saved-tensor total stays constant either way).
    chunked_topk_chunk_size: int = 4096

    use_policy_gradient: bool = True
    policy_loss_mode: str = "vanilla"
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    # Store distillation loss settings for computing the specified loss_mode
    # Not set by user, populated at runtime
    loss_settings: Optional[dict] = None

    def __post_init__(self):
        self._mutable_fields.add("loss_settings")
        from verl.trainer.distillation.losses import DistillationLossSettings, get_distillation_loss_settings

        self.loss_settings: DistillationLossSettings = get_distillation_loss_settings(self.loss_mode)

        if self.policy_loss_mode != "vanilla":
            raise NotImplementedError(
                f"Only vanilla policy loss is currently supported when use_policy_gradient is True, "
                f"but got {self.policy_loss_mode}."
            )

        if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
            print(
                "WARNING: forward_kl_topk is most effective as a supervised distillation loss "
                "(use_policy_gradient=False). With policy gradient, the update uses only the sampled"
                " token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits "
                "should move) is largely unused."
            )

        if not self.use_policy_gradient and self.loss_mode == "k1":
            raise ValueError(
                "Directly backpropagating k1 loss is incorrect since gradient of k1 loss"
                " wrt model weights does not depend on teacher log probabilities."
            )


@dataclass
class DistillationTeacherModelConfig(BaseConfig):
    """Configuration for on-policy distillation teacher.

    key (str, optional):
        Identifier to route examples to the teacher model in multi-teacher setting.
    model_path (str, optional):
        Model path for the teacher model. Can be a local path or a Hugging Face model
    inference (RolloutConfig):
        Rollout configuration for the teacher model inference during distillation.
    num_replicas (int):
        Number of inference replicas of this teacher to launch. Each replica occupies
        `per_replica_world_size` GPUs (= inference.data_parallel_size *
        inference.tensor_model_parallel_size * inference.pipeline_model_parallel_size),
        so the teacher's total GPU footprint is
        `num_replicas * per_replica_world_size`.
    """

    _mutable_fields = BaseConfig._mutable_fields | {"num_replicas", "key"}

    key: Optional[str] = None
    model_path: Optional[str] = None
    inference: RolloutConfig = field(default_factory=RolloutConfig)
    num_replicas: Optional[int] = 0

    @property
    def per_replica_world_size(self) -> int:
        return (
            self.inference.tensor_model_parallel_size
            * self.inference.data_parallel_size
            * self.inference.pipeline_model_parallel_size
        )

    @property
    def world_size(self) -> int:
        return self.num_replicas * self.per_replica_world_size

    def check_configured(self):
        if self.model_path is None:
            raise ValueError("model_path must be specified for distillation teacher model config.")
        if self.key is None:
            raise ValueError("key must be specified for distillation teacher model config.")
        if self.num_replicas is None:
            raise ValueError("num_replicas must be specified for distillation teacher model config.")

    def validate_and_prepare_for_distillation(
        self, use_topk: bool, topk: Optional[int], privileged_extra: int = 0
    ) -> None:
        # Prompt + Response from student are fed into teacher as context.
        # For OPSD the teacher context is the privileged [x, y*, bridge, ŷ], so reserve
        # `privileged_extra` (= max_reference_length + max_bridge_length) extra tokens.
        # privileged_extra=0 (plain OPD / non-OPSD) reproduces the original arithmetic.
        max_model_len = self.inference.max_model_len
        student_prompt_length = self.inference.prompt_length
        student_response_length = self.inference.response_length
        required_context_len = student_prompt_length + privileged_extra + student_response_length + 1
        if max_model_len is not None and required_context_len > max_model_len:
            raise ValueError(
                "Distillation teacher inference requires room for the student prompt, the privileged "
                "context (y* + bridge), the full student response, and one generated token, but got "
                f"{student_prompt_length=}, {privileged_extra=}, {student_response_length=}, "
                f"{required_context_len=}, {max_model_len=}."
            )
        self.inference.prompt_length = self.inference.prompt_length + privileged_extra + self.inference.response_length
        self.inference.response_length = 1
        self._validate_topk_logprobs(use_topk=use_topk, topk=topk)

    def _validate_topk_logprobs(self, use_topk: bool, topk: Optional[int]) -> None:
        if not use_topk:
            return
        if topk is None:
            raise ValueError("topk must be specified when use_topk is True.")

        engine_name = self.inference.name
        engine_kwargs = self.inference.engine_kwargs
        match engine_name:
            case "vllm":
                vllm_engine_kwargs = dict(engine_kwargs.get("vllm", {}))
                max_logprobs = vllm_engine_kwargs.get("max_logprobs")
                if max_logprobs is None:
                    vllm_engine_kwargs["max_logprobs"] = topk
                    max_logprobs = topk
                if max_logprobs < topk:
                    raise ValueError(
                        f"VLLM max_logprobs ({max_logprobs}) must be >= distillation_loss topk "
                        f"({topk}) to enable distillation loss computation."
                    )
                engine_kwargs["vllm"] = vllm_engine_kwargs
            case "sglang":
                # SGLang's top_logprobs_num is a per-request parameter, so there is no
                # engine-boot cap to align (unlike vLLM's max_logprobs). The async
                # server translates sampling_params["prompt_logprobs"] into
                # return_logprob + logprob_start_len=0 + top_logprobs_num at call time.
                pass
            case _:
                raise NotImplementedError(
                    f"DistillationTeacherModelConfig does not support inference engine {engine_name}"
                )


@dataclass
class SelfDistillationConfig(BaseConfig):
    """On-Policy Self-Distillation (OPSD, arXiv:2601.18734): teacher == the same
    model under a privileged context [x, y*, bridge]. See docs/algo/opsd.md.

    enabled (bool):
        Route teacher logprobs through the privileged-context builder instead of
        the plain OPD context. Requires ``distillation.enabled=True``.
    teacher_weights (str): "frozen" | "live".
        "frozen" (DEFAULT, reproduces the paper §4.1): teacher == the INITIAL
            checkpoint theta_0, served by the existing OPD teacher pool. Set
            ``distillation.teacher_models.<name>.model_path`` to the SAME base
            checkpoint as the student init, and keep a non-empty teacher pool.
        "live" (Algorithm-1 theory): teacher == the CURRENT student weights,
            served by the rollout engine. Requires n_gpus_per_node=nnodes=0 plus
            the trainer-level control flow (need_teacher_policy->False, the
            trainer-base pool guard gated off, teacher_client=None).
    reference_key (str):
        Per-sample non-tensor field holding the privileged solution y*. Dotted
        keys traverse nested dicts; the default ``reward_model.ground_truth``
        reads ``non_tensor_batch["reward_model"]`` (a per-sample dict) and indexes
        its ``ground_truth`` entry — the path the reward managers use.
    problem_key (str):
        Per-sample non-tensor field holding the raw problem text x. The teacher
        prompt is rebuilt from (problem, solution) via the chat template (see
        teacher_template), matching the reference impl — the privileged context is
        a proper user turn, NOT y* appended after the student's templated prompt.
    teacher_template (str):
        Teacher user-message template with ``{problem}`` and ``{solution}`` fields,
        rendered through ``tokenizer.apply_chat_template(add_generation_prompt=True,
        enable_thinking=teacher_thinking)``. Default is the reference impl's text.
    teacher_thinking (bool):
        ``enable_thinking`` passed to the teacher's chat template (Qwen3). Default
        True (the reference default; the student rolls out with thinking off).
    max_reference_length (int):
        Token budget for the reference solution y* inside the teacher context; the
        solution is truncated to this many tokens. The teacher's max_model_len /
        prompt_length sizing is extended by this + max_bridge_length so the
        privileged [teacher_prompt, ŷ] sequence is not silently truncated.
    max_bridge_length (int):
        Token budget for the template/transition text (everything in the teacher
        prompt other than the problem and the truncated solution).
    """

    enabled: bool = False
    teacher_weights: str = "frozen"  # paper default; "live" == Algorithm-1 theory
    reference_key: str = "reward_model.ground_truth"
    problem_key: str = "problem"
    # Reference impl's teacher user message (siyan-zhao/OPSD data_collator.py, non-reason_first).
    # {problem}/{solution} are substituted; \boxed{} is emitted literally after .format().
    teacher_template: str = (
        "Problem: {problem}\n\n"
        "Here is a reference solution to this problem:\n"
        "=== Reference Solution Begin ===\n{solution}\n=== Reference Solution End ===\n"
        "\n\nAfter reading the reference solution above, make sure you truly understand the "
        "reasoning behind each step — do not copy or paraphrase it. Now, using your own words "
        "and independent reasoning, derive the same final answer to the problem above. Think "
        "step by step, explore different approaches, and don't be afraid to backtrack or "
        "reconsider if something doesn't work out:\n"
        "\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    )
    teacher_thinking: bool = True
    max_reference_length: int = 2048
    max_bridge_length: int = 256

    def __post_init__(self):
        if self.enabled and self.teacher_weights not in ("frozen", "live"):
            raise ValueError(
                f"self_distillation.teacher_weights must be 'frozen' or 'live', got {self.teacher_weights!r}."
            )

    @property
    def privileged_extra(self) -> int:
        """Extra teacher-context tokens reserved for [y*, bridge] when OPSD is on."""
        return (self.max_reference_length + self.max_bridge_length) if self.enabled else 0


@dataclass
class DistillationConfig(BaseConfig):
    """Configuration for on-policy distillation.

    enabled (bool):
        Whether on-policy distillation is enabled.
    n_gpus_per_node (int):
        Number of GPUs per node in the teacher resource pool.
    nnodes (int):
        Number of nodes in the teacher resource pool.
    teacher_models (dict[str, TeacherModelConfig]):
        Configurations for teacher models used for multi-teacher distillation.
    teacher_key (str):
        Key to route examples to the appropriate teacher model in multi-teacher setups. Should correspond to a field in
        the data proto, e.g., data_source.
    distillation_loss (DistillationLossConfig):
    Configuration for distillation loss settings.

    NOTE: The `teacher_model` entry is in the `teacher_models` dict by default.
    Since it is popped when other teacher entries are added, using `teacher_model` as
    one of several keys silently drops it. For example, the following CLI overrides result
    in ONLY `teacher_model2` being used:

    ```bash
    distillation.teacher_models.teacher_model.key=openai/gsm8k
    distillation.teacher_models.teacher_model.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    Instead, give the first teacher a different name:

    ```bash
    +distillation.teacher_models.teacher_model1.key=openai/gsm8k
    +distillation.teacher_models.teacher_model1.model_path=Qwen/Qwen3-4B
    +distillation.teacher_models.teacher_model2.key=hiyouga/geometry3k
    +distillation.teacher_models.teacher_model2.model_path=Qwen/Qwen3-VL-4B-Instruct
    ```
    """

    _mutable_fields = BaseConfig._mutable_fields | {"teacher_models", "n_gpus_per_node", "nnodes"}

    enabled: bool = False
    n_gpus_per_node: int = 0
    nnodes: int = 0
    teacher_models: dict[str, DistillationTeacherModelConfig] = field(default_factory=dict)
    teacher_key: str = "data_source"
    distillation_loss: DistillationLossConfig = field(default_factory=DistillationLossConfig)
    self_distillation: SelfDistillationConfig = field(default_factory=SelfDistillationConfig)

    def __post_init__(self):
        if not self.enabled:
            return

        sd = self.self_distillation
        if sd.enabled and sd.teacher_weights == "live":
            # OPSD live: teacher is the rollout model itself; there is no separate
            # teacher pool, so skip the pool-sizing resolution/sum below. The 3
            # trainer-level sites (need_teacher_policy, the trainer-base pool guard,
            # MultiTeacherModelManager wiring) and the rollout-engine max_logprobs /
            # max_model_len validation are handled at trainer config-resolution time,
            # NOT here. The rollout engine serves p_T at the current weights.
            if self.n_gpus_per_node != 0 or self.nnodes != 0:
                raise ValueError(
                    "self_distillation.teacher_weights='live' uses the rollout model as the teacher; "
                    f"the teacher pool must be empty, but got {self.n_gpus_per_node=}, {self.nnodes=}."
                )
            return

        # Plain OPD, OR OPSD frozen (teacher == theta_0 on the OPD pool, with
        # teacher_models[*].model_path = student base). Frozen reuses the pool path
        # unchanged, only sizing the teacher context for the privileged sequence.
        self.teacher_models = self._resolve_teacher_models()
        teacher_world_size_sum = 0
        for teacher_model in self.teacher_models.values():
            teacher_model.validate_and_prepare_for_distillation(
                use_topk=self.distillation_loss.loss_settings.use_topk,
                topk=self.distillation_loss.topk,
                privileged_extra=sd.privileged_extra,
            )
            teacher_world_size_sum += teacher_model.world_size
        total_pool_size = self.n_gpus_per_node * self.nnodes
        if teacher_world_size_sum != total_pool_size:
            raise ValueError(
                f"Sum of teacher (num_replicas * per_replica_world_size) ({teacher_world_size_sum}) must match "
                f"the distillation resource pool size "
                f"({self.n_gpus_per_node=} * {self.nnodes=} = {total_pool_size})."
            )

    def _resolve_teacher_models(self) -> dict[str, DistillationTeacherModelConfig]:
        assert "teacher_model" in self.teacher_models
        if len(self.teacher_models) == 1:
            # Single teacher occupies the entire teacher resource pool.
            teacher_model = self.teacher_models["teacher_model"]
            inference = teacher_model.inference
            per_replica = (
                inference.tensor_model_parallel_size
                * inference.data_parallel_size
                * inference.pipeline_model_parallel_size
            )
            pool_size = self.n_gpus_per_node * self.nnodes
            if pool_size % per_replica != 0:
                raise ValueError(
                    f"Single teacher's per_replica_world_size ({per_replica}) must divide the distillation "
                    f"resource pool size ({self.n_gpus_per_node=} * {self.nnodes=} = {pool_size})."
                )
            teacher_model.num_replicas = pool_size // per_replica
            teacher_model.key = "default"
        else:
            # Multiple teachers: remove default single teacher config
            self.teacher_models.pop("teacher_model")

        # Teacher models dict is keyed by teacher_key instead of YAML entry name
        teacher_models = {}
        for teacher_config in self.teacher_models.values():
            teacher_config = omega_conf_to_dataclass(teacher_config, dataclass_type=DistillationTeacherModelConfig)
            teacher_config.check_configured()
            if teacher_config.key in teacher_models:
                raise ValueError(f"Duplicate teacher key {teacher_config.key} found in teacher models.")
            teacher_models[teacher_config.key] = teacher_config
        return teacher_models
