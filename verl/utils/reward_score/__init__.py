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

import importlib.util

from verl.utils.import_utils import deprecated

REWARD_PRESETS: tuple[str, ...] = (
    "gsm8k",
    "math_reward",
    "math_verify",
    "math_dapo",
    "prime_math",
    "prime_code",
    "geo3k",
    "search_r1_like_qa_em",
    "search_r1_like_qa_subem",
)


def _ensure_preset_dependency(preset: str, *, context: str = "reward.preset") -> None:
    """Verify optional runtime dependencies required by a selected preset scorer."""
    if preset == "math_verify" and importlib.util.find_spec("math_verify") is None:
        raise ImportError(
            f"The 'math_verify' reward preset configured in {context} requires the optional 'math-verify' package. "
            "Install it with `pip install math-verify`."
        )


def validate_reward_preset(preset: str, *, context: str = "reward.preset") -> str:
    """Validate that ``preset`` names a supported built-in reward scorer and its dependencies exist."""
    if not isinstance(preset, str) or preset not in REWARD_PRESETS:
        available = ", ".join(REWARD_PRESETS)
        raise ValueError(f"Unknown preset reward function {preset!r} in {context}. Available presets: {available}")
    _ensure_preset_dependency(preset, context=context)
    return preset


def _builtin_preset_for_data_source(data_source: str) -> str | None:
    """Return the built-in preset name matching ``data_source``, or ``None`` if unmapped."""
    if data_source == "openai/gsm8k":
        return "gsm8k"
    if data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        return "math_reward"
    if data_source in ["math_dapo", "math", "math_dapo_reasoning"] or (
        isinstance(data_source, str) and data_source.startswith("aime")
    ):
        return "math_dapo"
    if data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        return "prime_math"
    if data_source in ["codecontests", "apps", "codeforces", "taco"]:
        return "prime_code"
    if data_source in ["hiyouga/geometry3k"]:
        return "geo3k"
    if data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        return "search_r1_like_qa_em"
    return None


def resolve_reward_preset(data_source: str, *, preset: str | None = None) -> str | None:
    """Resolve which built-in preset scorer applies to ``data_source``.

    Any supplied ``preset`` fallback is validated upfront (including optional dependency checks)
    even when ``data_source`` matches a built-in route, so invalid fallback presets never fail
    lazily midway through training.
    """
    validated_fallback = validate_reward_preset(preset) if preset is not None else None
    builtin = _builtin_preset_for_data_source(data_source)
    if builtin is not None:
        return builtin
    return validated_fallback


def _compute_preset_score(
    preset_name: str,
    solution_str: str,
    ground_truth,
    *,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Invoke a validated built-in preset reward module and return its raw result."""
    if preset_name == "gsm8k":
        from . import gsm8k

        return gsm8k.compute_score(solution_str, ground_truth)
    if preset_name == "math_reward":
        from . import math_reward

        return math_reward.compute_score(solution_str, ground_truth)
    if preset_name == "math_verify":
        _ensure_preset_dependency("math_verify", context="preset_name='math_verify'")
        from . import math_verify

        return math_verify.compute_score(solution_str, ground_truth)
    if preset_name == "math_dapo":
        from . import math_dapo

        return math_dapo.compute_score(solution_str, ground_truth)
    if preset_name == "prime_math":
        from . import prime_math

        return prime_math.compute_score(solution_str, ground_truth)
    if preset_name == "prime_code":
        if sandbox_fusion_url:
            from . import sandbox_fusion

            return sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        from . import prime_code

        return prime_code.compute_score(solution_str, ground_truth, continuous=True)
    if preset_name == "geo3k":
        from . import geo3k

        return geo3k.compute_score(solution_str, ground_truth)
    if preset_name == "search_r1_like_qa_em":
        from . import search_r1_like_qa_em

        return search_r1_like_qa_em.compute_score(solution_str, ground_truth)
    if preset_name == "search_r1_like_qa_subem":
        from . import search_r1_like_qa_em

        return search_r1_like_qa_em.compute_score_subem(solution_str, ground_truth)
    raise NotImplementedError(f"Preset reward function is not implemented for {preset_name=}")


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    preset: str | None = None,
    **kwargs,
):
    """Compute the score for a given solution based on ``data_source`` or fallback ``preset``.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.
        preset (str, optional): Fallback built-in preset name when ``data_source`` is not mapped in the
            built-in table.

    Returns:
        float | dict: The computed score as a float, or a dictionary containing ``"score"`` and extra fields.

    Raises:
        NotImplementedError: If no reward function is configured or implemented for ``data_source``.
    """
    resolved_preset = resolve_reward_preset(data_source, preset=preset)
    if resolved_preset is None:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    res = _compute_preset_score(
        resolved_preset,
        solution_str,
        ground_truth,
        sandbox_fusion_url=sandbox_fusion_url,
        concurrent_semaphore=concurrent_semaphore,
        memory_limit_mb=memory_limit_mb,
    )

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


def get_default_compute_score(reward_name: str | None):
    """Get the default compute_score function based on the reward manager type."""
    return default_compute_score


__all__ = [
    "REWARD_PRESETS",
    "default_compute_score",
    "resolve_reward_preset",
    "validate_reward_preset",
]
