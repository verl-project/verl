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
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated

_GSM8K_DATA_SOURCE = "openai/gsm8k"
_GSM8K_PREPROCESS_CMD = "python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k"

_SUPPORTED_REWARD_DATA_SOURCES = {
    _GSM8K_DATA_SOURCE: "GSM8K rule reward",
    "lighteval/MATH": "MATH rule reward",
    "DigitalLearningGmbH/MATH-lighteval": "MATH rule reward",
    "HuggingFaceH4/MATH-500": "MATH rule reward",
    "math_dapo": "DAPO math rule reward",
    "math": "DAPO math rule reward",
    "math_dapo_reasoning": "DAPO math rule reward",
    "aime*": "AIME-style math rule reward",
    "numina_aops_forum": "Prime math reward",
    "numina_synthetic_math": "Prime math reward",
    "numina_amc_aime": "Prime math reward",
    "numina_synthetic_amc": "Prime math reward",
    "numina_cn_k12": "Prime math reward",
    "numina_olympiads": "Prime math reward",
    "codecontests": "Code reward",
    "apps": "Code reward",
    "codeforces": "Code reward",
    "taco": "Code reward",
    "hiyouga/geometry3k": "Geo3K rule reward",
    "searchR1_nq": "SearchR1 QA reward",
    "searchR1_triviaqa": "SearchR1 QA reward",
    "searchR1_popqa": "SearchR1 QA reward",
    "searchR1_hotpotqa": "SearchR1 QA reward",
    "searchR1_2wikimultihopqa": "SearchR1 QA reward",
    "searchR1_musique": "SearchR1 QA reward",
    "searchR1_bamboogle": "SearchR1 QA reward",
}


def _format_supported_data_sources() -> str:
    return ", ".join(sorted(_SUPPORTED_REWARD_DATA_SOURCES))


def _looks_like_gsm8k_or_path(data_source) -> bool:
    if data_source is None:
        return True
    data_source_str = str(data_source).strip()
    if not data_source_str:
        return True
    normalized = data_source_str.lower()
    return "gsm8k" in normalized or data_source_str.startswith(("/", "./", "../", "~")) or "\\" in data_source_str


def _raise_unsupported_data_source(data_source):
    message_parts = [
        f"Reward function is not implemented for data_source={data_source!r}.",
        f"Supported built-in reward data sources include: {_format_supported_data_sources()}.",
    ]
    if _looks_like_gsm8k_or_path(data_source):
        message_parts.append(
            "If you are running the GSM8K quickstart, regenerate the parquet files with "
            f"`{_GSM8K_PREPROCESS_CMD}` and keep the emitted "
            f"`data_source` value as `{_GSM8K_DATA_SOURCE}`. Do not replace it with a local dataset path "
            "or mirror name."
        )
    message_parts.append(
        "For custom datasets or custom data_source values, configure "
        "`reward.custom_reward_function.path` and `reward.custom_reward_function.name`."
    )
    raise NotImplementedError(" ".join(message_parts))


def _is_aime_data_source(data_source) -> bool:
    return isinstance(data_source, str) and data_source.startswith("aime")


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == _GSM8K_DATA_SOURCE:
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        from . import math_reward

        res = math_reward.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        # from . import math_verify
        # res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source in ["math_dapo", "math", "math_dapo_reasoning"] or _is_aime_data_source(data_source):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
    ]:
        from . import prime_math

        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        _raise_unsupported_data_source(data_source)

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


def default_compute_score_image(
    data_source,
    solution_image,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    **kwargs,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_image (Image.Image or torch.Tensor): The solution image to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """
    if data_source == "jpeg_compressibility":
        from . import jpeg_compressibility

        res = jpeg_compressibility.compute_score(solution_image)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

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
    if reward_name == "visual":
        return default_compute_score_image
    else:
        return default_compute_score


__all__ = ["default_compute_score"]
