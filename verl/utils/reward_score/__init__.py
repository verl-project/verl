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
import os

from verl.utils.import_utils import deprecated


def _code_sandbox_backend():
    backend = os.environ.get("SANDBOX_BACKEND", "kubernetes")
    if backend not in {"kubernetes", "codegym"}:
        raise ValueError("SANDBOX_BACKEND must be 'kubernetes' or 'codegym'")
    return backend


def _code_test_cases_for_prime_code(ground_truth, extra_info):
    if isinstance(extra_info, dict):
        for key in ("prime_code_input_output", "input_output"):
            value = extra_info.get(key)
            if value:
                return value
    return ground_truth


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    continuous=True,
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
    if data_source in [
        "lighteval/MATH",
        "DigitalLearningGmbH/MATH-lighteval",
        "HuggingFaceH4/MATH-500",
        "SynthLabsAI/Big-Math-RL-Verified",
        "zwhe99/DeepMath-103K",
        "deepmath",
        "deepscaler",
        "math500",
        "amc23",
        "amc2023",
        "aime2024",
        "aime2025",
        "aime2026",
        "beyondaime",
        "openai/gsm8k",
        "gsm8k_boxed",
        "dapo_en",
        "hendrycks-math-12k",
    ]:
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        from . import math_verify

        res = math_verify.compute_score(
            solution_str, ground_truth, data_source=data_source
        )
    elif data_source in [
        "mmlu",
        "gpqa_diamond",
        "gpqa",
        "Idavidrein/gpqa",
        "riddle_sense",
        "lexam_mcq",
    ]:
        from . import multiple_choice

        res = multiple_choice.compute_score(
            solution_str, ground_truth, data_source=data_source
        )
    elif data_source in [
        "allenai/IF_multi_constraints_upto5",
        "swiss-ai/if-rl-singleturn-prompts",
        "swiss-ai/if-rl-singleturn-hard-prompts",
        "google/IFEval",
        "allenai/IFBench_test",
    ]:
        from . import instruction_following

        res = instruction_following.compute_score(
            solution_str, ground_truth, extra_info=extra_info, data_source=data_source
        )
    elif data_source in [
        "humaneval",
        "openai/openai_humaneval",
    ]:
        from . import prime_code

        res = prime_code.compute_score(
            solution_str, ground_truth, continuous=True, data_source=data_source
        )
    elif data_source in [
        "taco",
        "likaixin/TACO-verified",
        "lighteval/code_generation_lite",
        "codecontests",
        "deepmind/code_contests",
        "code_contests",
        "apps",
        "codeforces",
    ]:
        # Select code evaluation sandbox backend
        sandbox_backend = _code_sandbox_backend()
        if sandbox_backend == "kubernetes":
            sandbox_url = sandbox_fusion_url or os.environ.get("KUBERNETES_SANDBOX_URL")
            from . import kubernetes_sandbox as code_sandbox
        elif sandbox_backend == "codegym":
            sandbox_url = sandbox_fusion_url or os.environ.get("SCHEDULER_URL")
            from . import codegym_sandbox as code_sandbox
        else:
            sandbox_url = None

        if sandbox_url:
            res = code_sandbox.compute_score(
                data_source=data_source,
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                sandbox_fusion_url=sandbox_url,
                concurrent_semaphore=concurrent_semaphore,
                memory_limit_mb=memory_limit_mb,
                continuous=continuous,
            )
        else:
            from . import prime_code

            test_cases = _code_test_cases_for_prime_code(ground_truth, extra_info)
            res = prime_code.compute_score(
                solution_str, test_cases, continuous=continuous, data_source=data_source
            )
    elif data_source == "rgym":
        from . import rgym

        res = rgym.compute_score(data_source, solution_str, ground_truth, extra_info)
    elif data_source == "qa_gym":
        from . import qa_gym

        res = qa_gym.compute_score(data_source, solution_str, ground_truth, extra_info)
    elif isinstance(data_source, str) and data_source.startswith("tablegpt/"):
        from . import table_gpt

        res = table_gpt.compute_score(
            data_source, solution_str, ground_truth, extra_info
        )
    elif isinstance(data_source, str) and data_source.startswith("blindtasks"):
        from . import blindtasks

        res = blindtasks.compute_score(
            data_source, solution_str, ground_truth, extra_info
        )
    elif isinstance(data_source, str) and data_source.startswith("tool_gym"):
        from . import toolgym

        res = toolgym.compute_score(
            data_source,
            solution_str,
            ground_truth,
            extra_info,
            **kwargs,
        )
    else:
        raise NotImplementedError(
            f"Reward function is not implemented for {data_source}"
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
        data_source,
        solution_str,
        ground_truth,
        extra_info,
        sandbox_fusion_url,
        concurrent_semaphore,
        memory_limit_mb,
    )


def get_default_compute_score(reward_name: str | None):
    """Get the default compute_score function based on the reward manager type."""
    return default_compute_score


__all__ = ["default_compute_score"]
