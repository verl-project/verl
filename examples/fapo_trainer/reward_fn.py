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

import asyncio
import logging
import os

import aiohttp
from transformers import PreTrainedTokenizer

from verl.utils.ray_utils import get_event_loop
from verl.utils.reward_score.math_dapo import last_boxed_only_string, normalize_final_answer, remove_boxed

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def verify(
    solution_str: str,
    gt: str,
) -> tuple[bool, str]:
    solution_str = solution_str[-300:]
    boxed_answer = last_boxed_only_string(solution_str)
    if boxed_answer is not None:
        extracted_answer = remove_boxed(boxed_answer)
    else:
        extracted_answer = "[INVALID]"

    pred = normalize_final_answer(extracted_answer)
    gt = normalize_final_answer(gt)
    return (pred == gt), pred


async def compute_score_baseline(
    solution_str: str,
    ground_truth: str,
    **kwargs,
):
    loop = get_event_loop()
    """Compute the reward score for Baseline."""
    correct, pred = await loop.run_in_executor(None, lambda: verify(solution_str, ground_truth))
    reward_score = 1.0 if correct else -1.0
    return {"score": reward_score, "acc": correct, "pred": pred}


# FAPO Hyper-parameters
FAPO_GENRM_TEMPLATE = (
    "The following is a math problem with its ground truth answer, along with an AI solution (split into steps):\n\n"
    "[Math Problem]\n\n"
    "{problem}\n\n"
    "[Ground Truth]\n\n"
    "{ground_truth}\n\n"
    "[AI Solution]\n\n"
    "{solution}\n\n"
    "Your task is to review and critique the solution step by step. "
    "Once you identify an error in a step, return the index of the step where the earliest error occurs. "
    "Otherwise, return the index of -1 (which typically denotes 'not found').\n\n"
    "Please reason step by step, put your final answer (i.e., the index) in \\boxed{{}}."
)
MAX_TOKENS = 16384
FLAWED_REWARD_PENALTY = 1.0


# async def generate_aiohttp(router_address: str, prompt_ids: list[int], sampling_params: dict):
#     payload = {
#         "input_ids": prompt_ids,
#         "sampling_params": sampling_params,
#     }
#     url = f"http://{router_address}/generate"
#     try:
#         session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
#         async with session.post(url, json=payload) as resp:
#             output = await resp.text()
#             try:
#                 output = json.loads(output)
#                 return output
#             except Exception:
#                 logger.error(f"Failed to parse JSON response: {output}")
#                 return {}
#     finally:
#         await session.close()


async def post_request(router_address: str, payload: dict, endpoint: str, max_retries: int = 5):
    url = f"http://{router_address}/{endpoint}"
    last_exception = None
    for attempt in range(max_retries):
        try:
            # It's safer to have a timeout instead of None, which can hang indefinitely.
            timeout = aiohttp.ClientTimeout(total=None)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as resp:
                    resp.raise_for_status()
                    return await resp.json()
        except aiohttp.ClientResponseError as e:
            # Do not retry on 4xx client errors, but retry on 5xx server errors.
            if 400 <= e.status < 500:
                logger.error(f"Request to {url} failed with client error HTTP {e.status}: {e}. Not retrying.")
                raise
            last_exception = e
            logger.warning(
                f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed with HTTP {e.status}: {e}. Retrying..."
            )
        except (asyncio.TimeoutError, aiohttp.ClientConnectorError) as e:
            last_exception = e
            logger.warning(f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed: {e}. Retrying...")
        except Exception as e:
            last_exception = e
            logger.warning(
                f"[Attempt {attempt + 1}/{max_retries}] Request to {url} failed with unexpected error: {e}. Retrying..."
            )

        if attempt < max_retries - 1:
            # Using exponential backoff is generally better than a fixed sleep.
            backoff_seconds = 2**attempt
            await asyncio.sleep(min(backoff_seconds, 30))

    logger.error(f"Max retries ({max_retries}) reached for request to {url}.")
    if last_exception:
        raise last_exception


async def compute_score_fapo(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer,
):
    """Compute the reward score for FAPO."""
    loop = get_event_loop()

    question, split = extra_info["question"], extra_info["split"]
    correct, pred = await loop.run_in_executor(None, lambda: verify(solution_str, ground_truth))
    reward_score = 1.0 if correct else -1.0
    is_flawed_positive = False

    # for test set or incorrect solution, directly return the reward score
    if split == "test" or not correct:
        return {"score": reward_score, "acc": correct, "pred": pred, "is_flawed_positive": is_flawed_positive}

    grm_prompt = FAPO_GENRM_TEMPLATE.format(
        problem=question,
        ground_truth=ground_truth,
        solution=solution_str,
    )
    messages = [{"role": "user", "content": grm_prompt}]
    grm_outputs = await post_request(
        router_address=reward_router_address,
        payload={
            "messages": messages,
            "max_tokens": MAX_TOKENS,
        },
        endpoint="v1/chat/completions",
    )
    grm_response = grm_outputs["choices"][0]["message"]["content"]
    try:
        err_location = remove_boxed(last_boxed_only_string(grm_response))
        is_flawed_positive = int(err_location) != -1
    except Exception:
        is_flawed_positive = False

    if is_flawed_positive:
        reward_score -= FLAWED_REWARD_PENALTY

    return {"score": reward_score, "acc": correct, "pred": pred, "is_flawed_positive": is_flawed_positive}
