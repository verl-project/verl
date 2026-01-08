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

import base64
import json
import os
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import aiohttp
import numpy as np
import torch
from openai.types.chat import ChatCompletion
from PIL import Image
from transformers import PreTrainedTokenizer

GRM_PROMPT_TEMPLATE = """
You are given a problem and a proposed solution.

Problem:
{problem}

Solution:
{solution}

Please evaluate how well the solution addresses the problem.
Give a score from 1 to 10, where:
- 1 means the solution is completely irrelevant or incorrect.
- 5 means the solution is partially correct but incomplete or not well reasoned.
- 10 means the solution is fully correct, well-reasoned, and directly solves the problem.

Only output the score as a single number (integer).
""".strip()


async def chat_complete(router_address: str, chat_complete_request: dict):
    url = f"http://{router_address}/v1/chat/completions"
    try:
        timeout = aiohttp.ClientTimeout(total=None)
        session = aiohttp.ClientSession(timeout=timeout)
        async with session.post(url, json=chat_complete_request) as resp:
            output = await resp.text()
            output = json.loads(output)
            return ChatCompletion(**output)
    except Exception as e:
        raise e
    finally:
        await session.close()


async def compute_score_gsm8k(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer,
):
    """Compute the reward score."""

    grm_prompt = GRM_PROMPT_TEMPLATE.format(problem=extra_info["question"], solution=solution_str)
    messages = [{"role": "user", "content": grm_prompt}]
    sampling_params = {"temperature": 0.7, "top_p": 0.8, "max_tokens": 4096}
    model_name = os.path.expanduser("~/models/Qwen/Qwen2.5-1.5B-Instruct")
    chat_complete_request = {
        "messages": messages,
        "model": model_name,
        **sampling_params,
    }
    result = await chat_complete(
        router_address=reward_router_address,
        chat_complete_request=chat_complete_request,
    )
    grm_response = result.choices[0].message.content
    try:
        score = int(grm_response.split("\n\n")[-1].strip())
    except Exception:
        score = 0
    return {"score": score, "acc": score == 10, "genrm_response": grm_response}


def compute_score_math_verify(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    **kwargs,
):
    """Compute the reward score."""
    from verl.utils.reward_score.math_verify import compute_score

    return compute_score(
        model_output=solution_str,
        ground_truth=ground_truth,
    )


def _pil_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded_image_text = base64.b64encode(buffered.getvalue()).decode("utf-8")
    base64_image = f"data:image;base64,{encoded_image_text}"
    return base64_image


async def compute_score_ocr(
    data_source: str,
    solution_str: Image.Image | np.ndarray | torch.Tensor,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer = None,
):
    """Compute the reward score."""
    import re

    import Levenshtein

    from verl.utils.ray_utils import get_event_loop

    # preprocess image to base64
    image = solution_str
    if isinstance(image, torch.Tensor):
        image = image.float().permute(1, 2, 0).cpu().numpy()
    if isinstance(image, np.ndarray):
        assert image.shape[-1] == 3, "must be in HWC format"
        image = (image * 255).round().clip(0, 255).astype(np.uint8)
        image = Image.fromarray(image)
    assert isinstance(image, Image.Image)

    with ThreadPoolExecutor(max_workers=1) as executor:
        image_base64 = await get_event_loop().run_in_executor(executor, _pil_image_to_base64, image)

    # prepare chat template
    grm_prompt = "Please output only the text content from the image without any additional descriptions or formatting."
    query = [
        {
            "type": "image_url",
            "image_url": {"url": image_base64},
        },
        {"type": "text", "text": grm_prompt},
    ]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": query,
        },
    ]

    sampling_params = {"temperature": 0.7, "top_p": 0.8, "max_tokens": 4096}
    model_name = os.path.expanduser("~/models/Qwen/Qwen2.5-1.5B-Instruct")
    chat_complete_request = {
        "messages": messages,
        "model": model_name,
        **sampling_params,
    }
    result = await chat_complete(
        router_address=reward_router_address,
        chat_complete_request=chat_complete_request,
    )
    grm_response = result.choices[0].message.content

    # compute OCR score
    text = grm_response
    # remove any nonvisible characters and convert to lowercase
    gt = re.sub(r"\s+", "", ground_truth).lower()
    text = re.sub(r"\s+", "", text).lower()
    if gt in text:
        dist = 0
    else:
        dist = Levenshtein.distance(text, gt)

    # recognized many unrelated characters, only add one character penalty
    dist = min(dist, len(gt))
    score = 1 - dist / len(gt)

    return {"score": score, "acc": score == 1, "genrm_response": grm_response}
