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

import json
import random
import re
from collections import Counter


def match_score(list1, list2):
    """Compute a similarity score considering element frequency, ignoring order.

    Reference: Liu S Y, Dong X, Lu X, et al. "Gdpo: Group reward-decoupled normalization policy
    optimization for multi-reward rl optimization."
    arXiv preprint arXiv:2601.05242, 2026.
    """
    if list1 == list2:
        return 1.0

    if not list1 or not list2:
        return 0.0

    count1 = Counter(list1)  # Frequency count for list1
    count2 = Counter(list2)  # Frequency count for list2

    intersection = sum(min(count1[k], count2[k]) for k in count1.keys() & count2.keys())
    max_possible = len(list1) + len(list2) - intersection

    return intersection / max_possible if max_possible > 0 else 0.0


# custoimzed reward functions: format
def customize_format_reward_func(
    completions, answer, step, max_possible_reward, min_possible_reward, do_print, **kwargs
):
    rewards = []
    responses = [completion[0]["content"] for completion in completions]

    if do_print:
        print("\n======= Answer ======= ")
        print(answer[0])
        print("\n======= Responses ======= ")
        for idx, response in enumerate(responses):
            print(f"*** Response {idx + 1}***\n{response}")

    for response, ans in zip(responses, answer, strict=False):
        reward = min_possible_reward
        if "<response>" in ans and "<tool_call>" not in ans:
            pattern = r"^<think>.*?</think>\n<response>.*?</response>$"
            if (
                re.search(pattern, response, re.DOTALL)
                and response.count("<response>") == 1
                and response.count("</response>") == 1
            ):
                reward = max_possible_reward
        elif "<response>" not in ans and "<tool_call>" in ans:
            pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>$"
            if (
                re.search(pattern, response, re.DOTALL)
                and response.count("<tool_call>") == 1
                and response.count("</tool_call>") == 1
            ):
                reward = max_possible_reward
        elif "<response>" in ans and "<tool_call>" in ans:
            pattern = r"^<think>.*?</think>\n<tool_call>\n.*?\n</tool_call>\n<response>.*?</response>$"
            if (
                re.search(pattern, response, re.DOTALL)
                and response.count("<tool_call>") == 1
                and response.count("</tool_call>") == 1
                and response.count("<response>") == 1
                and response.count("</response>") == 1
            ):
                reward = max_possible_reward
        else:
            pattern = r"^<think>.*?</think>$"
            if re.search(pattern, response, re.DOTALL):
                reward = max_possible_reward

        rewards.append(reward)

    if do_print:
        print("\n======= Reward for <format> =======")
        print("Reward function for <format> is called ...")
        print(rewards)

    return rewards


def compute_tool_call_reward(gt_tools, pd_tools, max_possible_reward, min_possible_reward, do_print):
    if gt_tools == pd_tools:
        if do_print:
            print("Max possible score:", "Exact Match!")
            print("Score:", max_possible_reward)
        return max_possible_reward

    gt_names = [tool["name"] for tool in gt_tools]
    pd_names = [tool["name"] for tool in pd_tools]
    score = match_score(list(gt_names), list(pd_names))

    local_max_possible = 1.0
    used_pd_indices = set()  # Keep track of matched pd_tools

    for gt_tool in gt_tools:
        gt_name = gt_tool["name"]
        gt_params = gt_tool["parameters"]

        local_max_possible += 1.0 + len(gt_params)

        best_match = None
        best_match_score = 0.0
        best_match_index = -1

        # Find the best matching unused pd_tool
        for i, pd_tool in enumerate(pd_tools):
            if i in used_pd_indices or pd_tool["name"] != gt_name:
                continue

            pd_params = pd_tool["parameters"]
            param_score = match_score(list(gt_params.keys()), list(pd_params.keys()))

            # Calculate correctness score for parameter values
            correctness_score = sum(1.0 for k, v in gt_params.items() if k in pd_params and pd_params[k] == v)

            total_score = param_score + correctness_score

            if total_score > best_match_score:
                best_match_score = total_score
                best_match = pd_tool
                best_match_index = i

        if best_match:
            used_pd_indices.add(best_match_index)
            score += best_match_score

    if do_print:
        print()
        print("Max possible score:", local_max_possible)
        print("Score:", score)

    return (max_possible_reward - min_possible_reward) * score / local_max_possible + min_possible_reward


# custoimzed reward functions: tool call correctness
def customize_correctness_reward_tool(
    completions, answer, step, max_possible_reward, min_possible_reward, do_print, **kwargs
):
    responses = [completion[0]["content"] for completion in completions]
    rewards = []

    for response, ans in zip(responses, answer, strict=False):
        reward = 0.0

        if "<tool_call>" not in ans:
            # if "<tool_call>" not in response and "</tool_call>" not in response:
            #     reward = max_possible_reward
            # else:
            #     reward = min_possible_reward
            rewards.append(reward)
            continue

        gt_tool_call = ans.split("<tool_call>")[1].split("</tool_call>")[0].strip()
        gt_tools = gt_tool_call.split("\n")
        gt_tools = [json.loads(tool) for tool in gt_tools]  # each diction contains "name" and "parameter"

        try:
            # Change here as a constrint in training: if the format is not correct,
            # directly give the lowest possible score
            assert "<tool_call>" in response
            assert "</tool_call>" in response
            pd_tools = response.split("<tool_call>")[1].split("</tool_call>")[0].strip().split("\n")
            pd_tools = [json.loads(tool) for tool in pd_tools]
            reward = compute_tool_call_reward(
                gt_tools, pd_tools, max_possible_reward, min_possible_reward, do_print
            )  # top reward is 2
        except Exception:
            reward = min_possible_reward

        rewards.append(reward)

    if do_print:
        print("\n======= Reward for <tool call> =======")
        print("Reward function for <tool call> correctness is called ...")
        print(rewards)
    return rewards


def compute_score(data_source, solution_str, ground_truth, extra_info, step=0):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning."
    Proceedings of the 62nd Annual Meeting of the Association for
    Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    exp_name = extra_info.get("experiment_name", "")
    if "llama" in exp_name:
        predict_str = (
            solution_str.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].strip()
        )
    elif "qwen" in exp_name:
        predict_str = solution_str.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    else:
        raise NotImplementedError(f"Unknown model name: {exp_name}")

    tool_max_possible = 3.0
    tool_min_possible = -3.0

    format_max_possible = 1.0
    format_min_possible = 0.0

    completions = [[{"role": "assistant", "content": predict_str}]]
    answer = [ground_truth]

    do_print = random.randint(1, 64) == 1

    fomrat_score = customize_format_reward_func(
        completions, answer, step, format_max_possible, format_min_possible, do_print
    )[0]
    correctness_score = customize_correctness_reward_tool(
        completions, answer, step, tool_max_possible, tool_min_possible, do_print
    )[0]

    score = fomrat_score + correctness_score

    result = {
        "score": score,
        "format_reward": fomrat_score,
        "accuracy_reward": correctness_score,
    }

    return result
