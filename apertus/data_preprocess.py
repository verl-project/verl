# Copyright 2026 The VERL Team and individual contributors.
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
"""
Preprocess Apertus demo RL data into VERL parquet format.

TODO: Move this hardcoded configuration into a config file and add command-line
arguments for selecting the config and output directory.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any
from datasets import load_from_disk

import datasets

from table_gpt_data import TABLE_GPT_DATASET_ID, load_table_gpt_mix
# from verl.utils.reward_score.gsm8k import extract_solution as extract_gsm8k_solution

LOCAL_SAVE_DIR = "./data/code"
DATASETS_CACHE_DIR = "./data/code/.hf_datasets_cache"
SEED = 85
EMPTY_SYSTEM_PROMPT = True
CODE_CONTESTS_MAX_GENERATED_TESTS = 50

MATH_FINAL_ANSWER_INSTRUCTION = (
    "Let's think step by step and output the final answer within \\boxed{}."
)
GSM8K_FINAL_ANSWER_INSTRUCTION = (
    'Let\'s think step by step and output the final answer after "####".'
)
MULTIPLE_CHOICE_FINAL_ANSWER_INSTRUCTION = (
    "Put the letter of the correct option in <answer></answer>, "
    "for example <answer>A</answer>."
)
CODE_FINAL_ANSWER_INSTRUCTION = (
    "You may reason before answering. End your response with a Python 3 code "
    "block containing the complete solution."
)


@dataclass(frozen=True)
class DatasetConfig:
    enabled: bool
    name: str
    dataset_id: str
    split: str
    adapter: str
    sample_size: int | None = None
    subset: str | None = None
    data_source: str | None = None
    question_key: str = "problem"
    answer_key: str = "answer"
    solution_key: str | None = None
    prompt_key: str = "prompt"
    choices_key: str = "choices"
    subject_key: str | None = None
    shuffle_choices: bool = False
    filter_key: str | None = None
    filter_value: Any = None


TRAIN_DATASETS = [
    # DatasetConfig(
    #     enabled=True,
    #     name="big_math_rl_verified",
    #     dataset_id="SynthLabsAI/Big-Math-RL-Verified",
    #     split="train",
    #     adapter="math",
    #     data_source="SynthLabsAI/Big-Math-RL-Verified",
    #     question_key="problem",
    #     answer_key="answer",
    #     subject_key="domain",
    #     sample_size=50_000,
    # ),
    # DatasetConfig(
    #     enabled=True,
    #     name="deepmath103k",
    #     dataset_id="zwhe99/DeepMath-103K",
    #     split="train",
    #     adapter="math",
    #     data_source="zwhe99/DeepMath-103K",
    #     question_key="question",
    #     answer_key="final_answer",
    #     solution_key="r1_solution_1",
    #     subject_key="topic",
    #     sample_size=50_000,
    # ),
    # DatasetConfig(
    #     enabled=True,
    #     name="if_rl_singleturn",
    #     dataset_id="swiss-ai/if-rl-singleturn-prompts",
    #     split="train",
    #     adapter="if_placeholder",
    #     data_source="swiss-ai/if-rl-singleturn-prompts",
    #     prompt_key="messages",
    #     answer_key="ground_truth",
    #     subject_key="constraint_type",
    #     sample_size=40_000,
    # ),
    # DatasetConfig(
    #     enabled=True,
    #     name="if_rl_singleturn_hard",
    #     dataset_id="swiss-ai/if-rl-singleturn-hard-prompts",
    #     split="train",
    #     adapter="if_placeholder",
    #     data_source="swiss-ai/if-rl-singleturn-hard-prompts",
    #     prompt_key="messages",
    #     answer_key="ground_truth",
    #     subject_key="constraint_type",
    #     sample_size=10_000,
    # ),
    DatasetConfig(
        enabled=True,
        name="KodCode-verified",
        dataset_id="/iopsstor/scratch/cscs/rmachace/code-gym/src/kodcode_50k_verified",
        split="train",
        adapter="kodcode",
        data_source="kodcode",
        question_key="question",
        solution_key="solution",
        answer_key="test_cases",
        sample_size=None,
    ),
    DatasetConfig(
        enabled=True,
        name="taco_verified",
        dataset_id="likaixin/TACO-verified",
        split="train",
        adapter="taco",
        data_source="taco",
        question_key="question",
        answer_key="input_output",
        solution_key="solutions",
        subject_key="source",
        sample_size=None,
    ),
    DatasetConfig(
        enabled=True,
        name="apps",
        dataset_id="ReactiveAI/codeparrot-apps-reupload",
        split="train",
        adapter="apps",
        data_source="apps",
        question_key="question",
        answer_key="input_output",
        solution_key="solutions",
        sample_size=None,
    ),
    # DatasetConfig(
    #     enabled=True,
    #     name="table_gpt",
    #     dataset_id=TABLE_GPT_DATASET_ID,
    #     split="mixed",
    #     adapter="table_gpt",
    #     prompt_key="prompt",
    #     answer_key="completion",
    #     sample_size=None,
    # ),
    DatasetConfig(
        enabled=True,
        name="code_contests",
        dataset_id="deepmind/code_contests",
        split="train",
        adapter="code_contests",
        data_source="code_contests",
        question_key="description",
        solution_key="solutions",
        sample_size=None,
    ),
    DatasetConfig(
        enabled=True,
        name="open_r1_codeforces",
        dataset_id="open-r1/codeforces",
        subset="verifiable",
        split="train",
        adapter="open_r1_codeforces",
        data_source="codeforces",
        sample_size=None,
    ),
]

EVAL_DATASETS = [
    # DatasetConfig(
    #     enabled=True,
    #     name="gsm8k",
    #     dataset_id="openai/gsm8k",
    #     subset="main",
    #     split="test",
    #     adapter="gsm8k",
    #     data_source="openai/gsm8k",
    #     question_key="question",
    #     answer_key="answer",
    #     sample_size=100,
    # ),
    # DatasetConfig(
    #     enabled=True,
    #     name="math500",
    #     dataset_id="HuggingFaceH4/MATH-500",
    #     split="test",
    #     adapter="math",
    #     data_source="HuggingFaceH4/MATH-500",
    #     question_key="problem",
    #     answer_key="answer",
    #     solution_key="solution",
    #     subject_key="subject",
    #     sample_size=50,
    # ),
    # DatasetConfig(
    #     enabled=True,
    #     name="aime2024",
    #     dataset_id="HuggingFaceH4/aime_2024",
    #     split="train",
    #     adapter="math",
    #     data_source="aime2024",
    #     question_key="problem",
    #     answer_key="answer",
    #     solution_key="solution",
    #     sample_size=None,
    # ),
    # DatasetConfig(
    #     enabled=True,
    #     name="aime2025",
    #     dataset_id="math-ai/aime25",
    #     split="test",
    #     adapter="math",
    #     data_source="aime2025",
    #     question_key="problem",
    #     answer_key="answer",
    #     sample_size=None,
    # ),
    # DatasetConfig(
    #     enabled=True,
    #     name="gpqa_diamond",
    #     dataset_id="Idavidrein/gpqa",
    #     subset="gpqa_diamond",
    #     split="train",
    #     adapter="gpqa",
    #     data_source="gpqa_diamond",
    #     question_key="Question",
    #     answer_key="Correct Answer",
    #     shuffle_choices=True,
    #     sample_size=100,
    # ),
    # DatasetConfig(
    #     enabled=True,
    #     name="mmlu",
    #     dataset_id="cais/mmlu",
    #     subset="all",
    #     split="test",
    #     adapter="multiple_choice",
    #     data_source="mmlu",
    #     question_key="question",
    #     choices_key="choices",
    #     answer_key="answer",
    #     subject_key="subject",
    #     sample_size=100,
    # ),
    # NOTE: atm humaneval is evaluated using prime_code
    DatasetConfig(
        enabled=True,
        name="openai_humaneval",
        dataset_id="openai/openai_humaneval",
        subset="openai_humaneval",
        split="test",
        adapter="humaneval",
        data_source="humaneval",
        question_key="prompt",
        answer_key="test",
        solution_key="canonical_solution",
        prompt_key="entry_point",
        sample_size=100,
    ),
    # DatasetConfig(
    #     enabled=True,
    #     name="ifeval",
    #     dataset_id="google/IFEval",
    #     split="train",
    #     adapter="if_eval",
    #     data_source="google/IFEval",
    #     prompt_key="prompt",
    #     sample_size=100,
    # ),
    # DatasetConfig(
    #     enabled=True,
    #     name="ifbench",
    #     dataset_id="allenai/IFBench_test",
    #     split="train",
    #     adapter="if_eval",
    #     data_source="allenai/IFBench_test",
    #     prompt_key="prompt",
    #     sample_size=100,
    # ),
]

ADAPTERS: dict[
    str, Callable[[dict[str, Any], int, str, DatasetConfig], dict[str, Any]]
] = {}


def register_adapter(name: str):
    def decorator(
        func: Callable[[dict[str, Any], int, str, DatasetConfig], dict[str, Any]],
    ):
        ADAPTERS[name] = func
        return func

    return decorator


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def make_prompt(user_content: str) -> list[dict[str, str]]:
    messages = []
    if EMPTY_SYSTEM_PROMPT:
        messages.append({"role": "system", "content": ""})
    messages.append({"role": "user", "content": user_content})
    return messages


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return make_prompt(messages)
    if not isinstance(messages, list):
        return make_prompt(normalize_text(messages))

    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_text(message.get("role", "user")) or "user"
        content = normalize_text(message.get("content", ""))
        normalized.append({"role": role, "content": content})

    if EMPTY_SYSTEM_PROMPT and not (
        normalized and normalized[0].get("role") == "system"
    ):
        normalized.insert(0, {"role": "system", "content": ""})
    return normalized


def parse_json_maybe(value: Any, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return default
    return value


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def make_row(
    *,
    config: DatasetConfig,
    split: str,
    index: int,
    prompt: list[dict[str, str]],
    ability: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "data_source": config.data_source or config.dataset_id,
        "prompt": prompt,
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "dataset": config.name,
            "dataset_id": config.dataset_id,
            "split": split,
            "index": index,
        },
    }
    if extra_info:
        row["extra_info"].update(extra_info)
    return row


@register_adapter("table_gpt")
def adapt_table_gpt(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    task = normalize_text(get_value(example, "task")) or normalize_text(
        get_value(example, "_table_gpt_task")
    )
    prompt_text = normalize_text(get_value(example, config.prompt_key))
    ground_truth = get_value(example, config.answer_key)
    if not isinstance(ground_truth, str):
        ground_truth = json_dumps(ground_truth)

    metadata = parse_json_maybe(get_value(example, "metadata"), default={})
    if not isinstance(metadata, dict):
        metadata = {}
    metadata = json_dumps(metadata)
    source_split = normalize_text(get_value(example, "_table_gpt_split")) or split

    return {
        "data_source": f"tablegpt/{task}",
        "prompt": make_prompt(prompt_text),
        "ability": task,
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "split": source_split,
            "index": idx,
            "task": task,
            "dataset": normalize_text(get_value(example, "dataset")),
            "metadata": metadata,
        },
    }


@register_adapter("math")
def adapt_math(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    question = normalize_text(get_value(example, config.question_key))
    answer = normalize_text(get_value(example, config.answer_key))
    prompt = make_prompt(f"{question} {MATH_FINAL_ANSWER_INSTRUCTION}")
    extra_info = {
        "question": question,
        "answer": answer,
    }
    if config.solution_key and config.solution_key in example:
        extra_info["solution"] = example[config.solution_key]
    if config.subject_key and config.subject_key in example:
        extra_info["subject"] = normalize_text(example[config.subject_key])
    if "difficulty" in example:
        extra_info["difficulty"] = normalize_text(example["difficulty"])
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=prompt,
        ability="math",
        ground_truth=answer,
        extra_info=extra_info,
    )


@register_adapter("gsm8k")
def adapt_gsm8k(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    question = normalize_text(get_value(example, config.question_key))
    answer_raw = normalize_text(get_value(example, config.answer_key))
    try:
        # answer = extract_gsm8k_solution(answer_raw)
        answer = ""
    except Exception:
        answer = answer_raw.split("####")[-1].strip()
    prompt = make_prompt(f"{question} {GSM8K_FINAL_ANSWER_INSTRUCTION}")
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=prompt,
        ability="math",
        ground_truth=answer,
        extra_info={"question": question, "answer": answer_raw},
    )


@register_adapter("taco")
def adapt_taco(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    question = normalize_text(get_value(example, config.question_key))
    starter_code = normalize_text(get_value(example, "starter_code"))
    raw_test_cases = parse_json_maybe(get_value(example, config.answer_key), default={})
    prime_code_test_cases = normalize_prime_code_test_cases(raw_test_cases)
    raw_solutions = get_value(example, config.solution_key)
    solutions = parse_json_maybe(raw_solutions, default=raw_solutions)
    reference_solution = first_solution(solutions)
    prompt = make_prompt(format_code_prompt(question, starter_code))
    extra_info = {
        "question": question,
        "starter_code": starter_code,
        "difficulty": get_value(example, "difficulty"),
        "source": get_value(example, "source"),
        "name": get_value(example, "name"),
        "url": get_value(example, "url"),
        "reference_solution": reference_solution,
        "language": "python",
        "input_output": json_dumps(raw_test_cases),
        "prime_code_input_output": json_dumps(prime_code_test_cases),
        "sandbox_data_source": "likaixin/TACO-verified",
    }
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=prompt,
        ability="code",
        ground_truth=reference_solution,
        extra_info=extra_info,
    )


@register_adapter("kodcode")
def adapt_kodcode(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    question = normalize_text(get_value(example, "question"))
    test_cases = get_value(example, "test_cases")
    prompt = make_prompt(format_code_prompt(question, ""))
    ground_truth = json_dumps(test_cases)
    extra_info = {
        "question": question,
        "question_id": normalize_text(get_value(example, "question_id")),
        "test_cases": test_cases,
        "num_used_tests": len(test_cases),
        "language": "python",
        "sandbox_data_source": "Muennighoff/mbpp",
    }
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=prompt,
        ability="code",
        ground_truth=ground_truth,
        extra_info=extra_info,
    )

@register_adapter("code_contests")
def adapt_code_contests(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    question = normalize_text(get_value(example, config.question_key))
    test_cases = normalize_code_contests_test_cases(example)
    solutions = get_value(example, config.solution_key)
    reference_solution = first_code_contests_solution(solutions)
    prompt = make_prompt(format_code_prompt(question, ""))
    extra_info = {
        "question": question,
        "difficulty": normalize_text(get_value(example, "difficulty")),
        "source": normalize_text(get_value(example, "source")),
        "name": normalize_text(get_value(example, "name")),
        "reference_solution": reference_solution,
        "language": "python",
        "input_output": json_dumps(test_cases),
        "prime_code_input_output": json_dumps(test_cases),
        "sandbox_data_source": "likaixin/TACO-verified", #"lighteval/code_generation_lite", 
        "num_used_tests": len(test_cases["inputs"]),
    }
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=prompt,
        ability="code",
        ground_truth=reference_solution,
        extra_info=extra_info,
    )


@register_adapter("apps")
def adapt_apps(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    question = normalize_text(get_value(example, config.question_key))
    starter_code = normalize_text(get_value(example, "starter_code"))
    raw_test_cases = parse_json_maybe(get_value(example, config.answer_key), default={})
    prime_code_test_cases = normalize_prime_code_test_cases(raw_test_cases)
    raw_solutions = get_value(example, config.solution_key)
    solutions = parse_json_maybe(raw_solutions, default=raw_solutions)
    reference_solution = first_solution(solutions)
    prompt = make_prompt(format_code_prompt(question, starter_code))
    extra_info = {
        "question": question,
        "starter_code": starter_code,
        "difficulty": normalize_text(get_value(example, "difficulty")),
        "url": normalize_text(get_value(example, "url")),
        "problem_id": normalize_text(get_value(example, "problem_id")),
        "reference_solution": reference_solution,
        "language": "python",
        "input_output": json_dumps(raw_test_cases),
        "prime_code_input_output": json_dumps(prime_code_test_cases),
        "sandbox_data_source": "likaixin/TACO-verified",#"lighteval/code_generation_lite",
    }
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=prompt,
        ability="code",
        ground_truth=reference_solution,
        extra_info=extra_info,
    )


@register_adapter("open_r1_codeforces")
def adapt_codeforces(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    question = format_codeforces_prompt(example)
    test_cases = normalize_codeforces_test_cases(example)
    prompt = make_prompt(format_code_prompt(question, ""))
    ground_truth = json_dumps(test_cases)
    extra_info = {
        "question": question,
        "problem_id": normalize_text(get_value(example, "id")),
        "contest_id": normalize_text(get_value(example, "contest_id")),
        "cf_index": normalize_text(get_value(example, "index")),
        "title": normalize_text(get_value(example, "title")),
        "difficulty": normalize_text(get_value(example, "rating")),
        "tags": json_dumps(get_value(example, "tags", [])),
        "num_used_tests": len(test_cases["inputs"]),
        "language": "python",
        "input_output": json_dumps(test_cases),
        "prime_code_input_output": json_dumps(test_cases),
        "sandbox_data_source": "likaixin/TACO-verified",#"lighteval/code_generation_lite",
    }
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=prompt,
        ability="code",
        ground_truth=ground_truth,
        extra_info=extra_info,
    )


@register_adapter("humaneval")
def adapt_humaneval(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    code_prompt = normalize_text(get_value(example, config.question_key)).rstrip()
    if not code_prompt.endswith("\n"):
        code_prompt += "\n"
    entry_point = normalize_text(get_value(example, config.prompt_key))
    test = normalize_text(get_value(example, config.answer_key)).rstrip()
    canonical_solution = get_value(example, config.solution_key, "")
    prompt = make_prompt(format_code_prompt(code_prompt, ""))
    ground_truth = {
        "prompt": code_prompt,
        "test": test,
        "entry_point": entry_point,
    }
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=prompt,
        ability="code",
        ground_truth=json_dumps(ground_truth),
        extra_info={
            "task_id": get_value(example, "task_id"),
            "entry_point": entry_point,
            "canonical_solution": canonical_solution,
        },
    )


@register_adapter("if_placeholder")
def adapt_if_placeholder(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    prompt = normalize_messages(get_value(example, config.prompt_key))
    ground_truth = get_value(example, config.answer_key)
    extra_info = {
        "key": get_value(example, "key"),
        "dataset": get_value(example, "dataset"),
        "constraint": get_value(example, "constraint"),
        "constraint_type": get_value(example, "constraint_type"),
        "verification": "placeholder",
    }
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=prompt,
        ability="instruction_following",
        ground_truth=ground_truth,
        extra_info=extra_info,
    )


@register_adapter("if_eval")
def adapt_if_eval(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    prompt_text = normalize_text(get_value(example, config.prompt_key))
    instruction_ids = get_value(example, "instruction_id_list", [])
    kwargs_list = get_value(example, "kwargs", [])
    ground_truth = {
        "instruction_id": instruction_ids,
        "kwargs": kwargs_list,
    }
    ground_truth_json = json.dumps(ground_truth, ensure_ascii=False)
    extra_info = {
        "key": normalize_text(get_value(example, "key")),
        "constraint": ground_truth_json,
        "instruction_id_list": json.dumps(instruction_ids, ensure_ascii=False),
    }
    if config.subject_key and config.subject_key in example:
        extra_info["subject"] = example[config.subject_key]
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=make_prompt(prompt_text),
        ability="instruction_following",
        ground_truth=ground_truth_json,
        extra_info=extra_info,
    )


@register_adapter("multiple_choice")
def adapt_multiple_choice(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    question = normalize_text(get_value(example, config.question_key))
    choices = list(get_value(example, config.choices_key, []))
    answer_index = normalize_answer_index(get_value(example, config.answer_key))
    choices, answer_letter = maybe_shuffle_choices(choices, answer_index, config, idx)
    prompt = make_prompt(format_multiple_choice_prompt(question, choices))
    extra_info = {
        "question": question,
        "choices": choices,
        "answer_index": answer_index,
    }
    if config.subject_key and config.subject_key in example:
        extra_info["subject"] = example[config.subject_key]
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=prompt,
        ability="knowledge",
        ground_truth=answer_letter,
        extra_info=extra_info,
    )


@register_adapter("gpqa")
def adapt_gpqa(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    question = normalize_text(get_value(example, config.question_key))
    correct = normalize_text(get_value(example, config.answer_key))
    choices = [
        correct,
        normalize_text(get_value(example, "Incorrect Answer 1")),
        normalize_text(get_value(example, "Incorrect Answer 2")),
        normalize_text(get_value(example, "Incorrect Answer 3")),
    ]
    choices = [choice for choice in choices if choice]
    choices, answer_letter = maybe_shuffle_choices(choices, 0, config, idx)
    prompt = make_prompt(format_multiple_choice_prompt(question, choices))
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=prompt,
        ability="knowledge",
        ground_truth=answer_letter,
        extra_info={"question": question, "choices": choices},
    )


def format_multiple_choice_prompt(question: str, choices: list[Any]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = "\n".join(
        f"{letters[i]}. {normalize_text(choice)}" for i, choice in enumerate(choices)
    )
    return f"{question}\n\n{options}\n\n{MULTIPLE_CHOICE_FINAL_ANSWER_INSTRUCTION}"


def format_code_prompt(question: str, starter_code: str) -> str:
    if starter_code:
        return f"{question}\n\n```python\n{starter_code}\n```\n\n{CODE_FINAL_ANSWER_INSTRUCTION}"
    return f"{question}\n\n{CODE_FINAL_ANSWER_INSTRUCTION}"


def format_codeforces_prompt(example: dict[str, Any]) -> str:
    sections = []
    title = normalize_text(get_value(example, "title"))
    if title:
        sections.append(title)

    description = normalize_text(get_value(example, "description"))
    if description:
        sections.append(description)

    # TODO: should we debias the prompt from Input/Output format?
    input_format = normalize_text(get_value(example, "input_format"))
    if input_format:
        sections.append(f"Input\n{input_format}")

    output_format = normalize_text(get_value(example, "output_format"))
    if output_format:
        sections.append(f"Output\n{output_format}")

    examples = get_value(example, "examples", []) or []
    example_text = format_codeforces_examples(examples)
    if example_text:
        sections.append(f"Examples\n{example_text}")

    note = normalize_text(get_value(example, "note"))
    if note:
        sections.append(f"Note\n{note}")

    return "\n\n".join(sections)


def format_codeforces_examples(examples: Any) -> str:
    if not isinstance(examples, list):
        return ""
    formatted = []
    for example in examples:
        if not isinstance(example, dict):
            continue
        test_input = normalize_text(example.get("input"))
        test_output = normalize_text(example.get("output"))
        if not test_input and not test_output:
            continue
        formatted.append(f"Input\n{test_input}\n\nOutput\n{test_output}")
    return "\n\n".join(formatted)


def first_solution(solutions: Any) -> str:
    if isinstance(solutions, list) and solutions:
        return normalize_text(solutions[0])
    return normalize_text(solutions)


def first_code_contests_solution(solutions: Any) -> str:
    if isinstance(solutions, dict):
        solution_values = solutions.get("solution", [])
        if not solution_values:
            return ""
        return first_solution(solution_values)
    return first_solution(solutions)


def normalize_prime_code_test_cases(test_cases: Any) -> dict[str, Any]:
    """Convert TACO input_output into the dict schema expected by prime_code."""
    if not isinstance(test_cases, dict):
        raise ValueError(f"Expected dict test cases, got {type(test_cases)}")

    inputs = test_cases.get("inputs", [])
    outputs = test_cases.get("outputs", [])
    normalized = {"inputs": [], "outputs": []}
    if test_cases.get("fn_name") is not None:
        normalized["fn_name"] = test_cases["fn_name"]
        normalized["inputs"] = [serialize_call_based_input(case) for case in inputs]
        normalized["outputs"] = [json_dumps(output) for output in outputs]
    else:
        normalized["inputs"] = [serialize_standard_input(case) for case in inputs]
        normalized["outputs"] = [serialize_standard_output(case) for case in outputs]
    return normalized


def normalize_code_contests_test_cases(example: dict[str, Any]) -> dict[str, list[str]]:
    """Convert CodeContests tests into the standard-input schema expected by prime_code."""
    inputs: list[str] = []
    outputs: list[str] = []
    for key, max_cases in (
        ("public_tests", None),
        ("private_tests", None),
        # These are not AI-generated, they are obtained by mutating existing test inputs
        ("generated_tests", CODE_CONTESTS_MAX_GENERATED_TESTS),
    ):
        test_group = get_value(example, key)
        test_inputs = get_test_values(test_group, "input")
        test_outputs = get_test_values(test_group, "output")
        if max_cases is not None:
            test_inputs = test_inputs[:max_cases]
            test_outputs = test_outputs[:max_cases]
        for test_input, test_output in zip(test_inputs, test_outputs):
            inputs.append(normalize_text(test_input))
            outputs.append(normalize_text(test_output))
    if not inputs:
        raise ValueError("CodeContests example has no usable tests")
    return {"inputs": inputs, "outputs": outputs}


def normalize_codeforces_test_cases(
    example: dict[str, Any],
) -> dict[str, list[str]]:
    """Convert Open-R1 Codeforces official tests into the code-gym stdio schema."""
    inputs: list[str] = []
    outputs: list[str] = []
    official_tests = get_value(example, "official_tests", []) or []
    for test_case in official_tests:
        if not isinstance(test_case, dict):
            continue
        test_input = normalize_text(test_case.get("input"))
        test_output = normalize_text(test_case.get("output"))
        if test_input or test_output:
            inputs.append(test_input)
            outputs.append(test_output)
    if not inputs:
        raise ValueError("Open-R1 Codeforces example has no usable official tests")
    return {"inputs": inputs, "outputs": outputs}


def code_contests_has_tests(example: dict[str, Any]) -> bool:
    for key, max_cases in (
        ("public_tests", None),
        ("private_tests", None),
        ("generated_tests", CODE_CONTESTS_MAX_GENERATED_TESTS),
    ):
        test_group = get_value(example, key)
        test_inputs = get_test_values(test_group, "input")
        test_outputs = get_test_values(test_group, "output")
        if max_cases is not None:
            test_inputs = test_inputs[:max_cases]
            test_outputs = test_outputs[:max_cases]
        if len(test_inputs) != len(test_outputs):
            continue
        if any(True for _ in zip(test_inputs, test_outputs)):
            return True
    return False


def apps_has_tests(example: dict[str, Any]) -> bool:
    test_cases = parse_json_maybe(get_value(example, "input_output"), default={})
    if not isinstance(test_cases, dict):
        return False
    inputs = test_cases.get("inputs", [])
    outputs = test_cases.get("outputs", [])
    return bool(inputs) and len(inputs) == len(outputs)


def codeforces_has_tests(example: dict[str, Any]) -> bool:
    """Keep rows whose official tests can be checked by code-gym's stdio diff runner."""
    if normalize_text(get_value(example, "input_mode")) != "stdio":
        return False
    if normalize_text(get_value(example, "interaction_format")):
        return False
    if normalize_text(get_value(example, "generated_checker")):
        return False
    # TODO: decide whether or not to also keep problems with incomplete official tests
    # if get_value(example, "official_tests_complete") is not True:
    #     return False
    official_tests = get_value(example, "official_tests", []) or []
    return bool(official_tests) and all(
        isinstance(test_case, dict)
        and "input" in test_case
        and "output" in test_case
        for test_case in official_tests
    )


def get_test_values(test_group: Any, key: str) -> list[Any]:
    if not test_group:
        return []
    if isinstance(test_group, dict):
        values = test_group.get(key, [])
        return list(values) if values is not None else []
    if isinstance(test_group, list):
        return [
            test_case.get(key, "")
            for test_case in test_group
            if isinstance(test_case, dict)
        ]
    return []


def serialize_call_based_input(value: Any) -> str:
    """Serialize one call-based test case as newline-separated JSON args."""
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        return "\n".join(json_dumps(item) for item in value)
    return json_dumps(value)


def serialize_standard_input(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        return "\n".join(str(item) for item in value)
    return str(value)


def serialize_standard_output(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list | tuple):
        return "\n".join(str(item) for item in value)
    return str(value)


def normalize_answer_index(answer: Any) -> int:
    if isinstance(answer, int):
        return answer
    if isinstance(answer, str):
        answer = answer.strip()
        if answer.isdigit():
            return int(answer)
        match = re.search(r"\b([A-D])\b", answer.upper())
        if match:
            return ord(match.group(1)) - ord("A")
    raise ValueError(f"Cannot normalize multiple-choice answer: {answer!r}")


def maybe_shuffle_choices(
    choices: list[Any], answer_index: int, config: DatasetConfig, idx: int
) -> tuple[list[Any], str]:
    indexed_choices = list(enumerate(choices))
    if config.shuffle_choices:
        random.Random(f"{SEED}:{config.name}:{idx}").shuffle(indexed_choices)
    new_choices = [choice for _, choice in indexed_choices]
    new_answer_index = next(
        new_idx
        for new_idx, (old_idx, _) in enumerate(indexed_choices)
        if old_idx == answer_index
    )
    return new_choices, chr(ord("A") + new_answer_index)


def get_value(example: dict[str, Any], key: str | None, default: Any = None) -> Any:
    if key is None:
        return default
    return example.get(key, default)


def load_raw_dataset(config: DatasetConfig) -> datasets.Dataset:
    load_kwargs = {}
    if config.subset is not None:
        load_kwargs["name"] = config.subset
    if config.adapter == "table_gpt":
        raw_dataset = load_table_gpt_mix(
            config.dataset_id,
            cache_dir=os.path.expanduser(DATASETS_CACHE_DIR),
        )
    elif os.path.isfile(os.path.expanduser(config.dataset_id)):
        raw_dataset = datasets.load_dataset(
            "parquet",
            data_files=os.path.expanduser(config.dataset_id),
            split=config.split,
            cache_dir=os.path.expanduser(DATASETS_CACHE_DIR),
        )
    elif os.path.isdir(config.dataset_id):
        raw_dataset = load_from_disk(
            config.dataset_id, 
        )
    else:
        raw_dataset = datasets.load_dataset(
            config.dataset_id,
            split=config.split,
            cache_dir=os.path.expanduser(DATASETS_CACHE_DIR),
            **load_kwargs,
        )
    print(f"Loaded {len(raw_dataset)} rows for {config.name}.", flush=True)

    if config.filter_key is not None:
        raw_dataset = raw_dataset.filter(
            lambda example: example.get(config.filter_key) == config.filter_value
        )
    if config.adapter == "code_contests":
        rows_before_filter = len(raw_dataset)
        raw_dataset = raw_dataset.filter(code_contests_has_tests)
        filtered = rows_before_filter - len(raw_dataset)
        if filtered:
            print(f"Filtered {filtered} rows from {config.name}.", flush=True)
    if config.adapter == "apps":
        rows_before_filter = len(raw_dataset)
        raw_dataset = raw_dataset.filter(apps_has_tests)
        filtered = rows_before_filter - len(raw_dataset)
        if filtered:
            print(f"Filtered {filtered} rows from {config.name}.", flush=True)
    if config.adapter == "open_r1_codeforces":
        rows_before_filter = len(raw_dataset)
        # Remove all interactive, file-I/O, or custom-checker tasks.
        raw_dataset = raw_dataset.filter(codeforces_has_tests)
        filtered = rows_before_filter - len(raw_dataset)
        if filtered:
            print(f"Filtered {filtered} rows from {config.name}.", flush=True)
    if config.sample_size is not None:
        sample_size = min(config.sample_size, len(raw_dataset))
        raw_dataset = raw_dataset.shuffle(seed=SEED).select(range(sample_size))
        print(
            f"Sampled {len(raw_dataset)} rows for {config.name} after filtering and sampling.",
            flush=True,
        )
    return raw_dataset


def preprocess_dataset(config: DatasetConfig) -> datasets.Dataset:
    if config.adapter not in ADAPTERS:
        raise KeyError(f"No adapter registered for {config.adapter!r}")
    print(f"Loading {config.name} from {config.dataset_id}...", flush=True)
    raw_dataset = load_raw_dataset(config)
    adapter = ADAPTERS[config.adapter]
    rows = [
        adapter(dict(example), idx, config.split, config)
        for idx, example in enumerate(raw_dataset)
    ]
    return datasets.Dataset.from_list(rows)


def concatenate_named(
    datasets_by_name: Iterable[tuple[str, datasets.Dataset]],
) -> datasets.Dataset:
    dataset_list = []
    for name, dataset in datasets_by_name:
        if len(dataset) == 0:
            print(f"Skipping empty dataset {name}.", flush=True)
            continue
        dataset_list.append(dataset)
    if not dataset_list:
        return datasets.Dataset.from_list([])
    return datasets.concatenate_datasets(dataset_list)


def save_json_example(dataset: datasets.Dataset, path: str) -> None:
    if len(dataset) == 0:
        return
    with open(path, "w") as f:
        json.dump(dataset[0], f, indent=2)


def write_outputs(
    train_dataset: datasets.Dataset, eval_datasets: dict[str, datasets.Dataset]
) -> None:
    local_dir = os.path.expanduser(LOCAL_SAVE_DIR)
    eval_dir = os.path.join(local_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    val_dataset = concatenate_named(eval_datasets.items())
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(local_dir, "val.parquet"))

    for name, dataset in eval_datasets.items():
        dataset.to_parquet(os.path.join(eval_dir, f"{name}.parquet"))

    save_json_example(train_dataset, os.path.join(local_dir, "train_example.json"))
    save_json_example(val_dataset, os.path.join(local_dir, "val_example.json"))

    print(
        f"Wrote {len(train_dataset)} training rows to {local_dir}/train.parquet",
        flush=True,
    )
    print(f"Wrote {len(val_dataset)} eval rows to {local_dir}/val.parquet", flush=True)


def main() -> None:
    train_dataset = concatenate_named(
        (config.name, preprocess_dataset(config))
        for config in TRAIN_DATASETS
        if config.enabled
    )
    eval_datasets = {
        config.name: preprocess_dataset(config)
        for config in EVAL_DATASETS
        if config.enabled
    }
    write_outputs(train_dataset, eval_datasets)


if __name__ == "__main__":
    main()
