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

from __future__ import annotations

import json
import os
import random
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import datasets
from verl.utils.reward_score.gsm8k import extract_solution as extract_gsm8k_solution

from utils.LEXam_mcq import normalize_lexam_mcq_sample
from utils.qa_gym import load_qa_gym_rl_pairs_jsonl
from utils.rgym import strip_rgym_format_instructions
from utils.riddle_sense import normalize_riddle_sense_sample
from utils.table_gpt import TABLE_GPT_DATASET_ID, load_table_gpt_mix


LOCAL_SAVE_DIR = "./data/apertus_demo_rl"
DATASETS_CACHE_DIR = "./data/apertus_demo_rl/.hf_datasets_cache"
SEED = 85
CODE_CONTESTS_MAX_GENERATED_TESTS = 50
PREPROCESS_NUM_PROC = 48

CODE_FINAL_ANSWER_INSTRUCTION = (
    "You may reason before answering. End your response with a Python 3 code "
    "block containing the complete solution."
)

# DISPLAY_ANSWERS_EBNF = r"""%llguidance {}
# start: (TEXT | tool_block)*
# tool_block: <|tools_prefix|> %json { "type": "array", "minItems": 1, "items": { "type": "object" } } <|tools_suffix|>
# TEXT: /(?:(?!<\|tools_prefix\|>)(.|\n))+/
# """


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    dataset_id: str
    split: str
    adapter: str
    data_source: str
    sample_size: int | None = None
    subset: str | None = None
    question_key: str = "problem"
    answer_key: str = "answer"
    prompt_key: str = "prompt"
    system_key: str | None = None
    choices_key: str = "choices"
    subject_key: str | None = None
    solution_key: str | None = None
    shuffle_choices: bool = False
    enable_thinking: bool = False
    tool_selection: tuple[str, ...] = ()


TRAIN_DATASETS = [
    DatasetConfig(
        name="big_math_rl_verified",
        dataset_id="SynthLabsAI/Big-Math-RL-Verified",
        split="train",
        adapter="math",
        data_source="SynthLabsAI/Big-Math-RL-Verified",
        question_key="problem",
        answer_key="answer",
        subject_key="domain",
        sample_size=50_000,
        enable_thinking=False,
        tool_selection=("display_answers",),
    ),
    DatasetConfig(
        name="deepmath103k",
        dataset_id="zwhe99/DeepMath-103K",
        split="train",
        adapter="math",
        data_source="zwhe99/DeepMath-103K",
        question_key="question",
        answer_key="final_answer",
        solution_key="r1_solution_1",
        subject_key="topic",
        sample_size=50_000,
        enable_thinking=False,
        tool_selection=("display_answers",),
    ),
    DatasetConfig(
        name="if_rl_singleturn",
        dataset_id="swiss-ai/if-rl-singleturn-prompts",
        split="train",
        adapter="if_data",
        data_source="swiss-ai/if-rl-singleturn-prompts",
        prompt_key="messages",
        answer_key="ground_truth",
        subject_key="constraint_type",
        sample_size=40_000,
    ),
    DatasetConfig(
        name="if_rl_singleturn_hard",
        dataset_id="swiss-ai/if-rl-singleturn-hard-prompts",
        split="train",
        adapter="if_data",
        data_source="swiss-ai/if-rl-singleturn-hard-prompts",
        prompt_key="messages",
        answer_key="ground_truth",
        subject_key="constraint_type",
        sample_size=10_000,
    ),
    DatasetConfig(
        name="taco_verified",
        dataset_id="likaixin/TACO-verified",
        split="train",
        adapter="taco",
        data_source="taco",
        question_key="question",
        answer_key="input_output",
        solution_key="solutions",
        subject_key="source",
    ),
    DatasetConfig(
        name="apps",
        dataset_id="ReactiveAI/codeparrot-apps-reupload",
        split="train",
        adapter="apps",
        data_source="apps",
        question_key="question",
        answer_key="input_output",
        solution_key="solutions",
    ),
    DatasetConfig(
        name="rgym",
        dataset_id="/capstor/store/cscs/swissai/infra01/reasoning/data/RL-prod/rgym/train.parquet",
        split="train",
        adapter="rgym",
        data_source="rgym",
        prompt_key="prompt",
        sample_size=50_000,
        tool_selection=("display_answers",),
    ),
    DatasetConfig(
        name="riddle_sense",
        dataset_id="/capstor/store/cscs/swissai/infra01/reasoning/data/RL-prod/riddle_sense/train.parquet",
        split="train",
        adapter="riddle_sense",
        data_source="riddle_sense",
        question_key="question",
        choices_key="choices",
        answer_key="answerKey",
        enable_thinking=False,
        tool_selection=("display_answers",),
    ),
    DatasetConfig(
        name="lexam_mcq",
        dataset_id="LEXam-Benchmark/LEXam",
        subset="mcq_4_choices",
        split="test",
        adapter="lexam_mcq",
        data_source="lexam_mcq",
        question_key="question",
        choices_key="choices",
        answer_key="gold",
        subject_key="area",
        enable_thinking=False,
        tool_selection=("display_answers",),
    ),
    DatasetConfig(
        name="table_gpt",
        dataset_id=TABLE_GPT_DATASET_ID,
        split="mixed",
        adapter="table_gpt",
        data_source="table_gpt",
        prompt_key="prompt",
        answer_key="completion",
    ),
    DatasetConfig(
        name="qa_gym",
        dataset_id="/capstor/store/cscs/swissai/infra01/reasoning/data/RL-prod/qa_gym/eval10_hybrid_multihop_rl_pairs.jsonl",
        split="train",
        adapter="qa_gym",
        data_source="qa_gym",
        question_key="question",
        sample_size=None,
        tool_selection=("display_answers",),
    ),
    DatasetConfig(
        name="vrl",
        dataset_id="/capstor/store/cscs/swissai/infra01/reasoning/data/RL-prod/vrl/train.parquet",
        split="train",
        adapter="vrl",
        data_source="blindtasks_rl",
        prompt_key="prompt",
        tool_selection=("display_answers",),
    ),
    DatasetConfig(
        name="code_contests",
        dataset_id="deepmind/code_contests",
        split="train",
        adapter="code_contests",
        data_source="codecontests",
        question_key="description",
        solution_key="solutions",
    ),
    DatasetConfig(
        name="open_r1_codeforces",
        dataset_id="open-r1/codeforces",
        subset="verifiable",
        split="train",
        adapter="open_r1_codeforces",
        data_source="codeforces",
    ),
    DatasetConfig(
        name="toolgym",
        dataset_id="/capstor/store/cscs/swissai/infra01/reasoning/data/RL-prod/toolgym_test_v2/dataset/train.parquet",
        split="train",
        adapter="tools",
        data_source="tool_gym",
        enable_thinking=True,
        # tool_selection is done in the adapter
    ),
]

EVAL_DATASETS = [
    DatasetConfig(
        name="rgym",
        dataset_id="/capstor/store/cscs/swissai/infra01/reasoning/data/RL-prod/rgym/val_mini.parquet",
        split="val_mini",
        adapter="rgym",
        data_source="rgym",
        prompt_key="prompt",
        sample_size=200,
    ),
    DatasetConfig(
        name="gsm8k",
        dataset_id="openai/gsm8k",
        subset="main",
        split="test",
        adapter="gsm8k",
        data_source="openai/gsm8k",
        question_key="question",
        answer_key="answer",
        sample_size=100,
        enable_thinking=False,
        tool_selection=("display_answers",),
    ),
    DatasetConfig(
        name="math500",
        dataset_id="HuggingFaceH4/MATH-500",
        split="test",
        adapter="math",
        data_source="HuggingFaceH4/MATH-500",
        question_key="problem",
        answer_key="answer",
        subject_key="subject",
        solution_key="solution",
        sample_size=50,
        enable_thinking=False,
        tool_selection=("display_answers",),
    ),
    DatasetConfig(
        name="aime2024",
        dataset_id="HuggingFaceH4/aime_2024",
        split="train",
        adapter="math",
        data_source="aime2024",
        question_key="problem",
        answer_key="answer",
        solution_key="solution",
        enable_thinking=False,
        tool_selection=("display_answers",),
    ),
    DatasetConfig(
        name="aime2025",
        dataset_id="math-ai/aime25",
        split="test",
        adapter="math",
        data_source="aime2025",
        question_key="problem",
        answer_key="answer",
        enable_thinking=False,
        tool_selection=("display_answers",),
    ),
    DatasetConfig(
        name="gpqa_diamond",
        dataset_id="Idavidrein/gpqa",
        subset="gpqa_diamond",
        split="train",
        adapter="gpqa",
        data_source="gpqa_diamond",
        question_key="Question",
        answer_key="Correct Answer",
        shuffle_choices=True,
        sample_size=100,
        enable_thinking=False,
        tool_selection=("display_answers",),
    ),
    DatasetConfig(
        name="mmlu",
        dataset_id="cais/mmlu",
        subset="all",
        split="test",
        adapter="multiple_choice",
        data_source="mmlu",
        question_key="question",
        choices_key="choices",
        answer_key="answer",
        subject_key="subject",
        sample_size=100,
        enable_thinking=False,
        tool_selection=("display_answers",),
    ),
    DatasetConfig(
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
    DatasetConfig(
        name="ifeval",
        dataset_id="google/IFEval",
        split="train",
        adapter="if_eval",
        data_source="google/IFEval",
        prompt_key="prompt",
        sample_size=100,
    ),
    DatasetConfig(
        name="ifbench",
        dataset_id="allenai/IFBench_test",
        split="train",
        adapter="if_eval",
        data_source="allenai/IFBench_test",
        prompt_key="prompt",
        sample_size=100,
    ),
    DatasetConfig(
        name="toolgym",
        dataset_id="/capstor/store/cscs/swissai/infra01/reasoning/data/RL-prod/toolgym_test_v2/dataset/val.parquet",
        split="train",
        adapter="tools",
        data_source="tool_gym",
        enable_thinking=False,
        # tool_selection is done in the adapter
    ),
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
    return "" if value is None else str(value).strip()


def get_value(example: dict[str, Any], key: str | None, default: Any = None) -> Any:
    if key is None:
        return default
    return example.get(key, default)


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


def make_prompt(user_content: str, system_content: str = "") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    messages = parse_json_maybe(messages, messages)
    if not isinstance(messages, list):
        return make_prompt(normalize_text(messages))

    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        normalized.append(
            {
                "role": normalize_text(message.get("role")) or "user",
                "content": normalize_text(message.get("content")),
            }
        )
    if not normalized or normalized[0].get("role") != "system":
        normalized.insert(0, {"role": "system", "content": ""})
    return normalized


def maybe_strip_rgym_format_instructions(
    prompt: list[dict[str, str]], config: DatasetConfig, language: str | None = None
) -> list[dict[str, str]]:
    """Strips out the output format instructions if "display_answers" tool is selected"""
    if "display_answers" not in config.tool_selection:
        return prompt

    normalized = []
    for message in prompt:
        content, _ = strip_rgym_format_instructions(message["content"], language)
        normalized.append({**message, "content": content})
    return normalized


def prompt_controls(config: DatasetConfig) -> dict[str, Any]:
    controls = {
        "tool_selection": list(config.tool_selection),
        "apply_chat_template_kwargs": {"enable_thinking": config.enable_thinking},
    }
    # if "display_answers" in config.tool_selection:
    #     controls["sampling_params"] = {"ebnf": DISPLAY_ANSWERS_EBNF}
    # controls["sampling_params"] = {"ebnf": DISPLAY_ANSWERS_EBNF}
    return controls


def agent_name(config: DatasetConfig) -> str:
    return agent_name_for_tools(config.tool_selection)


def agent_name_for_tools(tool_selection: Any) -> str:
    if tool_selection:
        return "tool_agent"
    return "single_turn_agent"


def source_controls(example: dict[str, Any]) -> dict[str, Any]:
    extra_info = parse_json_maybe(example.get("extra_info"), default={})
    if not isinstance(extra_info, dict):
        return {}
    # keys = ("tool_selection", "apply_chat_template_kwargs", "sampling_params")
    keys = ("tool_selection", "apply_chat_template_kwargs")
    return {key: extra_info[key] for key in keys if key in extra_info}


def system_prompt(example: dict[str, Any], config: DatasetConfig) -> str:
    if not config.system_key:
        return ""
    return normalize_text(get_value(example, config.system_key))


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
        "data_source": config.data_source,
        # "agent_name": agent_name(config),
        "prompt": prompt,
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "dataset": config.name,
            "dataset_id": config.dataset_id,
            "split": split,
            "index": index,
            "tool_selection": [],
            "apply_chat_template_kwargs": {"enable_thinking": False},
            **prompt_controls(config),
        },
    }
    if extra_info:
        row["extra_info"].update(extra_info)
    return row


@register_adapter("qa_gym")
def adapt_qa_gym(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    prompt = normalize_text(get_value(example, config.prompt_key))
    question = normalize_text(get_value(example, config.question_key)) or prompt
    answer = normalize_text(get_value(example, config.answer_key))
    extra_info = {
        "question": question,
        "qa_id": normalize_text(get_value(example, "id")),
    }
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=make_prompt(prompt),
        ability="long_context_qa",
        ground_truth=answer,
        extra_info=extra_info,
    )


@register_adapter("rgym")
def adapt_rgym(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    row = dict(example)
    prompt = normalize_messages(get_value(example, config.prompt_key))
    extra_info = dict(
        parse_json_maybe(get_value(example, "extra_info"), default={}) or {}
    )
    prompt = maybe_strip_rgym_format_instructions(
        prompt, config, normalize_text(extra_info.get("language")) or None
    )
    row["prompt"] = prompt
    row["data_source"] = config.data_source
    # row["agent_name"] = agent_name(config)
    extra_info.update(prompt_controls(config))
    extra_info["source_dataset"] = normalize_text(get_value(example, "data_source"))
    row["extra_info"] = extra_info
    return row


@register_adapter("table_gpt")
def adapt_table_gpt(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    task = normalize_text(get_value(example, "task")) or normalize_text(
        get_value(example, "_table_gpt_task")
    )
    ground_truth = get_value(example, config.answer_key)
    if not isinstance(ground_truth, str):
        ground_truth = json_dumps(ground_truth)
    metadata = parse_json_maybe(get_value(example, "metadata"), default={})
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "data_source": f"tablegpt/{task}",
        # "agent_name": agent_name(config),
        "prompt": make_prompt(normalize_text(get_value(example, config.prompt_key))),
        "ability": task,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "dataset": config.name,
            "dataset_id": config.dataset_id,
            "split": normalize_text(get_value(example, "_table_gpt_split")) or split,
            "index": idx,
            "task": task,
            "source_dataset": normalize_text(get_value(example, "dataset")),
            "metadata": json_dumps(metadata),
            "tool_selection": [],
            "apply_chat_template_kwargs": {"enable_thinking": False},
            **prompt_controls(config),
        },
    }


@register_adapter("vrl")
def adapt_vrl(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    reward_model = parse_json_maybe(get_value(example, "reward_model"), default={})
    if not isinstance(reward_model, dict):
        reward_model = {}
    ground_truth = reward_model.get(
        "ground_truth", get_value(example, config.answer_key)
    )
    if not isinstance(ground_truth, str):
        ground_truth = json_dumps(ground_truth)

    extra_info = parse_json_maybe(get_value(example, "extra_info"), default={})
    if not isinstance(extra_info, dict):
        extra_info = {}
    row = make_row(
        config=config,
        split=split,
        index=idx,
        prompt=normalize_messages(get_value(example, config.prompt_key)),
        ability=normalize_text(get_value(example, "ability")) or "vision.blindtasks",
        ground_truth=ground_truth,
        extra_info=dict(extra_info),
    )
    row["reward_model"]["style"] = reward_model.get("style", "rule")
    return row


@register_adapter("tools")
def adapt_tools(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    row = dict(example)
    extra_info = dict(row.get("extra_info") or {})
    extra_info["index"] = idx
    reward_model = dict(row.get("reward_model") or {})
    ground_truth = reward_model.get("ground_truth")
    if not isinstance(ground_truth, str):
        reward_model["ground_truth"] = json_dumps(ground_truth)
    row["data_source"] = normalize_text(row.get("data_source")) or config.data_source
    row["reward_model"] = reward_model
    row["extra_info"] = extra_info
    # row["agent_name"] = agent_name_for_tools(extra_info.get("tool_selection"))
    row.pop("tools", None)
    return row


@register_adapter("math")
def adapt_math(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    question = normalize_text(get_value(example, config.question_key))
    answer = normalize_text(get_value(example, config.answer_key))
    extra_info = {"question": question, "answer": answer}
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
        prompt=make_prompt(question, system_prompt(example, config)),
        ability="math",
        ground_truth=answer,
        extra_info=extra_info,
    )


@register_adapter("gsm8k")
def adapt_gsm8k(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    question = normalize_text(get_value(example, config.question_key))
    raw_answer = normalize_text(get_value(example, config.answer_key))
    try:
        answer = extract_gsm8k_solution(raw_answer)
    except Exception:
        answer = raw_answer.split("####")[-1].strip()
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=make_prompt(question, system_prompt(example, config)),
        ability="math",
        ground_truth=answer,
        extra_info={"question": question, "answer": raw_answer},
    )


def make_code_row(
    example: dict[str, Any],
    idx: int,
    split: str,
    config: DatasetConfig,
    sandbox_data_source: str,
) -> dict[str, Any]:
    question = normalize_text(get_value(example, config.question_key))
    starter_code = normalize_text(get_value(example, "starter_code"))
    raw_test_cases = parse_json_maybe(get_value(example, config.answer_key), default={})
    prime_code_test_cases = normalize_prime_code_test_cases(raw_test_cases)
    raw_solutions = get_value(example, config.solution_key)
    reference_solution = first_solution(parse_json_maybe(raw_solutions, raw_solutions))
    extra_info = {
        "question": question,
        "starter_code": starter_code,
        "difficulty": normalize_text(get_value(example, "difficulty")),
        "source": normalize_text(get_value(example, "source")),
        "name": normalize_text(get_value(example, "name")),
        "url": normalize_text(get_value(example, "url")),
        "problem_id": normalize_text(get_value(example, "problem_id")),
        "reference_solution": reference_solution,
        "language": "python",
        "input_output": json_dumps(raw_test_cases),
        "prime_code_input_output": json_dumps(prime_code_test_cases),
        "sandbox_data_source": sandbox_data_source,
    }
    if config.subject_key and config.subject_key in example:
        extra_info["subject"] = normalize_text(example[config.subject_key])
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=make_prompt(format_code_prompt(question, starter_code)),
        ability="code",
        ground_truth=reference_solution,
        extra_info=extra_info,
    )


@register_adapter("taco")
def adapt_taco(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    return make_code_row(example, idx, split, config, "likaixin/TACO-verified")


@register_adapter("apps")
def adapt_apps(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    return make_code_row(example, idx, split, config, "likaixin/TACO-verified")


@register_adapter("code_contests")
def adapt_code_contests(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    question = normalize_text(get_value(example, config.question_key))
    test_cases = normalize_code_contests_test_cases(example)
    reference_solution = first_code_contests_solution(
        get_value(example, config.solution_key)
    )
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=make_prompt(format_code_prompt(question, "")),
        ability="code",
        ground_truth=reference_solution,
        extra_info={
            "question": question,
            "difficulty": normalize_text(get_value(example, "difficulty")),
            "source": normalize_text(get_value(example, "source")),
            "name": normalize_text(get_value(example, "name")),
            "reference_solution": reference_solution,
            "language": "python",
            "input_output": json_dumps(test_cases),
            "prime_code_input_output": json_dumps(test_cases),
            "sandbox_data_source": "likaixin/TACO-verified",
            "num_used_tests": len(test_cases["inputs"]),
        },
    )


@register_adapter("open_r1_codeforces")
def adapt_codeforces(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    question = format_codeforces_prompt(example)
    test_cases = normalize_codeforces_test_cases(example)
    ground_truth = json_dumps(test_cases)
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=make_prompt(format_code_prompt(question, "")),
        ability="code",
        ground_truth=ground_truth,
        extra_info={
            "question": question,
            "problem_id": normalize_text(get_value(example, "id")),
            "contest_id": normalize_text(get_value(example, "contest_id")),
            "cf_index": normalize_text(get_value(example, "index")),
            "title": normalize_text(get_value(example, "title")),
            "difficulty": normalize_text(get_value(example, "rating")),
            "tags": json_dumps(get_value(example, "tags", [])),
            "num_used_tests": len(test_cases["inputs"]),
            "language": "python",
            "input_output": ground_truth,
            "prime_code_input_output": ground_truth,
            "sandbox_data_source": "likaixin/TACO-verified",
        },
    )


@register_adapter("humaneval")
def adapt_humaneval(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    code_prompt = normalize_text(get_value(example, config.question_key)).rstrip()
    if not code_prompt.endswith("\n"):
        code_prompt += "\n"
    entry_point = normalize_text(get_value(example, config.prompt_key))
    ground_truth = {
        "prompt": code_prompt,
        "test": normalize_text(get_value(example, config.answer_key)).rstrip(),
        "entry_point": entry_point,
    }
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=make_prompt(format_code_prompt(code_prompt, "")),
        ability="code",
        ground_truth=json_dumps(ground_truth),
        extra_info={
            "task_id": get_value(example, "task_id"),
            "entry_point": entry_point,
            "canonical_solution": get_value(example, config.solution_key, ""),
        },
    )


@register_adapter("if_data")
def adapt_if_data(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    extra_info = {
        "key": get_value(example, "key"),
        "source_dataset": get_value(example, "dataset"),
        "constraint": get_value(example, "constraint"),
        "constraint_type": get_value(example, "constraint_type"),
        "verification": "placeholder",
    }
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=normalize_messages(get_value(example, config.prompt_key)),
        ability="instruction_following",
        ground_truth=get_value(example, config.answer_key),
        extra_info=extra_info,
    )


@register_adapter("if_eval")
def adapt_if_eval(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    instruction_ids = get_value(example, "instruction_id_list", [])
    kwargs_list = get_value(example, "kwargs", [])
    ground_truth = {"instruction_id": instruction_ids, "kwargs": kwargs_list}
    ground_truth_json = json_dumps(ground_truth)
    extra_info = {
        "key": normalize_text(get_value(example, "key")),
        "constraint": ground_truth_json,
        "instruction_id_list": json_dumps(instruction_ids),
    }
    if config.subject_key and config.subject_key in example:
        extra_info["subject"] = example[config.subject_key]
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=make_prompt(
            normalize_text(get_value(example, config.prompt_key)),
            system_prompt(example, config),
        ),
        ability="instruction_following",
        ground_truth=ground_truth_json,
        extra_info=extra_info,
    )


@register_adapter("multiple_choice")
def adapt_multiple_choice(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    question = normalize_text(get_value(example, config.question_key))
    choices, answer_letter = maybe_shuffle_choices(
        list(get_value(example, config.choices_key, [])),
        normalize_answer_index(get_value(example, config.answer_key)),
        config,
        idx,
    )
    extra_info = {
        "question": question,
        "choices": choices,
        "answer_index": answer_letter,
    }
    if config.subject_key and config.subject_key in example:
        extra_info["subject"] = normalize_text(example[config.subject_key])
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=make_prompt(format_multiple_choice_prompt(question, choices)),
        ability="knowledge",
        ground_truth=answer_letter,
        extra_info=extra_info,
    )


@register_adapter("riddle_sense")
def adapt_riddle_sense(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    normalized = normalize_riddle_sense_sample(example)
    choices, answer_letter = maybe_shuffle_choices(
        normalized["choices"], normalized["answer_index"], config, idx
    )
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=make_prompt(
            format_multiple_choice_prompt(normalized["question"], choices)
        ),
        ability="knowledge",
        ground_truth=answer_letter,
        extra_info={
            "question": normalized["question"],
            "choices": choices,
            "answer_index": answer_letter,
            "answer_text": normalized["answer_text"],
            **normalized["metadata"],
        },
    )


@register_adapter("lexam_mcq")
def adapt_lexam_mcq(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    normalized = normalize_lexam_mcq_sample(example)
    choices, answer_letter = maybe_shuffle_choices(
        normalized["choices"], normalized["answer_index"], config, idx
    )
    metadata = dict(normalized["metadata"])
    if config.subject_key and config.subject_key in metadata:
        metadata["subject"] = normalize_text(metadata[config.subject_key])
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=make_prompt(
            format_multiple_choice_prompt(normalized["question"], choices)
        ),
        ability="knowledge",
        ground_truth=answer_letter,
        extra_info={
            "question": normalized["question"],
            "choices": choices,
            "answer_index": answer_letter,
            "answer_text": normalized["answer_text"],
            **metadata,
        },
    )


@register_adapter("gpqa")
def adapt_gpqa(
    example: dict[str, Any], idx: int, split: str, config: DatasetConfig
) -> dict[str, Any]:
    question = normalize_text(get_value(example, config.question_key))
    choices = [
        normalize_text(get_value(example, config.answer_key)),
        normalize_text(get_value(example, "Incorrect Answer 1")),
        normalize_text(get_value(example, "Incorrect Answer 2")),
        normalize_text(get_value(example, "Incorrect Answer 3")),
    ]
    choices, answer_letter = maybe_shuffle_choices(
        [choice for choice in choices if choice], 0, config, idx
    )
    return make_row(
        config=config,
        split=split,
        index=idx,
        prompt=make_prompt(format_multiple_choice_prompt(question, choices)),
        ability="knowledge",
        ground_truth=answer_letter,
        extra_info={"question": question, "choices": choices},
    )


def format_multiple_choice_prompt(question: str, choices: list[Any]) -> str:
    options = "\n".join(
        f"{chr(ord('A') + i)}. {normalize_text(choice)}"
        for i, choice in enumerate(choices)
    )
    return f"{question}\n\n{options}"


def format_code_prompt(question: str, starter_code: str) -> str:
    if starter_code:
        question = f"{question}\n\n```python\n{starter_code}\n```"
    return f"{question}\n\n{CODE_FINAL_ANSWER_INSTRUCTION}"


def format_codeforces_prompt(example: dict[str, Any]) -> str:
    sections = []
    for title, key in (
        ("", "title"),
        ("", "description"),
        ("Input", "input_format"),
        ("Output", "output_format"),
        ("Note", "note"),
    ):
        value = normalize_text(get_value(example, key))
        if value:
            sections.append(f"{title}\n{value}" if title else value)

    examples = format_codeforces_examples(get_value(example, "examples", []))
    if examples:
        sections.append(f"Examples\n{examples}")
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
        if test_input or test_output:
            formatted.append(f"Input\n{test_input}\n\nOutput\n{test_output}")
    return "\n\n".join(formatted)


def first_solution(solutions: Any) -> str:
    if isinstance(solutions, list) and solutions:
        return normalize_text(solutions[0])
    return normalize_text(solutions)


def first_code_contests_solution(solutions: Any) -> str:
    if isinstance(solutions, dict):
        return first_solution(solutions.get("solution", []))
    return first_solution(solutions)


def normalize_prime_code_test_cases(test_cases: Any) -> dict[str, Any]:
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
        normalized["inputs"] = [serialize_standard_io(case) for case in inputs]
        normalized["outputs"] = [serialize_standard_io(case) for case in outputs]
    return normalized


def normalize_code_contests_test_cases(example: dict[str, Any]) -> dict[str, list[str]]:
    inputs: list[str] = []
    outputs: list[str] = []
    for key, max_cases in (
        ("public_tests", None),
        ("private_tests", None),
        ("generated_tests", CODE_CONTESTS_MAX_GENERATED_TESTS),
    ):
        test_inputs = get_test_values(get_value(example, key), "input")
        test_outputs = get_test_values(get_value(example, key), "output")
        if max_cases is not None:
            test_inputs = test_inputs[:max_cases]
            test_outputs = test_outputs[:max_cases]
        for test_input, test_output in zip(test_inputs, test_outputs):
            inputs.append(normalize_text(test_input))
            outputs.append(normalize_text(test_output))
    if not inputs:
        raise ValueError("CodeContests example has no usable tests")
    return {"inputs": inputs, "outputs": outputs}


def normalize_codeforces_test_cases(example: dict[str, Any]) -> dict[str, list[str]]:
    inputs: list[str] = []
    outputs: list[str] = []
    for test_case in get_value(example, "official_tests", []) or []:
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
    try:
        tests = normalize_code_contests_test_cases(example)
    except ValueError:
        return False
    return len(tests["inputs"]) == len(tests["outputs"])


def apps_has_tests(example: dict[str, Any]) -> bool:
    test_cases = parse_json_maybe(get_value(example, "input_output"), default={})
    if not isinstance(test_cases, dict):
        return False
    inputs = test_cases.get("inputs", [])
    outputs = test_cases.get("outputs", [])
    return bool(inputs) and len(inputs) == len(outputs)


def codeforces_has_tests(example: dict[str, Any]) -> bool:
    if normalize_text(get_value(example, "input_mode")) != "stdio":
        return False
    if normalize_text(get_value(example, "interaction_format")):
        return False
    if normalize_text(get_value(example, "generated_checker")):
        return False
    official_tests = get_value(example, "official_tests", []) or []
    return bool(official_tests) and all(
        isinstance(test_case, dict) and "input" in test_case and "output" in test_case
        for test_case in official_tests
    )


def get_test_values(test_group: Any, key: str) -> list[Any]:
    if not test_group:
        return []
    if isinstance(test_group, dict):
        return list(test_group.get(key, []) or [])
    if isinstance(test_group, list):
        return [
            test_case.get(key, "")
            for test_case in test_group
            if isinstance(test_case, dict)
        ]
    return []


def serialize_call_based_input(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(json_dumps(item) for item in value)
    return json_dumps(value)


def serialize_standard_io(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    return str(value)


def normalize_answer_index(answer: Any) -> int:
    if isinstance(answer, int):
        return answer
    text = normalize_text(answer)
    if text.isdigit():
        return int(text)
    match = re.search(r"\b([A-D])\b", text.upper())
    if match:
        return ord(match.group(1)) - ord("A")
    raise ValueError(f"Cannot normalize multiple-choice answer: {answer!r}")


def maybe_shuffle_choices(
    choices: list[Any], answer_index: int, config: DatasetConfig, idx: int
) -> tuple[list[str], str]:
    indexed_choices = list(enumerate(normalize_text(choice) for choice in choices))
    if config.shuffle_choices:
        random.Random(f"{SEED}:{config.name}:{idx}").shuffle(indexed_choices)
    new_choices = [choice for _, choice in indexed_choices]
    new_answer_index = next(
        new_idx
        for new_idx, (old_idx, _) in enumerate(indexed_choices)
        if old_idx == answer_index
    )
    return new_choices, chr(ord("A") + new_answer_index)


def load_raw_dataset(config: DatasetConfig) -> datasets.Dataset:
    load_kwargs = {"name": config.subset} if config.subset else {}
    if config.adapter == "table_gpt":
        raw_dataset = load_table_gpt_mix(
            config.dataset_id,
            cache_dir=os.path.expanduser(DATASETS_CACHE_DIR),
        )
    elif config.adapter == "qa_gym":
        raw_dataset = load_qa_gym_rl_pairs_jsonl(config.dataset_id)
    elif config.dataset_id.endswith(".parquet"):
        raw_dataset = datasets.load_dataset(
            "parquet",
            data_files={config.split: config.dataset_id},
            split=config.split,
            cache_dir=os.path.expanduser(DATASETS_CACHE_DIR),
        )
    elif config.dataset_id.endswith((".json", ".jsonl")):
        raw_dataset = datasets.load_dataset(
            "json",
            data_files={config.split: config.dataset_id},
            split=config.split,
            cache_dir=os.path.expanduser(DATASETS_CACHE_DIR),
        )
    elif os.path.isdir(config.dataset_id):
        raw_dataset = datasets.load_from_disk(
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

    if config.adapter in {"apps", "taco"}:
        raw_dataset = raw_dataset.filter(lambda example: apps_has_tests(dict(example)))
    elif config.adapter == "code_contests":
        raw_dataset = raw_dataset.filter(
            lambda example: code_contests_has_tests(dict(example))
        )
    elif config.adapter == "open_r1_codeforces":
        raw_dataset = raw_dataset.filter(
            lambda example: codeforces_has_tests(dict(example))
        )

    if config.sample_size is not None:
        sample_size = min(config.sample_size, len(raw_dataset))
        raw_dataset = raw_dataset.shuffle(seed=SEED).select(range(sample_size))
        print(f"Sampled {len(raw_dataset)} rows for {config.name}.", flush=True)
    return raw_dataset


def preprocess_example(
    example: dict[str, Any], idx: int, config: DatasetConfig
) -> dict[str, Any]:
    adapter = ADAPTERS[config.adapter]
    example = dict(example)
    row = adapter(example, idx, config.split, config)
    row["extra_info"].update(source_controls(example))
    # row["agent_name"] = agent_name_for_tools(row["extra_info"].get("tool_selection"))
    return row


def normalize_feature_types(feature: Any) -> Any:
    """Normalize Arrow feature types that block concatenating processed datasets."""
    if isinstance(feature, datasets.Value) and feature.dtype == "large_string":
        return datasets.Value("string")
    if isinstance(feature, datasets.List):
        return datasets.List(
            normalize_feature_types(feature.feature),
            length=feature.length,
            id=feature.id,
        )
    if isinstance(feature, dict):
        return {key: normalize_feature_types(value) for key, value in feature.items()}
    return feature


def cast_output_features(dataset: datasets.Dataset) -> datasets.Dataset:
    """Cast output schema while preserving runtime values such as empty tool lists."""
    features = datasets.Features(normalize_feature_types(dataset.features))
    if "extra_info" not in features or "tool_selection" not in features["extra_info"]:
        return dataset.cast(features)
    features["extra_info"]["tool_selection"] = datasets.Sequence(
        datasets.Value("string")
    )
    return dataset.cast(features)


def preprocess_dataset(config: DatasetConfig) -> datasets.Dataset:
    if config.adapter not in ADAPTERS:
        raise KeyError(f"No adapter registered for {config.adapter!r}")
    print(f"Loading {config.name} from {config.dataset_id}...", flush=True)
    raw_dataset = load_raw_dataset(config)
    dataset = raw_dataset.map(
        preprocess_example,
        with_indices=True,
        fn_kwargs={"config": config},
        remove_columns=raw_dataset.column_names,
        num_proc=PREPROCESS_NUM_PROC,
        desc=f"Preprocessing {config.name}",
    )
    return cast_output_features(dataset)


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
    output_dir = os.path.expanduser(LOCAL_SAVE_DIR)
    eval_dir = os.path.join(output_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    val_dataset = concatenate_named(eval_datasets.items())
    train_dataset.to_parquet(os.path.join(output_dir, "train.parquet"))
    val_dataset.to_parquet(os.path.join(output_dir, "val.parquet"))
    for name, dataset in eval_datasets.items():
        dataset.to_parquet(os.path.join(eval_dir, f"{name}.parquet"))

    save_json_example(train_dataset, os.path.join(output_dir, "train_example.json"))
    save_json_example(val_dataset, os.path.join(output_dir, "val_example.json"))
    print(
        f"Wrote {len(train_dataset)} train and {len(val_dataset)} validation rows "
        f"to {output_dir}.",
        flush=True,
    )


def main() -> None:
    train_dataset = concatenate_named(
        (config.name, preprocess_dataset(config)) for config in TRAIN_DATASETS
    )
    eval_datasets = {
        config.name: preprocess_dataset(config) for config in EVAL_DATASETS
    }
    write_outputs(train_dataset, eval_datasets)


if __name__ == "__main__":
    main()
