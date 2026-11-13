"""
This file contains code inspired by and adapted from the Open-Reasoner-Zero project.
Original Repository: https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero
"""

import asyncio
import random
import re
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List, Optional, Tuple

import wandb
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from math_verify.errors import TimeoutException
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    ScoredDataGroup,
    ServerBaseline,
)

prompt_format = (
    "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant "
    "first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning "
    "process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, "
    "i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {prompt}\nAssistant: <think>"
)

problem_format = """You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{{}} tag.
This is the problem:
{problem}
"""  # noqa: E501

stop_list = ["User:", "Human:", "Assistant:", "</answer>"]


class RSConfig(BaseEnvConfig):
    run_evaluation: bool = Field(True, description="If this should run evaluation")
    mask_too_long_completions: bool = Field(
        True, description="If this should mask too long completions"
    )
    percent_length_penalty: float = Field(
        0.0, description="The percentage of items to have length penalty"
    )


def score_answer(gold, resp) -> Optional[bool]:
    pattern = re.compile(r"<answer>.*?(\\boxed{.*}).*?</answer>", re.DOTALL)
    matches = pattern.findall(resp)
    resp = matches[-1] if matches else None
    if resp is None:
        return False
    try:
        gold_parsed = parse(
            gold,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
    except (Exception, TimeoutException, KeyError, TypeError, NotImplementedError):
        return None
    if len(gold_parsed) != 0:
        try:
            answer_parsed = parse(
                resp,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
        except (
            Exception,
            TimeoutException,
            KeyError,
            TypeError,
            NotImplementedError,
        ):
            # Can't parse, so we skip
            return None
        # Reward 1 if the content is the same as the ground truth, 0 otherwise
        try:
            return verify(answer_parsed, gold_parsed)
        except (
            Exception,
            TimeoutException,
            KeyError,
            TypeError,
            NotImplementedError,
        ):
            return None
    return None


class MathEnv(BaseEnv):

    name = "math"
    env_config_cls = RSConfig

    def __init__(
        self,
        config: RSConfig,
        server_configs: ServerBaseline,
        slurm=True,
        testing=False,
    ):
        print("Initializing MathEnv")
        print(f"Slurm: {slurm}, Testing: {testing}")
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        self.mp_executor = ProcessPoolExecutor(64)
        self.percent_overanswer = list()
        self.correct_answer_len = list()
        self.incorrect_answer_len = list()
        self.normal_rollouts = list()
        self.pass_at_groupsize = list()
        self.iter = 0

    @classmethod
    def config_init(cls) -> Tuple[RSConfig, ServerBaseline]:
        env_config = RSConfig(
            tokenizer_name="Qwen/Qwen2.5-7B",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=1024,
            steps_per_eval=25,
            max_token_length=31000,  # 22000 // (2 ** i),
            wandb_name="math",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
        )
        server_configs = ServerBaseline(
            model_name="default",
            num_requests_for_eval=256,  # since evaling only on one...
        )

        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = dict()
        if len(self.pass_at_groupsize) > 0:
            wandb_metrics["train/pass_at_groupsize"] = sum(
                self.pass_at_groupsize
            ) / len(self.pass_at_groupsize)
            self.pass_at_8 = list()
        if len(self.percent_correct_buffer) > 0:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
            wandb_metrics["train/percent_overanswer"] = sum(
                self.percent_overanswer
            ) / len(self.percent_overanswer)
            self.percent_overthink = list()
            self.percent_overanswer = list()
            self.percent_correct_buffer = list()
        if len(self.correct_answer_len) > 0:
            wandb_metrics["train/avg_correct_answer_len"] = sum(
                self.correct_answer_len
            ) / len(self.correct_answer_len)
            self.correct_answer_len = list()
        if len(self.incorrect_answer_len) > 0:
            wandb_metrics["train/avg_incorrect_answer_len"] = sum(
                self.incorrect_answer_len
            ) / len(self.incorrect_answer_len)
            self.incorrect_answer_len = list()
        # create tables
        if len(self.normal_rollouts) > 0:
            table = wandb.Table(columns=["problem", "solution", "answer", "score"])
            for group in self.normal_rollouts:
                table.add_data(group[0], group[1], group[2], group[3])
            wandb_metrics["train/normal_rollouts"] = table
        wandb_metrics["train/iter"] = self.iter
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        self.train = load_dataset("zwhe99/DeepMath-103K", split="train").shuffle(
            seed=42
        )
        aime_test_data = load_dataset("HuggingFaceH4/aime_2024", split="train")
        math500_test_data = load_dataset("HuggingFaceH4/math-500", split="test")
        amc_test_data = load_dataset("math-ai/amc23", split="test")
        minerva_test_data = load_dataset("math-ai/minervamath", split="test")
        olympiad_test_data = load_dataset("math-ai/olympiadbench", split="test")
        self.test = list()
        for name, t_dataset in zip(
            ["aime24", "math500"], [aime_test_data, math500_test_data]
        ):
            for item in t_dataset:
                self.test.append(
                    (
                        prompt_format.format(
                            prompt=problem_format.format(problem=item["problem"])
                        ),
                        item["answer"],
                        name,
                    )
                )
        for name, t_dataset in zip(
            ["amc23", "minerva"],
            [amc_test_data, minerva_test_data],
        ):
            for item in t_dataset:
                self.test.append(
                    (
                        prompt_format.format(
                            prompt=problem_format.format(problem=item["question"])
                        ),
                        item["answer"],
                        name,
                    )
                )
        for name, t_dataset in zip(["olympiad"], [olympiad_test_data]):
            for item in t_dataset:
                self.test.append(
                    (
                        prompt_format.format(
                            prompt=problem_format.format(problem=item["question"])
                        ),
                        item["final_answer"][0],
                        name,
                    )
                )
        return

    async def rollout_and_score_eval(self, question, answer, subset):

        completion = await self.server.completion(
            prompt=question,
            n=1,
            max_tokens=32765,
            temperature=0.0,
            split="eval",
            stop=stop_list,
        )
        loop = asyncio.get_event_loop()
        gold = "\\boxed{" + answer + "}" if "\\boxed" not in answer else answer
        resp = completion.choices[0].text
        if completion.choices[0].finish_reason == "stop":
            if ("</answer>" not in completion.choices[0].text) and (
                "<answer>" in completion.choices[0].text
            ):
                # assume it stopped on </answer>
                resp = resp + " </answer>"
        task = loop.run_in_executor(self.mp_executor, score_answer, gold, resp)
        reward = await task
        if reward is None:
            return 0, subset
        score = 1 if reward else 0
        return score, subset

    async def evaluate(self, *args, **kwargs):
        if not self.config.run_evaluation:
            return
        eval_tasks = []
        for item in self.test:
            eval_tasks.append(self.rollout_and_score_eval(item[0], item[1], item[2]))
        parsing_data = await tqdm_asyncio.gather(*eval_tasks)
        task_lists = dict()
        for score, subset in parsing_data:
            if subset not in task_lists:
                task_lists[subset] = list()
            task_lists[subset].append(score)
        # Now get the average
        for subset, scores in task_lists.items():
            self.eval_metrics.append(
                (f"eval/{subset}_percent_correct", sum(scores) / len(scores))
            )
        # overall score
        scores = []
        for subset, score in task_lists.items():
            scores.extend(score)
        self.eval_metrics.append(
            ("eval/overall_percent_correct", sum(scores) / len(scores))
        )

    async def collect_trajectories(self, item) -> Tuple[List, List]:
        thinking_len = self.config.max_token_length
        user_prompt = prompt_format.format(
            prompt=problem_format.format(problem=item[0])
        )
        thinking_len = thinking_len - len(self.tokenizer.encode(user_prompt))
        completions = await self.server.completion(
            prompt=user_prompt,
            n=self.config.group_size,
            max_tokens=thinking_len,
            temperature=1.0,
            top_p=0.95,
            stop=stop_list,
        )
        to_score = list()
        to_backlog = list()
        for i, completion in enumerate(completions.choices):
            message = user_prompt + completion.text
            if completion.finish_reason == "stop":
                if ("</answer>" not in completion.text) and (
                    "<answer>" in completion.text
                ):
                    # assume it stopped on </answer>
                    message = message + " </answer>"
            to_score.append(
                (
                    message,
                    item[1],
                    completion.finish_reason,
                    user_prompt,
                )
            )
        to_postprocess = await self.score(to_score)
        if to_postprocess is None:
            return None, to_backlog
        if all(
            [to_postprocess["scores"][0] == score for score in to_postprocess["scores"]]
        ):
            return None, to_backlog
        self.normal_rollouts.append(
            (
                prompt_format.format(prompt=problem_format.format(problem=item[0])),
                to_postprocess["messages"][0][-1]["content"],
                item[1],
                to_postprocess["scores"][0],
            )
        )
        if len(self.normal_rollouts) > self.config.num_rollouts_to_keep:
            self.normal_rollouts.pop(0)
        print(f"Collected {len(to_postprocess['scores'])} trajectories")
        return to_postprocess, to_backlog

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        scores["overrides"] = list()
        scores["messages"] = list()
        gold = rollout_group_data[0][1]
        loop = asyncio.get_event_loop()
        random.shuffle(rollout_group_data)
        for item in rollout_group_data:
            resp = item[0]
            scores["overrides"].append(dict())
            if item[2] == "length":
                reward = False
                if self.config.mask_too_long_completions:
                    scores["overrides"][-1]["set_advantage_to_zero"] = True
            else:
                task = loop.run_in_executor(self.mp_executor, score_answer, gold, resp)
                reward = await task
                if reward is None:
                    return None
            tokens = self.tokenizer.encode(resp)
            user_prompt_tokens = self.tokenizer.encode(item[3])
            if user_prompt_tokens[-1] == self.tokenizer.eos_token_id:
                user_prompt_tokens = user_prompt_tokens[:-1]
            assert all(
                [
                    i == j
                    for i, j in zip(
                        user_prompt_tokens, tokens[: len(user_prompt_tokens)]
                    )
                ]
            )
            masks = [-100 for _ in range(len(user_prompt_tokens))]
            masks = masks + tokens[len(user_prompt_tokens) :]
            messages = [
                {"role": "user", "content": item[3]},
                {"role": "assistant", "content": resp[len(item[3]) :]},
            ]
            # remove obviously bad examples
            if len([1 for i in masks if i != -100]) < 10:
                continue
            if (item[2] == "length") and (not self.config.mask_too_long_completions):
                scores["overrides"][-1]["set_advantage_to_zero"] = True
            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(1.0 if reward else -1.0)
            scores["messages"].append(messages)
            if len(scores["tokens"]) >= self.config.group_size:
                break
        if any([score == 1.0 for score in scores["scores"]]):
            self.pass_at_groupsize.append(1.0)
        else:
            self.pass_at_groupsize.append(0.0)
        if len(scores["tokens"]) < self.config.group_size:
            # We don't have enough data to score
            return None
        for score in scores["scores"]:
            self.percent_correct_buffer.append(max(score, 0))
        self.percent_overanswer.extend(
            [item[2] == "length" for item in rollout_group_data]
        )
        # check if all the same
        # print(scores['scores'])
        # Fill in the correct/incorrect lenses after so we're only looking at actual training data
        self.correct_answer_len.extend(
            [
                len(scores["tokens"][i])
                for i in range(len(scores["scores"]))
                if scores["scores"][i] == 1.0
            ]
        )
        self.incorrect_answer_len.extend(
            [
                len(scores["tokens"][i])
                for i in range(len(scores["scores"]))
                if (scores["scores"][i] == -1.0)
                and (not scores["overrides"][i].get("set_advantage_to_zero", False))
            ]
        )
        return scores

    async def get_next_item(self):
        while True:
            next_item = self.train[self.iter % len(self.train)]
            self.iter += 1
            prompt = next_item["question"]
            try:
                answer = (
                    ("\\boxed{" + next_item["final_answer"] + "}")
                    if "\\boxed" not in next_item["final_answer"]
                    else next_item["final_answer"]
                )
                break
            except TypeError:
                print(
                    f"Error in getting next item, trying again, "
                    f"data: {next_item['question']} -> {next_item['final_answer']}"
                )
        return (prompt, answer, "normal")


if __name__ == "__main__":
    MathEnv.cli()
