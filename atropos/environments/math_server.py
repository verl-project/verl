import asyncio
import math
import random
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import wandb
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from math_verify.errors import TimeoutException
from pydantic import Field
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

problem_format = "{problem}"

judge_format = """Here is a math problem and a proposed solution:

[START PROBLEM]
{problem}
[END PROBLEM]
[START SOLUTION]
{solution}
[END SOLUTION]

Please verify if it is correct or not.

If it's correct submit your answer in your response with \\boxed{{True}}.
If it's incorrect, please submit your answer in your response with \\boxed{{False}}.

Please include how to solve the problem correctly in your answer."""


retry_format = """Here is a math problem, a proposed solution, and a verification of the solution:
[START PROBLEM]
{problem}
[END PROBLEM]
[START SOLUTION]
{solution}
[END SOLUTION]
[START VERIFICATION]
{verification}
[END VERIFICATION]

Please use this verification to help you solve the problem correctly.

Provide your answer in your response with \\boxed{{answer}}."""  # noqa: E501


rlaif_format = """Here is a math problem, and two solutions that are correct. Please choose whichever answer you prefer.
[START PROBLEM]
{problem}
[END PROBLEM]
[START SOLUTION 1]
{solution1}
[END SOLUTION 1]
[START SOLUTION 2]
{solution2}
[END SOLUTION 2]

Here are some metrics for you to use to grade the two solutions:
- Conciseness: How concise is the solution? Is it too long or too short?
- Clarity: How clear is the solution? Is it easy to understand?
- Correctness: Is the reasoning correct? The answer has been prechecked to be correct, but there may be errors in the reasoning.

Please use these metrics to help you choose the best solution, in order of priority.

Please provide your answer in your response with \\boxed{{1}}, for the first solution, or \\boxed{{2}} for the second solution."""  # noqa: E501


class RSConfig(BaseEnvConfig):
    run_evaluation: bool = Field(True, description="If this should run evaluation")
    mask_too_long_completions: bool = Field(
        True, description="If this should mask too long completions"
    )
    percent_to_judge: float = Field(0.3, description="The percentage of items to judge")
    percent_length_penalty: float = Field(
        0.0, description="The percentage of items to have length penalty"
    )


def quick_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def score_answer(gold, resp) -> Optional[bool]:
    try:
        gold_parsed = parse(
            gold,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
    except (Exception, TimeoutException, KeyError, TypeError, NotImplementedError):
        return None
    if len(gold_parsed) != 0:
        # print(item[0][-1]["content"])
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
        server_configs: List[APIServerConfig],
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
        self.percent_judge_correct = list()
        self.correct_answer_len = list()
        self.incorrect_answer_len = list()
        self.normal_rollouts = list()
        self.rlaif_rollouts = list()
        self.pass_at_groupsize = list()
        self.judge_rollouts = list()
        self.selfcorrect_rollouts = list()
        self.judge_success_rate = list()
        self.iter = 0

    @classmethod
    def config_init(self) -> Tuple[RSConfig, List[APIServerConfig]]:
        env_config = RSConfig(
            tokenizer_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
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
        server_configs = [
            APIServerConfig(
                model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_requests_for_eval=256,  # since evaling only on one...
            ),
        ]

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
        if len(self.percent_judge_correct) > 0:
            wandb_metrics["judge_train/percent_judge_correct"] = sum(
                self.percent_judge_correct
            ) / len(self.percent_judge_correct)
            self.percent_judge_correct = list()
        if len(self.judge_success_rate) > 0:
            wandb_metrics["judge_train/judge_success_rate"] = sum(
                self.judge_success_rate
            ) / len(self.judge_success_rate)
        # create tables
        if len(self.judge_rollouts) > 0:
            table = wandb.Table(
                columns=["problem", "solution", "answer", "correct", "judge"]
            )
            for group in self.judge_rollouts:
                table.add_data(group[0], group[1], group[2], group[3], group[4])
            wandb_metrics["judge_train/judge_rollouts"] = table
        if len(self.selfcorrect_rollouts) > 0:
            table = wandb.Table(columns=["problem", "solution1", "solution2", "score"])
            for group in self.selfcorrect_rollouts:
                table.add_data(group[0], group[1], group[2], group[3])
            wandb_metrics["judge_train/selfcorrect_rollouts"] = table
        if len(self.normal_rollouts) > 0:
            table = wandb.Table(columns=["problem", "solution", "answer", "score"])
            for group in self.normal_rollouts:
                table.add_data(group[0], group[1], group[2], group[3])
            wandb_metrics["train/normal_rollouts"] = table
        if len(self.rlaif_rollouts) > 0:
            table = wandb.Table(
                columns=["problem", "solution1", "solution2", "score", "rollout"]
            )
            for group in self.rlaif_rollouts:
                table.add_data(group[0], group[1], group[2], group[3], group[4])
            wandb_metrics["train/rlaif_rollouts"] = table
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
        self.test = list()
        for name, t_dataset in zip(
            ["aime24", "math500"], [aime_test_data, math500_test_data]
        ):
            for item in t_dataset:
                self.test.append(
                    (
                        problem_format.format(problem=item["problem"]),
                        item["answer"],
                        name,
                    )
                )
        for name, t_dataset in zip(
            ["amc23"],
            [amc_test_data],
        ):
            for item in t_dataset:
                self.test.append(
                    (
                        problem_format.format(problem=item["question"]),
                        item["answer"],
                        name,
                    )
                )

    async def rollout_and_score_eval(self, question, answer, subset):

        completion = await self.server.chat_completion(
            messages=[
                {"role": "user", "content": question},
            ],
            n=1,
            max_tokens=32765,
            temperature=0.0,
            split="eval",
        )
        loop = asyncio.get_event_loop()
        gold = "\\boxed{" + answer + "}" if "\\boxed" not in answer else answer
        resp = completion.choices[0].message.content.split("</think>")[-1]
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

    async def collect_trajectories_normal(self, item) -> Tuple[List, List]:
        thinking_len = self.config.max_token_length
        user_prompt = problem_format.format(problem=item[0])
        chat = [
            {"role": "user", "content": user_prompt},
        ]
        thinking_len = thinking_len - len(
            self.tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        )
        chat_completions = await self.server.chat_completion(
            messages=chat,
            n=self.config.group_size,
            max_tokens=thinking_len,
            temperature=1.0,
            top_p=0.95,
        )
        print("Finished generation", flush=True)
        to_score = list()
        to_backlog = list()
        for i, chat_completion in enumerate(chat_completions.choices):
            messages = (
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": chat_completion.message.content},
            )
            to_score.append(
                (
                    messages,
                    item[1],
                    chat_completion.finish_reason,
                )
            )

        to_postprocess = await self.score_normal(to_score)
        if to_postprocess is None:
            return None, to_backlog
        if all(
            [to_postprocess["scores"][0] == score for score in to_postprocess["scores"]]
        ):
            if to_postprocess["scores"][0] == 1.0:
                # we can do RLAIF
                # find the two most dissimilar messages
                messages = to_postprocess["messages"]
                score_matrix = []
                most_dissimilar = (0, 1)
                most_dissimilar_score = 1.0
                # find the two most dissimilar messages
                for i in range(len(messages) - 1):
                    score_matrix.append([])
                    for j in range(i + 1):
                        # Only need to compute half of the matrix
                        score_matrix[i].append(1.0)
                    for j in range(i + 1, len(messages)):
                        m1 = messages[i][-1]["content"].split("</think>")[-1]
                        m2 = messages[j][-1]["content"].split("</think>")[-1]
                        if m1 == m2:
                            score_matrix[i].append(1.0)
                        else:
                            score_matrix[i].append(quick_similarity(m1, m2))
                        if score_matrix[i][j] < most_dissimilar_score:
                            most_dissimilar = (i, j)
                            most_dissimilar_score = score_matrix[i][j]
                if most_dissimilar_score < 0.975:
                    # send over to RLAIF
                    to_backlog.append(
                        (
                            item[0],
                            item[1],
                            "rlaif",
                            tuple(
                                [
                                    frozenset(item.items())
                                    for item in messages[most_dissimilar[0]]
                                ]
                            ),
                            tuple(
                                [
                                    frozenset(item.items())
                                    for item in messages[most_dissimilar[1]]
                                ]
                            ),
                            most_dissimilar_score,
                        )
                    )
                    print(
                        "\n".join(
                            [
                                "["
                                + ", ".join([str(item) for item in score_matrix_row])
                                + "]"
                                for score_matrix_row in score_matrix
                            ]
                        )
                    )
                    print(
                        f"Sending to RLAIF, most dissimilar score: {most_dissimilar_score}"
                    )
                else:
                    print(
                        f"Unable to RLAIF, most dissimilar score: {most_dissimilar_score}"
                    )
                if random.random() < self.config.percent_length_penalty:
                    # Check if deltas of message lengths are different enough to want to length penalty on
                    message_lengths = [
                        len(tokens) for tokens in to_postprocess["tokens"]
                    ]
                    min_message_length = min(message_lengths)
                    max_message_delta = max(
                        [msg_len - min_message_length for msg_len in message_lengths]
                    )
                    if max_message_delta > 0.1 * min_message_length:
                        print(
                            "Max message delta is greater than 0.1 * shortest message, adding length penalty"
                        )
                        for i in range(len(to_postprocess["scores"])):
                            len_penalty = (
                                message_lengths[i] - min_message_length
                            ) / max_message_delta
                            len_penalty = math.cos(len_penalty * math.pi)
                            to_postprocess["scores"][i] = len_penalty
                    else:
                        print(
                            "Max message delta is less than 0.1 * shortest message, no length penalty"
                        )
                        return None, to_backlog
                else:
                    return None, to_backlog
            else:
                return None, to_backlog
        else:
            self.normal_rollouts.append(
                (
                    item[0],
                    to_postprocess["messages"][0],
                    item[1],
                    to_postprocess["scores"][0],
                )
            )
            print("Sending to judge potentially")
            if random.random() < self.config.percent_to_judge:
                # find first pos and neg scored answers.
                pos_idx = [
                    i
                    for i, score in enumerate(to_postprocess["scores"])
                    if score == 1.0
                ][0]
                neg_idx = [
                    i
                    for i, score in enumerate(to_postprocess["scores"])
                    if (score == -1.0)
                    and (
                        not to_postprocess["overrides"][i].get(
                            "set_advantage_to_zero", False
                        )
                    )
                ]
                if len(neg_idx) == 0:
                    return None, to_backlog
                neg_idx = neg_idx[0]
                if pos_idx is not None and neg_idx is not None:
                    to_backlog.append(
                        (
                            item[0],
                            item[1],
                            "judge",
                            to_postprocess["messages"][pos_idx][-1]["content"].split(
                                "</think>"
                            )[-1],
                            "True",
                        )
                    )
                    to_backlog.append(
                        (
                            item[0],
                            item[1],
                            "judge",
                            to_postprocess["messages"][neg_idx][-1]["content"].split(
                                "</think>"
                            )[-1],
                            "False",
                        )
                    )
                    print("sending to judge")
                else:
                    return None, to_backlog
        print(f"Collected {len(to_postprocess['scores'])} trajectories")
        if not self.config.mask_too_long_completions:
            to_postprocess["overrides"] = [
                {} for _ in range(len(to_postprocess["scores"]))
            ]
        return to_postprocess, to_backlog

    async def collect_trajectories(self, item) -> Tuple[List, List]:
        if item[2] == "normal":
            return await self.collect_trajectories_normal(item)
        elif item[2] == "rlaif":
            return await self.collect_trajectories_rlaif(item)
        elif item[2] == "judge":
            return await self.collect_trajectories_judge(item)
        elif item[2] == "selfcorrect":
            # selfcorrect is a special case where we are using the Judge rollout
            print("selfcorrect processing...")
            print("selfcorrect item:", item, flush=True)
            group = item[3]
            scores = item[4]
            finish_reasons = item[5]
            to_postprocess = ScoredDataGroup()
            to_postprocess["tokens"] = list()
            to_postprocess["masks"] = list()
            to_postprocess["scores"] = list()
            to_postprocess["overrides"] = list()
            to_postprocess["messages"] = list()
            for i in range(len(group)):
                # convert from frozen set to dict
                conv = [dict(x) for x in group[i]]
                if i == 0:
                    self.selfcorrect_rollouts.append(
                        (
                            item[0],
                            item[1],
                            conv[0]["content"],
                            conv[1]["content"],
                        )
                    )
                    if (
                        len(self.selfcorrect_rollouts)
                        >= self.config.num_rollouts_to_keep
                    ):
                        self.selfcorrect_rollouts.pop(0)
                out_dict = tokenize_for_trainer(
                    tokenizer=self.tokenizer,
                    chat=conv,
                    finish_reason=finish_reasons[i],
                    include_messages=True,
                )
                to_postprocess["tokens"].append(out_dict["tokens"])
                to_postprocess["masks"].append(out_dict["masks"])
                to_postprocess["scores"].append(scores[i])
                to_postprocess["overrides"].append(dict())
                if (finish_reasons[i] == "length") and (
                    self.config.mask_too_long_completions
                ):
                    to_postprocess["overrides"][-1]["set_advantage_to_zero"] = True
                to_postprocess["messages"].append(out_dict["messages"])
            print("selfcorrect done, sending batch off")
            return to_postprocess, []
        else:
            raise ValueError(f"Unknown rollout type: {item[2]}")

    async def score_normal(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
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
            resp = item[0][-1]["content"].split("</think>")[-1]
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
            out_dict = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=item[0],
                finish_reason=item[2],
                include_messages=True,
            )
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]
            messages = out_dict["messages"]
            # remove obviously bad examples
            if len([1 for i in masks if i != -100]) < 10:
                continue
            if item[2] == "length":
                # Note we set it here so we can filter out the examples that are too long
                # for the Judge loop. IF you set the config to not do this we fix it
                # in the collect_trajectories_normal function.
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

    async def collect_trajectories_rlaif(self, frozen_item) -> Tuple[List, List]:
        to_backlog = list()
        print("Attempting RLAIF")
        item = list(frozen_item)
        print("Converting to dicts")
        item[3] = [dict(x) for x in item[3]]
        item[4] = [dict(x) for x in item[4]]
        print("Formatting user prompts")
        user_prompt_fwd = rlaif_format.format(
            problem=item[0],
            solution1=item[3][-1]["content"].split("</think>")[-1],
            solution2=item[4][-1]["content"].split("</think>")[-1],
        )
        user_prompt_bwd = rlaif_format.format(
            problem=item[0],
            solution1=item[4][-1]["content"].split("</think>")[-1],
            solution2=item[3][-1]["content"].split("</think>")[-1],
        )
        print("Sending to server")
        chat = [
            {"role": "user", "content": user_prompt_fwd},
        ]
        max_token_length = self.config.max_token_length - len(
            self.tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        )
        chat_completions_fwd = self.server.chat_completion(
            messages=chat,
            n=3,
            max_tokens=max_token_length,
            temperature=1.0,
            top_p=0.95,
        )
        print("Sending to server")
        # Should be the same token length as the fwd but tokenizers are cursed
        chat = [
            {"role": "user", "content": user_prompt_bwd},
        ]
        max_token_length = self.config.max_token_length - len(
            self.tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        )
        chat_completions_bwd = self.server.chat_completion(
            messages=chat,
            n=3,
            max_tokens=self.config.max_token_length,
            temperature=1.0,
            top_p=0.95,
        )
        print("Gathering completions")
        chat_completions_fwd, chat_completions_bwd = await asyncio.gather(
            chat_completions_fwd, chat_completions_bwd
        )
        print("Grabbed RLAIF completions")
        # Check for correct answers
        score_1 = 0
        score_2 = 0
        for chat_completion in chat_completions_fwd.choices:
            score = (
                chat_completion.message.content.split("</think>")[-1]
                .split("\\boxed{")[-1]
                .split("}")[0]
                .strip()
            )
            if score == "1":
                score_1 += 1
            elif score == "2":
                score_2 += 1
        for chat_completion in chat_completions_bwd.choices:
            score = (
                chat_completion.message.content.split("</think>")[-1]
                .split("\\boxed{")[-1]
                .split("}")[0]
                .strip()
            )
            if score == "1":
                score_2 += 1
            elif score == "2":
                score_1 += 1
        print(f"Score 1: {score_1}, Score 2: {score_2}")
        if score_1 == score_2:
            return None, []
        self.rlaif_rollouts.append(
            (
                item[0],
                item[3][-1]["content"].split("</think>")[-1],
                item[4][-1]["content"].split("</think>")[-1],
                score_1 - score_2,
                chat_completions_fwd.choices[0].message.content,
            )
        )
        if len(self.rlaif_rollouts) >= self.config.num_rollouts_to_keep:
            self.rlaif_rollouts.pop(0)
        print("RLAIF rollout added")
        to_postprocess = ScoredDataGroup()
        to_postprocess["tokens"] = list()
        to_postprocess["masks"] = list()
        to_postprocess["scores"] = list()
        to_postprocess["overrides"] = list()
        to_postprocess["messages"] = list()
        # add the first message in
        out_dict = tokenize_for_trainer(
            tokenizer=self.tokenizer, chat=item[3], include_messages=True
        )
        tokens = out_dict["tokens"]
        masks = out_dict["masks"]
        to_postprocess["tokens"].append(tokens)
        to_postprocess["masks"].append(masks)
        to_postprocess["scores"].append(1.0 if score_1 > score_2 else -1.0)
        to_postprocess["messages"].append(out_dict["messages"])
        out_dict = tokenize_for_trainer(
            tokenizer=self.tokenizer, chat=item[4], include_messages=True
        )
        tokens = out_dict["tokens"]
        masks = out_dict["masks"]
        to_postprocess["tokens"].append(tokens)
        to_postprocess["masks"].append(masks)
        to_postprocess["scores"].append(1.0 if score_2 > score_1 else -1.0)
        to_postprocess["messages"].append(out_dict["messages"])
        to_postprocess["group_overrides"] = {
            "group_size": 2,
        }
        print("RLAIF rollout added")
        return to_postprocess, to_backlog

    async def collect_trajectories_judge(self, item) -> Tuple[List, List]:
        user_prompt = judge_format.format(
            problem=item[0],
            solution=item[3],
        )
        to_backlog = list()
        chat = [
            {"role": "user", "content": user_prompt},
        ]
        max_token_length = self.config.max_token_length - len(
            self.tokenizer.apply_chat_template(chat, add_generation_prompt=True)
        )
        chat_completions = await self.server.chat_completion(
            messages=chat,
            n=self.config.group_size,
            max_tokens=max_token_length,
            temperature=1.0,
            top_p=0.95,
        )
        is_correct = [
            (
                chat_completion.message.content.split("</think>")[-1]
                .split("\\boxed{")[-1]
                .split("}")[0]
                .strip()
                == item[4]
            )
            and (chat_completion.finish_reason != "length")
            for chat_completion in chat_completions.choices
        ]
        self.percent_judge_correct.append(
            sum([1.0 if val else 0.0 for val in is_correct]) / len(is_correct)
        )
        if all([not val for val in is_correct]):
            # Can't judge :(
            return None, []
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        scores["overrides"] = []
        scores["messages"] = []
        for_table = []
        for i, chat_completion in enumerate(chat_completions.choices):
            out_dict = tokenize_for_trainer(
                tokenizer=self.tokenizer,
                chat=[
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": chat_completion.message.content},
                ],
                include_messages=True,
            )
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]
            messages = out_dict["messages"]
            if not is_correct[i]:
                scores["tokens"].append(tokens)
                scores["masks"].append(masks)
                scores["scores"].append(-1.0)
                scores["messages"].append(messages)
                scores["overrides"].append(dict())
                if (chat_completion.finish_reason == "length") and (
                    self.config.mask_too_long_completions
                ):
                    scores["overrides"][-1]["set_advantage_to_zero"] = True
            else:
                if len(for_table) == 0:
                    # populate the table
                    for_table = [
                        item[0],
                        item[1],
                        item[3],
                        item[4],
                        chat_completion.message.content,
                    ]
                if item[4] == "False":
                    # Score based on percentage correct from retry
                    print("Scoring retry")
                    retry_prompt = retry_format.format(
                        problem=item[0],
                        solution=item[3],
                        verification=chat_completion.message.content.split("</think>")[
                            -1
                        ],
                    )
                    print("Sending to server")
                    retry_messages = [
                        {"role": "user", "content": retry_prompt},
                    ]
                    max_token_length = self.config.max_token_length - len(
                        self.tokenizer.apply_chat_template(
                            retry_messages, add_generation_prompt=True
                        )
                    )
                    retry_chat_completions = await self.server.chat_completion(
                        messages=retry_messages,
                        n=self.config.group_size,
                        max_tokens=max_token_length,
                        temperature=1.0,
                        top_p=0.95,
                    )
                    print("Gathering completions")
                    scoring_data = []
                    backlog_scores = []
                    backlog_reasons = []
                    backlog_messages = []
                    for j, retry_chat_completion in enumerate(
                        retry_chat_completions.choices
                    ):
                        print(f"Scoring generation {j} for retry...")
                        backlog_messages.append(
                            tuple(
                                [frozenset(msg.items()) for msg in retry_messages]
                                + [
                                    frozenset(
                                        {
                                            "role": "assistant",
                                            "content": retry_chat_completion.message.content,
                                        }.items()
                                    )
                                ]
                            )
                        )
                        backlog_reasons.append(retry_chat_completion.finish_reason)
                        if retry_chat_completion.finish_reason == "length":
                            scoring_data.append(0)
                            backlog_scores.append(0)
                        else:
                            loop = asyncio.get_event_loop()
                            task = loop.run_in_executor(
                                self.mp_executor,
                                score_answer,
                                item[1],
                                retry_chat_completion.message.content.split("</think>")[
                                    -1
                                ],
                            )
                            reward = await task
                            scoring_data.append(1.0 if reward else 0.0)
                            backlog_scores.append(1.0 if reward else -1.0)

                    if (
                        not all(
                            backlog_score == backlog_scores[0]
                            for backlog_score in backlog_scores
                        )
                    ) or (
                        all(
                            backlog_reasons == 1.0 for backlog_reason in backlog_reasons
                        )
                        and (random.random() < self.config.percent_length_penalty)
                    ):
                        to_backlog.append(
                            (
                                item[0],
                                item[1],
                                "selfcorrect",
                                tuple(backlog_messages),
                                tuple(backlog_scores),
                                tuple(backlog_reasons),
                            )
                        )
                        print(f"Sending to selfcorrect, {len(to_backlog)} in backlog")
                    scores["scores"].append(sum(scoring_data) / len(scoring_data))
                    self.judge_success_rate.append(
                        sum(scoring_data) / len(scoring_data)
                    )
                    if len(self.judge_success_rate) >= self.config.num_rollouts_to_keep:
                        self.judge_success_rate.pop(0)
                else:
                    scores["scores"].append(1.0)
                    scores["tokens"].append(tokens)
                    scores["masks"].append(masks)
                    scores["messages"].append(messages)
                    scores["overrides"].append(dict())
        if all([score == 1.0 for score in scores["scores"]]) and (
            random.random() < self.config.percent_length_penalty
        ):
            # Do len penalty
            message_lengths = [len(tokens) for tokens in scores["tokens"]]
            min_message_length = min(message_lengths)
            max_message_delta = max(
                [msg_len - min_message_length for msg_len in message_lengths]
            )
            if max_message_delta > 0.1 * min_message_length:
                print(
                    "Max message delta is greater than 0.1 * shortest message, adding length penalty"
                )
                for i in range(len(scores["scores"])):
                    len_penalty = (
                        message_lengths[i] - min_message_length
                    ) / max_message_delta
                    len_penalty = math.cos(len_penalty * math.pi)
                    scores["scores"][i] = len_penalty
            else:
                print(
                    "Max message delta is less than 0.1 * shortest message, no length penalty"
                )
                return None, []
        elif all([score == scores["scores"][0] for score in scores["scores"]]):
            return None, []
        if len(for_table) > 0:
            self.judge_rollouts.append(for_table)
            if len(self.judge_rollouts) >= self.config.num_rollouts_to_keep:
                self.judge_rollouts.pop(0)
        print(
            f"Collected {len(scores['scores'])} trajectories with {len(to_backlog)} in backlog"
        )
        return scores, to_backlog


if __name__ == "__main__":
    MathEnv.cli()
