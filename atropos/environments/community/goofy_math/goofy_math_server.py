import asyncio
import random
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import wandb
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item, number
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
)

system_prompt += """You are allocated a maximum of 2048 tokens, please strive to use less.

You will then provide your answer like this: \\boxed{your answer here}
It is important that you provide your answer in the correct format.
If you do not, you will not receive credit for your answer.
So please end your answer with \\boxed{your answer here}"""

# Define the goofiness preference string
goofiness_preference = (
    "be the GOOFIEST math solver ever! Use wild exaggerations, silly sound effects, "
    "dramatic reactions to calculations, personify numbers, and be totally over-the-top "
    "enthusiastic! Don't just solve the problem - make it a PERFORMANCE! Give your solution "
    "with maximum silliness - include dramatic gasps, unexpected tangents, and random sound effects. "
    "But still get the answer right, you absolute mathematical goofball! Your answers should "
    "feel like they're coming from an extremely enthusiastic but chaotic math genius."
)

# Judge system prompt
judge_system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
)
judge_system_prompt += """You are allocated a maximum of 8192 tokens, please strive to use less for thinking.

You will then provide your preferred response like this: \\boxed{A or B}.

It is important that you provide your answer in the correct format.
If you do not, you will not receive credit for your answer.

So please end your answer with \\boxed{A or B}"""

# Format string for the judge
GOOFY_JUDGE_prompt_format_str = """[Math Problem]
{problem}

[Response A]
{response_a}

[Response B]
{response_b}

[END]

Here's a math problem that has two correct solutions. BOTH solutions have the correct answer,
but I need you to pick the one that is GOOFIER and more FUN while still being mathematically sound.

A goofy math solution should have these qualities:
1. Silly expressions or sound effects
2. Dramatic reactions to calculations
3. Creative personification of numbers
4. Unexpected but relevant tangents
5. Exaggerated enthusiasm

Ignoring any differences in mathematical correctness (since both are correct),
which response is more ENTERTAINING and GOOFY?

Go ahead and think through it, then give me your answer with \\boxed{{A or B}}."""


class GSM8kRow(TypedDict):
    question: str
    answer: str


class GoofyMathEnv(BaseEnv):

    name = "goofy_math"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        # Add tracking for wandb visualizations
        self.rollouts_for_wandb = []
        self.completion_lengths = []
        self.judgement_strings = list()
        self.goofiness_scores = []

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="gpt2",  # Compatible with most models
            group_size=4,  # Generate 4 responses to compare
            use_wandb=True,  # Track experiments
            rollout_server_url="http://localhost:8000",
            total_steps=10,
            batch_size=8,  # Smaller batch for more frequent updates
            steps_per_eval=50,  # More frequent evaluation
            max_token_length=2048,
            wandb_name="goofy_math",
        )
        server_configs = [
            APIServerConfig(
                model_name="gpt-3.5-turbo",  # Use a widely available model
                server_type="openai",
                api_key=None,  # Will be provided at runtime
                num_requests_for_eval=64,
            ),
        ]

        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        # Try to calculate percent_correct, pass if there's a division by zero
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        # Add goofiness metrics
        try:
            if self.goofiness_scores:
                wandb_metrics["train/avg_goofiness_score"] = sum(
                    self.goofiness_scores
                ) / len(self.goofiness_scores)
                wandb_metrics["train/goofiness_histogram"] = wandb.Histogram(
                    self.goofiness_scores
                )
        except (ZeroDivisionError, Exception):
            pass

        # Log evaluation metrics
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()

        # Log judgment examples (similar to RLAIF)
        if len(self.judgement_strings) > 0:
            # setup wandb table
            table = wandb.Table(
                columns=["problem", "resp_a", "resp_b", "sample_judgement"]
            )
            for item in self.judgement_strings:
                table.add_data(item[0], item[1], item[2], item[3])
            self.judgement_strings.clear()
            wandb_metrics["train/judgement_table"] = table

        # Call the parent method to handle the server metrics
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        self.train = load_dataset("gsm8k", "main", split="train").shuffle(seed=42)
        test_data = load_dataset("gsm8k", "main", split="test").shuffle(seed=42)
        self.test = list()
        for item in test_data:
            self.test.append(
                {
                    "question": item["question"],
                    "gold_answer": item["answer"]
                    .split("#")[-1]
                    .strip()
                    .replace(",", ""),
                }
            )
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def rollout_and_score_eval(self, question: str, answer: str) -> number:
        completion = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )
        gold_parsed = parse(
            "\\boxed{" + answer + "}",
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        answer_parsed = parse(
            completion.choices[0].message.content.split("</think>")[-1],
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
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
        score = 1 if verify(answer_parsed, gold_parsed) else 0
        return score

    async def evaluate(self, *args, **kwargs):
        eval_tasks = []
        for item in self.test:
            eval_tasks.append(
                self.rollout_and_score_eval(item["question"], item["gold_answer"])
            )
        scores = await tqdm_asyncio.gather(*eval_tasks)
        self.eval_metrics.append(("eval/percent_correct", sum(scores) / len(scores)))

    async def collect_trajectories(
        self, item: GSM8kRow
    ) -> Tuple[ScoredDataGroup, list[Item]]:
        user_message = {"role": "user", "content": item["question"]}
        gold_answer = (
            "\\boxed{" + item["answer"].split("#")[-1].strip().replace(",", "") + "}"
        )

        # Similar to RLAIF, randomly add goofiness to system prompt
        added_goofy = random.random() < 0.5  # 50% chance of adding goofiness

        chat = []
        if added_goofy:
            # Add system prompt with goofiness instruction
            chat.append(
                {
                    "role": "system",
                    "content": system_prompt + "\n\n" + goofiness_preference,
                }
            )
        else:
            # Normal system prompt
            chat.append({"role": "system", "content": system_prompt})

        # Add user question
        chat.append(user_message)

        # Get responses
        chat_completions = await self.server.chat_completion(
            messages=chat,
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
        )

        to_score = list()
        to_backlog = list()

        for i, chat_completion in enumerate(chat_completions.choices):
            messages = (
                chat[0],  # System prompt (with or without goofiness)
                user_message,
                {"role": "assistant", "content": chat_completion.message.content},
            )
            to_score.append(
                {
                    "messages": messages,
                    "gold_answer": gold_answer,
                    "finish_reason": chat_completion.finish_reason,
                    "problem": item["question"],  # Store problem for judging
                }
            )

        to_postprocess = await self.score(to_score)
        return to_postprocess, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        # First, filter for mathematical correctness
        correct_solutions = []
        gold_parsed = parse(
            rollout_group_data[0]["gold_answer"],
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )

        if len(gold_parsed) == 0:
            # If the gold solution is not parseable, we return None
            return None

        # Check each solution for correctness
        for item in rollout_group_data:
            answer_parsed = parse(
                item["messages"][-1]["content"].split("</think>")[-1],
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
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
            # If correct, add to our list
            if verify(answer_parsed, gold_parsed):
                correct_solutions.append(item)

        # If we don't have at least 2 correct solutions, can't compare goofiness
        if len(correct_solutions) < 2:
            scores = ScoredDataGroup()
            scores["tokens"] = list()
            scores["masks"] = list()
            scores["scores"] = list()

            # Just score based on correctness (1.0 for correct, -1.0 for wrong)
            for item in rollout_group_data:
                answer_parsed = parse(
                    item["messages"][-1]["content"].split("</think>")[-1],
                    extraction_config=[LatexExtractionConfig()],
                    extraction_mode="first_match",
                )
                reward = 1.0 if verify(answer_parsed, gold_parsed) else -1.0

                out_dict = tokenize_for_trainer(
                    self.tokenizer, item["messages"], item["finish_reason"]
                )
                tokens = out_dict["tokens"]
                masks = out_dict["masks"]

                # remove obviously bad examples
                if len([1 for i in masks if i != -100]) < 10:
                    continue

                scores["tokens"].append(tokens)
                scores["masks"].append(masks)
                scores["scores"].append(reward)

            # Track correct solutions
            for score in scores["scores"]:
                self.percent_correct_buffer.append(max(score, 0))

            return scores

        # Now we have at least 2 correct solutions, judge goofiness
        # Randomly pair solutions for judging
        random.shuffle(correct_solutions)
        goofiness_scores = {}

        # Prepare to track all pair judgments
        judgments_to_make = []
        for i in range(0, len(correct_solutions), 2):
            if i + 1 < len(correct_solutions):
                judgments_to_make.append(
                    (correct_solutions[i], correct_solutions[i + 1])
                )

        # Prepare all judgment tasks
        judgment_tasks = []
        for sol_a, sol_b in judgments_to_make:
            # Forward format (A vs B)
            fwd_fmt = GOOFY_JUDGE_prompt_format_str.format(
                problem=sol_a["problem"],
                response_a=sol_a["messages"][-1]["content"],
                response_b=sol_b["messages"][-1]["content"],
            )

            # Reverse format (B vs A) to reduce position bias
            rvs_fmt = GOOFY_JUDGE_prompt_format_str.format(
                problem=sol_a["problem"],
                response_a=sol_b["messages"][-1]["content"],
                response_b=sol_a["messages"][-1]["content"],
            )

            # Create judging tasks
            fwd_judge = self.server.chat_completion(
                messages=[
                    {"role": "system", "content": judge_system_prompt},
                    {"role": "user", "content": fwd_fmt},
                ],
                n=1,
                max_tokens=self.config.max_token_length,
            )

            rvs_judge = self.server.chat_completion(
                messages=[
                    {"role": "system", "content": judge_system_prompt},
                    {"role": "user", "content": rvs_fmt},
                ],
                n=1,
                max_tokens=self.config.max_token_length,
            )

            judgment_tasks.append((fwd_judge, rvs_judge, sol_a, sol_b))

        # Execute all judgment tasks
        for fwd_judge_task, rvs_judge_task, sol_a, sol_b in judgment_tasks:
            fwd_judge, rvs_judge = await asyncio.gather(fwd_judge_task, rvs_judge_task)

            # Save example to wandb
            self.judgement_strings.append(
                (
                    sol_a["problem"],
                    sol_a["messages"][-1]["content"],
                    sol_b["messages"][-1]["content"],
                    fwd_judge.choices[0].message.content,
                )
            )

            # Calculate goofiness scores
            chosen_val_fwd = (
                fwd_judge.choices[0]
                .message.content.split("\\boxed{")[-1]
                .strip()
                .replace("}", "")
            )
            chosen_val_rvs = (
                rvs_judge.choices[0]
                .message.content.split("\\boxed{")[-1]
                .strip()
                .replace("}", "")
            )

            # Initial scores based on forward judgment
            if chosen_val_fwd == "A":
                goofiness_scores.setdefault(id(sol_a), 0)
                goofiness_scores[id(sol_a)] += 1
            elif chosen_val_fwd == "B":
                goofiness_scores.setdefault(id(sol_b), 0)
                goofiness_scores[id(sol_b)] += 1

            # Scores based on reverse judgment (swapped positions)
            if chosen_val_rvs == "A":
                goofiness_scores.setdefault(id(sol_b), 0)
                goofiness_scores[id(sol_b)] += 1
            elif chosen_val_rvs == "B":
                goofiness_scores.setdefault(id(sol_a), 0)
                goofiness_scores[id(sol_a)] += 1

        # Prepare the final scored data
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        # Process all correct solutions with their goofiness scores
        for solution in correct_solutions:
            out_dict = tokenize_for_trainer(
                self.tokenizer, solution["messages"], solution["finish_reason"]
            )
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Base score for correctness
            correct_score = 1.0

            # Add goofiness bonus (normalized to 0-1 range)
            goofiness_score = goofiness_scores.get(id(solution), 0)
            max_possible_goofiness = 2  # Maximum from 2 judgments (fwd+rvs)
            goofiness_bonus = goofiness_score / max_possible_goofiness

            # Track goofiness scores for analytics
            self.goofiness_scores.append(goofiness_bonus)

            # Combine scores: base correctness + weighted goofiness bonus
            final_score = correct_score + (
                goofiness_bonus * 0.5
            )  # Goofiness worth up to +0.5

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(final_score)

        # Track correctness in our buffer
        for _ in range(len(correct_solutions)):
            self.percent_correct_buffer.append(1.0)  # All are correct

        return scores

    async def get_next_item(self) -> GSM8kRow:
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item


if __name__ == "__main__":
    GoofyMathEnv.cli()
