import asyncio
import json
import logging
import os
import random
import re
from typing import Dict, List, Optional, Tuple, Union

from dotenv import load_dotenv
from openai import AsyncOpenAI

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

from .curriculum import MathCurriculum

load_dotenv()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

system_prompt = (
    "You are an expert mathematician that can use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering.\n"
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your final answer in a LaTeX format using \\boxed{your answer here}.\n\n"
    "The problems will be given in a LaTeX format, so be sure to follow the LaTeX "
    "syntax when writing your answer (although no $ delimiters are necessary).\n\n"
    "Follow these steps:\n"
    "1. Understand the problem carefully\n"
    "2. Plan your approach\n"
    "3. Execute the calculations step-by-step\n"
    "4. Verify your solution\n"
    "5. Express the final answer as \\boxed{your answer here}\n\n"
    "You may use extremely long chains of thought to deeply consider the problem "
    "and deliberate with yourself via systematic reasoning processes to help come "
    "to a correct solution prior to answering.\n\n"
    "Your answer format should be:\n"
    "<think>\n"
    "[Your detailed step-by-step reasoning process here]\n"
    "</think>\n\n"
    "\\boxed{your final answer here}\n\n"
    "Remember to format your final answer correctly as this is important for evaluation. "
    "Do not apply any rounding to your final answer, be as exact as possible."
)


class InfiniteMathEnvConfig(BaseEnvConfig):
    """Configuration for the InfiniteMath environment."""

    starting_level: int = 1
    progress_threshold: float = 0.8
    min_evaluations: int = 5

    max_attempts_per_problem: int = 3
    correct_reward: float = 1.0
    incorrect_reward: float = -1.0
    think_block_bonus: float = 0.2
    boxed_answer_bonus: float = 0.2

    apply_length_penalty: bool = True
    length_threshold_ratio: float = 0.5

    temperature: float = 0.7
    top_p: float = 0.9

    word_problem_model_name: Optional[str] = "gpt-4.1-mini"
    word_problem_openai_api_key: Optional[str] = os.environ.get("OPENAI_API_KEY")
    word_problem_openai_base_url: Optional[str] = None


class InfiniteMathEnv(BaseEnv):
    """Environment for procedurally generated math problems with curriculum advancement."""

    def __init__(
        self,
        config: InfiniteMathEnvConfig,
        server_configs: Union[List[APIServerConfig], APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.config = config

        self.percent_correct_buffer = []
        self.level_correct_buffer = {i: [] for i in range(1, 8)}
        self.eval_metrics = []

        self.curriculum = None

        self.system_prompt = system_prompt

    async def setup(self):
        """Initialize the environment and curriculum."""
        logger.info("Setting up InfiniteMathEnv")

        self.curriculum = MathCurriculum(
            starting_level=self.config.starting_level,
            progress_threshold=self.config.progress_threshold,
            min_evaluations=self.config.min_evaluations,
        )

        self.eval_problems = {}
        for level in range(1, 8):
            self.eval_problems[level] = []
            temp_curriculum = MathCurriculum(starting_level=level)
            attempts = 0
            max_attempts_per_level = 20

            while (
                len(self.eval_problems[level]) < 10
                and attempts < max_attempts_per_level
            ):
                try:
                    problem, solution, generator_id = temp_curriculum.get_problem()
                    problem = self._strip_latex_delimiters(problem)
                    solution = self._strip_latex_delimiters(solution)
                    self.eval_problems[level].append((problem, solution, generator_id))
                except Exception as e:
                    logger.warning(
                        f"Error generating evaluation problem for level {level}: {e}"
                    )
                attempts += 1

            logger.info(
                f"Generated {len(self.eval_problems[level])} evaluation problems for level {level}"
            )

        for level in range(1, 8):
            if not self.eval_problems[level]:
                logger.warning(
                    f"No valid evaluation problems for level {level}, adding fallback"
                )
                if level == 1:
                    self.eval_problems[level].append(("What is 2 + 3?", "5", 0))
                elif level == 2:
                    self.eval_problems[level].append(
                        ("What is the square root of 16?", "4", 6)
                    )
                elif level == 3:
                    self.eval_problems[level].append(
                        (
                            "What is the area of a triangle with base 6 and height 8?",
                            "24",
                            18,
                        )
                    )
                elif level == 4:
                    self.eval_problems[level].append(
                        ("What is the solution to x + 5 = 12?", "7", 26)
                    )
                elif level == 5:
                    self.eval_problems[level].append(
                        ("What is the volume of a cube with side length 3?", "27", 33)
                    )
                elif level == 6:
                    self.eval_problems[level].append(
                        ("What is 5 factorial?", "120", 31)
                    )
                else:
                    self.eval_problems[level].append(("What is |3 - 10|?", "7", 71))

    def _strip_latex_delimiters(self, text: str) -> str:
        """Strip LaTeX delimiters ($...$) from text."""
        return re.sub(r"\$(.*?)\$", r"\1", text)

    def save_checkpoint(self, step, data=None):
        """Save curriculum state in checkpoint."""
        if data is None:
            data = {}

        data["curriculum_level"] = self.curriculum.get_current_level()
        data["performance_history"] = {
            str(k): v for k, v in self.curriculum.performance_history.items()
        }

        super().save_checkpoint(step, data)

    def load_checkpoint(self):
        """Load curriculum state from checkpoint."""
        super().load_checkpoint()

        checkpoint_path = f"{self.checkpoint_dir}/env_checkpoints/{self.wandb_prepend}/step-{self.curr_step}.json"
        try:
            with open(checkpoint_path, "r") as f:
                data = json.load(f)

            if "curriculum_level" in data:
                level = data["curriculum_level"]
                self.curriculum.current_level = level

            if "performance_history" in data:
                self.curriculum.performance_history = {
                    int(k): v for k, v in data["performance_history"].items()
                }
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to wandb."""
        if wandb_metrics is None:
            wandb_metrics = {}

        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / max(1, len(self.percent_correct_buffer))
        except ZeroDivisionError:
            pass

        for level, buffer in self.level_correct_buffer.items():
            if buffer:
                wandb_metrics[f"train/level_{level}_correct"] = sum(buffer) / len(
                    buffer
                )
                wandb_metrics[f"train/level_{level}_count"] = len(buffer)

        if self.curriculum:
            current_level = self.curriculum.get_current_level()
            max_level = max(self.curriculum.DIFFICULTY_LEVELS.keys())

            wandb_metrics["curriculum/current_level"] = current_level
            wandb_metrics["curriculum/max_level"] = max_level
            wandb_metrics["curriculum/progress_percent"] = (
                current_level / max_level
            ) * 100

            wandb_metrics["curriculum/level_description"] = (
                self.curriculum.get_level_description()
            )

            if current_level in self.curriculum.performance_history:
                history = self.curriculum.performance_history[current_level]
                if history:
                    recent_history = history[
                        -min(len(history), self.curriculum.min_evaluations) :
                    ]
                    if recent_history:
                        success_rate = sum(recent_history) / len(recent_history)
                        wandb_metrics["curriculum/current_level_success_rate"] = (
                            success_rate
                        )
                        wandb_metrics["curriculum/threshold_to_advance"] = (
                            self.curriculum.progress_threshold
                        )
                        wandb_metrics["curriculum/remaining_to_threshold"] = max(
                            0, self.curriculum.progress_threshold - success_rate
                        )

        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]

        self.percent_correct_buffer = []
        for level in self.level_correct_buffer:
            self.level_correct_buffer[level] = []
        self.eval_metrics = []

        await super().wandb_log(wandb_metrics)

    async def _convert_to_word_problem(self, raw_problem_text: str) -> str:
        """Converts a raw math problem string into a word problem using an LLM."""
        system_prompt_word_problem = (
            "You are an expert creative writer. Your task is to transform a given raw "
            "mathematical expression into an engaging and imaginative word problem.\n\n"
            "**Critical Instructions:**\n"
            "1.  **Strict Preservation:** The core mathematical question, ALL numbers, and ALL operations "
            "from the raw problem MUST be EXACTLY preserved in the word problem. Do NOT change the calculation "
            "required. For example, if the raw problem is 'A - B', the word problem must represent subtraction "
            "of B from A, not any other operation.\n"
            "2.  **Clarity:** The word problem must clearly and unambiguously lead to solving the original "
            "mathematical expression.\n"
            "3.  **Conciseness:** Keep the word problem relatively short and to the point.\n"
            "4.  **Output Format:** Output ONLY the word problem text. Do NOT include any preambles, "
            "self-references (like 'Here is a word problem:'), special tokens (like '<|start_header_id|>'), "
            "or any text other than the word problem itself.\n\n"
            "**Examples of Correct Transformation:**\n"
            "Raw Problem: 5 * 3\n"
            "Word Problem: Sarah is baking cookies, and each batch requires 3 eggs. If Sarah wants to bake 5 batches, "
            "how many eggs will she need in total?\n\n"
            "Raw Problem: |10 - 15|\n"
            "Word Problem: A submarine is 10 meters below sea level. Another submarine is 15 meters below sea level. "
            "What is the absolute difference in their depths in meters?\n\n"
            "Raw Problem: sqrt(16)\n"
            "Word Problem: A square piece of land has an area of 16 square units. What is the length of one of its "
            "sides in units?\n\n"
            "**Example of Incorrect Transformation (Operation Changed):**\n"
            "Raw Problem: |3 - (-67)|  (This is 3 + 67)\n"
            "Incorrect Word Problem: In a magical forest, there are 3 enchanted trees,"
            " and each tree has 67 glowing fruits. "
            "How many glowing fruits are there in total? (This became 3 * 67)\n"
            "Correct Word Problem: A bird watcher is 3 meters up a tree. "
            "She spots a rare bird 67 meters below ground level "
            "in a cave. What is the total vertical distance between the bird watcher and the rare bird in meters?"
        )

        messages = [
            {"role": "system", "content": system_prompt_word_problem},
            {"role": "user", "content": f"Raw Problem: {raw_problem_text}"},
        ]

        try:
            api_key_to_use = self.config.word_problem_openai_api_key or os.environ.get(
                "OPENAI_API_KEY"
            )
            base_url_to_use = self.config.word_problem_openai_base_url
            model_to_use = self.config.word_problem_model_name or "gpt-4.1-mini"

            if not api_key_to_use:
                logger.error(
                    "OpenAI API key for word problem generation is not configured "
                    "(checked config and OPENAI_API_KEY env var)."
                )
                return raw_problem_text

            client = AsyncOpenAI(
                api_key=api_key_to_use,
                base_url=base_url_to_use,
            )

            chat_completions = await client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                n=1,
                max_tokens=512,
                temperature=0.7,
                top_p=1.0,
            )

            generated_text = chat_completions.choices[0].message.content

            original_llm_output = generated_text

            cleaned_text = original_llm_output.strip()

            if not cleaned_text:
                logger.warning(
                    f"Word problem conversion for '{raw_problem_text}' resulted in empty string after stripping. "
                    f"Original LLM output was: '{original_llm_output}'. Falling back to raw problem."
                )
                return raw_problem_text

            logger.info(
                f"Converted raw problem '{raw_problem_text}' to word problem: '{cleaned_text}'"
            )
            return cleaned_text
        except Exception as e:
            log_message_error = (
                f"Error converting to word problem for '{raw_problem_text}': {e}"
            )
            logger.error(log_message_error)
            return raw_problem_text

    async def get_next_item(self):
        """Get the next problem based on current curriculum level."""
        raw_problem, solution, generator_id = self.curriculum.get_problem()

        raw_problem_stripped = self._strip_latex_delimiters(raw_problem)
        solution_stripped = self._strip_latex_delimiters(solution)

        word_problem_text = await self._convert_to_word_problem(raw_problem_stripped)

        prompt = tuple(
            [frozenset({"role": "user", "content": word_problem_text}.items())]
        )

        return (prompt, solution_stripped, generator_id)

    async def evaluate(self, *args, **kwargs):
        """Evaluate the model on test problems at the current curriculum level."""
        current_level = self.curriculum.get_current_level()
        logger.info(f"Starting evaluation for curriculum level {current_level}")

        eval_tasks = []
        eval_generator_ids = []
        if current_level in self.eval_problems:
            for problem, solution, generator_id in self.eval_problems[current_level]:
                eval_tasks.append(
                    self.evaluate_single_problem(problem, solution, current_level)
                )
                eval_generator_ids.append(generator_id)

        if not eval_tasks:
            logger.warning(
                f"No evaluation problems available for level {current_level}"
            )
            return []

        logger.info(f"Evaluating {len(eval_tasks)} problems at level {current_level}")
        results = await asyncio.gather(*eval_tasks)

        correct_count = sum(1 for _, is_correct in results if is_correct)
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0

        logger.info(
            f"Level {current_level} accuracy: {accuracy:.2f} ({correct_count}/{total_count})"
        )

        self.eval_metrics.append((f"eval/level_{current_level}_accuracy", accuracy))
        self.eval_metrics.append(("eval/current_level", current_level))

        for i, (_, is_correct) in enumerate(results):
            if i < len(eval_generator_ids):
                self.curriculum.record_performance(eval_generator_ids[i], is_correct)
            else:
                sample_generator_id = random.choice(
                    self.curriculum.DIFFICULTY_LEVELS[current_level]
                )
                self.curriculum.record_performance(sample_generator_id, is_correct)

        advanced = self.curriculum.advance_difficulty()
        new_level = self.curriculum.get_current_level()

        if advanced:
            logger.info(f"Advanced from level {current_level} to level {new_level}!")
            self.eval_metrics.append(("eval/advanced_level", 1))
        else:
            logger.info(f"Remaining at level {current_level}")
            self.eval_metrics.append(("eval/advanced_level", 0))

        return self.eval_metrics

    async def evaluate_single_problem(
        self, problem: str, solution: str, level: int
    ) -> Tuple[int, bool]:
        """Evaluate a single problem."""
        try:
            word_problem_text = await self._convert_to_word_problem(problem)
            logger.debug(
                f"Evaluating level {level} word problem: {word_problem_text[:50]}... "
                f"(Original raw: {problem[:30]}...)"
            )

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": word_problem_text},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

            prefill = "\n<think>\n"
            prefilled_prompt = prompt + prefill

            logger.debug(f"Requesting completion for problem: {problem[:30]}...")
            completion = await self.server.completion(
                prompt=prefilled_prompt,
                n=1,
                max_tokens=self.config.max_token_length,
                temperature=0.0,
                top_p=1.0,
                split="eval",
            )

            model_answer = prefill + (
                completion.choices[0].text
                if hasattr(completion.choices[0], "text")
                else completion.choices[0].message.content
            )

            is_correct = self.check_answer(model_answer, solution)
            logger.debug(f"Problem evaluated: level={level}, correct={is_correct}")

            return level, is_correct
        except Exception as e:
            logger.error(f"Error evaluating problem: {e}")
            return level, False

    def check_answer(self, model_answer: str, solution: str) -> bool:
        """Check if the model's answer matches the solution."""
        after_think_part = (
            model_answer.split("</think>")[-1].strip()
            if "</think>" in model_answer
            else model_answer
        )

        boxed_answer = self._extract_boxed_answer(after_think_part)
        if not boxed_answer:
            lines = after_think_part.strip().split("\n")
            if lines:
                boxed_answer = lines[-1].strip()

        model_clean = self._clean_for_comparison(
            boxed_answer if boxed_answer else after_think_part
        )
        solution_clean = self._clean_for_comparison(solution)

        return model_clean == solution_clean

    def _extract_boxed_answer(self, text: str) -> Optional[str]:
        """Extract answer from a LaTeX boxed expression."""
        boxed_match = re.search(r"\\boxed{([^}]*)}", text)
        if boxed_match:
            return boxed_match.group(1)
        return None

    def _clean_for_comparison(self, text: str) -> str:
        """Clean text for comparison."""
        cleaned = re.sub(r"\\[a-zA-Z]+", "", text)
        cleaned = re.sub(r"[,\s]", "", cleaned)
        cleaned = cleaned.lower()
        return cleaned

    async def collect_trajectories(self, item) -> Tuple[List, List]:
        """Collect trajectories for the current item."""
        problem_prompt, solution, generator_id = item

        prefill = "\n<think>\n"
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": dict(problem_prompt[0])["content"]},
            {"role": "assistant", "content": prefill},
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        to_score = []

        level = None
        for lvl, generator_ids in self.curriculum.DIFFICULTY_LEVELS.items():
            if generator_id in generator_ids:
                level = lvl
                break

        for i, completion in enumerate(completions.choices):
            model_answer = prefill + (
                completion.text
                if hasattr(completion, "text")
                else completion.message.content
            )
            print("model_answer", model_answer)

            full_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": dict(problem_prompt[0])["content"]},
                {"role": "assistant", "content": model_answer},
            ]

            to_score.append((full_messages, solution, generator_id, level))

        backlog = []

        return to_score, backlog

    async def score(self, rollout_group_data) -> ScoredDataGroup:
        """Score the collected trajectories."""
        scored_data = ScoredDataGroup()
        scored_data["tokens"] = []
        scored_data["masks"] = []
        scored_data["scores"] = []
        scored_data["messages"] = []

        for i, (messages, solution, generator_id, level) in enumerate(
            rollout_group_data
        ):
            model_answer = messages[-1]["content"]
            current_score = 0.0

            is_correct = self.check_answer(model_answer, solution)
            if is_correct:
                current_score += self.config.correct_reward
            else:
                current_score += self.config.incorrect_reward

            self.percent_correct_buffer.append(1 if is_correct else 0)
            if level is not None:
                self.level_correct_buffer[level].append(1 if is_correct else 0)
            self.curriculum.record_performance(generator_id, is_correct)

            think_match = re.search(r"<think>(.*?)</think>", model_answer, re.DOTALL)
            if think_match:
                think_content = think_match.group(1).strip()
                if think_content:
                    current_score += self.config.think_block_bonus

            after_think_part = (
                model_answer.split("</think>")[-1].strip()
                if "</think>" in model_answer
                else model_answer
            )
            boxed_answer_content = self._extract_boxed_answer(after_think_part)
            if boxed_answer_content is not None:
                current_score += self.config.boxed_answer_bonus

            logger.info(
                f"Item {i}: Correct: {is_correct}, "
                f"Think Bonus: {self.config.think_block_bonus if think_match and think_match.group(1).strip() else 0}, "
                f"Boxed Bonus: {self.config.boxed_answer_bonus if boxed_answer_content is not None else 0}, "
                f"Final Score: {current_score}"
            )

            tokens_dict = tokenize_for_trainer(
                self.tokenizer,
                messages,
                None,
            )

            scored_data["tokens"].append(tokens_dict["tokens"])
            scored_data["masks"].append(tokens_dict["masks"])
            scored_data["scores"].append(current_score)
            scored_data["messages"].append(messages)

        self.curriculum.advance_difficulty()

        return scored_data

    @classmethod
    def config_init(cls) -> Tuple[InfiniteMathEnvConfig, List[APIServerConfig]]:
        """Initialize environment and OpenAI configurations with default values."""
        env_config = InfiniteMathEnvConfig(
            tokenizer_name="NousResearch/Nous-Hermes-2-Yi-34B",
            group_size=8,
            use_wandb=True,
            max_num_workers=64,
            rollout_server_url="http://localhost:8000",
            total_steps=10000,
            batch_size=1024,
            steps_per_eval=25,
            max_token_length=4096,
            inference_weight=1.0,
            wandb_name="infinimath",
            data_path_to_save_groups="data/infinite_math_groups.jsonl",
            starting_level=1,
            progress_threshold=0.8,
            min_evaluations=10,
            max_attempts_per_problem=3,
            correct_reward=1.0,
            incorrect_reward=-0.5,
            think_block_bonus=0.2,
            boxed_answer_bonus=0.2,
            apply_length_penalty=True,
            length_threshold_ratio=0.6,
            temperature=0.7,
            top_p=0.9,
            word_problem_model_name="gpt-4.1-mini",
            word_problem_openai_api_key=None,
            word_problem_openai_base_url=None,
        )

        server_configs = [
            APIServerConfig(
                model_name="NousResearch/Nous-Hermes-2-Yi-34B",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_requests_for_eval=64,
            )
        ]
        return env_config, server_configs


if __name__ == "__main__":
    InfiniteMathEnv.cli()
