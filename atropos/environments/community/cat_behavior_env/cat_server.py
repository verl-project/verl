import json
from typing import Dict, List, Optional, Tuple, TypedDict, Union

from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item, number
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Configs

CAT_BEHAVIORS_FILEPATH = "environments/community/cat_behavior_env/cat_behaviors.json"

# Prompts


def load_cat_behaviors_for_prompt(filepath: str) -> str:
    """Loads cat behaviors from a JSONL file and formats them for the system prompt."""
    behaviors_description = [
        "\n\nHere is a detailed list of behaviors you, as a cat, can use and what they generally mean:"
    ]

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            behaviors = json.load(f)  # <<< one big load
            for behavior_data in behaviors:
                behaviors_description.append(
                    f"- **{behavior_data['behavior']}**: {behavior_data['description']}"
                )
        return "\n".join(behaviors_description)
    except FileNotFoundError:
        return (
            "\n\nWarning: Cat behaviors file not found at '{filepath}'. "
            "You'll have to rely on your basic cat instincts (meow, hiss, purr, hairball, silence)."
        )
    except json.JSONDecodeError as e:
        return (
            f"\n\nWarning: Error decoding cat behaviors file '{filepath}'. "
            f"Please ensure it's valid JSONL. Error: {e}. Rely on basic instincts."
        )


cat_behaviors_list_string = load_cat_behaviors_for_prompt(CAT_BEHAVIORS_FILEPATH)

cat_system_prompt = (
    "You are a cat. The primary ways you can communicate are by meowing, hissing, "
    "purring, making a hairball sound, or remaining silent. "
    "You will be given a collection of scenarios which describe various needs you want "
    "to be met by your caretaker. "
    "Please try to communicate with your caretaker through your available cat-like "
    "expressions and actions, referring to the list of behaviors below if needed."
    "Rules:"
    "Do not speak in English"
    "No use of Emojis"
    "Format should be a sound then context in ()"
    "If no sound use ~Silent~"
    ""
    "Examples:"
    "Mew! (Looks at up at you)"
    "~Silent~ (Looks at up at you)"
    "Hiss! (Stares at the litterbox)"
    f"{cat_behaviors_list_string}"  # Appending the loaded behaviors here
)
cat_system_prompt += (
    """You are allocated a maximum of 2048 tokens, please strive to use less."""
)

caretaker_system_prompt = (
    "You are the caretaker of this cat. It is trying to communicate its various needs to you via cat language."
    "Provide a written string which provides a set of interventions."
    "You will only have 5 opportunities to interact with the cat. Choose what you say wisely."
)


class CatRow(TypedDict):
    scenario: str


class GSM8kEnv(BaseEnv):

    name = "gsm8k"

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

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=61,
            batch_size=1,
            steps_per_eval=60,
            max_token_length=2048,
            wandb_name="gsm8k",
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=256,
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

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        # Call the parent method to handle the server metrics
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        # self.train = load_dataset("gsm8k", "main", split="train").shuffle(seed=42)
        # test_data = load_dataset("gsm8k", "main", split="test").shuffle(seed=42)
        with open(
            "environments/community/cat_behavior_env/cat_scenarios.json",
            "r",
            encoding="utf-8",
        ) as f:
            test_data = json.load(f)
        self.test = list()
        self.train = list()
        for item in test_data:
            self.test.append(
                {
                    "scenario": item["scenario"],
                    # "gold_answer": item["answer"]
                    # .split("#")[-1]
                    # .strip()
                    # .replace(",", ""),
                }
            )
            self.train.append(
                {
                    "scenario": item["scenario"],
                }
            )
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def rollout_and_score_eval(self, scenario: str, answer: str) -> number:
        # completion = await self.server.chat_completion(
        #     messages=[
        #         {"role": "system", "content": system_prompt},
        #         {"role": "user", "content": scenario},
        #     ],
        #     n=1,
        #     max_tokens=self.config.max_token_length,
        #     temperature=0.0,
        #     split="eval",
        # )
        # gold_parsed = parse(
        #     "\\boxed{" + answer + "}",
        #     extraction_mode="first_match",
        #     extraction_config=[LatexExtractionConfig()],
        # )
        # answer_parsed = parse(
        #     completion.choices[0].message.content.split("</think>")[-1],
        #     extraction_config=[
        #         LatexExtractionConfig(
        #             normalization_config=NormalizationConfig(
        #                 nits=False,
        #                 malformed_operators=False,
        #                 basic_latex=True,
        #                 equations=True,
        #                 boxed="all",
        #                 units=True,
        #             ),
        #             # Ensures that boxed is tried first
        #             boxed_match_priority=0,
        #             try_extract_without_anchor=False,
        #         )
        #     ],
        #     extraction_mode="first_match",
        # )
        # score = 1 if verify(answer_parsed, gold_parsed) else 0
        # return score
        return 1

    async def evaluate(self, *args, **kwargs):
        eval_tasks = []
        for item in self.test:
            eval_tasks.append(self.rollout_and_score_eval(item["scenario"]))
        scores = await tqdm_asyncio.gather(*eval_tasks)
        self.eval_metrics.append(("eval/percent_correct", sum(scores) / len(scores)))

    async def collect_trajectories(
        self, item: CatRow
    ) -> Tuple[ScoredDataGroup, list[Item]]:
        user_message = {"role": "user", "content": item["scenario"]}
        to_score = list()
        to_backlog = list()
        for j in range(self.config.group_size):
            all_messages = []
            history = []
            cat_history = [user_message]
            for turn_iter in range(5):
                cat_completions = await self.server.chat_completion(
                    messages=[{"role": "system", "content": cat_system_prompt}]
                    + cat_history,
                    n=self.config.group_size,
                    max_tokens=self.config.max_token_length,
                )

                for i, cat_completion in enumerate(cat_completions.choices):
                    if i == 0:
                        cat_message = cat_completion.message.content
                cat_response = {"role": "system", "content": cat_message}
                cat_history.append(cat_response)
                caretaker_message = {"role": "user", "content": cat_message}
                history.append(caretaker_message)
                caretaker_completions = await self.server.chat_completion(
                    messages=[{"role": "system", "content": caretaker_system_prompt}]
                    + history,
                    n=1,
                    max_tokens=self.config.max_token_length,
                )
                caretaker_response = {
                    "role": "assistant",
                    "content": caretaker_completions.choices[0].message.content,
                }
                cat_history.append(caretaker_response)
                history.append(caretaker_response)

                if turn_iter == 0:
                    messages = [
                        {"role": "system", "content": cat_system_prompt},
                        user_message,
                        cat_response,
                        caretaker_response,
                    ]
                else:
                    messages = [cat_response, caretaker_response]
                all_messages.extend(messages)
            all_messages = tuple(all_messages)
            to_score.append(
                {
                    "messages": all_messages,
                }
            )
            # import pdb; pdb.set_trace()
        to_postprocess = await self.score(to_score)
        # import pdb; pdb.set_trace()
        return to_postprocess, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()

        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        # # random.shuffle(rollout_group_data)
        for item in rollout_group_data:
            final_question = list(item["messages"]) + [
                {
                    "role": "system",
                    "content": (
                        "The conversation is over. Say purr if the caretaker did everything perfectly "
                        "and there was nothing that the caretaker could have done even slightly better. "
                        "Otherwise, say meow. Make sure it is rare that you rate the caretaker with a purr."
                    ),
                }
            ]
            caretaker_completions = await self.server.chat_completion(
                messages=final_question,
                n=1,
                max_tokens=self.config.max_token_length,
            )
            final_out = {
                "role": "system",
                "content": [
                    row.message.content for row in caretaker_completions.choices
                ][0],
            }

            final_score = purrfect_eval(final_out["content"])

            out_dict = tokenize_for_trainer(
                self.tokenizer, [row for row in item["messages"]] + [final_out]
            )
            scores["tokens"].append(out_dict["tokens"])
            scores["masks"].append(out_dict["masks"])
            scores["scores"].append(final_score)

        #     tokens = out_dict["tokens"]
        #     masks = out_dict["masks"]
        #     # remove obviously bad examples
        #     if len([1 for i in masks if i != -100]) < 10:
        #         continue
        #     scores["tokens"].append(tokens)
        #     scores["masks"].append(masks)
        #     scores["scores"].append(1.0)
        #     if len(scores["tokens"]) >= self.config.group_size:
        #         break
        # for score in scores["scores"]:
        #     self.percent_correct_buffer.append(max(score, 0))
        # # check if all the same
        # # print(scores['scores'])
        # if all([score == 1 for score in scores["scores"]]):
        #     # Do length penalty :)
        #     token_lengths = [len(token) for token in scores["tokens"]]
        #     if max(token_lengths) == 0:
        #         # What? But don't want to crash a run so just in case...
        #         return None

        #     # Get max allowed token length from config
        #     max_allowed_length = self.config.max_token_length
        #     # Set threshold at 50% of max_token_length - no penalty below this
        #     length_threshold = max_allowed_length * 0.5

        #     # Apply modified length penalty with threshold
        #     scores["scores"] = []
        #     for length in token_lengths:
        #         if length <= length_threshold:
        #             # No penalty for responses under threshold
        #             scores["scores"].append(1.0)
        #         else:
        #             # Calculate how far we are between threshold and max as a percentage
        #             percentage_of_range = (length - length_threshold) / (
        #                 max_allowed_length - length_threshold
        #             )
        #             # Cap at 1.0 in case length exceeds max_allowed_length
        #             percentage_of_range = min(percentage_of_range, 1.0)
        #             # Apply linear penalty scaling from 1.0 down to 0.0
        #             scores["scores"].append(1.0 - percentage_of_range)
        #     if all([scores["scores"][0] == score for score in scores["scores"]]):
        #         return None  # If all the same, we return None
        #     return scores
        # else:
        #     # If the gold solution is not parseable, we return None
        #     return None
        return None

        # gold_parsed = parse(
        #     rollout_group_data[0]["gold_answer"],
        #     extraction_mode="first_match",
        #     extraction_config=[LatexExtractionConfig()],
        # )
        # if len(gold_parsed) != 0:
        #     # We require the answer to be provided in correct latex (no malformed operators)
        #     random.shuffle(rollout_group_data)
        #     for item in rollout_group_data:
        #         # print(item[0][-1]["content"])
        #         answer_parsed = parse(
        #             item["messages"][-1]["content"].split("</think>")[-1],
        #             extraction_config=[
        #                 LatexExtractionConfig(
        #                     normalization_config=NormalizationConfig(
        #                         nits=False,
        #                         malformed_operators=False,
        #                         basic_latex=True,
        #                         equations=True,
        #                         boxed="all",
        #                         units=True,
        #                     ),
        #                     # Ensures that boxed is tried first
        #                     boxed_match_priority=0,
        #                     try_extract_without_anchor=False,
        #                 )
        #             ],
        #             extraction_mode="first_match",
        #         )
        #         # Reward 1 if the content is the same as the ground truth, 0 otherwise
        #         reward = verify(answer_parsed, gold_parsed)
        #         # print(
        #         #     f"message: {item[0][-1]['content']}, ground_truth: {item[1]}, reward: {reward}"
        #         # )
        #         out_dict = tokenize_for_trainer(
        #             self.tokenizer, item["messages"], item["finish_reason"]
        #         )
        #         tokens = out_dict["tokens"]
        #         masks = out_dict["masks"]
        #         # remove obviously bad examples
        #         if len([1 for i in masks if i != -100]) < 10:
        #             continue
        #         scores["tokens"].append(tokens)
        #         scores["masks"].append(masks)
        #         scores["scores"].append(1.0 if reward else -1.0)
        #         if len(scores["tokens"]) >= self.config.group_size:
        #             break
        #     for score in scores["scores"]:
        #         self.percent_correct_buffer.append(max(score, 0))
        #     # check if all the same
        #     # print(scores['scores'])
        #     if all([score == 1 for score in scores["scores"]]):
        #         # Do length penalty :)
        #         token_lengths = [len(token) for token in scores["tokens"]]
        #         if max(token_lengths) == 0:
        #             # What? But don't want to crash a run so just in case...
        #             return None

        #         # Get max allowed token length from config
        #         max_allowed_length = self.config.max_token_length
        #         # Set threshold at 50% of max_token_length - no penalty below this
        #         length_threshold = max_allowed_length * 0.5

        #         # Apply modified length penalty with threshold
        #         scores["scores"] = []
        #         for length in token_lengths:
        #             if length <= length_threshold:
        #                 # No penalty for responses under threshold
        #                 scores["scores"].append(1.0)
        #             else:
        #                 # Calculate how far we are between threshold and max as a percentage
        #                 percentage_of_range = (length - length_threshold) / (
        #                     max_allowed_length - length_threshold
        #                 )
        #                 # Cap at 1.0 in case length exceeds max_allowed_length
        #                 percentage_of_range = min(percentage_of_range, 1.0)
        #                 # Apply linear penalty scaling from 1.0 down to 0.0
        #                 scores["scores"].append(1.0 - percentage_of_range)
        #     if all([scores["scores"][0] == score for score in scores["scores"]]):
        #         return None  # If all the same, we return None
        #     return scores
        # else:
        #     # If the gold solution is not parseable, we return None
        #     return None
        return None

    async def get_next_item(self) -> CatRow:
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        print(f"iteration: {self.iter}")
        return next_item


def purrfect_eval(st: str) -> float:
    if "purr" in st.lower():
        return 1.0
    return 0.0


if __name__ == "__main__":
    GSM8kEnv.cli()
