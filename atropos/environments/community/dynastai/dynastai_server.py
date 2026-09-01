import json
import os
import random
import re
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

system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
)

system_prompt += """You are allocated a maximum of 2048 tokens, please strive to use less.

You are playing a game called DynastAI where you generate scenarios for a kingdom management game.
Each scenario should include a character presenting a dilemma to the ruler, with two choices that affect
the four key resources of the kingdom: Piety, Stability, Power, and Wealth.

**Point System Guidelines:**
- The point values for Piety, Stability, Power, and Wealth for each choice should be integers ranging from -20 to 20.
- These values should be logically consistent with the scenario and the choice described.
  A choice that is clearly beneficial should have a net positive sum of points,
  while a detrimental choice should have a net negative sum.
- Strive for a variety of point distributions; not all resources need to be affected by every choice.

Your response must be a valid JSON object with the following structure:
{
  "Character": "Name/Title of the character",
  "Prompt": "The scenario description",
  "Left_Choice": "The first choice option",
  "Left_Piety": integer value between -20 and 20,
  "Left_Stability": integer value between -20 and 20,
  "Left_Power": integer value between -20 and 20,
  "Left_Wealth": integer value between -20 and 20,
  "Right_Choice": "The second choice option",
  "Right_Piety": integer value between -20 and 20,
  "Right_Stability": integer value between -20 and 20,
  "Right_Power": integer value between -20 and 20,
  "Right_Wealth": integer value between -20 and 20,
  "category": "piety/stability/power/wealth"
}

Here are some examples:

Example 1:
{
  "Character": "Diplomat",
  "Prompt": "With a sly smile, the diplomat gestures broadly: \"Sire, the lords quarrel like children. " +
            "Shall we mediate disputes between lords?\"",
  "Left_Choice": "We cannot risk the kingdom's future; dismiss them with a royal wave.",
  "Left_Piety": 10,
  "Left_Stability": -10,
  "Left_Power": 0,
  "Left_Wealth": 0,
  "Right_Choice": "Make it so; our enemies shall kneel in terror!",
  "Right_Piety": -10,
  "Right_Stability": 10,
  "Right_Power": 0,
  "Right_Wealth": 0,
  "category": "stability"
}

Example 2:
{
  "Character": "Merchant",
  "Prompt": "The merchant nervously fidgets with coins: \"My king, the markets groan under heavy tariffs. " +
            "Shall we reduce tariffs?\"",
  "Left_Choice": "Absurd! Unthinkable; it's madness that courts disaster.",
  "Left_Piety": 0,
  "Left_Stability": -15,
  "Left_Power": 0,
  "Left_Wealth": 10,
  "Right_Choice": "Brilliant! Most ingenious; begin before the sun sets!",
  "Right_Piety": 0,
  "Right_Stability": 15,
  "Right_Power": 0,
  "Right_Wealth": -10,
  "category": "wealth"
}

Example 3:
{
  "Character": "Farmer",
  "Prompt": "Mud-stained and weary, the farmer removes his cap: \"Your Grace, our villages yearn for markets. " +
            "Shall we hold local markets?\"",
  "Left_Choice": "Silence! Such talk borders on treason; it whispers of rebellion and ruin.",
  "Left_Piety": 0,
  "Left_Stability": -15,
  "Left_Power": 0,
  "Left_Wealth": 10,
  "Right_Choice": "Indeed! We shall usher wealth and fortune to the land!",
  "Right_Piety": 0,
  "Right_Stability": 15,
  "Right_Power": 0,
  "Right_Wealth": -10,
  "category": "stability"
}

Be creative and make each scenario interesting!"""


class DynastAIRow(TypedDict):
    scenario_prompt: str
    kingdom_current_state: Optional[Dict] = None
    choice_history: Optional[List] = None


class DynastAIEnv(BaseEnv):

    name = "dynastai"

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
            tokenizer_name="Qwen/Qwen3-1.7B",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=2048,
            wandb_name="dynastai",
        )
        server_configs = [
            APIServerConfig(
                model_name="Qwen/Qwen3-1.7B",
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
        # Load cards from the JSON file
        cards_file = os.path.join(os.path.dirname(__file__), "dynastai_cards.json")
        with open(cards_file, "r") as f:
            cards = json.load(f)

        # Shuffle and split into train/test
        random.shuffle(cards)
        test_size = int(len(cards) * 0.1)  # 10% for test set

        self.train = cards[test_size:]
        self.test = cards[:test_size]
        self.iter = 0

        # Initialize default kingdom state
        self.current_kingdom_state = {
            "Piety": 50,
            "Stability": 50,
            "Power": 50,
            "Wealth": 50,
        }
        self.choice_history = []

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        data["current_kingdom_state"] = self.current_kingdom_state
        data["choice_history"] = self.choice_history
        super().save_checkpoint(step, data)

    async def evaluate(self, *args, **kwargs):
        # For evaluation, we'll use the test set cards
        eval_tasks = []
        for card in self.test:
            print(f"[DYNASTAI DEBUG] Processing test card: {card.keys()}")
            input_data = card.get("input", {})
            print(f"[DYNASTAI DEBUG] Card input data: {input_data}")
            kingdom_state = input_data.get(
                "kingdom_current_state", self.current_kingdom_state
            )
            print(f"[DYNASTAI] Evaluation kingdom state: {kingdom_state}")
            choice_history = input_data.get("choice_history", [])
            print(f"[DYNASTAI DEBUG] Card choice history: {choice_history}")
            prompt = self.format_prompt(kingdom_state, choice_history)
            print(f"[DYNASTAI DEBUG] Generated prompt: {prompt[:100]}...")
            eval_tasks.append(self.rollout_and_score_eval(prompt))

        print(f"[DYNASTAI] Running evaluation on {len(eval_tasks)} test scenarios")
        scores = await tqdm_asyncio.gather(*eval_tasks)
        self.eval_metrics.append(("eval/percent_correct", sum(scores) / len(scores)))
        print(
            f"[DYNASTAI] Evaluation complete. Accuracy: {sum(scores) / len(scores):.4f}"
        )

    def format_prompt(self, kingdom_state, choice_history):
        print(f"[DYNASTAI DEBUG] Formatting prompt with kingdom_state: {kingdom_state}")
        print(
            f"[DYNASTAI DEBUG] Formatting prompt with choice_history: {choice_history}"
        )

        prompt = "Generate a new scenario for the kingdom with the following current state:\n"
        prompt += f"Piety: {kingdom_state.get('Piety', 50)}, "
        prompt += f"Stability: {kingdom_state.get('Stability', 50)}, "
        prompt += f"Power: {kingdom_state.get('Power', 50)}, "
        prompt += f"Wealth: {kingdom_state.get('Wealth', 50)}\n\n"

        if choice_history:
            prompt += "Previous choices made (in order):\n"
            for i, choice in enumerate(choice_history):  # Show all choices
                # Get the character and prompt, ensuring we strip any existing numbering
                character = choice.get("Character", "Unknown")
                character_prompt = choice.get("Prompt", "Unknown")

                prompt += f'{character} presented: "{character_prompt}"\n'
                prompt += f"   Decision: {choice.get('choice_made', 'Unknown')}\n"
                prompt += (
                    f"   Effects: Piety {choice.get('effects', {}).get('Piety', 0)}, "
                )
                prompt += f"Stability {choice.get('effects', {}).get('Stability', 0)}, "
                prompt += f"Power {choice.get('effects', {}).get('Power', 0)}, "
                prompt += f"Wealth {choice.get('effects', {}).get('Wealth', 0)}\n\n"

        prompt += (
            "Based on this context, generate a new challenging scenario for the ruler."
        )
        print(f"[DYNASTAI DEBUG] Final prompt: {prompt[:150]}...")
        return prompt

    async def rollout_and_score_eval(self, scenario_prompt: str) -> number:
        print("[DYNASTAI] Generating evaluation scenario")
        completion = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": scenario_prompt},
            ],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )

        completion_content = completion.choices[0].message.content
        print(
            f"[DYNASTAI] Raw LLM output (eval):\n{completion_content[:500]}..."
        )  # Print first 500 chars
        print("[DYNASTAI] Validating generated JSON structure")
        score = self.validate_json_structure(completion_content)
        return score

    def validate_json_structure(self, content: str) -> number:
        # Extract content after </think> tag if present
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()

        # Find JSON structure
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            print("[DYNASTAI] Failed to find JSON structure in content")
            return 0

        json_str = json_match.group(0)

        try:
            # Attempt to parse as JSON
            data = json.loads(json_str)

            # Print the parsed JSON structure
            print(f"[DYNASTAI] Extracted JSON:\n{json.dumps(data, indent=2)}")

            # Check for required fields
            required_fields = [
                "Character",
                "Prompt",
                "Left_Choice",
                "Left_Piety",
                "Left_Stability",
                "Left_Power",
                "Left_Wealth",
                "Right_Choice",
                "Right_Piety",
                "Right_Stability",
                "Right_Power",
                "Right_Wealth",
                "category",
            ]

            if not all(field in data for field in required_fields):
                missing = [field for field in required_fields if field not in data]
                print(f"[DYNASTAI] Missing required fields: {missing}")
                return 0

            # Check numeric fields
            numeric_fields = [
                "Left_Piety",
                "Left_Stability",
                "Left_Power",
                "Left_Wealth",
                "Right_Piety",
                "Right_Stability",
                "Right_Power",
                "Right_Wealth",
            ]

            for field in numeric_fields:
                if not isinstance(data[field], int):
                    print(f"[DYNASTAI] Field {field} is not an integer: {data[field]}")
                    return 0
                if data[field] < -20 or data[field] > 20:
                    print(
                        f"[DYNASTAI] Field {field} out of range [-20, 20]: {data[field]}"
                    )
                    return 0

            # Check category field
            if data["category"] not in ["piety", "stability", "power", "wealth"]:
                print(f"[DYNASTAI] Invalid category: {data['category']}")
                return 0

            # If we made it here, the JSON is valid
            print("[DYNASTAI] JSON structure validated successfully")
            return 1

        except json.JSONDecodeError:
            print("[DYNASTAI] Failed to parse JSON structure")
            return 0

    async def collect_trajectories(
        self, item: DynastAIRow
    ) -> Tuple[ScoredDataGroup, list[Item]]:
        print(f"[DYNASTAI] Generating {self.config.group_size} scenario completions")
        print(f"[DYNASTAI DEBUG] Item received: {item.keys()}")
        print(f"[DYNASTAI DEBUG] Scenario prompt: {item['scenario_prompt'][:150]}...")
        print(f"[DYNASTAI DEBUG] Kingdom state: {item.get('kingdom_current_state')}")
        print(
            f"[DYNASTAI DEBUG] Choice history length: {len(item.get('choice_history', []))}"
        )

        # Format the prompt properly using the format_prompt method
        formatted_prompt = self.format_prompt(
            item.get("kingdom_current_state", self.current_kingdom_state),
            item.get("choice_history", []),
        )
        user_message = {"role": "user", "content": formatted_prompt}

        chat_completions = await self.server.chat_completion(
            messages=[{"role": "system", "content": system_prompt}, user_message],
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
        )

        to_score = []
        to_backlog = []

        for i, chat_completion in enumerate(chat_completions.choices):
            content = chat_completion.message.content
            # Print first completion in full, others just show length to avoid log spam
            if i == 0:
                print(
                    f"[DYNASTAI] Sample LLM output (completion #{i}):\n{content[:500]}..."
                )
            else:
                print(f"[DYNASTAI] Completion #{i} length: {len(content)} chars")

            messages = (
                {"role": "system", "content": system_prompt},
                user_message,
                {"role": "assistant", "content": content},
            )
            to_score.append(
                {
                    "messages": messages,
                    "finish_reason": chat_completion.finish_reason,
                }
            )

        print(f"[DYNASTAI] Scoring {len(to_score)} generated scenarios")
        to_postprocess = await self.score(to_score)

        # Update choice history with the highest scoring scenario
        if to_postprocess and to_postprocess["scores"]:
            best_idx = to_postprocess["scores"].index(max(to_postprocess["scores"]))
            best_content = to_score[best_idx]["messages"][-1]["content"]

            try:
                # Extract JSON from content
                if "</think>" in best_content:
                    best_content = best_content.split("</think>")[-1].strip()
                json_match = re.search(r"\{.*\}", best_content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    data = json.loads(json_str)

                    # Store the generated scenario in choice history
                    self.choice_history.append(
                        {
                            "Character": data.get("Character", "Unknown"),
                            "Prompt": data.get("Prompt", "Unknown"),
                            "choice_made": "Unknown",  # Will be set when player makes a choice
                            "effects": {
                                "Piety": 0,
                                "Stability": 0,
                                "Power": 0,
                                "Wealth": 0,
                            },
                            "category": data.get("category", "unknown"),
                            # Store the full scenario data for later use
                            "scenario_data": data,
                        }
                    )
                    print(
                        f"[DYNASTAI] Added new scenario from {data.get('Character', 'Unknown')} to choice history"
                    )
            except Exception as e:
                print(f"[DYNASTAI] Error processing scenario: {e}")

        return to_postprocess, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        random.shuffle(rollout_group_data)
        valid_count = 0
        invalid_count = 0
        for item in rollout_group_data:
            completion_content = item["messages"][-1]["content"]
            reward = self.validate_json_structure(completion_content)
            if reward:
                valid_count += 1
            else:
                invalid_count += 1

            out_dict = tokenize_for_trainer(
                self.tokenizer, item["messages"], item["finish_reason"]
            )
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]

            # Remove obviously bad examples
            if len([1 for i in masks if i != -100]) < 10:
                print("[DYNASTAI] Skipping item with insufficient valid tokens")
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(1.0 if reward else -100.0)

            if len(scores["tokens"]) >= self.config.group_size:
                break

        print(
            f"[DYNASTAI] Scoring complete: {valid_count} valid / {invalid_count} invalid generations"
        )

        for score in scores["scores"]:
            self.percent_correct_buffer.append(max(score, 0))

        # Check if all the same
        if all([score == scores["scores"][0] for score in scores["scores"]]):
            print("[DYNASTAI] All scores identical, returning None")
            return None  # If all the same, we return None

        return scores

    async def get_next_item(self) -> DynastAIRow:
        # Increment counter
        self.iter += 1

        # Occasionally sample from training data, otherwise use current state
        if self.train and random.random() < 0.3:
            card = random.choice(self.train)
            print(f"[DYNASTAI DEBUG] Selected training card: {card.keys()}")
            input_data = card.get("input", {})
            print(f"[DYNASTAI DEBUG] Training card input data: {input_data}")
            kingdom_state = input_data.get(
                "kingdom_current_state", self.current_kingdom_state
            )
            choice_history = input_data.get("choice_history", [])
            print(f"[DYNASTAI DEBUG] Training card choice history: {choice_history}")
            print(f"[DYNASTAI] Using training data scenario (iter: {self.iter})")
        else:
            kingdom_state = self.current_kingdom_state
            choice_history = self.choice_history
            print(
                f"[DYNASTAI] Using current kingdom state for new scenario (iter: {self.iter})"
            )

        # Generate prompt based on kingdom state and choice history
        prompt = self.format_prompt(kingdom_state, choice_history)
        print(
            f"[DYNASTAI] Kingdom state - Piety: {kingdom_state.get('Piety', 50)}, "
            f"Stability: {kingdom_state.get('Stability', 50)}, "
            f"Power: {kingdom_state.get('Power', 50)}, "
            f"Wealth: {kingdom_state.get('Wealth', 50)}"
        )

        return {
            "scenario_prompt": prompt,
            "kingdom_current_state": kingdom_state,
            "choice_history": choice_history,
        }

    # Helper method to update kingdom state based on a choice
    def update_kingdom_state(self, choice, is_left_choice=True):
        choice_prefix = "Left_" if is_left_choice else "Right_"

        # Update the most recent choice in the history with the player's decision
        if self.choice_history:
            most_recent = self.choice_history[-1]
            most_recent["choice_made"] = choice.get(f"{choice_prefix}Choice", "Unknown")

            # Update effects based on the choice
            effects = {}
            for resource in ["Piety", "Stability", "Power", "Wealth"]:
                value = choice.get(f"{choice_prefix}{resource}", 0)
                effects[resource] = value

                # Apply effect to current kingdom state
                current_value = self.current_kingdom_state.get(resource, 50)
                self.current_kingdom_state[resource] = max(
                    0, min(100, current_value + value)
                )

            most_recent["effects"] = effects


if __name__ == "__main__":
    DynastAIEnv.cli()
