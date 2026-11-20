import json
import os
from typing import Dict, List, Optional, Tuple

from bs4 import BeautifulSoup
from pydantic import Field
from transformers.models.auto.tokenization_auto import AutoTokenizer

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

from .accessibility_rules import (
    AccessibilityRule,
    LabelAssociationRule,
    MissingAltTextRule,
)


class AccessibilityEnvConfig(BaseEnvConfig):
    dataset_path: str = Field(
        default="data/accessibility_dataset.jsonl",  # Default relative path
        description="Path to the JSONL file containing the accessibility dataset.",
    )


class AccessibilityEnv(BaseEnv):
    config: AccessibilityEnvConfig
    name = "accessibility_env"

    def __init__(
        self,
        config: AccessibilityEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.tokenizer = None

        # Initialize your list of rule instances
        self.accessibility_rules: List[AccessibilityRule] = [
            MissingAltTextRule(),
            LabelAssociationRule(),
        ]

        # For quick lookup if needed, though iterating self.accessibility_rules is fine
        self.rules_by_key = {rule.issue_key: rule for rule in self.accessibility_rules}

    @classmethod
    def config_init(cls) -> Tuple[AccessibilityEnvConfig, List[APIServerConfig]]:
        current_dataset_size = 10

        env_config = AccessibilityEnvConfig(
            tokenizer_name="gpt2",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=current_dataset_size,
            batch_size=1,
            steps_per_eval=current_dataset_size,
            max_token_length=2048,
            wandb_name="accessibility_openai_default_dev",
        )

        openai_api_key_from_env = os.environ.get("OPENAI_API_KEY")
        if not openai_api_key_from_env:
            print(
                "WARNING (from config_init): OPENAI_API_KEY environment variable not set for default config!"
            )

        server_configs = [
            APIServerConfig(
                model_name="gpt-3.5-turbo",
                api_key=openai_api_key_from_env,
            )
        ]
        return env_config, server_configs

    async def setup(self):
        print(f"[{self.name}] Setting up environment...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_name, trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            if self.tokenizer.chat_template is None:
                self.tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{{ message['role'] + ': ' + message['content'] + '\\n' }}"
                    "{% endfor %}"
                )

                print(
                    f"[{self.name}] Set a default chat_template for tokenizer '{self.config.tokenizer_name}'."
                )

            print(
                f"[{self.name}] Tokenizer '{self.config.tokenizer_name}' loaded successfully."
            )
        except Exception as e:
            print(
                f"[{self.name}] Error loading tokenizer '{self.config.tokenizer_name}': {e}"
            )
            raise RuntimeError(f"Failed to load tokenizer: {e}") from e

        # Load dataset from file
        self.dataset: List[Dict] = []
        env_script_dir = os.path.dirname(os.path.abspath(__file__))
        full_dataset_path = os.path.join(env_script_dir, self.config.dataset_path)

        print(f"[{self.name}] Attempting to load dataset from: {full_dataset_path}")
        try:
            with open(full_dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():  # Ensure line is not empty
                        self.dataset.append(json.loads(line))
            if not self.dataset:
                raise FileNotFoundError(
                    "Dataset file was empty or contained no valid JSON lines."
                )
        except FileNotFoundError:
            print(f"[{self.name}] ERROR: Dataset file not found at {full_dataset_path}")
            raise
        except json.JSONDecodeError as e:
            print(
                f"[{self.name}] ERROR: Failed to decode JSON from {full_dataset_path}. Error: {e}"
            )
            raise
        except Exception as e:
            print(
                f"[{self.name}] ERROR: An unexpected error occurred while loading dataset: {e}"
            )
            raise

        self.iter = 0
        print(
            f"""[{self.name}] Setup complete. Loaded {len(self.dataset)}
        items. Initialized {len(self.accessibility_rules)} accessibility rules."""
        )

    async def get_next_item(self) -> Optional[Item]:
        if self.iter >= len(self.dataset):
            if self.iter >= self.config.total_steps:
                return None

            print(f"[{self.name}] Reached end of dataset or total_steps.")
            return None

        item_data = self.dataset[self.iter]
        self.iter += 1

        return item_data

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        original_html = item["html"]
        system_message_content = (
            "You are an expert web developer specializing in accessibility. "
            "Given the following HTML snippet, please make the minimal necessary modifications "
            "to ensure it meets WCAG 2.1 AA standards for the issues present. "
            "Output only the complete, modified HTML snippet. Do not include explanations unless explicitly asked."
        )
        user_message_content = (
            f"Original HTML:\n```html\n{original_html}\n```\nModified HTML:"
        )

        messages = [
            {"role": "system", "content": system_message_content},
            {"role": "user", "content": user_message_content},
        ]

        chat_completions = await self.server.chat_completion(
            messages=messages,
            n=self.config.group_size,
            max_tokens=1024,
        )

        to_score_inputs = []
        if chat_completions is not None:
            for choice in chat_completions.choices:
                llm_response_content = choice.message.content
                full_exchange_messages = messages + [
                    {"role": "assistant", "content": llm_response_content}
                ]
                to_score_inputs.append(
                    {
                        "full_exchange_messages": full_exchange_messages,
                        "llm_modified_html": llm_response_content,
                        "original_html_info": item,
                    }
                )

        scored_data_group = await self.score(to_score_inputs)
        return scored_data_group, []  # Assuming no backlog for now

    async def score(self, rollout_group_data: List[dict]) -> Optional[ScoredDataGroup]:
        print(f"[{self.name}] Scoring {len(rollout_group_data)} rollouts...")

        # Initialize lists to store data for all successfully processed items in the batch
        final_tokens_batch: List[List[int]] = []
        final_masks_batch: List[List[int]] = []
        final_scores_batch: List[float] = []
        final_concatenated_dialogues_batch: List[str] = []

        # Optional fields for ScoredDataGroup, will remain None for this basic setup
        all_advantages: Optional[List[List[float]]] = None
        all_ref_logprobs: Optional[List[List[float]]] = None

        for data_item in rollout_group_data:
            llm_html_str = data_item["llm_modified_html"]
            original_info = data_item["original_html_info"]
            full_exchange_messages_list_of_dicts = data_item[
                "full_exchange_messages"
            ]  # This is List[Dict[str, str]]

            current_item_score = 0.0
            num_issues_actually_fixed = 0
            issues_expected_to_fix = original_info.get("issues_to_fix", [])
            num_issues_targeted = len(issues_expected_to_fix)

            soup: Optional[BeautifulSoup] = None
            can_proceed_with_rule_checks = False
            try:
                soup = BeautifulSoup(llm_html_str, "lxml")
                can_proceed_with_rule_checks = True
            except Exception as e:
                print(
                    f"[{self.name}] Item {original_info.get('id', 'N/A')}: Could not parse LLM output as HTML: {e}"
                )

            if can_proceed_with_rule_checks and soup is not None:
                for rule_instance in self.accessibility_rules:
                    if rule_instance.issue_key in issues_expected_to_fix:
                        try:
                            if rule_instance.check(soup, original_info):
                                num_issues_actually_fixed += 1
                                print(
                                    f"""[{self.name}] Item {original_info.get('id', 'N/A')}:
                                    Rule '{rule_instance.issue_key}' PASSED."""
                                )
                            else:
                                print(
                                    f"""[{self.name}] Item {original_info.get('id', 'N/A')}:
                                    Rule '{rule_instance.issue_key}' FAILED."""
                                )
                        except Exception as rule_e:
                            print(
                                f"""[{self.name}] Item {original_info.get('id', 'N/A')}:
                                Error executing rule '{rule_instance.issue_key}': {rule_e}"""
                            )

            # Determine score based on fixes and parseability
            if num_issues_targeted > 0:
                if not can_proceed_with_rule_checks:  # Parsing failed
                    current_item_score = (
                        -1.0 * num_issues_targeted
                    )  # Penalize per targeted issue if unparseable
                elif num_issues_actually_fixed == num_issues_targeted:
                    current_item_score = 1.0  # All targeted issues fixed
                elif (
                    num_issues_actually_fixed > 0
                ):  # Some, but not all, targeted issues fixed
                    current_item_score = 0.8 * (
                        num_issues_actually_fixed / num_issues_targeted
                    )
                else:  # Parseable, but no targeted issues fixed
                    current_item_score = -0.5
            else:  # No issues were targeted for this item (e.g., input was considered good by dataset design)
                if (
                    not can_proceed_with_rule_checks
                ):  # LLM made a good input unparseable
                    current_item_score = -1.0
                else:  # Parseable, and no issues were targeted (good input remained good)
                    current_item_score = 0.0  # Neutral score

            # Tokenization
            try:
                if not self.tokenizer:
                    raise ValueError("Tokenizer not initialized.")
                tokenized_output = tokenize_for_trainer(
                    self.tokenizer, full_exchange_messages_list_of_dicts
                )
            except Exception as e:
                print(
                    f"""[{self.name}] Error during tokenization for item
                    {original_info.get('id', 'N/A')}: {e}. Skipping this item."""
                )
                continue  # Skip to the next data_item in rollout_group_data

            if "tokens" not in tokenized_output or "masks" not in tokenized_output:
                print(
                    f"""[{self.name}] Tokenization did not produce 'tokens' or
                    'masks' for item {original_info.get('id', 'N/A')}. Skipping this item."""
                )
                continue  # Skip to the next data_item

            # If we reach here, scoring and tokenization for the current item were successful
            final_tokens_batch.append(tokenized_output["tokens"])
            final_masks_batch.append(tokenized_output["masks"])
            final_scores_batch.append(current_item_score)

            if self.config.include_messages:
                formatted_message_log = "".join(
                    f"{msg_dict['role']}: {msg_dict['content']}\n"
                    for msg_dict in full_exchange_messages_list_of_dicts
                )
                final_concatenated_dialogues_batch.append(formatted_message_log.strip())

        # After processing all items in rollout_group_data
        if (
            not final_scores_batch
        ):  # If all items were skipped (e.g., due to tokenization errors)
            print(
                f"""[{self.name}] No valid items to include in ScoredDataGroup
                after processing all rollouts, returning None."""
            )
            return None

        data_to_return: ScoredDataGroup = {
            "tokens": final_tokens_batch,
            "masks": final_masks_batch,
            "scores": final_scores_batch,
            "advantages": all_advantages,
            "ref_logprobs": all_ref_logprobs,
            "group_overrides": {},
            "messages": (
                final_concatenated_dialogues_batch
                if self.config.include_messages and final_concatenated_dialogues_batch
                else None
            ),  # type: ignore[assignment]
            "overrides": None,
        }

        print(
            f"[{self.name}] Scoring batch complete. Final scores for batch: {data_to_return['scores']}"
        )
        return data_to_return

    async def evaluate(
        self,
    ):
        print(f"[{self.name}] Evaluate method called (placeholder).")
        # Implement evaluation logic if you have a separate test set and metrics
        pass


if __name__ == "__main__":
    # This makes your environment runnable with `python accessibility_env.py process`
    AccessibilityEnv.cli()
