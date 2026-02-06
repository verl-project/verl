import ast  # For safely evaluating the LLM's string output
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item

try:
    from atroposlib.utils.tokenize_for_trainer import (
        tokenize_for_trainer_dpo as imported_tokenize_for_trainer_dpo,
    )
except ImportError:
    imported_tokenize_for_trainer_dpo = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- System Prompts for Generating Chosen/Rejected Responses ---
SYSTEM_PROMPT_CHOSEN = """
You are a helpful and engaging AI assistant. Your goal is to provide clear, empathetic, and
insightful responses that encourage further conversation. Be positive and proactive.
"""

SYSTEM_PROMPT_REJECTED = """"
You are an AI assistant. Provide very brief, blunt, and unhelpful responses.
Do not elaborate or ask follow-up questions.
"""
# --- End System Prompts ---

# --- Master Prompt for Generating Initial Prompts ---
PROMPT_GENERATION_MASTER_PROMPT = """
You are a creative assistant. Your task is to generate a list of 10 unique and random conversational prompts.
Each prompt should be suitable for starting a general conversation with an AI assistant.
The prompts should cover a variety of topics, tones (e.g., inquisitive, reflective, casual), and lengths.
Format your entire output as a single Python list of dictionaries, where each dictionary has a
single key "prompt" and the value is the prompt string.

Example format:
[
    {"prompt": "What's the most interesting dream you've ever had, if AIs could dream?"},
    {"prompt": "If you could learn any new skill instantly, what would it be and why?"}
]

Provide only the Python list of dictionaries, with no other surrounding text, explanations, or markdown formatting.
"""
# --- End Master Prompt ---


class GSM8KConversationalStyleDPOEnvConfig(BaseEnvConfig):
    """Config for GSM8KConversationalStyleDPOEnv."""

    dataset_name: str = Field(
        "synthetic_conversational_style_prompts_via_gsm8k_env",
        description="Name of the dataset to use (source of prompts).",
    )
    shuffle_dataset: bool = Field(
        True, description="Whether to shuffle the dataset of prompts"
    )
    data_path_to_save_groups: Optional[str] = Field(
        None, description="Path to save .jsonl and .html processed rollouts."
    )
    # Generation parameters for chosen responses
    chosen_temperature: float = Field(
        0.7, description="Temperature for generating chosen responses."
    )
    chosen_max_tokens: int = Field(150, description="Max tokens for chosen responses.")
    # Generation parameters for rejected responses
    rejected_temperature: float = Field(
        0.4, description="Temperature for generating rejected responses."
    )
    rejected_max_tokens: int = Field(
        50, description="Max tokens for rejected responses."
    )
    prompt_generation_temperature: float = Field(
        0.8, description="Temperature for LLM generating the initial list of prompts."
    )
    prompt_generation_max_tokens: int = Field(
        1000, description="Max tokens for LLM generating the initial list of prompts."
    )


class GSM8KConversationalStyleDPOEnv(BaseEnv):
    name = "gsm8k_dynamic_conversational_dpo"
    name_config_cls = GSM8KConversationalStyleDPOEnvConfig

    def __init__(
        self,
        config: GSM8KConversationalStyleDPOEnvConfig,
        server_configs: Optional[List[APIServerConfig]] = None,
        slurm=True,
        testing=False,
    ):
        # Ensure server_configs is not None if we intend to use self.server
        # The BaseEnv will select the first server_config if multiple are provided and split is not specified.
        resolved_server_configs = server_configs
        if not resolved_server_configs:
            logger.warning(
                f"No server_configs provided for {self.name}, chat_completion calls "
                f"will fail if not overridden in config_init or CLI."
            )
            # You might want to provide a default dummy one if testing without a server is intended for some paths
            # For this version, we expect it to be configured properly for LLM calls.

        super().__init__(config, resolved_server_configs, slurm, testing)
        self.config: GSM8KConversationalStyleDPOEnvConfig = config
        self.prompt_dataset: List[Dict[str, str]] = []  # Stores only prompts now
        self.iter: int = 0

        if imported_tokenize_for_trainer_dpo is not None:
            self._tokenize_dpo_fn = imported_tokenize_for_trainer_dpo
            logger.info(f"Using imported tokenize_for_trainer_dpo for {self.name}")
        else:
            self._tokenize_dpo_fn = self._placeholder_tokenize_for_dpo
            logger.info(f"Using placeholder tokenize_for_trainer_dpo for {self.name}")

    def _placeholder_tokenize_for_dpo(self, tokenizer, data, max_length, **kwargs):
        prompt = data["prompt"]
        chosen = data["chosen"]
        rejected = data["rejected"]
        chosen_full_text = prompt + chosen
        rejected_full_text = prompt + rejected
        chosen_tokenized = tokenizer(
            chosen_full_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        rejected_tokenized = tokenizer(
            rejected_full_text,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_input_ids": chosen_tokenized["input_ids"],
            "chosen_attention_mask": chosen_tokenized["attention_mask"],
            "rejected_input_ids": rejected_tokenized["input_ids"],
            "rejected_attention_mask": rejected_tokenized["attention_mask"],
        }

    async def setup(self):
        """Load and prepare the dataset of prompts, potentially by generating them via LLM."""

        generated_prompts = []
        if self.server:
            logger.info(
                f"Attempting to generate initial prompts for {self.name} using LLM..."
            )
            try:
                prompt_list_completion = await self.server.chat_completion(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that strictly follows formatting instructions.",
                        },
                        {"role": "user", "content": PROMPT_GENERATION_MASTER_PROMPT},
                    ],
                    n=1,
                    max_tokens=self.config.prompt_generation_max_tokens,
                    temperature=self.config.prompt_generation_temperature,
                )
                response_text = (
                    prompt_list_completion.choices[0].message.content.strip()
                    if prompt_list_completion.choices
                    else ""
                )
                logger.debug(f"LLM response for prompt generation: {response_text}")

                # Attempt to parse the string as a Python list of dictionaries
                try:
                    parsed_list = ast.literal_eval(response_text)
                    if isinstance(parsed_list, list) and all(
                        isinstance(item, dict) and "prompt" in item
                        for item in parsed_list
                    ):
                        generated_prompts = parsed_list
                        logger.info(
                            f"Successfully generated and parsed {len(generated_prompts)} prompts from LLM."
                        )
                    else:
                        logger.warning(
                            "LLM response for prompt generation was not a valid list of "
                            "prompt dictionaries. Using fallback."
                        )
                except (SyntaxError, ValueError) as e:
                    logger.warning(
                        f"Error parsing LLM response for prompt generation: {e}. Using fallback."
                    )
            except Exception as e:
                logger.error(
                    f"Error calling LLM for prompt generation: {e}. Using fallback."
                )
        else:
            logger.warning(
                f"LLM server not available for {self.name}. Using fallback prompts."
            )

        if (
            not generated_prompts
            or not isinstance(generated_prompts, list)
            or len(generated_prompts) < 10
        ):
            logger.info(f"Using fallback static prompt list for {self.name}.")
            generated_prompts = [
                {"prompt": "What are your thoughts on the future of renewable energy?"},
                {
                    "prompt": "If you could travel anywhere in the world, where would you go and why?"
                },
                {"prompt": "What's a book or movie that has deeply impacted you?"},
                {
                    "prompt": "Can you describe a complex scientific concept in simple terms?"
                },
                {"prompt": "What's a common misconception people have about AI?"},
                {
                    "prompt": "How do you think technology will change our daily lives in the next 20 years?"
                },
                {
                    "prompt": "What's a piece of advice you would give to someone learning a new skill?"
                },
                {
                    "prompt": "If you could have a conversation with any historical figure, who would it be?"
                },
                {"prompt": "What does 'creativity' mean to you as an AI?"},
                {"prompt": "Can you tell me a joke?"},
            ]
            if (
                len(generated_prompts) > 10
            ):  # Ensure we only use 10 if more are in fallback
                generated_prompts = generated_prompts[:10]

        self.prompt_dataset = generated_prompts
        if self.config.shuffle_dataset:
            random.shuffle(self.prompt_dataset)
        self.iter = 0
        logger.info(
            f"Initialized prompt dataset with {len(self.prompt_dataset)} examples for {self.name}."
        )

    async def get_next_item(self) -> Item:
        """
        Returns the next prompt from the dataset.
        Chosen and rejected responses will be generated dynamically.
        """
        if not self.prompt_dataset or self.iter >= len(self.prompt_dataset):
            await self.setup()  # This will now potentially call the LLM to generate prompts

        if not self.prompt_dataset:  # Should be populated by setup, even with fallback
            logger.error(
                f"Prompt dataset is STILL empty after setup in {self.name}. This is unexpected."
            )
            # Provide an absolute failsafe prompt
            return ("Failsafe prompt: What is 1+1?", "", "")

        entry = self.prompt_dataset[self.iter % len(self.prompt_dataset)]
        self.iter += 1
        return (entry["prompt"], "", "")

    async def collect_trajectories(
        self, items: List[Item]  # Changed to accept a list of items
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        """
        Receives a prompt, generates chosen and rejected responses using an LLM,
        then tokenizes for DPO.
        Assumes group_size is 1 for this DPO setup.
        """
        if not items:
            logger.warning("collect_trajectories received an empty list of items.")
            return None, []

        # item = items[0]
        prompt_str, _, _ = items

        if not self.server:
            logger.error(
                f"LLM server not configured or available for {self.name}. Cannot generate responses."
            )
            return None, []

        if not self.tokenizer:
            logger.error(
                f"Tokenizer not available in {self.name}. Attempting to initialize."
            )
            if (
                self.testing
                and hasattr(self.config, "tokenizer_name")
                and self.config.tokenizer_name
            ):
                try:
                    from transformers import AutoTokenizer

                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.config.tokenizer_name
                    )
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    logger.info(
                        f"Fallback tokenizer '{self.config.tokenizer_name}' initialized."
                    )
                except Exception as e:
                    logger.error(f"Failed to initialize fallback tokenizer: {e}")
                    return None, []
            else:
                return None, []

        try:
            # Generate Chosen Response
            logger.debug(f"Generating CHOSEN for prompt: {prompt_str[:100]}...")
            chosen_completion = await self.server.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_CHOSEN},
                    {"role": "user", "content": prompt_str},
                ],
                n=1,
                max_tokens=self.config.chosen_max_tokens,
                temperature=self.config.chosen_temperature,
            )
            chosen_response_str = (
                chosen_completion.choices[0].message.content.strip()
                if chosen_completion.choices
                else ""
            )
            logger.debug(f"Generated CHOSEN: {chosen_response_str[:100]}...")

            # Generate Rejected Response
            logger.debug(f"Generating REJECTED for prompt: {prompt_str[:100]}...")
            rejected_completion = await self.server.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_REJECTED},
                    {"role": "user", "content": prompt_str},
                ],
                n=1,
                max_tokens=self.config.rejected_max_tokens,
                temperature=self.config.rejected_temperature,
            )
            rejected_response_str = (
                rejected_completion.choices[0].message.content.strip()
                if rejected_completion.choices
                else ""
            )
            logger.debug(f"Generated REJECTED: {rejected_response_str[:100]}...")

            if not chosen_response_str or not rejected_response_str:
                logger.warning(
                    f"Failed to generate valid chosen or rejected response for prompt: {prompt_str}"
                )
                return None, []

            dpo_pair_data = {
                "prompt": prompt_str,
                "chosen": chosen_response_str,
                "rejected": rejected_response_str,
            }

            tokenized_output = self._tokenize_dpo_fn(
                self.tokenizer,
                dpo_pair_data,
                max_length=self.config.max_token_length,
            )

            scores_group = ScoredDataGroup()
            scores_group["prompt"] = [tokenized_output["prompt"]]
            scores_group["chosen"] = [tokenized_output["chosen"]]
            scores_group["rejected"] = [tokenized_output["rejected"]]
            scores_group["chosen_tokens"] = [tokenized_output["chosen_input_ids"]]
            scores_group["chosen_masks"] = [tokenized_output["chosen_attention_mask"]]
            scores_group["rejected_tokens"] = [tokenized_output["rejected_input_ids"]]
            scores_group["rejected_masks"] = [
                tokenized_output["rejected_attention_mask"]
            ]
            scores_group["tokens"] = [tokenized_output["chosen_input_ids"]]
            scores_group["masks"] = [tokenized_output["chosen_attention_mask"]]
            scores_group["scores"] = [1.0]
            scores_group["images"] = [None]
            scores_group["group_overrides"] = {"group_size": 1}

            return scores_group, []

        except Exception as e:
            logger.error(
                f"Error in collect_trajectories for {self.name} during DPO processing: {e}"
            )
            import traceback

            traceback.print_exc()
            return None, []

    async def score(self, rollout_group_data: Any) -> Optional[ScoredDataGroup]:
        if rollout_group_data and isinstance(rollout_group_data, ScoredDataGroup):
            return rollout_group_data
        logger.info(
            f"Data for {self.name} is not in ScoredDataGroup format or no data to score."
        )
        return None

    async def evaluate(self, *args, **kwargs):
        logger.info(
            f"Evaluation step called for {self.name}. No custom DPO evaluation implemented."
        )
        return None

    @classmethod
    def config_init(
        cls,
    ) -> Tuple[GSM8KConversationalStyleDPOEnvConfig, List[APIServerConfig]]:
        env_config = GSM8KConversationalStyleDPOEnvConfig(
            wandb_name="gsm8k_dynamic_conversational_dpo",
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            group_size=1,
            use_wandb=True,
            max_num_workers=1,
            rollout_server_url="http://localhost:8000",
            total_steps=100,
            batch_size=2,
            steps_per_eval=50,
            max_token_length=512,
            dataset_name="synthetic_conversational_style_prompts_via_gsm8k_env",
            shuffle_dataset=True,
            data_path_to_save_groups=None,
            chosen_temperature=0.7,
            chosen_max_tokens=150,
            rejected_temperature=0.4,
            rejected_max_tokens=50,
            prompt_generation_temperature=0.8,
            prompt_generation_max_tokens=1000,
        )
        # IMPORTANT: Configure your LLM inference server details here or via CLI/config file.
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
                base_url="https://inference-api.nousresearch.com/v1",
                api_key="sk-3DvYKMv_-BfAoDSTfdSvEQ",
                num_requests_for_eval=256,  # Copied from gsm8k_server.py
            )
        ]
        return env_config, server_configs


if __name__ == "__main__":
    GSM8KConversationalStyleDPOEnv.cli()
