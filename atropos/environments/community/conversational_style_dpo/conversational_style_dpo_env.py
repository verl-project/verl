import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple

from pydantic import Field

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer_dpo

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ConversationalStyleDPOEnvConfig(BaseEnvConfig):
    """Config for ConversationalStyleDPOEnv."""

    dataset_name: str = Field(
        "synthetic_conversational_style", description="Name of the dataset to use."
    )
    shuffle_dataset: bool = Field(True, description="Whether to shuffle the dataset")
    # Add any other environment-specific configurations here if needed


class ConversationalStyleDPOEnv(BaseEnv):
    name = "conversational_style_dpo"
    name_config_cls = ConversationalStyleDPOEnvConfig

    def __init__(
        self,
        config: ConversationalStyleDPOEnvConfig,
        server_configs: Optional[
            List[APIServerConfig]
        ] = None,  # server_configs might not be needed if we don't query a model
        slurm=True,
        testing=False,
    ):
        # If you're not calling an external model server for generation in this specific DPO setup
        # (because chosen/rejected are pre-defined), you might not need server_configs.
        # For simplicity, we'll keep it but not use it actively in this example.
        # If server_configs is None and BaseEnv requires it, initialize it as an empty list.
        resolved_server_configs = server_configs if server_configs is not None else []
        super().__init__(config, resolved_server_configs, slurm, testing)
        self.config: ConversationalStyleDPOEnvConfig = (
            config  # Ensure type for self.config
        )
        self.dataset: List[Dict[str, str]] = []
        self.iter: int = 0

    async def setup(self):
        """Load and prepare the synthetic dataset."""
        # Synthetic dataset: (prompt, chosen_response, rejected_response)
        # Chosen responses are more engaging, empathetic, or clear.
        # Rejected responses are blunt, generic, or less helpful.
        self.synthetic_data = [
            {
                "prompt": "I'm feeling a bit down today.",
                "chosen": "I'm sorry to hear that. Sometimes a little self-care can help. "
                "What's one small thing you could do for yourself right now?",
                "rejected": "Okay.",
            },
            {
                "prompt": "Can you explain how photosynthesis works?",
                "chosen": "Certainly! Photosynthesis is a fascinating process where plants use "
                "sunlight, water, and carbon dioxide to create their own food (glucose) and "
                "release oxygen. Think of it like a plant's kitchen!",
                "rejected": "Plants make food from light.",
            },
            {
                "prompt": "I'm excited about my new project!",
                "chosen": "That's fantastic news! Tell me more about it - "
                "what are you most looking forward to?",
                "rejected": "Good for you.",
            },
            {
                "prompt": "What's the weather like?",
                "chosen": "I can't check the real-time weather, but I hope it's pleasant where "
                "you are! If you'd like, you can tell me your location, and I could try to "
                "give you a general idea based on typical patterns, or help you find a "
                "weather service.",
                "rejected": "I don't know.",
            },
            {
                "prompt": "I'm having trouble understanding this concept.",
                "chosen": "I can understand that some concepts can be tricky! Could you tell me "
                "which part is confusing you? Maybe we can break it down together.",
                "rejected": "Read the manual.",
            },
        ]
        self.dataset = self.synthetic_data
        if self.config.shuffle_dataset:
            random.shuffle(self.dataset)
        self.iter = 0
        logger.info(f"Loaded synthetic dataset with {len(self.dataset)} examples.")

    async def get_next_item(self) -> Item:
        """
        Returns the next item from the dataset.
        For DPO, an "item" will be a tuple of (prompt, chosen_response, rejected_response).
        The BaseEnv expects (prompt_tuple, gold_answer, optional_extra_data),
        so we'll adapt our DPO item to this structure.
        The prompt_tuple will be the actual prompt.
        The gold_answer will be the chosen response.
        The optional_extra_data will be the rejected response.
        """
        if not self.dataset or self.iter >= len(self.dataset):
            await self.setup()  # Re-setup if dataset is exhausted or not loaded

        if not self.dataset:  # Still no dataset after setup
            logger.error("Dataset is empty even after setup.")
            # Return a fallback item or raise an error
            # For now, let's create a dummy item to avoid crashing, but this should be handled
            fallback_prompt = tuple(
                [frozenset({"role": "user", "content": "Fallback prompt"}.items())]
            )
            return (fallback_prompt, "Fallback chosen", "Fallback rejected")

        entry = self.dataset[self.iter % len(self.dataset)]
        self.iter += 1

        # For DPO, the direct strings are often more useful for tokenization with tokenize_for_trainer_dpo

        # We will pass the raw strings to collect_trajectories and handle tokenization there.
        # For Item, we can simplify or adjust based on how BaseEnv uses it.
        # Let's pass the prompt string directly as the first element of the "prompt_tuple".
        # The "gold_answer" will be the chosen response, and "extra_data" the rejected one.

        # Adapting to Item: Tuple[prompt_tuple, gold_answer_str, any_extra_data]
        # prompt_tuple: Tuple[FrozenSet[Tuple[str, str]], ...]
        # For DPO, the core elements are prompt, chosen, rejected. We'll pass them directly.
        return (entry["prompt"], entry["chosen"], entry["rejected"])

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[Optional[ScoredDataGroup], List[Item]]:
        """
        Processes a single DPO item (prompt, chosen, rejected) and prepares it for the trainer.
        Since we are not querying a model to generate responses (we have them),
        this method will directly use the provided chosen and rejected responses.
        """
        prompt_str, chosen_response_str, rejected_response_str = item

        # For DPO, we typically need to tokenize:
        # 1. Prompt + Chosen
        # 2. Prompt + Rejected
        # The `tokenize_for_trainer_dpo` function should handle this.
        # It usually expects (tokenizer, prompt, chosen, rejected)

        # We don't need to call self.server.chat_completion here as we have the data.
        # We directly prepare the ScoredDataGroup.

        # Create a dummy GameHistory if parts of the system expect it.
        # For DPO, the history might be just the prompt and the respective response.
        # This part might need adjustment based on how `tokenize_for_trainer_dpo`
        # and the DPO trainer expect the input.

        # Let's assume `tokenize_for_trainer_dpo` takes the raw strings.
        # The output of `tokenize_for_trainer_dpo` is expected to be a dictionary like:
        # {
        #     "chosen_tokens": [...], "chosen_masks": [...],
        #     "rejected_tokens": [...], "rejected_masks": [...]
        # }
        # or perhaps including prompt tokens as well.
        # For simplicity, we'll construct a ScoredDataGroup directly.

        # We need to ensure the tokenizer is available. It's set in BaseEnv.
        if not self.tokenizer:
            logger.error("Tokenizer not available. Cannot process DPO pair.")
            return None, []

        try:
            # Note: The actual structure of what `tokenize_for_trainer_dpo` returns
            # and how it's used in `ScoredDataGroup` is crucial.
            # This is a common pattern for DPO data preparation.
            # The function would typically create sequences like:
            #   <prompt_tokens><chosen_response_tokens>
            #   <prompt_tokens><rejected_response_tokens>
            # And corresponding attention masks. The loss is usually calculated only on response tokens.

            # Constructing the input for tokenize_for_trainer_dpo
            # It usually takes a list of dictionaries or a specific structure.
            # Let's assume it takes a dictionary with prompt, chosen, rejected.
            dpo_pair_data = {
                "prompt": prompt_str,
                "chosen": chosen_response_str,
                "rejected": rejected_response_str,
            }

            # This function needs to be defined or imported correctly.
            # It should handle the tokenization for DPO, creating chosen and rejected sequences.
            tokenized_output = tokenize_for_trainer_dpo(
                self.tokenizer,
                dpo_pair_data,  # Or (self.tokenizer, prompt_str, chosen_response_str, rejected_response_str)
                # depending on its signature.
                max_length=self.config.max_token_length,  # Ensure this config is available
                # Add other necessary args for tokenize_for_trainer_dpo
            )

            scores = ScoredDataGroup()
            # These keys depend on what your DPO trainer expects.
            # Common keys for DPO batches include:
            # - prompt_input_ids, prompt_attention_mask
            # - chosen_input_ids, chosen_attention_mask, chosen_labels
            # - rejected_input_ids, rejected_attention_mask, rejected_labels
            # `tokenize_for_trainer_dpo` should produce these.

            # Let's assume tokenize_for_trainer_dpo returns a dict with at least:
            # 'chosen_input_ids', 'chosen_attention_mask',
            # 'rejected_input_ids', 'rejected_attention_mask'
            # 'prompt_input_ids' (optional, might be part of chosen/rejected)

            # Adapting to a simpler ScoredDataGroup structure for this example:
            # We'll store the direct outputs of a hypothetical tokenize_for_trainer_dpo
            scores["chosen_tokens"] = [tokenized_output["chosen_input_ids"]]
            scores["chosen_masks"] = [tokenized_output["chosen_attention_mask"]]
            scores["rejected_tokens"] = [tokenized_output["rejected_input_ids"]]
            scores["rejected_masks"] = [tokenized_output["rejected_attention_mask"]]
            # Optionally, if prompts are tokenized separately:
            if "prompt_input_ids" in tokenized_output:
                scores["prompt_tokens"] = [tokenized_output["prompt_input_ids"]]
                scores["prompt_masks"] = [
                    tokenized_output.get("prompt_attention_mask")
                ]  # Handle if mask isn't there

            # DPO doesn't use a single "score" like reward models. The "reward" is implicit
            # in the preference (chosen > rejected). So, the "scores" field in ScoredDataGroup
            # might not be directly used or could be set to a placeholder if required by the trainer.
            scores["scores"] = [
                1.0
            ]  # Placeholder, DPO loss handles preference directly

            # Images are not used in this environment
            scores["images"] = [None]

            # Ensure group size logic if processing multiple items for a group
            # For this example, we process one item at a time.
            # If self.config.group_size > 1, you'd aggregate multiple tokenized_outputs
            # before returning the ScoredDataGroup.

            return scores, []  # No items to backlog

        except Exception as e:
            logger.error(f"Error in collect_trajectories during DPO processing: {e}")
            import traceback

            traceback.print_exc()
            return None, []

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        """
        This method is typically for scoring model generations against a gold answer or reward model.
        For DPO with a static dataset, the "scoring" is implicit in the chosen/rejected pair.
        The main processing happens in `collect_trajectories` which prepares tokenized pairs.
        So, this method might not be directly used if `collect_trajectories` already
        returns a `ScoredDataGroup`. If the main loop calls `score` after `collect_trajectories`,
        it might just pass through the data or perform some final aggregation.

        If `collect_trajectories` returns the raw data (prompt, chosen, rejected)
        and `score` is responsible for tokenization, then the logic from
        `collect_trajectories` related to tokenization would move here.

        Assuming `collect_trajectories` prepares the `ScoredDataGroup` with tokenized DPO pairs:
        """
        if rollout_group_data and isinstance(rollout_group_data, ScoredDataGroup):
            # If `collect_trajectories` already produced the ScoredDataGroup
            return rollout_group_data
        elif rollout_group_data and isinstance(rollout_group_data, list):
            # If `collect_trajectories` returned a list of items to be scored/tokenized here.
            # This would mean moving the tokenization logic from `collect_trajectories` to here.
            # For now, let's assume the former.
            logger.warning(
                "`score` method received a list; expecting ScoredDataGroup for pre-processed DPO."
            )
            # Fallback: if you need to process a list of (prompt, chosen, rejected) tuples here:
            # all_chosen_tokens = []
            # ... and so on, then call tokenize_for_trainer_dpo for each item.
            # This depends on the design of BaseEnv's main loop.
            return None

        logger.info("No data to score or data is already in ScoredDataGroup format.")
        return None

    async def evaluate(self, *args, **kwargs):
        """
        Evaluation for DPO might involve comparing the DPO-trained model's preferences
        against a held-out set of preferred/rejected pairs or other metrics.
        For this basic environment, we'll skip custom evaluation.
        """
        logger.info(
            "Evaluation step called. No custom DPO evaluation implemented in this basic environment."
        )
        # You could load a separate test set of (prompt, chosen, rejected)
        # and see if the model assigns higher logprobs to chosen than rejected.
        return None

    @classmethod
    def config_init(
        cls,
    ) -> Tuple[ConversationalStyleDPOEnvConfig, List[APIServerConfig]]:
        """
        Provides a default configuration for this environment.
        """
        env_config = ConversationalStyleDPOEnvConfig(
            wandb_name="conversational_style_dpo",  # For logging if wandb is used
            tokenizer_name="gpt2",  # Choose an appropriate tokenizer
            group_size=4,  # Number of DPO pairs to process in a "group" or "batch"
            use_wandb=False,  # Enable or disable wandb
            max_num_workers=1,  # Number of parallel workers for data collection (if applicable)
            rollout_server_url="http://localhost:8000",  # Corrected URL
            total_steps=100,  # Total DPO training steps (or epochs over the dataset)
            batch_size=2,  # DPO training batch size (distinct from group_size for data collection)
            steps_per_eval=50,
            max_token_length=512,  # Max length for tokenized sequences
            dataset_name="synthetic_conversational_style",
            shuffle_dataset=True,
        )

        server_configs = []  # Simplified as discussed

        return env_config, server_configs


if __name__ == "__main__":
    # This allows running the environment directly, e.g., to test data loading and processing.
    # The `BaseEnv.cli()` method usually sets up and runs the environment's main loop.
    # For DPO, the "main loop" might involve iterating through the dataset,
    # tokenizing pairs, and perhaps logging them or yielding them to a trainer.

    # To make this runnable and test the data processing:
    async def main_test():
        config, server_configs_list = ConversationalStyleDPOEnv.config_init()

        # Manually override tokenizer for local testing if needed and not using a server
        # that provides it, or if the default in BaseEnvConfig isn't what you want for DPO.
        # config.tokenizer_name = "EleutherAI/pythia-70m" # Example
        config.tokenizer_name = "distilgpt2"  # A small, fast tokenizer for testing
        config.group_size = (
            1  # Process one DPO item at a time for simplicity in this test
        )
        config.use_wandb = False

        env = ConversationalStyleDPOEnv(
            config=config, server_configs=server_configs_list, slurm=False, testing=True
        )

        # Initialize tokenizer (BaseEnv usually does this)
        from transformers import AutoTokenizer

        env.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        if env.tokenizer.pad_token is None:
            env.tokenizer.pad_token = env.tokenizer.eos_token

        print("Setting up environment...")
        await env.setup()
        print(f"Dataset size: {len(env.dataset)}")

        if not env.dataset:
            print("No data loaded. Exiting.")
            return

        print("Simulating DPO data processing for a few items...")
        for i in range(min(len(env.dataset), 3)):  # Test with a few items
            print(f"--- Item {i+1} ---")
            item = await env.get_next_item()
            if item:
                prompt, chosen, rejected = item
                print(f"Prompt: {prompt}")
                print(f"Chosen: {chosen}")
                print(f"Rejected: {rejected}")

                # Simulate calling collect_trajectories which should do the DPO tokenization
                # In a real run, this would be part of the BaseEnv's loop
                # For testing, we call it directly.
                # The `tokenize_for_trainer_dpo` function needs to exist and be importable.
                # Let's create a placeholder for it here for the test to run.
                global tokenize_for_trainer_dpo

                def placeholder_tokenize_for_dpo(tokenizer, data, max_length, **kwargs):
                    # This is a simplified placeholder. A real one would handle complex tokenization,
                    # padding, truncation, and creating labels for DPO loss.
                    prompt = data["prompt"]
                    chosen = data["chosen"]
                    rejected = " ".join(
                        data["rejected"].split()[: max_length // 3]
                    )  # Basic truncation

                    chosen_text = f"{prompt} {chosen}"
                    rejected_text = f"{prompt} {rejected}"

                    chosen_tok = tokenizer(
                        chosen_text,
                        truncation=True,
                        padding="max_length",
                        max_length=max_length,
                        return_tensors="pt",
                    )
                    rejected_tok = tokenizer(
                        rejected_text,
                        truncation=True,
                        padding="max_length",
                        max_length=max_length,
                        return_tensors="pt",
                    )

                    # A real DPO tokenizer would also prepare labels, where only response tokens are unmasked.
                    # For simplicity, we're just returning input_ids and attention_mask.
                    return {
                        "chosen_input_ids": chosen_tok["input_ids"].squeeze().tolist(),
                        "chosen_attention_mask": chosen_tok["attention_mask"]
                        .squeeze()
                        .tolist(),
                        "rejected_input_ids": rejected_tok["input_ids"]
                        .squeeze()
                        .tolist(),
                        "rejected_attention_mask": rejected_tok["attention_mask"]
                        .squeeze()
                        .tolist(),
                        # "prompt_input_ids": ..., # Optionally
                    }

                tokenize_for_trainer_dpo = placeholder_tokenize_for_dpo

                scored_data_group, _ = await env.collect_trajectories(item)

                if scored_data_group:
                    print("Tokenized DPO Pair (first item in group):")
                    print(
                        f"  Chosen Tokens (IDs): {scored_data_group['chosen_tokens'][0][:20]}..."
                    )  # Print first 20
                    # print(f"  Chosen Masks: {scored_data_group['chosen_masks'][0][:20]}...")
                    print(
                        f"  Rejected Tokens (IDs): {scored_data_group['rejected_tokens'][0][:20]}..."
                    )
                    # print(f"  Rejected Masks: {scored_data_group['rejected_masks'][0][:20]}...")
                    # print(f"  Scores: {scored_data_group['scores']}")
                else:
                    print("  Failed to process DPO pair.")
            else:
                print("Failed to get item.")

        print("--- End of Test ---")

    # To run the CLI, you would typically not need the main_test async function,
    # but BaseEnv.cli() would handle it.
    # For this example, we provide a way to test the data processing logic.
    # If you want to run with the actual CLI:
    # ConversationalStyleDPOEnv.cli()
    # But ensure `tokenize_for_trainer_dpo` is correctly implemented and importable in that context.

    # Running the test:
    if __name__ == "__main__":
        # Define a placeholder tokenize_for_trainer_dpo if it's not available globally
        # This is often part of a utils library.
        # Use the imported tokenize_for_trainer_dpo function

        asyncio.run(main_test())
