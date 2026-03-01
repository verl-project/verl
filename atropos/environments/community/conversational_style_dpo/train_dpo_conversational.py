import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
)
from trl import DPOTrainer

# Import your custom environment
# The import below assumes this script is in the same directory as conversational_style_dpo_env.py
from .conversational_style_dpo_env import (
    ConversationalStyleDPOEnv,
    ConversationalStyleDPOEnvConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class ScriptArguments:
    """
    Arguments for the DPO training script.
    """

    model_name_or_path: str = field(
        default="distilgpt2",
        metadata={"help": "The model name or path to load from."},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The tokenizer name or path. Defaults to model_name_or_path."
        },
    )
    beta: float = field(default=0.1, metadata={"help": "The beta factor in DPO loss."})
    max_prompt_length: int = field(
        default=256, metadata={"help": "Max length for prompts."}
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Max length for chosen/rejected responses including prompt."},
    )
    # Add any other TRL DPOTrainer arguments or TrainingArguments here if needed
    # For example, learning_rate, per_device_train_batch_size, num_train_epochs etc.
    # will be part of TrainingArguments.


async def get_dataset_from_env(
    env_config: ConversationalStyleDPOEnvConfig,
) -> Dataset:
    """
    Initializes the environment and extracts the synthetic dataset
    in the format required by DPOTrainer (list of dicts with prompt, chosen, rejected).
    """
    # We don't need server_configs if the env doesn't use them for static dataset loading
    env = ConversationalStyleDPOEnv(config=env_config, server_configs=[], testing=True)
    await env.setup()  # This loads env.synthetic_data

    # env.synthetic_data is already a List[Dict[str, str]] with "prompt", "chosen", "rejected"
    # Convert it to Hugging Face Dataset
    # Check if data is loaded
    if not env.dataset:
        raise ValueError(
            "Dataset is empty after environment setup. Check ConversationalStyleDPOEnv."
        )

    # The DPOTrainer expects columns named "prompt", "chosen", "rejected"
    # The synthetic_data in your environment is already in this format.
    # Example: [{"prompt": "...", "chosen": "...", "rejected": "..."}]
    hf_dataset = Dataset.from_list(list(env.dataset))  # Ensure it's a fresh list copy

    # Log a sample to verify
    if len(hf_dataset) > 0:
        logger.info(f"Sample from dataset: {hf_dataset[0]}")
    else:
        logger.warning("Dataset created from environment is empty!")

    return hf_dataset


def main():
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    if script_args.tokenizer_name_or_path is None:
        script_args.tokenizer_name_or_path = script_args.model_name_or_path

    # --- 1. Initialize Environment and Get Dataset ---
    logger.info("Initializing environment to get dataset...")
    # Use the default config from your environment, but ensure tokenizer matches
    env_dpo_config, _ = ConversationalStyleDPOEnv.config_init()
    env_dpo_config.tokenizer_name = (
        script_args.tokenizer_name_or_path
    )  # Align tokenizer
    # You might want to adjust other env_dpo_config parameters if needed

    # Run the async function to get the dataset
    dataset = asyncio.run(get_dataset_from_env(env_dpo_config))
    logger.info(f"Loaded dataset with {len(dataset)} examples.")

    if len(dataset) == 0:
        logger.error("No data loaded. Exiting training.")
        return

    # --- 2. Load Tokenizer and Models ---
    logger.info(f"Loading tokenizer: {script_args.tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad token. Setting to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        # For some models, you might also need to set tokenizer.pad_token_id

    logger.info(f"Loading policy model: {script_args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        # low_cpu_mem_usage=True, # Can be helpful for large models
        # torch_dtype=torch.float16, # For mixed precision if GPU supports
    )

    # Reference model for DPO. If not provided, DPOTrainer will create a copy of the model.
    # For simplicity, we'll let DPOTrainer handle creating the reference model by not passing one.
    # If you wanted to load a different SFT model as reference, you would do:
    # model_ref = AutoModelForCausalLM.from_pretrained(...)
    model_ref = None
    logger.info(
        "Reference model will be a copy of the policy model (handled by DPOTrainer)."
    )

    # --- 3. Set up Training Arguments ---
    # Default DPO training arguments. You might want to customize these.
    if training_args.output_dir == "output_dir":  # Default value from TrainingArguments
        # Output directory will be relative to the script's current working directory when run.
        # If run from environments/hack0/conversational_style_dpo/, it will be ./dpo_conversational_trainer_results
        training_args.output_dir = "./dpo_conversational_trainer_results"

    # training_args.per_device_train_batch_size = 2 # Adjust as needed
    # training_args.num_train_epochs = 1 # Keep low for a quick test
    # training_args.gradient_accumulation_steps = 1
    # training_args.learning_rate = 5e-5
    # training_args.logging_steps = 10
    # training_args.save_steps = 50
    # training_args.report_to = "none" # "wandb" or "tensorboard" if you want to log

    logger.info(f"Training Arguments: {training_args}")

    # --- 4. Initialize DPOTrainer ---
    logger.info("Initializing DPOTrainer...")
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,  # If None, a copy of model is made
        args=training_args,
        beta=script_args.beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,  # Max length of prompt + response
        # peft_config=peft_config, # If using PEFT/LoRA
    )
    logger.info("DPOTrainer initialized.")

    # --- 5. Start Training ---
    logger.info("Starting DPO training...")
    dpo_trainer.train()
    logger.info("DPO training completed.")

    # --- 6. Save the Model (Optional) ---
    if training_args.should_save:  # Checks if any save_strategy is enabled
        output_save_dir = training_args.output_dir
        logger.info(f"Saving model to {output_save_dir}")
        dpo_trainer.save_model(output_save_dir)
        logger.info("Model saved.")
        # Also save the tokenizer
        tokenizer.save_pretrained(output_save_dir)
        logger.info("Tokenizer saved.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
