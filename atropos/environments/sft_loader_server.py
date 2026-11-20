import asyncio
from enum import Enum
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
from pydantic import Field

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, OpenaiConfig, ScoredDataGroup
from atroposlib.type_definitions import Item


class DatasetFormat(Enum):
    COMPLETION = "completion"
    PREFIXED_COMPLETION = "prefix_completion"
    OAI_FORMAT = "oai_format"
    SHAREGPT_FORMAT = "sharegpt_format"


class SFTConfig(BaseEnvConfig):
    dataset_name: str = Field(
        default="gsm8k",
        description="The name of the HF dataset to use for training",
    )
    dataset_format: DatasetFormat = Field(
        default=DatasetFormat.COMPLETION,
        description="The format of the dataset to use for training",
    )
    dataset_column_name: str = Field(
        default="text",
        description="The name of the column to use for training",
    )
    prefix_column_name: Optional[str] = Field(
        default=None,
        description="The name of the column to use for the prefix, if applicable",
    )
    mask_everything_but_assistant_answer: bool = Field(
        default=True,
        description="Whether to mask everything but the assistant answer, if applicable",
    )
    mask_everything_but_last_step: bool = Field(
        default=False,
        description="Whether to mask everything but the last step, if applicable",
    )
    max_sft_per_step: int = Field(
        default=-1,
        description="The maximum number of SFTs to do per step, if -1 just sends all the data",
    )
    add_every_n_steps: int = Field(
        default=1,
        description="Only add SFT data every n steps",
    )


class SFTEnv(BaseEnv):

    name = "sft_loader"

    def __init__(
        self,
        config: SFTConfig,
        server_configs: List[OpenaiConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        # Add tracking for wandb visualizations
        self.rollouts_for_wandb = []
        self.completion_lengths = []
        self.idx = 0
        self.last_step = -1

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[OpenaiConfig]]:
        env_config = SFTConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=2048,
            ensure_scores_are_not_same=False,
            wandb_name="sft",
            dataset_name="AlignmentLab-AI/open-instruct-sharegpt",
            dataset_format=DatasetFormat.SHAREGPT_FORMAT,
            dataset_column_name="conversations",
            inference_weight=-1,
            max_sft_per_step=8,
        )
        server_configs = [
            OpenaiConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=256,
            ),
        ]

        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}
        wandb_metrics["train/idx"] = self.idx
        wandb_metrics["train/epoch"] = self.idx // len(self.train)
        # Call the parent method to handle the server metrics
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        self.train = load_dataset(self.config.dataset_name, split="train").shuffle(
            seed=42
        )
        self.iter = 0

    async def get_next_item(self) -> Item:
        next_item = self.train[self.idx % len(self.train)]
        # self.idx += 1
        return next_item

    async def evaluate(self, *args, **kwargs):
        pass

    async def add_train_workers(self):
        pass

    async def format_item(self, item: Item):
        group = ScoredDataGroup()
        group["tokens"] = list()
        group["masks"] = list()
        group["messages"] = list()
        group["scores"] = [1]
        group["advantages"] = [1]
        # GRPO/RLOO should natively work if they don't apply mean/std to a 1 item group and have 0 KL term,
        # but in case you are using something different, e.g. PPO,
        # we'll send an override to the server to tell it to do a SFT step.
        group["group_overrides"] = {"sft": True}
        group["overrides"] = [{"sft": True}]
        if self.config.dataset_format == DatasetFormat.COMPLETION:
            tokens = self.tokenizer.encode(item[self.config.dataset_column_name])
            group["tokens"].append(tokens)
            group["messages"].append(item[self.config.dataset_column_name])
            group["masks"].append(tokens)
        elif self.config.dataset_format == DatasetFormat.PREFIXED_COMPLETION:
            tokens = self.tokenizer.encode(
                item[self.config.prefix_column_name]
                + item[self.config.dataset_column_name]
            )
            prefix_tokens = self.tokenizer.encode(item[self.config.prefix_column_name])
            if prefix_tokens[-1] == self.tokenizer.eos_token_id:
                prefix_tokens = prefix_tokens[:-1]
            group["tokens"].append(tokens)
            group["messages"].append(
                item[self.config.prefix_column_name]
                + item[self.config.dataset_column_name]
            )
            group["masks"].append(
                [-100 for _ in range(len(prefix_tokens))] + tokens[len(prefix_tokens) :]
            )
        else:
            if self.config.dataset_format == DatasetFormat.OAI_FORMAT:
                messages = item[self.config.dataset_column_name]
            elif self.config.dataset_format == DatasetFormat.SHAREGPT_FORMAT:
                messages = item[self.config.dataset_column_name]
                # reformat to oai format
                role_map = {"human": "user", "gpt": "assistant", "system": "system"}
                messages = [
                    {"role": role_map[message["from"]], "content": message["value"]}
                    for message in messages
                ]
            group["messages"].append(messages)
            tokens = self.tokenizer.apply_chat_template(messages)
            group["tokens"].append(tokens)
            if self.config.mask_everything_but_last_step:
                masks = [
                    -100
                    for _ in self.tokenizer.apply_chat_template(
                        messages[:-1], add_generation_prompt=True
                    )
                ]
                masks.extend(tokens[len(masks) :])
                group["masks"].append(masks)
            elif self.config.mask_everything_but_assistant_answer:
                # find the assistant answer in the chats
                masks = []
                for i, msg in enumerate(messages):
                    if msg["role"] == "assistant":
                        prefix_tokens = self.tokenizer.apply_chat_template(
                            messages[:i], add_generation_prompt=True
                        )
                        masks.extend(
                            [-100 for _ in range(len(prefix_tokens[len(masks) :]))]
                        )
                        curr_assist_tokens = self.tokenizer.apply_chat_template(
                            messages[: i + 1]
                        )
                        masks.extend(curr_assist_tokens[len(masks) :])
                group["masks"].append(masks)
        return group

    async def env_step_checks(self):
        # Check if we need to run an eval or log...
        if self.curr_step != self.status_dict["current_step"]:
            next_step = self.status_dict["current_step"]
            if next_step % self.config.add_every_n_steps == 0:
                to_send = (
                    self.config.max_batches_offpolicy * self.config.batch_size
                ) - (self.status_dict["queue_size"])
                if self.config.max_sft_per_step != -1:
                    to_send = min(to_send, self.config.max_sft_per_step)
                self.items_sent_this_step = to_send
                if to_send > 0:
                    formatted_items = list()
                    for _ in range(to_send):
                        item = await self.get_next_item()
                        self.idx += 1
                        formatted_items.append(await self.format_item(item))
                    await asyncio.gather(
                        *[
                            self.handle_send_to_api(formatted_item, None)
                            for formatted_item in formatted_items
                        ]
                    )
        await super().env_step_checks()


async def checkout_formatting():
    # SFTEnv.cli()
    env = SFTEnv(
        config=SFTConfig(
            dataset_name="AlignmentLab-AI/open-instruct-sharegpt",
            dataset_format=DatasetFormat.SHAREGPT_FORMAT,
            dataset_column_name="conversations",
        ),
        server_configs=[
            OpenaiConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=256,
            )
        ],
        slurm=False,
        testing=True,
    )
    await env.setup()
    item = await env.get_next_item()
    item = {
        "conversations": [
            {"from": "human", "value": "What is the capital of France?"},
            {"from": "gpt", "value": "The capital of France is Paris."},
            {"from": "human", "value": "What is the capital of Germany?"},
            {"from": "gpt", "value": "The capital of Germany is Berlin."},
        ]
    }
    formatted_item = await env.format_item(item)
    print(formatted_item)
    print(env.tokenizer.decode(formatted_item["tokens"][0]))
    print("--- mask decoding ---")
    print(env.tokenizer.decode([x for x in formatted_item["masks"][0] if x != -100]))


if __name__ == "__main__":
    SFTEnv.cli()
