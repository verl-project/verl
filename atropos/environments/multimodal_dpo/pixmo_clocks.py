import base64
import io
import random
import re
import traceback
from typing import List, Optional, Tuple

from datasets import load_dataset

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import GameHistory, Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class ClockDatasetEnv(BaseEnv):
    name = "pixmo_clocks"
    name_config_cls = BaseEnvConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[GameHistory | None, List[Item]]:
        to_score: List[Tuple[GameHistory, str, Optional[str]]] = []
        to_backlog: List[Item] = []

        # Extract the base64 image
        base64_image = item[2]

        # Build system instruction and multimodal user message
        system_msg = {
            "role": "system",
            "content": (
                "You must submit your answer enclosed in <answer> tags, "
                "e.g., <answer>HH:MM</answer>"
            ),
        }
        user_prompt_text = "What time does the clock show?"
        user_msg_multimodal = {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        }

        messages = [system_msg, user_msg_multimodal]

        # Call chat completion
        chat_completions = await self.server.chat_completion(
            messages=messages,
            n=self.config.group_size,
            max_tokens=512,
            timeout=60,
        )

        # Prepare trajectories for scoring
        for choice in chat_completions.choices:
            # Use text-only prompt for history
            user_msg = {"role": "user", "content": user_prompt_text}
            assistant_msg = {"role": "assistant", "content": choice.message.content}
            history: GameHistory = (user_msg, assistant_msg)
            to_score.append((history, item[1], base64_image))

        to_postprocess = await self.score(to_score)

        return to_postprocess, to_backlog

    async def evaluate(self, *args, **kwargs):
        # No custom evaluation
        return

    async def setup(self):
        # Load the clock dataset
        self.dataset = load_dataset("junyeong-nero/clock-dataset")
        self.train = self.dataset["train"]
        self.iter = 0

    async def get_next_item(self) -> Item:

        try:
            entry = self.train[self.iter % len(self.train)]
            self.iter += 1

            text_prompt = "What time does the clock show"
            prompt = tuple(
                [frozenset({"role": "user", "content": text_prompt}.items())]
            )

            # Format gold answer
            hour = entry["hour"]
            minute = entry["minute"]
            gold_answer = f"<answer>{hour}:{minute:02d}</answer>"

            # Convert image to base64
            img = entry["image"]
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            image_bytes = buf.getvalue()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")

            return (prompt, gold_answer, base64_image)
        except Exception:
            traceback.print_exc()
            # Fallback
            fallback_prompt = tuple(
                [
                    frozenset(
                        {"role": "user", "content": "Please solve: 2 + 2 = ?"}.items()
                    )
                ]
            )
            return (fallback_prompt, "<answer>4</answer>", None)

    async def score(self, rollout_group_data) -> Optional[ScoredDataGroup]:
        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        scores["images"] = []
        random.shuffle(rollout_group_data)

        for item in rollout_group_data:
            out = tokenize_for_trainer(self.tokenizer, item[0])
            tokens = out["tokens"]
            masks = out["masks"]

            # Extract answers
            try:
                reply = item[0][-1]["content"]
                m_match = re.search(
                    r"<answer>\s*(.*?)\s*</answer>", reply, re.IGNORECASE
                )
                model_answer = m_match.group(1).strip() if m_match else reply.strip()

                gold = item[1]
                g_match = re.search(
                    r"<answer>\s*(.*?)\s*</answer>", gold, re.IGNORECASE
                )
                gold_answer = g_match.group(1).strip() if g_match else gold.strip()

                reward = model_answer == gold_answer
            except Exception:
                reward = False

            if len([i for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(1.0 if reward else -1.0)
            try:
                scores["images"].append(item[2])
            except Exception:
                scores["images"].append(None)

            if len(scores["tokens"]) >= self.config.group_size:
                break

        return scores

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        config = BaseEnvConfig(
            wandb_name="pixmo_clocks",
            tokenizer_name="Qwen/Qwen2-VL-2B-Instruct",
            group_size=8,
            use_wandb=True,
            max_num_workers=2,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=2048,
        )

        server_configs = [
            APIServerConfig(
                model_name="Qwen/Qwen2-VL-2B-Instruct",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=256,
            ),
        ]

        return config, server_configs


if __name__ == "__main__":
    ClockDatasetEnv.cli()
