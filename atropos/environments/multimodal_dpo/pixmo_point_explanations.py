import base64
import io
import random
import re
import traceback
from typing import List, Optional, Tuple

import requests
from datasets import load_dataset
from PIL import Image

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import GameHistory, Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class PixmoPointExplanationsEnv(BaseEnv):
    name = "pixmo_point_explanations"
    name_config_cls = BaseEnvConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def setup(self):
        # Load the pixmo-point-explanations dataset
        self.dataset = load_dataset("allenai/pixmo-point-explanations")
        self.train = self.dataset["train"]
        self.iter = 0

    async def get_next_item(self) -> Item:
        try:
            entry = self.train[self.iter % len(self.train)]
            self.iter += 1

            question = entry["question"]
            # Use the first inline text as the answer
            answer_text = entry["inline_text"][0]
            prompt = tuple([frozenset({"role": "user", "content": question}.items())])
            gold_answer = f"<answer>{answer_text}</answer>"

            # Load image from URL and convert to base64
            try:
                image_url = entry["image_url"]
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_bytes = buf.getvalue()
                base64_image = base64.b64encode(img_bytes).decode("utf-8")
            except Exception:
                base64_image = None

            return (prompt, gold_answer, base64_image)
        except Exception:
            traceback.print_exc()
            fallback = tuple(
                [
                    frozenset(
                        {"role": "user", "content": "Please solve: 2 + 2 = ?"}.items()
                    )
                ]
            )
            return (fallback, "<answer>4</answer>", None)

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[GameHistory | None, List[Item]]:
        to_score: List[Tuple[GameHistory, str, Optional[str]]] = []
        to_backlog: List[Item] = []

        prompt_tuple, gold, base64_image = item
        text_prompt = dict(prompt_tuple[0])["content"]

        system_msg = {
            "role": "system",
            "content": (
                "You must submit your answer enclosed in <answer> tags, "
                "e.g., <answer>YOUR_ANSWER</answer>"
            ),
        }
        user_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
            ],
        }

        # Only add image if we have a valid base64 image
        if base64_image:
            user_msg["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )
        messages = [system_msg, user_msg]

        # Call chat completion
        chat_completions = await self.server.chat_completion(
            messages=messages,
            n=self.config.group_size,
            max_tokens=512,
            timeout=60,
        )

        for choice in chat_completions.choices:
            user_hist = {"role": "user", "content": text_prompt}
            assistant_hist = {"role": "assistant", "content": choice.message.content}
            history: GameHistory = (user_hist, assistant_hist)
            to_score.append((history, gold, base64_image))

        to_postprocess = await self.score(to_score)

        return to_postprocess, to_backlog

    async def evaluate(self, *args, **kwargs):
        # No custom evaluation
        return

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

            try:
                reply = item[0][-1]["content"]
                m = re.search(r"<answer>\s*(.*?)\s*</answer>", reply, re.IGNORECASE)
                model_answer = m.group(1).strip() if m else reply.strip()

                gold = item[1]
                g = re.search(r"<answer>\s*(.*?)\s*</answer>", gold, re.IGNORECASE)
                gold_answer = g.group(1).strip() if g else gold.strip()

                reward = model_answer.lower() == gold_answer.lower()
            except Exception:
                reward = False

            # Filter out short examples
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
            wandb_name="pixmo_point_explanations",
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
    PixmoPointExplanationsEnv.cli()
