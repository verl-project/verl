import base64
import csv
import io
import os
import random
import sys
import traceback
from typing import List, Optional, Tuple

from PIL import Image
from pydantic import Field

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, OpenaiConfig, ScoredDataGroup
from atroposlib.type_definitions import GameHistory, Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class UFCImageEnvConfig(BaseEnvConfig):
    """Configuration for the UFC Image Environment"""

    fighter_stats_path: str = Field(
        os.path.join(os.path.dirname(__file__), "fighter_stats.csv"),
        description="Path to fighter stats CSV",
    )
    fight_data_path: str = Field(
        os.path.join(os.path.dirname(__file__), "large_dataset.csv"),
        description="Path to large fight dataset CSV",
    )
    image_folder: str = Field(
        os.path.join(os.path.dirname(__file__), "fighter_images"),
        description="Path to fighter images folder",
    )
    max_steps: int = Field(1, description="Only one step per fight prediction")
    temperature: float = Field(0.7, description="Temperature for generation diversity")
    top_p: float = Field(0.95, description="Top p for nucleus sampling")


class UFCImageEnv(BaseEnv):
    """UFC Fight Prediction Environment using only fighter images"""

    name = "ufc_image_predictor"
    env_config_cls = UFCImageEnvConfig

    def __init__(
        self,
        config: UFCImageEnvConfig,
        server_configs: List[OpenaiConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.fighter_stats = {}
        self.fight_data = []
        self.current_index = 0
        self.inference_server = self.server.servers[
            0
        ]  # Get first server as inference server

    async def setup(self):
        """Load the fighter stats and fight data"""
        try:
            print("Loading fighter stats from:", self.config.fighter_stats_path)
            with open(self.config.fighter_stats_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.fighter_stats = {row["name"]: row for row in reader}
            print(f"Loaded stats for {len(self.fighter_stats)} fighters")

            print("Loading fight data from:", self.config.fight_data_path)
            with open(self.config.fight_data_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.fight_data = list(reader)
            print(f"Loaded {len(self.fight_data)} fights")

            # Filter out fights where either fighter's image is missing
            filtered_fights = []
            missing_images = set()  # Track unique missing images
            for fight in self.fight_data:
                r_fighter = fight["r_fighter"]
                b_fighter = fight["b_fighter"]

                # Convert names to image filename format
                r_slug = r_fighter.lower().replace(" ", "-")
                b_slug = b_fighter.lower().replace(" ", "-")

                r_image_path = os.path.join(self.config.image_folder, f"{r_slug}.jpg")
                b_image_path = os.path.join(self.config.image_folder, f"{b_slug}.jpg")

                if os.path.exists(r_image_path) and os.path.exists(b_image_path):
                    filtered_fights.append(fight)
                else:
                    if not os.path.exists(r_image_path):
                        missing_images.add(r_fighter)
                    if not os.path.exists(b_image_path):
                        missing_images.add(b_fighter)

            if missing_images:
                print(
                    f"\nMissing images for {len(missing_images)} fighters. These fights will be skipped."
                )

            self.fight_data = filtered_fights
            print(f"Filtered to {len(self.fight_data)} fights with complete image sets")

        except Exception as e:
            print(f"Error loading data: {e}")
            traceback.print_exc()
            sys.exit(1)

    def get_fighter_image(self, fighter_name):
        """Convert fighter name to image path and return base64 encoded image"""
        try:
            # Convert name to slug format
            slug = fighter_name.lower().replace(" ", "-")

            image_path = os.path.join(self.config.image_folder, f"{slug}.jpg")
            if not os.path.exists(image_path):
                return None

            # Convert image to base64
            with Image.open(image_path) as img:
                # Convert RGBA to RGB if necessary
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                image_bytes = buf.getvalue()
                return base64.b64encode(image_bytes).decode("utf-8")
        except Exception as e:
            print(f"Error getting image for {fighter_name}: {e}")
            return None

    async def get_next_item(self) -> Optional[Item]:
        """Get the next fight from the dataset"""
        try:
            if self.current_index >= len(self.fight_data):
                return None
            fight = self.fight_data[self.current_index]
            self.current_index += 1

            r_fighter = fight["r_fighter"]
            b_fighter = fight["b_fighter"]

            # Get base64 encoded images
            r_image = self.get_fighter_image(r_fighter)
            b_image = self.get_fighter_image(b_fighter)

            if not r_image or not b_image:
                print(f"Skipping fight {self.current_index} due to missing images")
                return None

            # Format the prompt with images
            prompt_text = (
                "🎤 LADIES AND GENTLEMEN! Welcome to the most electrifying show in sports entertainment "
                "Let's break down this matchup that's got everyone talking!\n\n"
                "In the red corner, we have:(YOUR FIRST IMAGE):\n"
                "And in the blue corner: (YOUR SECOND IMAGE):\n\n"
                "Now, act as your favorite fight comentator, I want you to:\n"
                "create a fight commentary of whats happening in the fight live\n"
                "Give us your best fight commentary! Make it exciting, make it dramatic, "
                "make it sound like you're calling the fight live! "
                "Throw in some classic commentator phrases, maybe a 'OH MY GOODNESS!' or two, "
                "and definitely some dramatic pauses for effect.\n\n"
                "End your masterpiece with your prediction in this exact format:\n"
                "[S1]Hello im your host  [S2] And so am i (name) [S1] Wow. Amazing. (laughs) "
                "[S2] Lets get started! (coughs)\n\n"
                "The winner should always be announced with"
                "\\boxed{Red} or \\boxed{Blue}"
                "Or you will receive a score of -1.0"
            )

            # Create multimodal prompt with images
            prompt = tuple(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{r_image}"
                                },
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b_image}"
                                },
                            },
                        ],
                    }
                ]
            )

            winner = fight.get("winner", "")  # Red or Blue
            ground_truth = f"Answer: {winner}" if winner else ""

            return (prompt, ground_truth, None)

        except Exception as e:
            print(f"Error in get_next_item: {e}")
            traceback.print_exc()
            return None

    async def collect_trajectories(
        self, item: Item
    ) -> Tuple[List[Tuple[GameHistory, str, Optional[str]]], List[Item]]:
        to_score = []
        to_backlog = []

        system_msg = {
            "role": "system",
            "content": (
                "You are an expert MMA analyst. You will be given two UFC fighters' images. "
                "Your task is to predict the winner of the fight based on their appearance and physique.\n\n"
                "IMPORTANT: You MUST format your response in exactly two parts:\n"
                "1. First, analyze the fighters' appearances and create a fight commentary\n"
                "2. Then on a new line, give ONLY your final prediction in this exact format:\n"
                "\\boxed{Red} or \\boxed{Blue}\n\n"
                "For example:\n"
                "After analyzing the fighters' appearances... [your analysis here]\n"
                "\\boxed{Red}\n\n"
                "If you do not end your response with the \\boxed{} format containing either 'Red' or 'Blue', "
                "you will receive a score of -1.0."
            ),
        }

        user_msg = {"role": "user", "content": dict(item[0][0])["content"]}

        messages = [system_msg, user_msg]

        try:
            chat_completions = await self.inference_server.chat_completion(
                messages=messages,
                n=self.config.group_size,
                max_tokens=2048,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                timeout=60,
            )
            for choice in chat_completions.choices:
                history = [
                    {"role": "system", "content": system_msg["content"]},
                    {"role": "user", "content": user_msg["content"]},
                    {"role": "assistant", "content": choice.message.content},
                ]
                to_score.append((history, item[1], None))
        except Exception as e:
            print(f"Error in collect_trajectories: {e}")
            traceback.print_exc()
            to_backlog.append(item)

        if not to_score:
            return None, to_backlog

        scored_data = await self.score(to_score)
        return scored_data, to_backlog

    async def score(self, rollout_group_data) -> Optional[ScoredDataGroup]:
        if not rollout_group_data:
            return None

        scores = ScoredDataGroup()
        scores["tokens"] = []
        scores["masks"] = []
        scores["scores"] = []
        scores["advantages"] = None
        scores["ref_logprobs"] = None
        scores["messages"] = None
        scores["group_overrides"] = {"group_size": self.config.group_size}
        scores["overrides"] = None
        scores["ground_truths"] = []

        random.shuffle(rollout_group_data)
        for item in rollout_group_data:
            out = tokenize_for_trainer(self.tokenizer, item[0])
            tokens = out["tokens"]
            masks = out["masks"]

            try:
                # Extract prediction and ground truth
                reply = item[0][-1]["content"]
                ground_truth = item[1].strip().lower()

                # Extract color from ground truth (format: "answer: color")
                ground_truth_color = ground_truth.replace("answer:", "").strip()

                # Extract color from \boxed{color} format
                import re

                boxed_match = re.search(r"\\boxed{([^}]+)}", reply)
                if boxed_match:
                    prediction = boxed_match.group(1).strip().lower()
                    # Compare just the colors
                    reward = 1.0 if prediction == ground_truth_color else -1.0
                else:
                    # No boxed answer found
                    reward = -1.0

            except Exception as e:
                print(f"Error scoring response: {e}")
                reward = -1.0
                ground_truth = item[1] if isinstance(item[1], str) else ""

            if len([i for i in masks if i != -100]) < 10:
                continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(reward)
            scores["ground_truths"].append(ground_truth)

            if len(scores["tokens"]) >= self.config.group_size:
                break

        if not scores["tokens"]:
            return None

        return scores

    async def evaluate(self, *args, **kwargs):
        """No-op evaluation"""
        return

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[OpenaiConfig]]:
        """Initialize configuration for the environment"""
        if not os.environ.get("OPENAI_API_KEY"):
            print("ERROR: OPENAI_API_KEY environment variable is not set!")
            sys.exit(1)

        config = UFCImageEnvConfig(
            wandb_name="ufc_image",
            tokenizer_name="gpt2",
            group_size=2,
            use_wandb=False,
            max_num_workers=2,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=1,
            steps_per_eval=10,
            ensure_scores_are_not_same=False,
        )

        server_configs = [
            OpenaiConfig(
                model_name="gpt-4o",
                base_url=None,
                api_key=os.environ.get("OPENAI_API_KEY"),
                num_requests_for_eval=1,
            ),
        ]

        return config, server_configs


if __name__ == "__main__":
    UFCImageEnv.cli()
