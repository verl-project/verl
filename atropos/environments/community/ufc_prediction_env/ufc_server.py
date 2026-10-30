import csv
import os
import random
import sys
import traceback
from typing import List, Optional, Tuple

from pydantic import Field

from atroposlib.envs.base import BaseEnv, BaseEnvConfig, OpenaiConfig, ScoredDataGroup
from atroposlib.type_definitions import GameHistory, Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class UFCEnvConfig(BaseEnvConfig):
    """Configuration for the UFC Environment"""

    fighter_stats_path: str = Field(
        os.path.join(os.path.dirname(__file__), "fighter_stats.csv"),
        description="Path to fighter stats CSV",
    )
    fight_data_path: str = Field(
        os.path.join(os.path.dirname(__file__), "large_dataset.csv"),
        description="Path to large fight dataset CSV",
    )
    max_steps: int = Field(1, description="Only one step per fight prediction")
    temperature: float = Field(0.7, description="Temperature for generation diversity")
    top_p: float = Field(0.95, description="Top p for nucleus sampling")


class UFCEnv(BaseEnv):
    """UFC Fight Prediction Environment"""

    name = "ufc_predictor"
    env_config_cls = UFCEnvConfig

    def __init__(
        self,
        config: UFCEnvConfig,
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

        except Exception as e:
            print(f"Error loading data: {e}")
            traceback.print_exc()
            sys.exit(1)

    async def get_next_item(self) -> Optional[Item]:
        """Get the next fight from the dataset"""
        try:
            if self.current_index >= len(self.fight_data):
                return None
            fight = self.fight_data[self.current_index]
            self.current_index += 1

            r_fighter = fight["r_fighter"]
            b_fighter = fight["b_fighter"]
            r_stats = self.fighter_stats.get(r_fighter, {})
            b_stats = self.fighter_stats.get(b_fighter, {})

            # Format the prompt
            def stats_str(name, stats):
                if not stats:
                    return f"{name}: (No stats available)"
                return (
                    f"Name: {name}\n"
                    f"Wins: {stats.get('wins', '?')}  "
                    f"Losses: {stats.get('losses', '?')}  "
                    f"Age: {stats.get('age', '?')}\n"
                    f"Height: {stats.get('height', '?')} cm  "
                    f"Weight: {stats.get('weight', '?')} kg  "
                    f"Reach: {stats.get('reach', '?')} cm  "
                    f"Stance: {stats.get('stance', '?')}\n"
                    f"SLpM: {stats.get('SLpM', '?')}  "
                    f"Sig Str Acc: {stats.get('sig_str_acc', '?')}  "
                    f"SApM: {stats.get('SApM', '?')}  "
                    f"Str Def: {stats.get('str_def', '?')}\n"
                    f"TD Avg: {stats.get('td_avg', '?')}  "
                    f"TD Acc: {stats.get('td_acc', '?')}  "
                    f"TD Def: {stats.get('td_def', '?')}  "
                    f"Sub Avg: {stats.get('sub_avg', '?')}\n"
                )

            prompt_text = (
                "🎤 LADIES AND GENTLEMEN! Welcome to the most electrifying show in sports entertainment - "
                "the UFC Fight Prediction Show! "
                "Let's break down this matchup that's got everyone talking!\n\n"
                f"*Drumroll please* In the red corner, we have :\n{stats_str(r_fighter, r_stats)}\n\n"
                f"And in the blue corner:\n{stats_str(b_fighter, b_stats)}\n\n"
                "Now, as your favorite fight analyst who's definitely not just making this up as "
                "I go along, I want you to:\n"
                "1. Break down these fighters like you're explaining why your favorite TV show character "
                "would win in a fight\n"
                "2. Compare their styles\n"
                "3. Point out their advantages\n"
                "Give us your best fight commentary! Make it exciting, make it dramatic, make it sound "
                "like you're calling the fight live! "
                "Throw in some classic commentator phrases, maybe a 'OH MY GOODNESS!' or two, and "
                "definitely some dramatic pauses for effect.\n\n"
                "End your masterpiece with the winner's name in this exact format:\n"
                "\\boxed{fighter name}"
            )

            prompt = tuple(
                [frozenset({"role": "user", "content": prompt_text}.items())]
            )

            winner = fight.get("winner", "")  # Red or Blue
            winner_name = (
                r_fighter if winner == "Red" else b_fighter if winner == "Blue" else ""
            )
            ground_truth = f"Answer: {winner_name}" if winner_name else ""

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
                "You are an expert MMA analyst. You will be given two UFC fighters and their stats. "
                "Your task is to predict the winner of the fight based on their statistics.\n\n"
                "IMPORTANT: You MUST format your response in exactly two parts:\n"
                "1. First, analyze the fighters' stats and explain create a fight commentary\n"
                "2. Then on a new line, give ONLY your final prediction in this exact format:\n"
                "\\boxed{fighter name}\n\n"
                "For example:\n"
                "After analyzing stats... [your analysis here]\n"
                "\\boxed{John Smith}\n\n"
                "If you do not end your response with the \\boxed{} format, you will receive a score of -1.0."
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

                # Extract name from ground truth (format: "answer: name")
                ground_truth_name = ground_truth.replace("answer:", "").strip()

                # Extract name from \boxed{name} format
                import re

                boxed_match = re.search(r"\\boxed{([^}]+)}", reply)
                if boxed_match:
                    prediction = boxed_match.group(1).strip().lower()
                    # Compare just the names
                    reward = 1.0 if prediction == ground_truth_name else -1.0
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


if __name__ == "__main__":
    UFCEnv.cli()
