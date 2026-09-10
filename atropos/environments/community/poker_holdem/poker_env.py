import random
import re
from typing import Any, Dict, List, Optional, Tuple

# Import custom reward functions to ensure they're registered
import reward_fns as poker_reward_fns
from transformers import AutoTokenizer

from atroposlib.envs.base import APIServerConfig, BaseEnv, BaseEnvConfig

WANDB_PROJECT = "poker"


class PokerEnv(BaseEnv):
    """Poker training environment using processed hand histories of winning players"""

    name = "poker_env"

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        # configure for axolotl plugin
        env_config = BaseEnvConfig(
            group_size=8,
            use_wandb=True,
            tokenizer_name="Qwen/Qwen3-1.7B",
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=256,
            wandb_name="number_grid",
        )

        server_configs = [
            APIServerConfig(
                base_url="http://localhost:9002/v1",
                api_key="EMPTY",
                num_requests_for_eval=256,
                model_name="Qwen/Qwen3-1.7B",
                server_type="openai",
            ),
        ]

        return env_config, server_configs

    async def setup(self):
        """Load the dataset and prepare environment"""
        # Load dataset from Hugging Face with train/test splits
        from datasets import load_dataset

        # Load the specific dataset from Hugging Face
        self.train_dataset = load_dataset("yoniebans/6max-nlh-poker", split="train")
        self.eval_dataset = load_dataset("yoniebans/6max-nlh-poker", split="test")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)

        self.reward_function = poker_reward_fns.PokerCombinedReward(
            action_match_weight=0.6,
            bet_sizing_weight=0.4,
        )

        # Group hands by game stage for easier access by parsing from prompts
        self.preflop_hands = []
        self.flop_hands = []
        self.turn_hands = []
        self.river_hands = []

        # Process training dataset using the game_stage column
        for i, item in enumerate(self.train_dataset):
            stage = item["game_stage"].upper()
            if stage == "PREFLOP":
                self.preflop_hands.append(i)
            elif stage == "FLOP":
                self.flop_hands.append(i)
            elif stage == "TURN":
                self.turn_hands.append(i)
            elif stage == "RIVER":
                self.river_hands.append(i)

        # Set up evaluation indices based on the eval dataset
        self.eval_indices = list(range(len(self.eval_dataset)))

        # Create training indices and shuffle them
        self.train_indices = list(range(len(self.train_dataset)))
        random.shuffle(self.train_indices)
        self.training_queue = self.train_indices.copy()

        # Initialize cumulative evaluation stats
        self.eval_stats = {
            "PREFLOP": {"total": 0, "correct": 0, "scores": []},
            "FLOP": {"total": 0, "correct": 0, "scores": []},
            "TURN": {"total": 0, "correct": 0, "scores": []},
            "RIVER": {"total": 0, "correct": 0, "scores": []},
        }

        # Initialize training stats
        self.stats = {
            "total_evaluated": 0,
            "stages": {
                "PREFLOP": {"count": 0, "score_sum": 0},
                "FLOP": {"count": 0, "score_sum": 0},
                "TURN": {"count": 0, "score_sum": 0},
                "RIVER": {"count": 0, "score_sum": 0},
            },
            "actions": {
                "fold": 0,
                "check": 0,
                "call": 0,
                "bet": 0,
                "raise": 0,
                "re-raise": 0,
                "all-in": 0,
                "unknown": 0,
            },
            "epoch": 0,  # Track how many times we've gone through the dataset
        }

        print(f"Loaded training dataset with {len(self.train_dataset)} hands")
        print(f"  Preflop: {len(self.preflop_hands)}")
        print(f"  Flop: {len(self.flop_hands)}")
        print(f"  Turn: {len(self.turn_hands)}")
        print(f"  River: {len(self.river_hands)}")
        print(f"Loaded evaluation dataset with {len(self.eval_dataset)} examples")
        print(f"Training set indices: {len(self.train_indices)}")

    async def get_next_item(self) -> Dict:
        """Get the next poker hand for evaluation using a shuffled queue approach"""
        # If queue is empty, reshuffle and start over
        if not self.training_queue:
            random.shuffle(self.train_indices)
            self.training_queue = self.train_indices.copy()
            self.stats["epoch"] += 1
            print(f"Completed epoch {self.stats['epoch']}, reshuffled training queue")

        # Get next index from queue
        idx = self.training_queue.pop(0)
        hand = self.train_dataset[idx]

        # Get stage directly from the game_stage column
        stage = hand["game_stage"].upper()

        # Extract data according to the HuggingFace schema
        return {
            "situation": hand["pokergpt_prompt"],  # The formatted prompt
            "winner_action": hand["formatted_winning_action"],  # The expected action
            "hand_id": hand.get("hand_id", str(idx)),
            "bb_won": hand.get("bb_won", 0),
            "stage": stage,
            "hand_idx": idx,
        }

    def _parse_action(self, text: str) -> Tuple[str, Optional[float]]:
        """Extract action type and amount from model response"""
        text = text.strip().lower()

        # Define all possible action types
        action_types = ["fold", "check", "call", "bet", "raise", "re-raise", "all-in"]

        # Find which action type is in the response
        action_type = next((a for a in action_types if a in text), None)

        if not action_type:
            return "unknown", None

        # For actions that might have amounts, extract the amount
        amount = None
        if action_type in ["call", "bet", "raise", "re-raise"]:
            # Look for a number after the action
            amount_match = re.search(r"(\d+(?:\.\d+)?)", text)
            if amount_match:
                try:
                    amount = float(amount_match.group(1))
                except ValueError:
                    pass

        return action_type, amount

    async def collect_trajectory(self, item: Dict) -> Tuple[Any | None, List[Dict]]:
        """Get and score a single response from the model"""
        situation = item["situation"]

        # Use parameters from config with fallbacks if not specified
        temperature = getattr(self.config, "temperature", 0.7)
        max_tokens = getattr(self.config, "max_tokens", 50)

        # Get model response
        response = await self.server.chat_completion(
            messages=[{"role": "user", "content": situation}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

        text = response.choices[0].message.content.strip()

        # Parse the model's action for logging
        action_type, amount = self._parse_action(text)

        # Score using reward function
        scores = self.reward_function.compute(
            [text], winner_action=item["winner_action"]
        )
        score = scores[0] if scores else 0.0

        # Update stats
        self.stats["total_evaluated"] += 1
        stage = item["stage"]
        if stage in self.stats["stages"]:
            self.stats["stages"][stage]["count"] += 1
            self.stats["stages"][stage]["score_sum"] += score

        if action_type in self.stats["actions"]:
            self.stats["actions"][action_type] += 1

        # Tokenize the response
        tokens = self.tokenizer.encode(text)
        masks = [-100] * len(tokens)  # Standard masking - adjust if needed

        # Create scored data with metadata for analysis
        scored_data = {
            "tokens": [tokens],
            "masks": [masks],
            "scores": [score],
            "messages": [
                [
                    {"role": "user", "content": situation},
                    {"role": "assistant", "content": text},
                ]
            ],
            "metadata": {
                "hand_id": item["hand_id"],
                "stage": stage,
                "bb_won": item["bb_won"],
                "model_action": text,
                "parsed_action": {"type": action_type, "amount": amount},
                "winner_action": item["winner_action"],
            },
        }

        return scored_data, []  # No backlog items

    async def evaluate(self, *args, **kwargs):
        """Evaluate model performance on a sample of the evaluation set"""
        # Get evaluation parameters from config
        eval_sample_size = getattr(self.config, "eval_sample_size", 200)
        eval_temperature = getattr(self.config, "eval_temperature", 0.2)
        eval_max_tokens = getattr(self.config, "eval_max_tokens", 50)
        correctness_threshold = getattr(self.config, "correctness_threshold", 0.9)

        # Use configured sample size
        sample_size = min(eval_sample_size, len(self.eval_indices))
        sampled_indices = random.sample(self.eval_indices, sample_size)

        print(
            f"Starting evaluation on {sample_size} sampled examples (from {len(self.eval_indices)} total)..."
        )
        print(
            f"Using temperature={eval_temperature}, max_tokens={eval_max_tokens}, "
            f"correctness_threshold={correctness_threshold}"
        )

        # Reset per-evaluation stats
        current_eval_stats = {
            "PREFLOP": {"total": 0, "correct": 0, "score_sum": 0},
            "FLOP": {"total": 0, "correct": 0, "score_sum": 0},
            "TURN": {"total": 0, "correct": 0, "score_sum": 0},
            "RIVER": {"total": 0, "correct": 0, "score_sum": 0},
        }

        # Evaluate each example one at a time - Atropos will handle concurrency
        for idx in sampled_indices:
            hand = self.eval_dataset[idx]
            stage = hand["game_stage"].upper()

            try:
                # Get model response using configured parameters
                response = await self.server.chat_completion(
                    messages=[{"role": "user", "content": hand["pokergpt_prompt"]}],
                    temperature=eval_temperature,
                    max_tokens=eval_max_tokens,
                )

                text = response.choices[0].message.content.strip()

                # Score using reward function
                scores = self.reward_function.compute(
                    [text], winner_action=hand["winning_action"]
                )
                score = scores[0] if scores else 0.0

                # Consider "correct" if score exceeds the configured threshold
                is_correct = score > correctness_threshold

                # Update current evaluation stats
                if stage in current_eval_stats:
                    current_eval_stats[stage]["total"] += 1
                    current_eval_stats[stage]["score_sum"] += score
                    if is_correct:
                        current_eval_stats[stage]["correct"] += 1

                # Update cumulative evaluation stats
                if stage in self.eval_stats:
                    self.eval_stats[stage]["total"] += 1
                    self.eval_stats[stage]["scores"].append(score)
                    if is_correct:
                        self.eval_stats[stage]["correct"] += 1

            except Exception as e:
                print(f"Error evaluating hand: {e}")

        # Calculate overall metrics for this evaluation run
        total_correct = sum(stats["correct"] for stats in current_eval_stats.values())
        total_examples = sum(stats["total"] for stats in current_eval_stats.values())
        current_accuracy = total_correct / total_examples if total_examples > 0 else 0

        # Calculate cumulative metrics
        cumulative_correct = sum(stats["correct"] for stats in self.eval_stats.values())
        cumulative_total = sum(stats["total"] for stats in self.eval_stats.values())
        cumulative_accuracy = (
            cumulative_correct / cumulative_total if cumulative_total > 0 else 0
        )

        # Log evaluation results
        print("Evaluation Results (Current Run):")
        print(
            f"  Overall Accuracy: {current_accuracy:.4f} ({total_correct}/{total_examples})"
        )

        # Log stage-specific results for current run
        for stage, stats in current_eval_stats.items():
            if stats["total"] > 0:
                stage_accuracy = stats["correct"] / stats["total"]
                stage_avg_score = stats["score_sum"] / stats["total"]
                print(
                    f"  {stage}: Accuracy={stage_accuracy:.4f}, Avg Score={stage_avg_score:.4f}, Count={stats['total']}"
                )

        # Log cumulative stats
        print("Cumulative Evaluation Results:")
        print(
            f"  Overall Accuracy: {cumulative_accuracy:.4f} ({cumulative_correct}/{cumulative_total})"
        )

        # Log stage-specific cumulative results
        for stage, stats in self.eval_stats.items():
            if stats["total"] > 0:
                stage_accuracy = stats["correct"] / stats["total"]
                stage_avg_score = sum(stats["scores"]) / len(stats["scores"])
                print(
                    f"  {stage}: Accuracy={stage_accuracy:.4f}, Avg Score={stage_avg_score:.4f}, Count={stats['total']}"
                )

        return {
            "accuracy": current_accuracy,
            "cumulative_accuracy": cumulative_accuracy,
        }

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        """Log metrics to Weights & Biases"""
        if wandb_metrics is None:
            wandb_metrics = {}

        # Add epoch counter
        wandb_metrics["train/epoch"] = self.stats["epoch"]

        # Add stage-specific metrics
        for stage, stats in self.stats["stages"].items():
            if stats["count"] > 0:
                avg_score = stats["score_sum"] / stats["count"]
                wandb_metrics[f"train/avg_score_{stage.lower()}"] = avg_score
                wandb_metrics[f"train/count_{stage.lower()}"] = stats["count"]

        # Add action distribution metrics
        for action, count in self.stats["actions"].items():
            pct = (
                count / self.stats["total_evaluated"] * 100
                if self.stats["total_evaluated"] > 0
                else 0
            )
            wandb_metrics[f"train/action_{action}_pct"] = pct

        # Add evaluation metrics if available
        if hasattr(self, "eval_stats"):
            # Calculate cumulative metrics
            cumulative_correct = sum(
                stats["correct"] for stats in self.eval_stats.values()
            )
            cumulative_total = sum(stats["total"] for stats in self.eval_stats.values())
            if cumulative_total > 0:
                cumulative_accuracy = cumulative_correct / cumulative_total
                wandb_metrics["eval/cumulative_accuracy"] = cumulative_accuracy

            # Add stage-specific eval metrics
            for stage, stats in self.eval_stats.items():
                if stats["total"] > 0:
                    stage_accuracy = stats["correct"] / stats["total"]
                    stage_avg_score = (
                        sum(stats["scores"]) / len(stats["scores"])
                        if stats["scores"]
                        else 0
                    )
                    wandb_metrics[f"eval/accuracy_{stage.lower()}"] = stage_accuracy
                    wandb_metrics[f"eval/avg_score_{stage.lower()}"] = stage_avg_score

        # Call the parent method to log base metrics and any custom metrics
        await super().wandb_log(wandb_metrics)


if __name__ == "__main__":
    PokerEnv.cli()
