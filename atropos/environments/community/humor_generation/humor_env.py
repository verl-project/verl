import os
from typing import List, Optional, Tuple

from datasets import load_dataset

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


class HumorEnvConfig(BaseEnvConfig):
    data_path: str = "environments/community/humor_generation/humor_dataset.jsonl"


class HumorEnv(BaseEnv):
    env_config_cls = HumorEnvConfig
    name = "humor"

    @classmethod
    def config_init(cls) -> Tuple[HumorEnvConfig, List[APIServerConfig]]:
        env_config = cls.env_config_cls(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=2,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=1024,
            steps_per_eval=100,
            max_token_length=2048,
            wandb_name="humor",
        )
        server_configs = [
            APIServerConfig(
                model_name="gpt-4o-mini",
                base_url=None,
                api_key=os.environ.get("OPENAI_API_KEY"),
                num_requests_for_eval=256,
            )
        ]
        return env_config, server_configs

    async def setup(self):
        ds = load_dataset("json", data_files=self.config.data_path, split="train")
        self.train = ds
        self.iter = 0

    async def get_next_item(self) -> Tuple[dict]:
        record = self.train[self.iter % len(self.train)]
        self.iter += 1
        return (record,)

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, List]:
        record = item[0]
        prompt = record["question"]
        chat_completions = await self.server.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
        )
        to_score = []
        for choice in chat_completions.choices:
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": choice.message.content},
            ]
            to_score.append((tuple(messages), choice.finish_reason))
        scored = await self.score(to_score)
        return scored, []

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        """
        Score each generated joke using the detailed rubric via an LLM call.
        """
        scores = ScoredDataGroup(tokens=[], masks=[], scores=[])
        # All items share same comedian/format
        fmt = self.train[0]["format"]
        comedian = self.train[0]["comedian"]
        for messages, _ in rollout_group_data:
            joke = messages[-1]["content"].strip()
            # Build the rubric prompt
            rubric_prompt = (
                f'1. Relevance to the format ({fmt}): Evaluate the joke "{joke}". Score: X (0-2)\n'
                f'2. Style consistency ({comedian}): Evaluate the joke "{joke}". Score: X (0-2)\n'
                f'3. Creativity: Evaluate the joke "{joke}". Score: X (0-3)\n'
                f'4. Humor effectiveness: Evaluate the joke "{joke}". Score: X (0-3)\n'
                f'5. Virality: Evaluate the joke "{joke}". Score: X (0-3)\n'
                f'6. Cognitive coherence: Evaluate the joke "{joke}". Score: X (0-3)\n'
                "Please provide each score on its own line as 'Score: <number>'."
            )
            judge = await self.server.chat_completion(
                messages=[{"role": "user", "content": rubric_prompt}],
                n=1,
                max_tokens=512,
            )
            text = judge.choices[0].message.content
            # Parse out all Score: X lines
            nums = [
                int(line.split("Score:")[-1].strip().split()[0])
                for line in text.splitlines()
                if "Score:" in line
            ]
            avg_score = sum(nums) / len(nums) if nums else 0.0
            out = tokenize_for_trainer(self.tokenizer, messages)
            scores["tokens"].append(out["tokens"])
            scores["masks"].append(out["masks"])
            scores["scores"].append(avg_score)
        return scores

    async def wandb_log(self, wandb_metrics: Optional[dict] = None):
        await super().wandb_log(wandb_metrics)

    async def evaluate(self, *args, **kwargs):
        # No-op evaluation; required by BaseEnv abstract interface
        return None


if __name__ == "__main__":
    import sys

    # default to 'serve' if no subcommand provided
    if len(sys.argv) == 1:
        sys.argv.append("serve")
    HumorEnv.cli()
