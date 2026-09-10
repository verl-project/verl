import logging
from typing import List, Optional, Tuple

from pydantic import Field

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.envs.reward_fns.registry import registry
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

logger = logging.getLogger(__name__)


class SanskritPoetryEnvConfig(BaseEnvConfig):
    meter: str = Field("tristubh", description="Desired Sanskrit meter")
    system_prompt: Optional[str] = Field(
        "You are a Sanskrit poet. Respond only with the poem in IAST.",
        description="System prompt for the model",
    )
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 256


class SanskritPoetryEnv(BaseEnv):
    env_config_cls = SanskritPoetryEnvConfig

    def __init__(
        self,
        config: SanskritPoetryEnvConfig,
        server_configs: List[APIServerConfig],
        slurm: bool = True,
        testing: bool = False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        # Create reward function using registry for easy configuration
        self.reward_fn = registry.create(
            {"type": "chandas_meter", "params": {"meter": config.meter}}
        )
        self.iter = 0

    @classmethod
    def config_init(cls) -> Tuple[SanskritPoetryEnvConfig, List[APIServerConfig]]:
        env_config = SanskritPoetryEnvConfig(
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=32,
            steps_per_eval=50,
            max_token_length=512,
            wandb_name="sanskrit_poetry",
        )
        server_configs = [
            APIServerConfig(
                base_url="http://localhost:9001",
                api_key="x",
                num_requests_for_eval=64,
                model_name="Qwen/Qwen3-1.7B",
                server_type="trl",
            )
        ]
        return env_config, server_configs

    async def setup(self):
        self.iter = 0

    async def get_next_item(self):
        prompt = (
            f"Compose a four line Sanskrit poem in the {self.config.meter} meter. "
            "Use IAST transliteration only."
        )
        user_msg = {"role": "user", "content": prompt}
        return (tuple([frozenset(user_msg.items())]), None, None)

    async def collect_trajectories(self, item):
        user_content = dict(item[0][0])["content"]
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": user_content})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        completions = await self.server.completion(
            prompt=prompt,
            n=self.config.group_size,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )
        trajectories = []
        for completion in completions.choices:
            completion_text = (
                completion.text
                if hasattr(completion, "text")
                else completion.message.content
            )
            msg_seq = []
            if self.config.system_prompt:
                msg_seq.append({"role": "system", "content": self.config.system_prompt})
            msg_seq.append({"role": "user", "content": user_content})
            msg_seq.append({"role": "assistant", "content": completion_text})
            trajectories.append(msg_seq)
        return trajectories, []

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        scored = ScoredDataGroup()
        scored["tokens"] = []
        scored["masks"] = []
        scored["scores"] = []
        for traj in rollout_group_data:
            reward = self.reward_fn([traj[-1]["content"]])[0]
            out_dict = tokenize_for_trainer(self.tokenizer, traj)
            scored["tokens"].append(out_dict["tokens"])
            scored["masks"].append(out_dict["masks"])
            scored["scores"].append(reward)
        return scored
