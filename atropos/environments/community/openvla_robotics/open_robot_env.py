from typing import Dict, List, Optional, Tuple, Union

import robosuite as suite
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item, number

from .action_tokenizer import ActionTokenizer

prompt = "In: What action should the robot take to pick up the cube?\nOut:"


class RobotSimEnv(BaseEnv):

    name = "robot_sim"

    def __init__(
        self,
        config: BaseEnvConfig,
        server_configs: List[APIServerConfig],
        slurm=True,
        testing=False,
    ):
        super().__init__(config, server_configs, slurm, testing)
        self.percent_correct_buffer = list()
        self.eval_metrics = list()
        # Add tracking for wandb visualizations
        self.rollouts_for_wandb = []
        self.completion_lengths = []

    # Many methods here will have to be removed
    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:

        # vllm support for VLA models is a bit iffy. Therefore we don't use vllm.
        # Ideally no model would be used below, but empty/None values throw error
        # therefore placeholder tokenizer_name and model_name is passed.
        # These are not needed for the VLA to run.
        env_config = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
            group_size=8,
            use_wandb=True,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=12,
            steps_per_eval=100,
            max_token_length=2048,
            wandb_name="gsm8k",
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-3B-Preview",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=256,
            ),
        ]

        return env_config, server_configs

    # Not sure what to do with this
    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        # Try to calculate percent_correct, pass if there's a division by zero
        try:
            wandb_metrics["train/percent_correct"] = sum(
                self.percent_correct_buffer
            ) / len(self.percent_correct_buffer)
        except ZeroDivisionError:
            # Skip if buffer is empty
            pass

        self.percent_correct_buffer = list()
        for item in self.eval_metrics:
            wandb_metrics[item[0]] = item[1]
        self.eval_metrics = list()
        # Call the parent method to handle the server metrics
        await super().wandb_log(wandb_metrics)

    # There is NO test data for this environment
    async def setup(self):

        self.local_model_path = "openvla/openvla-7b"

        self.robosuite_env = suite.make(
            "Lift",  # This can be changed to any other envirionment like NutAssemblySquare
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names="frontview",
            camera_heights=640,
            camera_widths=480,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.local_model_path, trust_remote_code=True
        )

        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.local_model_path,
            # attn_implementation="flash_attention_2", #Removed for now. #TODO: Add this
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to("cuda:0")

        self.action_tokenizer = ActionTokenizer(
            self.processor.tokenizer, bins=256, min_action=-1, max_action=1
        )

        self.max_steps = 100

        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    # See where this is called from and then change. No eval data so commenting out for now.
    async def rollout_and_score_eval(self, question: str, answer: str) -> number:

        pass

    async def evaluate(self, *args, **kwargs):
        # No evaluation data available for this environment yet
        # eval_tasks = []
        # for item in self.test:
        #     eval_tasks.append(
        #         self.rollout_and_score_eval(item["question"], item["gold_answer"])
        #     )
        # scores = await tqdm_asyncio.gather(*eval_tasks)
        # self.eval_metrics.append(("eval/percent_correct", sum(scores) / len(scores)))
        return None

    async def collect_trajectories(
        self, item: bool
    ) -> Tuple[ScoredDataGroup, list[Item]]:

        obs = self.robosuite_env.reset()

        to_score = list()
        to_backlog = list()

        print("TRYING TO COLLECT TRAJECTORIES")

        print(self.max_steps)

        for i in range(self.max_steps):

            image = Image.fromarray(obs["frontview_image"])

            inputs = self.processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
            action = self.vla.predict_action(
                **inputs, unnorm_key="bridge_orig", do_sample=False
            )

            # Remove adjustment for now
            action[0:3] = action[0:3] * 100  # because the sensitivity is 0.01
            action[..., 2] = action[..., 2] * -1.0
            # action[..., -1] = np.sign(action[..., -1])

            orig_low, orig_high = 0.0, 1.0
            action[..., -1] = (
                2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1
            )
            # action[..., -1] = np.sign(action[..., -1])
            # action[6] = action[6]*2-1 #related to the gripper openvla is [0,1],robosuite is [-1,1]
            action[..., -1] = action[..., -1] * -1.0

            obs, reward, done, info = self.robosuite_env.step(action)

            print(action)

            to_score.append({"action:": action, "reward": reward})

        # user_message = {"role": "user", "content": item["question"]}
        # gold_answer = (
        #     "\\boxed{" + item["answer"].split("#")[-1].strip().replace(",", "") + "}"
        # )

        # chat_completions = await self.server.chat_completion(
        #     messages=[{"role": "system", "content": system_prompt}, user_message],
        #     n=self.config.group_size,
        #     max_tokens=self.config.max_token_length,
        # )

        # for i, chat_completion in enumerate(chat_completions.choices):
        #     messages = (
        #         {"role": "system", "content": system_prompt},
        #         user_message,
        #         {"role": "assistant", "content": chat_completion.message.content},
        #     )
        #     to_score.append(
        #         {
        #             "messages": messages,
        #             "gold_answer": gold_answer,
        #             "finish_reason": chat_completion.finish_reason,
        #         }
        #     )

        to_postprocess = await self.score(to_score)
        return to_postprocess, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        for item in rollout_group_data:
            action_tokens, mask = self.action_tokenizer.tokenize(item["action"])
            scores["tokens"].append(action_tokens)
            scores["masks"].append(mask)
            scores["scores"].append(item["reward"])
        return scores

    # What should this return? A full trajectory of the robot consisting of N steps
    async def get_next_item(self) -> bool:
        # There is no next item, only a reset (or a random seed).
        # See how we can take a random seed. For now just reset.

        # Another limitation is that right now it has to be 1 by 1.
        # It can't do multiple of this. It has to wait for one to finish.

        self.iter += 1
        return True


if __name__ == "__main__":
    RobotSimEnv.cli()
