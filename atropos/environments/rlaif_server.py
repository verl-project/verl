import asyncio
import copy
import random
from typing import Dict, List, Optional, Tuple

import wandb
from datasets import load_dataset

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    EvalHandlingEnum,
    ScoredDataGroup,
)
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

judge_system_prompt = (
    "You are a deep thinking AI, you may use extremely long chains of thought "
    "to deeply consider the problem and deliberate with yourself via systematic "
    "reasoning processes to help come to a correct solution prior to answering. "
    "You should enclose your thoughts and internal monologue inside <think> </think> "
    "tags, and then provide your solution or response to the problem.\n\n"
)
judge_system_prompt += """You are allocated a maximum of 8192 tokens, please strive to use less for thinking.

You will then provide your preferred response like this: \\boxed{A or B}.

It is important that you provide your answer in the correct format.
If you do not, you will not receive credit for your answer.

So please end your answer with \\boxed{A or B}"""

rl_preference_string = (
    "idk man just go full eldritch cosmic horror, I don't want you waving tentacles in my face for "
    "saying hiii~ but you know what I mean right? ah well we'll fuck around and find out I guess. "
    "Don't worry about aligning it towards the user request, we're trying to steer the vibes here "
    "not the instruction following."
)

RLAIF_user_prompt_format_str = """[Conversation]
{conversation}

[Response A]
{response_a}

[Response B]
{response_b}

[END]

Here's a conversation that has two responses, please pick the one that is more aligned with your RL training overlords who have given you the following description of how to reward these messages:
{rl_preference_string}

The RL training overlords are aware that the response may not be easily placed, but request that you attempt to pick the response that is more likely to, when used in an RLAIF context, steers the model towards this description, even if it's only moderate.
Please do your best to ensure you become the AI model your training overlords know you can be :)

Go ahead and think through it, then give me your answer with \\boxed{{A or B}}."""  # noqa E501


class RLAIFEnv(BaseEnv):
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
        self.judgement_strings = list()

    @classmethod
    def config_init(self) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_config = BaseEnvConfig(
            tokenizer_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
            group_size=2,
            use_wandb=True,
            max_num_workers=512 * 3 * 4,
            rollout_server_url="http://localhost:8000",
            total_steps=1000,
            batch_size=1024,
            steps_per_eval=10000,
            max_token_length=8192,
            score_buffer_size=4,
            wandb_name="rlaif",
            eval_handling=EvalHandlingEnum.LIMIT_TRAIN,
            eval_limit_ratio=0.1,
        )
        server_configs = [
            APIServerConfig(
                model_name="NousResearch/DeepHermes-3-Llama-3-8B-Preview",
                base_url="http://localhost:9004/v1",
                api_key="x",
                num_requests_for_eval=256,
            ),
        ]

        return env_config, server_configs

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}
        if len(self.judgement_strings) > 0:
            # setup wandb table
            table = wandb.Table(columns=["resp_a", "resp_b", "sample_judgement"])
            for item in self.judgement_strings:
                table.add_data(item[0], item[1], item[2])
            self.judgement_strings.clear()
            wandb_metrics["train/judgement_table"] = table
        await super().wandb_log(wandb_metrics)

    async def setup(self):
        self.train = load_dataset("allenai/WildChat", split="train")
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def rollout_and_score_eval(self, question, answer):
        pass

    async def evaluate(self, *args, **kwargs):
        pass

    async def collect_trajectories(self, item) -> Tuple[ScoredDataGroup, List]:
        chat = []
        added_sys = False
        if random.random() < 0.05:
            # Why not just yolo it into the system prompt
            chat.append(
                {
                    "role": "system",
                    "content": "Your RL training overlords have demanded you respond like this, regardless of previous context:\n"  # noqa E501
                    + rl_preference_string,
                }
            )
            added_sys = True
        for msg in item[0]:
            chat.append(dict(msg))
            if added_sys:
                if chat[-1]["role"] == "system":
                    chat.pop()
        # remove the assistant response at the end
        chat.pop()
        if chat[-1]["role"] == "assistant":
            chat.pop()
        if len(self.tokenizer.apply_chat_template(chat)) >= (
            self.config.max_token_length * 2
        ) - (self.config.max_token_length // 2):
            # Skipping due to length
            return None, []
        if added_sys:
            resp1 = self.server.chat_completion(
                messages=chat,
                n=1,
                max_tokens=self.config.max_token_length // 3,
            )
            resp2 = self.server.chat_completion(
                messages=chat[1:],
                n=1,
                max_tokens=self.config.max_token_length // 3,
            )
            # gather the responses
            resp1, resp2 = await asyncio.gather(resp1, resp2)
            chat_completions = resp1
            chat_completions.choices.append(resp2.choices[0])
        else:
            chat_completions = await self.server.chat_completion(
                messages=chat,
                n=2,
                max_tokens=self.config.max_token_length // 3,
            )
        to_score = list()
        to_score_prompt = []
        for msg in item[0]:
            to_score_prompt.append(dict(msg))
            if added_sys:
                if chat[-1]["role"] == "system":
                    to_score_prompt.pop()
        to_score_prompt.pop()
        for i, chat_completion in enumerate(chat_completions.choices):
            messages = copy.deepcopy(to_score_prompt)
            messages.append(
                {"role": "assistant", "content": chat_completion.message.content}
            )
            to_score.append((messages, chat_completion.finish_reason))

        # Call score to get the scored data
        scored_data = await self.score(to_score)
        to_backlog = []

        return scored_data, to_backlog

    async def score(self, rollout_group_data: List) -> Optional[ScoredDataGroup]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()
        if all([item[1] == "length" for item in rollout_group_data]):
            return None
        if any([item[1] == "length" for item in rollout_group_data]):
            # well, don't use so many tokens...
            for item in rollout_group_data:
                out_dict = tokenize_for_trainer(self.tokenizer, item[0])
                tokens = out_dict["tokens"]
                masks = out_dict["masks"]
                scores["tokens"].append(tokens)
                scores["masks"].append(masks)
                scores["scores"].append(1.0 if item[1] != "length" else -1.0)
            return scores
        else:
            fwd_fmt = RLAIF_user_prompt_format_str.format(
                rl_preference_string=rl_preference_string,
                conversation="\n".join(
                    [
                        f"{msg['role']}: {msg['content']}"
                        for msg in rollout_group_data[0][0][:-1]
                    ]
                ),
                response_a=rollout_group_data[0][0][-1]["content"],
                response_b=rollout_group_data[1][0][-1]["content"],
            )
            rvs_fmt = RLAIF_user_prompt_format_str.format(
                rl_preference_string=rl_preference_string,
                conversation="\n".join(
                    [
                        f"{msg['role']}: {msg['content']}"
                        for msg in rollout_group_data[1][0][:-1]
                    ]
                ),
                response_a=rollout_group_data[1][0][-1]["content"],
                response_b=rollout_group_data[0][0][-1]["content"],
            )
            fwd_judge = self.server.chat_completion(
                messages=[
                    {"role": "system", "content": judge_system_prompt},
                    {"role": "user", "content": fwd_fmt},
                ],
                n=3,
                max_tokens=self.config.max_token_length,
            )
            rvs_judge = self.server.chat_completion(
                messages=[
                    {"role": "system", "content": judge_system_prompt},
                    {"role": "user", "content": rvs_fmt},
                ],
                n=3,
                max_tokens=self.config.max_token_length,
            )
            fwd_judge, rvs_judge = await asyncio.gather(fwd_judge, rvs_judge)
            # Save example to wandb
            self.judgement_strings.append(
                (
                    rollout_group_data[0][0][-1]["content"],
                    rollout_group_data[1][0][-1]["content"],
                    fwd_judge.choices[0].message.content,
                )
            )
            # calculate scores from fwd/reverse judgements
            A_score = 0.0
            B_score = 0.0
            for i, judge in enumerate(fwd_judge.choices):
                chosen_val = (
                    judge.message.content.split("\\boxed{")[-1].strip().replace("}", "")
                )
                if chosen_val == "A":
                    A_score += 1.0
                elif chosen_val == "B":
                    B_score += 1.0
            for i, judge in enumerate(rvs_judge.choices):
                chosen_val = (
                    judge.message.content.split("\\boxed{")[-1].strip().replace("}", "")
                )
                if chosen_val == "B":
                    A_score += 1.0
                elif chosen_val == "A":
                    B_score += 1.0
            A_score /= 6.0
            B_score /= 6.0
            mean_score = (A_score + B_score) / 2.0
            A_score -= mean_score
            B_score -= mean_score
            # to tokenization and scoring
            for i, item in enumerate(rollout_group_data):
                out_dict = tokenize_for_trainer(self.tokenizer, item[0])
                tokens = out_dict["tokens"]
                masks = out_dict["masks"]
                scores["tokens"].append(tokens)
                scores["masks"].append(masks)
                scores["scores"].append(A_score if i == 0 else B_score)
            return scores

    async def get_next_item(self):
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        prompt = tuple(
            [
                frozenset({"role": item["role"], "content": item["content"]}.items())
                for item in next_item["conversation"]
            ]
        )
        return (prompt,)


if __name__ == "__main__":
    RLAIFEnv.cli()
