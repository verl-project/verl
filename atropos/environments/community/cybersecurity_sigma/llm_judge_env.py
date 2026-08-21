import os
import random
import re
from typing import Dict, List, Optional, Tuple, TypedDict, Union

import yaml
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse
from tqdm.asyncio import tqdm_asyncio

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item, number
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

os.environ["OPENAI_API_KEY"] = "x"

system_prompt = """
You are a systematic, deep-reasoning AI trained to solve detection engineering problems by
constructing Sigma rules. You must first **think carefully** through the problem using structured
internal monologue enclosed in <think>...</think> tags, and then provide your final Sigma rule as
a YAML block enclosed in a LaTeX \\boxed{...} environment.

**You MUST follow this exact output format:**
<think>
Step-by-step reasoning, outlining how you arrived at the rule. Include relevant knowledge about
Sigma syntax, threat detection context, and how you chose each field.
</think>

\\boxed{
<YAML Sigma Rule>
}

**Important Rules:**
- DO NOT skip the <think> tags — all your thoughts must be inside them.
- DO NOT skip the \\boxed{...} wrapper — your final YAML answer MUST go inside it.
- DO NOT output anything outside the <think> and \\boxed{} blocks.
- The content inside \\boxed{} must be **pure YAML**, with **no extra formatting characters**
  (no bullets, backticks, emojis, or markdown) so it can be passed **directly** to `yaml.safe_load()`.
- You are allocated a maximum of 2048 tokens — be detailed but concise.
- Your final output must be valid YAML for a Sigma rule, with correct indentation and field names.
- Use Sigma best practices: include `detection.condition`, `detection.selection`,
  `logsource`, and `timeframe` when appropriate.
- Match the format and style of this example:

Example:
<think>
This rule is intended to detect potential lateral movement attempts by monitoring for excessive
outbound connections. The condition 'selection | count(dst_ip) by src_ip > 10' flags when one
source IP connects to more than 10 destination IPs. The log source is firewall logs, and the
action is 'denied'.
</think>

\\boxed{
detection:
  condition: selection | count(dst_ip) by src_ip > 10
  selection:
    action: denied
  timeframe: 24h
logsource:
  category: firewall
}

Only produce answers in this format. Think first, then answer clearly in YAML. Follow YAML syntax
exactly.
"""


class SigmaRuleEntry(TypedDict):
    question: str
    answer: str


class SigmaRuleEnv(BaseEnv):

    name = "llm_judge_sigmarule"

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

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
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
                # api_key="x",
                num_requests_for_eval=256,
            ),
        ]

        return env_config, server_configs

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

    async def setup(self):
        self.train = load_dataset(
            "mmaisel1/nous-rl-hackathon-sigma", split="train"
        ).shuffle(seed=42)
        test_data = load_dataset(
            "mmaisel1/nous-rl-hackathon-sigma", split="test"
        ).shuffle(seed=42)
        self.test = list()
        for item in test_data:
            self.test.append({"question": item["prompt"], "gold_answer": item["rule"]})
        self.iter = 0

    def save_checkpoint(self, step, data=None):
        if data is None:
            data = {}
        data["iter"] = self.iter
        super().save_checkpoint(step, data)

    async def llm_judge_similarity(
        self, gold_yaml_str: str, gen_yaml_str: str
    ) -> float:
        """
        Uses an LLM to decide whether the generated Sigma rule is semantically equivalent
        to the gold answer. Returns 1.0 if yes, 0.0 if no.
        """
        prompt = f"""
            You are an expert in cybersecurity and YAML-based Sigma rules. Given a gold standard
            Sigma rule and a generated rule, determine if the generated rule is functionally
            equivalent and correct.

            Only respond with "1" if the generated Sigma rule is correct and matches the intent
            and structure of the gold rule.
            Otherwise, respond with "0".

            GOLD RULE:
            {gold_yaml_str}

            GENERATED RULE:
            {gen_yaml_str}

            Answer with only one character: "1" or "0".
        """

        try:
            completion = await self.server.chat_completion(
                messages=[
                    {"role": "user", "content": prompt},
                ],
                n=1,
                max_tokens=1,
                temperature=0.0,
                split="eval",
            )
            reply = completion.choices[0].message.content.strip()
            return 1.0 if reply == "1" else 0.0

        except Exception as e:
            print(f"LLM Judge error: {e}")
            return 0.0

    async def rollout_and_score_eval(self, question: str, answer: str) -> number:
        completion = await self.server.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            n=1,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
            split="eval",
        )
        gold_parsed = parse(
            "\\boxed{" + answer + "}",
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        answer_parsed = parse(
            completion.choices[0].message.content.split("</think>")[-1],
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    # Ensures that boxed is tried first
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
        score = self.llm_judge_similarity(
            gold_parsed["detection"],
            answer_parsed["detection"],
        )
        return score

    async def evaluate(self, *args, **kwargs):
        eval_tasks = []
        for item in self.test:
            eval_tasks.append(
                self.rollout_and_score_eval(item["question"], item["gold_answer"])
            )
        scores = await tqdm_asyncio.gather(*eval_tasks)
        self.eval_metrics.append(("eval/percent_correct", sum(scores) / len(scores)))

    async def collect_trajectories(
        self, item: SigmaRuleEntry
    ) -> Tuple[ScoredDataGroup, list[Item]]:
        user_message = {"role": "user", "content": item["prompt"]}
        gold_answer = (
            "\\boxed{" + item["rule"].split("#")[-1].strip().replace(",", "") + "}"
        )

        chat_completions = await self.server.chat_completion(
            messages=[{"role": "system", "content": system_prompt}, user_message],
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
        )
        to_score = list()
        to_backlog = list()
        for i, chat_completion in enumerate(chat_completions.choices):
            messages = (
                {"role": "system", "content": system_prompt},
                user_message,
                {"role": "assistant", "content": chat_completion.message.content},
            )
            to_score.append(
                {
                    "messages": messages,
                    "gold_answer": gold_answer,
                    "finish_reason": chat_completion.finish_reason,
                }
            )
        to_postprocess = await self.score(to_score)
        return to_postprocess, to_backlog

    async def score(
        self, rollout_group_data
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup()
        scores["tokens"] = list()
        scores["masks"] = list()
        scores["scores"] = list()

        try:
            # gold_yaml = yaml.safe_load(rollout_group_data[0]["gold_answer"])
            gold_det = rollout_group_data[0]["gold_answer"]
        except yaml.YAMLError:
            reward = 0

        random.shuffle(rollout_group_data)

        for item in rollout_group_data:
            out_dict = tokenize_for_trainer(
                self.tokenizer, item["messages"], item["finish_reason"]
            )
            tokens = out_dict["tokens"]
            masks = out_dict["masks"]
            boxed_match = re.search(
                r"\\boxed\{(.*)\}\s*$", item["messages"][-1]["content"], re.DOTALL
            )
            try:
                if boxed_match:
                    yaml_str = boxed_match.group(1)
                    # gen_yaml = yaml.safe_load(yaml_str)
                    # gen_det = self.flatten_detection(gen_yaml.get("detection", {}))
                    reward = await self.llm_judge_similarity(gold_det, yaml_str)
                else:
                    reward = 0
            except Exception as e:
                print(e)
                reward = 0

            print("GOLD ANSWER:", rollout_group_data[0]["gold_answer"])
            print("GEN OUTPUT:", item["messages"][-1]["content"])
            print("REWARD:", reward)

            # Optional: Add LLM-based score for semantic evaluation (not shown here)

            # if len([1 for i in masks if i != -100]) < 10:
            #     continue

            scores["tokens"].append(tokens)
            scores["masks"].append(masks)
            scores["scores"].append(reward)

            # if len(scores["tokens"]) >= self.config.group_size:
            #     break

        # Buffer for analysis
        for score in scores["scores"]:
            self.percent_correct_buffer.append(score)

        # Optional: Apply length penalty if all rewards are 1.0
        if all(score >= 0.99 for score in scores["scores"]):  # float-safe check
            token_lengths = [len(token) for token in scores["tokens"]]
            if max(token_lengths) == 0:
                return None
            max_allowed_length = self.config.max_token_length
            length_threshold = max_allowed_length * 0.5
            scores["scores"] = []
            for length in token_lengths:
                if length <= length_threshold:
                    scores["scores"].append(1.0)
                else:
                    pct_range = (length - length_threshold) / (
                        max_allowed_length - length_threshold
                    )
                    scores["scores"].append(1.0 - min(pct_range, 1.0))

        # if all([scores["scores"][0] == score for score in scores["scores"]]):
        #     return None

        return scores if scores["tokens"] else None

    async def get_next_item(self) -> SigmaRuleEntry:
        next_item = self.train[self.iter % len(self.train)]
        self.iter += 1
        return next_item


if __name__ == "__main__":
    SigmaRuleEnv.cli()
