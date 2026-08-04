"""
Punchline VR-CLI Environment for Atropos
"""

from __future__ import annotations

import asyncio
import math
import random
from typing import Dict, List, Optional, Tuple, TypedDict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer


# ───────────────────────────────────────────────────────
# Config & data row
# ───────────────────────────────────────────────────────
class PunchlineRow(TypedDict):
    question: str
    answer: str


class PunchEnvConfig(BaseEnvConfig):
    tokenizer_name: str = "Qwen/Qwen3-1.7B"
    group_size: int = 8
    use_wandb: bool = True
    rollout_server_url: str = "http://localhost:8000"
    total_steps: int = 1000
    batch_size: int = 12
    steps_per_eval: int = 100
    max_token_length: int = 3000
    wandb_name: str = "punchline_vrcli"
    gpu_device: int = 0


class PunchlineEnv(BaseEnv):
    name = "punchline_vrcli"

    # ───────────────────────────────────────────────
    # default config + server
    # ───────────────────────────────────────────────
    @classmethod
    def config_init(cls):
        cfg = PunchEnvConfig()
        servers = [
            APIServerConfig(
                model_name="Qwen/Qwen3-1.7B",
                base_url="http://localhost:9001/v1",
                api_key="x",
                num_requests_for_eval=64,
            )
        ]
        return cfg, servers

    # ───────────────────────────────────────────────
    # wandb logging helper
    # ───────────────────────────────────────────────
    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}
        if getattr(self, "_reward_buffer", None):
            wandb_metrics["train/mean_reward"] = sum(self._reward_buffer) / len(
                self._reward_buffer
            )
            self._reward_buffer = []
        await super().wandb_log(wandb_metrics)

    # ───────────────────────────────────────────────
    # setup
    # ───────────────────────────────────────────────
    async def setup(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_name)
        self.reward_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B-Base")
        self._ref = (
            AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen3-1.7B-Base", torch_dtype=torch.bfloat16
            )
            .eval()
            .to(
                f"cuda:{self.config.gpu_device}" if torch.cuda.is_available() else "cpu"
            )
        )

        raw = load_dataset(
            "SocialGrep/one-million-reddit-jokes",
            split="train",
            trust_remote_code=True,
        )

        self.data: List[PunchlineRow] = []
        for row in raw:
            if (
                row.get("selftext", "")
                and row.get("score", 0) >= 75
                and row.get("selftext", "") not in ["[removed]", "[deleted]"]
                and row.get("title", "").strip().endswith("?")
                and (
                    row.get("title", "").strip().startswith("What")
                    or row.get("title", "").strip().startswith("Why")
                    or row.get("title", "").strip().startswith("How")
                )
                and "(" not in row.get("title", "")
                and "[" not in row.get("title", "")
                and "\n" not in row.get("selftext", "")
                and "\r" not in row.get("selftext", "")
                and ";" not in row.get("selftext", "")
            ):
                q, a = row["title"], row["selftext"]
                if q and a:
                    self.data.append({"question": q, "answer": a})
        random.shuffle(self.data)
        self._idx = 0
        self._reward_buffer: List[float] = []

    # ───────────────────────────────────────────────
    # item iterator
    # ───────────────────────────────────────────────
    async def get_next_item(self) -> PunchlineRow:
        itm = self.data[self._idx % len(self.data)]
        self._idx += 1
        return itm

    # ───────────────────────────────────────────────
    # trajectory collection
    # ───────────────────────────────────────────────
    async def collect_trajectories(
        self, item: PunchlineRow
    ) -> Tuple[ScoredDataGroup, List[Item]]:
        system_msg = {
            "role": "system",
            "content": (
                "You have a strong sense of humor and answer the user's question with a punchline. "
                "You always give the funniest answer, even if it could offend some people. "
                "Consider the aspects that make a joke funny, for example the answer is usually "
                "surprising to hear but makes sense in hindsight. You shouldn't need to explain "
                "your answer, it should stand on its own."
            ),
        }
        user_msg = {"role": "user", "content": item["question"]}

        chat_comps = await self.server.chat_completion(
            messages=[system_msg, user_msg],
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
        )

        group = ScoredDataGroup(tokens=[], masks=[], scores=[])
        for choice in chat_comps.choices:
            assistant_content = choice.message.content
            reasoning, answer = self._parse_completion(assistant_content)

            rew = self._vrcli_reward(item["question"], reasoning, item["answer"])
            self._reward_buffer.append(rew)

            msgs = (
                user_msg,
                {"role": "assistant", "content": f"<think>\n{reasoning}\n</think>"},
            )
            td = tokenize_for_trainer(self.tokenizer, msgs, choice.finish_reason)

            group["tokens"].append(td["tokens"])
            group["masks"].append(td["masks"])
            group["scores"].append(rew)

            if len(group["tokens"]) >= self.config.group_size:
                break

        if len(group["tokens"]) < self.config.group_size or all(
            s == group["scores"][0] for s in group["scores"]
        ):
            return None, []
        return group, []

    # ───────────────────────────────────────────────
    # evaluation (average reward of random samples)
    # ───────────────────────────────────────────────
    async def evaluate(self, *args, **kwargs):
        # take 64 random jokes and see mean reward with greedy decoding
        sample = random.sample(self.data, k=64)
        tasks = []
        for row in sample:
            msg = {"role": "user", "content": row["question"]}
            tasks.append(
                self.server.chat_completion(
                    messages=[msg],
                    n=1,
                    temperature=0.0,
                    max_tokens=self.config.max_token_length,
                    split="eval",
                )
            )
        completions = await asyncio.gather(*tasks)
        rewards = []
        for row, comp in zip(sample, completions):
            txt = comp.choices[0].message.content
            r, a = self._parse_completion(txt)
            rewards.append(self._vrcli_reward(row["question"], r, row["answer"]))
        self.eval_metrics.append(("eval/mean_reward", sum(rewards) / len(rewards)))

    # ───────────────────────────────────────────────
    # helpers
    # ───────────────────────────────────────────────
    def _parse_completion(self, txt: str):
        if "<think>" in txt and "</think>" in txt:
            reasoning = txt.split("<think>", 1)[1].split("</think>", 1)[0].strip()
            answer = txt.split("</think>", 1)[1].strip()
        else:
            reasoning, answer = "", txt.strip()
        return reasoning, answer

    def _vrcli_reward(self, q: str, reasoning: str, gold: str) -> float:
        if not reasoning:
            return -1.0

        t = self.reward_tokenizer

        def ppl(prompt: str, comp: str):
            ids = t(prompt + comp, return_tensors="pt").to(self._ref.device)
            p_len = t(prompt, return_tensors="pt").input_ids.size(1)
            with torch.no_grad():
                logits = self._ref(**ids).logits[:, :-1]
            targets = ids.input_ids[:, 1:]
            lp = (
                torch.log_softmax(logits, -1)
                .gather(2, targets.unsqueeze(-1))
                .squeeze(-1)
            )
            return math.exp(-lp[:, p_len:].mean().item())

        base = ppl(f"Question: {q}\nAnswer:", gold)
        plus = ppl(f"Question: {q}\nReasoning: {reasoning}\nAnswer:", gold)
        return max(0.0, (base - plus) / base)


if __name__ == "__main__":
    PunchlineEnv.cli()
