"""
KernelBench Environment Setup Instructions
----------------------------------------

Before running this script, you need to install KernelBench:

1. Install KernelBench from source:
   pip install git@github.com:ScalingIntelligence/KernelBench.git
   cd KernelBench
   pip install -r requirements.txt
   pip install -e .
   cd -

2. Set variables at the top of this script:
   KERNELBENCH_LEVEL: The difficulty level (1-3)
   KERNELBENCH_PROBLEM_NUMBER: The specific problem number to solve
   KERNELBENCH_DIR: the absolute path to your KernelBench install

These environment variables will be used to configure the evaluation environment.
"""

import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypedDict, Union

from datasets import load_dataset

# KernelBench imports
from src.eval import eval_kernel_against_ref

from atroposlib.envs.base import (
    APIServerConfig,
    BaseEnv,
    BaseEnvConfig,
    ScoredDataGroup,
)
from atroposlib.type_definitions import Item
from atroposlib.utils.tokenize_for_trainer import tokenize_for_trainer

# Set the start method to 'spawn' for CUDA compatibility
mp.set_start_method("spawn", force=True)

KERNELBENCH_DIR = Path("/path/to/KernelBench")
KERNELBENCH_LEVEL = 1
KERNELBENCH_PROBLEM_NUMBER = 1

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"


def get_kernelbench_code(level: int, problem_id: int) -> str:
    """
    Return the `code` string for a given KernelBench level/problem combo.
    Raises ValueError if the problem_id is not found in that level.
    """
    split = f"level_{level}"
    ds = load_dataset("ScalingIntelligence/KernelBench", split=split)

    # Keep only rows whose `problem_id` exactly matches the desired one
    row = ds.filter(lambda x: x["problem_id"] == problem_id)
    if len(row) == 0:
        raise ValueError(f"{problem_id=} not found in {split=}")

    return row[0]["code"]


class KBRow(TypedDict):
    """Single‑task record (prompt text plus meta)."""

    prompt: str  # full prompt given to the LLM
    sample_path: str


def evaluate_single_kernel(args):
    """Helper function to evaluate a single kernel in a process."""
    item, build_dir, ref_code = args
    generated_src = item["messages"][-1]["content"].strip("```python\n").strip("```")

    # Initialize CUDA in the child process
    import torch

    torch.cuda.init()

    eval_result = eval_kernel_against_ref(
        original_model_src=ref_code,
        custom_model_src=generated_src,
        measure_performance=True,
        verbose=True,
        num_correct_trials=1,
        num_perf_trials=1,
        build_dir=build_dir,
        device="cuda:7",
    )

    compiled_flag = bool(getattr(eval_result, "compiled", False))
    runtime_val = float(getattr(eval_result, "runtime", -1.0))
    reward = 0.3 * (1 if compiled_flag else 0) + runtime_val

    # Note: We can't use the tokenizer here since it's not pickleable
    # We'll return the raw data and tokenize in the main process
    return {
        "messages": item["messages"],
        "finish_reason": item["finish_reason"],
        "reward": reward,
    }


class KernelBenchEnv(BaseEnv):

    name = "kernelbench_parallel"

    @classmethod
    def config_init(cls) -> Tuple[BaseEnvConfig, List[APIServerConfig]]:
        env_cfg = BaseEnvConfig(
            tokenizer_name="Qwen/Qwen3-4B",
            group_size=2,
            max_token_length=2048,
            batch_size=1,
            steps_per_eval=1,
            total_steps=1000,
            rollout_server_url="http://localhost:8000",
            use_wandb=False,
            wandb_name=f"kb_level{KERNELBENCH_LEVEL}_prob{KERNELBENCH_PROBLEM_NUMBER}_parallel",
        )

        server_cfgs = [
            APIServerConfig(
                model_name="Qwen/Qwen3-4B",
                base_url="http://localhost:9001/v1",
                api_key="DUMMY_KB_KEY",
                num_requests_for_eval=64,
            )
        ]
        return env_cfg, server_cfgs

    # --------------------- Data ------------------------------------------------
    async def setup(self):
        self.problem_spec = {
            "level": KERNELBENCH_LEVEL,
            "problem_id": KERNELBENCH_PROBLEM_NUMBER,
            "problem_file": f"{KERNELBENCH_PROBLEM_NUMBER}_Square_matrix_multiplication_.py",
        }
        self.iter = 0
        with open("prompt.txt", "r", encoding="utf-8") as f:
            self.prompt = f.read()

        # Get reference code directly from the dataset
        self.ref_code = get_kernelbench_code(
            KERNELBENCH_LEVEL, KERNELBENCH_PROBLEM_NUMBER
        )
        self.reward_buffer = list()
        # Create a process pool for parallel processing
        self.pool = mp.Pool(processes=24)

    # --------------------- Rollout / scoring ----------------------------------
    async def collect_trajectories(
        self, item: KBRow
    ) -> Tuple[ScoredDataGroup, List[Item]]:
        """
        Ask the LLM `group_size` times; each completion should be *only* the
        CUDA / Triton kernel (per KernelBench docs).  We store them to
        runs/{run_name}/{level}/{id}/sample_<n>.cu so that the official
        evaluator picks them up.
        """
        user_msg = {"role": "user", "content": self.prompt}

        chat_completions = await self.server.chat_completion(
            messages=[user_msg],
            n=self.config.group_size,
            max_tokens=self.config.max_token_length,
            temperature=0.0,
        )

        # Path: runs/<RUN_NAME>/level_1/1/
        run_dir = KERNELBENCH_DIR / "runs" / "wandb" / "level_1" / "1"
        run_dir.mkdir(parents=True, exist_ok=True)

        to_score: List[Dict] = []
        to_backlog: list() = []
        for i, choice in enumerate(chat_completions.choices):
            kernel_code = choice.message.content
            sample_path = run_dir / f"sample_{i}.cu"
            sample_path.write_text(kernel_code, encoding="utf‑8")

            messages = (user_msg, {"role": "assistant", "content": kernel_code})
            to_score.append(
                {
                    "messages": messages,
                    "finish_reason": choice.finish_reason,
                }
            )

        to_postprocess = await self.score(to_score)

        return to_postprocess, to_backlog

    async def score(
        self, rollout_group_data: List[Dict]
    ) -> Union[Optional[ScoredDataGroup], List[Optional[ScoredDataGroup]]]:
        scores = ScoredDataGroup(tokens=[], masks=[], scores=[])

        # where we will build + compile kernels
        build_dir = os.path.join("build", "kernelbench", f"{1}", f"{1}")
        os.makedirs(build_dir, exist_ok=True)

        # Create arguments for parallel evaluation
        eval_args = [(item, build_dir, self.ref_code) for item in rollout_group_data]

        # Run evaluations in parallel
        results = []
        for args in eval_args:
            result = self.pool.apply_async(evaluate_single_kernel, args=(args,))
            results.append(result)

        # Wait for all evaluations to complete and process results
        for result in results:
            eval_result = result.get()  # This will wait for the result
            reward = eval_result["reward"]

            # Tokenize in the main process since tokenizer isn't pickleable
            out_dict = tokenize_for_trainer(
                self.tokenizer, eval_result["messages"], eval_result["finish_reason"]
            )

            scores["tokens"].append(out_dict["tokens"])
            scores["masks"].append(out_dict["masks"])
            scores["scores"].append(reward)
            self.reward_buffer.append(max(reward, 0))

        return scores if scores["tokens"] else None

    async def get_next_item(self) -> KBRow:
        """Return the same single problem every time (env is tiny)."""
        return KBRow(
            prompt=self.prompt, sample_path=""
        )  # sample_path is no longer used

    async def evaluate(self, *args, **kwargs):
        """Evaluate the current model on a set of test problems."""
        if self.reward_buffer:
            avg_reward = sum(self.reward_buffer) / len(self.reward_buffer)
            self.eval_metrics.append(("eval/avg_reward", avg_reward))
            self.reward_buffer = list()

    async def wandb_log(self, wandb_metrics: Optional[Dict] = None):
        if wandb_metrics is None:
            wandb_metrics = {}

        try:
            wandb_metrics["train/reward"] = sum(self.reward_buffer) / len(
                self.reward_buffer
            )
        except ZeroDivisionError:
            pass

        self.reward_buffer = list()
        self.eval_metrics = list()
        await super().wandb_log(wandb_metrics)

    async def cleanup(self):
        """Clean up resources when done."""
        self.pool.close()
        self.pool.join()
        await super().cleanup()


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    KernelBenchEnv.cli()
