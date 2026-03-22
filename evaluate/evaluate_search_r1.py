
"""
1. update merge and evaluate March 20

  python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /ocean/projects/med230010p/yji3/BrowseCamp/verl/checkpoints/search_r1_like_async_rl/qwen2.5-7b-combined-search-checker-no-triage-explicitcheck-20-13-17/global_step_188/actor \
    --target_dir /ocean/projects/med230010p/yji3/BrowseCamp/verl/merged_models/qwen2.5-7b-combined-search-checker-no-triage-explicitcheck-20-13-17-step188


2.update merge and evaluate March 20 


 unset ROCR_VISIBLE_DEVICES
  module load cuda
  export PYTHONPATH=/ocean/projects/med230010p/yji3/BrowseCamp/verl:$PYTHONPATH
  export CUDA_HOME=/opt/packages/cuda/v12.6.1
  export CUDA_PATH=/opt/packages/cuda/v12.6.1
  export LD_LIBRARY_PATH=/opt/packages/cuda/v12.6.1/lib64:/opt/packages/cuda/v12.6.1/nvvm/lib64:/opt/packages/cuda/v12.6.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH
  export PATH=/opt/packages/cuda/v12.6.1/bin:$PATH
  CUDA_VISIBLE_DEVICES=1 python evaluate/evaluate_search_r1.py \
    --repo_root /ocean/projects/med230010p/yji3/BrowseCamp/verl \
    --model_path /ocean/projects/med230010p/yji3/BrowseCamp/verl/merged_models/qwen2.5-7b-combined-search-checker-no-triage-explicitcheck-20-13-17-step188 \
    --test_file /ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/test.parquet \
    --max_samples 100 \
    --eval_batch_size 4 \
    --output_file /ocean/projects/med230010p/yji3/BrowseCamp/verl/eval_no_triage_explicitcheck_step188.json \
    --tool_count_mode both \
    --tag_style auto \
    --prompt_mode explicit_check \
    --tool_config_path /ocean/projects/med230010p/yji3/BrowseCamp/verl/examples/sglang_multiturn/config/tool_config/medical_search_checker_tool_config.yaml \
    --multi_turn_format search_r1_with_checker \
    --tensor_parallel_size 1 \
    --nnodes 1 \
    --n_gpus_per_node 1 \
    --gpu_memory_utilization 0.4 \
    --max_model_len 8000 \
    --max_prompt_length 3072 \
    --max_response_length 2000 \
    --max_assistant_turns 7
这里是结果 AgentLoopWorker pid=5945) WARNING:2026-03-20 16:10:24,660:[CHECKER] answer=Low serum levels of vitamin D are associated with post-stroke depression. supports=0 contradictions=0 neutrals=1 num_claims=1 avg_confidence=0.500 reward=0.100

============================================================
Evaluation Results
============================================================
num_samples: 87
candidate_samples: 88
filtered_samples: 1
f1_mean: 0.1974
em_mean: 0.0000
fuzzy_accuracy: 0.0000
avg_searches: 0.2069
avg_checks: 1.1264
avg_explicit_checks: 0.3333
avg_auto_checks: 0.8046
avg_tools: 1.3333
avg_thinks: 0.0000
avg_turns: 4.6667
avg_search_budget_utilization: 0.0000
avg_check_budget_utilization: 0.0000
avg_turn_budget_utilization: 0.0000
search_budget_saturation_rate: 0.0000
check_budget_saturation_rate: 0.0000
turn_budget_saturation_rate: 0.0000
has_answer_tag_rate: 0.9540
samples_with_search: 0.1839
samples_with_check: 0.9770
samples_with_explicit_check: 0.1724
samples_with_tool: 0.9770
tool_count_mode: both
tag_style: auto
eval_mode: agent_loop

Results saved to /ocean/projects/med230010p/yji3/BrowseCamp/verl/eval_no_triage_explicitcheck_step188.json

============================================================
excitedshrub705

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /ocean/projects/med230010p/yji3/BrowseCamp/verl/checkpoints/search_r1_like_async_rl/qwen2.5-7b-combined-search-checker-
  triage-explicitcheck-20-10-41/global_step_188/actor \
    --target_dir /ocean/projects/med230010p/yji3/BrowseCamp/verl/merged_models/qwen2.5-7b-combined-search-checker-triage-explicitcheck-20-
  10-41-step188


    

  
CUDA_VISIBLE_DEVICES=1 python -m verl.model_merger merge     --backend fsdp --local_dir /ocean/projects/med230010p/yji3/BrowseCamp/verl/checkpoints/search_r1_like_async_rl/qwen2.5-7b-combined-search-checker-triage-15-17-50/global_step_93/actor --target_dir merged_qwen2.5_7b_combined_search_checker_triage_step_93


CUDA_VISIBLE_DEVICES=1 python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /ocean/projects/med230010p/yji3/BrowseCamp/verl/checkpoints/search_r1_like_async_rl/qwen2.5-7b-combined-search-checker-no-triage-15-19-18/global_step_93/actor \
  --target_dir merged_qwen2.5_7b_combined_search_checker_no_triage_step_93



CUDA_VISIBLE_DEVICES=1 python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /ocean/projects/med230010p/yji3/BrowseCamp/verl/checkpoints/search_r1_like_async_rl/qwen2.5-7b-combined-search-only-no-triage-14-22-04/global_step_93/actor \
  --target_dir merged_qwen2.5_7b_combined_search_only_no_triage_step_93


CUDA_VISIBLE_DEVICES=1 python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir /ocean/projects/med230010p/yji3/BrowseCamp/verl/checkpoints/search_r1_like_async_rl/qwen2.5-7b-combined-search-only-triage-14-23-57/global_step_93/actor \
  --target_dir merged_qwen2.5_7b_combined_search_only_triage_step_93


 unset ROCR_VISIBLE_DEVICES
  module load cuda
  export PYTHONPATH=/ocean/projects/med230010p/yji3/BrowseCamp/verl:$PYTHONPATH
  export CUDA_HOME=/opt/packages/cuda/v12.6.1
  export CUDA_PATH=/opt/packages/cuda/v12.6.1
  export LD_LIBRARY_PATH=/opt/packages/cuda/v12.6.1/lib64:/opt/packages/cuda/v12.6.1/nvvm/lib64:/opt/packages/cuda/v12.6.1/extras/CUPTI/lib64:$LD_LIBRARY_PATH
  export PATH=/opt/packages/cuda/v12.6.1/bin:$PATH

CUDA_VISIBLE_DEVICES=1 python evaluate/evaluate_search_r1.py \
    --repo_root /ocean/projects/med230010p/yji3/BrowseCamp/verl \
    --model_path merged_qwen2.5_7b_combined_search_checker_triage_step_93 \
    --test_file /ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/test.parquet \
    --max_samples 100 \
    --eval_batch_size 8 \
    --output_file eval_combined_search_checker_triage_step93.json \
    --tool_count_mode both \
    --tag_style auto \
    --tool_config_path /ocean/projects/med230010p/yji3/BrowseCamp/verl/examples/sglang_multiturn/config/tool_config/medical_search_checker_tool_config.yaml \
    --multi_turn_format search_r1_with_checker \
    --tensor_parallel_size 1 \
    --nnodes 1 \
    --n_gpus_per_node 1 \
    --gpu_memory_utilization 0.4 \
    --max_model_len 8000 \
    --max_prompt_length 2048 \
    --max_response_length 2000 \
    --max_assistant_turns 7 \
    --enable_triage \
    --online_escalation



 1. checker no triage:

  CUDA_VISIBLE_DEVICES=3 python evaluate/evaluate_search_r1.py \
    --repo_root /ocean/projects/med230010p/yji3/BrowseCamp/verl \
    --model_path merged_qwen2.5_7b_combined_search_checker_no_triage_step_93 \
    --test_file /ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/test.parquet \
    --max_samples 100 \
    --eval_batch_size 4 \
    --output_file eval_checker_no_triage.json \
    --tool_count_mode both \
    --tag_style auto \
    --prompt_mode clean \
    --tool_config_path /ocean/projects/med230010p/yji3/BrowseCamp/verl/examples/sglang_multiturn/config/tool_config/medical_search_checker_tool_config.yaml \
    --multi_turn_format search_r1_with_checker \
    --tensor_parallel_size 1 \
    --nnodes 1 \
    --n_gpus_per_node 1 \
    --gpu_memory_utilization 0.4 \
    --max_model_len 8000 \
    --max_prompt_length 2048 \
    --max_response_length 2000 \
    --max_assistant_turns 7

结果 
============================================================
Evaluation Results
============================================================
num_samples: 87
candidate_samples: 88
filtered_samples: 1
f1_mean: 0.2100
em_mean: 0.0000
fuzzy_accuracy: 0.0000
avg_searches: 0.0460
avg_checks: 0.0000
avg_auto_checks: 0.0000
avg_tools: 0.0460
avg_thinks: 0.2299
avg_turns: 2.0920
has_answer_tag_rate: 0.9885
samples_with_search: 0.0460
samples_with_check: 0.0000
samples_with_tool: 0.0460
tool_count_mode: both
tag_style: auto
eval_mode: agent_loop

Results saved to eval_checker_no_triage.json

============================================================
Sample Results
============================================================

--- Sample 1 ---
Q: Hey so ive been extremely worried for the past day that i might have prion disease/am in the incubation period, the reas...
Golden: ["You don't have a prion disease. Get help for your anxiety."]
Model: Prion diseases are rare and it is highly unlikely that you have prion disease based on your symptoms. Psoriasis is a skin condition and there is no evidence to ...
Searches: 0, Checks: 0, Tools(both): 0, Turns: 2, F1: 0.242, EM: 0.0

--- Sample 2 ---
Q: 19yo F here, 60kg, 5'5'', very healthy, religious and virgin.

I have been researching the HPV gardasil shot and am real...
Golden: ["Honestly I have no idea why the uptake of HPV vaccine is so low. Rates of cervical cancer have dramatically fallen. You can't control what may happen in the future with your relationship. To me it is a no brainer. I will even be taking my son for it when he is old enough."]
Model: The HPV vaccine, Gardasil, has been extensively studied and is considered safe by medical experts. While no vaccine is 100% risk-free, the benefits of the HPV v...
Searches: 0, Checks: 0, Tools(both): 0, Turns: 2, F1: 0.219, EM: 0.0

--- Sample 3 ---
Q: Is what my primary doctor told me over the phone as he blew off the symptoms I've been having. Let me first explain i do...
Golden: ["26 year olds can have heart problems of course.  That said, it is less likely, and doesn't fit perfectly with your symptoms."]
Model: When visiting your doctor, it's important to be clear and concise in describing your symptoms. Start by mentioning the duration and severity of your symptoms. E...
Searches: 0, Checks: 0, Tools(both): 0, Turns: 2, F1: 0.132, EM: 0.0

 2. checker triage:

  CUDA_VISIBLE_DEVICES=1 python evaluate/evaluate_search_r1.py \
    --repo_root /ocean/projects/med230010p/yji3/BrowseCamp/verl \
    --model_path merged_qwen2.5_7b_combined_search_checker_triage_step_93 \
    --test_file /ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/test.parquet \
    --max_samples 100 \
    --eval_batch_size 4 \
    --output_file eval_checker_triage.json \
    --tool_count_mode both \
    --tag_style auto \
    --prompt_mode clean \
    --tool_config_path /ocean/projects/med230010p/yji3/BrowseCamp/verl/examples/sglang_multiturn/config/tool_config/medical_search_checker_tool_config.yaml \
    --multi_turn_format search_r1_with_checker \
    --tensor_parallel_size 1 \
    --nnodes 1 \
    --n_gpus_per_node 1 \
    --gpu_memory_utilization 0.4 \
    --max_model_len 8000 \
    --max_prompt_length 2048 \
    --max_response_length 2000 \
    --max_assistant_turns 7 \
    --enable_triage \
    --online_escalation

结果


============================================================
Evaluation Results
============================================================
num_samples: 87
candidate_samples: 88
filtered_samples: 1
f1_mean: 0.1637
em_mean: 0.0000
fuzzy_accuracy: 0.0000
avg_searches: 0.3333
avg_checks: 0.3333
avg_auto_checks: 0.3333
avg_tools: 0.6667
avg_thinks: 0.0000
avg_turns: 3.3333
has_answer_tag_rate: 0.7126
samples_with_search: 0.3333
samples_with_check: 0.3333
samples_with_tool: 0.3333
tool_count_mode: both
tag_style: auto
eval_mode: agent_loop

Results saved to eval_force_search.json

============================================================
Sample Results
============================================================

--- Sample 1 ---
Q: Hey so ive been extremely worried for the past day that i might have prion disease/am in the incubation period, the reas...
Golden: ["You don't have a prion disease. Get help for your anxiety."]
Model: Prion disease is a rare neurological disorder that is caused by misfolded prion proteins. While it is true that consuming prions can increase the risk of prion ...
Searches: 0, Checks: 0, Tools(both): 0, Turns: 2, F1: 0.160, EM: 0.0

--- Sample 2 ---
Q: 19yo F here, 60kg, 5'5'', very healthy, religious and virgin.

I have been researching the HPV gardasil shot and am real...
Golden: ["Honestly I have no idea why the uptake of HPV vaccine is so low. Rates of cervical cancer have dramatically fallen. You can't control what may happen in the future with your relationship. To me it is a no brainer. I will even be taking my son for it when he is old enough."]
Model: The HPV Gardasil shot is generally considered safe, and the benefits of preventing HPV infection and related cancers and diseases outweigh the risks for most pe...
Searches: 0, Checks: 0, Tools(both): 0, Turns: 2, F1: 0.208, EM: 0.0

--- Sample 3 ---
Q: Is what my primary doctor told me over the phone as he blew off the symptoms I've been having. Let me first explain i do...
Golden: ["26 year olds can have heart problems of course.  That said, it is less likely, and doesn't fit perfectly with your symptoms."]
Model: To convey your concerns to your doctor, it's important to be clear and concise. You can start by summarizing your symptoms and how they have been affecting your...
Searches: 0, Checks: 0, Tools(both): 0, Turns: 2, F1: 0.123, EM: 0.0
 3. search only no triage:

  CUDA_VISIBLE_DEVICES=0 python evaluate/evaluate_search_r1.py \
    --repo_root /ocean/projects/med230010p/yji3/BrowseCamp/verl \
    --model_path merged_qwen2.5_7b_combined_search_only_no_triage_step_93 \
    --test_file /ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/test.parquet \
    --max_samples 100 \
    --eval_batch_size 4 \
    --output_file eval_search_only_no_triage.json \
    --tool_count_mode both \
    --tag_style auto \
    --prompt_mode clean \
    --tool_config_path /ocean/projects/med230010p/yji3/BrowseCamp/verl/examples/sglang_multiturn/config/tool_config/medical_search_tool_config.yaml \
    --multi_turn_format search_r1 \
    --tensor_parallel_size 1 \
    --nnodes 1 \
    --n_gpus_per_node 1 \
    --gpu_memory_utilization 0.4 \
    --max_model_len 8000 \
    --max_prompt_length 2048 \
    --max_response_length 2000 \
    --max_assistant_turns 5
    ============================================================
Evaluation Results
============================================================
num_samples: 87
candidate_samples: 88
filtered_samples: 1
f1_mean: 0.1903
em_mean: 0.0000
fuzzy_accuracy: 0.0000
avg_searches: 0.6552
avg_checks: 0.0000
avg_auto_checks: 0.0000
avg_tools: 0.6552
avg_thinks: 1.6322
avg_turns: 3.3103
has_answer_tag_rate: 0.5862
samples_with_search: 0.2644
samples_with_check: 0.0000
samples_with_tool: 0.2644
tool_count_mode: both
tag_style: auto
eval_mode: agent_loop

Results saved to eval_search_only_no_triage.json

============================================================
Sample Results
============================================================

--- Sample 1 ---
Q: Hey so ive been extremely worried for the past day that i might have prion disease/am in the incubation period, the reas...
Golden: ["You don't have a prion disease. Get help for your anxiety."]
Model: Based on the search results, there is no evidence to support the idea that psoriasis contains prions or that consuming psoriasis flakes poses a significant risk...
Searches: 1, Checks: 0, Tools(both): 1, Turns: 4, F1: 0.131, EM: 0.0

--- Sample 2 ---
Q: 19yo F here, 60kg, 5'5'', very healthy, religious and virgin.

I have been researching the HPV gardasil shot and am real...
Golden: ["Honestly I have no idea why the uptake of HPV vaccine is so low. Rates of cervical cancer have dramatically fallen. You can't control what may happen in the future with your relationship. To me it is a no brainer. I will even be taking my son for it when he is old enough."]
Model: After searching for information on the HPV vaccine safety and Gardasil adverse events, I found that the vaccine is generally considered safe and effective. The ...
Searches: 0, Checks: 0, Tools(both): 0, Turns: 2, F1: 0.208, EM: 0.0

--- Sample 3 ---
Q: Is what my primary doctor told me over the phone as he blew off the symptoms I've been having. Let me first explain i do...
Golden: ["26 year olds can have heart problems of course.  That said, it is less likely, and doesn't fit perfectly with your symptoms."]
Model: To convey your symptoms to your doctor, try to be as specific and concise as possible. Mention the onset of symptoms, their severity, and any triggers. Here is ...
Searches: 1, Checks: 0, Tools(both): 1, Turns: 4, F1: 0.124, EM: 0.0

 4. search only triage:

  CUDA_VISIBLE_DEVICES=1 python evaluate/evaluate_search_r1.py \
    --repo_root /ocean/projects/med230010p/yji3/BrowseCamp/verl \
    --model_path merged_qwen2.5_7b_combined_search_only_triage_step_93 \
    --test_file /ocean/projects/med230010p/yji3/MedicalRagChecker/verl/searchr1_data/combined__medical/test.parquet \
    --max_samples 100 \
    --eval_batch_size 4 \
    --output_file eval_search_only_triage.json \
    --tool_count_mode both \
    --tag_style auto \
    --prompt_mode clean \
    --tool_config_path /ocean/projects/med230010p/yji3/BrowseCamp/verl/examples/sglang_multiturn/config/tool_config/medical_search_tool_config.yaml \
    --multi_turn_format search_r1 \
    --tensor_parallel_size 1 \
    --nnodes 1 \
    --n_gpus_per_node 1 \
    --gpu_memory_utilization 0.4 \
    --max_model_len 8000 \
    --max_prompt_length 2048 \
    --max_response_length 2000 \
    --max_assistant_turns 5 \
    --enable_triage \
    --online_escalation
结果
============================================================
Evaluation Results
============================================================
num_samples: 87
candidate_samples: 88
filtered_samples: 1
f1_mean: 0.1157
em_mean: 0.0000
fuzzy_accuracy: 0.0000
avg_searches: 0.0000
avg_checks: 0.0000
avg_auto_checks: 0.0000
avg_tools: 0.0000
avg_thinks: 0.0000
avg_turns: 2.0000
has_answer_tag_rate: 0.0575
samples_with_search: 0.0000
samples_with_check: 0.0000
samples_with_tool: 0.0000
tool_count_mode: both
tag_style: auto
eval_mode: agent_loop

Results saved to eval_search_only_triage.json

============================================================
Sample Results
============================================================

--- Sample 1 ---
Q: Hey so ive been extremely worried for the past day that i might have prion disease/am in the incubation period, the reas...
Golden: ["You don't have a prion disease. Get help for your anxiety."]
Model: <search query_list=["prion disease incubation period", "psoriasis and prions", "eating prions from own species risk"]>...
Searches: 0, Checks: 0, Tools(both): 0, Turns: 2, F1: 0.000, EM: 0.0

--- Sample 2 ---
Q: 19yo F here, 60kg, 5'5'', very healthy, religious and virgin.

I have been researching the HPV gardasil shot and am real...
Golden: ["Honestly I have no idea why the uptake of HPV vaccine is so low. Rates of cervical cancer have dramatically fallen. You can't control what may happen in the future with your relationship. To me it is a no brainer. I will even be taking my son for it when he is old enough."]
Model: <search query_list=["HPV Gardasil shot safety", "HPV Gardasil shot adverse events", "HPV Gardasil shot risks vs benefits"] />...
Searches: 0, Checks: 0, Tools(both): 0, Turns: 2, F1: 0.000, EM: 0.0

--- Sample 3 ---
Q: Is what my primary doctor told me over the phone as he blew off the symptoms I've been having. Let me first explain i do...
Golden: ["26 year olds can have heart problems of course.  That said, it is less likely, and doesn't fit perfectly with your symptoms."]
Model: <search query_list=["shortness of breath dry cough dizziness fast heart rate heart skipping beats loss of appetite bloatedness asthma seasonal allergies"]>...
Searches: 0, Checks: 0, Tools(both): 0, Turns: 2, F1: 0.100, EM: 0.0

    
"""
import argparse
import asyncio
import json
import os
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import ray
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer

from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import DataProto
from verl.tools.utils.tool_registry import initialize_tools_from_config


def find_repo_root(start_path: Path) -> Path:
    """Walk upward until we find the repo root that contains both `verl/` and `examples/`."""
    start = start_path.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "verl").is_dir() and (candidate / "examples").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate repo root from {start_path}")


REPO_ROOT = find_repo_root(Path(__file__).parent)
DEFAULT_TOOL_CONFIG = (
    REPO_ROOT / "examples/sglang_multiturn/config/tool_config/medical_search_checker_tool_config.yaml"
)
DEFAULT_CONFIG_DIR = REPO_ROOT / "verl/trainer/config"


class SearchR1Evaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.repo_root = find_repo_root(Path(__file__).parent if args.repo_root is None else Path(args.repo_root))
        self.config_dir = self.repo_root / "verl/trainer/config"
        self.tool_config_path = self._resolve_tool_config_path(args.tool_config_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        self.tool_schemas = self._load_tool_schemas()
        self.tool_count_mode = args.tool_count_mode
        self.tag_style = "auto" if args.tag_style == "both" else args.tag_style
        self.config = self._build_config(args)
        self.agent_loop_manager: AgentLoopManager | None = None

    def _resolve_tool_config_path(self, tool_config_path: str) -> str:
        candidate = Path(tool_config_path).expanduser()
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)
        if candidate.exists():
            return str(candidate.resolve())
        repo_relative = self.repo_root / tool_config_path
        if repo_relative.exists():
            return str(repo_relative.resolve())
        return str(candidate)

    def _load_tool_schemas(self) -> list[dict[str, Any]] | None:
        if not self.tool_config_path:
            return None
        try:
            tool_list = initialize_tools_from_config(self.tool_config_path)
            return [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        except Exception as exc:
            print(f"Warning: failed to initialize tools from {self.tool_config_path}: {exc}")
            return None

    @staticmethod
    def _normalize_content(value: Any) -> Any:
        if hasattr(value, "tolist"):
            value = value.tolist()
        return value

    def _normalize_prompt(self, prompt: Any, question: str) -> list[dict[str, Any]]:
        prompt = self._normalize_content(prompt)
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
            return prompt
        if isinstance(prompt, str) and prompt.strip():
            return [{"role": "user", "content": prompt}]
        if question.strip():
            return [{"role": "user", "content": question}]
        return [{"role": "user", "content": ""}]

    def _build_inference_prompt(self, question: str) -> list[dict[str, Any]]:
        question = question.strip()
        if self.args.prompt_mode == "force_search":
            system_prompt = (
                "Answer the given medical question. "
                "You must first reason inside <think> and </think>. "
                "For factual medical questions, first output exactly one <search>query</search> tag before answering. "
                "Do not answer directly before producing a search query. "
                "After enough information is available, provide the final answer inside <answer> and </answer>."
            )
        elif self.args.prompt_mode == "explicit_check":
            system_prompt = (
                "Answer the given medical question. "
                "You may use tools before the final answer. "
                "Use <search> query </search> when the question requires factual medical evidence. "
                "After search, if your draft answer involves diagnosis, treatment, prognosis, medication safety, "
                "interactions, contraindications, dosing, or if evidence may be incomplete or conflicting, "
                "you should verify the draft answer with <check> draft answer </check> before the final answer. "
                "Do not check the search query itself; check a candidate answer. "
                "Provide the final answer inside <answer> and </answer>."
            )
        else:
            system_prompt = (
                "Answer the given medical question. "
                "You must first reason inside <think> and </think>. "
                "For medical questions involving diagnosis, treatment, prognosis, medication effects, drug safety, "
                "interactions, contraindications, dosing, or other factual uncertainty, you should use "
                "<search> query </search> before answering. "
                "Do not skip search when external medical knowledge would improve reliability. "
                "After you have enough information, provide the final answer inside <answer> and </answer>."
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

    def _extract_question(self, row: pd.Series) -> str:
        if "question" in row and row["question"] is not None:
            return str(row["question"])
        prompt = self._normalize_content(row.get("prompt", None))
        if isinstance(prompt, list):
            for msg in prompt:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        parts = [str(item.get("text", "")) for item in content if isinstance(item, dict)]
                        return "\n".join([p for p in parts if p])
        extra = self._normalize_content(row.get("extra_info", None))
        if isinstance(extra, dict):
            ik = extra.get("interaction_kwargs", {})
            if isinstance(ik, dict) and ik.get("question") is not None:
                return str(ik["question"])
        return ""

    def _extract_golden_answers(self, row: pd.Series) -> list[str]:
        if "golden_answers" in row and row["golden_answers"] is not None:
            value = self._normalize_content(row["golden_answers"])
            if isinstance(value, list):
                vals = [str(x) for x in value if x is not None]
                return vals if vals else [""]
            return [str(value)]
        rm = self._normalize_content(row.get("reward_model", None))
        if isinstance(rm, dict):
            gt = rm.get("ground_truth", {})
            if isinstance(gt, dict) and "target" in gt:
                target = self._normalize_content(gt["target"])
                if isinstance(target, list):
                    vals = [str(x) for x in target if x is not None]
                    return vals if vals else [""]
                return [str(target)]
        return [""]

    def extract_tags(self, text: str) -> dict[str, Any]:
        xml_searches = []
        for match in re.finditer(r"<search(?P<attrs>\s+[^>]*)?>(?P<body>.*?)</search>", text, re.DOTALL | re.IGNORECASE):
            attrs = match.group("attrs") or ""
            body = match.group("body") or ""
            attr_match = re.search(r"""query\s*=\s*["'](.*?)["']""", attrs, re.DOTALL | re.IGNORECASE)
            query = attr_match.group(1).strip() if attr_match else body.strip()
            if query:
                xml_searches.append(query)
        xml_checks = re.findall(r"<check>(.*?)</check>", text, re.DOTALL | re.IGNORECASE)
        thinks = re.findall(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        prefix_searches = [m.strip() for m in re.findall(r"(?im)^\s*search\s*:\s*(.+?)\s*$", text) if m.strip()]
        prefix_checks = [m.strip() for m in re.findall(r"(?im)^\s*check\s*:\s*(.+?)\s*$", text) if m.strip()]

        if self.tag_style == "xml":
            searches = xml_searches
            checks = xml_checks
        elif self.tag_style == "prefix":
            searches = prefix_searches
            checks = prefix_checks
        else:
            searches = xml_searches + prefix_searches
            checks = xml_checks + prefix_checks

        answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        return {
            "thinks": thinks,
            "searches": searches,
            "checks": checks,
            "answer": answer_match.group(1).strip() if answer_match else None,
        }

    @staticmethod
    def extract_checker_stats(text: str) -> dict[str, Any]:
        def _find_ints(pattern: str, block: str) -> list[int]:
            return [int(x) for x in re.findall(pattern, block)]

        def _find_floats(pattern: str, block: str) -> list[float]:
            return [float(x) for x in re.findall(pattern, block)]

        tool_blocks = re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", text, re.DOTALL | re.IGNORECASE)
        summary_blocks = []
        for block in tool_blocks:
            if any(key in block for key in ("'supports':", '"supports":', "'contradictions':", '"contradictions":')):
                summary_blocks.append(block.split("'raw':", 1)[0].split('"raw":', 1)[0])

        supports: list[int] = []
        contradictions: list[int] = []
        neutrals: list[int] = []
        num_claims: list[int] = []
        support_rates: list[float] = []
        contradiction_rates: list[float] = []
        avg_confidences: list[float] = []

        for block in summary_blocks:
            supports.extend(_find_ints(r"""['"]supports['"]\s*:\s*(\d+)""", block))
            contradictions.extend(_find_ints(r"""['"]contradictions['"]\s*:\s*(\d+)""", block))
            neutrals.extend(_find_ints(r"""['"]neutrals['"]\s*:\s*(\d+)""", block))
            num_claims.extend(_find_ints(r"""['"]num_claims['"]\s*:\s*(\d+)""", block))
            support_rates.extend(_find_floats(r"""['"]support_rate['"]\s*:\s*([0-9]*\.?[0-9]+)""", block))
            contradiction_rates.extend(
                _find_floats(r"""['"]contradiction_rate['"]\s*:\s*([0-9]*\.?[0-9]+)""", block)
            )
            avg_confidences.extend(_find_floats(r"""['"]avg_confidence['"]\s*:\s*([0-9]*\.?[0-9]+)""", block))

        checker_calls = max(
            len(summary_blocks),
            len(supports),
            len(contradictions),
            len(neutrals),
            0,
        )

        total_supports = sum(supports)
        total_contradictions = sum(contradictions)
        total_neutrals = sum(neutrals)
        total_claims = sum(num_claims)

        return {
            "checker_calls_parsed": checker_calls,
            "checker_total_supports": total_supports,
            "checker_total_contradictions": total_contradictions,
            "checker_total_neutrals": total_neutrals,
            "checker_total_claims": total_claims,
            "checker_mean_support_rate": statistics.mean(support_rates) if support_rates else None,
            "checker_mean_contradiction_rate": (
                statistics.mean(contradiction_rates) if contradiction_rates else None
            ),
            "checker_mean_avg_confidence": statistics.mean(avg_confidences) if avg_confidences else None,
            "checker_has_support_signal": total_supports > 0,
            "checker_has_contradiction_signal": total_contradictions > 0,
            "checker_only_neutral": (
                checker_calls > 0 and total_supports == 0 and total_contradictions == 0 and total_neutrals > 0
            ),
        }

    def get_num_tools(self, tags: dict[str, Any]) -> int:
        num_searches = len(tags["searches"])
        num_checks = len(tags["checks"])
        if self.tool_count_mode == "search":
            return num_searches
        if self.tool_count_mode == "check":
            return num_checks
        return num_searches + num_checks

    @staticmethod
    def compute_f1(prediction: str, ground_truth: str) -> float:
        pred_tokens = set(prediction.lower().split())
        gold_tokens = set(ground_truth.lower().split())
        if not pred_tokens or not gold_tokens:
            return 0.0
        common = pred_tokens & gold_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common) / len(gold_tokens) if gold_tokens else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def compute_em(prediction: str, ground_truth: str) -> float:
        return float(prediction.lower().strip() == ground_truth.lower().strip())

    @staticmethod
    def _safe_row_dict(row: pd.Series, key: str, default: Any) -> Any:
        value = row.get(key, default)
        if value is None:
            return default
        if hasattr(value, "tolist"):
            value = value.tolist()
        return value

    def _compute_prompt_length(self, prompt_messages: list[dict[str, Any]]) -> int:
        apply_kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
        }
        if self.tool_schemas is not None:
            apply_kwargs["tools"] = self.tool_schemas

        tokenized = self.tokenizer.apply_chat_template(prompt_messages, **apply_kwargs)
        if hasattr(tokenized, "shape"):
            return int(tokenized.shape[-1])
        return len(tokenized)

    def _filter_overlong_rows(
        self, test_data: pd.DataFrame, max_samples: int | None
    ) -> tuple[list[pd.Series], dict[str, Any]]:
        candidate_count = min(len(test_data), max_samples) if max_samples else len(test_data)
        kept_rows: list[pd.Series] = []
        filtered_indices: list[int] = []
        filtered_lengths: list[int] = []

        for idx in range(candidate_count):
            row = test_data.iloc[idx]
            question = self._extract_question(row)
            prompt = self._build_inference_prompt(question)
            prompt_len = self._compute_prompt_length(prompt)
            if prompt_len > self.args.max_prompt_length:
                filtered_indices.append(idx)
                filtered_lengths.append(prompt_len)
                continue
            kept_rows.append(row)

        return kept_rows, {
            "candidate_samples": candidate_count,
            "kept_samples": len(kept_rows),
            "filtered_samples": len(filtered_indices),
            "filtered_indices": filtered_indices,
            "filtered_lengths": filtered_lengths,
        }

    def _build_config(self, args: argparse.Namespace):
        overrides = [
            f"actor_rollout_ref.model.path={args.model_path}",
            f"actor_rollout_ref.model.trust_remote_code={str(args.trust_remote_code)}",
            f"actor_rollout_ref.rollout.name={args.rollout_backend}",
            "actor_rollout_ref.rollout.mode=async",
            f"actor_rollout_ref.rollout.nnodes={args.nnodes}",
            f"actor_rollout_ref.rollout.n_gpus_per_node={args.n_gpus_per_node}",
            f"trainer.n_gpus_per_node={args.n_gpus_per_node}",
            f"trainer.nnodes={args.nnodes}",
            f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.tensor_parallel_size}",
            "actor_rollout_ref.rollout.data_parallel_size=1",
            "actor_rollout_ref.rollout.pipeline_model_parallel_size=1",
            "actor_rollout_ref.rollout.load_format=auto",
            f"actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_memory_utilization}",
            f"actor_rollout_ref.rollout.max_model_len={args.max_model_len}",
            f"actor_rollout_ref.rollout.prompt_length={args.max_prompt_length}",
            f"actor_rollout_ref.rollout.response_length={args.max_response_length}",
            "actor_rollout_ref.rollout.n=1",
            f"actor_rollout_ref.rollout.temperature={args.temperature}",
            f"actor_rollout_ref.rollout.top_p={args.top_p}",
            f"actor_rollout_ref.rollout.top_k={args.top_k}",
            f"actor_rollout_ref.rollout.do_sample={str(args.do_sample)}",
            f"actor_rollout_ref.rollout.enforce_eager={str(args.enforce_eager)}",
            "actor_rollout_ref.rollout.skip_tokenizer_init=False",
            "actor_rollout_ref.rollout.disable_log_stats=True",
            f"actor_rollout_ref.rollout.agent.num_workers={args.agent_num_workers}",
            "actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent",
            "actor_rollout_ref.rollout.multi_turn.enable=True",
            f"actor_rollout_ref.rollout.multi_turn.format={args.multi_turn_format}",
            f"actor_rollout_ref.rollout.multi_turn.max_assistant_turns={args.max_assistant_turns}",
            f"actor_rollout_ref.rollout.multi_turn.max_parallel_calls={args.max_parallel_calls}",
            f"actor_rollout_ref.rollout.multi_turn.max_tool_response_length={args.max_tool_response_length}",
            f"actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side={args.tool_response_truncate_side}",
            "actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=False",
            f"actor_rollout_ref.rollout.multi_turn.tool_config_path={self.tool_config_path}",
            f"actor_rollout_ref.rollout.val_kwargs.temperature={args.val_temperature}",
            f"actor_rollout_ref.rollout.val_kwargs.top_p={args.val_top_p}",
            f"actor_rollout_ref.rollout.val_kwargs.top_k={args.val_top_k}",
            f"actor_rollout_ref.rollout.val_kwargs.do_sample={str(args.val_do_sample)}",
            "reward.reward_model.enable=False",
        ]

        if args.enable_triage:
            overrides.extend(
                [
                    "+actor_rollout_ref.rollout.multi_turn.triage.enable=True",
                    f"+actor_rollout_ref.rollout.multi_turn.triage.online_escalation={str(args.online_escalation)}",
                    f"+actor_rollout_ref.rollout.multi_turn.triage.budget.easy.max_search={args.easy_max_search}",
                    f"+actor_rollout_ref.rollout.multi_turn.triage.budget.easy.max_check={args.easy_max_check}",
                    f"+actor_rollout_ref.rollout.multi_turn.triage.budget.easy.max_turn={args.easy_max_turn}",
                    f"+actor_rollout_ref.rollout.multi_turn.triage.budget.medium.max_search={args.medium_max_search}",
                    f"+actor_rollout_ref.rollout.multi_turn.triage.budget.medium.max_check={args.medium_max_check}",
                    f"+actor_rollout_ref.rollout.multi_turn.triage.budget.medium.max_turn={args.medium_max_turn}",
                    f"+actor_rollout_ref.rollout.multi_turn.triage.budget.hard.max_search={args.hard_max_search}",
                    f"+actor_rollout_ref.rollout.multi_turn.triage.budget.hard.max_check={args.hard_max_check}",
                    f"+actor_rollout_ref.rollout.multi_turn.triage.budget.hard.max_turn={args.hard_max_turn}",
                    (
                        "+actor_rollout_ref.rollout.multi_turn.triage.escalation."
                        f"contradiction_threshold={args.contradiction_threshold}"
                    ),
                    (
                        "+actor_rollout_ref.rollout.multi_turn.triage.escalation."
                        f"support_threshold={args.support_threshold}"
                    ),
                ]
            )

        with initialize_config_dir(config_dir=str(self.config_dir), version_base=None):
            config = compose(config_name="ppo_trainer", overrides=overrides)
        return config

    def _init_ray(self) -> None:
        if ray.is_initialized():
            return
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "VLLM_DISABLE_COMPILE_CACHE": "1",
                "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
                "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
            }
        }
        ray_kwargs = {"runtime_env": runtime_env, "ignore_reinit_error": True}
        if self.args.ray_object_store_memory is not None:
            ray_kwargs["object_store_memory"] = self.args.ray_object_store_memory
        ray.init(**ray_kwargs)

    @staticmethod
    def _run_async(coro):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            new_loop = asyncio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        return loop.run_until_complete(coro)

    def _create_agent_loop_manager_compat(self) -> AgentLoopManager:
        if hasattr(AgentLoopManager, "create"):
            return AgentLoopManager.create(config=self.config)

        manager = AgentLoopManager(config=self.config)
        for method_name in ("_initialize_llm_servers", "_init_global_load_balancer", "_init_agent_loop_workers"):
            method = getattr(manager, method_name, None)
            if method is None:
                raise AttributeError(f"AgentLoopManager is missing required initializer: {method_name}")
            self._run_async(method())
        return manager

    def _ensure_agent_loop_manager(self) -> AgentLoopManager:
        if self.agent_loop_manager is None:
            self._init_ray()
            self.agent_loop_manager = self._create_agent_loop_manager_compat()
        return self.agent_loop_manager

    def _build_batch(self, rows: list[pd.Series], start_index: int) -> DataProto:
        prompts = []
        data_sources = []
        reward_models = []
        extra_infos = []
        uids = []

        for offset, row in enumerate(rows):
            question = self._extract_question(row)
            prompt = self._build_inference_prompt(question)
            prompts.append(np.array(prompt, dtype=object))
            data_sources.append(str(self._safe_row_dict(row, "data_source", "search_r1_like_eval")))
            reward_models.append(self._safe_row_dict(row, "reward_model", {"style": "rule", "ground_truth": ""}))
            extra_infos.append(self._safe_row_dict(row, "extra_info", {}))
            uids.append(str(self._safe_row_dict(row, "uid", f"eval-{start_index + offset}")))

        return DataProto(
            non_tensor_batch={
                "raw_prompt": np.array(prompts, dtype=object),
                "agent_name": np.array(["tool_agent"] * len(prompts), dtype=object),
                "data_source": np.array(data_sources, dtype=object),
                "reward_model": np.array(reward_models, dtype=object),
                "extra_info": np.array(extra_infos, dtype=object),
                "uid": np.array(uids, dtype=object),
                "index": np.arange(start_index, start_index + len(prompts)),
            },
            meta_info={
                "validate": self.args.validate_mode,
                "global_steps": 0,
            },
        )

    @staticmethod
    def _pad_batch_to_worker_multiple(batch: DataProto, worker_count: int) -> tuple[DataProto, int]:
        if worker_count <= 0:
            return batch, len(batch)
        batch_size = len(batch)
        remainder = batch_size % worker_count
        if remainder == 0:
            return batch, batch_size

        pad_count = worker_count - remainder
        padded_non_tensor_batch = {}
        for key, value in batch.non_tensor_batch.items():
            if not isinstance(value, np.ndarray):
                value = np.array(value, dtype=object)
            if value.shape[0] == 0:
                padded_non_tensor_batch[key] = value
                continue
            tail = value[-1:]
            if pad_count == 1:
                extra = tail.copy()
            else:
                extra = np.repeat(tail, pad_count, axis=0)
            padded_non_tensor_batch[key] = np.concatenate([value, extra], axis=0)

        padded_batch = DataProto(
            batch=batch.batch,
            non_tensor_batch=padded_non_tensor_batch,
            meta_info=batch.meta_info,
        )
        return padded_batch, batch_size

    def _decode_outputs(self, result: DataProto) -> list[dict[str, Any]]:
        decoded = []
        responses = result.batch["responses"].cpu()
        response_masks = result.batch["response_mask"].cpu()
        attention_masks = result.batch["attention_mask"].cpu()
        response_len = responses.shape[1]
        num_turns = result.non_tensor_batch.get("__num_turns__", np.array([None] * len(responses)))
        search_used = result.non_tensor_batch.get("search_used", np.array([None] * len(responses)))
        check_used = result.non_tensor_batch.get("check_used", np.array([None] * len(responses)))
        difficulty_current = result.non_tensor_batch.get("difficulty_current", np.array([None] * len(responses), dtype=object))
        tool_budget = result.non_tensor_batch.get("tool_budget", np.array([None] * len(responses), dtype=object))

        for i in range(len(responses)):
            full_mask = attention_masks[i][-response_len:].bool()
            assistant_mask = response_masks[i].bool()
            full_response = self.tokenizer.decode(responses[i][full_mask].tolist(), skip_special_tokens=True)
            assistant_response = self.tokenizer.decode(
                responses[i][assistant_mask].tolist(), skip_special_tokens=True
            )
            decoded.append(
                {
                    "assistant_response": assistant_response,
                    "full_response": full_response,
                    "num_turns": int(num_turns[i]) if num_turns[i] is not None else None,
                    "search_used": int(search_used[i]) if search_used[i] is not None else None,
                    "check_used": int(check_used[i]) if check_used[i] is not None else None,
                    "difficulty_current": difficulty_current[i],
                    "tool_budget": tool_budget[i],
                }
            )
        return decoded

    def evaluate_dataset(self, test_data: pd.DataFrame, max_samples: int | None) -> list[dict[str, Any]]:
        manager = self._ensure_agent_loop_manager()
        results = []
        filtered_rows, filter_info = self._filter_overlong_rows(test_data, max_samples=max_samples)
        self.filter_info = filter_info
        if filter_info["filtered_samples"] > 0:
            print(
                "Filtered prompts longer than "
                f"{self.args.max_prompt_length} tokens: {filter_info['filtered_samples']} "
                f"of {filter_info['candidate_samples']}"
            )

        n_samples = len(filtered_rows)
        for start in tqdm(range(0, n_samples, self.args.eval_batch_size), desc="Evaluating"):
            end = min(start + self.args.eval_batch_size, n_samples)
            rows = filtered_rows[start:end]
            batch = self._build_batch(rows, start)
            worker_count = len(getattr(manager, "agent_loop_workers", []))
            padded_batch, original_batch_size = self._pad_batch_to_worker_multiple(batch, worker_count)
            result = manager.generate_sequences(prompts=padded_batch)
            decoded = self._decode_outputs(result)
            decoded = decoded[:original_batch_size]

            if start == 0 and decoded:
                first_question = self._extract_question(rows[0])
                first_prompt = self._build_inference_prompt(first_question)
                print("\n" + "=" * 60)
                print("Debug First Sample")
                print("=" * 60)
                print("validate_mode:", self.args.validate_mode)
                print("raw_prompt:")
                print(first_prompt)
                print("\nfull_response:")
                print(decoded[0]["full_response"])
                print("\nassistant_response:")
                print(decoded[0]["assistant_response"])
                print("=" * 60 + "\n")

            for local_idx, row in enumerate(rows):
                question = self._extract_question(row)
                golden_answers = self._extract_golden_answers(row)
                assistant_response = decoded[local_idx]["assistant_response"]
                full_response = decoded[local_idx]["full_response"]
                tags = self.extract_tags(assistant_response)
                model_answer = tags["answer"] or assistant_response.strip()
                actual_searches = (
                    decoded[local_idx]["search_used"]
                    if decoded[local_idx]["search_used"] is not None
                    else len(tags["searches"])
                )
                actual_checks = (
                    decoded[local_idx]["check_used"]
                    if decoded[local_idx]["check_used"] is not None
                    else len(tags["checks"])
                )
                explicit_checks = len(tags["checks"])
                auto_checks = max(0, actual_checks - explicit_checks)
                budget = decoded[local_idx]["tool_budget"] if isinstance(decoded[local_idx]["tool_budget"], dict) else {}
                max_search = int(budget.get("max_search", 0) or 0)
                max_check = int(budget.get("max_check", 0) or 0)
                max_turn = int(budget.get("max_turn", 0) or 0)
                num_turns = decoded[local_idx]["num_turns"] or 0
                best_f1 = max(self.compute_f1(model_answer, ga) for ga in golden_answers)
                best_em = max(self.compute_em(model_answer, ga) for ga in golden_answers)
                fuzzy_correct = any(ga.lower() in assistant_response.lower() for ga in golden_answers)
                checker_stats = self.extract_checker_stats(full_response)

                results.append(
                    {
                        "question": question,
                        "golden_answers": golden_answers,
                        "model_answer": model_answer,
                        "assistant_response": assistant_response,
                        "full_response": full_response,
                        "num_thinks": len(tags["thinks"]),
                        "num_searches": actual_searches,
                        "num_checks": actual_checks,
                        "num_explicit_checks": explicit_checks,
                        "num_tools": (
                            actual_searches
                            if self.tool_count_mode == "search"
                            else actual_checks
                            if self.tool_count_mode == "check"
                            else actual_searches + actual_checks
                        ),
                        "num_turns": num_turns,
                        "num_auto_checks": auto_checks,
                        "difficulty_current": decoded[local_idx]["difficulty_current"],
                        "tool_budget": budget,
                        "max_search_budget": max_search,
                        "max_check_budget": max_check,
                        "max_turn_budget": max_turn,
                        "search_budget_utilization": (actual_searches / max_search) if max_search > 0 else None,
                        "check_budget_utilization": (actual_checks / max_check) if max_check > 0 else None,
                        "turn_budget_utilization": (num_turns / max_turn) if max_turn > 0 else None,
                        "search_budget_saturated": bool(max_search > 0 and actual_searches >= max_search),
                        "check_budget_saturated": bool(max_check > 0 and actual_checks >= max_check),
                        "turn_budget_saturated": bool(max_turn > 0 and num_turns >= max_turn),
                        "search_queries": tags["searches"],
                        "check_queries": tags["checks"],
                        "has_answer_tag": tags["answer"] is not None,
                        "f1": best_f1,
                        "em": best_em,
                        "fuzzy_correct": fuzzy_correct,
                        **checker_stats,
                    }
                )
        return results

    def compute_metrics(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        if not results:
            return {"num_samples": 0}
        filter_info = getattr(self, "filter_info", {})
        search_util_values = [r["search_budget_utilization"] for r in results if r["search_budget_utilization"] is not None]
        check_util_values = [r["check_budget_utilization"] for r in results if r["check_budget_utilization"] is not None]
        turn_util_values = [r["turn_budget_utilization"] for r in results if r["turn_budget_utilization"] is not None]
        checker_support_rates = [r["checker_mean_support_rate"] for r in results if r["checker_mean_support_rate"] is not None]
        checker_contradiction_rates = [
            r["checker_mean_contradiction_rate"] for r in results if r["checker_mean_contradiction_rate"] is not None
        ]
        checker_avg_confidences = [
            r["checker_mean_avg_confidence"] for r in results if r["checker_mean_avg_confidence"] is not None
        ]
        return {
            "num_samples": len(results),
            "candidate_samples": filter_info.get("candidate_samples", len(results)),
            "filtered_samples": filter_info.get("filtered_samples", 0),
            "f1_mean": sum(r["f1"] for r in results) / len(results),
            "em_mean": sum(r["em"] for r in results) / len(results),
            "fuzzy_accuracy": sum(r["fuzzy_correct"] for r in results) / len(results),
            "avg_searches": sum(r["num_searches"] for r in results) / len(results),
            "avg_checks": sum(r["num_checks"] for r in results) / len(results),
            "avg_explicit_checks": sum(r["num_explicit_checks"] for r in results) / len(results),
            "avg_auto_checks": sum(r["num_auto_checks"] for r in results) / len(results),
            "avg_tools": sum(r["num_tools"] for r in results) / len(results),
            "avg_thinks": sum(r["num_thinks"] for r in results) / len(results),
            "avg_turns": sum((r["num_turns"] or 0) for r in results) / len(results),
            "avg_checker_calls_parsed": sum(r["checker_calls_parsed"] for r in results) / len(results),
            "avg_checker_total_claims": sum(r["checker_total_claims"] for r in results) / len(results),
            "avg_checker_supports": sum(r["checker_total_supports"] for r in results) / len(results),
            "avg_checker_contradictions": sum(r["checker_total_contradictions"] for r in results) / len(results),
            "avg_checker_neutrals": sum(r["checker_total_neutrals"] for r in results) / len(results),
            "avg_checker_support_rate": (
                sum(checker_support_rates) / len(checker_support_rates) if checker_support_rates else 0.0
            ),
            "avg_checker_contradiction_rate": (
                sum(checker_contradiction_rates) / len(checker_contradiction_rates)
                if checker_contradiction_rates
                else 0.0
            ),
            "avg_checker_avg_confidence": (
                sum(checker_avg_confidences) / len(checker_avg_confidences) if checker_avg_confidences else 0.0
            ),
            "avg_search_budget_utilization": (
                sum(search_util_values) / len(search_util_values) if search_util_values else 0.0
            ),
            "avg_check_budget_utilization": (
                sum(check_util_values) / len(check_util_values) if check_util_values else 0.0
            ),
            "avg_turn_budget_utilization": (
                sum(turn_util_values) / len(turn_util_values) if turn_util_values else 0.0
            ),
            "search_budget_saturation_rate": sum(r["search_budget_saturated"] for r in results) / len(results),
            "check_budget_saturation_rate": sum(r["check_budget_saturated"] for r in results) / len(results),
            "turn_budget_saturation_rate": sum(r["turn_budget_saturated"] for r in results) / len(results),
            "has_answer_tag_rate": sum(r["has_answer_tag"] for r in results) / len(results),
            "samples_with_search": sum(1 for r in results if r["num_searches"] > 0) / len(results),
            "samples_with_check": sum(1 for r in results if r["num_checks"] > 0) / len(results),
            "samples_with_explicit_check": sum(1 for r in results if r["num_explicit_checks"] > 0) / len(results),
            "samples_with_tool": sum(1 for r in results if r["num_tools"] > 0) / len(results),
            "checker_parsed_coverage": sum(1 for r in results if r["checker_calls_parsed"] > 0) / len(results),
            "samples_with_checker_support_signal": (
                sum(1 for r in results if r["checker_has_support_signal"]) / len(results)
            ),
            "samples_with_checker_contradiction_signal": (
                sum(1 for r in results if r["checker_has_contradiction_signal"]) / len(results)
            ),
            "samples_with_checker_only_neutral": sum(1 for r in results if r["checker_only_neutral"]) / len(results),
            "tool_count_mode": self.tool_count_mode,
            "tag_style": self.tag_style,
            "eval_mode": "agent_loop",
        }

    def close(self) -> None:
        if ray.is_initialized():
            ray.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_root", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--output_file", type=str, default="eval_results.json")
    parser.add_argument("--tool_count_mode", choices=["search", "check", "both"], default="both")
    parser.add_argument("--tag_style", choices=["auto", "xml", "prefix", "both"], default="auto")
    parser.add_argument("--prompt_mode", choices=["clean", "force_search", "explicit_check"], default="clean")
    parser.add_argument("--tool_config_path", type=str, default=str(DEFAULT_TOOL_CONFIG))
    parser.add_argument("--multi_turn_format", type=str, default="search_r1_with_checker")
    parser.add_argument("--rollout_backend", type=str, default="sglang")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--n_gpus_per_node", type=int, default=1)
    parser.add_argument("--agent_num_workers", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.4)
    parser.add_argument("--max_model_len", type=int, default=8000)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_response_length", type=int, default=2000)
    parser.add_argument("--max_assistant_turns", type=int, default=7)
    parser.add_argument("--max_parallel_calls", type=int, default=1)
    parser.add_argument("--max_tool_response_length", type=int, default=256)
    parser.add_argument("--tool_response_truncate_side", type=str, default="middle")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--val_do_sample", action="store_true")
    parser.add_argument("--val_temperature", type=float, default=0.0)
    parser.add_argument("--val_top_p", type=float, default=1.0)
    parser.add_argument("--val_top_k", type=int, default=-1)
    parser.add_argument("--enable_triage", action="store_true")
    parser.add_argument("--online_escalation", action="store_true")
    parser.add_argument("--easy_max_search", type=int, default=1)
    parser.add_argument("--easy_max_check", type=int, default=1)
    parser.add_argument("--easy_max_turn", type=int, default=3)
    parser.add_argument("--medium_max_search", type=int, default=2)
    parser.add_argument("--medium_max_check", type=int, default=2)
    parser.add_argument("--medium_max_turn", type=int, default=5)
    parser.add_argument("--hard_max_search", type=int, default=4)
    parser.add_argument("--hard_max_check", type=int, default=3)
    parser.add_argument("--hard_max_turn", type=int, default=7)
    parser.add_argument("--contradiction_threshold", type=float, default=0.30)
    parser.add_argument("--support_threshold", type=float, default=0.40)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--validate_mode", action="store_true")
    parser.add_argument("--ray_object_store_memory", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

    print(f"Loading test data from {args.test_file}")
    test_data = pd.read_parquet(args.test_file)
    print(f"Total samples: {len(test_data)}")
    print(f"Loading model from {args.model_path}")
    print("Evaluation mode: agent_loop")
    print(f"Tool config: {args.tool_config_path}")
    print(f"Multi-turn format: {args.multi_turn_format}")
    print(f"Triage enabled: {args.enable_triage}")

    evaluator = SearchR1Evaluator(args)
    try:
        results = evaluator.evaluate_dataset(test_data, max_samples=args.max_samples)
        metrics = evaluator.compute_metrics(results)
    finally:
        evaluator.close()

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "metrics": metrics,
                "config": OmegaConf.to_container(evaluator.config, resolve=True),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 60)
    print("Sample Results")
    print("=" * 60)
    for i, r in enumerate(results[:3]):
        print(f"\n--- Sample {i + 1} ---")
        print(f"Q: {r['question'][:120]}...")
        print(f"Golden: {r['golden_answers']}")
        print(f"Model: {r['model_answer'][:160]}...")
        print(
            f"Searches: {r['num_searches']}, Checks: {r['num_checks']}, "
            f"Tools({args.tool_count_mode}): {r['num_tools']}, Turns: {r['num_turns']}, "
            f"F1: {r['f1']:.3f}, EM: {r['em']}"
        )


if __name__ == "__main__":
    main()
