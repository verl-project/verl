# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import asyncio
import socket
import sys
import threading
from collections import defaultdict
from typing import Optional

import ray
import torch

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopMetrics,
    _InternalAgentLoopOutput,
)
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopWorker as _AgentLoopWorker,
)
from verl.protocol import DataProto
from verl.utils.model import compute_position_id_with_mask
from verl.utils.ray_utils import auto_await

# _postprocess is an AgentLoopWorker method that stacks _InternalAgentLoopOutput
# into a DataProto batch. It accesses self.distillation_enabled and
# self.reward_loop_worker_handles, both of which we set in _init_nemo_gym.
_postprocess = _AgentLoopWorker._postprocess


class NemoGymAgentLoopManager(AgentLoopManager):
    @classmethod
    @auto_await
    async def create(
        cls,
        config,
        worker_group=None,
        rollout_resource_pool=None,
        reward_loop_worker_handles=None,
        teacher_model_manager=None,
    ) -> NemoGymAgentLoopManager:
        instance = cls(
            config,
            worker_group,
            rollout_resource_pool,
            teacher_model_manager,
            reward_loop_worker_handles,
        )
        await instance._initialize_llm_servers()
        await instance._init_global_load_balancer()
        await instance._apply_server_patch()
        await instance._init_nemo_gym()
        return instance

    async def _apply_server_patch(self) -> None:
        futs = [handle.apply_nemo_gym_server_patch.remote() for handle in self.server_handles]
        await asyncio.get_event_loop().run_in_executor(None, ray.get, futs)

    async def _init_nemo_gym(self) -> None:
        nemo_gym_cfg = self.rollout_config.agent.nemo_gym

        nemo_gym_root = nemo_gym_cfg.nemo_gym_root if nemo_gym_cfg else None

        if nemo_gym_root and str(nemo_gym_root) not in sys.path:
            sys.path.insert(0, str(nemo_gym_root))

        from omegaconf import DictConfig

        try:
            from nemo_gym.cli import GlobalConfigDictParserConfig, RunHelper
            from nemo_gym.rollout_collection import RolloutCollectionHelper
            from nemo_gym.server_utils import HEAD_SERVER_KEY_NAME, BaseServerConfig
        except ModuleNotFoundError as e:
            raise ImportError("nemo-gym not found. Install it with: pip install -e /path/to/gym-ref") from e

        config_paths = list(nemo_gym_cfg.config_paths or []) if nemo_gym_cfg else []
        initial_global_cfg = {"config_paths": config_paths} if config_paths else {}

        uses_reasoning_parser = nemo_gym_cfg.uses_reasoning_parser if nemo_gym_cfg else False
        vllm_model_cfg = (
            initial_global_cfg.setdefault("policy_model", {})
            .setdefault("responses_api_models", {})
            .setdefault("vllm_model", {})
        )
        vllm_model_cfg["uses_reasoning_parser"] = uses_reasoning_parser

        if not uses_reasoning_parser:
            vllm_model_cfg.setdefault("extra_body", {}).setdefault("chat_template_kwargs", {})["enable_thinking"] = (
                False
            )

        base_urls = [
            (addr if addr.startswith("http") else f"http://{addr}").rstrip("/") + "/v1"
            for addr in self.server_addresses
        ]
        initial_global_cfg["policy_model_name"] = self.model_config.get("path", "")
        initial_global_cfg["policy_api_key"] = "dummy_key"
        initial_global_cfg["policy_base_url"] = base_urls
        initial_global_cfg.setdefault("global_aiohttp_connector_limit_per_host", 16_384)
        initial_global_cfg.setdefault("global_aiohttp_connector_limit", 65_536)

        ray_context = ray.get_runtime_context()
        initial_global_cfg["ray_head_node_address"] = ray_context.gcs_address

        if nemo_gym_root:
            initial_global_cfg.setdefault("uv_venv_dir", str(nemo_gym_root))
            initial_global_cfg.setdefault("skip_venv_if_present", True)

        node_ip = socket.gethostbyname(socket.gethostname())
        with socket.socket() as s:
            s.bind(("", 0))
            head_port = s.getsockname()[1]
        initial_global_cfg[HEAD_SERVER_KEY_NAME] = {"host": "0.0.0.0", "port": head_port}
        self._head_server_config = BaseServerConfig(host=node_ip, port=head_port)

        self._rh = RunHelper()
        self._rh.start(
            global_config_dict_parser_config=GlobalConfigDictParserConfig(
                dotenv_path=None,
                initial_global_config_dict=DictConfig(initial_global_cfg),
                skip_load_from_cli=True,
            )
        )

        self._rch = RolloutCollectionHelper()

        from verl.utils.config import omega_conf_to_dataclass

        self._tokenizer = omega_conf_to_dataclass(self.model_config).tokenizer
        self.distillation_enabled = False

        self._rollout_loop = asyncio.new_event_loop()
        self._rollout_thread = threading.Thread(
            target=self._rollout_loop.run_forever,
            daemon=True,
            name="nemo-gym-rollout-loop",
        )
        self._rollout_thread.start()

        print(f"NemoGymAgentLoopManager ready: {len(base_urls)} vLLM endpoints: {base_urls}")

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        future = asyncio.run_coroutine_threadsafe(self._async_generate_sequences(prompts), self._rollout_loop)
        return future.result()

    async def _async_generate_sequences(self, prompts: DataProto) -> DataProto:
        validate = prompts.meta_info.get("validate", False)
        global_steps = prompts.meta_info.get("global_steps", -1)

        nemo_gym_examples = _build_nemo_gym_examples(
            prompts,
            self.rollout_config,
            validate=validate,
        )

        nemo_gym_result_iterator = self._rch.run_examples(
            examples=nemo_gym_examples,
            head_server_config=self._head_server_config,
        )

        rowidxs, raw_results = [], []
        for task in nemo_gym_result_iterator:
            nemo_gym_row, nemo_gym_result = await task
            try:
                result = _postprocess_nemo_gym_result(nemo_gym_result, self._tokenizer)
            except ValueError:
                result = _empty_result(nemo_gym_row, self._tokenizer)
            result["env"] = nemo_gym_row["agent_ref"]["name"]
            rowidxs.append(nemo_gym_row["_rowidx"])
            raw_results.append(result)

        results = [None] * len(nemo_gym_examples)
        for rowidx, result in zip(rowidxs, raw_results, strict=True):
            results[rowidx] = result
        missing = [i for i, r in enumerate(results) if r is None]
        if missing:
            raise RuntimeError(f"nemo-gym did not return results for samples: {missing}")

        prompt_lens = [sum(len(m["token_ids"]) for m in r["input_message_log"]) for r in results]
        response_lens = [
            sum(len(m["token_ids"]) for m in r["message_log"][len(r["input_message_log"]) :]) for r in results
        ]
        prompt_length = max(prompt_lens) if prompt_lens else self.rollout_config.prompt_length
        _vllm_kwargs = (getattr(self.rollout_config, "engine_kwargs", {}) or {}).get("vllm", {}) or {}
        max_model_len = (
            getattr(self.rollout_config, "max_model_len", None)
            or _vllm_kwargs.get("max-model-len")
            or _vllm_kwargs.get("max_model_len")
        )
        response_budget = (int(max_model_len) - prompt_length) if max_model_len else None
        response_length = max(response_lens) if response_lens else self.rollout_config.response_length
        if response_budget:
            response_length = min(response_length, response_budget)

        internal_outputs = [
            _nemo_gym_result_to_verl(
                result=result,
                tokenizer=self._tokenizer,
                prompt_length=prompt_length,
                response_length=response_length,
            )
            for result in results
        ]

        output = _postprocess(
            self,
            internal_outputs,
            input_non_tensor_batch=prompts.non_tensor_batch,
            validate=validate,
        )
        output.meta_info["global_steps"] = global_steps
        rollout_metrics = _compute_rollout_metrics(results, getattr(self.rollout_config, "max_model_len", None))
        output.meta_info["timing"] = {}
        output.meta_info["rollout_metrics"] = rollout_metrics

        env_metrics = {k: v for k, v in rollout_metrics.items() if k.startswith("env/")}
        if env_metrics:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log({**env_metrics, "train/global_step": global_steps}, step=global_steps)
            except Exception:
                pass

        return output


def _build_nemo_gym_examples(
    prompts: DataProto,
    rollout_config,
    validate: bool = False,
) -> list[dict]:
    cfg = rollout_config
    temperature = cfg.val_kwargs.temperature if validate else cfg.temperature
    top_p = cfg.val_kwargs.top_p if validate else cfg.top_p

    non_tensor = prompts.non_tensor_batch
    examples = []
    for i in range(len(prompts)):
        messages = list(non_tensor["raw_prompt"][i])

        if "agent_ref" not in non_tensor:
            raise ValueError(f"dataset row {i} is missing agent_ref")
        agent_ref = non_tensor["agent_ref"][i]

        rcp = {}
        if "extra_env_info" in non_tensor and "_rcp_extra" in (non_tensor["extra_env_info"][i] or {}):
            rcp.update(non_tensor["extra_env_info"][i]["_rcp_extra"])
        rcp.update({"input": messages, "temperature": temperature, "top_p": top_p})

        row = {
            "responses_create_params": rcp,
            "agent_ref": agent_ref,
            "_rowidx": i,
        }

        if "extra_env_info" in non_tensor:
            env_info = non_tensor["extra_env_info"][i]
            if isinstance(env_info, dict):
                for k, v in env_info.items():
                    if k not in row:
                        row[k] = v

        examples.append(row)
    return examples


def _postprocess_nemo_gym_result(nemo_gym_result: dict, tokenizer) -> dict:
    message_log = []
    seen_token_ids: list[int] = []

    for item in nemo_gym_result["response"]["output"]:
        if "generation_token_ids" not in item:
            continue

        prompt_ids = item["prompt_token_ids"]

        if seen_token_ids != prompt_ids[: len(seen_token_ids)]:
            raise ValueError(
                f"Non-contiguous token IDs (server_patch active?). seen={len(seen_token_ids)} prompt={len(prompt_ids)}"
            )

        message_log.append(
            {
                "role": "user",
                "content": "",
                "token_ids": torch.tensor(prompt_ids[len(seen_token_ids) :]),
            }
        )
        message_log.append(
            {
                "role": "assistant",
                "content": "",
                "token_ids": torch.tensor(item["generation_token_ids"]),
                "generation_logprobs": torch.tensor(item["generation_log_probs"]),
            }
        )

        seen_token_ids.extend(message_log[-2]["token_ids"].tolist())
        seen_token_ids.extend(message_log[-1]["token_ids"].tolist())

        item.pop("prompt_token_ids", None)
        item["prompt_str"] = tokenizer.decode(prompt_ids)
        item["generation_str"] = tokenizer.decode(item.pop("generation_token_ids"))
        item.pop("generation_log_probs")

    if not message_log:
        raise ValueError(
            "nemo-gym returned a result with no generation data. The prompt may exceed vLLM's max_model_len."
        )

    return {
        "message_log": message_log,
        "input_message_log": message_log[:1],
        "full_result": nemo_gym_result,
    }


def _empty_result(nemo_gym_row: dict, tokenizer) -> dict:
    """empty result for overlong samples
    TODO: should we truncate or something else? what is best practice here?"""

    messages = nemo_gym_row.get("responses_create_params", {}).get("input", [])
    raw_prompt = [{"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages]
    prompt_ids = tokenizer.apply_chat_template(raw_prompt, tokenize=True, add_generation_prompt=False)[-1:]
    dummy_tok = torch.tensor(prompt_ids, dtype=torch.long)
    return {
        "message_log": [
            {"role": "user", "token_ids": dummy_tok},
            {"role": "assistant", "token_ids": dummy_tok, "generation_logprobs": torch.zeros(len(dummy_tok))},
        ],
        "input_message_log": [{"role": "user", "token_ids": dummy_tok}],
        "full_result": {"reward": 0.0},
    }


def _nemo_gym_result_to_verl(
    result: dict,
    tokenizer,
    prompt_length: int,
    response_length: int,
) -> _InternalAgentLoopOutput:
    """Pack message_log into padded tensors for verl and mask non assistant messages"""
    message_log = result["message_log"]
    input_message_log = result["input_message_log"]

    prompt_ids_raw: list[int] = []
    for msg in input_message_log:
        tids = msg["token_ids"]
        prompt_ids_raw.extend(tids.tolist() if isinstance(tids, torch.Tensor) else tids)

    n_prompt_msgs = len(input_message_log)
    response_ids_raw: list[int] = []
    response_mask_raw: list[int] = []
    response_logprobs_raw: list[float] = []

    for msg in message_log[n_prompt_msgs:]:
        tids = msg["token_ids"]
        toks = tids.tolist() if isinstance(tids, torch.Tensor) else list(tids)
        if msg["role"] == "assistant":
            response_ids_raw.extend(toks)
            response_mask_raw.extend([1] * len(toks))
            lp = msg.get("generation_logprobs")
            response_logprobs_raw.extend(
                lp.tolist() if isinstance(lp, torch.Tensor) else list(lp) if lp is not None else [0.0] * len(toks)
            )
        else:
            response_ids_raw.extend(toks)
            response_mask_raw.extend([0] * len(toks))
            response_logprobs_raw.extend([0.0] * len(toks))

    response_ids_raw = response_ids_raw[:response_length]
    response_mask_raw = response_mask_raw[:response_length]
    response_logprobs_raw = response_logprobs_raw[:response_length]

    tokenizer.padding_side = "left"
    prompt_out = tokenizer.pad(
        {"input_ids": prompt_ids_raw},
        padding="max_length",
        max_length=prompt_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    prompt_ids = prompt_out["input_ids"]
    if prompt_ids.dim() == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
        prompt_out["attention_mask"] = prompt_out["attention_mask"].unsqueeze(0)

    tokenizer.padding_side = "right"
    response_out = tokenizer.pad(
        {"input_ids": response_ids_raw},
        padding="max_length",
        max_length=response_length,
        return_tensors="pt",
        return_attention_mask=True,
    )
    response_ids = response_out["input_ids"]
    response_attn = response_out["attention_mask"]
    if response_ids.dim() == 1:
        response_ids = response_ids.unsqueeze(0)
        response_attn = response_attn.unsqueeze(0)

    pad = response_length - len(response_mask_raw)
    response_mask = torch.tensor(response_mask_raw + [0] * pad, dtype=torch.long).unsqueeze(0) * response_attn

    pad = response_length - len(response_logprobs_raw)
    response_logprobs = torch.tensor(response_logprobs_raw + [0.0] * pad, dtype=torch.float32).unsqueeze(0)

    attention_mask = torch.cat([prompt_out["attention_mask"], response_attn], dim=1)
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)

    position_ids = compute_position_id_with_mask(attention_mask)

    reward_score: Optional[float] = result.get("full_result", {}).get("reward", None)
    num_turns = sum(1 for m in message_log if m["role"] == "assistant")

    return _InternalAgentLoopOutput(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        input_ids=input_ids,
        position_ids=position_ids,
        response_mask=response_mask,
        attention_mask=attention_mask,
        response_logprobs=response_logprobs,
        routed_experts=None,
        multi_modal_inputs={},
        teacher_logprobs=None,
        teacher_ids=None,
        reward_score=reward_score,
        num_turns=num_turns,
        metrics=AgentLoopMetrics(),
        extra_fields={},
    )


def _compute_rollout_metrics(results: list[dict], max_model_len: Optional[int] = None) -> dict:
    batch_size = len(results)
    if batch_size == 0:
        return {}

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    stats = []
    for r in results:
        ml = r["message_log"]
        total = sum(len(m["token_ids"]) for m in ml)
        asst = sum(len(m["token_ids"]) for m in ml if m["role"] == "assistant")
        turns = sum(1 for m in ml if m["role"] == "user")
        hit_max = (max_model_len is not None) and (total >= max_model_len)
        stats.append(
            {
                "reward": float(r.get("full_result", {}).get("reward", 0.0)),
                "asst": asst,
                "total": total,
                "turns": turns,
                "hit_max": hit_max,
            }
        )

    metrics = {
        "turns_per_sample/mean": _mean([s["turns"] for s in stats]),
        "total_tokens_per_sample/mean": _mean([s["total"] for s in stats]),
        "gen_tokens_per_sample/mean": _mean([s["asst"] for s in stats]),
        "mean_gen_tokens_per_sample": _mean([s["asst"] for s in stats]),
        "total_reward/mean": _mean([s["reward"] for s in stats]),
        "natural_termination_rate": sum(not s["hit_max"] for s in stats) / batch_size,
        "truncation_rate": sum(s["hit_max"] for s in stats) / batch_size,
    }

    # per environment metrics
    env_rewards: dict = defaultdict(list)
    for r, s in zip(results, stats, strict=True):
        env_rewards[r.get("env", "unknown")].append(s["reward"])
    for env, rewards in env_rewards.items():
        metrics[f"env/{env}/reward_mean"] = _mean(rewards)

    return metrics
