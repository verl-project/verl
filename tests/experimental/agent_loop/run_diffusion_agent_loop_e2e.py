#!/usr/bin/env python
# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Manual end-to-end verifier for the diffusion agent loop.

This script reuses the same AgentLoopManager path as
`tests/experimental/agent_loop/test_diffusion_agent_loop.py`, but saves the
generated images and writes a JSON report with the rollout artifacts needed for
manual inspection.

Example:
    python tests/experimental/agent_loop/run_diffusion_agent_loop_e2e.py ^
        --model-path ~/models/Qwen/Qwen-Image ^
        --output-dir outputs/diffusion_agent_loop_e2e ^
        --gpus-per-node 4 ^
        --tensor-parallel-size 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import ray
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, open_dict
from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto

DEFAULT_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)
DEFAULT_PROMPTS = [
    "A cinematic portrait of a golden retriever wearing a red scarf in snowfall.",
    "A watercolor painting of a lighthouse on a cliff at sunset with crashing waves.",
]
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, bad anatomy, extra limbs"
DEFAULT_QWEN_PIPELINE = "verl.models.diffusion.vllm_omni.pipeline_qwenimage.QwenImagePipelineWithLogProb"
PROMPT_TEMPLATE_ENCODE_START_IDX = 34


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a manual diffusion agent-loop e2e check.")
    parser.add_argument("--model-path", required=True, help="Path to the full Qwen-Image model directory.")
    parser.add_argument("--tokenizer-path", default=None, help="Optional tokenizer path. Defaults to <model-path>/tokenizer.")
    parser.add_argument(
        "--output-dir",
        default="outputs/diffusion_agent_loop_e2e",
        help="Directory to store generated images and a JSON report.",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional JSON file containing either a list of strings or a list of {prompt, negative_prompt} objects.",
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt used for all requests.")
    parser.add_argument("--n", type=int, default=1, help="Number of rollouts per prompt.")
    parser.add_argument("--height", type=int, default=512, help="Requested image height.")
    parser.add_argument("--width", type=int, default=512, help="Requested image width.")
    parser.add_argument("--num-inference-steps", type=int, default=10, help="Number of diffusion denoising steps.")
    parser.add_argument("--true-cfg-scale", type=float, default=4.0, help="True CFG scale for negative-prompt classifier-free guidance.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed. When set, the script uses validation-mode rollout generation.",
    )
    parser.add_argument("--noise-level", type=float, default=1.0, help="Diffusion noise level used by the model/backend.")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="GPUs visible to the standalone rollout server.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallel size for rollout.")
    parser.add_argument("--agent-num-workers", type=int, default=2, help="Number of agent-loop workers.")
    parser.add_argument(
        "--custom-pipeline",
        default=DEFAULT_QWEN_PIPELINE,
        help="Custom vllm_omni diffusion pipeline class path.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=1024,
        help="Prompt token budget before the Qwen template prefix offset is added.",
    )
    return parser.parse_args()


def load_prompt_specs(prompt_file: str | None) -> list[dict[str, str]]:
    if prompt_file is None:
        return [{"prompt": prompt, "negative_prompt": DEFAULT_NEGATIVE_PROMPT} for prompt in DEFAULT_PROMPTS]

    with open(prompt_file, encoding="utf-8") as f:
        raw_specs = json.load(f)

    prompt_specs: list[dict[str, str]] = []
    for item in raw_specs:
        if isinstance(item, str):
            prompt_specs.append({"prompt": item, "negative_prompt": DEFAULT_NEGATIVE_PROMPT})
            continue
        if not isinstance(item, dict):
            raise TypeError("Prompt file entries must be strings or objects.")

        prompt = item.get("prompt") or item.get("user_prompt")
        if not prompt:
            raise ValueError("Prompt object must define `prompt` or `user_prompt`.")

        prompt_specs.append(
            {
                "prompt": prompt,
                "negative_prompt": item.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
            }
        )

    return prompt_specs


def build_config(args: argparse.Namespace) -> DictConfig:
    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config")):
        config = compose(config_name="ppo_diffusion_trainer")

    model_path = os.path.expanduser(args.model_path)
    tokenizer_path = os.path.expanduser(args.tokenizer_path) if args.tokenizer_path else os.path.join(model_path, "tokenizer")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_path}")

    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.model.tokenizer_path = tokenizer_path
    max_length = args.max_prompt_tokens + PROMPT_TEMPLATE_ENCODE_START_IDX
    with open_dict(config.actor_rollout_ref.model.extra_configs):
        config.actor_rollout_ref.model.extra_configs.true_cfg_scale = args.true_cfg_scale
        config.actor_rollout_ref.model.extra_configs.max_sequence_length = max_length
        config.actor_rollout_ref.model.extra_configs.noise_level = args.noise_level
    config.actor_rollout_ref.rollout.name = "vllm_omni"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.enforce_eager = True
    config.actor_rollout_ref.rollout.n = args.n
    config.actor_rollout_ref.rollout.height = args.height
    config.actor_rollout_ref.rollout.width = args.width
    config.actor_rollout_ref.rollout.num_inference_steps = args.num_inference_steps
    config.actor_rollout_ref.rollout.agent.num_workers = args.agent_num_workers
    config.actor_rollout_ref.rollout.agent.default_agent_loop = "diffusion_single_turn_agent"
    config.actor_rollout_ref.rollout.calculate_log_probs = True
    config.actor_rollout_ref.rollout.nnodes = 1
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = args.tensor_parallel_size
    config.actor_rollout_ref.rollout.engine_kwargs.vllm_omni = {"custom_pipeline": args.custom_pipeline}
    config.reward.reward_manager.name = "image"
    config.trainer.n_gpus_per_node = args.gpus_per_node
    config.actor_rollout_ref.rollout.n_gpus_per_node = args.gpus_per_node

    config.data.apply_chat_template_kwargs = dict(max_length=max_length, padding=True, truncation=True)
    config.data.max_prompt_length = max_length
    config.actor_rollout_ref.rollout.max_model_len = max_length

    if args.seed is not None:
        config.actor_rollout_ref.rollout.val_kwargs.seed = args.seed
        config.actor_rollout_ref.rollout.val_kwargs.num_inference_steps = args.num_inference_steps
        config.actor_rollout_ref.rollout.val_kwargs.noise_level = args.noise_level

    return config


def build_batch(
    prompt_specs: list[dict[str, str]],
    *,
    system_prompt: str,
    n: int,
    use_negative_prompt: bool,
    validate: bool,
) -> DataProto:
    raw_prompts = []
    raw_negative_prompts = []
    for spec in prompt_specs:
        raw_prompts.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": spec["prompt"]},
            ]
        )
        raw_negative_prompts.append(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": spec["negative_prompt"] if use_negative_prompt else " "},
            ]
        )

    batch = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts, dtype=object),
            "raw_negative_prompt": np.array(raw_negative_prompts, dtype=object),
            "data_source": np.array(["jpeg_compressibility"] * len(raw_prompts), dtype=object),
            "reward_model": np.array([{"style": "rule", "ground_truth": ""}] * len(raw_prompts), dtype=object),
        },
        meta_info={"validate": validate},
    )
    return batch.repeat(n)


def sanitize_filename(value: str, *, max_len: int = 60) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return sanitized[:max_len] or "sample"


def save_response_image(image_tensor: torch.Tensor, output_path: Path) -> dict[str, Any]:
    image_tensor = image_tensor.detach().cpu().float().clamp(0.0, 1.0)
    chw = image_tensor
    if chw.dim() != 3 or chw.shape[0] != 3:
        raise ValueError(f"Expected CHW image with 3 channels, got shape {tuple(chw.shape)}")

    hwc_uint8 = (chw.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(hwc_uint8).save(output_path)
    return {
        "shape": list(chw.shape),
        "min": float(chw.min().item()),
        "max": float(chw.max().item()),
    }


def summarize_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        summary: dict[str, Any] = {
            "type": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.numel() > 0:
            flat = value.detach().cpu().reshape(-1)
            preview = flat[: min(5, flat.numel())]
            summary["preview"] = preview.tolist()
        return summary

    if isinstance(value, np.ndarray):
        summary = {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.size > 0 and value.dtype != object:
            preview = value.reshape(-1)[: min(5, value.size)]
            summary["preview"] = preview.tolist()
        return summary

    if isinstance(value, list):
        return {"type": "list", "length": len(value), "preview": value[:5]}

    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(value.keys())}

    return value


def build_report(
    result: DataProto,
    *,
    prompt_specs: list[dict[str, str]],
    output_dir: Path,
    n: int,
) -> dict[str, Any]:
    responses = result.batch["responses"].detach().cpu()
    batch_keys = sorted(result.batch.keys())
    metrics = result.meta_info.get("metrics", [])
    num_turns = result.non_tensor_batch.get("__num_turns__")
    prompt_count = len(prompt_specs)

    report: dict[str, Any] = {
        "batch_keys": batch_keys,
        "num_samples": len(result),
        "prompt_count": prompt_count,
        "rollouts_per_prompt": n,
        "output_dir": str(output_dir),
        "samples": [],
        "manual_checks": [
            "Inspect saved images and confirm different prompts produce different outputs.",
            "Confirm prompt/output count equals prompt_count * rollouts_per_prompt.",
            "Confirm image ranges look sane and no saved image is empty or corrupted.",
            "Confirm rollout artifacts such as all_latents, all_timesteps, prompt_embeds, and rollout_log_probs are present.",
            "If true_cfg_scale > 0, confirm negative-prompt CFG path is active and outputs are stable enough for spot checks.",
        ],
    }

    for sample_idx in range(len(result)):
        prompt_idx = sample_idx // n
        rollout_idx = sample_idx % n
        prompt_spec = prompt_specs[prompt_idx]
        stem = sanitize_filename(prompt_spec["prompt"])
        image_path = output_dir / f"{sample_idx:03d}_prompt{prompt_idx:02d}_rollout{rollout_idx:02d}_{stem}.png"
        image_stats = save_response_image(responses[sample_idx], image_path)

        sample_report: dict[str, Any] = {
            "sample_idx": sample_idx,
            "prompt_idx": prompt_idx,
            "rollout_idx": rollout_idx,
            "prompt": prompt_spec["prompt"],
            "negative_prompt": prompt_spec["negative_prompt"],
            "image_path": str(image_path),
            "image_stats": image_stats,
            "num_turns": int(num_turns[sample_idx]) if num_turns is not None else None,
            "metrics": metrics[sample_idx] if sample_idx < len(metrics) else None,
            "artifacts": {},
        }

        for key in [
            "all_latents",
            "all_timesteps",
            "prompt_embeds",
            "prompt_embeds_mask",
            "rollout_log_probs",
            "input_ids",
            "attention_mask",
        ]:
            if key in result.batch:
                sample_report["artifacts"][key] = summarize_value(result.batch[key][sample_idx])
            elif key in result.non_tensor_batch:
                sample_report["artifacts"][key] = summarize_value(result.non_tensor_batch[key][sample_idx])
            else:
                sample_report["artifacts"][key] = {"present": False}

        report["samples"].append(sample_report)

    return report


def print_report_summary(report: dict[str, Any]) -> None:
    print(f"Saved {report['num_samples']} samples to {report['output_dir']}")
    print(f"Batch keys: {report['batch_keys']}")
    print("Manual checks:")
    for item in report["manual_checks"]:
        print(f"  - {item}")

    print("\nPer-sample summary:")
    for sample in report["samples"]:
        print(
            f"[sample {sample['sample_idx']}] prompt_idx={sample['prompt_idx']} "
            f"rollout_idx={sample['rollout_idx']} turns={sample['num_turns']}"
        )
        print(f"  prompt: {sample['prompt']}")
        print(f"  negative_prompt: {sample['negative_prompt']}")
        print(f"  image: {sample['image_path']} stats={sample['image_stats']}")
        for artifact_key, artifact_summary in sample["artifacts"].items():
            print(f"  {artifact_key}: {artifact_summary}")
        if sample["metrics"] is not None:
            print(f"  metrics: {sample['metrics']}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_specs = load_prompt_specs(args.prompt_file)
    config = build_config(args)
    batch = build_batch(
        prompt_specs,
        system_prompt=args.system_prompt,
        n=args.n,
        use_negative_prompt=args.true_cfg_scale > 0,
        validate=args.seed is not None,
    )

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
            }
        }
    )

    try:
        agent_loop_manager = AgentLoopManager.create(config)
        result = agent_loop_manager.generate_sequences(prompts=batch)
        report = build_report(result, prompt_specs=prompt_specs, output_dir=output_dir, n=args.n)
        report["config"] = {
            "model_path": os.path.expanduser(args.model_path),
            "tokenizer_path": os.path.expanduser(args.tokenizer_path)
            if args.tokenizer_path
            else os.path.join(os.path.expanduser(args.model_path), "tokenizer"),
            "n": args.n,
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.num_inference_steps,
            "true_cfg_scale": args.true_cfg_scale,
            "seed": args.seed,
            "noise_level": args.noise_level,
            "tensor_parallel_size": args.tensor_parallel_size,
            "agent_num_workers": args.agent_num_workers,
            "gpus_per_node": args.gpus_per_node,
            "custom_pipeline": args.custom_pipeline,
        }

        report_path = output_dir / "report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print_report_summary(report)
        print(f"\nFull JSON report: {report_path}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
