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

"""Exercise fused output readiness with a real engine, DDP, and distributed optimizer.

Run with two GPUs and the repository's Megatron dependencies, without downloading
model weights or data::

    torchrun --standalone --nproc-per-node=2 \
        tests/special_distributed/test_megatron_fused_param_sync.py \
        --output /tmp/fused-ready --overlap 1 --expect pass

The output contains per-step gradients, optimizer state, and the final model for
comparison with an overlap-disabled run. ``--expect repro`` checks the original
missing-wait failure when this script is run against an unpatched checkout.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import Qwen3Config, Qwen3ForCausalLM

from verl.models.mcore import model_forward_fused as mff
from verl.trainer.config import CheckpointConfig
from verl.utils.megatron_utils import unwrap_model
from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead


def emit(event, **fields):
    """Emit a structured event with the current rank for distributed-run diagnostics."""
    print("FUSED_READY " + json.dumps({"event": event, "rank": dist.get_rank(), **fields}, sort_keys=True), flush=True)


def cpu_tree(value):
    """Copy nested tensor state to detached CPU snapshots that survive later updates."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(child) for key, child in value.items()}
    if isinstance(value, tuple | list):
        return [cpu_tree(child) for child in value]
    return value


def optimizer_state(engine):
    """Snapshot each non-stub distributed optimizer's master weights and Adam state."""
    return [
        {
            "master_float16": cpu_tree(opt.shard_fp32_from_float16_groups),
            "master_float32": cpu_tree(opt.shard_fp32_groups),
            "optimizer": cpu_tree(opt.optimizer.state_dict()),
        }
        for opt in getattr(engine.optimizer, "chained_optimizers", [engine.optimizer])
        if not getattr(opt, "is_stub_optimizer", False)
    ]


def prepare_model(path, tied):
    """Save a seeded tiny Qwen3 on rank zero and wait until all ranks can load it."""
    if dist.get_rank() == 0:
        assert not path.exists(), f"Refusing to overwrite {path}"
        torch.manual_seed(20261019)
        config = Qwen3Config(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            max_position_embeddings=128,
            tie_word_embeddings=tied,
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
        config.architectures = ["Qwen3ForCausalLM"]
        Qwen3ForCausalLM(config).save_pretrained(path)
    dist.barrier()


def make_engine(model_path, overlap):
    """Initialize a DP2 fused engine with the requested parameter-gather overlap setting."""
    model = HFModelConfig(
        path=str(model_path),
        load_tokenizer=False,
        use_remove_padding=True,
        use_fused_kernels=True,
        enable_gradient_checkpointing=False,
    )
    config = McoreEngineConfig(
        dtype="bfloat16",
        seed=20261019,
        param_offload=False,
        optimizer_offload=False,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=1,
        max_token_len_per_gpu=32,
        infer_micro_batch_size_per_gpu=1,
        infer_max_token_len_per_gpu=32,
        use_remove_padding=True,
        use_fused_kernels=True,
        use_distributed_optimizer=True,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=1,
        override_ddp_config={
            "overlap_grad_reduce": True,
            "overlap_param_gather": overlap,
            "align_param_gather": False,
            "bucket_size": 8192,
        },
        override_transformer_config={
            "gradient_accumulation_fusion": False,
            "hidden_dropout": 0.0,
            "attention_dropout": 0.0,
        },
    )
    optimizer = McoreOptimizerConfig(
        lr=1e-3,
        min_lr=1e-3,
        lr_decay_style="constant",
        total_training_steps=4,
        lr_decay_steps=4,
        lr_warmup_steps=0,
        weight_decay=0.0,
        clip_grad=1.0,
        override_optimizer_config={"overlap_param_gather": overlap},
    )
    engine = MegatronEngineWithLMHead(model, config, optimizer, CheckpointConfig())
    engine.initialize()
    return engine


def main():
    """Run four training steps and assert either clean fused consumption or the baseline failure."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlap", type=int, choices=[0, 1], required=True)
    parser.add_argument("--tied", type=int, choices=[0, 1], default=0)
    parser.add_argument("--expect", choices=["pass", "repro"], required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 2
    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "tiny-qwen3"
    prepare_model(model_path, bool(args.tied))
    engine = make_engine(model_path, bool(args.overlap))
    assert len(engine.module) == 1
    ddp = engine.module[0]
    raw = unwrap_model(engine.module)[0]
    weight = raw.shared_embedding_or_output_weight() if args.tied else raw.output_layer.weight
    bucket = ddp.param_to_bucket_group[weight]
    names = {id(param): name for name, param in raw.named_parameters()}
    bucket_names = [names[id(param)] for param, group in ddp.param_to_bucket_group.items() if group is bucket]
    assert len(bucket_names) == 1, bucket_names
    assert raw.share_embeddings_and_output_weights == bool(args.tied)
    assert ddp.ddp_config.overlap_param_gather == bool(args.overlap)
    import megatron.core.utils as mcore_utils

    emit(
        "environment",
        versions={
            name: importlib.metadata.version(name)
            for name in ["torch", "megatron-core", "megatron-bridge", "transformer-engine", "transformers"]
        },
        fused_source=inspect.getfile(mff),
        fused_source_sha256=hashlib.sha256(Path(inspect.getfile(mff)).read_bytes()).hexdigest(),
        engine_source=inspect.getfile(type(engine)),
        native_ready_available=hasattr(mcore_utils, "ensure_params_ready"),
        bucket_names=bucket_names,
        bucket_count=len({id(group) for group in ddp.param_to_bucket_group.values()}),
        overlap=bool(args.overlap),
        tied=bool(args.tied),
        ddp_config=str(ddp.ddp_config),
    )
    original_kernel = mff.linear_cross_entropy
    observations = []

    def observed_kernel(hidden, actual_weight, *positional, **kwargs):
        """Record the owning bucket's gather state immediately before the real fused kernel."""
        assert actual_weight is weight
        row = {
            "call": len(observations) + 1,
            "dispatched": bool(bucket.param_gather_dispatched),
            "handle_live": bucket.param_gather_handle is not None,
        }
        observations.append(row)
        emit("before_kernel", **row)
        return original_kernel(hidden, actual_weight, *positional, **kwargs)

    mff.linear_cross_entropy = observed_kernel
    steps = []
    phase = "before_training"
    failure = None
    try:
        for step in range(1, 5):
            phase = f"step_{step}_zero_grad"
            engine.optimizer_zero_grad()
            generator = torch.Generator(device="cuda").manual_seed(1000 + step + dist.get_rank() * 100)
            tokens = torch.randint(3, 256, (1, 32), device="cuda", generator=generator)
            labels = torch.roll(tokens, shifts=-1, dims=1)
            positions = torch.arange(32, device="cuda").unsqueeze(0)
            phase = f"step_{step}_forward"
            output = ddp(
                input_ids=tokens,
                position_ids=positions,
                attention_mask=None,
                labels=labels,
                output_processor=mff.fused_output_processor,
                output_processor_context=mff.FusedOutputProcessorContext(temperature=1.0),
            )
            loss = -output.log_probs.float().mean()
            phase = f"step_{step}_backward"
            loss.backward()
            ddp.finish_grad_sync()
            gradients = {
                name: cpu_tree(param.main_grad) for name, param in raw.named_parameters() if param.requires_grad
            }
            phase = f"step_{step}_optimizer"
            grad_norm = engine.optimizer_step()
            lr = engine.lr_scheduler_step()
            row = {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad_norm), "lr": lr}
            steps.append(row)
            torch.save(
                {"metrics": row, "gradients": gradients, "state": optimizer_state(engine)},
                args.output / f"rank{dist.get_rank()}-step{step}.pt",
            )
            emit("step_complete", **row, handle_live=bucket.param_gather_handle is not None)
    except AssertionError:
        failure = {"phase": phase, "traceback": traceback.format_exc()}
        emit("training_assertion", **failure)
    finally:
        mff.linear_cross_entropy = original_kernel
    if args.expect == "repro":
        assert failure is not None, "Baseline completed without the reported assertion"
        assert "param_gather_handle" in failure["traceback"] and (
            "start_param_sync" in failure["traceback"] or "reset_param_sync_dispatch_state" in failure["traceback"]
        ), failure
        assert any(row["handle_live"] for row in observations), observations
    else:
        assert failure is None, failure
        assert len(steps) == 4
        assert not any(row["handle_live"] for row in observations), observations
        # This happens only after all training steps; never repairs a baseline forward.
        ddp.start_param_sync(force_sync=True)
        torch.save(
            {name: cpu_tree(param) for name, param in raw.named_parameters()},
            args.output / f"rank{dist.get_rank()}-final-model.pt",
        )
    report = {"expect": args.expect, "steps": steps, "observations": observations, "failure": failure}
    (args.output / f"rank{dist.get_rank()}-report.json").write_text(json.dumps(report, indent=2))
    emit("CASE_COMPLETE", expect=args.expect, completed_steps=len(steps), target_reproduced=failure is not None)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
