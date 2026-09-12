# Copyright 2026 Individual Contributor: Zupeng Wang
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

"""Run with torchrun --nproc_per_node=2 test_torchtitan_critic.py --work-dir /tmp/critic.

Uses a randomly initialized Qwen3 debug model and a local tokenizer; no model
download or rollout server is needed. Add --with-score to load an HF critic.
"""

import argparse
import json
import os
from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from tensordict import TensorDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.distributed.tensor import DTensor
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, Qwen3ForTokenClassification, Qwen3Model

from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.workers.config import CriticConfig, HFModelConfig, TorchtitanEngineConfig, TorchtitanOptimizerConfig
from verl.workers.engine.torchtitan import TorchTitanEngineWithValueHead
from verl.workers.utils.losses import value_loss


def make_checkpoint(path, with_score):
    torch.manual_seed(42)
    config = Qwen3Config(
        vocab_size=2048,
        hidden_size=256,
        intermediate_size=3072,
        num_hidden_layers=8,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        tie_word_embeddings=True,
        max_position_embeddings=4096,
        rope_theta=1000000.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        num_labels=1,
        classifier_dropout=0.0,
    )
    cls = Qwen3ForTokenClassification if with_score else Qwen3ForCausalLM
    model = cls(config).to(torch.bfloat16)
    if with_score:
        with torch.no_grad():
            model.score.bias.fill_(0.125)
    model.save_pretrained(path)
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    vocab.update({f"token{i}": i for i in range(4, config.vocab_size)})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    tokenizer.save_pretrained(path)


def make_batch():
    rank = dist.get_rank()
    sequences = [torch.arange(11, 27, device="cuda") + rank, torch.arange(21, 33, device="cuda") + rank]
    prompt_lengths = [8, 6]

    def nested(values):
        return torch.nested.as_nested_tensor(values, layout=torch.jagged)

    data = TensorDict(
        {
            "input_ids": nested(sequences),
            "position_ids": nested([torch.arange(len(seq), device="cuda") for seq in sequences]),
            "prompts": nested([seq[:n] for seq, n in zip(sequences, prompt_lengths, strict=True)]),
            "responses": nested([seq[n:] for seq, n in zip(sequences, prompt_lengths, strict=True)]),
            "loss_mask": nested(
                [
                    torch.cat([torch.zeros(n, device="cuda"), torch.ones(len(seq) - n, device="cuda")])
                    for seq, n in zip(sequences, prompt_lengths, strict=True)
                ]
            ),
            "response_mask": nested(
                [torch.ones(len(seq) - n, device="cuda") for seq, n in zip(sequences, prompt_lengths, strict=True)]
            ),
            "values": nested(
                [torch.zeros(len(seq) - n, device="cuda") for seq, n in zip(sequences, prompt_lengths, strict=True)]
            ),
            "returns": nested(
                [
                    torch.linspace(-0.4, 0.7, len(seq) - n, device="cuda")
                    for seq, n in zip(sequences, prompt_lengths, strict=True)
                ]
            ),
        },
        batch_size=[2],
    )
    tu.assign_non_tensor(
        data,
        use_remove_padding=True,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=1,
        global_batch_size=2 * dist.get_world_size(),
    )
    return data


def clone_state(value):
    if isinstance(value, DTensor):
        value = value.full_tensor()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: clone_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clone_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(clone_state(v) for v in value)
    return value


def snapshot(engine):
    with engine.trainer.train_context():
        return clone_state(
            {
                "model": engine.module[0].state_dict(),
                "optimizer": engine.optimizer.state_dict(),
                "scheduler": engine.lr_scheduler.state_dict(),
            }
        )


def infer(engine, data, packed):
    tu.assign_non_tensor(data, use_remove_padding=packed)
    with engine.eval_mode(), engine.trainer.train_context(), torch.no_grad():
        _, output = engine.forward_step(data, loss_function=None, forward_only=True)
    return output["model_output"]["values"].values().float()


def train_step(engine, data):
    tu.assign_non_tensor(data, use_remove_padding=True)
    config = CriticConfig(strategy="torchtitan", ppo_micro_batch_size_per_gpu=1)
    with engine.train_mode():
        engine.optimizer_zero_grad()
        output = engine.forward_backward_batch(data, loss_function=partial(value_loss, config), forward_only=False)
        grad_norm = engine.optimizer_step()
        engine.lr_scheduler_step()
    assert torch.isfinite(torch.tensor(grad_norm)) and grad_norm > 0
    assert "critic/vf_loss" in output["metrics"]
    return grad_norm


def run(args):
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    engine = None
    try:
        work = Path(args.work_dir).resolve()
        assets = work / ("hf-critic" if args.with_score else "hf-lm")
        if dist.get_rank() == 0:
            work.mkdir(parents=True, exist_ok=True)
            make_checkpoint(assets, args.with_score)
        dist.barrier()
        os.chdir(work)
        engine = TorchTitanEngineWithValueHead(
            model_config=HFModelConfig(path=str(assets), use_remove_padding=True, model_type="value_model"),
            engine_config=TorchtitanEngineConfig(
                data_parallel_shard_size=dist.get_world_size(),
                attn_type="flex",
                activation_checkpoint="none",
                use_torch_compile=False,
                spmd_backend="spmd_types",
                max_seq_len=64,
                param_offload=False,
                optimizer_offload=False,
            ),
            optimizer_config=TorchtitanOptimizerConfig(lr=1e-4, total_training_steps=4),
            checkpoint_config=CheckpointConfig(),
        )
        engine.initialize()
        initial = snapshot(engine)
        state = initial["model"]
        assert state["lm_head.weight"].shape == (1, 256)
        assert state["tok_embeddings.weight"].shape == (2048, 256)
        source = load_file(assets / "model.safetensors")
        torch.testing.assert_close(
            state["tok_embeddings.weight"], source["model.embed_tokens.weight"].float(), rtol=0, atol=0
        )
        if args.with_score:
            torch.testing.assert_close(state["lm_head.weight"], source["score.weight"].float(), rtol=0, atol=0)
            torch.testing.assert_close(state["lm_head.bias"], source["score.bias"].float(), rtol=0, atol=0)
        else:
            assert torch.count_nonzero(state["lm_head.bias"]) == 0

        data = make_batch()
        packed = infer(engine, data, True)
        padded = infer(engine, data, False)
        torch.testing.assert_close(packed, padded, rtol=0.05, atol=0.03)
        reference = (
            Qwen3Model.from_pretrained(assets, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        )
        head = state["lm_head.weight"].to(device, dtype=torch.bfloat16)
        bias = state["lm_head.bias"].to(device, dtype=torch.bfloat16)
        with torch.no_grad():
            expected = torch.cat(
                [
                    torch.nn.functional.linear(reference(seq.unsqueeze(0)).last_hidden_state, head, bias).flatten()
                    for seq in data["input_ids"].unbind()
                ]
            ).float()
        max_error = (packed - expected).abs().max().item()
        torch.testing.assert_close(packed, expected, rtol=0.05, atol=0.03)
        del reference, source, initial

        grad_norm = train_step(engine, data)
        after_one = snapshot(engine)
        assert not torch.equal(after_one["model"]["lm_head.weight"], state["lm_head.weight"])
        assert not torch.equal(after_one["model"]["tok_embeddings.weight"], state["tok_embeddings.weight"])
        del state
        # Exercise the engine's public checkpoint API at TorchTitan's save interval.
        checkpoint_step = engine.checkpointer.interval
        checkpoint = work / "resume" / f"global_step_{checkpoint_step}"
        engine.save_checkpoint(str(checkpoint), global_step=checkpoint_step)
        assert (checkpoint.parent / f"step-{checkpoint_step}" / ".metadata").exists()
        train_step(engine, data)
        after_two = snapshot(engine)
        engine.load_checkpoint(str(checkpoint))
        restored = snapshot(engine)
        torch.testing.assert_close(restored, after_one, rtol=0, atol=0)
        train_step(engine, data)
        resumed_two = snapshot(engine)
        torch.testing.assert_close(resumed_two, after_two, rtol=0, atol=0)
        print(
            json.dumps(
                {
                    "result": "PASS",
                    "rank": dist.get_rank(),
                    "world_size": dist.get_world_size(),
                    "with_score": args.with_score,
                    "hf_max_abs_error": max_error,
                    "grad_norm": grad_norm,
                    "checkpoint_model_optimizer_scheduler": "exact",
                    "next_update_after_resume": "exact",
                }
            ),
            flush=True,
        )
    finally:
        if engine is not None:
            engine.trainer.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--with-score", action="store_true")
    run(parser.parse_args())
