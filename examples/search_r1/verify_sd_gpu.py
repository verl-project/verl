"""Eight-rank FSDP check of the real SD backward, before loading the retriever."""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tensordict import TensorDict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import Qwen2Config, Qwen2ForCausalLM

from examples.search_r1.sd_search import MASK, backward_sd
from verl.utils import tensordict_utils as tu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        torch.manual_seed(42)
        config = Qwen2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            attention_dropout=0.0,
            attn_implementation="eager",
        )
        model = Qwen2ForCausalLM(config).cuda()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model = FSDP(model, device_id=rank, use_orig_params=False)
        model.train()
        # Mirror the integration order: ordinary actor backward, then SD before step.
        model(
            input_ids=torch.tensor([[1, 2, 3]], device=f"cuda:{rank}"), use_cache=False
        ).logits.float().square().mean().backward()
        fields = {}
        for slot in range(4):
            rows = {
                "student_ids": [[1, 2, 3], [0, 0]],
                "teacher_ids": [[4, 1, 2, 3], [0, 0]],
                "student_positions": [[2], [1]],
                "teacher_positions": [[3], [1]],
                "weights": [[0.25], [0.0]],
            }
            for key, values in rows.items():
                # Deliberately leave ALL auxiliary columns on CPU as in TransferQueue.
                dtype = torch.float32 if key == "weights" else torch.long
                fields[f"sd_{slot}_{key}"] = torch.nested.as_nested_tensor(
                    [torch.tensor(v, dtype=dtype) for v in values], layout=torch.jagged
                )
        data = TensorDict(fields, batch_size=[2])
        tu.assign_non_tensor(
            data,
            sd_alpha=0.001,
            sd_slots=4,
            sd_top_k=5,
            sd_micro_batch=1,
            auxiliary_loss_mask_counts={MASK: 2 * dist.get_world_size()},
            dp_size=dist.get_world_size(),
        )
        metrics = backward_sd(SimpleNamespace(module=model, scaler=None), data)
        value = float(metrics["sd/jsd"].aggregate())
        assert value > 0 and torch.isfinite(torch.tensor(value)), value
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        record = {"rank": rank, "jsd": value, "finite_gradients": True, "cpu_auxiliary_fields": True}
        records = [None] * dist.get_world_size()
        dist.all_gather_object(records, record)
        if rank == 0:
            args.output.write_text(
                json.dumps({"passed": True, "world_size": dist.get_world_size(), "ranks": records}, indent=2)
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
