import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import LlamaConfig, LlamaForCausalLM

from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

WORLD_SIZE = 4  # a 2 x 2 HSDP mesh: two replicas of two shards


def _build(seed):
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    model = LlamaForCausalLM(config)
    mesh = init_device_mesh("cpu", (2, WORLD_SIZE // 2), mesh_dim_names=("ddp", "fsdp"))
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model, torch.optim.AdamW(model.parameters(), lr=1e-2)


def _worker(rank, world_size, rendezvous_file, ckpt_dir, results_dir):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous_file}", rank=rank, world_size=world_size)
    try:
        # one step: the optimizer holds state, and every replica the same parameters (HSDP all-reduces the gradients)
        model, optimizer = _build(seed=0)
        ids = torch.arange(8).view(1, 8)
        model(input_ids=ids, labels=ids).loss.backward()
        optimizer.step()
        FSDPCheckpointManager(model, optimizer).save_checkpoint(ckpt_dir, global_step=1)

        fresh, fresh_optimizer = _build(seed=1)  # other weights until loaded
        FSDPCheckpointManager(fresh, fresh_optimizer).load_checkpoint(ckpt_dir)
        pairs = list(zip(model.parameters(), fresh.parameters(), strict=True))
        same_params = all(torch.equal(a.to_local(), b.to_local()) for a, b in pairs)
        same_state = all(
            torch.equal(optimizer.state[a]["exp_avg"].to_local(), fresh_optimizer.state[b]["exp_avg"].to_local())
            for a, b in pairs
        )
        with open(os.path.join(results_dir, f"{rank}.txt"), "w") as handle:
            handle.write(f"{same_params} {same_state}")
    finally:
        dist.destroy_process_group()


def test_hsdp_replica_zero_writes_the_shards_every_replica_loads_them(tmp_path):
    ckpt_dir, results_dir = tmp_path / "ckpt", tmp_path / "results"
    results_dir.mkdir()
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, str(tmp_path / "rdzv"), str(ckpt_dir), str(results_dir)),
        nprocs=WORLD_SIZE,
        join=True,
    )

    files = sorted(os.listdir(ckpt_dir))
    # replica 0 (ranks 0 and 1) writes the model and optimizer shards once; every rank writes its extra state
    shard_files = [f"{kind}_world_size_4_rank_{rank}.pt" for kind in ("model", "optim") for rank in (0, 1)]
    assert [f for f in files if f.startswith(("model_", "optim_"))] == shard_files
    assert [f for f in files if f.startswith("extra_state_")] == [
        f"extra_state_world_size_4_rank_{rank}.pt" for rank in range(WORLD_SIZE)
    ]
    for rank in range(WORLD_SIZE):
        assert (results_dir / f"{rank}.txt").read_text() == "True True"
