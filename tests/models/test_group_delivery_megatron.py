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
"""Real Megatron fixed-batch vs group-delivery updates; requires CUDA and the megatron extra.

Run with CUDA_VISIBLE_DEVICES=0,1 pytest -q -s tests/models/test_group_delivery_megatron.py.
Larger topology cases skip when there are not enough visible GPUs. No model download is needed.
"""

import math
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

TOPOLOGIES = [
    (2, 1, 1, 1),
    (1, 2, 1, 1),
    (1, 1, 2, 1),
    (1, 1, 1, 2),
    (4, 1, 1, 1),
    (2, 2, 1, 1),
    (1, 2, 2, 1),
    (1, 2, 2, 2),
]


def _make_model(path):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(19)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=256,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    Qwen3ForCausalLM(config).save_pretrained(path)


def _batch(rows, dp_rank, dp_size, mode, group_id=None):
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding

    generator = torch.Generator().manual_seed(23)
    tokens = torch.randint(3, 128, (6, 64), generator=generator)
    lengths = torch.tensor([4, 8, 15, 23, 8, 31])
    rows = list(rows)
    # Include real zero-loss DP lanes when rollout.n < DP.
    padded = rows + [-1] * (-len(rows) % dp_size)
    local = padded[dp_rank * (len(padded) // dp_size) : (dp_rank + 1) * (len(padded) // dp_size)]
    ids = torch.stack([tokens[max(row, 0)] for row in local])
    mask = torch.stack([torch.arange(32) < (lengths[row] if row >= 0 else 0) for row in local]).long()
    attention = torch.cat([torch.ones_like(mask), mask], dim=1)
    advantages = torch.tensor([(-1.0 if row % 2 == 0 else 1.0) if row >= 0 else 0.0 for row in local])
    data = TensorDict(
        dict(
            input_ids=ids,
            prompts=ids[:, :32],
            responses=ids[:, 32:],
            attention_mask=attention,
            position_ids=torch.arange(64).expand(len(local), -1),
            response_mask=mask,
            old_log_probs=torch.full_like(mask, -math.log(128), dtype=torch.float32),
            advantages=advantages[:, None].expand(-1, 32),
        ),
        batch_size=len(local),
    ).to("cuda")
    data = left_right_2_no_padding(data)
    weight = int(lengths[rows].sum()) if mode == "token-mean" else len(rows)
    tu.assign_non_tensor(
        data,
        global_batch_size=len(rows),
        temperature=1.0,
        calculate_entropy=False,
        group_delivery_id=group_id,
        group_delivery_weight=weight,
    )
    return data


def _run_rank(rank, world_size, topology, mode, streaming, root, rendezvous):
    from verl.utils import tensordict_utils as tu
    from verl.workers.config import ActorConfig, HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
    from verl.workers.engine import EngineRegistry
    from verl.workers.engine_workers_group_delivery import GroupDeliveryTrainingWorker
    from verl.workers.utils.losses import ppo_loss

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=f"file://{rendezvous}")
    _, tp, pp, cp = topology
    model = HFModelConfig(path=str(Path(root) / "model"), load_tokenizer=False, use_remove_padding=True)
    model.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=2)
    engine_config = McoreEngineConfig(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        use_mbridge=True,
        use_distributed_optimizer=True,
        param_offload=False,
        optimizer_offload=False,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=1,
        use_remove_padding=True,
        use_fused_kernels=False,
        sequence_parallel=tp > 1,
    )
    optimizer_config = McoreOptimizerConfig(lr=1e-3, lr_decay_steps=2, clip_grad=0.1)
    engine = EngineRegistry.new(
        model_type="language_model",
        backend="megatron",
        model_config=model,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=None,
    )
    engine.initialize()
    worker = object.__new__(GroupDeliveryTrainingWorker)
    worker.engine, worker.engine_config = engine, engine_config
    worker.optimizer_config, worker.model_config = optimizer_config, model
    worker.device_name, worker.flops_counter = "cuda", None
    actor_config = ActorConfig(strategy="megatron", ppo_micro_batch_size_per_gpu=1, loss_agg_mode=mode)
    worker.loss_fn = partial(ppo_loss, config=actor_config)
    snapshots = []
    actual_step = engine.optimizer_step

    def capture_step():
        # Inspect owned FP32 shards, not non-owned positions left by reduce-scatter.
        # This harness targets the repository's pinned Megatron-Core 0.19.2 interface.
        optimizers = getattr(engine.optimizer, "chained_optimizers", [engine.optimizer])
        parameters = []
        for optimizer in optimizers:
            optimizer._copy_model_grads_to_main_grads()
            parameters.extend(p for group in optimizer.optimizer.param_groups for p in group["params"])
        before = [p.detach().float().cpu().clone() for p in parameters]
        grads = [p.grad.detach().float().cpu().clone() for p in parameters]
        norm = actual_step()
        updates = [p.detach().float().cpu() - old for p, old in zip(parameters, before, strict=True)]
        snapshots.append(dict(gradients=grads, updates=updates, norm=float(norm)))
        return norm

    engine.optimizer_step = capture_step
    dp_rank, dp_size = engine.get_data_parallel_rank(), engine.get_data_parallel_size()
    # Two consecutive batches detect stale bucket state and accidental scheduler steps.
    for step in range(2):
        if streaming:
            worker.begin_group_batch(["a", "b", "c"])
            for group_index in (2, 0, 1):
                uid = "abc"[group_index]
                worker.backward_group(_batch([2 * group_index, 2 * group_index + 1], dp_rank, dp_size, mode, uid))
                assert len(snapshots) == step
            worker.finish_group_batch()
        else:
            worker.optimizer_zero_grad()
            data = _batch(range(6), dp_rank, dp_size, mode)
            tu.assign_non_tensor(data, return_model_output=False)
            worker.forward_backward(data)
            worker.optimizer_step()
            engine.lr_scheduler_step()
    torch.save(snapshots, Path(root) / f"{streaming}-{rank}.pt")
    dist.destroy_process_group()


def _relative_error(actual, expected):
    numerator = sum((a - b).double().square().sum() for a, b in zip(actual, expected, strict=True))
    denominator = sum(b.double().square().sum() for b in expected)
    assert denominator > 0, "This regression must exercise a nonzero gradient/update"
    return float((numerator / denominator).sqrt())


@pytest.mark.parametrize("topology", TOPOLOGIES, ids=[f"dp{d}-tp{t}-pp{p}-cp{c}" for d, t, p, c in TOPOLOGIES])
@pytest.mark.parametrize("mode", ["token-mean", "seq-mean-token-mean"])
def test_group_delivery_matches_full_batch_on_megatron(tmp_path, topology, mode):
    world_size = math.prod(topology)
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} CUDA GPUs")
    # With CUDA present, missing Megatron is an installation failure, not a passing skip.
    __import__("megatron.core")
    _make_model(tmp_path / "model")
    for streaming in (False, True):
        mp.spawn(
            _run_rank,
            args=(world_size, topology, mode, streaming, str(tmp_path), str(tmp_path / f"rendezvous-{streaming}")),
            nprocs=world_size,
            join=True,
        )
    for rank in range(world_size):
        baseline = torch.load(tmp_path / f"False-{rank}.pt", weights_only=True)
        delivered = torch.load(tmp_path / f"True-{rank}.pt", weights_only=True)
        assert len(baseline) == len(delivered) == 2
        for expected, actual in zip(baseline, delivered, strict=True):
            assert _relative_error(actual["gradients"], expected["gradients"]) < 0.02
            assert _relative_error(actual["updates"], expected["updates"]) < 0.03
            assert actual["norm"] == pytest.approx(expected["norm"], rel=0.02)
