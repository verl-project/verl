from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from verl.models.transformers.qwen3_5 import patch_vision_model_for_ulysses_sharding
from verl.utils.ulysses import set_ulysses_sequence_parallel_group

WORLD = 2
MERGE = 4  # the stub tower's spatial_merge_unit: tokens pooled into one feature
GRID_THW = [[1, 2, 4], [1, 4, 4], [1, 2, 2]]  # three images of 8, 16 and 4 tokens: rank 0 encodes one, rank 1 two
FEATURES = 3


class TowerStub:
    """A vision tower whose features depend on each token alone, pooled over MERGE tokens."""

    spatial_merge_unit = MERGE

    def forward(self, hidden_states, grid_thw):
        tokens = hidden_states * 2 + 1
        return SimpleNamespace(last_hidden_state=tokens, pooler_output=tokens.view(-1, MERGE, FEATURES).sum(1))


def sharded_forward_on_rank(rank, store):
    dist.init_process_group("gloo", init_method=f"file://{store}", rank=rank, world_size=WORLD)
    set_ulysses_sequence_parallel_group(dist.group.WORLD)
    grid_thw = torch.tensor(GRID_THW)
    hidden = torch.arange(int(grid_thw.prod(dim=1).sum()) * FEATURES, dtype=torch.float32).view(-1, FEATURES)
    expected = TowerStub().forward(hidden, grid_thw)

    class ShardedTower(TowerStub):
        pass

    patch_vision_model_for_ulysses_sharding(ShardedTower)
    output = ShardedTower().forward(hidden, grid_thw)
    assert torch.equal(output.last_hidden_state, expected.last_hidden_state)
    assert torch.equal(output.pooler_output, expected.pooler_output)
    dist.destroy_process_group()


def test_sharded_vision_tower_gathers_the_unsharded_features(tmp_path):
    mp.spawn(sharded_forward_on_rank, args=(str(tmp_path / "store"),), nprocs=WORLD, join=True)
