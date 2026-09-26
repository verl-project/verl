# Copyright 2026 Bytedance Ltd. and/or its affiliates

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from verl.utils.megatron.router_replay_patch import (
    RouterReplay,
    RouterReplayAction,
    apply_router_replay_patch,
)
from verl.utils.megatron.router_replay_utils import iter_model_routers

deepseek_v41_moe = pytest.importorskip("megatron.core.models.deepseek_v41.moe")
ModalityRouter = deepseek_v41_moe.ModalityRouter


class _Balance(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.register_buffer("expert_bias", torch.tensor(bias, dtype=torch.float32))
        self.register_buffer("local_tokens_per_expert", torch.zeros(len(bias), dtype=torch.int64))


@pytest.fixture
def isolated_router_registry():
    previous = RouterReplay.router_instances
    RouterReplay.router_instances = []
    try:
        yield
    finally:
        RouterReplay.router_instances = previous


def test_modality_router_records_and_replays_experts_without_changing_scores(isolated_router_registry):
    del isolated_router_registry
    apply_router_replay_patch()
    router = ModalityRouter.__new__(ModalityRouter)
    nn.Module.__init__(router)
    router.config = SimpleNamespace(
        moe_router_topk=1,
        moe_router_topk_scaling_factor=2.0,
        moe_router_enable_expert_bias=True,
    )
    router.num_experts = 4
    router.text_balance = _Balance([10.0, 0.0, 0.0, 0.0])
    router.image_balance = _Balance([0.0, 0.0, 0.0, 10.0])
    router.router_replay = RouterReplay()
    router.train()

    logits = torch.zeros((2, 1, 4), dtype=torch.float32)
    image_mask = torch.tensor([[False], [True]])
    router.router_replay.set_router_replay_action(RouterReplayAction.RECORD)
    recorded_probs, recorded_route = router.routing(logits, image_mask=image_mask)

    assert torch.equal(router.router_replay.recorded_topk_idx, torch.tensor([[0], [3]]))
    assert torch.equal(recorded_route.nonzero()[:, 1], torch.tensor([0, 3]))

    replay_indices = torch.tensor([[2], [1]])
    router.router_replay.set_target_indices(replay_indices)
    router.router_replay.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    replay_probs, replay_route = router.routing(logits, image_mask=image_mask)

    expected_weight = torch.nn.functional.softplus(torch.tensor(0.0)).sqrt() * 2.0
    assert torch.equal(replay_route.nonzero()[:, 1], replay_indices[:, 0])
    assert torch.allclose(replay_probs.sum(-1), torch.full((2,), expected_weight))
    assert torch.allclose(recorded_probs.sum(-1), replay_probs.sum(-1))


def test_modality_router_replay_keeps_padding_dispatch_rows_out_of_balance_counts(isolated_router_registry):
    del isolated_router_registry
    apply_router_replay_patch()
    router = ModalityRouter.__new__(ModalityRouter)
    nn.Module.__init__(router)
    router.config = SimpleNamespace(
        moe_router_topk=1,
        moe_router_topk_scaling_factor=1.0,
        moe_router_enable_expert_bias=True,
    )
    router.num_experts = 2
    router.text_balance = _Balance([1.0, 0.0])
    router.image_balance = _Balance([0.0, 1.0])
    router.router_replay = RouterReplay()
    router.router_replay.set_router_replay_action(RouterReplayAction.RECORD)
    router.train()

    probs, route = router.routing(
        torch.zeros((3, 1, 2)),
        image_mask=torch.tensor([[False], [True], [False]]),
        padding_mask=torch.tensor([[False], [False], [True]]),
    )

    torch.testing.assert_close(route.sum(-1), torch.ones(3, dtype=torch.long))
    torch.testing.assert_close(probs[2], torch.zeros(2))
    assert router.text_balance.local_tokens_per_expert.sum() == 1
    assert router.image_balance.local_tokens_per_expert.sum() == 1


def test_model_walk_finds_deepseek_v41_modality_router(isolated_router_registry):
    del isolated_router_registry
    router = ModalityRouter.__new__(ModalityRouter)
    nn.Module.__init__(router)
    router.layer_number = 3
    router.router_replay = RouterReplay()

    model = nn.Module()
    model.decoder = nn.ModuleList([router])

    assert list(iter_model_routers(model)) == [(3, router.router_replay)]
