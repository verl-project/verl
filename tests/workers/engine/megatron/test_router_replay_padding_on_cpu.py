import torch

from verl.utils.megatron.router_replay_patch import (
    RouterReplay,
    RouterReplayAction,
    _patched_topk_routing_with_score_function,
)


def test_record_canonicalizes_thd_alignment_padding_before_dispatch():
    router_replay = RouterReplay()
    router_replay.set_router_replay_action(RouterReplayAction.RECORD)
    router_replay.record_padding_mask = torch.tensor([False, True])
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    _probs, routing_map = _patched_topk_routing_with_score_function(
        logits=logits,
        topk=1,
        use_pre_softmax=False,
        num_groups=None,
        group_topk=None,
        score_function="softmax",
        expert_bias=None,
        fused=False,
        router_replay=router_replay,
        scaling_factor=1.0,
    )

    expected = torch.tensor([[2], [0]])
    assert torch.equal(router_replay.recorded_topk_idx, expected)
    assert routing_map[0, 2].item()
    assert routing_map[1, 0].item()


def test_record_padding_mask_shape_mismatch_hard_fails():
    router_replay = RouterReplay()
    router_replay.record_padding_mask = torch.tensor([True])

    try:
        router_replay.canonicalize_record_topk_indices(torch.tensor([[1], [2]]))
    except RuntimeError as exc:
        assert "padding mask does not match" in str(exc)
    else:
        raise AssertionError("expected router padding-mask shape mismatch to fail")
