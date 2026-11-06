import numpy as np
import torch

from verl.trainer.ppo.core_algos import get_adv_estimator_fn


def test_grpo_segment_normalizes_within_uid_and_broadcasts():
    estimator = get_adv_estimator_fn("grpo_segment")
    rewards = torch.zeros(2, 4)
    mask = torch.ones(2, 4)
    advantages, returns = estimator(
        token_level_rewards=rewards,
        response_mask=mask,
        index=np.array(["g", "g"]),
        non_tensor_batch={
            "segment_boundaries": [
                [{"response_start": 0, "response_end": 4, "num_segments_in_rollout": 1}],
                [{"response_start": 0, "response_end": 4, "num_segments_in_rollout": 1}],
            ],
            "combined_rewards": [[0.0], [2.0]],
        },
    )
    scores = torch.tensor([0.0, 2.0])
    scale = torch.std(scores) + 1e-6
    assert torch.allclose(advantages[0], torch.full((4,), float((0.0 - scores.mean()) / scale)))
    assert torch.allclose(advantages[1], torch.full((4,), float((2.0 - scores.mean()) / scale)))
    assert torch.equal(advantages, returns)


def test_grpo_segment_single_segment_group_is_zero():
    estimator = get_adv_estimator_fn("grpo_segment")
    advantages, _ = estimator(
        token_level_rewards=torch.zeros(1, 2),
        response_mask=torch.ones(1, 2),
        index=np.array(["only"]),
        non_tensor_batch={
            "segment_boundaries": [[{"response_start": 0, "response_end": 2}]],
            "combined_rewards": [[1.5]],
        },
    )
    assert torch.equal(advantages, torch.zeros(1, 2))
