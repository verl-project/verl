import numpy as np
import torch

from verl.experimental.agent_framework.assembler import TrajectoryAssembler
from verl.experimental.agent_framework.types import Trajectory


def test_trajectory_assembler_matches_training_batch_contract():
    trajectories = [
        Trajectory(
            uid="sample-0",
            session_id="session-0",
            trajectory_id=0,
            prompt_ids=[10, 11],
            response_ids=[20, 21, 22],
            response_mask=[1, 1, 0],
            response_logprobs=[-0.1, -0.2, 0.0],
            reward_info={"score": 0.5, "label": "alpha"},
            reward_score=0.5,
            num_turns=2,
            routed_experts=torch.tensor(
                [
                    [[1], [2]],
                    [[3], [4]],
                    [[5], [6]],
                    [[7], [8]],
                    [[9], [10]],
                ],
                dtype=torch.int64,
            ),
        ),
        Trajectory(
            uid="sample-1",
            session_id="session-1",
            trajectory_id=0,
            prompt_ids=[30],
            response_ids=[40, 41],
            response_mask=[1, 1],
            response_logprobs=[-0.3, -0.4],
            reward_info={"score": 1.5, "label": "beta"},
            reward_score=1.5,
            num_turns=3,
            routed_experts=torch.tensor(
                [
                    [[11], [12]],
                    [[13], [14]],
                    [[15], [16]],
                ],
                dtype=torch.int64,
            ),
        ),
    ]

    output = TrajectoryAssembler(pad_token_id=0).assemble(trajectories)

    assert tuple(output.batch["prompts"].shape) == (2, 2)
    assert tuple(output.batch["responses"].shape) == (2, 3)
    assert tuple(output.batch["response_mask"].shape) == (2, 3)
    assert tuple(output.batch["input_ids"].shape) == (2, 5)
    assert tuple(output.batch["attention_mask"].shape) == (2, 5)
    assert tuple(output.batch["position_ids"].shape) == (2, 5)
    assert tuple(output.batch["rollout_log_probs"].shape) == (2, 3)
    assert tuple(output.batch["routed_experts"].shape) == (2, 5, 2, 1)
    assert tuple(output.batch["rm_scores"].shape) == (2, 3)

    torch.testing.assert_close(
        output.batch["prompts"],
        torch.tensor(
            [
                [10, 11],
                [0, 30],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(
        output.batch["responses"],
        torch.tensor(
            [
                [20, 21, 22],
                [40, 41, 0],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(
        output.batch["response_mask"],
        torch.tensor(
            [
                [1, 1, 0],
                [1, 1, 0],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(
        output.batch["attention_mask"],
        torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(
        output.batch["position_ids"],
        torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [0, 0, 1, 2, 2],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(
        output.batch["rm_scores"],
        torch.tensor(
            [
                [0.0, 0.0, 0.5],
                [0.0, 1.5, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        output.batch["rollout_log_probs"],
        torch.tensor(
            [
                [-0.1, -0.2, 0.0],
                [-0.3, -0.4, 0.0],
            ],
            dtype=torch.float32,
        ),
    )

    second_experts = output.batch["routed_experts"][1]
    expected_second_experts = torch.zeros((5, 2, 1), dtype=torch.int64)
    expected_second_experts[1:4] = torch.tensor(
        [
            [[11], [12]],
            [[13], [14]],
            [[15], [16]],
        ],
        dtype=torch.int64,
    )
    torch.testing.assert_close(second_experts, expected_second_experts)

    assert np.array_equal(output.non_tensor_batch["__num_turns__"], np.array([2, 3], dtype=np.int32))
    assert np.array_equal(output.non_tensor_batch["label"], np.array(["alpha", "beta"], dtype=object))
    assert output.meta_info["reward_extra_keys"] == ["score", "label"]
