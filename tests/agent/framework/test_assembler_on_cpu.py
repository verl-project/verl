import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl.agent.framework.assembler import TrajectoryAssembler
from verl.agent.framework.types import Trajectory
from verl.utils import tensordict_utils as tu


def _build_trajectory(**overrides) -> Trajectory:
    fields = {
        "uid": "sample-0",
        "session_id": "session-0",
        "trajectory_id": 0,
        "prompt_ids": [10],
        "response_ids": [20, 21],
        "response_mask": [1, 1],
        "response_logprobs": [-0.1, -0.2],
        "reward_info": {},
        "reward_score": 1.0,
        "num_turns": 1,
        "routed_experts": None,
    }
    fields.update(overrides)
    return Trajectory(**fields)


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

    assert isinstance(output, TensorDict)
    assert tuple(output["prompts"].shape) == (2, 2)
    assert tuple(output["responses"].shape) == (2, 3)
    assert tuple(output["response_mask"].shape) == (2, 3)
    assert tuple(output["input_ids"].shape) == (2, 5)
    assert tuple(output["attention_mask"].shape) == (2, 5)
    assert tuple(output["position_ids"].shape) == (2, 5)
    assert tuple(output["rollout_log_probs"].shape) == (2, 3)
    assert tuple(output["routed_experts"].shape) == (2, 5, 2, 1)
    assert tuple(output["rm_scores"].shape) == (2, 3)

    torch.testing.assert_close(
        output["prompts"],
        torch.tensor(
            [
                [10, 11],
                [0, 30],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(
        output["responses"],
        torch.tensor(
            [
                [20, 21, 22],
                [40, 41, 0],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(
        output["response_mask"],
        torch.tensor(
            [
                [1, 1, 0],
                [1, 1, 0],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(
        output["attention_mask"],
        torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [0, 1, 1, 1, 0],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(
        output["position_ids"],
        torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [0, 0, 1, 2, 2],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(
        output["rm_scores"],
        torch.tensor(
            [
                [0.0, 0.0, 0.5],
                [0.0, 1.5, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
    torch.testing.assert_close(
        output["rollout_log_probs"],
        torch.tensor(
            [
                [-0.1, -0.2, 0.0],
                [-0.3, -0.4, 0.0],
            ],
            dtype=torch.float32,
        ),
    )

    second_experts = output["routed_experts"][1]
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

    assert np.array_equal(np.array(tu.get(output, "__num_turns__"), dtype=np.int32), np.array([2, 3], dtype=np.int32))
    assert np.array_equal(np.array(tu.get(output, "label"), dtype=object), np.array(["alpha", "beta"], dtype=object))
    assert tu.get(output, "reward_extra_keys") == ["score", "label"]


def test_trajectory_assembler_rejects_empty_trajectories():
    with pytest.raises(ValueError, match="non-empty"):
        TrajectoryAssembler().assemble([])


def test_trajectory_assembler_rejects_response_mask_length_mismatch():
    with pytest.raises(ValueError, match="response_mask length must match response_ids length"):
        TrajectoryAssembler().assemble(
            [
                _build_trajectory(
                    response_ids=[20, 21],
                    response_mask=[1],
                )
            ]
        )


def test_trajectory_assembler_rejects_response_logprobs_length_mismatch():
    with pytest.raises(ValueError, match="response_logprobs length must match response_ids length"):
        TrajectoryAssembler().assemble(
            [
                _build_trajectory(
                    response_ids=[20, 21],
                    response_logprobs=[-0.1],
                )
            ]
        )


def test_trajectory_assembler_requires_reward_score():
    with pytest.raises(ValueError, match="has no reward_score"):
        TrajectoryAssembler().assemble(
            [
                _build_trajectory(reward_score=None),
            ]
        )


def test_trajectory_assembler_supports_numpy_routed_experts():
    routed_experts = np.array(
        [
            [[1], [2]],
            [[3], [4]],
            [[5], [6]],
        ],
        dtype=np.int64,
    )

    output = TrajectoryAssembler().assemble(
        [
            _build_trajectory(
                prompt_ids=[10],
                response_ids=[20, 21],
                response_mask=[1, 1],
                routed_experts=routed_experts,
            )
        ]
    )

    assert tuple(output["routed_experts"].shape) == (1, 3, 2, 1)
    assert output["routed_experts"].dtype == torch.int64
    torch.testing.assert_close(output["routed_experts"][0], torch.from_numpy(routed_experts))
