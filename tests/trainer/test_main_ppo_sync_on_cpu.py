# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import importlib
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from tensordict import TensorDict

from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import AdvantageEstimator

_MAIN_PPO_SYNC = None


def _build_grpo_multi_session_batch() -> DataProto:
    """Build one prompt with two sessions and uneven trajectory counts."""
    return DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": torch.tensor(
                    [
                        [100.0, 0.0, 0.0],
                        [1.0, 2.0, 3.0],
                        [200.0, 0.0, 0.0],
                        [150.0, 150.0, 0.0],
                        [4.0, 5.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                "values": torch.zeros((5, 3), dtype=torch.float32),
                "response_mask": torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [1.0, 1.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
            },
            batch_size=(5,),
        ),
        non_tensor_batch={"uid": np.array(["prompt_a", "prompt_a", "prompt_a", "prompt_a", "prompt_a"], dtype=object)},
    )


def _build_grpo_batch_with_missing_session_output() -> DataProto:
    """Build one prompt where session 1 has no output and contributes no row."""
    return DataProto(
        batch=TensorDict(
            {
                "token_level_rewards": torch.tensor(
                    [
                        [100.0, 0.0, 0.0],
                        [1.0, 2.0, 3.0],
                        [3.0, 3.0, 3.0],
                        [12.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
                "values": torch.zeros((4, 3), dtype=torch.float32),
                "response_mask": torch.tensor(
                    [
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                ),
            },
            batch_size=(4,),
        ),
        non_tensor_batch={"uid": np.array(["prompt_a", "prompt_a", "prompt_a", "prompt_a"], dtype=object)},
    )


def _import_main_ppo_sync():
    global _MAIN_PPO_SYNC
    transfer_queue_stub = types.ModuleType("transfer_queue")
    transfer_queue_stub.KVBatchMeta = type("KVBatchMeta", (), {})

    if _MAIN_PPO_SYNC is not None:
        return _MAIN_PPO_SYNC

    with patch.dict(sys.modules, {"transfer_queue": transfer_queue_stub}):
        _MAIN_PPO_SYNC = importlib.import_module("verl.trainer.main_ppo_sync")
    return _MAIN_PPO_SYNC


class TestMainPPOSync(unittest.TestCase):
    def _assert_allclose(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        self.assertTrue(torch.allclose(actual, expected))

    def _assert_zeros(self, tensor: torch.Tensor) -> None:
        self.assertTrue(torch.equal(tensor, torch.zeros_like(tensor)))

    def test_compute_advantage_for_multi_trajectories_delegates_non_grpo_estimators(self):
        """Test that non-GRPO estimators reuse the original compute_advantage path."""
        main_ppo_sync = _import_main_ppo_sync()
        data = _build_grpo_multi_session_batch()

        result = main_ppo_sync.compute_advantage_for_multi_trajectories(
            data=data,
            batch_keys=["bad-key-0", "bad-key-1", "bad-key-2", "bad-key-3", "bad-key-4"],
            adv_estimator=AdvantageEstimator.GAE,
            gamma=1.0,
            lam=1.0,
            num_repeat=1,
            config={},
        )
        expected = main_ppo_sync.compute_advantage(
            _build_grpo_multi_session_batch(),
            adv_estimator=AdvantageEstimator.GAE,
            gamma=1.0,
            lam=1.0,
            num_repeat=1,
            config={},
        )

        self._assert_allclose(result.batch["advantages"], expected.batch["advantages"])
        self._assert_allclose(result.batch["returns"], expected.batch["returns"])

    def test_compute_advantage_for_multi_trajectories_uses_final_output_per_session_in_grpo(self):
        """Test that GRPO only uses each session's final output to compute advantage."""
        main_ppo_sync = _import_main_ppo_sync()
        data = _build_grpo_multi_session_batch()

        result = main_ppo_sync.compute_advantage_for_multi_trajectories(
            data=data,
            batch_keys=["prompt_a_0_0", "prompt_a_0_1", "prompt_a_1_0", "prompt_a_1_1", "prompt_a_1_2"],
            adv_estimator=AdvantageEstimator.GRPO,
            num_repeat=1,
            norm_adv_by_std_in_grpo=True,
            config={},
        )

        expected_final = main_ppo_sync.compute_advantage(
            _build_grpo_multi_session_batch().select_idxs([1, 4]),
            adv_estimator=AdvantageEstimator.GRPO,
            num_repeat=1,
            norm_adv_by_std_in_grpo=True,
            config={},
        )

        expected_session0_scalar_adv = expected_final.batch["advantages"][0, 2]
        expected_session0_scalar_ret = expected_final.batch["returns"][0, 2]
        self._assert_allclose(result.batch["advantages"][0, :1], expected_session0_scalar_adv.expand(1))
        self._assert_allclose(result.batch["returns"][0, :1], expected_session0_scalar_ret.expand(1))
        self._assert_zeros(result.batch["advantages"][0, 1:])
        self._assert_zeros(result.batch["returns"][0, 1:])
        self._assert_allclose(result.batch["advantages"][1, :3], expected_final.batch["advantages"][0, :3])
        self._assert_allclose(result.batch["returns"][1, :3], expected_final.batch["returns"][0, :3])

        expected_session1_scalar_adv = expected_final.batch["advantages"][1, 1]
        expected_session1_scalar_ret = expected_final.batch["returns"][1, 1]
        self._assert_allclose(result.batch["advantages"][2, :1], expected_session1_scalar_adv.expand(1))
        self._assert_allclose(result.batch["returns"][2, :1], expected_session1_scalar_ret.expand(1))
        self._assert_zeros(result.batch["advantages"][2, 1:])
        self._assert_zeros(result.batch["returns"][2, 1:])
        self._assert_allclose(result.batch["advantages"][3, :2], expected_final.batch["advantages"][1, :2])
        self._assert_allclose(result.batch["returns"][3, :2], expected_final.batch["returns"][1, :2])
        self._assert_allclose(result.batch["advantages"][4, :2], expected_final.batch["advantages"][1, :2])
        self._assert_allclose(result.batch["returns"][4, :2], expected_final.batch["returns"][1, :2])
        self._assert_zeros(result.batch["advantages"][3:, 2])
        self._assert_zeros(result.batch["returns"][3:, 2])

        self.assertLess(result.batch["advantages"][0, 0], 0)
        self.assertGreater(result.batch["advantages"][2, 0], 0)

    def test_compute_advantage_for_multi_trajectories_skips_sessions_with_none_output(self):
        """Test that absent AgentLoop outputs simply leave those sessions out of GRPO."""
        main_ppo_sync = _import_main_ppo_sync()
        data = _build_grpo_batch_with_missing_session_output()

        result = main_ppo_sync.compute_advantage_for_multi_trajectories(
            data=data,
            batch_keys=["prompt_a_0_0", "prompt_a_0_1", "prompt_a_2_0", "prompt_a_3_0"],
            adv_estimator=AdvantageEstimator.GRPO,
            num_repeat=1,
            norm_adv_by_std_in_grpo=True,
            config={},
        )

        expected_final = main_ppo_sync.compute_advantage(
            _build_grpo_batch_with_missing_session_output().select_idxs([1, 2, 3]),
            adv_estimator=AdvantageEstimator.GRPO,
            num_repeat=1,
            norm_adv_by_std_in_grpo=True,
            config={},
        )

        expected_session0_scalar_adv = expected_final.batch["advantages"][0, 2]
        expected_session0_scalar_ret = expected_final.batch["returns"][0, 2]
        self._assert_allclose(result.batch["advantages"][0, :1], expected_session0_scalar_adv.expand(1))
        self._assert_allclose(result.batch["returns"][0, :1], expected_session0_scalar_ret.expand(1))
        self._assert_zeros(result.batch["advantages"][0, 1:])
        self._assert_zeros(result.batch["returns"][0, 1:])
        self._assert_allclose(result.batch["advantages"][1, :3], expected_final.batch["advantages"][0, :3])
        self._assert_allclose(result.batch["returns"][1, :3], expected_final.batch["returns"][0, :3])

        self._assert_allclose(result.batch["advantages"][2, :3], expected_final.batch["advantages"][1, :3])
        self._assert_allclose(result.batch["returns"][2, :3], expected_final.batch["returns"][1, :3])
        self._assert_allclose(result.batch["advantages"][3, :1], expected_final.batch["advantages"][2, :1])
        self._assert_allclose(result.batch["returns"][3, :1], expected_final.batch["returns"][2, :1])
        self._assert_zeros(result.batch["advantages"][3, 1:])
        self._assert_zeros(result.batch["returns"][3, 1:])

        self.assertLess(result.batch["advantages"][0, 0], 0)
        self.assertAlmostEqual(result.batch["advantages"][2, 0].item(), 0.0, places=6)
        self.assertGreater(result.batch["advantages"][3, 0], 0)

    def test_compute_advantage_for_multi_trajectories_fails_for_unexpected_keys(self):
        """Test that invalid key format raises instead of silently falling back."""
        main_ppo_sync = _import_main_ppo_sync()
        data = _build_grpo_multi_session_batch()

        with self.assertRaisesRegex(ValueError, r"Unexpected batch key format: bad-key-0"):
            main_ppo_sync.compute_advantage_for_multi_trajectories(
                data=data,
                batch_keys=["bad-key-0", "bad-key-1", "bad-key-2", "bad-key-3", "bad-key-4"],
                adv_estimator=AdvantageEstimator.GRPO,
                num_repeat=1,
                norm_adv_by_std_in_grpo=True,
                config={},
            )


if __name__ == "__main__":
    unittest.main()
