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

"""Unit tests for MOPD sub-batch teacher routing logic.

Tests the routing logic in isolation without requiring Ray or GPU resources.
The core function `compute_teacher_log_probs_standalone` replicates the
routing algorithm from `RayPPOTrainer._compute_teacher_log_probs`.
"""

import numpy as np
import torch

from verl import DataProto


class MockTeacherWG:
    """Mock teacher worker group that returns deterministic ref_log_prob.

    Each mock teacher produces a unique constant value so tests can verify
    that results are correctly scattered back to the right batch indices.
    """

    def __init__(self, fill_value: float = 0.0):
        self.fill_value = fill_value
        self.call_count = 0
        self.last_batch_size = 0

    def compute_ref_log_prob(self, sub_batch: DataProto) -> DataProto:
        """Return a DataProto with ref_log_prob filled with self.fill_value."""
        batch_size = sub_batch.batch["responses"].shape[0]
        response_len = sub_batch.batch["responses"].shape[1]
        self.call_count += 1
        self.last_batch_size = batch_size
        result = DataProto.from_single_dict(
            {
                "ref_log_prob": torch.full(
                    (batch_size, response_len),
                    self.fill_value,
                    dtype=torch.float32,
                ),
            }
        )
        return result


def compute_teacher_log_probs_standalone(
    batch: DataProto,
    teacher_wgs: dict,
) -> torch.Tensor:
    """Standalone implementation of sub-batch teacher routing.

    This replicates the logic of RayPPOTrainer._compute_teacher_log_probs
    without requiring a trainer instance, for unit testing purposes.
    """
    teacher_ids = batch.non_tensor_batch["teacher_id"]
    batch_size = len(teacher_ids)
    response_len = batch.batch["responses"].shape[1]

    # Initialize output tensor (Fix 2: stacked storage)
    teacher_log_probs = torch.zeros(
        batch_size,
        response_len,
        dtype=torch.float32,
        device=batch.batch["responses"].device,
    )

    # Group by teacher_id and forward sub-batches (Fix 3)
    for teacher_name, teacher_wg in teacher_wgs.items():
        # Get indices for this teacher (Fix 7: integer tensor)
        mask = teacher_ids == teacher_name
        indices = torch.tensor(np.where(mask)[0], dtype=torch.long)

        if len(indices) == 0:
            continue

        # Select sub-batch (Fix 7: use select_idxs, not select)
        sub_batch = batch.select_idxs(indices)

        # Forward to teacher
        teacher_output = teacher_wg.compute_ref_log_prob(sub_batch)
        sub_log_probs = teacher_output.batch["ref_log_prob"]

        # Scatter back to full batch (Fix 4: correct broadcasting)
        teacher_log_probs[indices] = sub_log_probs

    return teacher_log_probs


def test_teacher_log_prob_basic_shape():
    """Test that teacher log prob computation returns correct shape."""
    # Arrange
    batch = DataProto.from_single_dict(
        {
            "input_ids": torch.randint(0, 1000, (4, 128)),
            "responses": torch.randint(0, 1000, (4, 64)),
            "attention_mask": torch.ones(4, 192),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "math", "code", "code"])
    teacher_wgs = {"math": MockTeacherWG(), "code": MockTeacherWG()}

    # Act
    teacher_log_prob = compute_teacher_log_probs_standalone(batch, teacher_wgs)

    # Assert: correct shape and dtype
    assert teacher_log_prob.shape == (4, 64)
    assert teacher_log_prob.dtype == torch.float32


def test_teacher_log_prob_correct_routing():
    """Test that sub-batches are correctly split by teacher_id and results scattered back."""
    # Arrange
    batch = DataProto.from_single_dict(
        {
            "input_ids": torch.randint(0, 1000, (6, 32)),
            "responses": torch.randint(0, 1000, (6, 16)),
            "attention_mask": torch.ones(6, 48),
        }
    )
    # Indices 0, 2, 4 -> math; indices 1, 3, 5 -> code
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "code", "math", "code", "math", "code"])

    math_wg = MockTeacherWG(fill_value=1.0)
    code_wg = MockTeacherWG(fill_value=2.0)
    teacher_wgs = {"math": math_wg, "code": code_wg}

    # Act
    teacher_log_prob = compute_teacher_log_probs_standalone(batch, teacher_wgs)

    # Assert: math samples (indices 0, 2, 4) should have value 1.0
    torch.testing.assert_close(teacher_log_prob[0], torch.ones(16) * 1.0)
    torch.testing.assert_close(teacher_log_prob[2], torch.ones(16) * 1.0)
    torch.testing.assert_close(teacher_log_prob[4], torch.ones(16) * 1.0)

    # Assert: code samples (indices 1, 3, 5) should have value 2.0
    torch.testing.assert_close(teacher_log_prob[1], torch.ones(16) * 2.0)
    torch.testing.assert_close(teacher_log_prob[3], torch.ones(16) * 2.0)
    torch.testing.assert_close(teacher_log_prob[5], torch.ones(16) * 2.0)


def test_teacher_log_prob_sub_batch_sizes():
    """Test that each teacher receives the correct sub-batch size."""
    # Arrange
    batch = DataProto.from_single_dict(
        {
            "input_ids": torch.randint(0, 1000, (5, 32)),
            "responses": torch.randint(0, 1000, (5, 16)),
            "attention_mask": torch.ones(5, 48),
        }
    )
    # 3 math samples, 2 code samples
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "math", "code", "math", "code"])

    math_wg = MockTeacherWG(fill_value=1.0)
    code_wg = MockTeacherWG(fill_value=2.0)
    teacher_wgs = {"math": math_wg, "code": code_wg}

    # Act
    compute_teacher_log_probs_standalone(batch, teacher_wgs)

    # Assert: each teacher was called once with correct sub-batch size
    assert math_wg.call_count == 1
    assert math_wg.last_batch_size == 3
    assert code_wg.call_count == 1
    assert code_wg.last_batch_size == 2


def test_teacher_log_prob_single_teacher():
    """Test routing when all samples go to a single teacher."""
    # Arrange
    batch = DataProto.from_single_dict(
        {
            "input_ids": torch.randint(0, 1000, (3, 32)),
            "responses": torch.randint(0, 1000, (3, 16)),
            "attention_mask": torch.ones(3, 48),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "math", "math"])

    math_wg = MockTeacherWG(fill_value=3.0)
    code_wg = MockTeacherWG(fill_value=4.0)
    teacher_wgs = {"math": math_wg, "code": code_wg}

    # Act
    teacher_log_prob = compute_teacher_log_probs_standalone(batch, teacher_wgs)

    # Assert: all samples routed to math teacher
    expected = torch.ones(3, 16) * 3.0
    torch.testing.assert_close(teacher_log_prob, expected)
    assert math_wg.call_count == 1
    assert math_wg.last_batch_size == 3
    # code teacher should not be called (skipped due to empty indices)
    assert code_wg.call_count == 0


def test_teacher_log_prob_empty_teacher():
    """Test that teachers with no matching samples are skipped gracefully."""
    # Arrange
    batch = DataProto.from_single_dict(
        {
            "input_ids": torch.randint(0, 1000, (2, 32)),
            "responses": torch.randint(0, 1000, (2, 16)),
            "attention_mask": torch.ones(2, 48),
        }
    )
    batch.non_tensor_batch["teacher_id"] = np.array(["math", "math"])

    math_wg = MockTeacherWG(fill_value=5.0)
    code_wg = MockTeacherWG(fill_value=6.0)
    unused_wg = MockTeacherWG(fill_value=7.0)
    teacher_wgs = {"math": math_wg, "code": code_wg, "unused": unused_wg}

    # Act
    teacher_log_prob = compute_teacher_log_probs_standalone(batch, teacher_wgs)

    # Assert: only math teacher was called
    assert math_wg.call_count == 1
    assert code_wg.call_count == 0
    assert unused_wg.call_count == 0
    expected = torch.ones(2, 16) * 5.0
    torch.testing.assert_close(teacher_log_prob, expected)
