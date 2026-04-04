import math

import numpy as np
import pytest

# Adjust the import below if your functions are in a different module.
from atroposlib.utils.advantages import (
    allclose_to_first,
    compute_discounted_returns,
    compute_grpo_process_supervision_advantages,
    compute_stats,
)


def test_allclose_to_first_all_close():
    """Test that identical values return True."""
    values = [1.0, 1.0, 1.0]
    result = allclose_to_first(values)
    assert result is True


def test_allclose_to_first_vector():
    """Test that return_vector=True returns a tensor of booleans."""
    values = [1.0, 1.000000001, 1.000000002]
    result = allclose_to_first(values, return_vector=True)
    assert isinstance(result, np.ndarray)
    # All comparisons should be True.
    assert np.all(result)


def test_allclose_to_first_not_close():
    """Test that values which are not close yield False."""
    values = [1.0, 1.0, 1.1]
    result = allclose_to_first(values)
    assert result is False


def test_allclose_to_first_nan():
    """Test handling of NaN values with equal_nan parameter."""
    values = [float("nan"), float("nan")]
    # With equal_nan False, the result should be False.
    result = allclose_to_first(values, equal_nan=False)
    assert result is False
    # With equal_nan True, NaNs are treated as equal.
    result = allclose_to_first(values, equal_nan=True)
    assert result is True


def test_compute_stats():
    """Test compute_stats with a nested list of numbers."""
    data = [1, 2, 3, [4, 5]]
    stats = compute_stats(data)
    # mean = (1+2+3+4+5)/5 = 3.0
    assert math.isclose(stats["mean"], 3.0, rel_tol=1e-5)
    # variance = (11 - 9) = 2.0, since average of squares = 55/5 = 11 and mean^2 = 9.
    assert math.isclose(stats["var"], 2.0, rel_tol=1e-5)


def test_compute_stats_empty():
    """Test that an empty list raises a ValueError."""
    with pytest.raises(ValueError):
        compute_stats([])


def test_compute_stats_jagged():
    """Test compute_stats with a deeper, jagged nested list."""
    data = [[1, 2], 3, [4, [5, 6]]]
    stats = compute_stats(data)
    expected_mean = (1 + 2 + 3 + 4 + 5 + 6) / 6  # 21/6 = 3.5
    expected_var = ((1**2 + 2**2 + 3**2 + 4**2 + 5**2 + 6**2) / 6) - expected_mean**2
    assert math.isclose(stats["mean"], expected_mean, rel_tol=1e-5)
    assert math.isclose(stats["var"], expected_var, rel_tol=1e-5)


def test_compute_discounted_returns():
    """Test compute_discounted_returns with a tensor input."""
    rewards = np.array([1.0, 1.0, 1.0])
    gamma = 0.9
    returns = compute_discounted_returns(rewards, gamma)
    # For a 3-element vector:
    # t=2: 1.0
    # t=1: 1.0 + 0.9*1.0 = 1.9
    # t=0: 1.0 + 0.9*1.9 = 2.71
    expected = np.array([2.71, 1.9, 1.0])
    assert np.allclose(returns, expected, rtol=1e-5, atol=1e-8)


def test_compute_discounted_returns_list_input():
    """Test compute_discounted_returns when the input is a list."""
    rewards = [1, 1, 1]
    gamma = 0.0  # With gamma=0, the returns should equal the rewards.
    returns = compute_discounted_returns(rewards, gamma)
    expected = np.array([1.0, 1.0, 1.0])
    assert np.allclose(returns, expected, rtol=1e-5, atol=1e-8)


def test_compute_grpo_process_supervision_advantages_cumsum():
    """
    Test compute_grpo_process_supervision_advantages with gamma=None,
    which should now compute a reversed cumulative sum on normalized rewards.
    For each trajectory, the expected advantage at index i is the sum of normalized rewards from i to the end.
    """
    rewards = [[1, 2, 3], [4, 5]]
    advantages = compute_grpo_process_supervision_advantages(rewards, gamma=None)
    # Compute normalized rewards using flattened stats (mean=3, var=2 so std=sqrt(2))
    sqrt2 = math.sqrt(2)
    # For trajectory 1, normalized rewards:
    # Reversed cumulative sum:
    # index 0: sum(traj1) = (-2/sqrt2) + (-1/sqrt2) + 0 = -3/sqrt2
    # index 1: sum(traj1[1:]) = (-1/sqrt2) + 0 = -1/sqrt2
    # index 2: sum(traj1[2:]) = 0
    expected_traj1 = [-3 / sqrt2, -1 / sqrt2, 0]
    # For trajectory 2, normalized rewards:
    # Reversed cumulative sum:
    # index 0: (1/sqrt2) + (2/sqrt2) = 3/sqrt2
    # index 1: (2/sqrt2)
    expected_traj2 = [3 / sqrt2, 2 / sqrt2]

    adv1 = advantages[0].tolist()
    adv2 = advantages[1].tolist()

    for computed, expected in zip(adv1, expected_traj1):
        assert math.isclose(
            computed, expected, rel_tol=1e-5
        ), f"Computed {computed} vs expected {expected} in trajectory 1"
    for computed, expected in zip(adv2, expected_traj2):
        assert math.isclose(
            computed, expected, rel_tol=1e-5
        ), f"Computed {computed} vs expected {expected} in trajectory 2"


def test_compute_grpo_process_supervision_advantages_discounted():
    """
    Test compute_grpo_process_supervision_advantages with a provided gamma,
    which should compute discounted returns on normalized rewards.
    """
    rewards = [[1, 2, 3], [4, 5]]
    gamma = 0.9
    advantages = compute_grpo_process_supervision_advantages(rewards, gamma=gamma)
    sqrt2 = math.sqrt(2)
    # Normalized first trajectory:
    a1 = (1 - 3) / sqrt2  # -2/sqrt2
    a2 = (2 - 3) / sqrt2  # -1/sqrt2
    a3 = (3 - 3) / sqrt2  # 0
    # Discounted returns for trajectory 1:
    # t=2: a3
    # t=1: a2 + gamma * a3 = a2
    # t=0: a1 + gamma * (a2 + gamma * a3) = a1 + gamma * a2
    expected_traj1 = [a1 + gamma * a2, a2, a3]
    # Normalized second trajectory:
    b1 = (4 - 3) / sqrt2  # 1/sqrt2
    b2 = (5 - 3) / sqrt2  # 2/sqrt2
    # Discounted returns for trajectory 2:
    # t=1: b2
    # t=0: b1 + gamma * b2
    expected_traj2 = [b1 + gamma * b2, b2]
    adv1 = advantages[0].tolist()
    adv2 = advantages[1].tolist()
    for computed, expected in zip(adv1, expected_traj1):
        assert math.isclose(computed, expected, rel_tol=1e-5)
    for computed, expected in zip(adv2, expected_traj2):
        assert math.isclose(computed, expected, rel_tol=1e-5)


def test_compute_grpo_process_supervision_advantages_std_tol():
    """Test that a constant reward trajectory raises ValueError due to low std."""
    rewards = [[1, 1, 1]]
    with pytest.raises(ValueError):
        compute_grpo_process_supervision_advantages(rewards)
