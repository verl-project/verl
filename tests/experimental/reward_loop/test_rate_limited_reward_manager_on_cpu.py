# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import asyncio
import os.path
import time

import pytest
import torch
from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto
from verl.experimental.reward_loop.reward_manager.limited import RateLimitedRewardManager


# Mock API reward functions for testing
class MockAPICounter:
    """Shared counter to track API calls across tests."""

    def __init__(self):
        self.call_count = 0
        self.call_times = []
        self.lock = asyncio.Lock()

    async def record_call(self):
        async with self.lock:
            self.call_count += 1
            self.call_times.append(time.time())

    def reset(self):
        self.call_count = 0
        self.call_times.clear()

    def get_rate_per_second(self, window_start: float = None):
        """Calculate API call rate over a time window."""
        if window_start is None:
            if not self.call_times:
                return 0.0
            window_start = self.call_times[0]

        if not self.call_times:
            return 0.0

        window_end = self.call_times[-1]
        duration = window_end - window_start

        if duration <= 0:
            return 0.0

        calls_in_window = sum(1 for t in self.call_times if t >= window_start)
        return calls_in_window / duration


# Global counter instance
api_counter = MockAPICounter()


def mock_sync_reward_function(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict, **kwargs
) -> float:
    """Synchronous mock reward function that simulates API call."""
    # Simulate API processing time
    time.sleep(0.01)

    # Simple scoring logic
    score = 1.0 if solution_str.strip() == ground_truth.strip() else 0.0
    return score


async def mock_async_reward_function(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict, **kwargs
) -> float:
    """Asynchronous mock reward function that simulates API call."""
    # Record API call for rate tracking
    await api_counter.record_call()

    # Simulate async API call (e.g., HTTP request)
    await asyncio.sleep(0.01)

    # Simple scoring logic
    score = 1.0 if solution_str.strip() == ground_truth.strip() else 0.0
    return score


async def mock_slow_api_function(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict, **kwargs
) -> float:
    """Slow mock API function for timeout testing."""
    await asyncio.sleep(2.0)  # Simulate slow API
    return 0.5


async def mock_failing_api_function(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict, **kwargs
) -> float:
    """Mock API function that raises an exception."""
    await api_counter.record_call()
    raise ValueError("Simulated API error")


async def mock_dict_result_function(
    data_source: str, solution_str: str, ground_truth: str, extra_info: dict, **kwargs
) -> dict:
    """Mock API function that returns dict result."""
    await api_counter.record_call()
    await asyncio.sleep(0.01)

    correct = solution_str.strip() == ground_truth.strip()
    return {"score": 1.0 if correct else 0.0, "correct": correct, "reasoning": "Mock reasoning"}


def create_test_data_proto(tokenizer, response_text: str, ground_truth: str, data_source: str = "test"):
    """Helper to create DataProto for testing."""
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    response_tensor = torch.tensor([response_ids], dtype=torch.long)
    attention_mask = torch.ones_like(response_tensor)

    data = DataProto.from_dict(
        {
            "responses": response_tensor,
            "attention_mask": attention_mask,
        }
    )

    # Wrap non-tensor values in lists to match batch dimension
    data.non_tensor_batch = {"data_source": [data_source], "reward_model": [{"ground_truth": ground_truth}]}

    return data


class TestRateLimitedRewardManager:
    """Integration tests for RateLimitedRewardManager with mock API functions."""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Reset global state before each test."""
        api_counter.reset()
        # Reset class state
        RateLimitedRewardManager._class_initialized = False
        RateLimitedRewardManager._semaphore = None
        RateLimitedRewardManager._rpm_limiter = None
        RateLimitedRewardManager._tpm_limiter = None
        yield
        # Cleanup
        api_counter.reset()

    @pytest.fixture
    def tokenizer(self):
        """Load a simple tokenizer for testing."""
        return AutoTokenizer.from_pretrained(os.path.expanduser("~/models/Qwen/Qwen2.5-0.5B-Instruct"))

    @pytest.mark.asyncio
    async def test_basic_reward_computation(self, tokenizer):
        """Test basic reward computation without rate limiting."""
        config = DictConfig({"reward_model": {"max_concurrent": 10, "timeout": 10.0}})

        RateLimitedRewardManager.init_class(config, tokenizer)
        manager = RateLimitedRewardManager(config=config, tokenizer=tokenizer, compute_score=mock_async_reward_function)

        # Create test data
        data = create_test_data_proto(tokenizer, "correct answer", "correct answer")

        # Compute reward
        result = await manager.run_single(data)

        assert "reward_score" in result
        assert result["reward_score"] == 1.0
        assert api_counter.call_count == 1

    @pytest.mark.asyncio
    async def test_rpm_rate_limiting(self, tokenizer):
        """Test request per minute (RPM) rate limiting."""
        # Set RPM limit to 60 (1 request per second)
        config = DictConfig(
            {
                "reward_model": {
                    "max_concurrent": 10,
                    "max_rpm": 60,  # 1 request per second
                    "timeout": 10.0,
                }
            }
        )

        RateLimitedRewardManager.init_class(config, tokenizer)
        manager = RateLimitedRewardManager(config=config, tokenizer=tokenizer, compute_score=mock_async_reward_function)

        # Create test data
        data = create_test_data_proto(tokenizer, "answer", "answer")

        # Make 3 requests - should be rate limited
        start_time = time.time()

        results = []
        for _ in range(3):
            result = await manager.run_single(data)
            results.append(result)

        elapsed = time.time() - start_time

        # Should take at least ~2 seconds for 3 requests at 1 req/sec
        assert elapsed >= 1.8, f"RPM limiting failed: {elapsed:.3f}s for 3 requests"
        assert all(r["reward_score"] == 1.0 for r in results)
        assert api_counter.call_count == 3

    @pytest.mark.asyncio
    async def test_tpm_rate_limiting(self, tokenizer):
        """Test tokens per minute (TPM) rate limiting."""
        # Set TPM limit to 6000 (100 tokens per second)
        # With 2000 tokens per request, that's 0.05 req/sec or 20 seconds per request
        config = DictConfig(
            {
                "reward_model": {
                    "max_concurrent": 10,
                    "max_tpm": 6000,  # 100 tokens per second
                    "estimated_tokens_per_request": 2000,  # Each request = 2000 tokens
                    "timeout": 30.0,
                }
            }
        )

        RateLimitedRewardManager.init_class(config, tokenizer)
        manager = RateLimitedRewardManager(config=config, tokenizer=tokenizer, compute_score=mock_async_reward_function)

        data = create_test_data_proto(tokenizer, "answer", "answer")

        # Make 2 requests
        start_time = time.time()

        result1 = await manager.run_single(data)
        result2 = await manager.run_single(data)

        elapsed = time.time() - start_time

        # First request: consumes 2000 tokens (immediate)
        # Second request: needs 2000 tokens, waits for refill
        # Wait time: 2000 tokens / 100 tokens per second = 20 seconds
        assert elapsed >= 18.0, f"TPM limiting failed: {elapsed:.3f}s for 2 requests"
        assert result1["reward_score"] == 1.0
        assert result2["reward_score"] == 1.0

    @pytest.mark.asyncio
    async def test_concurrency_limiting(self, tokenizer):
        """Test concurrent request limiting."""
        config = DictConfig(
            {
                "reward_model": {
                    "max_concurrent": 2,  # Only 2 concurrent requests
                    "timeout": 10.0,
                }
            }
        )

        RateLimitedRewardManager.init_class(config, tokenizer)
        manager = RateLimitedRewardManager(config=config, tokenizer=tokenizer, compute_score=mock_async_reward_function)

        data = create_test_data_proto(tokenizer, "answer", "answer")

        # Launch 5 concurrent requests
        start_time = time.time()

        tasks = [manager.run_single(data) for _ in range(5)]
        results = await asyncio.gather(*tasks)

        elapsed = time.time() - start_time

        # All should succeed
        assert len(results) == 5
        assert all(r["reward_score"] == 1.0 for r in results)

        # With concurrency=2 and 0.01s per request, should take at least 0.03s
        # (3 batches: 2+2+1)
        assert elapsed >= 0.02, f"Concurrency limiting may not be working: {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_timeout_handling(self, tokenizer):
        """Test timeout handling for slow API."""
        config = DictConfig(
            {
                "reward_model": {
                    "max_concurrent": 10,
                    "timeout": 0.5,  # 500ms timeout
                }
            }
        )

        RateLimitedRewardManager.init_class(config, tokenizer)
        manager = RateLimitedRewardManager(config=config, tokenizer=tokenizer, compute_score=mock_slow_api_function)

        data = create_test_data_proto(tokenizer, "answer", "answer")

        # Should timeout and return 0.0
        result = await manager.run_single(data)

        assert result["reward_score"] == 0.0
        assert result["reward_extra_info"].get("timeout") is True
        assert result["reward_extra_info"].get("acc") == 0.0

    @pytest.mark.asyncio
    async def test_error_handling(self, tokenizer):
        """Test error handling for failing API."""
        config = DictConfig({"reward_model": {"max_concurrent": 10, "timeout": 10.0}})

        RateLimitedRewardManager.init_class(config, tokenizer)
        manager = RateLimitedRewardManager(config=config, tokenizer=tokenizer, compute_score=mock_failing_api_function)

        data = create_test_data_proto(tokenizer, "answer", "answer")

        # Should catch exception and return 0.0
        result = await manager.run_single(data)

        assert result["reward_score"] == 0.0
        assert "error" in result["reward_extra_info"]
        assert "Simulated API error" in result["reward_extra_info"]["error"]
        assert result["reward_extra_info"].get("acc") == 0.0
        assert api_counter.call_count == 1

    @pytest.mark.asyncio
    async def test_dict_result_format(self, tokenizer):
        """Test handling of dict return format from reward function."""
        config = DictConfig({"reward_model": {"max_concurrent": 10, "timeout": 10.0}})

        RateLimitedRewardManager.init_class(config, tokenizer)
        manager = RateLimitedRewardManager(config=config, tokenizer=tokenizer, compute_score=mock_dict_result_function)

        data = create_test_data_proto(tokenizer, "correct", "correct")

        result = await manager.run_single(data)

        assert result["reward_score"] == 1.0
        assert result["reward_extra_info"]["score"] == 1.0
        assert result["reward_extra_info"]["correct"] is True
        assert result["reward_extra_info"]["reasoning"] == "Mock reasoning"

    @pytest.mark.asyncio
    async def test_sync_reward_function(self, tokenizer):
        """Test that synchronous reward functions work correctly."""
        config = DictConfig({"reward_model": {"max_concurrent": 10, "timeout": 10.0}})

        RateLimitedRewardManager.init_class(config, tokenizer)
        manager = RateLimitedRewardManager(config=config, tokenizer=tokenizer, compute_score=mock_sync_reward_function)

        data = create_test_data_proto(tokenizer, "answer", "answer")

        result = await manager.run_single(data)

        assert result["reward_score"] == 1.0
        assert manager.is_async_reward_score is False

    @pytest.mark.asyncio
    async def test_combined_rate_limits(self, tokenizer):
        """Test all three rate limiting layers together."""
        config = DictConfig(
            {
                "reward_model": {
                    "max_concurrent": 2,
                    "max_rpm": 120,  # 2 requests per second
                    "max_tpm": 12000,  # 200 tokens per second
                    "estimated_tokens_per_request": 100,  # 0.5 seconds per request
                    "timeout": 10.0,
                }
            }
        )

        RateLimitedRewardManager.init_class(config, tokenizer)
        manager = RateLimitedRewardManager(config=config, tokenizer=tokenizer, compute_score=mock_async_reward_function)

        data = create_test_data_proto(tokenizer, "answer", "answer")

        # Make 6 requests to exceed burst capacity (RPM bucket starts with 2 tokens)
        start_time = time.time()

        tasks = [manager.run_single(data) for _ in range(6)]
        results = await asyncio.gather(*tasks)

        elapsed = time.time() - start_time

        # Bucket starts with 2 RPM tokens and 200 TPM tokens
        # First 2 requests: use burst capacity (2 RPM tokens, 200 TPM tokens)
        # Next 4 requests: need 4 RPM tokens (wait 2 seconds) and 400 TPM tokens (wait 2 seconds)
        # Limiting factor: RPM at 2 seconds
        assert elapsed >= 1.8, f"Combined rate limiting: {elapsed:.3f}s"
        assert all(r["reward_score"] == 1.0 for r in results)
        assert api_counter.call_count == 6

    @pytest.mark.asyncio
    async def test_correct_vs_incorrect_answers(self, tokenizer):
        """Test scoring of correct vs incorrect answers."""
        config = DictConfig({"reward_model": {"max_concurrent": 10, "timeout": 10.0}})

        RateLimitedRewardManager.init_class(config, tokenizer)
        manager = RateLimitedRewardManager(config=config, tokenizer=tokenizer, compute_score=mock_async_reward_function)

        # Test correct answer
        data_correct = create_test_data_proto(tokenizer, "right answer", "right answer")
        result_correct = await manager.run_single(data_correct)

        # Test incorrect answer
        data_incorrect = create_test_data_proto(tokenizer, "wrong answer", "right answer")
        result_incorrect = await manager.run_single(data_incorrect)

        assert result_correct["reward_score"] == 1.0
        assert result_incorrect["reward_score"] == 0.0

    @pytest.mark.asyncio
    async def test_high_throughput(self, tokenizer):
        """Test high throughput with many concurrent requests."""
        config = DictConfig(
            {
                "reward_model": {
                    "max_concurrent": 20,
                    "max_rpm": 6000,  # 100 requests per second
                    "timeout": 10.0,
                }
            }
        )

        RateLimitedRewardManager.init_class(config, tokenizer)
        manager = RateLimitedRewardManager(config=config, tokenizer=tokenizer, compute_score=mock_async_reward_function)

        data = create_test_data_proto(tokenizer, "answer", "answer")

        # Launch 200 concurrent requests (more than burst capacity of 100)
        start_time = time.time()

        tasks = [manager.run_single(data) for _ in range(200)]
        results = await asyncio.gather(*tasks)

        elapsed = time.time() - start_time

        assert len(results) == 200
        assert all(r["reward_score"] == 1.0 for r in results)

        # Bucket starts with 100 tokens (burst capacity)
        # First 100 requests: use burst capacity instantly
        # Next 100 requests: need to wait for refill at 100 tokens/sec = 1 second minimum
        # Total time should be at least 1 second
        assert elapsed >= 0.9, f"Should take at least 0.9s for rate limiting, took {elapsed:.3f}s"

        # Calculate actual rate over the time window
        actual_rate = api_counter.call_count / elapsed

        # Average rate should not significantly exceed 100 req/sec
        # Allow some burst overhead due to initial capacity
        assert actual_rate <= 200, f"Rate limiting failed: {actual_rate:.1f} req/sec (max 200)"

    @pytest.mark.asyncio
    async def test_class_initialization_once(self, tokenizer):
        """Test that class initialization only happens once."""
        config = DictConfig({"reward_model": {"max_concurrent": 5, "timeout": 10.0}})

        # Initialize multiple times
        RateLimitedRewardManager.init_class(config, tokenizer)
        first_semaphore = RateLimitedRewardManager._semaphore

        RateLimitedRewardManager.init_class(config, tokenizer)
        second_semaphore = RateLimitedRewardManager._semaphore

        # Should be the same object
        assert first_semaphore is second_semaphore

    @pytest.mark.asyncio
    async def test_extra_info_handling(self, tokenizer):
        """Test that extra_info is properly passed to reward function."""
        received_extra_info = {}

        async def mock_reward_with_extra_info(
            data_source: str, solution_str: str, ground_truth: str, extra_info: dict, **kwargs
        ):
            received_extra_info.update(extra_info)
            return 1.0

        config = DictConfig({"reward_model": {"max_concurrent": 10, "timeout": 10.0}})

        RateLimitedRewardManager.init_class(config, tokenizer)
        manager = RateLimitedRewardManager(
            config=config, tokenizer=tokenizer, compute_score=mock_reward_with_extra_info
        )

        data = create_test_data_proto(tokenizer, "answer", "answer")
        data.non_tensor_batch["extra_info"] = [{"custom_field": "test_value"}]

        await manager.run_single(data)

        assert "custom_field" in received_extra_info
        assert received_extra_info["custom_field"] == "test_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
