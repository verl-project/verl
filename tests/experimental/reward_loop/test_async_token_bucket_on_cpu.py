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
import time

import pytest

from verl.experimental.reward_loop.reward_manager.limited import AsyncTokenBucket


class TestAsyncTokenBucket:
    """Unit tests for AsyncTokenBucket rate limiter."""

    @pytest.mark.asyncio
    async def test_basic_acquire(self):
        """Test basic token acquisition."""
        bucket = AsyncTokenBucket(rate_limit=10.0, max_tokens=10.0)

        # Should be able to acquire tokens immediately when bucket is full
        start = time.time()
        await bucket.acquire(5.0)
        elapsed = time.time() - start

        assert elapsed < 0.1, "Initial acquire should be immediate"
        assert bucket.tokens == pytest.approx(5.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_refill_mechanism(self):
        """Test that tokens refill over time."""
        bucket = AsyncTokenBucket(rate_limit=10.0, max_tokens=10.0)

        # Consume all tokens
        await bucket.acquire(10.0)
        assert bucket.tokens == pytest.approx(0.0, abs=0.1)

        # Wait for refill (should get ~5 tokens in 0.5 seconds at 10 tokens/sec)
        await asyncio.sleep(0.5)

        # Try to acquire 4 tokens (should succeed without waiting)
        start = time.time()
        await bucket.acquire(4.0)
        elapsed = time.time() - start

        assert elapsed < 0.1, "Acquire should be quick after refill"

    @pytest.mark.asyncio
    async def test_waiting_for_tokens(self):
        """Test that acquire waits when insufficient tokens available."""
        bucket = AsyncTokenBucket(rate_limit=10.0, max_tokens=10.0)

        # Consume all tokens
        await bucket.acquire(10.0)

        # Try to acquire more tokens (should wait ~0.5 seconds for 5 tokens)
        start = time.time()
        await bucket.acquire(5.0)
        elapsed = time.time() - start

        # Should wait approximately 0.5 seconds (5 tokens / 10 tokens per second)
        assert 0.4 < elapsed < 0.7, f"Expected ~0.5s wait, got {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_max_tokens_cap(self):
        """Test that tokens don't exceed max_tokens capacity."""
        bucket = AsyncTokenBucket(rate_limit=10.0, max_tokens=5.0)

        # Wait for potential overflow
        await asyncio.sleep(1.0)

        # Tokens should be capped at max_tokens
        await bucket.acquire(1.0)

        # After 1 second at 10 tokens/sec, should have max_tokens (5.0)
        # After acquiring 1, should have 4.0 remaining
        assert bucket.tokens <= 5.0, "Tokens should not exceed max_tokens"

    @pytest.mark.asyncio
    async def test_fractional_tokens(self):
        """Test acquiring fractional tokens."""
        bucket = AsyncTokenBucket(rate_limit=100.0, max_tokens=100.0)

        # Acquire fractional amounts
        await bucket.acquire(0.5)
        await bucket.acquire(1.5)
        await bucket.acquire(2.3)

        assert bucket.tokens == pytest.approx(100.0 - 0.5 - 1.5 - 2.3, abs=0.1)

    @pytest.mark.asyncio
    async def test_concurrent_acquires(self):
        """Test multiple concurrent acquire operations."""
        bucket = AsyncTokenBucket(rate_limit=10.0, max_tokens=10.0)

        async def acquire_task(num_tokens: float, task_id: int):
            await bucket.acquire(num_tokens)
            return task_id

        # Launch 5 concurrent tasks, each acquiring 3 tokens (15 total)
        # Bucket only has 10, so some will need to wait
        start = time.time()
        tasks = [acquire_task(3.0, i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start

        # Should take at least 0.5 seconds to refill 5 tokens
        # (15 needed - 10 available) / 10 tokens per second = 0.5 seconds
        assert elapsed >= 0.4, f"Expected >=0.4s for concurrent acquires, got {elapsed:.3f}s"
        assert len(results) == 5, "All tasks should complete"

    @pytest.mark.asyncio
    async def test_high_rate_limit(self):
        """Test with high rate limit (simulating high-throughput scenarios)."""
        bucket = AsyncTokenBucket(rate_limit=1000.0, max_tokens=1000.0)

        # Rapidly acquire tokens
        start = time.time()
        for _ in range(100):
            await bucket.acquire(10.0)  # 1000 tokens total
        elapsed = time.time() - start

        # Should complete in approximately 1 second
        assert elapsed < 1.5, f"High rate limit test took too long: {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_zero_initial_state(self):
        """Test that bucket starts with full tokens."""
        bucket = AsyncTokenBucket(rate_limit=10.0, max_tokens=10.0)

        assert bucket.tokens == 10.0, "Bucket should start full"
        assert bucket.last_update is None, "last_update should be None initially"

        # After first acquire, last_update should be set
        await bucket.acquire(1.0)
        assert bucket.last_update is not None, "last_update should be set after acquire"

    @pytest.mark.asyncio
    async def test_rate_limit_accuracy(self):
        """Test rate limit accuracy over time."""
        rate = 50.0  # 50 tokens per second
        bucket = AsyncTokenBucket(rate_limit=rate, max_tokens=rate)

        # Consume all tokens and measure refill time for 25 tokens
        await bucket.acquire(50.0)

        start = time.time()
        await bucket.acquire(25.0)
        elapsed = time.time() - start

        expected_time = 25.0 / rate  # 0.5 seconds
        # Allow 20% margin for timing inaccuracy
        assert abs(elapsed - expected_time) < expected_time * 0.2, f"Expected ~{expected_time:.3f}s, got {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_sequential_acquires(self):
        """Test sequential acquire operations."""
        bucket = AsyncTokenBucket(rate_limit=20.0, max_tokens=20.0)

        # Sequential acquires without waiting
        await bucket.acquire(5.0)
        await bucket.acquire(5.0)
        await bucket.acquire(5.0)
        await bucket.acquire(5.0)

        # Bucket should be empty
        assert bucket.tokens == pytest.approx(0.0, abs=0.1)

        # Next acquire should wait
        start = time.time()
        await bucket.acquire(10.0)
        elapsed = time.time() - start

        assert elapsed >= 0.4, "Should wait for token refill"

    @pytest.mark.asyncio
    async def test_default_max_tokens(self):
        """Test that max_tokens defaults to rate_limit."""
        bucket = AsyncTokenBucket(rate_limit=15.0)

        assert bucket.max_tokens == 15.0, "max_tokens should default to rate_limit"
        assert bucket.tokens == 15.0, "Initial tokens should equal max_tokens"

    @pytest.mark.asyncio
    async def test_single_token_acquire(self):
        """Test default acquire of 1 token."""
        bucket = AsyncTokenBucket(rate_limit=10.0, max_tokens=10.0)

        await bucket.acquire()  # Default num_tokens=1.0

        assert bucket.tokens == pytest.approx(9.0, abs=0.1)

    @pytest.mark.asyncio
    async def test_large_token_acquire(self):
        """Test acquiring more tokens than bucket capacity."""
        bucket = AsyncTokenBucket(rate_limit=10.0, max_tokens=10.0)

        # Try to acquire 50 tokens (5x capacity)
        start = time.time()
        await bucket.acquire(50.0)
        elapsed = time.time() - start

        # Should wait for: (50 - 10) / 10 = 4 seconds
        assert 3.5 < elapsed < 5.0, f"Expected ~4s wait for large acquire, got {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_thread_safety_with_lock(self):
        """Test that lock prevents race conditions."""
        bucket = AsyncTokenBucket(rate_limit=100.0, max_tokens=100.0)
        results = []

        async def acquire_and_record():
            await bucket.acquire(10.0)
            results.append(1)

        # Launch many concurrent tasks
        tasks = [acquire_and_record() for _ in range(10)]
        await asyncio.gather(*tasks)

        # All tasks should complete
        assert len(results) == 10, "All tasks should complete successfully"

        # Bucket should have consumed exactly 100 tokens
        assert bucket.tokens == pytest.approx(0.0, abs=0.5)

    @pytest.mark.asyncio
    async def test_multiple_wait_cycles(self):
        """Test multiple wait cycles in the acquire loop."""
        bucket = AsyncTokenBucket(rate_limit=10.0, max_tokens=10.0)

        # Consume all tokens
        await bucket.acquire(10.0)

        # Acquire tokens that require multiple refill cycles
        start = time.time()
        await bucket.acquire(15.0)
        elapsed = time.time() - start

        # Should wait for 15 tokens / 10 tokens per second = 1.5 seconds
        assert 1.3 < elapsed < 1.8, f"Expected ~1.5s for multiple refill cycles, got {elapsed:.3f}s"

    @pytest.mark.asyncio
    async def test_rapid_small_acquires(self):
        """Test many rapid small acquisitions."""
        bucket = AsyncTokenBucket(rate_limit=100.0, max_tokens=100.0)

        start = time.time()
        for _ in range(50):
            await bucket.acquire(2.0)  # 100 tokens total
        elapsed = time.time() - start

        # Should complete quickly since we're within capacity
        assert elapsed < 0.5, f"Rapid small acquires took too long: {elapsed:.3f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
