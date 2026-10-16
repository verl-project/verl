# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""CPU tests for the math_verify process-pool recovery in compute_score.

A ``ProcessPoolExecutor`` singleton is shared by every ``compute_score`` call.
These tests pin the recovery behavior requested in issue #8011: a dead worker
must not silently zero out all later scores, and a timed-out task must not
leave its worker stuck in the pool forever.
"""

import os
import signal
import threading
import time
from concurrent.futures.process import BrokenProcessPool

import pytest

from verl.utils.reward_score import math_verify
from verl.utils.reward_score.math_verify import _get_pool, compute_score


def _math_verify_available():
    """Check whether the real math-verify package can be imported."""
    try:
        import math_verify.grader  # noqa: F401
        import math_verify.parser  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _math_verify_available(), reason="math-verify is not installed")


def _return_one():
    """Trivial picklable task used to spawn/warm worker processes."""
    return 1


def _sleep_forever():
    """Picklable task that never returns, standing in for a stuck verify()."""
    time.sleep(1000)


@pytest.fixture
def fresh_pool():
    """Reset the module-level pool before and after each test."""
    math_verify._pool = None
    yield
    pool = math_verify._pool
    math_verify._pool = None
    if pool is not None:
        for proc in pool._processes.values():
            proc.kill()
        pool.shutdown(wait=False)


def _warm_pool(pool, num_workers=4):
    """Force the pool to actually spawn its workers and wait for them."""
    futures = [pool.submit(_return_one) for _ in range(num_workers)]
    for future in futures:
        assert future.result(timeout=60) == 1


def _kill_one_worker(pool):
    """Kill one worker process the way the OOM killer would (SIGKILL)."""
    pids = list(pool._processes.keys())
    assert pids, "pool has no workers yet"
    os.kill(pids[0], signal.SIGKILL)


class TestComputeScore:
    def test_correct_answer_scores_one(self, fresh_pool):
        assert compute_score("The answer is \\boxed{42}", "42") == 1.0

    def test_wrong_answer_scores_zero(self, fresh_pool):
        assert compute_score("The answer is \\boxed{41}", "42") == 0.0

    def test_equivalent_answer_scores_one(self, fresh_pool):
        assert compute_score("The answer is \\boxed{\\frac{1}{2}}", "0.5") == 1.0


class TestBrokenPoolRecovery:
    def test_scores_recover_after_worker_is_killed(self, fresh_pool):
        pool = _get_pool()
        _warm_pool(pool)
        _kill_one_worker(pool)
        # Every later call must recover instead of scoring 0.0 forever (#8011).
        assert compute_score("\\boxed{42}", "42") == 1.0
        assert compute_score("\\boxed{42}", "42") == 1.0
        assert math_verify._pool is not pool

    def test_second_broken_pool_propagates(self, fresh_pool):
        """A pool that breaks again right after the retry must raise, not return 0.0."""
        real_get_pool = math_verify._get_pool

        def _breaking_get_pool():
            p = real_get_pool()
            _warm_pool(p)
            _kill_one_worker(p)
            # Give the executor's management thread time to flag the pool as
            # broken, so the caller's submit() deterministically fails.
            time.sleep(0.5)
            return p

        math_verify._get_pool = _breaking_get_pool
        try:
            with pytest.raises(BrokenProcessPool):
                compute_score("\\boxed{42}", "42")
        finally:
            math_verify._get_pool = real_get_pool


class TestTimeoutKillsStuckWorkers:
    def test_timed_out_task_does_not_occupy_workers(self, fresh_pool):
        """Fill all workers with non-returning tasks; the next call must not wait for them."""
        pool = _get_pool()
        _warm_pool(pool)
        stuck = [pool.submit(_sleep_forever) for _ in range(4)]
        time.sleep(0.5)  # let the workers pick up the stuck tasks

        t0 = time.monotonic()
        score = compute_score("\\boxed{42}", "42", timeout_score=0.123, timeout=1.0)
        elapsed = time.monotonic() - t0

        assert score == 0.123
        assert elapsed < 30  # did not block behind the stuck workers
        # The stuck workers were killed and the pool was replaced.
        assert math_verify._pool is not pool
        deadline = time.monotonic() + 10
        while any(proc.is_alive() for proc in pool._processes.values()) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not any(proc.is_alive() for proc in pool._processes.values())
        for future in stuck:
            future.cancel()

        # The next call runs on a fresh pool and scores correctly.
        assert compute_score("\\boxed{42}", "42", timeout=30.0) == 1.0

    def test_concurrent_healthy_call_survives_a_timeout(self, fresh_pool):
        """Killing the pool on one timeout must not break another caller's retry.

        Two threads share the pool: one submits a task that times out (the pool
        is killed under it, breaking every in-flight future on that pool),
        while the other thread's healthy call is in flight on the same pool.
        The healthy caller must still get a correct score via the retry.
        """
        pool = _get_pool()
        _warm_pool(pool)
        # Occupy all workers so both callers' tasks cannot complete in time.
        stuck = [pool.submit(_sleep_forever) for _ in range(4)]
        time.sleep(0.5)  # let the workers pick up the stuck tasks

        results = {}
        start = threading.Barrier(2)

        def _stuck_caller():
            start.wait(timeout=10)
            results["stuck"] = compute_score("\\boxed{42}", "42", timeout_score=0.5, timeout=1.0)

        def _healthy_caller():
            start.wait(timeout=10)
            # This call shares the pool; when the stuck caller kills it, the
            # in-flight verify breaks with BrokenProcessPool and this thread
            # retries on the fresh pool.
            results["healthy"] = compute_score("\\boxed{42}", "42", timeout=30.0)

        stuck_thread = threading.Thread(target=_stuck_caller)
        healthy_thread = threading.Thread(target=_healthy_caller)
        stuck_thread.start()
        healthy_thread.start()
        stuck_thread.join(timeout=120)
        healthy_thread.join(timeout=120)

        for future in stuck:
            future.cancel()
        assert results["stuck"] == 0.5
        assert results["healthy"] == 1.0
