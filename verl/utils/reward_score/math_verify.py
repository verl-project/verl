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

import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from concurrent.futures.process import BrokenProcessPool

try:
    from math_verify.errors import TimeoutException
except ImportError:

    class TimeoutException(Exception):
        pass

    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn"))
    return _pool


def _reset_pool(pool):
    """Drop the given pool (identity-checked) so the next call builds a fresh one.

    Workers that survive are terminated first: a timed-out task keeps its worker
    busy forever because ``future.result(timeout=...)`` does not cancel it. The
    identity check keeps concurrent callers from racing to build two fresh pools,
    and stops a slow caller from killing a pool that has already been replaced.
    """
    global _pool
    with _pool_lock:
        if _pool is pool:
            _pool = None
        else:
            return
    # Snapshot: the executor's management thread may mutate this dict while we kill.
    for proc in list(pool._processes.values()):
        proc.kill()


def _verify_in_subprocess(ground_truth_boxed: str, model_output: str) -> float:
    """Run math_verify in a subprocess where signal.alarm() works."""
    from math_verify.grader import verify
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

    gold_targets = (LatexExtractionConfig(),)
    pred_targets = (ExprExtractionConfig(), LatexExtractionConfig())

    extracted_gold = parse(ground_truth_boxed, gold_targets)
    extracted_pred = parse(model_output, pred_targets)
    if extracted_gold and extracted_pred:
        return max(1.0 if any(verify(g, p) for g in extracted_gold) else 0.0 for p in extracted_pred)
    return 0.0


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0, timeout: float = 30.0) -> float:
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    for attempt in range(2):
        pool = _get_pool()
        future = None
        try:
            future = pool.submit(_verify_in_subprocess, ground_truth_boxed, model_output)
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            # The worker keeps running the task past the timeout, so it must be
            # killed before the pool is reused, otherwise the pool slowly fills
            # with stuck workers and every later call blocks for the full timeout.
            if future is not None:
                future.cancel()
            _reset_pool(pool)
            return timeout_score
        except TimeoutException:
            # math_verify's own alarm fired inside the worker: the worker finished
            # promptly and the pool is still healthy, so only the score degrades.
            return timeout_score
        except BrokenProcessPool:
            # A worker died (e.g. OOM-killed); the pool is unusable. Retry once on
            # a fresh pool, then propagate so the reward manager can flag the row.
            if attempt == 0:
                _reset_pool(pool)
            else:
                raise
    raise AssertionError("unreachable")
