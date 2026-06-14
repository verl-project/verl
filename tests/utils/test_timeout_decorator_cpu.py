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
import sys
import threading
import time

import pytest  # Import pytest

from verl.utils.py_functional import timeout_limit as timeout

# --- Test Task Functions ---
TEST_TIMEOUT_SECONDS = 1.5  # Timeout duration for tests
LONG_TASK_DURATION = TEST_TIMEOUT_SECONDS + 0.5  # Duration slightly longer than timeout


@timeout(seconds=TEST_TIMEOUT_SECONDS)  # Keep global decorator for mp tests
def quick_task(x):
    """A task that completes quickly."""
    time.sleep(0.1)
    return "quick_ok"


@timeout(seconds=TEST_TIMEOUT_SECONDS)  # Keep global decorator for mp tests
def slow_task(x):
    """A task that takes longer than the timeout."""
    time.sleep(LONG_TASK_DURATION)
    return "slow_finished"  # This return value indicates it didn't time out


# REMOVE global decorator here
def task_raises_value_error():  # Now truly not globally decorated
    """A task that intentionally raises a ValueError."""
    raise ValueError("Specific value error from task")


# --- Top-level function for signal test in subprocess ---
# Keep this decorated globally for the specific subprocess test case
@timeout(seconds=TEST_TIMEOUT_SECONDS, use_signals=True)
def top_level_decorated_quick_task_signal():
    """A pickleable top-level function decorated with signal timeout."""
    # Assuming this calls the logic of quick_task directly for the test purpose
    time.sleep(0.1)
    return "quick_ok_signal_subprocess"  # Different return for clarity if needed


# --- Top-level function for signal test in subprocess ---
# Keep this decorated globally for the specific subprocess test case
@timeout(seconds=TEST_TIMEOUT_SECONDS, use_signals=True)
def top_level_decorated_slow_task_signal():
    """A pickleable top-level function decorated with signal timeout."""
    time.sleep(LONG_TASK_DURATION)
    return "slow_finished"


# --- NEW: Top-level helper function to run target in process ---
def run_target_and_put_in_queue(target_func, q):
    """
    Top-level helper function to run a target function and put its result or exception into a queue.
    This function is pickleable and can be used as the target for multiprocessing.Process.
    """
    try:
        result = target_func()
        q.put(("success", result))
    except Exception as e:
        q.put(("error", e))


# Use a module-level fixture to set the start method on macOS
@pytest.fixture(scope="module", autouse=True)  # Changed scope to module
def set_macos_start_method():
    if sys.platform == "darwin":
        # Force fork method on macOS to avoid pickling issues with globally decorated functions
        # when running tests via pytest discovery.
        current_method = multiprocessing.get_start_method(allow_none=True)
        # Only set if not already set or if set to something else (less likely in test run)
        if current_method is None or current_method != "fork":
            try:
                multiprocessing.set_start_method("fork", force=True)
            except RuntimeError:
                # Might fail if context is already started, ignore in that case.
                pass


def test_quick_task():  # Renamed from test_multiprocessing_quick_task
    """Tests timeout handles a quick task correctly."""
    # Call the globally decorated function directly
    result = quick_task(1)
    assert result == "quick_ok"  # Use pytest assert


def test_slow_task_timeout():  # Renamed from test_multiprocessing_slow_task_timeout
    """Tests timeout correctly raises TimeoutError for a slow task."""
    # Call the globally decorated function directly within pytest.raises
    with pytest.raises(TimeoutError) as excinfo:  # Use pytest.raises
        slow_task(1)
    # Check the error message from the multiprocessing implementation
    assert f"timed out after {TEST_TIMEOUT_SECONDS} seconds" in str(excinfo.value)  # Use pytest assert


def test_internal_exception():  # Renamed from test_multiprocessing_internal_exception
    """Tests timeout correctly propagates internal exceptions."""
    # Apply the default timeout decorator dynamically to the undecorated function
    decorated_task = timeout(seconds=TEST_TIMEOUT_SECONDS)(task_raises_value_error)  # Apply decorator dynamically
    with pytest.raises(ValueError) as excinfo:  # Use pytest.raises
        decorated_task()  # Call the dynamically decorated function
    assert str(excinfo.value) == "Specific value error from task"  # Use pytest assert


# --- Test the signal implementation (use_signals=True) ---
# Note: As per py_functional.py, use_signals=True currently falls back to
# multiprocessing on POSIX. These tests verify that behavior.


def test_signal_quick_task_main_process():  # Removed self
    """Tests signal timeout handles a quick task correctly in the main process."""

    # Apply the signal decorator dynamically
    def plain_quick_task_logic():
        time.sleep(0.1)
        return "quick_ok_signal"

    decorated_task = timeout(seconds=TEST_TIMEOUT_SECONDS, use_signals=True)(plain_quick_task_logic)
    assert decorated_task() == "quick_ok_signal"  # Use pytest assert


def test_signal_slow_task_main_process_timeout():  # Removed self
    """Tests signal timeout correctly raises TimeoutError for a slow task in the main process."""

    # Apply the signal decorator dynamically
    def plain_slow_task_logic():
        time.sleep(LONG_TASK_DURATION)
        return "slow_finished_signal"

    decorated_task = timeout(seconds=TEST_TIMEOUT_SECONDS, use_signals=True)(plain_slow_task_logic)
    with pytest.raises(TimeoutError) as excinfo:  # Use pytest.raises
        decorated_task()
    # Check the error message (falls back to multiprocessing message on POSIX)
    assert f"timed out after {TEST_TIMEOUT_SECONDS} seconds" in str(excinfo.value)  # Use pytest assert


@pytest.mark.skip(reason="this test won't pass. Just to show why use_signals should not be used")
def test_signal_in_thread_does_not_timeout():
    """
    Tests that signal-based timeout does NOT work reliably in a child thread.
    The TimeoutError from the signal handler is not expected to be raised.
    """
    result_container = []  # Use a list to store result from thread
    exception_container = []  # Use a list to store exception from thread

    @timeout(seconds=TEST_TIMEOUT_SECONDS, use_signals=True)
    def slow_task_in_thread():
        try:
            print("Thread: Starting slow task...")
            time.sleep(LONG_TASK_DURATION)
            print("Thread: Slow task finished.")
            return "slow_finished_in_thread"
        except Exception as e:
            # Catch any exception within the thread's target function
            print(f"Thread: Caught exception: {e}")
            exception_container.append(e)
            return None  # Indicate failure

    def thread_target():
        try:
            # Run the decorated function inside the thread
            res = slow_task_in_thread()
            if res is not None:
                result_container.append(res)
        except Exception as e:
            # This might catch exceptions happening *outside* the decorated function
            # but still within the thread target, though less likely here.
            print(f"Thread Target: Caught exception: {e}")
            exception_container.append(e)

    thread = threading.Thread(target=thread_target)
    print("Main: Starting thread...")
    thread.start()
    # Wait longer than the timeout + task duration to ensure the thread finishes
    # regardless of whether timeout worked or not.
    thread.join(timeout=LONG_TASK_DURATION + 1)

    assert len(exception_container) == 1
    assert isinstance(exception_container[0], TimeoutError)
    assert not result_container


def test_in_thread_timeout():
    result_container = []  # Use a list to store result from thread
    exception_container = []  # Use a list to store exception from thread

    @timeout(seconds=TEST_TIMEOUT_SECONDS, use_signals=False)
    def slow_task_in_thread():
        try:
            print("Thread: Starting slow task...")
            time.sleep(LONG_TASK_DURATION)
            print("Thread: Slow task finished.")
            return "slow_finished_in_thread"
        except Exception as e:
            # Catch any exception within the thread's target function
            print(f"Thread: Caught exception: {e}")
            exception_container.append(e)
            return None  # Indicate failure

    def thread_target():
        try:
            # Run the decorated function inside the thread
            res = slow_task_in_thread()
            if res is not None:
                result_container.append(res)
        except Exception as e:
            # This might catch exceptions happening *outside* the decorated function
            # but still within the thread target, though less likely here.
            print(f"Thread Target: Caught exception: {e}")
            exception_container.append(e)

    thread = threading.Thread(target=thread_target)
    print("Main: Starting thread...")
    thread.start()
    # Wait longer than the timeout + task duration to ensure the thread finishes
    # regardless of whether timeout worked or not.
    thread.join(timeout=LONG_TASK_DURATION + 1)

    assert len(exception_container) == 1
    assert isinstance(exception_container[0], TimeoutError)
    assert not result_container
