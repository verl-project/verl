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
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import pytest

# Import the function to be tested
from verl.utils.reward_score.sandbox_fusion.utils import check_correctness

# Get SANDBOX_URL from environment variable
SANDBOX_URL = os.environ.get("SANDBOX_FUSION_URL")
# Define skip condition and reason
skip_reason = "SANDBOX_FUSION_URL environment variable not set"
skip_condition = not SANDBOX_URL

# --- Test code (for real API calls) ---
CODE_SUCCESS = """
import sys
data = sys.stdin.read()
if data == 'input1':
    print('output1\\n', end='')
elif data == 'input2':
    print('output2\\n', end='')
else:
    print('unexpected input', end='')
"""

CODE_WRONG_OUTPUT = """
print('wrong_output\\n', end='')
"""

CODE_COMPILE_ERROR = """
a=b
"""

CODE_RUNTIME_ERROR = """
import sys
print("About to raise error", file=sys.stderr)
raise ValueError("This is a runtime error")
"""

CODE_TIMEOUT = """
import time
import sys
print("Sleeping...", file=sys.stderr)
time.sleep(10) # Sleep time should be longer than the timeout set in the test
print("Finished sleeping", file=sys.stderr)
"""

# --- Test input/output data ---
INPUT_OUTPUT_VALID = {"inputs": ["input1", "input2"], "outputs": ["output1\n", "output2\n"]}

INPUT_OUTPUT_SINGLE = {"inputs": ["input1"], "outputs": ["output1\n"]}

INPUT_OUTPUT_MISMATCH = {"inputs": ["input1"], "outputs": ["output1\n", "output2\n"]}

INPUT_OUTPUT_INVALID_MISSING_KEY = {"inputs": ["input1"]}

# --- Integration test cases (calling real API) ---


@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_success_correct():
    """Integration test: Code is correct, output is correct"""
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_VALID, CODE_SUCCESS)
    assert results == [True, True]
    assert metadata_list[0]["status"] == "success"
    assert metadata_list[0]["stdout"] == "output1\n"
    assert metadata_list[1]["status"] == "success"
    assert metadata_list[1]["stdout"] == "output2\n"


@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_success_wrong_output():
    """Integration test: Code runs successfully, but output is wrong"""
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_VALID, CODE_WRONG_OUTPUT)
    assert results == [False, False]
    assert metadata_list[0]["status"] == "wrong_answer"
    assert metadata_list[0]["stdout"] == "wrong_output\n"
    assert metadata_list[1]["status"] == "wrong_answer"


@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_compile_error():
    """Integration test: Code causes compile error"""
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_VALID, CODE_COMPILE_ERROR, language="cpp")
    assert results == [-4, -4]
    assert metadata_list[0]["status"] == "compile_error"
    assert metadata_list[1]["status"] == "compile_error"


@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_runtime_error():
    """Integration test: Code causes runtime error"""
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_SINGLE, CODE_RUNTIME_ERROR)
    assert results == [-2]
    assert metadata_list[0]["status"] == "runtime_error"
    # More assertions can be added based on the actual API response, e.g., exit_code, stderr


@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_runtime_timeout():
    """Integration test: Code causes runtime timeout"""
    test_timeout = 5  # Set a timeout shorter than the sleep time in CODE_TIMEOUT
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_SINGLE, CODE_TIMEOUT, timeout=test_timeout)
    assert results == [-3]
    assert metadata_list[0]["status"] == "timeout"
    # More assertions can be added based on the actual API response, e.g., run_status


@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_concurrency_high_load():
    """Integration test: High concurrency (100 cases) against real API with mixed results (success, wrong
    answer, timeout)"""
    concurrency_level = 100
    # Indices for different expected outcomes
    wrong_answer_indices = {10, 25, 50}
    timeout_indices = {5, 30, 60, 90}  # Indices where we expect a timeout

    # Generate 100 input/output pairs and code
    high_load_inputs = []
    high_load_outputs = []
    expected_results_map = {}  # Store expected result for each index

    for i in range(concurrency_level):
        if i in timeout_indices:
            # Use a special input to trigger timeout in the code
            high_load_inputs.append(f"input_timeout_{i}")
            # Output doesn't matter for timeout, but keep it consistent
            high_load_outputs.append(f"output_{i}\n")
            expected_results_map[i] = -3  # Expect timeout
        elif i in wrong_answer_indices:
            high_load_inputs.append(f"input_{i}")
            # Intentionally set wrong expected output
            high_load_outputs.append(f"wrong_output_{i}\n")
            expected_results_map[i] = False  # Expect wrong answer
        else:
            high_load_inputs.append(f"input_{i}")
            # Correct expected output
            high_load_outputs.append(f"output_{i}\n")
            expected_results_map[i] = True  # Expect success

    high_load_in_outs = {"inputs": high_load_inputs, "outputs": high_load_outputs}

    # Code that handles normal inputs, and sleeps on specific "timeout" inputs
    code_mixed_concurrent = """
import sys
import time
data = sys.stdin.read()
if data.startswith('input_timeout_'):
    time.sleep(20) # Sleep longer than the test timeout
    print(f"output_{data.split('_')[-1]}\\n", end='') # Still print something in case it finishes early
elif data.startswith('input_'):
    print(f"output_{data.split('_')[-1]}\\n", end='')
else:
    print("unknown_input\\n", end='')
"""
    # Set a reasonable timeout per case (must be less than the sleep time in the code)
    test_timeout = 15  # Allow slightly more time due to potential API load, but less than 20s sleep

    start_time = time.time()
    results, metadata_list = check_correctness(
        SANDBOX_URL,
        high_load_in_outs,
        code_mixed_concurrent,  # Use the new code
        timeout=test_timeout,
    )
    end_time = time.time()
    duration = end_time - start_time
    print(
        f"\nHigh concurrency test ({concurrency_level} cases with {len(wrong_answer_indices)} wrong answers, "
        f"{len(timeout_indices)} timeouts) duration: {duration:.2f} seconds"
    )

    # Verify results against the expected map
    assert len(results) == concurrency_level, f"Expected {concurrency_level} results, got {len(results)}"

    correct_count = 0
    wrong_count = 0
    timeout_count = 0
    unexpected_results = []
    for i, r in enumerate(results):
        expected = expected_results_map[i]
        if r == expected:
            if expected is True:
                correct_count += 1
            elif expected is False:
                wrong_count += 1
            elif expected == -3:
                timeout_count += 1
        else:
            unexpected_results.append((i, r, f"Expected {expected}"))

    print(
        f"Correct results (True): {correct_count}/"
        f"{concurrency_level - len(wrong_answer_indices) - len(timeout_indices)}"
    )
    print(f"Expected wrong answers (False, correctly identified): {wrong_count}/{len(wrong_answer_indices)}")
    print(f"Expected timeouts (-3, correctly identified): {timeout_count}/{len(timeout_indices)}")

    if unexpected_results:
        print("Unexpected results found:")
        for idx, res, expected_str in unexpected_results[:10]:  # Print first 10 unexpected
            print(f"  Index {idx}: Got {res}, {expected_str}. Metadata: {metadata_list[idx]}")
        raise AssertionError(f"Found {len(unexpected_results)} unexpected results.")

    assert correct_count == concurrency_level - len(wrong_answer_indices) - len(timeout_indices), (
        "Incorrect number of successful results"
    )
    assert wrong_count == len(wrong_answer_indices), "Incorrect number of identified wrong answers"
    assert timeout_count == len(timeout_indices), "Incorrect number of identified timeouts"

    # Verify metadata count and basic status of one of each type
    assert len(metadata_list) == concurrency_level
    # Find the first correct index
    first_correct_index = next(
        i for i in range(concurrency_level) if i not in wrong_answer_indices and i not in timeout_indices
    )
    assert metadata_list[first_correct_index]["status"] == "success"
    assert metadata_list[first_correct_index]["stdout"] == f"output_{first_correct_index}\n"

    # Check the status of the first intentionally wrong case
    first_wrong_index = min(wrong_answer_indices)
    assert metadata_list[first_wrong_index]["status"] == "wrong_answer"
    assert metadata_list[first_wrong_index]["stdout"] == f"output_{first_wrong_index}\n"
    assert metadata_list[first_wrong_index]["expected_output"] == f"wrong_output_{first_wrong_index}\n"

    # Check the status of the first intentionally timeout case
    first_timeout_index = min(timeout_indices)
    assert metadata_list[first_timeout_index]["status"] == "timeout"
    # For timeout, stdout might be None or empty depending on when the timeout occurred
    # assert metadata_list[first_timeout_index]["stdout"] is None or metadata_list[first_timeout_index]["stdout"] == ""


# --- Unit test cases (using mock) ---


@patch("verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api")
def test_unit_concurrency_order(mock_call_sandbox_api):
    sandbox_url = "mock_url"
    generation = "print(input())"
    language = "python"
    timeout = 5
    in_outs = {"inputs": ["input1", "input2", "input3"], "outputs": ["output1", "output2", "output3"]}

    def side_effect(*args, **kwargs):
        stdin = kwargs.get("stdin")
        if stdin == "input1":
            return (
                {"status": "Success", "run_result": {"status": "Finished", "stdout": "output1", "return_code": 0}},
                None,
            )
        elif stdin == "input2":
            time.sleep(0.1)
            return (
                {"status": "Success", "run_result": {"status": "Finished", "stdout": "output2", "return_code": 0}},
                None,
            )
        elif stdin == "input3":
            return (
                {"status": "Success", "run_result": {"status": "Finished", "stdout": "output3", "return_code": 0}},
                None,
            )
        else:
            return (None, "Unknown input in mock")

    mock_call_sandbox_api.side_effect = side_effect

    results, metadata_list = check_correctness(sandbox_url, in_outs, generation, timeout, language)

    assert results == [True, True, True]
    assert len(metadata_list) == 3
    assert metadata_list[0]["case_index"] == 0
    assert metadata_list[0]["status"] == "success"
    assert metadata_list[1]["case_index"] == 1
    assert metadata_list[1]["status"] == "success"
    assert metadata_list[2]["case_index"] == 2
    assert metadata_list[2]["status"] == "success"
    assert mock_call_sandbox_api.call_count == 3


@patch("verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api")
def test_unit_api_timeout_error_concurrent(mock_call_sandbox_api):
    sandbox_url = "mock_url"
    generation = "print(input())"
    language = "python"
    timeout = 5
    in_outs = {"inputs": ["input1", "input2_timeout", "input3"], "outputs": ["output1", "output2", "output3"]}

    api_error_message = "API Call Failed: Gateway Timeout (504) on attempt 3/3"

    def side_effect(*args, **kwargs):
        stdin = kwargs.get("stdin")
        if stdin == "input1":
            return (
                {"status": "Success", "run_result": {"status": "Finished", "stdout": "output1", "return_code": 0}},
                None,
            )
        elif stdin == "input2_timeout":
            return (None, api_error_message)
        elif stdin == "input3":
            return (
                {"status": "Success", "run_result": {"status": "Finished", "stdout": "output3", "return_code": 0}},
                None,
            )
        else:
            return (None, "Unknown input in mock")

    mock_call_sandbox_api.side_effect = side_effect

    results, metadata_list = check_correctness(sandbox_url, in_outs, generation, timeout, language)

    assert results == [True, -1, True]
    assert len(metadata_list) == 3
    assert metadata_list[0]["status"] == "success"
    assert metadata_list[1]["status"] == "api_error"
    assert metadata_list[1]["api_request_error"] == api_error_message
    assert metadata_list[2]["status"] == "success"
    assert mock_call_sandbox_api.call_count == 3


# --- Constants for the new concurrency test ---
# Define a low global concurrency limit to test the semaphore's effect
MAX_GLOBAL_CONCURRENCY_LIMIT_TEST = 5
# Define the number of processes used in the test
NUM_PROCESSES_TEST = 4
# Define the number of tasks processed by check_correctness in each process (i.e., internal
# ThreadPoolExecutor's concurrency potential)
NUM_TASKS_PER_PROCESS_TEST = 3
# Simulate API call duration to ensure calls can overlap
SIMULATED_API_CALL_DURATION_TEST = 0.2  # seconds


# --- Mock API call function for concurrency tracking ---
# This function will replace the real call_sandbox_api and use shared variables to track concurrency
def _mock_api_call_for_concurrency_tracking(
    active_calls_counter,  # multiprocessing.Value
    max_calls_tracker,  # multiprocessing.Value
    call_lock,  # multiprocessing.Lock
    # Standard call_sandbox_api parameters
    sandbox_fusion_url,
    code,
    stdin,
    compile_timeout,
    run_timeout,
    memory_limit_mb,
    language,
):
    # entry_time = time.time() # For detailed logging
    with call_lock:
        active_calls_counter.value += 1
        if active_calls_counter.value > max_calls_tracker.value:
            max_calls_tracker.value = active_calls_counter.value
        # Optional debug log:
        # print(f"[PID:{os.getpid()}-TID:{threading.get_ident()}] API Call Start. Active: "
        #       f"{active_calls_counter.value}, Max Observed: {max_calls_tracker.value}, Input: {stdin}")

    time.sleep(SIMULATED_API_CALL_DURATION_TEST)  # Simulate actual work duration

    # exit_time = time.time() # For detailed logging
    with call_lock:
        active_calls_counter.value -= 1
        # Optional debug log:
        # print(f"[PID:{os.getpid()}-TID:{threading.get_ident()}] API Call End. Active: "
        #       f"{active_calls_counter.value}, Input: {stdin}, Duration: {exit_time - entry_time:.2f}s")

    # Return a simulated successful API response
    return {
        "status": "Success",
        "run_result": {"status": "Finished", "stdout": f"mock_output_for_{stdin}", "return_code": 0},
    }, None


# --- Worker function for ProcessPoolExecutor ---
# This function runs in each child process of ProcessPoolExecutor
def _process_pool_worker_for_concurrency_test(
    sandbox_url,
    in_outs,
    generation,
    memory_limit_mb,
    language,
    timeout,
    mp_semaphore_for_check_correctness,
    active_calls_counter,
    max_calls_tracker,
    call_lock,
):
    # Corrected lambda to accept keyword arguments matching call_sandbox_api's usage
    curried_mock_api_call = (
        lambda sandbox_fusion_url, code, stdin, compile_timeout, run_timeout, memory_limit_mb, language: (
            _mock_api_call_for_concurrency_tracking(
                active_calls_counter,
                max_calls_tracker,
                call_lock,
                sandbox_fusion_url,
                code,
                stdin,
                compile_timeout,
                run_timeout,
                memory_limit_mb,
                language,
            )
        )
    )

    # ---- START DEBUG PRINTS ----
    import os

    import verl.utils.reward_score.sandbox_fusion.utils

    print(
        f"[Worker PID:{os.getpid()}] Original call_sandbox_api: "
        f"{verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api}",
        flush=True,
    )
    # ---- END DEBUG PRINTS ----

    with patch(
        "verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api", side_effect=curried_mock_api_call
    ) as mock_obj:
        # ---- START DEBUG PRINTS ----
        print(
            f"[Worker PID:{os.getpid()}] Patched call_sandbox_api: "
            f"{verl.utils.reward_score.sandbox_fusion.utils.call_sandbox_api}",
            flush=True,
        )
        print(f"[Worker PID:{os.getpid()}] Mock object: {mock_obj}", flush=True)
        # ---- END DEBUG PRINTS ----
        results, metadata_list = check_correctness(
            sandbox_fusion_url=sandbox_url,
            in_outs=in_outs,
            generation=generation,
            timeout=timeout,
            memory_limit_mb=memory_limit_mb,
            language=language,
            concurrent_semaphore=mp_semaphore_for_check_correctness,  # Pass multiprocessing.Semaphore
        )
        # print(f"Process {os.getpid()} finished check_correctness. Processed {len(results)} tasks.")
    return len(results)  # Return the number of processed tasks for basic validation


# --- The actual test case for multiprocess concurrency control ---
def test_multiprocess_global_concurrency_limit_with_semaphore():
    """
    Tests that the global concurrent_semaphore (multiprocessing.Semaphore)
    correctly limits the number of concurrent calls to call_sandbox_api
    across multiple processes, each potentially running multiple threads
    via check_correctness's internal ThreadPoolExecutor.
    """
    manager = multiprocessing.Manager()
    active_calls_counter = manager.Value("i", 0)  # Current active mock API calls
    max_calls_tracker = manager.Value("i", 0)  # Observed maximum concurrent mock API calls
    call_lock = manager.Lock()  # Lock to protect counters

    # Create a multiprocessing.Semaphore instance, this is the global semaphore we are testing.
    # It will be passed to check_correctness and used by _process_single_case to limit calls to call_sandbox_api.
    global_mp_semaphore = manager.Semaphore(MAX_GLOBAL_CONCURRENCY_LIMIT_TEST)

    mock_sandbox_url = "mock_url_for_concurrency_test"
    mock_generation = "pass"  # Specific code content is not important as API call is mocked
    mock_memory_limit_mb = 1024
    mock_language = "python"
    mock_timeout = 5  # Timeout setting, not critical for mock calls

    # Input/output data for each process
    # NUM_TASKS_PER_PROCESS_TEST tasks will be handled by check_correctness's internal ThreadPoolExecutor
    process_in_outs = {
        "inputs": [f"task_input_{i}" for i in range(NUM_TASKS_PER_PROCESS_TEST)],
        "outputs": [f"task_output_{i}" for i in range(NUM_TASKS_PER_PROCESS_TEST)],
    }

    futures = []
    total_tasks_expected_to_run = NUM_PROCESSES_TEST * NUM_TASKS_PER_PROCESS_TEST

    test_start_time = time.time()

    with ProcessPoolExecutor(max_workers=NUM_PROCESSES_TEST) as executor:
        for i in range(NUM_PROCESSES_TEST):
            future = executor.submit(
                _process_pool_worker_for_concurrency_test,  # Worker function
                mock_sandbox_url,
                process_in_outs,
                mock_generation,
                mock_memory_limit_mb,
                mock_language,
                mock_timeout,
                global_mp_semaphore,  # Global semaphore to test
                active_calls_counter,  # Shared variables for tracking
                max_calls_tracker,
                call_lock,
            )
            futures.append(future)

    # Wait for all processes to complete and collect results
    num_tasks_processed_per_worker = [f.result() for f in futures]
    test_end_time = time.time()
    total_execution_time = test_end_time - test_start_time

    # Print some test statistics for debugging and validation
    print("\n--- Global Concurrency Test Stats ---")
    print(f"Semaphore Limit (MAX_GLOBAL_CONCURRENCY_LIMIT_TEST): {MAX_GLOBAL_CONCURRENCY_LIMIT_TEST}")
    print(f"Number of Processes (NUM_PROCESSES_TEST): {NUM_PROCESSES_TEST}")
    print(f"Tasks per Process (NUM_TASKS_PER_PROCESS_TEST): {NUM_TASKS_PER_PROCESS_TEST}")
    print(f"Total Tasks Submitted: {total_tasks_expected_to_run}")
    print(f"Simulated API Call Duration: {SIMULATED_API_CALL_DURATION_TEST}s")
    print(f"Total Test Execution Time: {total_execution_time:.2f}s")
    print(f"Max Concurrent Mock API Calls Observed: {max_calls_tracker.value}")
    # print(f"Tasks processed per worker: {num_tasks_processed_per_worker}")

    # Verify that all submitted tasks have been processed
    assert sum(num_tasks_processed_per_worker) == total_tasks_expected_to_run, (
        "Mismatch in the number of tasks processed."
    )

    # Verify that the mock API was called at least once
    assert max_calls_tracker.value > 0, "The mocked API call_sandbox_api was not called."

    # Core assertion: Observed maximum concurrent calls should not exceed the semaphore's limit
    assert max_calls_tracker.value <= MAX_GLOBAL_CONCURRENCY_LIMIT_TEST, (
        f"Observed concurrency ({max_calls_tracker.value}) exceeded semaphore limit "
        f"({MAX_GLOBAL_CONCURRENCY_LIMIT_TEST})."
    )

    # Optional: Rough check on execution time to verify semaphore is working to limit concurrency
    # Theoretical minimum execution time = (Total tasks / Concurrency limit) * Single task duration
    # Actual time will be longer due to various overheads
    min_expected_duration = (
        total_tasks_expected_to_run * SIMULATED_API_CALL_DURATION_TEST
    ) / MAX_GLOBAL_CONCURRENCY_LIMIT_TEST
    # print(f"Minimum Expected Execution Time (approx): {min_expected_duration:.2f}s")
    # Allow some margin, e.g., 80% of theoretical minimum time
    assert total_execution_time >= min_expected_duration * 0.8, (
        f"Total execution time ({total_execution_time:.2f}s) was unexpectedly short, suggesting the "
        f"semaphore might not be effectively limiting concurrency as expected "
        f"(min expected: {min_expected_duration * 0.8:.2f}s)."
    )


# Ensure there is no more code after this point if these were the last functions.
# If there was other code, it would follow here.
def test_unit_invalid_input_format():
    """Unit test: Invalid in_outs format passed"""
    results, metadata_list = check_correctness(SANDBOX_URL, None, CODE_SUCCESS)
    assert results == [-1]
    assert metadata_list[0]["error"] == "Invalid input/output data"

    results, metadata_list = check_correctness(SANDBOX_URL, {}, CODE_SUCCESS)
    assert results == [-1]
    assert metadata_list[0]["error"] == "Invalid input/output data"

    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_INVALID_MISSING_KEY, CODE_SUCCESS)
    assert results == [-1]
    assert metadata_list[0]["error"] == "Invalid input/output data"


@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_unit_input_output_mismatch():
    """Unit test: Mismatch between the number of inputs and outputs"""
    results, metadata_list = check_correctness(SANDBOX_URL, INPUT_OUTPUT_MISMATCH, CODE_SUCCESS)
    assert results == [-1]
    assert len(metadata_list) == 1
    assert metadata_list[0]["error"] == "Input/output count mismatch"


@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_integration_concurrency_all_timeout():
    """Integration test: High concurrency (100 cases) against real API, all causing timeout"""
    concurrency_level = 100
    code_infinite_loop = """
def knight_moves(X, Y):
    MOD = 10**9 + 7
    dp = [[0] * (Y + 1) for _ in range(X + 1)]
    dp[0][0] = 1
    for i in range(1, X + 1):
        for j in range(1, Y + 1):
            dp[i][j] = (dp[i - 1][j] + dp[i][j - 1]) % MOD
    return dp[X][Y]

def solve():
    X, Y = map(int, input().split())
    print(knight_moves(X, Y))

if __name__ == "__main__":
    solve()
    """

    # Generate 100 simple input/output pairs (content doesn't matter)
    timeout_inputs = ["324 384429" for i in range(concurrency_level)]
    timeout_outputs = [f"output_{i}\n" for i in range(concurrency_level)]
    timeout_in_outs = {"inputs": timeout_inputs, "outputs": timeout_outputs}

    # Set a timeout for the test cases
    test_timeout = 10  # Set a timeout value

    start_time = time.time()
    results, metadata_list = check_correctness(SANDBOX_URL, timeout_in_outs, code_infinite_loop, timeout=test_timeout)
    end_time = time.time()
    duration = end_time - start_time
    print(f"\nHigh concurrency all timeout test ({concurrency_level} cases) duration: {duration:.2f} seconds")

    # Verify all results are -3 (timeout)
    assert len(results) == concurrency_level, f"Expected {concurrency_level} results, got {len(results)}"
    all_timed_out = all(r == -3 for r in results)
    if not all_timed_out:
        non_timeout_indices = [i for i, r in enumerate(results) if r != -3]
        print(f"Indices that did not time out: {non_timeout_indices}")
        # Print metadata for the first few non-timeout cases for debugging
        for i in non_timeout_indices[:5]:
            print(f"Metadata for non-timeout case {i}: {metadata_list[i]}")
    assert all_timed_out, f"Not all {concurrency_level} concurrent tests resulted in timeout (-3). Results: {results}"

    # Verify metadata count and status of the first case
    assert len(metadata_list) == concurrency_level
    assert metadata_list[0]["status"] == "timeout"


@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_fn_name_success_single_case():
    """Tests successful execution for a single test case with fn_name.
    from livecodebench/code_generation_lite test 510
    """
    generation_code = """
class Solution:
    def occurrencesOfElement(self, nums: List[int], queries: List[int], x: int) -> List[int]:
        positions = defaultdict(list)
        for idx, num in enumerate(nums):
            positions[num].append(idx)

        x_positions = positions[x]
        answer = []
        for k in queries:
            if k > len(x_positions):
                answer.append(-1)
            else:
                answer.append(x_positions[k-1])
        return answer
"""
    in_outs = {
        "fn_name": "occurrencesOfElement",
        "inputs": ["[1, 3, 1, 7]\n[1, 3, 2, 4]\n1", "[1, 2, 3]\n[10]\n5"],
        "outputs": ["[0, -1, 2, -1]", "[-1]"],
    }

    # Use a short timeout for fast tests
    results, metadata_list = check_correctness(SANDBOX_URL, in_outs, generation_code, timeout=5)
    # from verl.utils.reward_score.prime_code import apps_check_correctness
    # results, metadata_list = apps_check_correctness(in_outs=in_outs, generation=generation_code,
    #                                                        timeout=50000, debug=True)

    assert results == [True, True]
    assert "error" not in metadata_list[0]
    assert metadata_list[0].get("status") != "compile_error"
    assert metadata_list[0].get("status") != "runtime_error"


@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_none_and_empty_stdin_passed_correctly():
    """
    Tests that when stdin data is set to an empty string or None, it is still
    is passed correctly to Sandbox Fusion as an empty string.
    """
    echo_code = """
import sys
print(f"You said '{sys.stdin.readline().strip()}'")
"""
    in_outs = {
        "inputs": [None, "", "hello"],
        "outputs": ["You said ''", "You said ''", "You said 'hello'"],
    }

    # Use a short timeout for fast tests
    results, metadata_list = check_correctness(SANDBOX_URL, in_outs, echo_code, timeout=5)

    assert results == [True, True, True]
    assert "error" not in metadata_list[0]
    assert metadata_list[0].get("status") != "compile_error"
    assert metadata_list[0].get("status") != "runtime_error"


@pytest.mark.skipif(skip_condition, reason=skip_reason)
def test_assert_case_success():
    """Tests successful execution for assert case.
    from KodCode
    """
    generation_code = """
from typing import List, Tuple

def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []

    # Sort intervals by the start time
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]
        # If intervals overlap, merge them
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)

    return merged
"""
    test_cases = {
        "fn_name": "merge_intervals",
        "assert_case": [
            "assert merge_intervals([(0, 1), (3, 5), (4, 7), (6, 8), (10, 12),"
            " (12, 14)]) == [(0, 1), (3, 8), (10, 14)]",
            "assert merge_intervals([(1, 2), (2, 3), (3, 4)]) == [(1, 4)]",
            "assert merge_intervals([(1, 2), (3, 4), (5, 6)]) == [(1, 2), (3, 4), (5, 5)]",
        ],
    }

    assert_cases = test_cases.get("assert_case")
    test_cases.setdefault("inputs", ["" for _ in assert_cases])
    test_cases.setdefault("outputs", [None for _ in assert_cases])

    # Use a short timeout for fast tests
    results, metadata_list = check_correctness(SANDBOX_URL, test_cases, generation_code, timeout=5)
    assert results == [True, True, -2]
    for i in range(2):
        assert "error" not in metadata_list[i]
        assert metadata_list[i].get("status") == "success"
        assert metadata_list[i].get("expected_output") is None
        assert metadata_list[i].get("status") != "runtime_error"
    assert "error" not in metadata_list[2]
    assert metadata_list[2].get("status") != "success"
    assert metadata_list[2].get("expected_output") is None
    assert metadata_list[2].get("status") == "runtime_error"
