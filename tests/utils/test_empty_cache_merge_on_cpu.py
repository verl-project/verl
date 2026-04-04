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

"""Unit tests for empty cache deduplication and merge mechanisms.

These tests run on CPU (no GPU required) by mocking the device layer.
They validate the cooldown, suppression, and context manager logic
in `verl.utils.memory_utils`.
"""

import importlib
import importlib.util
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: import memory_utils without pulling in heavy deps (torch, ray, tensordict)
# This allows the test to run on minimal Python environments.
# ---------------------------------------------------------------------------

_VERL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Pre-mock heavy dependencies so importing memory_utils doesn't fail
for _mod in ("torch", "verl", "verl.utils", "verl.utils.device"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Provide a concrete get_torch_device and is_cuda_available in the mock
_device_mod = sys.modules["verl.utils.device"]
_device_mod.get_torch_device = MagicMock()
_device_mod.is_cuda_available = False

# Now load memory_utils from file
_mu_path = os.path.join(_VERL_ROOT, "verl", "utils", "memory_utils.py")
_spec = importlib.util.spec_from_file_location("verl.utils.memory_utils", _mu_path)
mem_utils = importlib.util.module_from_spec(_spec)
sys.modules["verl.utils.memory_utils"] = mem_utils
_spec.loader.exec_module(mem_utils)

aggressive_empty_cache = mem_utils.aggressive_empty_cache
empty_cache_context = mem_utils.empty_cache_context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_global_state():
    """Reset the module-level global state between tests."""
    mem_utils._last_call_time = 0.0
    mem_utils._suppress_depth = 0
    mem_utils._suppressed_calls = 0


class _CallCounter:
    """Tracks how many times the *real* cleanup body executes (gc + empty_cache)."""

    def __init__(self):
        self.gc_collect_count = 0
        self.empty_cache_count = 0
        self.synchronize_count = 0

    def mock_gc_collect(self):
        self.gc_collect_count += 1

    def make_mock_device(self):
        """Create a mock device that counts calls."""
        device = MagicMock()
        device.is_available.return_value = True
        device.memory_reserved.return_value = 0
        device.memory_allocated.return_value = 0

        def _empty_cache():
            self.empty_cache_count += 1

        def _synchronize():
            self.synchronize_count += 1

        device.empty_cache = _empty_cache
        device.synchronize = _synchronize
        return device


# ---------------------------------------------------------------------------
# Tests: Cooldown mechanism
# ---------------------------------------------------------------------------


class TestCooldownMechanism:
    """Tests for the time-window deduplication (cooldown) in aggressive_empty_cache."""

    def setup_method(self):
        _reset_global_state()

    @patch("verl.utils.memory_utils._COOLDOWN_SECONDS", 0.2)
    def test_second_call_within_cooldown_is_skipped(self):
        """Two calls within the cooldown window → only the first should execute."""
        counter = _CallCounter()
        device = counter.make_mock_device()

        with patch("verl.utils.memory_utils.get_torch_device", return_value=device):
            with patch("gc.collect", side_effect=counter.mock_gc_collect):
                aggressive_empty_cache(force_sync=False)
                assert counter.gc_collect_count == 1

                # Immediately call again — should be skipped
                aggressive_empty_cache(force_sync=False)
                assert counter.gc_collect_count == 1  # still 1!

    @patch("verl.utils.memory_utils._COOLDOWN_SECONDS", 0.05)
    def test_call_after_cooldown_expires(self):
        """After the cooldown expires, the next call should execute."""
        counter = _CallCounter()
        device = counter.make_mock_device()

        with patch("verl.utils.memory_utils.get_torch_device", return_value=device):
            with patch("gc.collect", side_effect=counter.mock_gc_collect):
                aggressive_empty_cache(force_sync=False)
                assert counter.gc_collect_count == 1

                time.sleep(0.06)  # exceed cooldown

                aggressive_empty_cache(force_sync=False)
                assert counter.gc_collect_count == 2

    @patch("verl.utils.memory_utils._COOLDOWN_SECONDS", 0.2)
    def test_bypass_cooldown_flag(self):
        """bypass_cooldown=True should force execution even within cooldown."""
        counter = _CallCounter()
        device = counter.make_mock_device()

        with patch("verl.utils.memory_utils.get_torch_device", return_value=device):
            with patch("gc.collect", side_effect=counter.mock_gc_collect):
                aggressive_empty_cache(force_sync=False)
                assert counter.gc_collect_count == 1

                aggressive_empty_cache(force_sync=False, bypass_cooldown=True)
                assert counter.gc_collect_count == 2  # forced!


# ---------------------------------------------------------------------------
# Tests: Suppression context manager
# ---------------------------------------------------------------------------


class TestEmptyCacheContext:
    """Tests for the empty_cache_context suppression mechanism."""

    def setup_method(self):
        _reset_global_state()

    @patch("verl.utils.memory_utils._COOLDOWN_SECONDS", 0.0)  # disable cooldown for clarity
    def test_suppression_inside_context(self):
        """Calls inside the context should be suppressed, cleanup on exit."""
        counter = _CallCounter()
        device = counter.make_mock_device()

        with patch("verl.utils.memory_utils.get_torch_device", return_value=device):
            with patch("gc.collect", side_effect=counter.mock_gc_collect):
                with empty_cache_context(force_sync=True):
                    # These should all be suppressed
                    aggressive_empty_cache(force_sync=True)
                    aggressive_empty_cache(force_sync=True)
                    aggressive_empty_cache(force_sync=True)
                    assert counter.gc_collect_count == 0  # nothing yet

                # Context exit → one deferred cleanup
                assert counter.gc_collect_count == 1
                assert counter.synchronize_count == 1

    @patch("verl.utils.memory_utils._COOLDOWN_SECONDS", 0.0)
    def test_no_cleanup_if_no_suppressed_calls(self):
        """If no calls were suppressed, context exit should not trigger cleanup."""
        counter = _CallCounter()
        device = counter.make_mock_device()

        with patch("verl.utils.memory_utils.get_torch_device", return_value=device):
            with patch("gc.collect", side_effect=counter.mock_gc_collect):
                with empty_cache_context(force_sync=True):
                    pass  # no aggressive_empty_cache calls inside

                assert counter.gc_collect_count == 0  # no cleanup needed

    @patch("verl.utils.memory_utils._COOLDOWN_SECONDS", 0.0)
    def test_nested_contexts(self):
        """Nested contexts: only the outermost exit triggers cleanup."""
        counter = _CallCounter()
        device = counter.make_mock_device()

        with patch("verl.utils.memory_utils.get_torch_device", return_value=device):
            with patch("gc.collect", side_effect=counter.mock_gc_collect):
                with empty_cache_context(force_sync=True):
                    aggressive_empty_cache(force_sync=True)  # suppressed
                    assert counter.gc_collect_count == 0

                    with empty_cache_context(force_sync=True):
                        aggressive_empty_cache(force_sync=True)  # still suppressed
                        aggressive_empty_cache(force_sync=True)  # still suppressed
                        assert counter.gc_collect_count == 0

                    # Inner context exit — NOT outermost, no cleanup
                    assert counter.gc_collect_count == 0

                # Outer context exit — IS outermost, triggers cleanup
                assert counter.gc_collect_count == 1

    @patch("verl.utils.memory_utils._COOLDOWN_SECONDS", 0.0)
    def test_suppressed_calls_count(self):
        """The context should track how many calls were suppressed."""
        counter = _CallCounter()
        device = counter.make_mock_device()

        with patch("verl.utils.memory_utils.get_torch_device", return_value=device):
            with patch("gc.collect", side_effect=counter.mock_gc_collect):
                with empty_cache_context(force_sync=True):
                    for _ in range(5):
                        aggressive_empty_cache(force_sync=True)

                # All 5 suppressed, 1 deferred cleanup
                assert counter.gc_collect_count == 1

    @patch("verl.utils.memory_utils._COOLDOWN_SECONDS", 0.0)
    def test_context_exit_uses_bypass_cooldown(self):
        """The deferred cleanup on context exit should bypass cooldown."""
        counter = _CallCounter()
        device = counter.make_mock_device()

        with patch("verl.utils.memory_utils.get_torch_device", return_value=device):
            with patch("gc.collect", side_effect=counter.mock_gc_collect):
                # First: a normal call to set _last_call_time
                aggressive_empty_cache(force_sync=False)
                assert counter.gc_collect_count == 1

        # Now set a long cooldown
        _reset_global_state()
        counter = _CallCounter()
        device = counter.make_mock_device()

        with patch("verl.utils.memory_utils._COOLDOWN_SECONDS", 999):
            with patch("verl.utils.memory_utils.get_torch_device", return_value=device):
                with patch("gc.collect", side_effect=counter.mock_gc_collect):
                    with empty_cache_context(force_sync=False):
                        aggressive_empty_cache(force_sync=False)  # suppressed

                    # Context exit should still trigger cleanup despite long cooldown
                    assert counter.gc_collect_count == 1

    @patch("verl.utils.memory_utils._COOLDOWN_SECONDS", 0.0)
    def test_context_exception_still_cleans_up(self):
        """If an exception occurs inside the context, cleanup should still happen."""
        counter = _CallCounter()
        device = counter.make_mock_device()

        with patch("verl.utils.memory_utils.get_torch_device", return_value=device):
            with patch("gc.collect", side_effect=counter.mock_gc_collect):
                with pytest.raises(ValueError):
                    with empty_cache_context(force_sync=True):
                        aggressive_empty_cache(force_sync=True)
                        raise ValueError("test error")

                # Despite the exception, cleanup should have run
                assert counter.gc_collect_count == 1

    @patch("verl.utils.memory_utils._COOLDOWN_SECONDS", 0.0)
    def test_suppress_depth_resets_after_exception(self):
        """After an exception in context, suppression state should be clean."""
        counter = _CallCounter()
        device = counter.make_mock_device()

        with patch("verl.utils.memory_utils.get_torch_device", return_value=device):
            with patch("gc.collect", side_effect=counter.mock_gc_collect):
                with pytest.raises(ValueError):
                    with empty_cache_context(force_sync=True):
                        aggressive_empty_cache(force_sync=True)
                        raise ValueError("test error")

                # After exception, should NOT be suppressed anymore
                assert mem_utils._suppress_depth == 0

                # Normal call should work
                aggressive_empty_cache(force_sync=False)
                assert counter.gc_collect_count == 2  # 1 from context exit + 1 normal


# ---------------------------------------------------------------------------
# Tests: force_sync parameter
# ---------------------------------------------------------------------------


class TestForceSync:
    """Tests for force_sync behavior."""

    def setup_method(self):
        _reset_global_state()

    @patch("verl.utils.memory_utils._COOLDOWN_SECONDS", 0.0)
    def test_force_sync_true_calls_synchronize(self):
        """force_sync=True should call device.synchronize()."""
        counter = _CallCounter()
        device = counter.make_mock_device()

        with patch("verl.utils.memory_utils.get_torch_device", return_value=device):
            with patch("gc.collect", side_effect=counter.mock_gc_collect):
                aggressive_empty_cache(force_sync=True)
                assert counter.synchronize_count == 1

    @patch("verl.utils.memory_utils._COOLDOWN_SECONDS", 0.0)
    def test_force_sync_false_skips_synchronize(self):
        """force_sync=False should NOT call device.synchronize()."""
        counter = _CallCounter()
        device = counter.make_mock_device()

        with patch("verl.utils.memory_utils.get_torch_device", return_value=device):
            with patch("gc.collect", side_effect=counter.mock_gc_collect):
                aggressive_empty_cache(force_sync=False)
                assert counter.synchronize_count == 0


# ---------------------------------------------------------------------------
# Tests: Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Basic thread safety tests for global state."""

    def setup_method(self):
        _reset_global_state()

    def test_concurrent_contexts_do_not_interfere(self):
        """Two threads using empty_cache_context should not corrupt global state.

        We only test that the suppress_depth returns to 0 after concurrent contexts
        complete. We don't call aggressive_empty_cache because `patch` on
        get_torch_device is not thread-safe for concurrent use.
        """
        errors = []

        def worker():
            try:
                # Just exercise the context manager enter/exit logic
                # Do NOT increment _suppressed_calls (avoids triggering cleanup on exit)
                with empty_cache_context(force_sync=False):
                    time.sleep(0.01)
                # Context exited without triggering cleanup (suppressed_calls == 0)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        # After all threads finish, suppress depth should be 0
        assert mem_utils._suppress_depth == 0


# ---------------------------------------------------------------------------
# Tests: Integration — simulating a Megatron-style call sequence
# ---------------------------------------------------------------------------


class TestMegatronStyleIntegration:
    """Simulates the call pattern in MegatronHybridEngineWorker to verify merge benefit."""

    def setup_method(self):
        _reset_global_state()

    @patch("verl.utils.memory_utils._COOLDOWN_SECONDS", 0.0)
    def test_generate_sequences_merges_three_calls(self):
        """Simulates generate_sequences wrapping rollout_mode + generate + cleanup.

        Original: 3 aggressive_empty_cache calls.
        After optimization: 1 effective call (via empty_cache_context).
        """
        counter = _CallCounter()
        device = counter.make_mock_device()

        def fake_rollout_mode():
            """Simulates rollout_mode() which has 2 aggressive_empty_cache calls."""
            aggressive_empty_cache(force_sync=True)
            # ... export weights, update_weights ...
            aggressive_empty_cache(force_sync=True)

        def fake_generate():
            """Simulates rollout.generate_sequences()."""
            pass  # actual vLLM generation

        with patch("verl.utils.memory_utils.get_torch_device", return_value=device):
            with patch("gc.collect", side_effect=counter.mock_gc_collect):
                # WITHOUT optimization: 3 calls
                _reset_global_state()
                counter2 = _CallCounter()
                device2 = counter2.make_mock_device()

                with patch("verl.utils.memory_utils.get_torch_device", return_value=device2):
                    with patch("gc.collect", side_effect=counter2.mock_gc_collect):
                        fake_rollout_mode()
                        fake_generate()
                        aggressive_empty_cache(force_sync=True)

                # All 3 calls executed (due to 0 cooldown)
                assert counter2.gc_collect_count == 3

        # WITH optimization: wrapped in empty_cache_context
        _reset_global_state()
        counter3 = _CallCounter()
        device3 = counter3.make_mock_device()

        with patch("verl.utils.memory_utils.get_torch_device", return_value=device3):
            with patch("gc.collect", side_effect=counter3.mock_gc_collect):
                with empty_cache_context(force_sync=True):
                    fake_rollout_mode()   # 2 calls suppressed
                    fake_generate()
                    aggressive_empty_cache(force_sync=True)  # 1 call suppressed

                # Only 1 deferred cleanup on context exit
                assert counter3.gc_collect_count == 1  # 3x → 1x!

    @patch("verl.utils.memory_utils._COOLDOWN_SECONDS", 0.0)
    def test_full_training_step_simulation(self):
        """Simulates a full training step with multiple phases.

        Original Megatron path: 6 aggressive_empty_cache calls per step.
        After optimization:
          - generate_sequences: 3 → 1 (context merge)
          - compute_log_prob: 1 (force_sync=False, cheaper)
          - compute_ref_log_prob: 1 (force_sync=False, cheaper)
          - update_actor: 1 (force_sync=False, cheaper)
          Total: 4 effective calls (down from 6), 3 without synchronize
        """
        counter = _CallCounter()
        device = counter.make_mock_device()

        with patch("verl.utils.memory_utils.get_torch_device", return_value=device):
            with patch("gc.collect", side_effect=counter.mock_gc_collect):
                # Phase 1: generate_sequences (with context merge)
                with empty_cache_context(force_sync=True):
                    aggressive_empty_cache(force_sync=True)   # rollout_mode start
                    aggressive_empty_cache(force_sync=True)   # rollout_mode end
                    aggressive_empty_cache(force_sync=True)   # generate end

                assert counter.gc_collect_count == 1  # 3 → 1
                assert counter.synchronize_count == 1

                # Phase 2: compute_log_prob (force_sync=False)
                aggressive_empty_cache(force_sync=False)
                assert counter.gc_collect_count == 2
                assert counter.synchronize_count == 1  # no sync!

                # Phase 3: compute_ref_log_prob (force_sync=False)
                aggressive_empty_cache(force_sync=False)
                assert counter.gc_collect_count == 3
                assert counter.synchronize_count == 1  # still no sync!

                # Phase 4: update_actor (force_sync=False)
                aggressive_empty_cache(force_sync=False)
                assert counter.gc_collect_count == 4
                assert counter.synchronize_count == 1

        # Total: 4 gc_collect (down from 6), 1 synchronize (down from 6)
