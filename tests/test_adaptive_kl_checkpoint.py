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

"""
Test that AdaptiveKLController state is preserved across checkpoint save/load.

Bug: When training with use_kl_in_reward=True and kl_ctrl.type="adaptive",
the AdaptiveKLController.value evolves during training via its update() method.
However, _save_checkpoint() and _load_checkpoint() do not persist this value.
On resume, the controller is re-initialized with init_kl_coef, causing a
discontinuity in the KL penalty schedule.
"""

import os
import tempfile

import pytest
import torch

from verl.trainer.ppo.core_algos import AdaptiveKLController, FixedKLController, get_kl_controller


class TestAdaptiveKLControllerStateEvolution:
    """Verify that AdaptiveKLController.value actually changes during training."""

    def test_value_increases_when_kl_too_high(self):
        """When current KL > target KL, the coefficient should increase."""
        ctrl = AdaptiveKLController(init_kl_coef=0.1, target_kl=1.0, horizon=10000)
        initial = ctrl.value

        # Simulate KL being much higher than target (current_kl=5.0 vs target=1.0)
        ctrl.update(current_kl=5.0, n_steps=256)

        assert ctrl.value > initial, (
            f"KL coef should increase when KL > target, but {ctrl.value} <= {initial}"
        )

    def test_value_decreases_when_kl_too_low(self):
        """When current KL < target KL, the coefficient should decrease."""
        ctrl = AdaptiveKLController(init_kl_coef=0.1, target_kl=1.0, horizon=10000)
        initial = ctrl.value

        # Simulate KL being much lower than target (current_kl=0.1 vs target=1.0)
        ctrl.update(current_kl=0.1, n_steps=256)

        assert ctrl.value < initial, (
            f"KL coef should decrease when KL < target, but {ctrl.value} >= {initial}"
        )

    def test_value_diverges_significantly_over_many_steps(self):
        """After many updates, the value drifts far from init_kl_coef."""
        init_coef = 0.1
        ctrl = AdaptiveKLController(init_kl_coef=init_coef, target_kl=1.0, horizon=10000)

        # Simulate 100 training steps with KL consistently above target
        for _ in range(100):
            ctrl.update(current_kl=3.0, n_steps=256)

        # The value should have diverged significantly from the initial
        ratio = ctrl.value / init_coef
        assert ratio > 1.5, (
            f"After 100 high-KL updates, coef should be >1.5x initial, "
            f"but ratio={ratio:.2f} (value={ctrl.value}, init={init_coef})"
        )

    def test_fixed_controller_value_never_changes(self):
        """FixedKLController.value should not change regardless of updates."""
        ctrl = FixedKLController(kl_coef=0.5)
        ctrl.update(current_kl=10.0, n_steps=1000)
        assert ctrl.value == 0.5


class TestKLControllerCheckpointRoundTrip:
    """Test that save/load round-trip preserves KL controller state."""

    def test_save_load_preserves_value(self):
        """The core round-trip test: save evolved value, load it back."""
        init_coef = 0.1
        ctrl = AdaptiveKLController(init_kl_coef=init_coef, target_kl=1.0, horizon=10000)

        # Evolve the controller
        for _ in range(50):
            ctrl.update(current_kl=3.0, n_steps=256)
        evolved_value = ctrl.value
        assert evolved_value != init_coef, "Value should have changed after updates"

        # Save (matching the fix in ray_trainer.py _save_checkpoint)
        with tempfile.TemporaryDirectory() as tmpdir:
            kl_ctrl_path = os.path.join(tmpdir, "kl_ctrl.pt")
            torch.save({"value": torch.tensor(ctrl.value)}, kl_ctrl_path)

            # Simulate resume: create a fresh controller (as __init__ would)
            new_ctrl = AdaptiveKLController(init_kl_coef=init_coef, target_kl=1.0, horizon=10000)
            assert new_ctrl.value == init_coef, "Fresh controller should have init value"

            # Load (matching the fix in ray_trainer.py _load_checkpoint)
            kl_state = torch.load(kl_ctrl_path, weights_only=True)
            new_ctrl.value = kl_state["value"].item()

            assert new_ctrl.value == evolved_value, (
                f"After load, value should be {evolved_value}, got {new_ctrl.value}"
            )

    def test_without_fix_value_resets(self):
        """Demonstrate the bug: without save/load, value resets to init."""
        init_coef = 0.1
        ctrl = AdaptiveKLController(init_kl_coef=init_coef, target_kl=1.0, horizon=10000)

        # Evolve
        for _ in range(50):
            ctrl.update(current_kl=3.0, n_steps=256)
        evolved_value = ctrl.value

        # Simulate resume WITHOUT the fix: just re-create the controller
        new_ctrl = AdaptiveKLController(init_kl_coef=init_coef, target_kl=1.0, horizon=10000)

        # BUG: the value resets to init_kl_coef instead of the evolved value
        assert new_ctrl.value == init_coef, "Without fix, value resets to init"
        assert new_ctrl.value != evolved_value, (
            "Without fix, evolved state is lost"
        )

    def test_quantify_discontinuity(self):
        """Quantify the magnitude of the KL penalty discontinuity on resume."""
        init_coef = 0.1
        ctrl = AdaptiveKLController(init_kl_coef=init_coef, target_kl=0.5, horizon=10000)

        # Simulate training where KL is consistently high, so beta grows
        for _ in range(200):
            ctrl.update(current_kl=2.0, n_steps=512)
        evolved_value = ctrl.value

        # The ratio between evolved and init shows the magnitude of discontinuity
        ratio = evolved_value / init_coef
        # With KL consistently 4x target over 200 steps with large n_steps,
        # the coefficient should have grown substantially
        assert ratio > 1.5, (
            f"KL coef should grow >1.5x under sustained high KL, got {ratio:.2f}x "
            f"(evolved={evolved_value:.4f}, init={init_coef})"
        )

        # On resume without the fix, we'd jump from evolved_value back to init_coef
        # This represents a relative change of (evolved_value - init_coef) / evolved_value
        relative_change = abs(evolved_value - init_coef) / evolved_value
        assert relative_change > 0.3, (
            f"Resume discontinuity should be >30%, got {relative_change*100:.1f}%"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
