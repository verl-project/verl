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

"""Guard the SAPO example scripts against silently-ineffective overrides.

``compute_policy_loss_sapo`` reads ``config.tau_pos`` / ``config.tau_neg`` off
``ActorConfig``. ``PolicyLossConfig`` has no such fields, so an override spelled
``+actor_rollout_ref.actor.policy_loss.tau_pos=...`` lands on a key nobody reads:
hydra accepts it, the run proceeds, and the temperature stays at its default.
That is invisible in logs and makes the paper's key hyper-parameter untunable.

Text-only checks, so this runs anywhere -- no torch, no NPU, no hydra.
"""

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SAPO_DIR = REPO_ROOT / "examples" / "sapo_trainer"

# Fields that live on ActorConfig and must never be nested under policy_loss.
ACTOR_LEVEL_FIELDS = ("tau_pos", "tau_neg")


class TestSapoExampleFlags(unittest.TestCase):
    """SAPO example scripts must override tau at the actor level."""

    def _sapo_files(self):
        files = sorted(SAPO_DIR.glob("run_*.sh")) + [SAPO_DIR / "README.md"]
        return [f for f in files if f.exists()]

    def test_sapo_dir_exists(self):
        self.assertTrue(SAPO_DIR.is_dir(), f"missing {SAPO_DIR}")
        self.assertTrue(list(SAPO_DIR.glob("run_*.sh")), "no SAPO run scripts found")

    def test_tau_is_never_nested_under_policy_loss(self):
        for path in self._sapo_files():
            text = path.read_text()
            for field in ACTOR_LEVEL_FIELDS:
                with self.subTest(file=path.name, field=field):
                    self.assertNotIn(
                        f"policy_loss.{field}",
                        text,
                        f"{path.name}: '{field}' must be overridden as "
                        f"actor_rollout_ref.actor.{field}; nesting it under "
                        f"policy_loss is accepted by hydra but never read.",
                    )

    def test_scripts_setting_tau_use_the_actor_path(self):
        """Any script mentioning tau must set it on the actor config."""
        for path in sorted(SAPO_DIR.glob("run_*.sh")):
            text = path.read_text()
            if "tau_pos" not in text:
                continue
            with self.subTest(file=path.name):
                self.assertIn(
                    "actor_rollout_ref.actor.tau_pos",
                    text,
                    f"{path.name} references tau_pos but never sets actor_rollout_ref.actor.tau_pos",
                )

    def test_sapo_scripts_select_the_sapo_loss_mode(self):
        for path in sorted(SAPO_DIR.glob("run_*.sh")):
            with self.subTest(file=path.name):
                self.assertIn(
                    "policy_loss.loss_mode=sapo",
                    path.read_text(),
                    f"{path.name} lives in sapo_trainer but does not select loss_mode=sapo",
                )


if __name__ == "__main__":
    unittest.main()
