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

The same failure mode generalises: any override without a ``+`` prefix is meant
to land on a key the config schema already declares, and hydra gives no warning
when it does not. ``test_plain_overrides_exist_in_config_schema`` checks every
such override in these scripts against the checked-in generated config, which is
what turns "hydra accepted it" into "something actually reads it".

Checks read the scripts as text and the generated config as YAML, so this runs
anywhere -- no torch, no NPU, no hydra.
"""

import re
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SAPO_DIR = REPO_ROOT / "examples" / "sapo_trainer"
CONFIG_DIR = REPO_ROOT / "verl" / "trainer" / "config"

# Fields that live on ActorConfig and must never be nested under policy_loss.
ACTOR_LEVEL_FIELDS = ("tau_pos", "tau_neg")

# A hydra override in these scripts is a dotted ``key=value`` at line start,
# optionally ``+``-prefixed to create a new key. Bash forbids dots in variable
# names, so requiring a dot excludes shell assignments and config-group
# selectors (``model_engine=megatron``) without an allowlist.
OVERRIDE_RE = re.compile(r"^\s*(\+?)([A-Za-z_]\w*(?:\.\w+)+)=", re.M)


def _load_schema(script_text: str) -> dict:
    """Generated config the script's engine selection resolves to."""
    name = (
        "_generated_ppo_megatron_trainer.yaml"
        if "model_engine=megatron" in script_text
        else "_generated_ppo_trainer.yaml"
    )
    return yaml.safe_load((CONFIG_DIR / name).read_text())


def _declares(schema: dict, dotted_key: str) -> bool:
    node = schema
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True


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

    def test_plain_overrides_exist_in_config_schema(self):
        """Every non-``+`` override must land on a key the schema declares.

        This is the general form of the tau bug: hydra silently accepts an
        override on an undeclared key, so a misspelled path costs a whole run
        before anyone notices the value never took effect.
        """
        for path in sorted(SAPO_DIR.glob("run_*.sh")):
            text = path.read_text()
            schema = _load_schema(text)
            for prefix, key in OVERRIDE_RE.findall(text):
                if prefix == "+":
                    continue
                with self.subTest(file=path.name, key=key):
                    self.assertTrue(
                        _declares(schema, key),
                        f"{path.name}: '{key}' is overridden without a '+' prefix but no "
                        f"such key exists in the generated config. Either fix the path or "
                        f"prefix it with '+' if creating a new key is genuinely intended.",
                    )


class TestSapoMegatronProfiling(unittest.TestCase):
    """The Megatron script carries the profiler behind an opt-in toggle.

    Profiling this run has to reproduce its exact parallel layout (TP/EP/ETP
    over 16 ranks) to say anything transferable about where the step time goes,
    so the profiler rides on the same canonical script rather than a forked
    copy -- which also keeps it clear of the ``npu`` token that
    ``check_example_naming.py`` forbids in filenames.
    """

    SCRIPT = SAPO_DIR / "run_qwen3_30b_a3b_megatron.sh"

    # Keys the profiler toggle must drive. All are declared by the schema, so
    # none of them may be '+'-prefixed.
    REQUIRED_PROFILER_KEYS = (
        "global_profiler.tool",
        "global_profiler.steps",
        "global_profiler.save_path",
        "actor_rollout_ref.actor.profiler.enable",
        "actor_rollout_ref.actor.profiler.ranks",
        "actor_rollout_ref.actor.profiler.tool_config.npu.discrete",
        "actor_rollout_ref.actor.profiler.tool_config.npu.level",
        "actor_rollout_ref.rollout.profiler.enable",
        "actor_rollout_ref.ref.profiler.enable",
    )

    def setUp(self):
        self.text = self.SCRIPT.read_text()

    def test_profiling_defaults_to_off(self):
        self.assertTrue(
            "PROFILE=${PROFILE:-0}" in self.text,
            f"{self.SCRIPT.name}: profiling must be opt-in via PROFILE=${{PROFILE:-0}}; "
            f"a default-on profiler would silently tax every run",
        )

    def test_profiler_keys_are_present_and_schema_backed(self):
        schema = _load_schema(self.text)
        emitted = {key: prefix for prefix, key in OVERRIDE_RE.findall(self.text)}
        for key in self.REQUIRED_PROFILER_KEYS:
            with self.subTest(key=key):
                self.assertTrue(key in emitted, f"{self.SCRIPT.name} never overrides '{key}'")
                self.assertEqual(
                    "",
                    emitted[key],
                    f"'{key}' is declared by the schema; a '+' prefix would create a parallel key that nothing reads.",
                )
                self.assertTrue(_declares(schema, key), f"'{key}' missing from generated config")

    def test_checkpoint_retention_is_bounded(self):
        """Local-disk checkpoints need a retention bound or they fill the node."""
        self.assertTrue(
            "trainer.max_actor_ckpt_to_keep" in self.text,
            f"{self.SCRIPT.name}: checkpoints now land on node-local disk; without "
            f"max_actor_ckpt_to_keep the overlay fills up over a long run",
        )


if __name__ == "__main__":
    unittest.main()
