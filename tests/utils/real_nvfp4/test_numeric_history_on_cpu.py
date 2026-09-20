# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Run only in the scheduled numeric-gate regression job."""

import re
import unittest

from examples.real_nvfp4.check_history import REQUIRED, finite_scalar, validate_lines


def row(step, overrides=None, omit=None, wrapped=True):
    values = {key: "0.01" for key in REQUIRED}
    values.update(
        {
            "training/global_step": str(step),
            "train/actor_optimizer_steps": "1",
            "rollout_corr/rollout_is_eff_sample_size": "0.99",
        }
    )
    values.update(overrides or {})
    fields = [
        f"{key}:{'np.float64(' + value + ')' if wrapped else value}" for key, value in values.items() if key != omit
    ]
    return f"\x1b[36m(Runner pid=1)\x1b[0m step:{step} - " + " - ".join(fields)


class NumericHistoryTests(unittest.TestCase):
    def test_raw_and_wrapped_scientific_numbers(self):
        for raw in (
            "1e-6",
            "-2.4E+2",
            "+.125",
            "np.float64(-1e-9)",
            "np.float32( 0.1 )",
            "np.int64(1)",
        ):
            with self.subTest(raw=raw):
                self.assertIsInstance(finite_scalar(raw), float)

    def test_nonfinite_all_critical_metrics(self):
        for key in REQUIRED:
            for value in ("nan", "+nan", "NaN", "inf", "-inf", "+Infinity", "1e999"):
                for wrapped in (False, True):
                    with (
                        self.subTest(key=key, value=value, wrapped=wrapped),
                        self.assertRaises(ValueError),
                    ):
                        validate_lines([row(1, {key: value}, wrapped=wrapped)], 1, 1)

    def test_old_regex_misses_actual_numpy_format(self):
        line = row(1, {"actor/loss": "nan"})
        self.assertIsNone(re.search(r"actor/(loss|pg_loss|grad_norm):\s*(nan|[-+]?inf)", line, re.IGNORECASE))
        with self.assertRaises(ValueError):
            validate_lines([line], 1, 1)

    def test_malformed_and_missing(self):
        for key in REQUIRED:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_lines([row(1, omit=key)], 1, 1)
        for raw in ("tensor(1.)", "np.float64(1.)trailing", "nantext", "", "True"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_lines([row(1, {"actor/loss": raw}, wrapped=False)], 1, 1)

    def test_all_steps_including_final(self):
        lines = [row(i) for i in range(1, 7)]
        self.assertEqual(sorted(validate_lines(lines, 1, 6)), list(range(1, 7)))
        lines[-1] = row(6, {"actor/grad_norm": "nan"})
        with self.assertRaises(ValueError):
            validate_lines(lines, 1, 6)

    def test_resume_interval(self):
        self.assertEqual(sorted(validate_lines([row(41), row(42)], 41, 42)), [41, 42])
        with self.assertRaises(ValueError):
            validate_lines([row(1), row(41), row(42)], 41, 42)

    def test_missing_duplicate_and_inconsistent_steps(self):
        for lines in (
            [row(1)],
            [row(1), row(3)],
            [row(1), row(1, {"actor/loss": "0.02"}), row(2)],
            [row(1, {"training/global_step": "2"}), row(2)],
        ):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                validate_lines(lines, 1, 2)
        self.assertEqual(len(validate_lines([row(1), row(1), row(2)], 1, 2)), 2)

    def test_duplicate_key(self):
        with self.assertRaises(ValueError):
            validate_lines([row(1) + " - actor/loss:0.01"], 1, 1)

    def test_existing_numeric_bounds_every_step(self):
        for override in (
            {"rollout_corr/kl": "0.6"},
            {"rollout_corr/rollout_is_eff_sample_size": "0.79"},
            {"train/actor_optimizer_steps": "2"},
        ):
            with self.subTest(override=override), self.assertRaises(ValueError):
                validate_lines([row(1, override), row(2)], 1, 2)

    def test_ignore_config_and_eval(self):
        lines = ["timing_s/step:999", "step:0 - val/acc:0.2", row(1)]
        self.assertEqual(len(validate_lines(lines, 1, 1)), 1)


if __name__ == "__main__":
    unittest.main()
