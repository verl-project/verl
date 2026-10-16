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
"""Unit tests for the dump-side join-key helpers on RayPPOTrainer."""

import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


class TestBuildJoinKeys(unittest.TestCase):
    def test_happy_path(self):
        """All three columns provided with matching length yield a full dict."""
        out = RayPPOTrainer._build_join_keys(
            uid=np.array(["a", "b", "c"], dtype=object),
            data_source=np.array(["ds1", "ds2", "ds1"], dtype=object),
            index=np.array([10, 11, 12]),
            n=3,
        )
        self.assertEqual(out["uid"], ["a", "b", "c"])
        self.assertEqual(out["data_source"], ["ds1", "ds2", "ds1"])
        self.assertEqual(out["index"], [10, 11, 12])
        for v in out["index"]:
            self.assertIsInstance(v, int)

    def test_missing_columns_are_skipped(self):
        """Columns passed as None are omitted from the returned dict."""
        out = RayPPOTrainer._build_join_keys(
            uid=None,
            data_source=None,
            index=np.array([0, 1]),
            n=2,
        )
        self.assertNotIn("uid", out)
        self.assertNotIn("data_source", out)
        self.assertEqual(out["index"], [0, 1])

    def test_empty_batch(self):
        """n=0 with empty sequences produces empty columns, not a crash."""
        out = RayPPOTrainer._build_join_keys(
            uid=[],
            data_source=[],
            index=[],
            n=0,
        )
        self.assertEqual(out, {"uid": [], "data_source": [], "index": []})

    def test_length_mismatch_warns_and_skips(self):
        """Length mismatch is observational: warn to stdout, drop the column,
        do not raise. This keeps training/validation alive when the dump path
        has a shape bug."""
        for kwargs in (
            {"uid": ["a", "b"], "n": 3},
            {"data_source": ["ds1"], "n": 0},
            {"index": np.array([1, 2, 3]), "n": 2},
        ):
            buf = io.StringIO()
            with redirect_stdout(buf):
                out = RayPPOTrainer._build_join_keys(**kwargs)
            # Column was skipped.
            colliding_key = next(k for k in kwargs if k != "n")
            self.assertNotIn(colliding_key, out)
            self.assertIn("skipping column", buf.getvalue())

    def test_scalar_input_skipped(self):
        """A 0-D scalar (e.g. np.asarray('foo')) is defensively rejected and
        the column is dropped with a warning."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = RayPPOTrainer._build_join_keys(data_source="scalar-ds", n=3)
        self.assertNotIn("data_source", out)
        self.assertIn("0-D scalar", buf.getvalue())

    def test_ragged_input_conversion_failure_warns_and_skips(self):
        """A ragged sequence rejected by numpy is an observational warning,
        not an exception escaping into validation or training."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            out = RayPPOTrainer._build_join_keys(
                data_source=[["a"], ["b", "c"]],
                n=2,
            )
        self.assertNotIn("data_source", out)
        self.assertIn("failed to normalize data_source", buf.getvalue())
        self.assertIn("skipping column", buf.getvalue())

    def test_uid_transformation_failure_warns_and_skips(self):
        """An exception raised while canonicalizing uid is contained to the
        bad dump column."""

        class BrokenString:
            """Value whose string conversion fails for the test."""

            def __str__(self):
                """Raise to exercise the transform-error boundary."""
                raise RuntimeError("cannot stringify")

        buf = io.StringIO()
        with redirect_stdout(buf):
            out = RayPPOTrainer._build_join_keys(
                uid=np.array([BrokenString()], dtype=object),
                n=1,
            )
        self.assertNotIn("uid", out)
        self.assertIn("failed to normalize uid", buf.getvalue())
        self.assertIn("RuntimeError", buf.getvalue())

    def test_data_source_native_dtype_preserved(self):
        """Numeric data_source stays numeric (no stringification)."""
        out = RayPPOTrainer._build_join_keys(
            data_source=np.array([1, 2, 3]),
            n=3,
        )
        self.assertEqual(out["data_source"], [1, 2, 3])
        for v in out["data_source"]:
            self.assertIsInstance(v, int)

    def test_index_native_dtype_preserved_from_ndarray(self):
        """np.int64 inputs round-trip to python int via ndarray.tolist()."""
        out = RayPPOTrainer._build_join_keys(
            index=np.array([0, 1, 2], dtype=np.int64),
            n=3,
        )
        self.assertEqual(out["index"], [0, 1, 2])
        for v in out["index"]:
            self.assertIsInstance(v, int)

    def test_uid_stringified_from_numpy_scalar(self):
        """numpy str_ / integer uids are canonicalized to python str."""
        out = RayPPOTrainer._build_join_keys(
            uid=np.array([np.str_("x"), np.str_("y")], dtype=object),
            n=2,
        )
        self.assertEqual(out["uid"], ["x", "y"])
        for v in out["uid"]:
            self.assertIsInstance(v, str)

    def test_accepts_plain_list_inputs(self):
        """The validation path passes already-flattened python lists."""
        out = RayPPOTrainer._build_join_keys(
            uid=["u1", "u2"],
            data_source=["ds1", "ds2"],
            index=[7, 8],
            n=2,
        )
        self.assertEqual(out["uid"], ["u1", "u2"])
        self.assertEqual(out["data_source"], ["ds1", "ds2"])
        self.assertEqual(out["index"], [7, 8])


class TestMergeJoinKeysWarnOnCollision(unittest.TestCase):
    def test_no_collision_injects_all(self):
        """When target lacks the keys, all join_keys are injected unchanged."""
        target = {"reward": [0.1, 0.2]}
        RayPPOTrainer._merge_join_keys_warn_on_collision(
            target,
            {"uid": ["a", "b"], "data_source": ["ds", "ds"], "index": [0, 1]},
        )
        self.assertEqual(target["reward"], [0.1, 0.2])
        self.assertEqual(target["uid"], ["a", "b"])
        self.assertEqual(target["data_source"], ["ds", "ds"])
        self.assertEqual(target["index"], [0, 1])

    def test_collision_warns_and_preserves_target(self):
        """On collision the target's value wins; non-colliding keys still land."""
        target = {"uid": ["reward-uid-1", "reward-uid-2"], "reward": [0.0, 0.0]}
        buf = io.StringIO()
        with redirect_stdout(buf):
            RayPPOTrainer._merge_join_keys_warn_on_collision(
                target,
                {"uid": ["dump-uid-1", "dump-uid-2"], "index": [0, 1]},
            )
        # Target's uid preserved; index still injected since no collision there.
        self.assertEqual(target["uid"], ["reward-uid-1", "reward-uid-2"])
        self.assertEqual(target["index"], [0, 1])
        self.assertIn("'uid'", buf.getvalue())
        self.assertIn("keeping reward value", buf.getvalue())

    def test_empty_join_keys_is_noop(self):
        """Merging an empty dict does not touch the target."""
        target = {"reward": [0.0]}
        RayPPOTrainer._merge_join_keys_warn_on_collision(target, {})
        self.assertEqual(target, {"reward": [0.0]})


if __name__ == "__main__":
    unittest.main()
