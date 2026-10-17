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

"""Exercise the production checkpoint boundary without importing GPU-only MCore.

CPU CI does not install Megatron-Core. Execute the unchanged save method from
its source, mocking DDP and I/O rather than copying its synchronization logic.
Actual parameter AllGather and checkpoint contents require distributed tests.
"""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


def _load_save_checkpoint():
    """Load the production save method without importing GPU-only dependencies."""
    source = Path(__file__).resolve().parents[2] / "verl/workers/engine/megatron/transformer_impl.py"
    tree = ast.parse(source.read_text())
    engine = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronEngine")
    method = next(node for node in engine.body if isinstance(node, ast.FunctionDef) and node.name == "save_checkpoint")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace["save_checkpoint"]


_SAVE_CHECKPOINT = _load_save_checkpoint()


class CheckpointParamSyncTest(unittest.TestCase):
    def _check_case(
        self, overlap, use_distributed_optimizer, offload, origin_device, ddp, *, tf_config=None, expect_sync=None
    ):
        """Check save ordering and autograd preservation for one engine configuration."""
        events = []
        test = self
        if tf_config is None:
            tf_config = SimpleNamespace(fp8=None, fp4=None)
        if expect_sync is None:
            expect_sync = ddp and use_distributed_optimizer and overlap

        class FakeDDP:
            ddp_config = SimpleNamespace(
                use_distributed_optimizer=use_distributed_optimizer, overlap_param_gather=overlap
            )

            def __init__(self, chunk_id):
                """Identify the model chunk whose synchronization is recorded."""
                self.chunk_id = chunk_id

            def start_param_sync(self, *, force_sync):
                """Assert forced synchronization runs without autograd and record the chunk."""
                test.assertIs(force_sync, True)
                test.assertFalse(torch.is_grad_enabled())
                events.append(f"sync:{self.chunk_id}")

        def save(**kwargs):
            """Check checkpoint arguments and record the write event."""
            self.assertEqual(kwargs, dict(local_path="checkpoint", hdfs_path=None, global_step=2, max_ckpt_to_keep=3))
            events.append("save")

        engine = SimpleNamespace(
            module=[FakeDDP(0), FakeDDP(1)] if ddp else [object()],
            _is_offload_param=offload,
            tf_config=tf_config,
            checkpoint_mananager=SimpleNamespace(save_checkpoint=save),
        )
        grad_enabled = torch.is_grad_enabled()
        with (
            patch.dict(
                _SAVE_CHECKPOINT.__globals__,
                DistributedDataParallel=FakeDDP,
                get_megatron_module_device=lambda _: origin_device,
                load_megatron_model_to_gpu=lambda _, load_grad: events.append("restore"),
                offload_megatron_model_to_cpu=lambda _: events.append("offload"),
            ),
            patch.object(torch.distributed, "barrier", side_effect=lambda: events.append("barrier")),
        ):
            _SAVE_CHECKPOINT(engine, "checkpoint", global_step=2, max_ckpt_to_keep=3)
        expected = ["restore"] if offload or origin_device == "cpu" else []
        if expect_sync:
            expected += ["sync:0", "sync:1"]
        expected += ["save", "barrier"]
        if offload:
            expected += ["offload"]
        self.assertEqual(events, expected)
        self.assertEqual(torch.is_grad_enabled(), grad_enabled)

    def test_sync_after_restore_before_save(self):
        """Verify synchronization eligibility and ordering around restore, save, and offload."""
        cases = [
            (True, True, False, "cuda", True),
            (True, True, True, "cpu", True),
            (True, True, False, "cpu", True),
            (False, True, False, "cuda", True),
            (False, True, True, "cpu", True),
            (True, True, False, "cuda", False),
            (True, False, False, "cuda", True),
            (True, False, True, "cpu", True),
            (False, False, False, "cuda", True),
        ]
        for grad_enabled in (True, False):
            with torch.set_grad_enabled(grad_enabled):
                for case in cases:
                    with self.subTest(case=case, grad_enabled=grad_enabled):
                        self._check_case(*case)

    def test_precision_config_controls_sync(self):
        """Verify effective precision settings control the extra save-boundary synchronization."""
        cases = [
            ("missing_precision_fields", {}, True),
            ("legacy_without_fp4", {"fp8": None}, True),
            ("disabled", {"fp8": None, "fp4": None}, True),
            ("disabled_boolean", {"fp8": False, "fp4": False}, True),
            ("recipe_only", {"fp8_recipe": "mxfp8", "fp4_recipe": "nvfp4"}, True),
            ("fp8_e4m3", {"fp8": "e4m3", "fp4": None}, False),
            ("fp8_delayed", {"fp8": "hybrid", "fp8_recipe": "delayed"}, False),
            ("fp8_tensorwise", {"fp8": "hybrid", "fp8_recipe": "tensorwise"}, False),
            ("fp8_blockwise", {"fp8": "hybrid", "fp8_recipe": "blockwise"}, False),
            ("fp8_mxfp8", {"fp8": "hybrid", "fp8_recipe": "mxfp8"}, False),
            ("fp8_custom", {"fp8": "hybrid", "fp8_recipe": "custom"}, False),
            ("fp4_nvfp4", {"fp8": None, "fp4": "e2m1", "fp4_recipe": "nvfp4"}, False),
            ("fp4_custom", {"fp8": None, "fp4": "e2m1", "fp4_recipe": "custom"}, False),
        ]
        for grad_enabled in (True, False):
            with torch.set_grad_enabled(grad_enabled):
                for offload, origin_device in ((False, "cuda"), (True, "cpu"), (False, "cpu")):
                    for name, fields, expect_sync in cases:
                        with self.subTest(precision=name, offload=offload, device=origin_device, grad=grad_enabled):
                            self._check_case(
                                True,
                                True,
                                offload,
                                origin_device,
                                True,
                                tf_config=SimpleNamespace(**fields),
                                expect_sync=expect_sync,
                            )

    def test_sync_failure_prevents_save(self):
        """Verify a synchronization failure prevents checkpoint writes and preserves autograd."""

        class FakeDDP:
            ddp_config = SimpleNamespace(use_distributed_optimizer=True, overlap_param_gather=True)

            def start_param_sync(self, *, force_sync):
                """Simulate a failed parameter gather before checkpoint writing."""
                raise RuntimeError("parameter gather failed")

        engine = SimpleNamespace(
            module=[FakeDDP()],
            _is_offload_param=False,
            tf_config=SimpleNamespace(fp8=None, fp4=None),
            checkpoint_mananager=SimpleNamespace(
                save_checkpoint=lambda **_: self.fail("Do not write a checkpoint after failed synchronization")
            ),
        )
        grad_enabled = torch.is_grad_enabled()
        with (
            patch.dict(
                _SAVE_CHECKPOINT.__globals__,
                DistributedDataParallel=FakeDDP,
                get_megatron_module_device=lambda _: "cuda",
            ),
            patch.object(torch.distributed, "barrier") as barrier,
            self.assertRaisesRegex(RuntimeError, "parameter gather failed"),
        ):
            _SAVE_CHECKPOINT(engine, "checkpoint")
        barrier.assert_not_called()
        self.assertEqual(torch.is_grad_enabled(), grad_enabled)


if __name__ == "__main__":
    unittest.main()
