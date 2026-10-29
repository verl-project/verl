# Copyright 2025 Bytedance Ltd. and/or its affiliates
"""CPU tests for asymmetric teacher system-prompt OPD helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from verl.experimental.teacher_loop.teacher_manager import (
    align_teacher_outputs_to_student,
    inject_system_message,
    resolve_teacher_system_prompt,
)
from verl.workers.config.distillation import DistillationTeacherModelConfig
from verl.workers.config.rollout import RolloutConfig


class TeacherSystemPromptTests(unittest.TestCase):
    def test_inject_system_message_prepends_when_missing(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = inject_system_message(msgs, "SYS")
        self.assertEqual(out[0], {"role": "system", "content": "SYS"})
        self.assertEqual(out[1], {"role": "user", "content": "hi"})
        self.assertEqual(msgs[0]["role"], "user")

    def test_inject_system_message_replaces_existing_system(self):
        msgs = [{"role": "system", "content": "old"}, {"role": "user", "content": "hi"}]
        out = inject_system_message(msgs, "NEW")
        self.assertEqual(out[0]["content"], "NEW")
        self.assertEqual(msgs[0]["content"], "old")

    def test_resolve_teacher_system_prompt_inline_and_path(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = DistillationTeacherModelConfig(
                key="default",
                model_path="dummy",
                num_replicas=1,
                inference=RolloutConfig(),
                system_prompt="  inline  ",
                system_prompt_path=str(Path(td) / "ignored.txt"),
            )
            self.assertEqual(resolve_teacher_system_prompt(cfg), "  inline  ")

            path = Path(td) / "sys.txt"
            path.write_text("from-file", encoding="utf-8")
            cfg2 = DistillationTeacherModelConfig(
                key="default",
                model_path="dummy",
                num_replicas=1,
                inference=RolloutConfig(),
                system_prompt=None,
                system_prompt_path=str(path),
            )
            self.assertEqual(resolve_teacher_system_prompt(cfg2), "from-file")

        cfg3 = DistillationTeacherModelConfig(
            key="default",
            model_path="dummy",
            num_replicas=1,
            inference=RolloutConfig(),
        )
        self.assertIsNone(resolve_teacher_system_prompt(cfg3))

    def test_align_identity_passthrough(self):
        n, m = 4, 3
        ids = torch.arange(n + m, dtype=torch.int32).unsqueeze(-1).expand(-1, 2).contiguous()
        lps = torch.arange(n + m, dtype=torch.float32).unsqueeze(-1).expand(-1, 2).contiguous() * 0.1
        out_ids, out_lps = align_teacher_outputs_to_student(
            ids,
            lps,
            student_prompt_len=n,
            student_response_len=m,
            teacher_prompt_len=n,
            pad_token_id=0,
        )
        self.assertTrue(torch.equal(out_ids, ids))
        self.assertTrue(torch.equal(out_lps, lps))

    def test_align_asymmetric_preserves_left_shift_response_window(self):
        n, m, t = 5, 4, 9
        teacher_ids = torch.arange(t + m, dtype=torch.int32).unsqueeze(-1)
        teacher_lps = (torch.arange(t + m, dtype=torch.float32) + 100.0).unsqueeze(-1)

        aligned_ids, aligned_lps = align_teacher_outputs_to_student(
            teacher_ids,
            teacher_lps,
            student_prompt_len=n,
            student_response_len=m,
            teacher_prompt_len=t,
            pad_token_id=7,
        )
        self.assertEqual(aligned_ids.shape[0], n + m)
        self.assertEqual(aligned_lps.shape[0], n + m)
        self.assertTrue(torch.equal(aligned_lps[n - 1 : n + m - 1], teacher_lps[t - 1 : t + m - 1]))
        self.assertTrue(torch.equal(aligned_ids[n - 1 : n + m - 1], teacher_ids[t - 1 : t + m - 1]))
        self.assertTrue(torch.all(aligned_ids[: n - 1] == 7))
        self.assertTrue(torch.all(aligned_lps[: n - 1] == 0))

    def test_align_rejects_length_mismatch(self):
        ids = torch.zeros(10, 1, dtype=torch.int32)
        lps = torch.zeros(10, 1)
        with self.assertRaisesRegex(ValueError, "teacher output length"):
            align_teacher_outputs_to_student(
                ids,
                lps,
                student_prompt_len=3,
                student_response_len=4,
                teacher_prompt_len=5,
                pad_token_id=0,
            )

    def test_validate_and_prepare_reserves_system_budget(self):
        inf = RolloutConfig(
            name="vllm",
            prompt_length=1024,
            response_length=4096,
            max_model_len=8192,
            engine_kwargs={"vllm": {}},
        )
        cfg = DistillationTeacherModelConfig(
            key="default",
            model_path="dummy",
            num_replicas=1,
            inference=inf,
            system_prompt="x" * 2000,
        )
        cfg.validate_and_prepare_for_distillation(use_topk=False, topk=None)
        self.assertEqual(cfg.inference.response_length, 1)
        self.assertGreaterEqual(cfg.inference.prompt_length, 1024 + 4096)
        self.assertLessEqual(cfg.inference.prompt_length, 8191)


if __name__ == "__main__":
    unittest.main()
