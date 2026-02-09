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

import unittest

import torch
import torch.nn as nn
from tensordict import TensorDict
from transformers import AutoModelForCausalLM, Qwen3Config

from verl import DataProto
from verl.utils.device import get_device_name
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config import FSDPActorConfig, OptimizerConfig


class MockTransformerModel(nn.Module):
    """Mock transformer model for testing DataParallelPPOActor"""

    def __init__(self, vocab_size=1000, hidden_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True), num_layers=2
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False, **kwargs):
        batch_size, seq_len = input_ids.shape

        embeddings = self.embedding(input_ids)
        hidden_states = self.transformer(embeddings)
        logits = self.lm_head(hidden_states)

        class MockOutput:
            def __init__(self, logits):
                self.logits = logits

        return MockOutput(logits)


class TestDataParallelPPOActor(unittest.TestCase):
    """Test DataParallelPPOActor compute_log_prob and update_policy methods"""

    @classmethod
    def setUpClass(cls):
        """Set up distributed environment"""
        if get_device_name() == "cuda":
            backend_name = "nccl"
        elif get_device_name() == "npu":
            backend_name = "hccl"
        else:
            backend_name = "gloo"

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend_name, init_method="env://")

        cls.rank = torch.distributed.get_rank()
        cls.world_size = torch.distributed.get_world_size()

        if get_device_name() == "cuda":
            torch.cuda.set_device(cls.rank)
            cls.device = torch.device(f"cuda:{cls.rank}")
        elif get_device_name() == "npu":
            torch.npu.set_device(cls.rank)
            cls.device = torch.device(f"npu:{cls.rank}")
        else:
            cls.device = torch.device("cpu")

    def setUp(self):
        """Set up test fixtures"""
        self.config = FSDPActorConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=4,
            ppo_micro_batch_size_per_gpu=2,
            ppo_epochs=1,
            clip_ratio=0.2,
            entropy_coeff=0.01,
            grad_clip=1.0,
            use_dynamic_bsz=False,
            use_torch_compile=False,  # Disable torch.compile for testing
            ulysses_sequence_parallel_size=1,
            optim=OptimizerConfig(lr=1e-6),
            rollout_n=1,
        )

        self.mock_model = MockTransformerModel(vocab_size=1000, hidden_size=64).to(self.device)
        self.mock_optimizer = torch.optim.Adam(self.mock_model.parameters(), lr=1e-4)

        self.actor = DataParallelPPOActor(
            config=self.config, actor_module=self.mock_model, actor_optimizer=self.mock_optimizer
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up distributed environment"""
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def _create_test_data_for_compute_log_prob(self):
        """Create test DataProto for compute_log_prob method"""
        batch_size = 2
        prompt_length = 8
        response_length = 4
        total_length = prompt_length + response_length
        vocab_size = 1000

        input_ids = torch.randint(0, vocab_size, (batch_size, total_length)).to(self.device)
        attention_mask = torch.ones(batch_size, total_length).to(self.device)
        position_ids = torch.arange(total_length).unsqueeze(0).expand(batch_size, -1).to(self.device)
        responses = input_ids[:, -response_length:]  # Last part is the response

        tensor_dict = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
            },
            batch_size=[batch_size],
        )

        meta_info = {"micro_batch_size": batch_size, "temperature": 1.0, "use_dynamic_bsz": False}

        return DataProto(batch=tensor_dict, meta_info=meta_info)

    def _create_test_data_for_update_policy(self):
        """Create test DataProto for update_policy method"""
        batch_size = 4  # Must match ppo_mini_batch_size
        prompt_length = 8
        response_length = 4
        total_length = prompt_length + response_length
        vocab_size = 1000

        input_ids = torch.randint(0, vocab_size, (batch_size, total_length)).to(self.device)
        attention_mask = torch.ones(batch_size, total_length).to(self.device)
        position_ids = torch.arange(total_length).unsqueeze(0).expand(batch_size, -1).to(self.device)
        responses = input_ids[:, -response_length:]
        response_mask = torch.ones(batch_size, response_length).to(self.device)
        old_log_probs = torch.randn(batch_size, response_length).to(self.device) * 0.1  # Small values
        advantages = torch.randn(batch_size, response_length).to(self.device) * 0.5

        tensor_dict = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
                "old_log_probs": old_log_probs,
                "advantages": advantages,
            },
            batch_size=[batch_size],
        )

        meta_info = {"temperature": 1.0}

        return DataProto(batch=tensor_dict, meta_info=meta_info)

    def test_compute_log_prob(self):
        """Test compute_log_prob method"""
        data = self._create_test_data_for_compute_log_prob()

        outputs = self.actor.compute_log_prob(data, calculate_entropy=True)
        log_probs = outputs["log_probs"]
        entropys = outputs["entropys"]

        batch_size = data.batch["responses"].shape[0]
        response_length = data.batch["responses"].shape[1]

        self.assertIsInstance(log_probs, torch.Tensor)
        self.assertEqual(log_probs.shape, (batch_size, response_length))
        self.assertTrue(torch.all(torch.isfinite(log_probs)))

        self.assertIsInstance(entropys, torch.Tensor)
        self.assertEqual(entropys.shape, (batch_size, response_length))
        self.assertTrue(torch.all(torch.isfinite(entropys)))
        self.assertTrue(torch.all(entropys >= 0))  # Entropy should be non-negative

    def test_compute_log_prob_without_entropy(self):
        """Test compute_log_prob method without entropy calculation"""
        data = self._create_test_data_for_compute_log_prob()

        outputs = self.actor.compute_log_prob(data, calculate_entropy=False)
        log_probs = outputs["log_probs"]
        entropys = outputs.get("entropys", None)

        batch_size = data.batch["responses"].shape[0]
        response_length = data.batch["responses"].shape[1]

        self.assertIsInstance(log_probs, torch.Tensor)
        self.assertEqual(log_probs.shape, (batch_size, response_length))
        self.assertTrue(torch.all(torch.isfinite(log_probs)))
        self.assertIsNone(entropys)

    def test_update_policy(self):
        """Test update_policy method"""
        data = self._create_test_data_for_update_policy()

        metrics = self.actor.update_policy(data)

        self.assertIsInstance(metrics, dict)

        expected_metric_keys = [
            "actor/pg_loss",
            "actor/pg_clipfrac",
            "actor/ppo_kl",
            "actor/pg_clipfrac_lower",
            "actor/grad_norm",
        ]

        for key in expected_metric_keys:
            self.assertIn(key, metrics)
            if isinstance(metrics[key], list):
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics[key]))
            else:
                self.assertIsInstance(metrics[key], (float, int))
                self.assertTrue(torch.isfinite(torch.tensor(metrics[key])))

    def test_dataparallelppoactor_initialization(self):
        """Test DataParallelPPOActor initialization"""
        self.assertIsNotNone(self.actor.actor_module)
        self.assertIsNotNone(self.actor.actor_optimizer)
        self.assertEqual(self.actor.config, self.config)

        self.assertEqual(self.actor.config.strategy, "fsdp2")
        self.assertEqual(self.actor.config.ppo_mini_batch_size, 4)
        self.assertEqual(self.actor.config.clip_ratio, 0.2)

    def test_dataparallelppoactor_with_qwen3_model(self):
        """Test DataParallelPPOActor with real Qwen3ForCausalLM model"""
        qwen_config = Qwen3Config(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=512,
            torch_dtype=torch.float32,
            use_cache=False,
        )

        with torch.device(self.device):
            qwen_model = AutoModelForCausalLM.from_config(config=qwen_config, torch_dtype=torch.float32).to(self.device)

        qwen_optimizer = torch.optim.Adam(qwen_model.parameters(), lr=1e-4)

        qwen_actor = DataParallelPPOActor(config=self.config, actor_module=qwen_model, actor_optimizer=qwen_optimizer)

        data = self._create_test_data_for_compute_log_prob()
        outputs = qwen_actor.compute_log_prob(data, calculate_entropy=True)
        log_probs = outputs["log_probs"]
        entropys = outputs["entropys"]

        batch_size = data.batch["responses"].shape[0]
        response_length = data.batch["responses"].shape[1]

        self.assertIsInstance(log_probs, torch.Tensor)
        self.assertEqual(log_probs.shape, (batch_size, response_length))
        self.assertTrue(torch.all(torch.isfinite(log_probs)))

        self.assertIsInstance(entropys, torch.Tensor)
        self.assertEqual(entropys.shape, (batch_size, response_length))
        self.assertTrue(torch.all(torch.isfinite(entropys)))
        self.assertTrue(torch.all(entropys >= 0))

        policy_data = self._create_test_data_for_update_policy()
        metrics = qwen_actor.update_policy(policy_data)

        self.assertIsInstance(metrics, dict)

        expected_metric_keys = [
            "actor/pg_loss",
            "actor/pg_clipfrac",
            "actor/ppo_kl",
            "actor/pg_clipfrac_lower",
            "actor/grad_norm",
        ]

        for key in expected_metric_keys:
            self.assertIn(key, metrics)
            if isinstance(metrics[key], list):
                self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in metrics[key]))
            else:
                self.assertIsInstance(metrics[key], (float, int))
                self.assertTrue(torch.isfinite(torch.tensor(metrics[key])))


if __name__ == "__main__":
    unittest.main()
