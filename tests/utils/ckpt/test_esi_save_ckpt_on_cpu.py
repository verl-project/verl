# Copyright 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
from datetime import datetime, timedelta
from unittest import TestCase

from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi


class TestShouldSaveCkptEsi(TestCase):
    def test_no_expiration_timestamp(self):
        """Test case when no expiration timestamp is set"""
        os.environ.pop("MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP", None)
        os.environ.pop("SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP", None)
        self.assertFalse(should_save_ckpt_esi(100))

    def test_mlp_expiration_valid(self):
        """Test valid MLP expiration timestamp requiring save"""
        current_time = time.time()
        os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(current_time + 90)
        self.assertTrue(should_save_ckpt_esi(30))  # max_steps_duration=30 seconds

    def test_mlp_expiration_passed(self):
        """Test expired MLP timestamp"""
        current_time = time.time()
        os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(current_time - 10)
        self.assertFalse(should_save_ckpt_esi(30))

    def test_mlp_invalid_timestamp(self):
        """Test invalid MLP timestamp format"""
        os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = "invalid"
        self.assertFalse(should_save_ckpt_esi(30))

    def test_mlp_expiration_not_reached(self):
        """Test MLP expiration timestamp with insufficient remaining time"""
        current_time = time.time()
        os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(current_time + 200)
        self.assertFalse(should_save_ckpt_esi(30))  # max_steps_duration=30

    def test_aws_expiration_not_reached(self):
        """Test AWS expiration timestamp with sufficient remaining time"""
        now = datetime.now()
        expiration = now + timedelta(minutes=100)  # Exceeds 90-minute threshold
        os.environ["SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(int(expiration.timestamp()))
        self.assertFalse(should_save_ckpt_esi(30 * 60))

    def test_redundant_time(self):
        """Test redundant_time parameter effect"""
        current_time = time.time()
        # Total required: 60+30+30=120 seconds
        os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(current_time + 120)
        self.assertTrue(should_save_ckpt_esi(30, redundant_time=30))

    def test_zero_max_steps_duration(self):
        """Test zero max_steps_duration"""
        current_time = time.time()
        os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(current_time + 60)
        self.assertFalse(should_save_ckpt_esi(0))
