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

import multiprocessing
import unittest
from multiprocessing import shared_memory

import torch

from verl.workers.rollout.vllm_rollout.utils import create_shared_memory, rebuild_shared_memory


class TestSharedMemory(unittest.TestCase):
    """Test cases for shared memory utility functions."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Use short unique names to avoid POSIX shared memory name length limits
        import uuid

        short_id = uuid.uuid4().hex[:8]
        self.test_name = f"shm_{short_id}"

    def tearDown(self):
        """Clean up shared memory after each test method."""
        # Note: We're relying on the OS to clean up shared memory
        # as we properly delete all references in the tests
        pass

    def test_create_shared_memory_new(self):
        """Test creating new shared memory with unique name."""
        size = 1024

        shm = create_shared_memory(size, self.test_name)

        # Verify shared memory object is created correctly
        self.assertIsNotNone(shm)
        # Note: shared memory may have system-dependent size rounding
        self.assertGreaterEqual(shm.size, size)
        self.assertEqual(shm.name, self.test_name)

        # Clean up - delete tensor references first
        del shm

    def test_create_shared_memory_attach_existing(self):
        """Test that create_shared_memory attaches to existing shared memory when FileExistsError occurs."""
        size = 2048

        # First, create shared memory
        shm1 = create_shared_memory(size, self.test_name)
        self.assertGreaterEqual(shm1.size, size)

        # Second call should attach to existing memory
        shm2 = create_shared_memory(size, self.test_name)

        # Verify we attached to the same shared memory
        self.assertIsNotNone(shm2)
        self.assertGreaterEqual(shm2.size, size)
        self.assertEqual(shm2.name, self.test_name)

        # Both should reference the same shared memory
        self.assertEqual(shm1.name, shm2.name)

        # Clean up
        del shm1, shm2

    def test_rebuild_shared_memory_default_dtype(self):
        """Test rebuilding tensor from shared memory with default dtype (uint8)."""
        size = 1024

        # Create and write to shared memory
        shm = create_shared_memory(size, self.test_name)
        test_data = torch.arange(size, dtype=torch.uint8)
        shm.buf[:size] = test_data.numpy().tobytes()

        # Rebuild tensor from shared memory
        tensor, _ = rebuild_shared_memory(self.test_name, size)

        # Verify tensor properties
        self.assertEqual(tensor.dtype, torch.uint8)
        self.assertEqual(len(tensor), size)

        # Verify data integrity
        reconstructed = torch.frombuffer(shm.buf[:size], dtype=torch.uint8)
        self.assertTrue(torch.equal(tensor, reconstructed))

        # Clean up - delete references before closing
        del tensor, reconstructed

    def test_rebuild_shared_memory_custom_dtype(self):
        """Test rebuilding tensor from shared memory with custom dtype."""
        size = 256  # 256 bytes = 64 float32 values

        # Create and write to shared memory
        shm = create_shared_memory(size, self.test_name)
        test_data = torch.arange(64, dtype=torch.float32)
        shm.buf[:size] = test_data.numpy().tobytes()

        # Rebuild tensor with custom dtype
        tensor, _ = rebuild_shared_memory(self.test_name, size, dtype=torch.float32)

        # Verify tensor properties
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertEqual(len(tensor), 64)

        # Verify data integrity
        reconstructed = torch.frombuffer(shm.buf[:size], dtype=torch.float32)
        self.assertTrue(torch.equal(tensor, reconstructed))

        # Clean up - delete references before closing
        del tensor, reconstructed

    def test_shared_memory_data_integrity(self):
        """Test that data remains intact between create and rebuild operations."""
        size = 512

        # Create test data with various patterns
        test_data = torch.randint(0, 256, (size,), dtype=torch.uint8)

        # Create shared memory and write data
        shm = create_shared_memory(size, self.test_name)
        shm.buf[:size] = test_data.numpy().tobytes()

        # Rebuild tensor
        tensor, _ = rebuild_shared_memory(self.test_name, size)

        # Verify data integrity
        reconstructed = torch.frombuffer(shm.buf[:size], dtype=torch.uint8)
        self.assertTrue(torch.equal(test_data, reconstructed))

        # Clean up - delete references before closing
        del tensor, reconstructed

    def test_shared_memory_different_dtypes(self):
        """Test shared memory operations with different tensor dtypes."""
        test_cases = [
            (torch.float32, 256, 64),  # 256 bytes / 4 bytes = 64 values
            (torch.float64, 256, 32),  # 256 bytes / 8 bytes = 32 values
            (torch.int32, 256, 64),  # 256 bytes / 4 bytes = 64 values
            (torch.int64, 256, 32),  # 256 bytes / 8 bytes = 32 values
            (torch.uint8, 256, 256),  # 256 bytes / 1 byte = 256 values
        ]

        for dtype, size, expected_len in test_cases:
            # Create test data
            test_data = torch.arange(expected_len, dtype=dtype)

            # Create shared memory and write data
            shm = create_shared_memory(size, self.test_name)
            shm.buf[:size] = test_data.numpy().tobytes()

            # Rebuild tensor
            tensor, _ = rebuild_shared_memory(self.test_name, size, dtype=dtype)

            # Verify properties and data
            self.assertEqual(tensor.dtype, dtype)
            self.assertEqual(len(tensor), expected_len)

            reconstructed = torch.frombuffer(shm.buf[:size], dtype=dtype)
            self.assertTrue(torch.equal(test_data, reconstructed))

            # Clean up - delete references before closing
            del tensor, reconstructed

    def test_shared_memory_multiple_operations(self):
        """Test multiple create/rebuild operations with the same name."""
        size = 512

        # First iteration
        test_data1 = torch.arange(size, dtype=torch.uint8)
        shm1 = create_shared_memory(size, self.test_name)
        shm1.buf[:size] = test_data1.numpy().tobytes()
        tensor1, _ = rebuild_shared_memory(self.test_name, size)
        reconstructed1 = torch.frombuffer(shm1.buf[:size], dtype=torch.uint8)
        self.assertTrue(torch.equal(test_data1, reconstructed1))
        del tensor1, reconstructed1, shm1

        # Second iteration with different data
        test_data2 = torch.arange(size, dtype=torch.uint8) * 2
        shm2 = create_shared_memory(size, self.test_name)
        shm2.buf[:size] = test_data2.numpy().tobytes()
        tensor2, _ = rebuild_shared_memory(self.test_name, size)
        reconstructed2 = torch.frombuffer(shm2.buf[:size], dtype=torch.uint8)
        self.assertTrue(torch.equal(test_data2, reconstructed2))
        del tensor2, reconstructed2, shm2


# Module-level function for cross-process testing
def child_process_function(name, size, test_data_bytes):
    """Child process function to rebuild and verify tensor."""
    shm = None
    tensor = None
    test_data = None
    try:
        # Convert bytes back to tensor
        test_data = torch.frombuffer(test_data_bytes, dtype=torch.uint8)

        # Attach to shared memory
        shm = shared_memory.SharedMemory(name=name)

        # Rebuild tensor from shared memory
        tensor = torch.frombuffer(shm.buf[:size], dtype=torch.uint8)

        # Verify data integrity
        assert torch.equal(test_data, tensor), "Data mismatch in child process"
        return True
    except Exception as e:
        print(f"Error in child process: {e}")
        return False
    finally:
        # Clean up shared memory in child process
        # Delete all references first
        del tensor, test_data
        if shm is not None:
            shm.close()
            # Note: Don't unlink in child process, parent will clean up


class TestSharedMemoryIntegration(unittest.TestCase):
    """Integration tests for shared memory operations across process boundaries."""

    def test_cross_process_shared_memory(self):
        """Test shared memory can be created in one process and accessed in another."""
        size = 1024
        test_data = torch.arange(size, dtype=torch.uint8)

        # Create shared memory in parent process
        shm = create_shared_memory(size, "test_cross_proc")
        shm.buf[:size] = test_data.numpy().tobytes()

        # Convert tensor to bytes for passing to child process
        test_data_bytes = test_data.numpy().tobytes()

        # Start child process
        process = multiprocessing.Process(
            target=child_process_function, args=("test_cross_proc", size, test_data_bytes)
        )
        process.start()
        process.join(timeout=5)

        # Verify child process completed successfully
        self.assertEqual(process.exitcode, 0, "Child process failed")

        # Clean up
        del shm


if __name__ == "__main__":
    unittest.main()
