# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# This code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest
from unittest.mock import Mock, mock_open, patch

from verl.utils.device import check_ipc_version_support, get_npu_versions


class TestCheckIPCVersionSupport(unittest.TestCase):
    """Test cases for the check_ipc_version_support function."""

    def setUp(self):
        """Set up test logging to suppress INFO messages."""
        # Suppress INFO log messages during testing
        logging.disable(logging.INFO)

    def tearDown(self):
        """Restore logging."""
        logging.disable(logging.NOTSET)

    def test_standard_version_with_support(self):
        """Test standard version that meets minimum requirements."""
        # Software 25.5.0 >= 25.3.rc1, CANN 8.3.0 >= 8.3.rc1
        result = check_ipc_version_support("25.5.0", "8.3.0")
        self.assertTrue(result)

    def test_standard_version_newer(self):
        """Test newer standard versions."""
        # Software 26.0.0 >= 25.3.rc1, CANN 9.0.0 >= 8.3.rc1
        result = check_ipc_version_support("26.0.0", "9.0.0")
        self.assertTrue(result)

    def test_rc_version_format(self):
        """Test RC version format with additional parts."""
        # Software 25.3.rc1.2 -> 25.3.rc1 >= 25.3.rc1
        # CANN 8.3.rc1.2 -> 8.3.rc1 >= 8.3.rc1
        result = check_ipc_version_support("25.3.rc1.2", "8.3.rc1.2")
        self.assertTrue(result)

    def test_exact_rc_version(self):
        """Test exact RC version."""
        # Software 25.3.rc1 >= 25.3.rc1
        # CANN 8.3.rc1 >= 8.3.rc1
        result = check_ipc_version_support("25.3.rc1", "8.3.rc1")
        self.assertTrue(result)

    def test_t_suffix_version(self):
        """Test version with lowercase t suffix."""
        # Software 25.5.t3.b001 -> 25.5 >= 25.3.rc1
        # CANN 8.3.rc1 >= 8.3.rc1
        result = check_ipc_version_support("25.5.t3.b001", "8.3.rc1")
        self.assertTrue(result)

    def test_t_suffix_version_older(self):
        """Test version with lowercase t suffix that's too old."""
        # Software 25.5.t3.b001 -> 25.5 >= 25.3.rc1 (should pass)
        # CANN 8.2.rc1 < 8.3.rc1 (should fail)
        result = check_ipc_version_support("25.5.t3.b001", "8.2.rc1")
        self.assertFalse(result)

    def test_software_version_below_minimum(self):
        """Test software version below minimum requirement."""
        # Software 25.2.0 < 25.3.rc1
        result = check_ipc_version_support("25.2.0", "8.3.0")
        self.assertFalse(result)

    def test_cann_version_below_minimum(self):
        """Test CANN version below minimum requirement."""
        # Software 25.5.0 >= 25.3.rc1
        # CANN 8.2.0 < 8.3.rc1
        result = check_ipc_version_support("25.5.0", "8.2.0")
        self.assertFalse(result)

    def test_both_versions_below_minimum(self):
        """Test both versions below minimum requirement."""
        # Software 25.2.0 < 25.3.rc1
        # CANN 8.2.0 < 8.3.rc1
        result = check_ipc_version_support("25.2.0", "8.2.0")
        self.assertFalse(result)

    def test_invalid_software_version(self):
        """Test invalid software version format."""
        with self.assertRaises(RuntimeError) as context:
            check_ipc_version_support("invalid.version", "8.3.0")
        self.assertIn("Invalid software version format", str(context.exception))

    def test_invalid_cann_version(self):
        """Test invalid CANN version format."""
        with self.assertRaises(RuntimeError) as context:
            check_ipc_version_support("25.5.0", "invalid.version")
        self.assertIn("Invalid CANN version format", str(context.exception))

    def test_rc_with_more_parts(self):
        """Test RC version with more than 3 parts."""
        # Should extract only first 3 parts: 25.3.rc1
        result = check_ipc_version_support("25.3.rc1.2.3.4", "8.3.rc1.2.3.4")
        self.assertTrue(result)

    def test_standard_with_more_parts(self):
        """Test standard version with more than 3 parts."""
        # Should extract only first 3 parts: 25.5.0
        result = check_ipc_version_support("25.5.0.1.2.3", "8.3.0.1.2.3")
        self.assertTrue(result)

    def test_rc_edge_case_versions(self):
        """Test edge case RC versions."""
        # RC1 is the minimum
        result = check_ipc_version_support("25.3.rc1", "8.3.rc1")
        self.assertTrue(result)

        # RC0 should fail
        result = check_ipc_version_support("25.3.rc0", "8.3.rc1")
        self.assertFalse(result)

    def test_major_version_differences(self):
        """Test major version number differences."""
        # Much newer major versions
        result = check_ipc_version_support("30.0.0", "10.0.0")
        self.assertTrue(result)

        # Older major versions
        result = check_ipc_version_support("24.0.0", "7.0.0")
        self.assertFalse(result)


class TestGetNPUVersions(unittest.TestCase):
    """Test cases for the get_npu_versions function."""

    @patch("subprocess.run")
    @patch("platform.machine")
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="version=8.3.rc1\n")
    def test_get_npu_versions_success(self, mock_file, mock_exists, mock_machine, mock_run):
        """Test successful retrieval of versions."""
        # Mock npu-smi output
        mock_run.return_value = Mock(stdout="Software Version : 25.5.0\nOther Info\n", check=True)

        # Mock architecture
        mock_machine.return_value = "x86_64"

        # Mock path exists
        mock_exists.return_value = True

        software_version, cann_version = get_npu_versions()

        self.assertEqual(software_version, "25.5.0")
        self.assertEqual(cann_version, "8.3.rc1")

    @patch("subprocess.run")
    def test_get_npu_versions_missing_software_version(self, mock_run):
        """Test error when Software Version is missing."""
        mock_run.return_value = Mock(stdout="Other Info Without Software Version\n", check=True)

        with self.assertRaises(RuntimeError) as context:
            get_npu_versions()

        self.assertIn("Could not find Software Version", str(context.exception))

    @patch("subprocess.run")
    @patch("platform.machine")
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="version=8.3.rc1\n")
    def test_get_npu_versions_unsupported_architecture(self, mock_file, mock_exists, mock_machine, mock_run):
        """Test error with unsupported architecture."""
        mock_run.return_value = Mock(stdout="Software Version : 25.5.0\n", check=True)

        mock_machine.return_value = "armv7l"  # Unsupported architecture
        mock_exists.return_value = True

        with self.assertRaises(RuntimeError) as context:
            get_npu_versions()

        self.assertIn("Unsupported architecture", str(context.exception))

    @patch("subprocess.run")
    @patch("platform.machine")
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="version=8.3.rc1\n")
    def test_get_npu_versions_cann_path_not_exists(self, mock_file, mock_exists, mock_machine, mock_run):
        """Test error when CANN path doesn't exist."""
        mock_run.return_value = Mock(stdout="Software Version : 25.5.0\n", check=True)

        mock_machine.return_value = "x86_64"
        mock_exists.return_value = False  # Path doesn't exist

        with self.assertRaises(RuntimeError) as context:
            get_npu_versions()

        self.assertIn("CANN toolkit path does not exist", str(context.exception))

    @patch("subprocess.run")
    @patch("platform.machine")
    @patch("os.path.exists")
    @patch("builtins.open")
    def test_get_npu_versions_info_file_not_exists(self, mock_file, mock_exists, mock_machine, mock_run):
        """Test error when CANN info file doesn't exist."""
        mock_run.return_value = Mock(stdout="Software Version : 25.5.0\n", check=True)

        mock_machine.return_value = "x86_64"

        # First call is for CANN path exists, second call is for info file exists
        mock_exists.side_effect = [True, False]

        with self.assertRaises(RuntimeError) as context:
            get_npu_versions()

        self.assertIn("CANN toolkit info file does not exist", str(context.exception))

    @patch("subprocess.run")
    @patch("platform.machine")
    @patch("os.path.exists")
    @patch("builtins.open", new_callable=mock_open, read_data="other_info=no_version\n")
    def test_get_npu_versions_missing_cann_version(self, mock_file, mock_exists, mock_machine, mock_run):
        """Test error when CANN version is missing from info file."""
        mock_run.return_value = Mock(stdout="Software Version : 25.5.0\n", check=True)

        mock_machine.return_value = "x86_64"
        mock_exists.return_value = True

        with self.assertRaises(RuntimeError) as context:
            get_npu_versions()

        self.assertIn("Could not find version in CANN toolkit info file", str(context.exception))


if __name__ == "__main__":
    unittest.main()
