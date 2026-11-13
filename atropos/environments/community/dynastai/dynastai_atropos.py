#!/usr/bin/env python3
"""
DynastAI Environment - Direct entry point for Atropos integration

This script provides a direct entry point for running the DynastAI environment
with the Atropos framework. It serves as a simple wrapper around the dynastai_environment.py
script in the parent directory.
"""

import os
import sys

# Add parent directory to path to find dynastai_environment.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the main environment module
import environments.dynastai_environment as dynastai_environment  # noqa: E402

if __name__ == "__main__":
    # Pass all arguments to the main entry point
    sys.exit(dynastai_environment.__file__)
