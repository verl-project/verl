#!/usr/bin/env python3
"""
Standalone entry point for InternBootcamp environment.
This script avoids relative import issues when running directly.
"""

import os
import sys

# Add the atropos root directory to Python path
atropos_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, atropos_root)

# Now import with absolute imports
from environments.intern_bootcamp.intern_bootcamp_env import (  # noqa: E402
    InternBootcampEnv,
)

if __name__ == "__main__":
    InternBootcampEnv.cli()
