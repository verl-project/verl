#!/usr/bin/env python3
"""
DynastAI Setup Script

This script ensures that all dependencies for DynastAI are properly installed
and compatible with your Python version.
"""

import os
import platform
import subprocess
import sys


def main():
    """Run the setup process"""
    print("DynastAI Setup")
    print("=============")

    # Check Python version
    python_version = tuple(map(int, platform.python_version_tuple()))
    print(f"Python version: {platform.python_version()}")

    if python_version < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)

    # Check built-in uuid module
    print("Checking UUID module...")
    try:
        import uuid

        print(
            f"UUID module version: {uuid.__version__ if hasattr(uuid, '__version__') else 'built-in'}"
        )
    except ImportError:
        print(
            "Warning: UUID module not found. This is unexpected as it should be built into Python."
        )

    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_file = os.path.join(script_dir, "requirements.txt")

    # Ensure pip is up-to-date
    print("\nUpdating pip...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install dependencies with special handling
    print("\nInstalling dependencies...")

    # First, install key packages that others might depend on
    print("Installing core dependencies...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "wheel",
            "setuptools>=68.0.0",
            "typing-extensions>=4.9.0",
        ]
    )

    # For Python 3.13+, special handling for aiohttp
    if python_version >= (3, 13):
        print("\nDetected Python 3.13+, installing compatible versions of packages...")
        try:
            # For Python 3.13, use the newest compatible version or install from source if needed
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--force-reinstall",
                    "--no-binary",
                    "aiohttp",
                    "aiohttp>=3.9.0",
                ]
            )
            print("Successfully installed aiohttp from source.")
        except Exception as e:
            print(f"Warning: Failed to install aiohttp from source: {e}")
            print("Continuing with installation, but some features might not work.")

    # Install main requirements
    print("\nInstalling main requirements...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", requirements_file]
    )

    if result.returncode != 0:
        print("\nTrying an alternative installation method for problematic packages...")

        # Read requirements file
        with open(requirements_file, "r") as f:
            requirements = [
                line.strip() for line in f if line.strip() and not line.startswith("#")
            ]

        # Install packages one by one
        for req in requirements:
            print(f"Installing {req}...")
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", req], check=True
                )
            except subprocess.CalledProcessError:
                pkg_name = (
                    req.split(">=")[0]
                    if ">=" in req
                    else req.split("==")[0] if "==" in req else req
                )
                print(
                    f"Warning: Failed to install {pkg_name}, trying without version constraint..."
                )
                subprocess.run([sys.executable, "-m", "pip", "install", pkg_name])

    print("\nSetup complete! You can now run DynastAI.")
    print("To start the game with web interface, run: python run_dynastai.py")


if __name__ == "__main__":
    main()
