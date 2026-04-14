#!/usr/bin/env python3
"""
DynastAI Quick Test Script

This script provides a simple way to test the DynastAI environment without requiring the full Atropos setup.
It will start a local server and open the web UI in your default browser.
"""

import argparse
import os
import subprocess
import sys
import time
import webbrowser

# Ensure the script works from any directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Add the parent directory to sys.path to allow importing atroposlib if available
parent_dir = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run DynastAI test environment")
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't open browser automatically"
    )
    parser.add_argument(
        "--api-port", type=int, default=9001, help="API port (default: 9001)"
    )
    parser.add_argument(
        "--web-port", type=int, default=3000, help="Web UI port (default: 3000)"
    )
    args = parser.parse_args()

    # Install dependencies if needed
    try:
        import fastapi  # noqa: F401
        import pydantic  # noqa: F401
        import uvicorn  # noqa: F401

        print("Dependencies already installed.")
    except ImportError:
        print("Installing dependencies...")
        try:
            # First try using the setup script
            if os.path.exists(os.path.join(SCRIPT_DIR, "setup.py")):
                print("Running setup script...")
                subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, "setup.py")])
            else:
                # Fall back to direct installation
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
                )
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
                )
        except Exception as e:
            print(f"Warning: Error installing dependencies: {e}")
            print("Please run 'python setup.py' manually before continuing.")

    # Create data directory if it doesn't exist
    os.makedirs(os.path.join(SCRIPT_DIR, "src/data"), exist_ok=True)

    # Start the local server
    print(f"Starting DynastAI server on http://localhost:{args.web_port}")
    server_process = subprocess.Popen(
        [
            sys.executable,
            os.path.join(SCRIPT_DIR, "dynastai_local_server.py"),
            "--api-port",
            str(args.api_port),
            "--web-port",
            str(args.web_port),
        ]
    )

    try:
        # Give the server time to start
        time.sleep(2)

        # Open the browser if requested
        if not args.no_browser:
            print("Opening web browser...")
            webbrowser.open(f"http://localhost:{args.web_port}")

        print("Press Ctrl+C to stop the server")
        server_process.wait()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server_process.terminate()
        server_process.wait()
        print("Server stopped.")


if __name__ == "__main__":
    main()
