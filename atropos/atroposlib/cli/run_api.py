"""
Run the Trajectory API server.
"""

import argparse

import uvicorn


def main():
    """
    Run the API server.
    Args:
        host: The host to run the API server on.
        port: The port to run the API server on.
        reload: Whether to reload the API server on code changes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "atroposlib.api:app", host=args.host, port=args.port, reload=args.reload
    )


if __name__ == "__main__":
    main()
