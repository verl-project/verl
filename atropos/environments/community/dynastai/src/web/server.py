# FastAPI server for DynastAI game
"""
This module initializes and runs the FastAPI server for the DynastAI game.
"""

import os
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import api as dynastai_api


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""

    app = FastAPI(
        title="DynastAI Server",
        description="DynastAI game server with FastAPI",
        version="1.0.0",
    )

    # Add CORS middleware to allow requests from the web frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins in development
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount the DynastAI API
    app.mount("/api", dynastai_api)

    # Determine the static files directory
    static_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "web", "static"
    )
    if os.path.exists(static_dir):
        # Mount static files if directory exists
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def root():
        """Root endpoint that redirects to static/index.html if available"""
        return {
            "message": "DynastAI Server running. Access the API at /api or web UI at /static/index.html"
        }

    return app


def run_server(host: str = "0.0.0.0", port: int = 9001):
    """Run the FastAPI server"""
    app = create_app()

    # Run with uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    # Default port
    port = 9001

    # Override port from command line if provided
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}, using default port {port}")

    run_server(port=port)
