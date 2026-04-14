import asyncio
import http.server
import json
import os
import socketserver
import threading

import websockets


# Temporary websocket handler without pybullet dependency
async def visualization_websocket_handler(websocket):
    print(f"Client connected from {websocket.remote_address}")
    try:
        # Send a test scene
        test_scene = [
            {
                "id": "test_cube",
                "type": "cube",
                "position": [0, 0, 0],
                "orientation_quaternion": [0, 0, 0, 1],
                "scale": [1, 1, 1],
                "color_rgba": [1, 0, 0, 1],
            }
        ]
        await websocket.send(
            json.dumps({"type": "initial_scene", "payload": test_scene})
        )

        async for message in websocket:
            print(f"Received message: {message}")

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in websocket handler: {e}")


def run_http_server():
    # Change to the visualization directory
    os.chdir(os.path.join(os.path.dirname(__file__), "visualization"))

    # Create an HTTP server
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", 8080), Handler) as httpd:
        print("HTTP Server running on http://localhost:8080")
        httpd.serve_forever()


async def main():
    # Start WebSocket server
    async with websockets.serve(visualization_websocket_handler, "localhost", 8765):
        print("WebSocket Server running on ws://localhost:8765")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    # Start HTTP server in a separate thread
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()

    # Run WebSocket server in the main thread
    asyncio.run(main())
