"""Head-node monotonic clock calibration endpoint."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(dict(monotonic=time.monotonic())).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def start_server(host, port):
    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
