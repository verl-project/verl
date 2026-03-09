"""
Mock Atropos server - dynamic scores that improve over time
"""
import json
import uuid
import random
from http.server import HTTPServer, BaseHTTPRequestHandler

step = [0]

class MockAtroposHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/register':
            length = int(self.headers['Content-Length'])
            data = json.loads(self.rfile.read(length))
            print(f"\n[MOCK] Registered: {data}")
            resp = {'uuid': str(uuid.uuid4()), 'status': 'ok'}
            self._respond(resp)

    def do_GET(self):
        if self.path == '/batch':
            # Scores improve over time simulating learning
            base = min(0.9, 0.2 + step[0] * 0.05)
            scores = [round(base + random.uniform(-0.1, 0.1), 3) for _ in range(4)]
            step[0] += 1
            batch = {
                'batch': {
                    'tokens': [[1,2,3,4,5]] * 4,
                    'masks': [[1,1,1,1,1]] * 4,
                    'scores': scores,
                }
            }
            print(f"[MOCK] Step {step[0]} batch scores: {scores}")
            self._respond(batch)

    def _respond(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass

if __name__ == '__main__':
    server = HTTPServer(('localhost', 8000), MockAtroposHandler)
    print("[MOCK] Atropos server running on localhost:8000")
    server.serve_forever()
