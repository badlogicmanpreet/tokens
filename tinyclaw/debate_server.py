"""Debate backend — serves the UI and triggers debates via openclaw CLI.

Runs two servers:
  - Port 8420: static file server for debate.html
  - Port 8421: API server to trigger debates

Usage:
    python3 debate_server.py
"""

import http.server
import json
import os
import subprocess
import threading
import sys
from http.server import HTTPServer

UI_PORT = 8420
API_PORT = 8421
DIR = os.path.dirname(os.path.abspath(__file__))


# ---- Static file server (port 8420) ----

class StaticHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def log_message(self, format, *args):
        pass  # silence logs


# ---- API server (port 8421) ----

class APIHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        if self.path != '/start':
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(length)) if length else {}
        topic = body.get('topic', 'AI will replace most jobs within 10 years')

        message = (
            f'Debate topic: {topic}. '
            f'You MUST use the sessions_send tool with sessionKey '
            f'"agent:skeptic:main" to send your opening position. '
            f'Do not just print it — use the tool.'
        )

        # Run openclaw agent in background
        def run_agent():
            try:
                subprocess.run(
                    ['openclaw', 'agent', '--agent', 'optimist', '--message', message],
                    timeout=120,
                    capture_output=True,
                    text=True,
                )
            except Exception as e:
                print(f"Agent error: {e}", file=sys.stderr)

        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()

        self.send_response(200)
        self._cors()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"status": "started", "topic": topic}).encode())
        print(f"Debate started: {topic}")

    def _cors(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def log_message(self, format, *args):
        pass  # silence logs


def run():
    # Start static server
    static = HTTPServer(('127.0.0.1', UI_PORT), StaticHandler)
    static_thread = threading.Thread(target=static.serve_forever, daemon=True)
    static_thread.start()

    # Start API server
    api = HTTPServer(('127.0.0.1', API_PORT), APIHandler)
    api_thread = threading.Thread(target=api.serve_forever, daemon=True)
    api_thread.start()

    print(f"\033[1mTinyClaw Debate Arena\033[0m")
    print(f"  UI:  http://127.0.0.1:{UI_PORT}/debate.html")
    print(f"  API: http://127.0.0.1:{API_PORT}")
    print(f"\n\033[2mPress Ctrl+C to stop\033[0m\n")

    try:
        static_thread.join()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == '__main__':
    run()
