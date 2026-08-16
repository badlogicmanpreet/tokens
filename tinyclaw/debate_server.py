"""Debate backend — serves the UI and triggers debates via openclaw CLI.

Runs two servers:
  - Port 8420: static file server for debate.html
  - Port 8421: API server to trigger debates

Security model (important — the agent this triggers has real tool permissions):
  - A random debate key is generated at startup and written to debate_key.txt.
    The UI (same-origin on :8420) fetches it and sends it as the X-Debate-Key
    header. Requiring a custom header forces a CORS preflight, and CORS is
    locked to the UI's exact origin — so arbitrary websites in your browser
    cannot trigger debates.
  - Topics are validated: plain string, single line, max 300 chars.

Usage:
    python3 debate_server.py
"""

import http.server
import json
import os
import re
import secrets
import subprocess
import threading
import sys
from http.server import HTTPServer

UI_PORT = 8420
API_PORT = 8421
DIR = os.path.dirname(os.path.abspath(__file__))
UI_ORIGIN = f"http://127.0.0.1:{UI_PORT}"
MAX_ROUNDS = 5

# Per-launch shared secret between the UI and this API.
DEBATE_KEY = secrets.token_hex(16)
with open(os.path.join(DIR, "debate_key.txt"), "w") as f:
    f.write(DEBATE_KEY)

_debate_lock = threading.Lock()
_debate_running = False


# ---- Static file server (port 8420) ----

class StaticHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def log_message(self, format, *args):
        pass  # silence logs


# ---- API server (port 8421) ----

def kickoff_message(topic):
    return (
        f'Debate topic: {topic}. '
        f'You MUST use the sessions_send tool with sessionKey '
        f'"agent:skeptic:main" to send your opening position. '
        f'Do not just print it — use the tool. '
        f'The debate ends after at most {MAX_ROUNDS} rounds '
        f'({MAX_ROUNDS * 2} total turns). When the limit is reached, give a '
        f'short closing statement and do NOT call sessions_send again.'
    )


class APIHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self._cors()
        self.end_headers()

    def do_POST(self):
        global _debate_running
        if self.path != '/start':
            self.send_response(404)
            self.end_headers()
            return

        if self.headers.get('X-Debate-Key', '') != DEBATE_KEY:
            self._reply(403, {"error": "bad or missing X-Debate-Key"})
            return

        try:
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length)) if length else {}
        except (ValueError, json.JSONDecodeError):
            self._reply(400, {"error": "invalid JSON body"})
            return

        topic = body.get('topic', 'AI will replace most jobs within 10 years')
        if not isinstance(topic, str):
            self._reply(400, {"error": "topic must be a string"})
            return
        topic = re.sub(r'\s+', ' ', topic).strip()[:300]
        if not topic:
            self._reply(400, {"error": "topic is empty"})
            return

        with _debate_lock:
            if _debate_running:
                self._reply(409, {"error": "a debate is already running"})
                return
            _debate_running = True

        message = kickoff_message(topic)

        def run_agent():
            global _debate_running
            try:
                result = subprocess.run(
                    ['openclaw', 'agent', '--agent', 'optimist',
                     '--message', message],
                    timeout=120,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    print(f"Agent exited {result.returncode}:\n"
                          f"{result.stderr[-2000:]}", file=sys.stderr)
            except FileNotFoundError:
                print("ERROR: 'openclaw' not found on PATH — install it first "
                      "(see TUTORIAL.md).", file=sys.stderr)
            except Exception as e:
                print(f"Agent error: {e}", file=sys.stderr)
            finally:
                with _debate_lock:
                    _debate_running = False

        threading.Thread(target=run_agent, daemon=True).start()
        self._reply(200, {"status": "started", "topic": topic})
        print(f"Debate started: {topic}")

    def _reply(self, code, obj):
        self.send_response(code)
        self._cors()
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode())

    def _cors(self):
        # Exact origin, never '*': combined with the custom-header requirement
        # this blocks cross-site POSTs from arbitrary pages.
        self.send_header('Access-Control-Allow-Origin', UI_ORIGIN)
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers',
                         'Content-Type, X-Debate-Key')

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
