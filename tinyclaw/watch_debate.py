"""Watch two OpenClaw agents debate in real time.

Connects to the Gateway WebSocket and streams agent events
as they happen — you see each agent's arguments appear live.

Usage:
    python3 watch_debate.py

Requires: pip3 install websockets
"""

import asyncio
import json
import os
import sys

try:
    import websockets
except ImportError:
    print("Install websockets: pip3 install websockets")
    sys.exit(1)

GATEWAY_URL = os.environ.get("OPENCLAW_GATEWAY_URL", "ws://127.0.0.1:18789/ws")
GATEWAY_TOKEN = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")
if not GATEWAY_TOKEN:
    print("Set OPENCLAW_GATEWAY_TOKEN (see the token in ~/.openclaw/openclaw.json)")
    sys.exit(1)

AGENT_COLORS = {
    "optimist": "\033[92m",  # green
    "skeptic": "\033[91m",   # red
}
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def agent_from_session_key(key: str) -> str:
    """Extract agent id from session key like 'agent:skeptic:main'."""
    if not key:
        return "unknown"
    parts = key.split(":")
    if len(parts) >= 2 and parts[0] == "agent":
        return parts[1]
    return key


def colorize(agent_id: str, text: str) -> str:
    color = AGENT_COLORS.get(agent_id, "\033[97m")
    return f"{color}{BOLD}[{agent_id}]{RESET} {text}"


async def watch():
    print(f"{BOLD}Connecting to Gateway at {GATEWAY_URL}...{RESET}\n")

    async with websockets.connect(GATEWAY_URL) as ws:
        # Handshake — Gateway requires protocol version + client identity
        connect_msg = {
            "type": "req",
            "id": "1",
            "method": "connect",
            "params": {
                "minProtocol": 3,
                "maxProtocol": 3,
                "client": {
                    "id": "cli",
                    "version": "0.1.0",
                    "platform": "python",
                    "mode": "cli",
                },
                "auth": {"token": GATEWAY_TOKEN},
            },
        }
        await ws.send(json.dumps(connect_msg))
        # read until the connect response arrives (the gateway may emit
        # unrelated events before it)
        while True:
            hello = json.loads(await ws.recv())
            if hello.get("type") == "res" and str(hello.get("id")) == "1":
                break

        if hello.get("error"):
            print(f"Connection failed: {hello['error']}")
            return

        print(f"{DIM}Connected. Waiting for debate events...{RESET}\n")
        print("=" * 60)

        printed = {}  # runId -> chars already printed
        async for raw in ws:
            msg = json.loads(raw)

            if msg.get("type") != "event":
                continue

            event_name = msg.get("event", "")
            payload = msg.get("payload", {})

            if event_name != "agent":
                continue

            # Agent events have: stream, data, sessionKey, runId
            stream = payload.get("stream", "")
            data = payload.get("data", {})
            session_key = payload.get("sessionKey", "")
            agent_id = agent_from_session_key(session_key)

            if stream == "assistant":
                # data.text is CUMULATIVE (full response so far), so print
                # only the new suffix since the last frame for this run.
                run_id = payload.get("runId", session_key)
                text = data.get("text", "")
                if text:
                    prev = printed.get(run_id, 0)
                    if len(text) > prev:
                        print(colorize(agent_id, text[prev:]), end="", flush=True)
                        printed[run_id] = len(text)
                else:
                    delta = data.get("delta", "")
                    if delta:
                        print(colorize(agent_id, delta), end="", flush=True)

            elif stream == "tool":
                tool_name = data.get("name", "") or data.get("tool", "")
                if tool_name:
                    print(f"{DIM}  [{agent_id}] tool: {tool_name}{RESET}")

            elif stream == "lifecycle":
                phase = data.get("phase", "")
                if phase == "end":
                    print()  # finish the streamed line
                if phase == "start":
                    print(f"\n{DIM}--- {agent_id} thinking... ---{RESET}")
                elif phase == "end":
                    print(f"{DIM}--- {agent_id} done ---{RESET}\n")


if __name__ == "__main__":
    try:
        asyncio.run(watch())
    except KeyboardInterrupt:
        print(f"\n{DIM}Debate ended.{RESET}")
