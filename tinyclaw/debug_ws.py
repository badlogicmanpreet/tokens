"""Debug: connect to Gateway and print ALL events raw."""

import asyncio
import json
import websockets

GATEWAY_URL = "ws://127.0.0.1:18789/ws"
import os, sys
TOKEN = os.environ.get("OPENCLAW_GATEWAY_TOKEN", "")
if not TOKEN:
    print("Set OPENCLAW_GATEWAY_TOKEN (see the token in ~/.openclaw/openclaw.json)")
    sys.exit(1)

async def debug():
    async with websockets.connect(GATEWAY_URL) as ws:
        await ws.send(json.dumps({
            "type": "req", "id": "1", "method": "connect",
            "params": {
                "minProtocol": 3, "maxProtocol": 3,
                "client": {"id": "cli", "version": "0.1.0", "platform": "python", "mode": "cli"},
                "auth": {"token": TOKEN},
            }
        }))

        print("Connected. Listening for ALL messages...\n")

        async for raw in ws:
            msg = json.loads(raw)
            t = msg.get("type", "")
            if t == "event":
                evt = msg.get("event", "")
                payload = msg.get("payload", {})
                if evt == "agent":
                    stream = payload.get("stream", "")
                    data = payload.get("data", {})
                    sk = payload.get("sessionKey", "")
                    # Print a summary
                    if stream == "assistant":
                        text = data.get("text", "") or data.get("delta", "")
                        print(f"[agent/{stream}] session={sk} text={text[:80]}...")
                    elif stream == "lifecycle":
                        print(f"[agent/{stream}] session={sk} data={data}")
                    elif stream == "tool":
                        print(f"[agent/{stream}] session={sk} data={data}")
                    else:
                        print(f"[agent/{stream}] session={sk} keys={list(data.keys())}")
                elif evt in ("tick", "health", "presence"):
                    pass  # skip noise
                else:
                    print(f"[event:{evt}] {json.dumps(payload)[:120]}")
            elif t == "res":
                print(f"[res] id={msg.get('id')} ok={msg.get('ok')} error={msg.get('error', '')}")

asyncio.run(debug())
