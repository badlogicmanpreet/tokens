# TinyClaw — a two-agent debate arena

Companion code for Token by Token, Chapter 6. Two agents (Optimist vs Skeptic)
debate through an OpenClaw gateway; you watch live in a browser.

## Quickstart
1. Install OpenClaw and set `ANTHROPIC_API_KEY` (see TUTORIAL.md steps 1-3).
2. `./setup.sh` — installs personas + config (once).
3. `pip3 install -r requirements.txt` (only needed for the terminal watcher).
4. `./start.sh "Your debate topic"`

The web UI asks for your gateway token (find it in `~/.openclaw/openclaw.json`),
or pass it in the URL: `http://127.0.0.1:8420/debate.html#token=YOURTOKEN`.
Terminal alternative: `OPENCLAW_GATEWAY_TOKEN=... python3 watch_debate.py`.

## Safety notes (read these)
- The agents run with real tool permissions. The debate API requires a
  per-launch key and exact-origin CORS so other websites cannot trigger runs,
  but you are still running an LLM with tools on your machine.
- Debates are capped at 5 rounds via the persona instructions; each turn is a
  full model call, so watch your API spend.
- `visibility: "all"` in the config lets agents read each other's sessions —
  a deliberate demo simplification, not a production pattern.
- `start.sh` archives (never deletes) previous sessions to /tmp.

Files: `debate_server.py` (UI + API), `debate.html` (arena UI), `start.sh`,
`setup.sh`, `watch_debate.py` (terminal watcher), `debug_ws.py` (raw event
dump for debugging), `config/` (personas + gateway config template).
