#!/bin/bash
# TinyClaw Debate Arena — one command to run everything
#
# Usage: ./start.sh [topic]
#   Example: ./start.sh "AI is overhyped"
#   Default topic: "Open source AI will overtake closed-source AI within 5 years"

set -e

PORT=18789
DIR="$(cd "$(dirname "$0")" && pwd)"
TOPIC="${1:-Open source AI will overtake closed-source AI within 5 years}"

GREEN='\033[0;92m'
DIM='\033[2m'
RESET='\033[0m'
BOLD='\033[1m'

cleanup() {
  echo -e "\n${DIM}Shutting down...${RESET}"
  kill $GATEWAY_PID 2>/dev/null || true
  kill $SERVER_PID 2>/dev/null || true
  exit 0
}
trap cleanup INT TERM

# 1. Kill any stale gateway
if lsof -ti:$PORT > /dev/null 2>&1; then
  echo -e "${DIM}Killing existing process on port $PORT...${RESET}"
  kill $(lsof -ti:$PORT) 2>/dev/null || true
  sleep 2
fi

# 2. Start gateway
echo -e "${BOLD}Starting OpenClaw gateway...${RESET}"
openclaw gateway --port $PORT > /tmp/openclaw-gateway.log 2>&1 &
GATEWAY_PID=$!

echo -n "  Waiting"
for i in $(seq 1 30); do
  if curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
    echo -e " ${GREEN}ready${RESET}"
    break
  fi
  echo -n "."
  sleep 1
done

if ! curl -s http://127.0.0.1:$PORT/health > /dev/null 2>&1; then
  echo -e "\n  Failed. Check /tmp/openclaw-gateway.log"
  exit 1
fi

# 3. Reset old sessions so debate starts fresh
rm -f ~/.openclaw/agents/optimist/sessions/*.jsonl 2>/dev/null
rm -f ~/.openclaw/agents/skeptic/sessions/*.jsonl 2>/dev/null

# 4. Start debate server (UI + API)
echo -e "${BOLD}Starting debate server...${RESET}"
python3 "$DIR/debate_server.py" &
SERVER_PID=$!
sleep 1

# 5. Open browser
echo -e "${GREEN}${BOLD}Opening debate arena...${RESET}"
open "http://127.0.0.1:8420/debate.html" 2>/dev/null || \
  xdg-open "http://127.0.0.1:8420/debate.html" 2>/dev/null || \
  echo "Open http://127.0.0.1:8420/debate.html in your browser"

# 6. Kick off the debate
echo -e "\n${BOLD}Topic: ${TOPIC}${RESET}"
echo -e "${DIM}Launching debate...${RESET}\n"

sleep 2

openclaw agent --agent optimist \
  --message "Debate topic: ${TOPIC}. You MUST use the sessions_send tool with sessionKey \"agent:skeptic:main\" to send your opening position. Do not just print it — use the tool." \
  > /tmp/openclaw-debate.log 2>&1 &

echo -e "${GREEN}Debate started! Watch it in the browser.${RESET}"
echo -e "  UI:      http://127.0.0.1:8420/debate.html"
echo -e "  Logs:    /tmp/openclaw-gateway.log"
echo -e "  Debate:  /tmp/openclaw-debate.log"
echo ""
echo -e "${DIM}Press Ctrl+C to stop everything${RESET}"

wait $GATEWAY_PID
