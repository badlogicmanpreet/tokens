#!/bin/bash
# TinyClaw one-time setup: create workspaces and install personas.
set -e
mkdir -p ~/.openclaw/workspace-optimist ~/.openclaw/workspace-skeptic
mkdir -p ~/.openclaw/agents/optimist/sessions ~/.openclaw/agents/skeptic/sessions
DIR="$(cd "$(dirname "$0")" && pwd)"
cp "$DIR/config/SOUL-optimist.md"   ~/.openclaw/workspace-optimist/SOUL.md
cp "$DIR/config/AGENTS-optimist.md" ~/.openclaw/workspace-optimist/AGENTS.md
cp "$DIR/config/SOUL-skeptic.md"    ~/.openclaw/workspace-skeptic/SOUL.md
cp "$DIR/config/AGENTS-skeptic.md"  ~/.openclaw/workspace-skeptic/AGENTS.md
echo "Personas installed."
if [ ! -f ~/.openclaw/openclaw.json ]; then
  cp "$DIR/config/openclaw.json.example" ~/.openclaw/openclaw.json
  TOKEN=$(openssl rand -hex 24 2>/dev/null || head -c 24 /dev/urandom | xxd -p)
  # replace the placeholder token with a fresh random one
  sed -i.bak "s/REPLACE-WITH-YOUR-OWN-RANDOM-TOKEN/$TOKEN/" ~/.openclaw/openclaw.json && rm -f ~/.openclaw/openclaw.json.bak
  echo "Installed ~/.openclaw/openclaw.json with a fresh gateway token."
else
  echo "~/.openclaw/openclaw.json already exists — merge config/openclaw.json.example manually (see TUTORIAL.md step 6)."
fi
echo "Done. Next: ./start.sh \"your debate topic\""
