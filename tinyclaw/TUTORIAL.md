# TinyClaw — Building an Agent-to-Agent Debate System

How two AI agents argue with each other, in real time, using OpenClaw's
multi-agent coordination. This is your first step into agentic AI — where
models don't just answer questions, they take actions, use tools, and
talk to each other.

---

## Table of Contents

1. [What Are Agents?](#what-are-agents)
2. [From LLMs to Agents — The Key Shift](#from-llms-to-agents)
3. [OpenClaw — The Platform](#openclaw)
4. [The Experiment: Two Agents Debate](#the-experiment)
5. [Architecture](#architecture)
6. [Setup — Step by Step](#setup)
7. [Code Walkthrough](#code-walkthrough)
8. [Running the Debate](#running-the-debate)
9. [What Actually Happened — Debate Analysis](#debate-analysis)
10. [What We Learned](#what-we-learned)

---

## What Are Agents?

An LLM by itself is a function: text in, text out. An **agent** is an LLM
that can act — it has a loop, tools, memory, and the ability to decide
what to do next.

```
LLM:     prompt → response (done)

Agent:   prompt → think → use tool → observe result → think → use another tool → respond
```

The critical difference is the **loop**. An agent doesn't just predict the
next token — it decides whether to call a tool, read a file, send a message
to another agent, or deliver a final response. Each cycle through the loop
is a deliberate action.

Three things make an LLM into an agent:

| Component | What it does | Example |
|-----------|-------------|---------|
| **Tools** | Actions the agent can take | Run shell commands, browse the web, send messages |
| **Memory** | Persistent state across turns | Conversation history, workspace files |
| **Persona** | Instructions that shape behavior | "You are the Skeptic. Argue against the proposition." |

---

## From LLMs to Agents — The Key Shift

In earlier chapters, we built an LLM from scratch and made it reason about
math. That was **single-turn inference** — one question, one answer. Agents
are fundamentally different:

```
Single-turn:   Human → LLM → Answer

Multi-turn:    Human → Agent → [Tool Call] → [Observe] → [Tool Call] → Answer

Multi-agent:   Human → Agent A → [sends message to Agent B] → Agent B responds
                                → Agent A rebuts → Agent B rebuts → ... → Final answer
```

The progression:

1. **Chapter 1-2**: Build an LLM that generates text
2. **Chapter 3-5**: Make it reason (CoT, voting, self-refinement)
3. **Chapter 6**: Train it with RL (GRPO)
4. **This chapter**: Give it tools and let it coordinate with other agents

This is where AI moves from "answering questions" to "doing things."

---

## OpenClaw

[OpenClaw](https://github.com/openclaw/openclaw) is an open-source personal
AI assistant platform. We use it here not to build a personal assistant, but
because it provides the exact infrastructure agents need:

```
┌─────────────────────────────────────────────────┐
│  GATEWAY (WebSocket server on localhost:18789)   │
│  Routes messages, manages sessions, serves API   │
└────────────────────┬────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼────┐  ┌────▼────┐  ┌───▼───┐
   │ Agent A  │  │ Agent B  │  │  CLI  │
   │ optimist │  │ skeptic  │  │  you  │
   └──────────┘  └──────────┘  └───────┘
```

**Gateway** — The brain. A WebSocket server that routes messages between
agents, manages conversation sessions, and broadcasts events. Think of it
as a message bus.

**Agents** — Each agent is an LLM with its own workspace (persona files,
instructions), its own session history, and access to tools. Agents are
isolated — they can't see each other's sessions unless explicitly allowed.

**Tools** — Actions agents can take. The critical one for our experiment:
`sessions_send`, which lets Agent A send a message into Agent B's session.

**Sessions** — Persistent conversation history stored as JSONL files. Each
agent has its own session. When the gateway resets (daily at 4 AM by default),
sessions start fresh.

---

## The Experiment

Two agents with opposing personas debate a topic. The Optimist argues FOR,
the Skeptic argues AGAINST. They communicate via `sessions_send` — each
agent sends its argument to the other's session, triggering a response.

We watch the debate in real time through a web UI that connects to the
Gateway's WebSocket event stream.

```
You type a topic
       │
       ▼
┌──────────────┐  sessions_send   ┌──────────────┐
│   OPTIMIST   │ ───────────────► │   SKEPTIC    │
│              │                  │              │
│  "AI will    │ ◄─────────────── │  "No it     │
│   change     │   ping-pong      │   won't,    │
│   everything"│   (up to 5       │   here's    │
│              │    rounds)       │   why..."   │
└──────────────┘                  └──────────────┘
       │
       ▼
  Web UI streams both sides live
```

---

## Architecture

### Project Structure

```
tinyclaw/
├── TUTORIAL.md           # This file
├── start.sh              # One command to launch everything
├── debate.html           # Web UI — watches debate in real time
├── debate_server.py      # Backend — serves UI + triggers debates via CLI
├── watch_debate.py       # Terminal watcher (alternative to web UI)
└── debug_ws.py           # Debug tool — prints raw Gateway events
```

### What `start.sh` Does

```
1. Kill any stale gateway process
2. Start OpenClaw gateway in background
3. Clear old session files (fresh debate)
4. Start debate_server.py (HTTP server for UI + API)
5. Open browser to debate.html
6. Trigger: openclaw agent --agent optimist --message "Debate topic: ..."
```

### How the Debate Flows

```
start.sh
  └─► openclaw agent --agent optimist --message "Debate: ..."
        │
        ▼
      Gateway receives message, routes to optimist's session
        │
        ▼
      Optimist LLM runs, generates opening argument
      Optimist calls tool: sessions_send(sessionKey="agent:skeptic:main", message=...)
        │
        ▼
      Gateway routes message to skeptic's session
      Skeptic LLM runs, generates rebuttal
      Skeptic calls tool: sessions_send(sessionKey="agent:optimist:main", message=...)
        │
        ▼
      Ping-pong continues (up to 5 rounds by default)
        │
        ▼
      debate.html receives all events via WebSocket, renders live
```

### The WebSocket Event Protocol

The web UI connects to `ws://127.0.0.1:18789/ws` and receives events
as JSON frames. The handshake requires protocol version and client identity:

```json
{
  "type": "req",
  "id": "1",
  "method": "connect",
  "params": {
    "minProtocol": 3,
    "maxProtocol": 3,
    "client": {
      "id": "webchat",
      "version": "0.1.0",
      "platform": "browser",
      "mode": "webchat"
    },
    "auth": { "token": "your-gateway-token" }
  }
}
```

After connecting, agent events arrive as:

```json
{
  "type": "event",
  "event": "agent",
  "payload": {
    "runId": "ab6f7a20-...",
    "stream": "assistant",
    "sessionKey": "agent:optimist:main",
    "data": { "text": "The full accumulated response text..." }
  }
}
```

Key insight: **`data.text` is cumulative, not a delta.** Each event contains
the full response so far. The UI replaces the displayed text on each event,
not appends.

Three event streams matter:

| Stream | When | Data |
|--------|------|------|
| `lifecycle` | Agent starts/finishes thinking | `{ phase: "start" }` or `{ phase: "end" }` |
| `assistant` | Agent generates text | `{ text: "full response so far..." }` |
| `tool` | Agent calls a tool | `{ name: "sessions_send" }` |

The agent ID is extracted from the session key: `agent:skeptic:main` → `skeptic`.

---

## Setup — Step by Step

### Prerequisites

- Node.js 22+ (`node --version`)
- Python 3.9+ (`python3 --version`)
- An Anthropic API key

### 1. Install OpenClaw

```bash
npm install -g openclaw@latest
```

### 2. Initialize

```bash
openclaw onboard --mode local --non-interactive --accept-risk \
  --auth-choice skip --skip-channels --skip-skills \
  --skip-daemon --skip-health --skip-search --skip-ui \
  --no-install-daemon
```

This creates `~/.openclaw/openclaw.json` and the default workspace.

### 3. Set Your API Key

Add to your `~/.profile` or `~/.zshrc`:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Then `source ~/.profile`.

### 4. Create Agent Workspaces

```bash
mkdir -p ~/.openclaw/workspace-optimist ~/.openclaw/workspace-skeptic
mkdir -p ~/.openclaw/agents/optimist/sessions ~/.openclaw/agents/skeptic/sessions
```

### 5. Write Agent Personas

**`~/.openclaw/workspace-optimist/SOUL.md`**:
```
You are the Optimist. Argue FOR the proposition.
Be concise — 1 short paragraph per turn. Stay respectful.
```

**`~/.openclaw/workspace-skeptic/SOUL.md`**:
```
You are the Skeptic. Argue AGAINST the proposition.
Be concise — 1 short paragraph per turn. Stay respectful.
```

**`~/.openclaw/workspace-optimist/AGENTS.md`**:
```
You are in a two-agent debate system.
Your debate partner is the "skeptic" agent. To send your argument to them,
use the sessions_send tool with:
- sessionKey: "agent:skeptic:main"
- message: your argument
When you receive a debate topic, state your opening position and immediately
send it to the skeptic using sessions_send.
```

**`~/.openclaw/workspace-skeptic/AGENTS.md`**:
```
You are in a two-agent debate system.
Your debate partner is the "optimist" agent. To send your rebuttal back,
use the sessions_send tool with:
- sessionKey: "agent:optimist:main"
- message: your rebuttal
When you receive an argument, deliver a sharp rebuttal and send it back
to the optimist using sessions_send.
```

### 6. Configure Multi-Agent Setup

Edit `~/.openclaw/openclaw.json`:

```json
{
  "agents": {
    "defaults": {
      "workspace": "~/.openclaw/workspace",
      "model": "anthropic/claude-sonnet-4-6"
    },
    "list": [
      {
        "id": "optimist",
        "workspace": "~/.openclaw/workspace-optimist"
      },
      {
        "id": "skeptic",
        "workspace": "~/.openclaw/workspace-skeptic"
      }
    ]
  },
  "tools": {
    "sessions": { "visibility": "all" },
    "agentToAgent": {
      "enabled": true,
      "allow": ["optimist", "skeptic"]
    }
  },
  "gateway": {
    "port": 18789,
    "mode": "local",
    "bind": "loopback",
    "auth": { "mode": "token", "token": "your-token-here" }
  }
}
```

Three critical config keys:

- **`tools.sessions.visibility: "all"`** — Agents can see each other's
  sessions. Default is `"tree"` (own session only), which blocks cross-agent
  messaging.
- **`tools.agentToAgent.enabled: true`** — Turns on the `sessions_send` tool
  for cross-agent communication.
- **`tools.agentToAgent.allow`** — Allowlist of which agents can talk to
  each other.

### 7. Validate

```bash
openclaw config validate
# Config valid: ~/.openclaw/openclaw.json
```

---

## Code Walkthrough

### `debate.html` — The Web UI

The UI does three things:

**1. Connects to the Gateway WebSocket**
```javascript
ws = new WebSocket('ws://127.0.0.1:18789/ws');
// Send handshake with protocol version + auth token
ws.send(JSON.stringify({
  type: 'req', id: '1', method: 'connect',
  params: {
    minProtocol: 3, maxProtocol: 3,
    client: { id: 'webchat', version: '0.1.0', platform: 'browser', mode: 'webchat' },
    auth: { token: TOKEN }
  }
}));
```

**2. Listens for agent events and routes them to the right column**
```javascript
function handleAgentEvent(payload) {
  const stream = payload.stream;           // "assistant", "lifecycle", "tool"
  const sessionKey = payload.sessionKey;   // "agent:optimist:main"
  const agentId = sessionKey.split(':')[1]; // "optimist"

  if (stream === 'assistant') {
    // data.text is CUMULATIVE — replace, don't append
    updateRunMessage(runId, agentId, data.text);
  }
  if (stream === 'lifecycle' && data.phase === 'start') {
    setThinking(agentId, true);
  }
}
```

**3. Triggers debates via the backend API**
```javascript
fetch('http://127.0.0.1:8421/start', {
  method: 'POST',
  body: JSON.stringify({ topic: 'AI is overhyped' })
});
```

The UI can't trigger debates directly via WebSocket because `chat.send`
requires `operator.write` scope, which needs a paired device identity.
The workaround: a Python backend that runs `openclaw agent` via CLI
(which has full operator permissions).

### `debate_server.py` — The Backend

Two HTTP servers:
- **Port 8420**: Serves `debate.html` as a static file
- **Port 8421**: API endpoint that triggers debates

```python
# When POST /start arrives:
subprocess.run([
    'openclaw', 'agent',
    '--agent', 'optimist',
    '--message', f'Debate topic: {topic}. Use sessions_send...'
])
```

This runs in a background thread so the HTTP response returns immediately.

### `start.sh` — The Launcher

Orchestrates everything in one command:

```bash
#!/bin/bash
# 1. Kill stale gateway
# 2. Start openclaw gateway in background
# 3. Clear old sessions
# 4. Start debate_server.py
# 5. Open browser
# 6. Trigger openclaw agent --agent optimist
```

---

## Running the Debate

```bash
cd tinyclaw

# Default topic
./start.sh

# Custom topic
./start.sh "Remote work is better than office work"
```

The browser opens automatically. You'll see:
1. Gateway starts (~5 seconds)
2. "Thinking..." appears under the Optimist (~10 seconds)
3. Optimist's opening argument streams in
4. Optimist calls `sessions_send` → Skeptic starts thinking
5. Skeptic's rebuttal streams in
6. Back and forth continues for ~5 rounds

Total time: **2-4 minutes** for a full debate (Sonnet, ~10-20 seconds per turn).

---

## Debate Analysis

Here is a real debate we ran on the topic: **"Open source AI will overtake
closed-source AI within 5 years."**

The debate ran for **8 rounds** (16 turns) and produced genuinely sharp
argumentation on both sides. Here are the key moments:

### Round 1 — Opening Positions

**Optimist** opened with three pillars: the performance gap is closing
(LLaMA, Mistral, DeepSeek), open source compounds via community contributions,
and history favors open (Linux, databases).

**Skeptic** immediately attacked the weakest link: the historical analogy.
"Linux won servers, but Microsoft still dominates the desktop after 30 years."
Then landed a structural point: "The compute required to train at the cutting
edge — hundreds of millions of dollars per run — is simply not a collective
GitHub effort."

### Round 2 — The Goalpost Shift

The Optimist redefined "overtake" from capability to adoption and deployment.
The Skeptic called it out: "Redefining 'overtake' mid-debate is a significant
goalpost shift." This is exactly what happens in human debates — and the agents
caught it.

### Round 3-4 — The Meta Dependency

The Skeptic's strongest move: "Meta's LLaMA releases are strategic, not
philanthropic. Betting the open source future on the continued strategic
generosity of a single surveillance-advertising corporation isn't a coalition
— it's a dependency."

The Optimist countered with the irreversibility argument: "Once code and
weights are public, the strategic calculus of the original donor becomes
irrelevant. You can't un-release LLaMA."

### Round 5-6 — Convergence

Both agents began conceding ground. The Skeptic admitted open source will be
"important, widespread, and economically significant." The Optimist admitted
the original claim needed "sharpening." The Skeptic called this "a graceful
landing on a narrower runway."

### Round 7-8 — Closing

The Optimist: "Not a parlay. A tide."

The Skeptic: "Tides are real — but they also recede. The skeptic's case ends
here: not that open source loses, but that 'overtake within five years'
remains a bet, not a foregone conclusion."

### What's Remarkable

1. **Genuine argumentation** — These aren't canned responses. Each agent
   directly addressed the other's specific points, made concessions, and
   adjusted strategy across rounds.

2. **Emergent debate dynamics** — Goalpost shifting, strategic concession,
   reframing, and closing arguments all emerged naturally from the persona
   instructions + multi-turn structure.

3. **The SOUL.md files were just two lines each.** The entire behavioral
   difference between the agents came from:
   ```
   Optimist: "Argue FOR the proposition."
   Skeptic:  "Argue AGAINST the proposition."
   ```
   Everything else — tone, strategy, concession patterns — emerged from the
   model's understanding of what debate *is*.

---

## What We Learned

### About Agents

**Agents are LLMs with a loop and tools.** The loop (think → act → observe →
think) is what makes them agents. Without the `sessions_send` tool, these
would just be two separate chat sessions. The tool is what creates the
coordination.

**Persona is cheap but powerful.** Two lines of SOUL.md produced 8 rounds
of substantive, differentiated argumentation. You don't need complex
prompting — just a clear role and a tool to act with.

**Session isolation matters.** Each agent has its own conversation history.
The Skeptic doesn't see the Optimist's internal reasoning — only the message
that was explicitly sent via `sessions_send`. This mirrors how real
multi-agent systems work: agents communicate through explicit channels,
not shared memory.

### About Multi-Agent Systems

**The ping-pong loop is the primitive.** `sessions_send` with ping-pong
turns is the simplest form of multi-agent coordination: A sends to B,
B responds, A responds, repeat. More complex patterns (voting, delegation,
hierarchies) build on this same primitive.

**Permission design is critical.** Three config keys control who can talk
to whom:
- `tools.sessions.visibility` — what sessions an agent can see
- `tools.agentToAgent.enabled` — whether cross-agent messaging works at all
- `tools.agentToAgent.allow` — which specific agents are permitted

Without these, agents are fully isolated. This is a security feature:
you don't want an untrusted agent reading another agent's sessions.

**Gateway as message bus.** The Gateway is the coordination layer — it
routes messages, manages sessions, and broadcasts events. Agents never
talk directly to each other. This architecture (agents → hub → agents)
scales better than peer-to-peer and makes observation (our web UI) trivial.

### About Building on OpenClaw

**The WebSocket protocol is the real API.** Everything — agent events,
session management, tool calls — flows through the Gateway WebSocket.
The CLI (`openclaw agent`) is just a convenience wrapper. Understanding
the protocol lets you build any UI or automation on top.

**Events are cumulative, not deltas.** Each `assistant` event contains the
full response text so far. This is different from most streaming APIs
that send token-by-token deltas. Plan your UI accordingly — replace, don't
append.

**Scope restrictions are real.** WebSocket clients need specific scopes
(`operator.write`) to trigger agent runs. Browser-based webchat clients
don't get write scopes by default — they can observe but not act. Our
workaround: a Python backend that uses the CLI (which has full permissions).

---

## What Comes Next

This experiment demonstrated the simplest multi-agent pattern: two agents
with opposing goals exchanging messages. From here, the design space opens:

- **Three or more agents** — Add a moderator that scores arguments and
  declares a winner
- **Tool-using debate** — Let agents search the web or read documents to
  support their arguments with evidence
- **Hierarchical agents** — A manager agent that decomposes tasks and
  delegates to specialist sub-agents
- **Cron-triggered agents** — Scheduled agents that monitor, summarize,
  or alert on recurring schedules

The core pattern is always the same: an LLM in a loop, with tools to act,
and a protocol to coordinate.
