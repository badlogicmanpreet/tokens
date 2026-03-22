# Building a Reasoning Model — From Base LLM to Text Generation

An end-to-end walkthrough of the pipeline, what every piece does, and how it all connects.

Our standard model throughout this project is **Qwen3-1.7B** (2 billion parameters).

---

## Table of Contents

1. [Project Structure](#project-structure)
2. [The Pipeline — Bird's Eye View](#the-pipeline)
3. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
   - [Configuration](#1-configuration-configpy)
   - [Tokenization](#2-tokenization-tokenizerpy)
   - [Positional Encoding (RoPE)](#3-positional-encoding-ropepy)
   - [The Model](#4-the-model-modelpy)
   - [KV Cache](#5-kv-cache-cachepy)
   - [Text Generation](#6-text-generation-generatepy)
   - [Weight Loading](#7-weight-loading-weightspy)
4. [Running the Pipeline](#running-the-pipeline)
5. [What Comes Next — Reasoning](#what-comes-next)

---

## Project Structure

```
reasoning/
├── pyproject.toml          # Dependencies and build config
├── run.py                  # Interactive CLI entrypoint
├── evaluate.py             # MATH-500 evaluation script
├── src/
│   ├── __init__.py
│   ├── config.py           # Model architecture configs
│   ├── rope.py             # Rotary position embeddings
│   ├── cache.py            # KV cache for fast decoding
│   ├── model.py            # Transformer LM (the neural network)
│   ├── tokenizer.py        # Text <-> token IDs
│   ├── generate.py         # Decoding strategies (greedy, cached)
│   ├── download.py         # Download weight files
│   ├── weights.py          # Load & remap checkpoints
│   └── verify.py           # Math answer verification (Ch.3)
└── models/                 # Downloaded HF model weights (local, gitignored)
```

Each file has one job. No circular dependencies. The flow is always left-to-right:

```
config → model → weights → generate → run.py
              ↑
         rope, cache
```

---

## The Pipeline

Here is what happens from the moment you type a prompt to seeing the output:

```
 "What is gravity?"                    ← your text
        │
        ▼
 ┌──────────────┐
 │  Tokenizer    │  text → integer IDs
 │  (tokenizer)  │  "What" → 3923, " is" → 374, ...
 └──────┬───────┘
        │  [3923, 374, 24128, 30]
        ▼
 ┌──────────────┐
 │  TransformerLM│  IDs → probability scores over vocab
 │  (model)      │  28 layers of attention + feed-forward
 └──────┬───────┘
        │  logits: shape (1, seq_len, 151936)
        ▼
 ┌──────────────┐
 │  Generate     │  pick highest-scoring token (argmax)
 │  (generate)   │  append to input, repeat until EOS
 └──────┬───────┘
        │  [29587, 382, 1059, ...]
        ▼
 ┌──────────────┐
 │  Tokenizer    │  integer IDs → text
 │  (decode)     │  29587 → " Gravity", ...
 └──────────────┘
        │
        ▼
 " Gravity is the force that..."       ← output text
```

The model generates **one token per loop iteration**. Each new token is fed back in as
input to predict the next one. This is called **autoregressive generation**.

---

## Step-by-Step Walkthrough

### 1. Configuration (`config.py`)

A config is a plain dictionary that fully describes the shape of a model.
The model code reads these values — it never hardcodes sizes.

```python
QWEN3_17B = {
    "name": "qwen3-1.7b",
    "repo_id": "Qwen/Qwen3-1.7B",
    "vocab_size": 151_936,     # how many tokens the model knows
    "max_seq_len": 40_960,     # maximum input length in tokens
    "dim": 2048,               # size of each hidden vector
    "n_heads": 16,             # number of attention query heads
    "n_kv_heads": 8,           # number of key-value heads (GQA)
    "n_layers": 28,            # depth: how many transformer blocks
    "ffn_dim": 6144,           # feed-forward intermediate size
    "head_dim": 128,           # dimension per attention head
    "rope_theta": 1_000_000.0, # RoPE frequency base
    "norm_eps": 1e-6,          # RMSNorm epsilon
    "qk_norm": True,           # normalize Q and K before attention
    "dtype": torch.bfloat16,   # weight precision
}
```

**Key relationship — `n_heads` vs `n_kv_heads`:**
- `n_heads = 16` query heads, `n_kv_heads = 8` key-value heads
- Every 2 query heads share 1 KV head (Grouped-Query Attention)
- This cuts memory usage for KV by 50% with minimal quality loss

---

### 2. Tokenization (`tokenizer.py`)

Neural networks process numbers, not text. The tokenizer is the bridge.

**Encoding** — text to IDs:
```
"Explain large language models."
    ↓ split into subwords (BPE algorithm)
["Ex", "plain", " large", " language", " models", "."]
    ↓ lookup each in vocabulary (152K entries)
[840, 20772, 3460, 4128, 4119, 13]
```

**Decoding** — IDs back to text:
```
[840, 20772, 3460, 4128, 4119, 13]
    ↓ lookup + concatenate
"Explain large language models."
```

**Why subwords?** A vocabulary can't contain every possible word. BPE (Byte Pair
Encoding) splits rare words into common pieces. "unhappiness" might become
["un", "happiness"]. This keeps the vocabulary finite (~152K tokens) while
representing any text.

**EOS token:** `<|endoftext|>` (ID: 151643) — tells the model where a response ends.

---

### 3. Positional Encoding (`rope.py`)

Attention is position-blind — "the cat sat" and "sat cat the" would produce
identical attention scores without position information. RoPE (Rotary Position
Embeddings) solves this by rotating Q and K vectors based on their position.

**How it works:**

```
Step 1: Precompute rotation angles for every possible position
        build_rope_table(head_dim=128, max_seq_len=40960, theta=1e6)
        → cos_table: (40960, 128)
        → sin_table: (40960, 128)

Step 2: For each token at position p, rotate its Q and K vectors
        apply_rope(q, cos, sin, offset=p)
```

The rotation uses this formula (split-halves style):
```
Given vector x = [x_left | x_right]  (split in half)

x_rotated = x * cos(position) + [-x_right | x_left] * sin(position)
```

**Why rotation?** The dot product between two rotated vectors depends on their
*relative* position difference, not absolute positions. This lets the model
generalize to different sequence lengths.

**The offset parameter:** During cached generation, new tokens arrive one at a
time but need position numbers that continue from where the prompt left off.
If the prompt was 10 tokens long, the next generated token is at position 10,
then 11, etc. The `offset` parameter handles this.

---

### 4. The Model (`model.py`)

The model is a stack of components. Here's each one, bottom-up:

#### RMSNorm
```
input x → normalize to unit variance → scale by learned parameter
```
Simpler than LayerNorm (no mean centering, no bias). Used before every
attention and FFN sublayer.

#### Attention (Grouped-Query Attention)

The core mechanism that lets tokens "look at" other tokens.

```
Input: x (batch, seq_len, 2048)
  │
  ├─→ wq(x) → Q: (batch, 16 heads, seq_len, 128)    "what am I looking for?"
  ├─→ wk(x) → K: (batch,  8 heads, seq_len, 128)    "what do I contain?"
  └─→ wv(x) → V: (batch,  8 heads, seq_len, 128)    "what info do I carry?"
  │
  │   [QK norm — normalize Q and K per head]
  │   [apply RoPE — inject position information]
  │   [expand K,V from 8→16 heads by repeating each 2x]
  │
  ├─→ scores = (Q × Kᵀ) / √128              similarity between all token pairs
  ├─→ mask out future tokens (causal mask)   can't look ahead
  ├─→ weights = softmax(scores)              normalize to probabilities
  └─→ output = weights × V                  weighted sum of values
  │
  └─→ wo(output) → (batch, seq_len, 2048)   project back to model dimension
```

**Causal mask:** Token at position 5 can only attend to positions 0-5, never
positions 6+. This makes the model autoregressive.

**GQA:** 16 query heads share 8 KV heads (2:1 ratio). Halves KV memory
with negligible quality loss.

#### SwiGLU (Feed-Forward Network)

After attention mixes information across tokens, the FFN processes each token
independently:

```
x → gate(x) → SiLU activation ─┐
                                 × → down projection → output
x → up(x) ─────────────────────┘
```

In code: `down(silu(gate(x)) * up(x))`

**Dimensions:** 2048 → 6144 (expand) → 2048 (compress back)

#### TransformerBlock

One block = attention + FFN, both with pre-norm and residual connections:

```
x ─────────────────────────── + ──────────────────── + ── output
   │                          ↑    │                  ↑
   └→ RMSNorm → Attention ────┘    └→ RMSNorm → FFN ─┘
```

**Residual connections** (the + arrows) let gradients flow directly through
the network and let each layer make incremental refinements.

#### TransformerLM (Full Model)

Stacks everything together:

```
Token IDs (batch, seq_len)
    │
    ▼
Embedding          151,936 → 2048 dim lookup table
    │
    ▼
TransformerBlock × 28      each block: attention + FFN
    │
    ▼
RMSNorm            final normalization
    │
    ▼
Linear Head        2048 → 151,936 scores (one per vocab token)
    │
    ▼
Logits (batch, seq_len, 151936)
```

Note: Qwen3-1.7B uses **tied embeddings** — the embedding matrix and the
output head share the same weights, reducing model size.

---

### 5. KV Cache (`cache.py`)

**The problem:** In autoregressive generation, step N reprocesses all N-1
previous tokens to generate token N. This is O(n²) total work.

**The solution:** Cache the Key and Value projections from previous steps.
Each new step only computes K,V for the single new token and concatenates
with the cache.

```
Step 1 (prefill):  Prompt has 10 tokens
                   Compute K,V for all 10 → store in cache
                   Cache per layer: keys(1, 8, 10, 128), values(1, 8, 10, 128)

Step 2 (decode):   1 new token
                   Compute K,V for just this token
                   Concatenate: keys(1, 8, 11, 128), values(1, 8, 11, 128)
                   Q only needs shape (1, 16, 1, 128) — attend to all 11

Step 3 (decode):   1 new token → keys(1, 8, 12, 128) ...
```

**Speed impact (Qwen3-1.7B on Apple Silicon MPS):**
| Method           | tok/s |
|------------------|-------|
| No cache         | ~10   |
| KV cache         | ~15   |
| Cache + compile  | ~20+  |

---

### 6. Text Generation (`generate.py`)

#### `greedy_cached(model, token_ids, max_tokens, eos_id)`

Greedy decoding with KV cache — our standard generation method:

```python
cache = KVCache(n_layers)
model.reset_cache_state()

# Prefill: process entire prompt, populate cache
logits = model(full_prompt, cache=cache)

for each step:
    next_token = argmax(logits[:, -1])
    if next_token == eos_id: break
    append next_token to sequence
    logits = model(just_next_token, cache=cache)  # only 1 token!
```

#### `generate(model, tokenizer, prompt, device, max_tokens)`

High-level wrapper that handles encoding, generation, and decoding:

```python
response = generate(model, tokenizer, "What is 2+2?", device, max_tokens=512)
```

---

### 7. Weight Loading (`weights.py`)

Pre-trained weights from HuggingFace use different parameter names than our
model. The remapping layer translates them:

```
HuggingFace checkpoint              →  Our model
model.embed_tokens.weight            →  embedding.weight
model.layers.0.self_attn.q_proj      →  layers.0.attn.wq
model.layers.0.mlp.gate_proj         →  layers.0.ffn.gate
model.norm.weight                    →  norm.scale
```

**Tied embeddings:** Qwen3-1.7B shares embedding and output head weights.
When `lm_head.weight` is missing from the checkpoint, the remapper copies
from `embedding.weight` automatically.

**Loading:**
```python
model, model_dir = load_hf_model("qwen3-1.7b", device=device, local_dir="models")
tok = Tokenizer(model_dir / "tokenizer.json")
```

---

## Running the Pipeline

```bash
cd ~/git/tokens/reasoning

# Interactive prompt
uv run python run.py

# Evaluate on MATH-500 (10 problems)
uv run python evaluate.py --n 10

# Evaluate with verbose output
uv run python evaluate.py --n 10 --verbose
```

Qwen3-1.7B is a **base** (pre-trained) model — it completes text, it doesn't
follow instructions. This is intentional. The reasoning chapters add that capability.

---

## What Comes Next

What we have now is a **base LLM that completes text**. It predicts the most
likely next token based on patterns learned during pre-training. It doesn't
reason, follow instructions, or explain its thinking.

The upcoming chapters add reasoning capabilities on top of this foundation:

```
 [Chapter 2] Base LLM + text generation     ← DONE
      │
      ▼
 [Chapter 3] Evaluation — math verifier      ← DONE
      │       Checks if the LLM's answer is correct using SymPy.
      │       Qwen3-1.7B scores ~40% on MATH-500 (base model).
      ▼
 [Chapter 4-5] Inference-time scaling
      │         Make the model "think harder" at generation time without
      │         changing its weights: chain-of-thought prompting,
      │         best-of-N sampling, self-refinement.
      ▼
 [Chapter 6] Reinforcement learning
      │       Train the model to reason by rewarding correct answers
      │       and penalizing wrong ones (GRPO / reward-based RL).
      ▼
 [Chapter 7] Distillation
      │       Transfer reasoning ability from a large model to a
      │       small one via supervised fine-tuning on reasoning traces.
      ▼
 [Chapter 8] Advanced techniques + future directions
```

The key insight: the base model already has knowledge (from pre-training).
Reasoning techniques teach it to *use* that knowledge step-by-step instead of
just pattern-matching the most likely next word.
