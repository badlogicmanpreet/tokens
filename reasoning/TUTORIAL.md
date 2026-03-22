# Building a Reasoning Model — From Base LLM to Text Generation

An end-to-end walkthrough of the pipeline, what every piece does, and how to extend it.

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
   - [Weight Loading](#7-weight-loading-weightspy--downloadpy)
4. [Running the Pipeline](#running-the-pipeline)
5. [Adding a New Model](#adding-a-new-model)
6. [What Comes Next — Reasoning](#what-comes-next)

---

## Project Structure

```
reasoning/
├── pyproject.toml          # Dependencies and build config
├── run.py                  # Interactive CLI entrypoint
├── src/
│   ├── __init__.py
│   ├── config.py           # Model architecture configs
│   ├── rope.py             # Rotary position embeddings
│   ├── cache.py            # KV cache for fast decoding
│   ├── model.py            # Transformer LM (the neural network)
│   ├── tokenizer.py        # Text <-> token IDs
│   ├── generate.py         # Decoding strategies (greedy, cached)
│   ├── download.py         # Download Qwen3 weights
│   └── weights.py          # Load & remap checkpoints
├── qwen3/                  # Downloaded Qwen3 weights (local)
└── models/                 # Downloaded HF models (local)
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
QWEN3_06B = {
    "name": "qwen3-0.6b",
    "vocab_size": 151_936,     # how many tokens the model knows
    "max_seq_len": 40_960,     # maximum input length in tokens
    "dim": 1024,               # size of each hidden vector
    "n_heads": 16,             # number of attention query heads
    "n_kv_heads": 8,           # number of key-value heads (GQA)
    "n_layers": 28,            # depth: how many transformer blocks
    "ffn_dim": 3072,           # feed-forward intermediate size
    "head_dim": 128,           # dimension per attention head
    "rope_theta": 1_000_000.0, # RoPE frequency base
    "norm_eps": 1e-6,          # RMSNorm epsilon
    "qk_norm": True,           # normalize Q and K before attention
    "dtype": torch.bfloat16,   # weight precision
}
```

**Why this matters:** To add a new model, you only need a new config dict. The
architecture code doesn't change.

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
["un", "happiness"]. This keeps the vocabulary finite (~150K tokens) while
representing any text.

**Special tokens:** The tokenizer auto-detects model-specific tokens:
- Qwen3: `<|endoftext|>` (EOS ID: 151643)
- Llama 3: `<|end_of_text|>` (EOS ID: 128001)

These tell the model where a response ends.

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
*relative* position difference, not absolute positions. Token at position 3
attending to position 1 produces the same angle-dependent score regardless of
whether those positions are 3,1 or 103,101. This lets the model generalize to
different sequence lengths.

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

```python
x = x.float()  # compute in float32 for stability
x = x * rsqrt(mean(x²) + eps)
return (x * scale).to(original_dtype)
```

#### Attention (Grouped-Query Attention)

This is the core mechanism that lets tokens "look at" other tokens.

```
Input: x (batch, seq_len, 1024)
  │
  ├─→ wq(x) → Q: (batch, 16 heads, seq_len, 128)    query: "what am I looking for?"
  ├─→ wk(x) → K: (batch,  8 heads, seq_len, 128)    key:   "what do I contain?"
  └─→ wv(x) → V: (batch,  8 heads, seq_len, 128)    value: "what info do I carry?"
  │
  │   [optional: QK norm — normalize Q and K per head]
  │   [apply RoPE — inject position information]
  │   [expand K,V from 8→16 heads by repeating each 2x]
  │
  ├─→ scores = (Q × Kᵀ) / √128              similarity between all token pairs
  ├─→ mask out future tokens (causal mask)   can't look ahead
  ├─→ weights = softmax(scores)              normalize to probabilities
  └─→ output = weights × V                  weighted sum of values
  │
  └─→ wo(output) → (batch, seq_len, 1024)   project back to model dimension
```

**Causal mask:** Token at position 5 can only attend to positions 0-5, never
positions 6+. This is what makes the model autoregressive — it can't cheat by
looking at future tokens.

**GQA (Grouped-Query Attention):** Instead of 16 separate KV heads (one per
query head), we use 8 KV heads. Each KV head is shared by 2 query heads via
`repeat_interleave`. This halves KV memory with <1% quality loss.

#### SwiGLU (Feed-Forward Network)

After attention mixes information across tokens, the FFN processes each token
independently to transform its representation:

```
x → gate(x) → SiLU activation ─┐
                                 × → down projection → output
x → up(x) ─────────────────────┘
```

In code: `down(silu(gate(x)) * up(x))`

The gating mechanism lets the network learn which features to pass through
and which to suppress. SiLU (Sigmoid Linear Unit) is a smooth activation
that works better than ReLU for language models.

**Dimensions:** 1024 → 3072 (expand) → 1024 (compress back)

#### TransformerBlock

One block = attention + FFN, both with pre-norm and residual connections:

```
x ─────────────────────────── + ──────────────────── + ── output
   │                          ↑    │                  ↑
   └→ RMSNorm → Attention ────┘    └→ RMSNorm → FFN ─┘
```

**Residual connections** (the + arrows) are critical. They let gradients flow
directly through the network during training and let each layer make
incremental refinements rather than needing to reconstruct the full
representation.

#### TransformerLM (Full Model)

Stacks everything together:

```
Token IDs (batch, seq_len)
    │
    ▼
Embedding          151,936 → 1024 dim lookup table
    │
    ▼
TransformerBlock × 28      each block: attention + FFN
    │
    ▼
RMSNorm            final normalization
    │
    ▼
Linear Head        1024 → 151,936 scores (one per vocab token)
    │
    ▼
Logits (batch, seq_len, 151936)
```

The model also:
- Precomputes RoPE tables once in `__init__` (registered as buffers)
- Builds causal masks dynamically based on sequence length
- Tracks position state (`_pos`) for KV cache offsets

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

Step 3 (decode):   1 new token
                   keys(1, 8, 12, 128), values(1, 8, 12, 128)
                   ...
```

**Implementation:** `KVCache` is a list of slots (one per layer). Each slot
holds a `(keys, values)` tuple or `None`. The attention layer reads from and
writes to its slot via `cache[layer_idx]`.

**Speed impact on Apple Silicon MPS:**
| Method           | tok/s |
|------------------|-------|
| No cache         | 10    |
| KV cache         | 29    |
| Cache + compile  | 38    |

---

### 6. Text Generation (`generate.py`)

#### `greedy(model, token_ids, max_tokens, eos_id)`

The simplest strategy — always pick the highest-probability token:

```python
for each step:
    logits = model(full_sequence)       # run entire sequence through model
    next_token = argmax(logits[:, -1])  # pick highest score at last position
    if next_token == eos_id: break      # stop if end-of-sequence
    append next_token to sequence       # grow the sequence by 1
```

Simple but slow — reprocesses everything every step.

#### `greedy_cached(model, token_ids, max_tokens, eos_id)`

Same output, but uses KV cache:

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

#### `pick_device()`

Auto-selects the best available hardware:
CUDA GPU → Apple MPS → Intel XPU → CPU

#### `benchmark(fn, *args, warmup=0)`

Wraps a generation call to measure wall-clock time. The `warmup` parameter
handles `torch.compile`'s first-run compilation overhead.

---

### 7. Weight Loading (`weights.py` + `download.py`)

Pre-trained weights come in different formats with different naming conventions.
This module translates them into our model's naming.

#### The remapping problem

Our model names parameters like:
```
layers.0.attn.wq.weight
layers.0.ffn.gate.weight
norm.scale
```

But checkpoints use different names:

| Source | Checkpoint key | Our key |
|--------|---------------|---------|
| Qwen3 .pth | `trf_blocks.0.att.W_query.weight` | `layers.0.attn.wq.weight` |
| HuggingFace | `model.layers.0.self_attn.q_proj.weight` | `layers.0.attn.wq.weight` |

`_remap_qwen()` and `_remap_hf()` handle this translation via simple
dictionaries.

#### Tied embeddings

Some models (like Llama 3.2 1B) share the embedding matrix with the output
head — `lm_head.weight` doesn't exist in the checkpoint. The remapper detects
this and copies the embedding weights:

```python
if "head.weight" not in out and "embedding.weight" in out:
    out["head.weight"] = out["embedding.weight"]
```

#### Two loading paths

**Qwen3** — custom `.pth` file from a specific mirror:
```python
model = load_model("qwen3/qwen3-0.6B-base.pth", cfg=QWEN3_06B, device=device)
```

**HuggingFace models** — `.safetensors` from the Hub:
```python
model, model_dir = load_hf_model("llama-3.2-1b", device=device)
# model_dir also contains the tokenizer
```

---

## Running the Pipeline

```bash
cd ~/git/tokens/reasoning

# Qwen3 0.6B (default, ~1.4GB download first time)
uv run python run.py

# Llama 3.2 1B (requires HF login for gated model, ~2.4GB)
uv run python run.py llama
```

Both models are **base** (pre-trained) models — they complete text, they don't
follow instructions. This is intentional. The reasoning chapters will add that.

---

## Adding a New Model

Any model that uses the same architecture pattern (RoPE + GQA + SwiGLU +
RMSNorm) can be added in three steps. Here's a concrete example with a
hypothetical "Mistral 7B":

### Step 1: Add config in `config.py`

```python
MISTRAL_7B = {
    "name": "mistral-7b",
    "repo_id": "mistralai/Mistral-7B-v0.3",
    "vocab_size": 32_768,
    "max_seq_len": 32_768,
    "dim": 4096,
    "n_heads": 32,
    "n_kv_heads": 8,
    "n_layers": 32,
    "ffn_dim": 14336,
    "head_dim": 128,          # 4096 / 32
    "rope_theta": 1_000_000.0,
    "norm_eps": 1e-5,
    "qk_norm": False,
    "dtype": torch.bfloat16,
}

# Add to registry
MODELS = {
    "qwen3-0.6b": QWEN3_06B,
    "llama-3.2-1b": LLAMA_32_1B,
    "mistral-7b": MISTRAL_7B,      # ← new
}
```

**Where to find these values:** Look at the model's `config.json` on HuggingFace.
The mapping is:
```
hidden_size          → dim
num_attention_heads  → n_heads
num_key_value_heads  → n_kv_heads
num_hidden_layers    → n_layers
intermediate_size    → ffn_dim
head_dim             → head_dim  (or dim // n_heads)
rope_theta           → rope_theta
rms_norm_eps         → norm_eps
vocab_size           → vocab_size
max_position_embeddings → max_seq_len
```

### Step 2: Check weight naming

Most HuggingFace models use the same naming convention (the `_HF_LAYER` and
`_HF_TOP` maps in `weights.py`). If your model follows the standard
`model.layers.N.self_attn.q_proj.weight` pattern, no changes needed.

If it uses different names, add a new remap function in `weights.py`:

```python
_MYSTRAL_LAYER = {
    "attention.wq.weight": "attn.wq.weight",
    # ... custom mappings
}
```

### Step 3: Add to `run.py`

```python
elif choice == "mistral":
    model, model_dir = load_hf_model("mistral-7b", device=device, local_dir=ROOT / "models")
    tok = Tokenizer(model_dir / "tokenizer.json")
```

Then run: `uv run python run.py mistral`

### Models that WON'T work without code changes

- **Models with different attention** (e.g., sliding window, linear attention)
- **Models with different FFN** (e.g., MoE / Mixture of Experts like Mixtral)
- **Models with different normalization** (e.g., LayerNorm instead of RMSNorm)
- **Models with different position encoding** (e.g., ALiBi instead of RoPE)

Most modern open models (Llama, Qwen, Gemma, Phi, SmolLM, OLMo) use the same
pattern and will work.

---

## What Comes Next

What we have now is a **base LLM that completes text**. It predicts the most
likely next token based on patterns learned during pre-training. It doesn't
reason, follow instructions, or explain its thinking.

The upcoming chapters add reasoning capabilities on top of this foundation:

```
 [Chapter 2] Base LLM + text generation     ← YOU ARE HERE
      │
      ▼
 [Chapter 3] Evaluation — math verifier
      │       Build a system that checks if the LLM's answer is correct
      │       by comparing against reference solutions.
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
