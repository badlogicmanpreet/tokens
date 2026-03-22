# Inference-Time Scaling — Better Reasoning Without Retraining

How to make Qwen3-1.7B reason better by spending more compute at generation
time, without touching the model's weights.

---

## Table of Contents

1. [The Core Idea](#the-core-idea)
2. [Method 1: Chain-of-Thought Prompting](#method-1-chain-of-thought-prompting)
3. [Method 2: Temperature Scaling + Top-p Sampling](#method-2-temperature-scaling--top-p-sampling)
   - [Temperature](#temperature)
   - [Top-p (Nucleus) Filtering](#top-p-nucleus-filtering)
4. [Method 3: Self-Consistency (Majority Voting)](#method-3-self-consistency-majority-voting)
5. [How They Combine](#how-they-combine)
6. [Running the Evaluations](#running-the-evaluations)
7. [Code Walkthrough](#code-walkthrough)
8. [What Comes Next](#what-comes-next)

---

## The Core Idea

There are two ways to make an LLM smarter:

```
1. Train harder    → change the weights (expensive, one-time cost)
2. Think harder    → spend more compute at inference (no training needed)
```

This chapter is about option 2. We don't modify Qwen3-1.7B at all — same
weights, same model file. We just change *how* we use it at generation time.

The three techniques we implement:

```
┌─────────────────────────┐
│ Chain-of-Thought (CoT)  │  Prompt the model to explain its reasoning
│                         │  "Explain step by step."
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│ Temperature + Top-p     │  Generate diverse responses instead of
│                         │  always picking the single most likely token
└────────────┬────────────┘
             │
┌────────────▼────────────┐
│ Self-Consistency        │  Generate N answers, pick the one that
│ (Majority Voting)       │  appears most frequently
└─────────────────────────┘
```

Each technique trades more compute for better answers.

---

## Method 1: Chain-of-Thought Prompting

The simplest technique — just append an instruction to think step by step.

**Without CoT:**
```
Prompt:  "Half the value of 3x-9 is x+37. What is x?"
Answer:  \boxed{20}   ← WRONG (jumped to answer)
```

**With CoT:**
```
Prompt:  "Half the value of 3x-9 is x+37. What is x?

         Explain step by step."
Answer:  Step 1: Write equation (3x-9)/2 = x+37
         Step 2: Multiply both sides by 2 → 3x-9 = 2x+74
         Step 3: Subtract 2x → x-9 = 74
         Step 4: Add 9 → x = 83
         \boxed{83}   ← CORRECT
```

**Why does this work?**

1. Walking through steps gives the model more chances to self-correct
2. Step-by-step reasoning matches training data patterns (math textbooks,
   StackOverflow answers, etc.)
3. Each intermediate token provides context that helps predict the next one

**The trade-off:** CoT generates 5-10x more tokens per problem. The model is
slower but more accurate on complex problems. On simple problems, it can
sometimes *hurt* accuracy by overthinking.

**Implementation** (`src/inference.py`):
```python
def cot_prompt(prompt: str) -> str:
    return prompt + " \n\nExplain step by step."
```

That's it — one line. The power is in what it triggers in the model.

---

## Method 2: Temperature Scaling + Top-p Sampling

### Why we need this

Greedy decoding (argmax) always picks the single highest-probability token.
This is deterministic — same input always produces the same output. That's
fine for a single answer, but if we want to generate *multiple different*
answers (for voting in Method 3), we need randomness.

### Temperature

Temperature controls how "peaked" or "flat" the probability distribution is
before sampling:

```
logits = model(tokens)[:, -1]     # raw scores for next token
scaled = logits / temperature      # rescale
probs  = softmax(scaled)           # convert to probabilities
```

| Temperature | Effect | Use case |
|-------------|--------|----------|
| 0.0 | Greedy (argmax) — always pick top token | Deterministic answers |
| 0.1 - 0.5 | Very focused — mostly picks top token, rare variation | Math, code |
| 0.7 - 0.9 | Balanced diversity | Self-consistency voting |
| 1.0 | No change (original distribution) | General use |
| > 1.5 | Very diverse — often picks unlikely tokens | Creative writing (not math) |

**Visually:**
```
Temperature = 0.3 (sharp)         Temperature = 2.0 (flat)

  ▌                                 ▌ ▌
  ▌                                 ▌ ▌ ▌
  ▌                                 ▌ ▌ ▌ ▌
  ▌ ▌                               ▌ ▌ ▌ ▌ ▌
  ▌ ▌ ▌                             ▌ ▌ ▌ ▌ ▌ ▌
──────────                        ──────────────
  Berlin is the clear winner        Many tokens are roughly equal
```

### Top-p (Nucleus) Filtering

Even with temperature, very unlikely tokens can occasionally be sampled.
Top-p filtering removes them:

```
1. Sort tokens by probability (highest first)
2. Compute cumulative sum
3. Keep only tokens where cumsum <= top_p threshold
4. Zero out everything else
5. Renormalize so remaining probabilities sum to 1
```

**Example with top_p = 0.8:**
```
Token       Prob    CumSum   Keep?
Berlin      0.45    0.45     Yes ✓
______      0.21    0.66     Yes ✓
____        0.17    0.83     No  ✗  (exceeded 0.8)
Munich      0.03    ...      No  ✗
Hamburg     0.03    ...      No  ✗
...
```

Only "Berlin" and "______" can be sampled. The rare junk tokens are gone.

**Typical settings for math reasoning:** `temperature=0.8, top_p=0.9`

---

## Method 3: Self-Consistency (Majority Voting)

This is where temperature + top-p pay off. The algorithm:

```
For the same problem, generate N independent answers:

  Sample 1:  → 83
  Sample 2:  → 22
  Sample 3:  → 54
  Sample 4:  → 83
  Sample 5:  → 26

Count:  83 appears 2x, everything else 1x
Winner: 83   ← CORRECT
```

**Why does this work?**

The model might reach the correct answer through different reasoning paths.
Wrong answers tend to be random (each wrong in a different way), while
correct answers cluster together. By taking the majority vote, noise cancels
out and the correct answer surfaces.

**The trade-off:** N× more compute. 5 samples = 5× the generation time.
But accuracy often improves significantly, especially on hard problems.

**Tie-breaking:** When multiple answers tie for the most frequent, we pick
the first one that appeared in the sample list (rather than returning None).

**Implementation** (`src/inference.py`):
```python
def self_consistency(model, tokenizer, prompt, device,
                     num_samples=5, temperature=0.8, top_p=0.9, ...):
    answers = []
    for i in range(num_samples):
        torch.manual_seed(seed + i + 1)          # reproducible diversity
        resp = generate(..., temperature=temperature, top_p=top_p)
        answers.append(extract_answer(resp))

    counts = Counter(answers)
    winner = counts.most_common(1)[0][0]          # majority vote
    return {"winner": winner, "counts": dict(counts), ...}
```

---

## How They Combine

These techniques are independent and stackable:

| Combination | What happens |
|-------------|-------------|
| Greedy only | Deterministic baseline (Ch.3) |
| CoT only | Longer reasoning, still deterministic |
| Temperature + top-p only | One diverse sample per problem |
| **CoT + vote** | Best of all worlds: reason step-by-step, sample N, vote |

The evaluate script supports all modes:

```bash
# Greedy baseline
uv run python evaluate.py --mode greedy

# Chain-of-thought (still greedy, but longer reasoning)
uv run python evaluate.py --mode cot

# Self-consistency voting (uses temperature + top-p internally)
uv run python evaluate.py --mode vote --samples 5
```

CoT can be combined with vote mode by modifying the prompt template in
`evaluate.py` — the `cot_prompt()` function can be applied inside the
voting loop.

---

## Running the Evaluations

```bash
cd ~/git/tokens/reasoning

# Baseline: greedy decoding, 10 problems
uv run python evaluate.py --mode greedy --n 10

# Chain-of-thought on 10 problems
uv run python evaluate.py --mode cot --n 10

# Self-consistency: 5 samples per problem, 10 problems
uv run python evaluate.py --mode vote --n 10 --samples 5

# Tune the sampling parameters
uv run python evaluate.py --mode vote --n 5 --samples 3 \
    --temperature 0.8 --top-p 0.9 --seed 42

# Full benchmark (slow)
uv run python evaluate.py --mode cot --n 0
```

Results are saved to `math500_{mode}.jsonl`.

---

## Code Walkthrough

### `src/generate.py` — Token-level generation

**`_top_p_filter(probs, top_p)`**
Takes a probability distribution, keeps only the smallest set of tokens
whose cumulative probability stays under `top_p`, zeros the rest, and
renormalizes.

**`sample_cached(model, token_ids, max_tokens, temperature, top_p)`**
KV-cached generation that supports both greedy (temperature=0) and
sampling (temperature>0 with top-p filtering). This is the engine that
powers diverse response generation.

**`generate(model, tokenizer, prompt, device, temperature, top_p)`**
High-level wrapper: text in → text out. Handles encoding, cached generation,
and decoding. Supports both greedy and sampling modes via temperature.

### `src/inference.py` — Reasoning strategies

**`cot_prompt(prompt)`**
One-liner that appends "Explain step by step." to any prompt.

**`self_consistency(model, tokenizer, prompt, device, num_samples, ...)`**
Generates `num_samples` responses with temperature>0, extracts the answer
from each, and returns the majority winner. Uses per-sample seeding for
reproducibility.

### `evaluate.py` — Evaluation harness

Supports three `--mode` options:
- `greedy`: standard deterministic evaluation (baseline)
- `cot`: appends chain-of-thought instruction to the prompt
- `vote`: runs self-consistency with configurable `--samples`, `--temperature`, `--top-p`

---

## What Comes Next

So far we've improved reasoning at inference time — no weight changes.
The remaining chapters modify the model itself:

```
 [Chapter 4] Inference-time scaling       ← DONE (this chapter)
      │       CoT prompting, temperature/top-p, self-consistency
      ▼
 [Chapter 5] Self-refinement
      │       Model reviews and iterates on its own answer
      ▼
 [Chapter 6] Reinforcement learning
      │       Train weights using the math verifier as reward signal
      ▼
 [Chapter 7] Distillation
      │       Transfer reasoning from large model to small model
      ▼
 [Chapter 8] Advanced techniques
```

The key insight from this chapter: **a model already knows more than it shows
in a single greedy pass.** Chain-of-thought unlocks hidden reasoning ability.
Self-consistency surfaces the correct answer from noise. These techniques
are why reasoning models generate long "thinking" traces — it's not wasted
tokens, it's the model using more compute to find better answers.
