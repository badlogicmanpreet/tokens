# Self-Refinement — The Model as Its Own Critic

How to make Qwen3-1.7B iteratively improve its answers by critiquing
and revising them, plus two scoring methods for measuring answer quality.

---

## Table of Contents

1. [The Idea](#the-idea)
2. [Scoring Answers](#scoring-answers)
   - [Heuristic Scorer](#1-heuristic-scorer)
   - [Logprob Scorer](#2-logprob-scorer)
3. [Self-Refinement Loop](#self-refinement-loop)
4. [How It All Fits Together](#how-it-all-fits-together)
5. [Running the Evaluation](#running-the-evaluation)
6. [Code Walkthrough](#code-walkthrough)
7. [What Comes Next](#what-comes-next)

---

## The Idea

In Chapter 4 we generated multiple independent answers and voted on them
(self-consistency). That works, but it requires extractable short answers
for comparison and treats each sample independently.

Self-refinement takes a different approach — instead of generating N
separate answers, the model **iterates on a single answer**:

```
                     ┌──────────────────┐
                     │  Initial Answer   │
                     │  \boxed{18} WRONG │
                     └────────┬─────────┘
                              │
                     ┌────────▼─────────┐
                     │  Critique         │
                     │  "The equation    │
                     │   setup is wrong. │
                     │   Fix: multiply   │
                     │   both sides by 2"│
                     └────────┬─────────┘
                              │
                     ┌────────▼─────────┐
                     │  Revised Answer   │
                     │  \boxed{83} RIGHT │
                     └────────┬─────────┘
                              │
                     Score improved? ──→ Keep revision
                              │
                     (repeat N times)
```

The model plays three roles in sequence:
1. **Solver** — generates an answer
2. **Critic** — finds errors in the answer
3. **Editor** — writes a revised answer using the critique

---

## Scoring Answers

Before we can decide whether a revision is *better*, we need a way to
score answers. We built two scorers in `src/scoring.py`.

### 1. Heuristic Scorer

A rule-based score that rewards good formatting and brevity:

```python
def heuristic_score(answer, prompt=None):
    score = 0.0

    # Does it have a \boxed{} answer? (+2.0)
    # Else a number? (+1.0)
    # Else anything? (+0.0)

    # Brevity bonus: shorter = better (exponential decay)
    score += 1.5 * exp(-len(answer) / 500)

    return score
```

**Score breakdown:**

| Component | Value | Why |
|-----------|-------|-----|
| Has `\boxed{}` | +2.0 | Best extraction format |
| Has a number (no box) | +1.0 | Fallback extraction |
| Brevity (100 chars) | +1.2 | Short answers preferred |
| Brevity (500 chars) | +0.6 | Medium |
| Brevity (2000 chars) | +0.03 | Long answer, minimal bonus |

**Example:**
```
Response 1 (1422 chars, has \boxed{}):  score = 2.087
Response 2 (651 chars, has \boxed{}):   score = 2.408  ← winner (shorter)
```

The heuristic scorer doesn't know if the answer is *correct* — it only
judges format and conciseness. It's fast, cheap, and deterministic.

### 2. Logprob Scorer

Uses the model's own confidence to score an answer. The idea: feed
`[prompt + answer]` through the model and check how likely the model
thinks each answer token is.

```
Prompt: "What is the capital of Germany?"
Answer: " The capital of Germany is Berlin."

For each answer token, look up its probability:
  "The"     → log_prob = -0.015
  "capital" → log_prob = -0.000
  "of"      → log_prob = -0.008
  "Germany" → log_prob = -0.075
  "is"      → log_prob = -0.158
  "Berlin"  → log_prob = -1.172
  "."       → log_prob = -0.000

Average: -0.204  (close to 0 = high confidence)
```

Compare with a wrong answer:
```
Answer: " The capital of Germany is Bridge."
  ...
  "Bridge"  → log_prob = -15.0   (model is very unsure!)

Average: -3.891  (far from 0 = low confidence)
```

**Why average instead of sum?** Summing penalizes longer answers
unfairly. A 100-token correct answer would score lower than a 5-token
wrong one. Averaging normalizes for length.

**Why log-probabilities instead of raw probabilities?**
- Probabilities multiply: `0.01 × 0.01 × 0.01 = 0.000001` (underflows)
- Log-probs add: `-4.6 + (-4.6) + (-4.6) = -13.8` (stable arithmetic)

**Implementation:**
```python
@torch.inference_mode()
def logprob_score(model, tokenizer, prompt, answer, device):
    full_ids = tensor(tokenizer.encode(prompt) + tokenizer.encode(answer))
    logits = model(full_ids.unsqueeze(0)).squeeze(0)
    log_probs = log_softmax(logits, dim=-1)

    # Only score answer tokens, not the prompt
    start = len(prompt_ids) - 1
    answer_logprobs = log_probs[start:end, target_ids]

    return answer_logprobs.mean().item()
```

---

## Self-Refinement Loop

The full loop in `src/inference.py`:

### Step 1: Generate initial answer

```python
current = generate(model, tokenizer, prompt, device, temperature=0.7)
current_score = score_fn(answer=current, prompt=prompt)  # e.g., 2.09
```

### Step 2: Critique

The model receives a critique prompt containing the original question
and the draft answer:

```
"You are a meticulous reviewer. Identify logical errors, missing
steps, or arithmetic mistakes..."

Question: Half the value of 3x-9 is x+37. What is x?
Draft answer: \boxed{18}

Write a short critique and bullet-point fix plan.
```

The model generates something like:
```
The answer 18 is incorrect.
Fix plan:
1. Rewrite equation: (3x-9)/2 = x+37
2. Multiply both sides by 2: 3x-9 = 2x+74
3. Solve: x = 83
```

### Step 3: Revise

The model receives the original question, its draft, and the critique:

```
Revise the answer using the critique. End with \boxed{ANSWER}

Question: ...
Previous answer: \boxed{18}
Critique: The answer 18 is incorrect. Fix plan: ...

Revised answer:
```

The model generates: `\boxed{83}` — correct!

### Step 4: Repeat steps 2-3 for N iterations

Each iteration takes the current best answer, critiques it, and
proposes a revision.

### Where does the scorer fit in?

The scorer is **not** part of the 3 core steps (solve → critique →
revise). It acts as a **gatekeeper between iterations**, deciding
whether to accept or reject each revision:

```
Iteration 0:  Solve          → \boxed{18}    score = 2.09
Iteration 1:  Critique + Revise → \boxed{83}  score = 2.41
                                    ↓
                          2.41 > 2.09? → YES, accept
                                    ↓
Iteration 2:  Critique + Revise → \boxed{42}  score = 1.80
                                    ↓
                          1.80 > 2.41? → NO, reject. Keep \boxed{83}
                                    ↓
Final answer: \boxed{83} ✓
```

**Without a scorer**, every revision is blindly accepted — and sometimes
the model "corrects" a right answer into a wrong one. The scorer
prevents that regression.

**With a scorer**, the loop can only improve or stay the same — never
get worse. This is the key insight: the scorer acts as a ratchet.

In a single-iteration run (`--samples 1`), the scorer has no practical
effect since there's only one revision to compare against the initial
answer.

**Which scorer to use?**
- **Heuristic** — fast, no model calls. Good default. Used in `evaluate.py`.
- **Logprob** — uses model confidence. Better signal but slower (requires
  a forward pass per score). Best for harder problems where format-based
  scoring isn't enough.

---

## How It All Fits Together

The complete inference-time scaling toolkit:

```
Chapter 3:  Greedy baseline                    15% accuracy (book, 500 problems)
    │
Chapter 4:  + CoT prompting                   40% accuracy
    │       + Self-consistency (n=10)          52% accuracy (with CoT)
    │
Chapter 5:  + Self-refinement                  Can fix wrong → right
    │
Chapter 6:  Reinforcement learning             Train weights (coming next)
```

Each technique is independent and can be combined:

| Mode | What happens | Cost |
|------|-------------|------|
| `greedy` | Single deterministic answer | 1× |
| `cot` | Longer reasoning via prompt | ~5× tokens |
| `vote` | N independent samples → majority | N× |
| `refine` | 1 answer → critique → revise × K iterations | ~3K× |
| `cot + vote` | CoT prompts + majority voting | ~5N× |
| `cot + refine` | CoT prompts + iterative refinement | ~15K× |

---

## Running the Evaluation

```bash
cd ~/git/tokens/reasoning

# Self-refinement with 2 iterations, heuristic scoring
uv run python evaluate.py --mode refine --n 5 --samples 2

# More iterations for harder problems
uv run python evaluate.py --mode refine --n 5 --samples 3

# With verbose output to see the critique/revision process
uv run python evaluate.py --mode refine --n 3 --samples 2 --verbose

# Compare all modes
uv run python evaluate.py --mode greedy --n 10
uv run python evaluate.py --mode cot --n 10
uv run python evaluate.py --mode vote --n 10 --samples 5
uv run python evaluate.py --mode refine --n 10 --samples 2
```

The `--samples` flag means different things per mode:
- `vote`: number of independent samples
- `refine`: number of critique-revise iterations

---

## Code Walkthrough

### `src/scoring.py`

**`heuristic_score(answer, prompt)`**
Rule-based scorer. Rewards `\boxed{}` format (+2), numeric answers (+1),
and brevity (exponential decay). No model calls needed — instant.

**`logprob_score(model, tokenizer, prompt, answer, device)`**
Feeds prompt+answer through the model, extracts log-probabilities for
answer tokens only, returns the mean. Requires a forward pass — slower
but measures actual model confidence.

### `src/inference.py`

**`_critique_prompt(question, draft)`**
Builds the prompt that asks the model to find errors and propose fixes.

**`_refine_prompt(question, draft, critique)`**
Builds the prompt that asks the model to write a revised answer using
the critique feedback.

**`self_refine(model, tokenizer, raw_question, formatted_prompt, ...)`**
The main loop:
1. Generate initial answer and score it
2. For each iteration: critique → revise → score → accept if better
3. Return best answer with full history

Returns a dict with `answer`, `extracted`, `score`, and `history`
(all intermediate answers and scores for inspection).

---

## What Comes Next

Everything so far has been **inference-time scaling** — spending more
compute at generation time with a frozen model. The accuracy ceiling
is limited by what the base model already knows.

**Chapter 6 changes the game: we modify the model's weights.**

Using reinforcement learning (specifically GRPO), we'll train Qwen3-1.7B
to reason better by:
1. Generating answers to math problems
2. Using the math verifier from Chapter 3 as a **reward signal**
3. Updating weights to make correct-answer paths more likely

This is how models like DeepSeek-R1 were trained. The verifier we built
in Chapter 3 isn't just for evaluation — it becomes the reward function
that teaches the model to reason.

```
 [Chapter 5] Self-refinement (inference-time)  ← DONE
      │
      ▼
 [Chapter 6] Reinforcement learning (GRPO)     ← NEXT
      │       Train weights using math verifier as reward
      │       The model learns to generate reasoning traces
      ▼
 [Chapter 7] Distillation
      │       Transfer reasoning to smaller models
      ▼
 [Chapter 8] Advanced techniques
```
