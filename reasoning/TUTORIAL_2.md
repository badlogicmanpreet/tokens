# Evaluating Reasoning Models — Building a Math Verifier

How to measure whether an LLM can actually solve math problems, not just
generate convincing-looking text.

Model: **Qwen3-1.7B** (base, no reasoning training)

---

## Table of Contents

1. [Why Evaluation Matters](#why-evaluation-matters)
2. [The Verification Pipeline](#the-verification-pipeline)
3. [Step-by-Step Walkthrough](#step-by-step-walkthrough)
   - [Prompt Template](#1-prompt-template)
   - [Answer Extraction](#2-answer-extraction)
   - [Normalization](#3-normalization)
   - [Symbolic Grading](#4-symbolic-grading-with-sympy)
4. [MATH-500 Benchmark](#math-500-benchmark)
5. [Running the Evaluation](#running-the-evaluation)
6. [Results and Analysis](#results-and-analysis)
7. [Why This Matters for Reasoning](#why-this-matters-for-reasoning)

---

## Why Evaluation Matters

An LLM can generate text that *looks* correct but is mathematically wrong.
Evaluation gives us a concrete number — accuracy — that tells us how often
the model actually gets the right answer. Without this, we can't measure
whether our reasoning improvements (Chapters 4-8) are working.

We focus on math because:
- Answers are **objectively verifiable** (unlike creative writing)
- Problems benefit from **step-by-step reasoning** (chain-of-thought)
- The same verifier becomes the **reward signal** for RL training in Chapter 6

---

## The Verification Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Math Problem │────→│  LLM         │────→│  Raw Output  │
│  from dataset │     │  (Qwen3-1.7B)│     │  with LaTeX  │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
         ┌────────────────────────────────────────┘
         ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Extract      │────→│  Normalize   │────→│  Grade       │
│  \boxed{...}  │     │  LaTeX→plain │     │  via SymPy   │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                          True or False
```

Each step in `src/verify.py`:

| Step | Function | What it does |
|------|----------|-------------|
| Extract | `extract_answer()` | Pulls the last `\boxed{...}` from LLM output |
| Normalize | `normalize()` | Strips LaTeX, converts to calculator-style math |
| Grade | `grade()` | Compares prediction to ground truth symbolically |

---

## Step-by-Step Walkthrough

### 1. Prompt Template

We wrap each math problem in a template that instructs the model to put its
final answer in a `\boxed{}`:

```python
def render_prompt(problem):
    return (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{problem}\n\nAnswer:"
    )
```

**Example input:**
```
You are a helpful math assistant.
Answer the question and write the final result on a new line as:
\boxed{ANSWER}

Question:
If $a+b=3$ and $ab=\tfrac{13}{6}$, what is the value of $a^2+b^2$?

Answer:
```

The model then generates a response that (hopefully) includes `\boxed{\frac{14}{3}}`.

### 2. Answer Extraction

The model's output is a long string with reasoning steps and a boxed answer.
We need to pull out just the answer.

**`extract_boxed(text)`** — finds the last `\boxed{...}` and handles nested braces:

```
Input:  "... Therefore \boxed{\dfrac{14}{3}} is the answer"
Output: "\dfrac{14}{3}"
```

Nested braces matter: `\boxed{\frac{14}{3}}` has braces inside braces.
A naive search for the first `}` would fail. The parser tracks brace depth:

```python
depth = 1
while depth > 0:
    if char == "{": depth += 1
    if char == "}": depth -= 1
```

**`extract_answer(text)`** — tries boxed first, falls back to finding
the last number in the text:

```python
extract_answer(r"Final: \boxed{42}")        # → "42"
extract_answer("The result is 14/3 done")   # → "14/3" (fallback)
extract_answer("No numbers here")           # → "No numbers here" (full text)
```

### 3. Normalization

The same mathematical value can be written many ways in LaTeX:

```
\dfrac{14}{3}     \frac{14}{3}     14/3     (14)/(3)
```

`normalize()` converts all of these to a single canonical form:

```python
normalize(r"\dfrac{14}{3}")    # → "(14)/(3)"
normalize(r"\sqrt{8}")         # → "sqrt(8)"
normalize(r"2^{10}")           # → "2**10"
normalize(r"90^\circ")         # → "90"
normalize(r"1,234")            # → "1234"
```

**What it does, in order:**
1. Strip special tokens (`<|assistant|>`, etc.)
2. Remove multiple-choice labels (`"c. 3"` → `"3"`)
3. Remove degree markers (`°`, `^\circ`)
4. Unwrap `\text{...}`
5. Strip math delimiters (`\(`, `\)`, `\[`, `\]`)
6. Apply LaTeX substitutions (`\cdot` → `*`, `\dfrac` → `\frac`)
7. Convert unicode superscripts (`²` → `**2`)
8. Convert `\sqrt{x}` → `sqrt(x)`
9. Convert `\frac{a}{b}` → `(a)/(b)`
10. Convert `^` → `**`
11. Handle mixed numbers (`2 3/4` → `2+3/4`)
12. Remove thousands separators (`1,234` → `1234`)
13. Strip braces, lowercase

### 4. Symbolic Grading with SymPy

String comparison isn't enough — `"14/3"` and `"(14)/(3)"` are the same
value but different strings. Even `"28/6"` equals `"14/3"`.

We use SymPy to parse both strings into symbolic expressions and check
if their difference simplifies to zero:

```python
from sympy import simplify
from sympy.parsing import sympy_parser as spp

def _exprs_equal(a_str, b_str):
    if a_str == b_str:          # fast path: exact string match
        return True
    a = spp.parse_expr(a_str)   # parse to SymPy expression
    b = spp.parse_expr(b_str)
    return simplify(a - b) == 0 # algebraically equivalent?
```

**The `grade()` function** also handles tuples — answers like `(3, π/2)`:

```python
grade("14/3", r"\frac{14}{3}")          # True  (same value)
grade("0.5", "1/2")                      # True  (0.5 == 1/2)
grade("(14/3, 2/3)", "(14/3, 4/6)")     # True  (2/3 == 4/6)
grade("15/3", "14/3")                    # False (different values)
```

It splits tuples into parts and checks each one independently.

---

## MATH-500 Benchmark

MATH-500 is a curated set of 500 math problems spanning multiple
difficulty levels and subjects:

```python
{
    "problem": "Convert the point (0,3) in rectangular coordinates to polar...",
    "answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
    "solution": "We have that r = \\sqrt{0^2 + 3^2} = 3...",
    "subject": "Precalculus",
    "level": 2,
}
```

**Subjects:** Algebra, Precalculus, Number Theory, Geometry, Counting &
Probability, Intermediate Algebra, Prealgebra

**Levels:** 1 (easiest) to 5 (hardest)

The dataset is auto-downloaded on first run and cached locally as
`math500_test.json`.

---

## Running the Evaluation

```bash
cd ~/git/tokens/reasoning

# Quick test: 10 problems (~7 min on Apple Silicon)
uv run python evaluate.py --n 10

# See the model's work in real time
uv run python evaluate.py --n 10 --verbose

# Full benchmark: all 500 problems
uv run python evaluate.py --n 0

# Shorter responses (faster but may cut off answers)
uv run python evaluate.py --n 10 --max-tokens 512
```

Results are saved to `math500_results.jsonl` — one JSON object per line:

```json
{
    "index": 3,
    "problem": "If f(x)=(3x-2)/(x-2), what is the value of f(-2)+f(-1)+f(0)?",
    "ground_truth": "\\frac{14}{3}",
    "response": "... step by step work ...",
    "predicted": "\\frac{14}{3}",
    "correct": true
}
```

---

## Results and Analysis

**Qwen3-1.7B base model on 10 MATH-500 problems: 40% accuracy**

```
  [+] Problem 3:  f(-2)+f(-1)+f(0) = 14/3           ✓ correct
  [+] Problem 4:  divisors of 196 = 9                ✓ correct
  [+] Problem 7:  smallest cube as 3 consecutive = 27 ✓ correct
  [+] Problem 9:  distance formula = 3√13            ✓ correct
  [x] Problem 1:  polar coordinates — answer truncated
  [x] Problem 2:  double sum — ran out of tokens
  [x] Problem 6:  hexagon perimeter — wrong reasoning
  [x] Problem 8:  angle between lines — incomplete
```

**Why is it only 40%?** This is a base model with no reasoning training:
- It tends to **repeat** the boxed answer endlessly instead of stopping
- Some problems need more tokens than the limit allows
- Complex multi-step problems trip up the base model's pattern matching
- No chain-of-thought training means it often jumps to wrong conclusions

**This is the baseline.** Every technique in the remaining chapters aims
to push this number higher:

```
Chapter 4-5:  Inference-time scaling  → ~2x improvement expected
Chapter 6:    Reinforcement learning  → significant accuracy gains
Chapter 7:    Distillation            → efficient reasoning
```

---

## Why This Matters for Reasoning

The math verifier we built here serves two purposes:

1. **Evaluation tool** — measures how well the model reasons about math,
   giving us a concrete accuracy number to track improvements

2. **Reward signal for RL** — in Chapter 6, we'll train the model using
   reinforcement learning. The reward function is essentially:
   ```
   reward = 1.0 if grade(model_answer, correct_answer) else 0.0
   ```
   The model learns to produce answers that pass this verifier.

This is the core insight behind reasoning models like DeepSeek-R1: you
don't need human feedback to train math reasoning. A deterministic
verifier is cheaper, faster, and more consistent than human evaluators.
