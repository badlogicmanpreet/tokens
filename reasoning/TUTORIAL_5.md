# Training a Reasoning Model with Reinforcement Learning

How to train Qwen3-1.7B to reason about math using GRPO — the same algorithm
behind DeepSeek-R1. This is where we actually modify the model's weights.

---

## Table of Contents

1. [Why RL for Reasoning?](#why-rl-for-reasoning)
2. [RLHF vs RLVR](#rlhf-vs-rlvr)
3. [The GRPO Algorithm — Step by Step](#the-grpo-algorithm)
4. [Code Walkthrough](#code-walkthrough)
5. [Running Training](#running-training)
6. [What to Expect](#what-to-expect)
7. [What Comes Next](#what-comes-next)

---

## Why RL for Reasoning?

Everything in Chapters 4-5 was **inference-time scaling** — we made the model
use more compute at generation time, but never changed its weights. The ceiling
is limited by what the model already knows.

Now we cross that boundary. Reinforcement learning **modifies the weights**
to make the model better at reasoning:

```
Pre-training:          "Learn language patterns from text"  (next-token prediction)
RL for reasoning:      "Learn to solve math problems"      (reward for correct answers)
```

The key insight: we already have a verifier from Chapter 3 that can check if
a math answer is correct. We can use this as a reward signal — the model
generates answers, correct ones get reward +1, wrong ones get 0. Over many
training steps, the model learns to produce correct answers more often.

---

## RLHF vs RLVR

Two flavors of RL for LLMs:

### RLHF (RL with Human Feedback)
Used for preference tuning (making the model polite, helpful, etc.):
```
1. Human ranks model outputs: "Response B is better than A"
2. Train a reward model to predict these rankings
3. Use reward model scores to train the LLM
```
Complex — requires human annotators and a separate reward model.

### RLVR (RL with Verifiable Rewards)
Used for reasoning training (math, code, logic):
```
1. Model generates answer to a math problem
2. Verifier checks: correct answer = reward 1.0, wrong = 0.0
3. Use this reward directly to train the LLM
```
Simpler — no human annotators, no reward model. Just a deterministic
verifier. This is what DeepSeek-R1 used, and what we implement here.

**We use RLVR because:**
- Math has objectively correct answers (no need for human judgment)
- We already built the verifier in Chapter 3
- It's simpler to implement and understand

---

## The GRPO Algorithm

GRPO = Group Relative Policy Optimization. Six stages per training step:

```
┌──────────────────────────────────────────────────────┐
│                   One GRPO Step                       │
│                                                       │
│  1. SAMPLE       Generate N rollouts (answers)        │
│       │          for the same math problem            │
│       ▼                                               │
│  2. REWARD       Check each answer with verifier      │
│       │          correct=1.0, wrong=0.0               │
│       ▼                                               │
│  3. ADVANTAGE    Normalize rewards within the group   │
│       │          better than avg → positive            │
│       │          worse than avg → negative             │
│       ▼                                               │
│  4. LOGPROB      Compute how likely the model thinks  │
│       │          each rollout is (with gradients)      │
│       ▼                                               │
│  5. LOSS         loss = -(advantages × logprobs).mean │
│       │          Maximize logprob of good answers      │
│       │          Minimize logprob of bad answers       │
│       ▼                                               │
│  6. UPDATE       Backpropagate → optimizer.step()     │
│                  Model weights change                  │
└──────────────────────────────────────────────────────┘
```

### Stage 1: Sample Rollouts

"Rollout" = RL term for a complete model response. We generate N answers
for the same prompt using temperature sampling:

```
Prompt: "What is the value of x if 3x-9 = x+37?"

  Rollout 1: \boxed{83}                    (correct)
  Rollout 2: The correct answer is \boxed{83}  (correct)
  Rollout 3: The final answer is 83             (no box — no reward)
  Rollout 4: We get \boxed{38}                  (wrong)
```

### Stage 2: Compute Rewards

Using our Chapter 3 verifier. An answer gets reward 1.0 only if it's both
correct AND in `\boxed{}` format:

```
  Rollout 1: reward = 1.0  ✓ correct + boxed
  Rollout 2: reward = 1.0  ✓ correct + boxed
  Rollout 3: reward = 0.0  ✗ correct but no box
  Rollout 4: reward = 0.0  ✗ wrong
```

### Stage 3: Compute Advantages

Raw rewards tell us "good" or "bad". Advantages tell us "how much better
or worse than the group average":

```
rewards    = [1.0,  1.0,   0.0,    0.0]
mean       = 0.5
std        = 0.577

advantages = [ 0.87,  0.87,  -0.87,  -0.87]
               ↑ better than avg    ↑ worse than avg
```

**Why not just use rewards directly?** Advantages normalize across the
group. If all rollouts are correct (rewards=[1,1,1,1]), advantages are
all zero → no gradient → no update. This is correct behavior: the model
already knows this problem, no need to change weights.

### Stage 4: Compute Sequence Log-Probabilities

For each rollout, compute how likely the model thinks it is (sum of
per-token log-probs over the answer tokens):

```
  Rollout 1: logprob = -7.92   (short, confident)
  Rollout 2: logprob = -20.15  (longer, less confident)
  Rollout 3: logprob = -16.61
  Rollout 4: logprob = -23.37  (wrong answer, lowest confidence)
```

These are computed **with gradient tracking** — this is what connects
the loss back to the model weights.

### Stage 5: Policy Gradient Loss

```python
loss = -(advantages.detach() * logprobs).mean()
```

What this does:
- **Positive advantage × logprob** → gradient increases that rollout's
  probability (make correct answers more likely)
- **Negative advantage × logprob** → gradient decreases that rollout's
  probability (make wrong answers less likely)
- The negative sign converts maximization to minimization for PyTorch

### Stage 6: Weight Update

Standard PyTorch training:
```python
optimizer.zero_grad()
loss.backward()       # compute gradients
optimizer.step()      # update weights
```

After this step, the model is slightly better at producing correct,
boxed math answers.

---

## Code Walkthrough

### `src/grpo.py`

**`sample_rollout(model, tokenizer, prompt, device, ...)`**
Generates one complete answer using KV cache + temperature/top-p.
Uses `@torch.no_grad()` (not `@inference_mode()`) because we need
PyTorch's computation graph intact for the logprob step.

**`compute_reward(answer_text, ground_truth)`**
Returns 1.0 if answer is correct AND boxed, 0.0 otherwise.
Uses the same `extract_answer()` and `grade()` from Chapter 3.

**`compute_advantages(rewards)`**
`(rewards - mean) / (std + epsilon)` — group-relative normalization.

**`sequence_logprob(model, token_ids, prompt_len)`**
Sum of log-probabilities over answer tokens. No `@inference_mode`
decorator — gradients flow through this for backpropagation.

**`grpo_loss(model, tokenizer, prompt, ground_truth, device, ...)`**
Combines stages 1-5 into a single function call. Returns `(loss,
avg_reward, avg_response_length)`. Returns `None` for loss when all
rewards are identical (no learning signal from this group).

**`train_grpo(model, tokenizer, train_data, device, ...)`**
Full training loop: iterates over math problems, computes GRPO loss,
updates weights, saves checkpoints.

### `train.py`

CLI script that loads the model, downloads the MATH training set
(12K problems), and runs `train_grpo()`.

---

## Running Training

```bash
cd ~/git/tokens/reasoning

# Default: 50 steps, 4 rollouts per step
uv run python train.py

# Lighter run (less memory, faster per step)
uv run python train.py --rollouts 2 --max-tokens 256

# Longer training
uv run python train.py --steps 100 --checkpoint-every 25

# All options
uv run python train.py \
    --steps 50 \
    --rollouts 4 \
    --max-tokens 512 \
    --temperature 0.8 \
    --top-p 0.9 \
    --lr 1e-5 \
    --checkpoint-every 10
```

Checkpoints are saved to `checkpoints/grpo-stepNNNNN.pth`.

### Evaluating a checkpoint

```bash
# After training, evaluate the trained model on MATH-500
# (load checkpoint manually or modify evaluate.py to accept --checkpoint)
```

### Hardware requirements

| Setting | Approximate memory | Time per step |
|---------|-------------------|---------------|
| rollouts=2, max_tokens=256 | ~6 GB | ~1 min (MPS) |
| rollouts=4, max_tokens=512 | ~10 GB | ~3 min (MPS) |
| rollouts=8, max_tokens=512 | ~16 GB | ~6 min (MPS) |

If you run out of memory, reduce `--rollouts` and `--max-tokens`.

---

## What to Expect

### Training output

```
[Step 1/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=5.5
[Step 2/50] loss=-0.0000 reward_avg=0.000 avg_resp_len=6.8
[Step 3/50] loss=0.3592  reward_avg=0.250 avg_resp_len=7.8
[Step 4/50] loss=2.7401  reward_avg=0.250 avg_resp_len=56.5
[Step 5/50] loss=3.3214  reward_avg=0.500 avg_resp_len=251.2
```

**What the numbers mean:**
- `loss=0.0000` → all rollouts got same reward, no gradient signal
- `reward_avg=0.250` → 1 of 4 rollouts was correct
- `reward_avg=0.500` → 2 of 4 were correct
- `avg_resp_len` should grow over training as model learns to write
  step-by-step reasoning

**What to look for:**
- `reward_avg` should trend upward over many steps
- Response lengths should increase (model writes explanations)
- Loss fluctuation is normal in RL (not smooth like supervised training)

### Expected accuracy improvement

From the book's results (Qwen3-0.6B, 500 MATH problems):

| Model | MATH-500 Accuracy |
|-------|-------------------|
| Base model (no training) | 15.2% |
| After 50 GRPO steps (4 rollouts, 512 tokens) | 45.6% |
| After 50 GRPO steps (8 rollouts, 512 tokens) | 47.4% |
| Official reasoning model | 48.2% |

**50 steps of GRPO tripled the base model's accuracy** — from 15% to 47%,
nearly matching the official reasoning model trained by the Qwen team.

---

## What Comes Next

```
 [Chapter 6] Reinforcement learning (GRPO)     ← DONE (this chapter)
      │       Base model 15% → 47% on MATH-500
      ▼
 [Chapter 7] Distillation
      │       Transfer reasoning from a large model to a small one
      │       via supervised fine-tuning on reasoning traces
      ▼
 [Chapter 8] Advanced techniques
              Improving the training pipeline, KL divergence,
              future research directions
```

The model we trained here has learned to reason about math by discovering
that step-by-step solutions lead to correct answers. This is the same
emergence of reasoning that the DeepSeek team observed with R1 — the model
wasn't explicitly taught to show its work, but learned that doing so
increases its reward.
