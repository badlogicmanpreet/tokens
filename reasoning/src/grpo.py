"""Group Relative Policy Optimization (GRPO) for reasoning training.

Implements RLVR (Reinforcement Learning with Verifiable Rewards) using
GRPO — the algorithm behind DeepSeek-R1 and similar reasoning models.

The pipeline for each training step:
  1. Sample N rollouts (answers) per prompt
  2. Compute rewards using the math verifier
  3. Convert rewards to group-relative advantages
  4. Compute sequence log-probabilities for each rollout
  5. Compute policy gradient loss = -(advantages * logprobs).mean()
  6. Update model weights via backpropagation
"""

import torch
import torch.nn.functional as F

from .cache import KVCache
from .generate import _top_p_filter
from .verify import extract_answer, grade


# -----------------------------------------------------------------------
# 1. Rollout sampling (generate answers without grad tracking)
# -----------------------------------------------------------------------

@torch.no_grad()
def sample_rollout(
    model, tokenizer, prompt, device,
    max_tokens=512, temperature=0.8, top_p=0.9,
):
    """Generate a single rollout (complete answer) for a prompt.

    Uses KV cache with temperature/top-p sampling but no gradient tracking.
    Returns (full_token_ids, prompt_length, answer_text).
    """
    input_ids = torch.tensor(tokenizer.encode(prompt), device=device)

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_cache_state()
    logits = model(input_ids.unsqueeze(0), cache=cache)[:, -1]

    generated = []
    for _ in range(max_tokens):
        if temperature and temperature != 0.0:
            scaled = logits / temperature
        else:
            scaled = logits
        probs = F.softmax(scaled.float(), dim=-1)
        probs = _top_p_filter(probs, top_p)
        next_tok = torch.multinomial(probs.cpu(), num_samples=1).to(device)

        if tokenizer.eos_id is not None and next_tok.item() == tokenizer.eos_id:
            break

        generated.append(next_tok.item())
        logits = model(next_tok, cache=cache)[:, -1]

    full_ids = torch.cat([
        input_ids,
        torch.tensor(generated, device=device, dtype=input_ids.dtype),
    ])
    answer_text = tokenizer.decode(generated)
    return full_ids, input_ids.numel(), answer_text


# -----------------------------------------------------------------------
# 2. Reward function (uses math verifier from Ch.3)
# -----------------------------------------------------------------------

def compute_reward(answer_text: str, ground_truth: str) -> float:
    """Reward = 1.0 if answer is correct AND in \\boxed{} format, else 0.0."""
    extracted = extract_answer(answer_text, fallback="none")
    if not extracted:
        return 0.0
    return float(grade(extracted, ground_truth))


# -----------------------------------------------------------------------
# 3. Advantages (group-relative normalization)
# -----------------------------------------------------------------------

def compute_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Normalize rewards relative to the group mean and std.

    Positive advantage = better than average in this group.
    Negative advantage = worse than average.
    Zero advantage = all rollouts got the same reward (no learning signal).
    """
    return (rewards - rewards.mean()) / (rewards.std() + 1e-4)


# -----------------------------------------------------------------------
# 4. Sequence log-probabilities (with gradient tracking)
# -----------------------------------------------------------------------

def sequence_logprob(model, token_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Compute the sum of log-probabilities over answer tokens.

    Unlike the scoring version in scoring.py, this does NOT use
    @torch.inference_mode() — gradients flow through for training.
    """
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = F.log_softmax(logits, dim=-1)
    # Use gather for efficiency: select logprob of each actual next token
    selected = logprobs[:-1].gather(1, token_ids[1:].unsqueeze(-1)).squeeze(-1)
    return selected[prompt_len - 1:].sum()


# -----------------------------------------------------------------------
# 5. GRPO loss (combines everything)
# -----------------------------------------------------------------------

def grpo_loss(
    model, tokenizer, prompt, ground_truth, device,
    num_rollouts=4, max_tokens=512, temperature=0.8, top_p=0.9,
):
    """Compute the GRPO policy gradient loss for one prompt.

    Steps:
      1. Sample N rollouts
      2. Compute rewards
      3. Compute advantages
      4. Compute sequence logprobs (with gradients)
      5. Return loss = -(advantages * logprobs).mean()

    Returns (loss, avg_reward, avg_response_length) or (None, 0, 0)
    if all rewards are identical (no learning signal).
    """
    model.eval()  # eval mode for rollout sampling (dropout off, etc.)

    # Step 1: Sample rollouts
    rollouts = []
    for _ in range(num_rollouts):
        full_ids, prompt_len, answer_text = sample_rollout(
            model, tokenizer, prompt, device,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
        )
        rollouts.append((full_ids, prompt_len, answer_text))

    # Step 2: Compute rewards
    rewards_list = [compute_reward(text, ground_truth) for _, _, text in rollouts]
    rewards = torch.tensor(rewards_list, device=device)
    avg_reward = rewards.mean().item()
    avg_len = sum(len(text) for _, _, text in rollouts) / num_rollouts

    # Step 3: Compute advantages
    if rewards.std() < 1e-6:
        # All rewards identical → no learning signal
        return None, avg_reward, avg_len

    advantages = compute_advantages(rewards)

    # Step 4: Compute sequence logprobs (with gradients)
    model.train()
    logprobs = []
    for full_ids, prompt_len, _ in rollouts:
        lp = sequence_logprob(model, full_ids, prompt_len)
        logprobs.append(lp)

    logps = torch.stack(logprobs)

    # Step 5: Policy gradient loss
    loss = -(advantages.detach() * logps).mean()

    return loss, avg_reward, avg_len


# -----------------------------------------------------------------------
# 6. Training loop
# -----------------------------------------------------------------------

def train_grpo(
    model, tokenizer, train_data, device,
    steps=50, num_rollouts=4, max_tokens=512,
    temperature=0.8, top_p=0.9, lr=1e-5,
    checkpoint_every=10, checkpoint_dir="checkpoints",
    render_prompt_fn=None,
):
    """Train the model using GRPO with verifiable rewards.

    Args:
        model: the TransformerLM to train
        train_data: list of dicts with 'problem' and 'answer' keys
        steps: number of training steps
        num_rollouts: answers to sample per prompt
        max_tokens: max tokens per rollout
        lr: learning rate
        checkpoint_every: save model every N steps
        render_prompt_fn: function to format problem into prompt
    """
    from pathlib import Path
    import time

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if render_prompt_fn is None:
        def render_prompt_fn(problem):
            return (
                "You are a helpful math assistant.\n"
                "Answer the question and write the final result on a new line as:\n"
                "\\boxed{ANSWER}\n\n"
                f"Question:\n{problem}\n\nAnswer:"
            )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"Training GRPO: {steps} steps, {num_rollouts} rollouts/step, "
          f"max_tokens={max_tokens}, lr={lr}")

    for step in range(1, steps + 1):
        # Pick a training example (cycle through data)
        row = train_data[(step - 1) % len(train_data)]
        prompt = render_prompt_fn(row["problem"])

        torch.manual_seed(step)

        loss, avg_reward, avg_len = grpo_loss(
            model, tokenizer, prompt, row["answer"], device,
            num_rollouts=num_rollouts, max_tokens=max_tokens,
            temperature=temperature, top_p=top_p,
        )

        if loss is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.item()
        else:
            loss_val = 0.0

        print(f"[Step {step}/{steps}] loss={loss_val:.4f} "
              f"reward_avg={avg_reward:.3f} avg_resp_len={avg_len:.1f}")

        # Checkpoint
        if step % checkpoint_every == 0:
            path = ckpt_dir / f"grpo-step{step:05d}.pth"
            torch.save(model.state_dict(), path)
            print(f"  Saved checkpoint: {path}")

    # Final checkpoint
    path = ckpt_dir / f"grpo-step{steps:05d}-final.pth"
    torch.save(model.state_dict(), path)
    print(f"Training complete. Final checkpoint: {path}")
    return path
