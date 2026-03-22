"""Answer scoring functions for ranking and comparing LLM responses.

Two scoring approaches:
  1. Heuristic — rule-based: rewards boxed answers + brevity
  2. Logprob   — model confidence: average log-probability of answer tokens

These scorers are used by the self-refinement loop to decide whether a
revised answer is better than the previous one.
"""

import math

import torch
import torch.nn.functional as F

from .verify import extract_answer


# -----------------------------------------------------------------------
# 1. Heuristic (rule-based) scorer
# -----------------------------------------------------------------------

def heuristic_score(
    answer: str,
    prompt: str | None = None,
    brevity_decay: float = 500.0,
    boxed_bonus: float = 2.0,
    number_bonus: float = 1.0,
) -> float:
    """Score an answer based on format quality and brevity.

    Awards points for having a \\boxed{} answer (highest), a numeric
    answer (medium), or any answer at all (low). Adds a brevity bonus
    that decays exponentially with answer length.

    Returns a float score (higher is better).
    """
    score = 0.0

    # Reward based on answer extractability
    boxed = extract_answer(answer, fallback="none")
    if boxed:
        score += boxed_bonus
    else:
        num = extract_answer(answer, fallback="number_only")
        if num:
            score += number_bonus

    # Brevity: shorter answers get higher bonus (decays exponentially)
    score += 1.5 * math.exp(-len(answer) / brevity_decay)

    return score


# -----------------------------------------------------------------------
# 2. Logprob (model confidence) scorer
# -----------------------------------------------------------------------

@torch.inference_mode()
def logprob_score(
    model,
    tokenizer,
    prompt: str,
    answer: str,
    device,
) -> float:
    """Score an answer by the model's average log-probability of answer tokens.

    Feeds [prompt + answer] through the model, extracts the log-probability
    assigned to each answer token, and returns the mean. Higher (less
    negative) values indicate the model is more confident in the answer.

    This measures: "how likely does the model think this answer is,
    given the prompt?"
    """
    prompt_ids = tokenizer.encode(prompt)
    answer_ids = tokenizer.encode(answer)

    if not answer_ids:
        return -float("inf")

    full_ids = torch.tensor(prompt_ids + answer_ids, device=device)
    logits = model(full_ids.unsqueeze(0)).squeeze(0)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # Only score the answer tokens (skip the prompt portion)
    start = len(prompt_ids) - 1
    end = full_ids.shape[0] - 1

    positions = torch.arange(start, end, device=device)
    target_ids = full_ids[start + 1 : end + 1]
    answer_logprobs = log_probs[positions, target_ids]

    return answer_logprobs.mean().item()
