"""Inference-time scaling techniques.

Methods that improve reasoning accuracy without modifying model weights:

  1. Chain-of-thought prompting — append "Explain step by step." to elicit reasoning
  2. Self-consistency voting  — sample N answers, pick the majority answer

These techniques trade more compute at inference time for better accuracy.
"""

from collections import Counter

import torch

from .generate import generate
from .verify import extract_answer


# -----------------------------------------------------------------------
# 1. Chain-of-thought prompting
# -----------------------------------------------------------------------

def cot_prompt(prompt: str) -> str:
    """Append a chain-of-thought instruction to a prompt."""
    return prompt + " \n\nExplain step by step."


# -----------------------------------------------------------------------
# 2. Self-consistency (majority voting)
# -----------------------------------------------------------------------

def self_consistency(
    model,
    tokenizer,
    prompt: str,
    device,
    num_samples: int = 5,
    temperature: float = 0.8,
    top_p: float = 0.9,
    max_tokens: int = 2048,
    seed: int | None = None,
    verbose: bool = False,
) -> dict:
    """Generate multiple answers and select the most frequent one.

    For each sample:
      1. Generate a response with temperature > 0 (diverse sampling)
      2. Extract the final answer (boxed or numeric fallback)
    Then pick the answer that appears most often (majority vote).

    Args:
        num_samples: how many independent answers to generate
        temperature: sampling temperature (>0 required for diversity)
        top_p: nucleus sampling threshold
        seed: base random seed (each sample uses seed+i for reproducibility)
        verbose: print each sample's extracted answer

    Returns:
        dict with keys:
          - responses: list of full response strings
          - answers: list of extracted short answers
          - counts: dict of answer → frequency
          - winner: the majority answer (or None on tie)
    """
    responses: list[str] = []
    answers: list[str] = []

    for i in range(num_samples):
        if seed is not None:
            torch.manual_seed(seed + i + 1)

        resp = generate(
            model, tokenizer, prompt, device,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        answer = extract_answer(resp)
        responses.append(resp)
        answers.append(answer)

        if verbose:
            print(f"  [Sample {i+1}/{num_samples}] → {answer!r}")

    # Majority vote
    counts = Counter(answers)
    if not counts:
        return {"responses": responses, "answers": answers, "counts": {},
                "winner": None}

    mc = counts.most_common()
    top_freq = mc[0][1]
    winners = [ans for ans, freq in mc if freq == top_freq]

    # Break ties by first occurrence order
    if len(winners) == 1:
        winner = winners[0]
    else:
        for ans in answers:
            if ans in winners:
                winner = ans
                break

    return {
        "responses": responses,
        "answers": answers,
        "counts": dict(counts),
        "winner": winner,
    }
