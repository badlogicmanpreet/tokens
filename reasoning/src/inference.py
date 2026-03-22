"""Inference-time scaling techniques.

Methods that improve reasoning accuracy without modifying model weights:

  1. Chain-of-thought prompting — append "Explain step by step." to elicit reasoning
  2. Self-consistency voting  — sample N answers, pick the majority answer
  3. Self-refinement          — critique and revise answers iteratively

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


# -----------------------------------------------------------------------
# 3. Self-refinement (critique → revise → iterate)
# -----------------------------------------------------------------------

def _critique_prompt(question: str, draft: str) -> str:
    return (
        "You are a meticulous reviewer. Identify logical errors, missing "
        "steps, or arithmetic mistakes. If the answer seems correct, "
        "say so briefly. Then propose a concise plan to fix issues.\n\n"
        f"Question:\n{question}\n\n"
        f"Draft answer:\n{draft}\n\n"
        "Write a short critique and bullet-point fix plan "
        "(under ~120 words).\n"
        "Critique:"
    )


def _refine_prompt(question: str, draft: str, critique: str) -> str:
    return (
        "Revise the answer using the critique. Keep it concise and "
        "end with a final boxed result: \\boxed{ANSWER}\n\n"
        f"Question:\n{question}\n\n"
        f"Previous answer:\n{draft}\n\n"
        f"Critique:\n{critique}\n\n"
        "Revised answer:"
    )


def self_refine(
    model,
    tokenizer,
    raw_question: str,
    formatted_prompt: str,
    device,
    iterations: int = 2,
    max_tokens: int = 2048,
    max_critique_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    score_fn=None,
    verbose: bool = False,
) -> dict:
    """Iteratively critique and revise an answer.

    Process:
      1. Generate initial answer
      2. For each iteration:
         a. Generate a critique of the current answer
         b. Generate a revised answer using the critique
         c. If score_fn provided, keep revision only if score improves
      3. Return the best answer

    Args:
        raw_question: the original math problem (without template)
        formatted_prompt: the full prompt with template applied
        score_fn: callable(answer, prompt) → float. If None, always accept revisions.
        iterations: number of critique-revise cycles

    Returns:
        dict with keys: answer, extracted, score, history
    """
    # Step 1: initial answer
    current = generate(
        model, tokenizer, formatted_prompt, device,
        max_tokens=max_tokens, temperature=temperature, top_p=top_p,
    )
    current_extracted = extract_answer(current)
    current_score = score_fn(answer=current, prompt=formatted_prompt) if score_fn else 0.0

    history = [{
        "iteration": 0,
        "answer": current,
        "extracted": current_extracted,
        "score": current_score,
    }]

    if verbose:
        print(f"  [iter 0] extracted={current_extracted!r}  score={current_score:.3f}")

    # Step 2: iterate critique → refine
    for it in range(1, iterations + 1):
        # Critique
        cp = _critique_prompt(raw_question, current)
        critique = generate(
            model, tokenizer, cp, device,
            max_tokens=max_critique_tokens, temperature=temperature, top_p=top_p,
        )

        # Refine
        rp = _refine_prompt(raw_question, current, critique)
        revised = generate(
            model, tokenizer, rp, device,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
        )
        revised_extracted = extract_answer(revised)
        revised_score = score_fn(answer=revised, prompt=formatted_prompt) if score_fn else 0.0

        history.append({
            "iteration": it,
            "answer": revised,
            "extracted": revised_extracted,
            "score": revised_score,
            "critique": critique,
        })

        if verbose:
            print(f"  [iter {it}] extracted={revised_extracted!r}  score={revised_score:.3f}")

        # Accept revision if score improves (or no scorer)
        if score_fn is None or revised_score >= current_score:
            current = revised
            current_extracted = revised_extracted
            current_score = revised_score

    return {
        "answer": current,
        "extracted": current_extracted,
        "score": current_score,
        "history": history,
    }
