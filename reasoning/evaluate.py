"""Evaluate a model on the MATH-500 benchmark.

Usage:
    uv run python evaluate.py                          # greedy, 10 problems
    uv run python evaluate.py --mode cot               # chain-of-thought
    uv run python evaluate.py --mode vote --samples 5  # self-consistency voting
    uv run python evaluate.py --n 0                    # full 500 problems
    uv run python evaluate.py --verbose                # see generated answers
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src.config import MODELS
from src.generate import generate, pick_device
from src.inference import cot_prompt, self_consistency, self_refine
from src.scoring import heuristic_score
from src.tokenizer import Tokenizer
from src.verify import extract_answer, grade
from src.weights import load_hf_model

ROOT = Path(__file__).resolve().parent


# -----------------------------------------------------------------------
# Prompt template
# -----------------------------------------------------------------------

def render_prompt(problem: str) -> str:
    return (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{problem}\n\nAnswer:"
    )


# -----------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------

def load_math500(path: Path | None = None) -> list[dict]:
    if path is None:
        path = ROOT / "math500_test.json"
    if path.exists():
        with path.open("r") as f:
            return json.load(f)

    import urllib.request
    url = (
        "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "main/ch03/01_main-chapter-code/math500_test.json"
    )
    print("Downloading MATH-500 dataset...")
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode())
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {path} ({len(data)} problems)")
    return data


# -----------------------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------------------

def evaluate(
    model, tokenizer, device, data,
    mode: str = "greedy",
    max_tokens: int = 2048,
    num_samples: int = 5,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seed: int = 42,
    verbose: bool = False,
    out_path: Path | None = None,
):
    """Run evaluation on math problems.

    Modes:
      - 'greedy':  deterministic argmax decoding
      - 'cot':     chain-of-thought prompting (greedy)
      - 'vote':    self-consistency with majority voting
      - 'refine':  self-refinement with iterative critique
    """
    n = len(data)
    correct = 0
    start = time.time()

    if out_path is None:
        out_path = ROOT / f"math500_{mode}.jsonl"

    print(f"Mode: {mode} | Problems: {n}")
    if mode == "vote":
        print(f"  samples={num_samples}, temp={temperature}, top_p={top_p}")
    if mode == "refine":
        print(f"  iterations={num_samples}, temp={temperature}, top_p={top_p}")

    with open(out_path, "w") as f:
        for i, row in enumerate(data, 1):
            prompt = render_prompt(row["problem"])

            if mode == "cot":
                prompt = cot_prompt(prompt)

            if mode == "vote":
                result = self_consistency(
                    model, tokenizer, prompt, device,
                    num_samples=num_samples,
                    temperature=temperature, top_p=top_p,
                    max_tokens=max_tokens,
                    seed=seed + i,
                    verbose=verbose,
                )
                predicted = result["winner"] or ""
                response = f"votes: {result['counts']}"
            elif mode == "refine":
                if seed is not None:
                    torch.manual_seed(seed + i)
                result = self_refine(
                    model, tokenizer,
                    raw_question=row["problem"],
                    formatted_prompt=prompt,
                    device=device,
                    iterations=num_samples,  # reuse --samples as iteration count
                    max_tokens=max_tokens,
                    temperature=temperature, top_p=top_p,
                    score_fn=heuristic_score,
                    verbose=verbose,
                )
                predicted = result["extracted"]
                response = f"score: {result['score']:.3f}"
            else:
                response = generate(
                    model, tokenizer, prompt, device,
                    max_tokens=max_tokens, verbose=verbose,
                )
                predicted = extract_answer(response)

            is_correct = grade(predicted, row["answer"])
            correct += int(is_correct)

            record = {
                "index": i,
                "problem": row["problem"],
                "ground_truth": row["answer"],
                "predicted": predicted,
                "correct": is_correct,
                "mode": mode,
            }
            if mode != "vote":
                record["response"] = response
            else:
                record["votes"] = result["counts"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            elapsed = time.time() - start
            eta = (elapsed / i) * (n - i)
            mark = "+" if is_correct else "x"
            print(
                f"\r  [{mark}] {i}/{n} | acc={correct/i*100:.1f}% | ETA={eta:.0f}s",
                end="", flush=True,
            )

            if verbose and mode != "vote":
                print(
                    f"\n    predicted: {predicted}"
                    f"\n    expected:  {row['answer']}"
                    f"\n    {'CORRECT' if is_correct else 'WRONG'}\n"
                )

    acc = correct / n if n else 0
    elapsed = time.time() - start
    print(f"\n\nResults: {correct}/{n} = {acc*100:.1f}%")
    print(f"Time: {elapsed/60:.1f} min")
    print(f"Log: {out_path}")
    return correct, n, acc


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate on MATH-500")
    parser.add_argument("--model", default="qwen3-1.7b", choices=list(MODELS))
    parser.add_argument("--mode", default="greedy", choices=["greedy", "cot", "vote", "refine"])
    parser.add_argument("--n", type=int, default=10, help="Number of problems (0=all)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--samples", type=int, default=5, help="Samples for vote mode")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = pick_device()
    model, model_dir = load_hf_model(args.model, device=device, local_dir=ROOT / "models")
    tok = Tokenizer(model_dir / "tokenizer.json")

    data = load_math500()
    if args.n > 0:
        data = data[:args.n]

    print(f"\nEvaluating {args.model} on {len(data)} MATH-500 problems\n")
    evaluate(
        model, tok, device, data,
        mode=args.mode,
        max_tokens=args.max_tokens,
        num_samples=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
