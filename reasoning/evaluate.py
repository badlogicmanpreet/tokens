"""Evaluate a model on the MATH-500 benchmark.

Usage:
    uv run python evaluate.py                     # default: qwen3-1.7b, 10 problems
    uv run python evaluate.py --n 500             # full benchmark
    uv run python evaluate.py --n 10 --verbose    # see generated answers live
"""

import argparse
import json
import time
from pathlib import Path

import torch

from src.config import MODELS
from src.generate import generate, pick_device
from src.tokenizer import Tokenizer
from src.verify import extract_answer, grade
from src.weights import load_hf_model

ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------
# Prompt template
# -----------------------------------------------------------------------

def render_prompt(problem: str) -> str:
    """Wrap a math problem in an evaluation prompt template."""
    return (
        "You are a helpful math assistant.\n"
        "Answer the question and write the final result on a new line as:\n"
        "\\boxed{ANSWER}\n\n"
        f"Question:\n{problem}\n\nAnswer:"
    )


# -----------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------

def load_math500(path: Path | None = None) -> list[dict]:
    """Load the MATH-500 dataset (downloads if not cached locally)."""
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
    print(f"Downloading MATH-500 dataset...")
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
    model,
    tokenizer,
    device,
    data: list[dict],
    max_tokens: int = 2048,
    verbose: bool = False,
    out_path: Path | None = None,
):
    """Run the evaluation pipeline on a list of math problems.

    For each problem:
      1. Render the prompt template
      2. Generate a response from the model
      3. Extract the predicted answer
      4. Grade it against the ground truth

    Returns (n_correct, n_total, accuracy).
    """
    n = len(data)
    correct = 0
    start = time.time()

    if out_path is None:
        out_path = ROOT / "math500_results.jsonl"

    with open(out_path, "w") as f:
        for i, row in enumerate(data, 1):
            prompt = render_prompt(row["problem"])

            # Generate
            response = generate(
                model, tokenizer, prompt, device,
                max_tokens=max_tokens, verbose=verbose,
            )

            # Extract and grade
            predicted = extract_answer(response)
            is_correct = grade(predicted, row["answer"])
            correct += int(is_correct)

            # Save result
            record = {
                "index": i,
                "problem": row["problem"],
                "ground_truth": row["answer"],
                "response": response,
                "predicted": predicted,
                "correct": is_correct,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Progress
            elapsed = time.time() - start
            eta = (elapsed / i) * (n - i) if i > 0 else 0
            mark = "+" if is_correct else "x"
            print(
                f"\r  [{mark}] {i}/{n} | acc={correct/i*100:.1f}% | ETA={eta:.0f}s",
                end="", flush=True,
            )

            if verbose:
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
    parser.add_argument("--n", type=int, default=10, help="Number of problems (0=all)")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    device = pick_device()
    model, model_dir = load_hf_model(args.model, device=device, local_dir=ROOT / "models")
    tok = Tokenizer(model_dir / "tokenizer.json")

    data = load_math500()
    if args.n > 0:
        data = data[:args.n]

    print(f"\nEvaluating {args.model} on {len(data)} MATH-500 problems\n")
    evaluate(model, tok, device, data, max_tokens=args.max_tokens, verbose=args.verbose)


if __name__ == "__main__":
    main()
