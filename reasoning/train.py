"""Train Qwen3-1.7B with GRPO (reinforcement learning with verifiable rewards).

Usage:
    uv run python train.py                          # 50 steps, defaults
    uv run python train.py --steps 100              # more training
    uv run python train.py --rollouts 2 --max-tokens 256  # lighter run
"""

import argparse
import json
from pathlib import Path

import torch

from src.config import MODELS
from src.generate import pick_device
from src.grpo import train_grpo
from src.tokenizer import Tokenizer
from src.weights import load_hf_model

ROOT = Path(__file__).resolve().parent


def load_math_train(path: Path | None = None) -> list[dict]:
    """Load MATH training data (12K problems)."""
    if path is None:
        path = ROOT / "math_train.json"

    if path.exists():
        with path.open("r") as f:
            return json.load(f)

    import urllib.request
    url = (
        "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "main/ch06/01_main-chapter-code/math_train.json"
    )
    print("Downloading MATH training dataset...")
    with urllib.request.urlopen(url) as resp:
        data = json.loads(resp.read().decode())
    with path.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {path} ({len(data)} problems)")
    return data


def main():
    parser = argparse.ArgumentParser(description="Train with GRPO")
    parser.add_argument("--model", default="qwen3-1.7b", choices=list(MODELS))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    args = parser.parse_args()

    device = pick_device()
    model, model_dir = load_hf_model(args.model, device=device, local_dir=ROOT / "models")
    tok = Tokenizer(model_dir / "tokenizer.json")

    train_data = load_math_train()

    print(f"\nTraining {args.model} with GRPO on {len(train_data)} math problems\n")

    train_grpo(
        model=model,
        tokenizer=tok,
        train_data=train_data,
        device=device,
        steps=args.steps,
        num_rollouts=args.rollouts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        lr=args.lr,
        checkpoint_every=args.checkpoint_every,
        checkpoint_dir=str(ROOT / "checkpoints"),
    )


if __name__ == "__main__":
    main()
