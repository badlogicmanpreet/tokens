# reasoning — build a reasoning model from a base LLM

Companion code for Token by Token, Chapter 5. Loads Qwen3-1.7B from scratch
(no transformers library), evaluates it on MATH-500, and improves it with
chain-of-thought, self-consistency, self-refinement, and GRPO-style RL.

## Install
Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/):
```bash
uv sync          # creates .venv from pyproject.toml
```
Models download from Hugging Face on first run (~3.5 GB for Qwen3-1.7B).

## Run
```bash
uv run python run.py                       # interactive generation
uv run python evaluate.py --mode greedy    # MATH-500 baseline
uv run python evaluate.py --mode cot       # chain-of-thought
uv run python evaluate.py --mode vote      # self-consistency
uv run python evaluate.py --mode refine    # self-refinement
uv run python train.py                     # GRPO RL (50 steps by default)
```

Start with TUTORIAL_1.md and work through TUTORIAL_5.md — they mirror the
chapter section by section.

Notes:
- Training casts the model to fp32 (bf16 master weights would swallow
  lr=1e-5 updates) — budget ~20 GB GPU memory for the 1.7B model.
- Checkpoints (~7 GB each in fp32) land in `checkpoints/` (gitignored).
- The verifier (src/verify.py) has a 5s per-answer timeout; see its tests
  in tests/test_verify.py.
