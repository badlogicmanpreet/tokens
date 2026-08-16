# Token by Token — companion code

Code for the book *Token by Token: Building modern AI from scratch* by
Manpreet Singh. Each directory is a self-contained project matching a chapter:

| Directory | Chapter | What you build |
|---|---|---|
| `tiny-autograd/` | 3 | A ~300-line autograd engine: Value, backward(), Neuron → Layer → MLP |
| `tiny-models/` | 3 | Bigram + neural language models in PyTorch (names dataset) |
| `gpt/` | 4 | GPT-2 (124M) from scratch, trained on FineWeb-Edu, HellaSwag eval |
| `reasoning/` | 5 | Qwen3-1.7B loaded from scratch: CoT, self-consistency, self-refinement, GRPO RL, MATH-500 |
| `tinyclaw/` | 6 | A two-agent debate arena on an OpenClaw gateway |

Each project has its own README with install and run instructions.
Start with the chapter you're reading; the projects are independent.

License: MIT (see LICENSE; `tiny-models/` carries Apache-2.0).
