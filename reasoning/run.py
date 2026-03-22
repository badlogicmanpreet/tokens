import sys
import torch
from pathlib import Path
from src.tokenizer import Tokenizer
from src.generate import greedy_cached, pick_device
from src.weights import load_hf_model
from src.config import MODELS

ROOT = Path(__file__).resolve().parent

# --- Default: Qwen3-1.7B. Override with: python run.py qwen3-0.6b ---
model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3-1.7b"
if model_name not in MODELS:
    sys.exit(f"Unknown model: {model_name}. Choose from: {', '.join(MODELS)}")

device = pick_device()
model, model_dir = load_hf_model(model_name, device=device, local_dir=ROOT / "models")
tok = Tokenizer(model_dir / "tokenizer.json")

print(f"\nReady ({model_name}). Type a prompt and press Enter. Type 'quit' to exit.\n")

while True:
    try:
        prompt = input("> ")
    except (EOFError, KeyboardInterrupt):
        break
    if prompt.strip().lower() in ("quit", "exit", "q"):
        break
    if not prompt.strip():
        continue

    ids = torch.tensor(tok.encode(prompt), device=device).unsqueeze(0)
    out = greedy_cached(model, ids, max_tokens=200, eos_id=tok.eos_id)
    print(tok.decode(out.squeeze(0).tolist()))
