import sys
import torch
from pathlib import Path
from src.tokenizer import Tokenizer
from src.generate import greedy_cached, pick_device
from src.weights import load_model, load_hf_model
from src.config import QWEN3_06B

ROOT = Path(__file__).resolve().parent

# --- Pick model from CLI arg: python run.py [qwen|llama] ---
choice = sys.argv[1] if len(sys.argv) > 1 else "qwen"
device = pick_device()

if choice == "llama":
    model, model_dir = load_hf_model("llama-3.2-1b", device=device, local_dir=ROOT / "models")
    tok = Tokenizer(model_dir / "tokenizer.json")
else:
    tok = Tokenizer(ROOT / "qwen3" / "tokenizer-base.json")
    model = load_model(ROOT / "qwen3" / "qwen3-0.6B-base.pth", cfg=QWEN3_06B, device=device)

print(f"\nReady ({choice}). Type a prompt and press Enter. Type 'quit' to exit.\n")

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
