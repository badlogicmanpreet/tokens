"""Text generation strategies for autoregressive LLMs.

Provides greedy decoding in two flavours:
  - `greedy`       — simple loop, reprocesses the full sequence each step
  - `greedy_cached` — uses a KV cache so each step only processes one token

Both are thin wrappers that keep generation logic separate from the model.
"""

import time

import torch

from .cache import KVCache
from .model import TransformerLM


# -----------------------------------------------------------------------
# Greedy decoding (no cache)
# -----------------------------------------------------------------------

@torch.inference_mode()
def greedy(
    model: TransformerLM,
    token_ids: torch.Tensor,
    max_tokens: int,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Generate tokens one-by-one using argmax selection.

    Args:
        model: the language model
        token_ids: prompt tensor of shape (1, prompt_len)
        max_tokens: upper bound on new tokens to generate
        eos_id: stop early when this token is produced

    Returns:
        Tensor containing only the newly generated token IDs.
    """
    prompt_len = token_ids.shape[1]
    model.eval()

    for _ in range(max_tokens):
        logits = model(token_ids)[:, -1]
        next_tok = logits.argmax(dim=-1, keepdim=True)

        if eos_id is not None and next_tok.item() == eos_id:
            break
        token_ids = torch.cat([token_ids, next_tok], dim=1)

    return token_ids[:, prompt_len:]


# -----------------------------------------------------------------------
# Greedy decoding with KV cache
# -----------------------------------------------------------------------

@torch.inference_mode()
def greedy_cached(
    model: TransformerLM,
    token_ids: torch.Tensor,
    max_tokens: int,
    eos_id: int | None = None,
) -> torch.Tensor:
    """Generate tokens with KV caching for faster inference.

    The first forward pass processes the full prompt and populates the
    cache.  Subsequent steps feed only the latest token, reusing
    cached keys and values from all previous positions.
    """
    prompt_len = token_ids.shape[1]
    model.eval()

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_cache_state()

    # Prefill: encode the entire prompt
    logits = model(token_ids, cache=cache)[:, -1]

    for _ in range(max_tokens):
        next_tok = logits.argmax(dim=-1, keepdim=True)

        if eos_id is not None and next_tok.item() == eos_id:
            break
        token_ids = torch.cat([token_ids, next_tok], dim=1)

        # Decode: process only the new token
        logits = model(next_tok, cache=cache)[:, -1]

    return token_ids[:, prompt_len:]


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def pick_device() -> torch.device:
    """Select the best available compute device."""
    if torch.cuda.is_available():
        print("Device: CUDA GPU")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Device: Apple Silicon (MPS)")
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print("Device: Intel XPU")
        return torch.device("xpu")
    print("Device: CPU")
    return torch.device("cpu")


def generate(model, tokenizer, prompt: str, device, max_tokens: int = 2048,
             verbose: bool = False) -> str:
    """High-level helper: encode prompt → generate → decode to string.

    Args:
        model: the language model
        tokenizer: Tokenizer instance
        prompt: raw text prompt
        device: torch device
        max_tokens: max new tokens to generate
        verbose: print tokens as they are generated
    """
    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device
    ).unsqueeze(0)

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_cache_state()
    model.eval()

    logits = model(input_ids, cache=cache)[:, -1]
    generated = []

    with torch.inference_mode():
        for _ in range(max_tokens):
            next_tok = logits.argmax(dim=-1, keepdim=True)
            tok_id = next_tok.item()

            if tok_id == tokenizer.eos_id:
                break

            generated.append(tok_id)
            if verbose:
                print(tokenizer.decode([tok_id]), end="", flush=True)

            logits = model(next_tok, cache=cache)[:, -1]

    return tokenizer.decode(generated)


def benchmark(fn, *args, warmup: int = 0, **kwargs) -> tuple[torch.Tensor, float]:
    """Run *fn* and return (result, elapsed_seconds).

    Args:
        fn: callable (e.g. greedy or greedy_cached)
        warmup: number of untimed warmup calls (useful with torch.compile)
    """
    for _ in range(warmup):
        fn(*args, **kwargs)

    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed
