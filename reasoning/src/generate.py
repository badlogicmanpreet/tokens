"""Text generation strategies for autoregressive LLMs.

Provides:
  - `greedy`         — deterministic argmax decoding (no cache)
  - `greedy_cached`  — deterministic argmax with KV cache
  - `generate`       — high-level: prompt in, text out (supports temperature + top-p)
  - `sample_cached`  — KV-cached generation with temperature scaling and top-p filtering
"""

import time

import torch
import torch.nn.functional as F

from .cache import KVCache
from .model import TransformerLM


# -----------------------------------------------------------------------
# Greedy decoding
# -----------------------------------------------------------------------

@torch.inference_mode()
def greedy(model, token_ids, max_tokens, eos_id=None):
    """Deterministic argmax decoding without cache."""
    prompt_len = token_ids.shape[1]
    model.eval()

    for _ in range(max_tokens):
        logits = model(token_ids)[:, -1]
        next_tok = logits.argmax(dim=-1, keepdim=True)
        if eos_id is not None and next_tok.item() == eos_id:
            break
        token_ids = torch.cat([token_ids, next_tok], dim=1)

    return token_ids[:, prompt_len:]


@torch.inference_mode()
def greedy_cached(model, token_ids, max_tokens, eos_id=None):
    """Deterministic argmax decoding with KV cache."""
    prompt_len = token_ids.shape[1]
    model.eval()

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_cache_state()

    logits = model(token_ids, cache=cache)[:, -1]

    for _ in range(max_tokens):
        next_tok = logits.argmax(dim=-1, keepdim=True)
        if eos_id is not None and next_tok.item() == eos_id:
            break
        token_ids = torch.cat([token_ids, next_tok], dim=1)
        logits = model(next_tok, cache=cache)[:, -1]

    return token_ids[:, prompt_len:]


# -----------------------------------------------------------------------
# Temperature scaling + top-p (nucleus) filtering
# -----------------------------------------------------------------------

def _top_p_filter(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose cumulative probability <= top_p.

    Zeroes out low-probability tokens and renormalizes so probabilities sum to 1.
    """
    if top_p is None or top_p >= 1.0:
        return probs

    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)

    # Keep tokens where cumulative prob hasn't exceeded top_p yet
    keep = cumulative <= top_p
    keep[..., 0] = True  # always keep the most probable token

    kept_sorted = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    filtered = torch.zeros_like(probs).scatter(-1, sorted_idx, kept_sorted)

    return filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.inference_mode()
def sample_cached(
    model,
    token_ids: torch.Tensor,
    max_tokens: int,
    eos_id: int | None = None,
    temperature: float = 0.0,
    top_p: float | None = None,
) -> torch.Tensor:
    """Generate tokens with KV cache, temperature scaling, and top-p filtering.

    When temperature=0 (default), falls back to deterministic argmax.
    When temperature>0, samples from the probability distribution
    filtered by top-p.
    """
    prompt_len = token_ids.shape[1]
    model.eval()

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_cache_state()

    logits = model(token_ids, cache=cache)[:, -1]

    for _ in range(max_tokens):
        if temperature is None or temperature == 0.0:
            next_tok = logits.argmax(dim=-1, keepdim=True)
        else:
            scaled = logits / temperature
            probs = F.softmax(scaled.float(), dim=-1)
            probs = _top_p_filter(probs, top_p)
            next_tok = torch.multinomial(probs.cpu(), num_samples=1)
            next_tok = next_tok.to(token_ids.device)

        if eos_id is not None and next_tok.item() == eos_id:
            break

        token_ids = torch.cat([token_ids, next_tok], dim=1)
        logits = model(next_tok, cache=cache)[:, -1]

    return token_ids[:, prompt_len:]


# -----------------------------------------------------------------------
# High-level generation: prompt in → text out
# -----------------------------------------------------------------------

def generate(
    model,
    tokenizer,
    prompt: str,
    device,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    top_p: float | None = None,
    verbose: bool = False,
) -> str:
    """Encode a prompt, generate tokens, decode to string.

    Args:
        temperature: 0 = greedy, >0 = sampling with this temperature
        top_p: nucleus sampling threshold (e.g. 0.9). Only used when temperature > 0.
        verbose: print tokens as they are generated
    """
    input_ids = torch.tensor(
        tokenizer.encode(prompt), device=device
    ).unsqueeze(0)

    cache = KVCache(n_layers=model.cfg["n_layers"])
    model.reset_cache_state()
    model.eval()

    logits = model(input_ids, cache=cache)[:, -1]
    generated: list[int] = []

    with torch.inference_mode():
        for _ in range(max_tokens):
            if temperature is None or temperature == 0.0:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = logits / temperature
                probs = F.softmax(scaled.float(), dim=-1)
                probs = _top_p_filter(probs, top_p)
                next_tok = torch.multinomial(probs.cpu(), num_samples=1)
                next_tok = next_tok.to(device)

            tok_id = next_tok.item()
            if tok_id == tokenizer.eos_id:
                break

            generated.append(tok_id)
            if verbose:
                print(tokenizer.decode([tok_id]), end="", flush=True)

            logits = model(next_tok, cache=cache)[:, -1]

    return tokenizer.decode(generated)


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


def benchmark(fn, *args, warmup: int = 0, **kwargs) -> tuple[torch.Tensor, float]:
    """Run *fn* and return (result, elapsed_seconds)."""
    for _ in range(warmup):
        fn(*args, **kwargs)

    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed
