"""Model architecture configurations.

Each config is a plain dict describing a transformer LLM variant.
Keeping configs separate from model code makes it easy to swap
architectures or experiment with smaller/larger models.
"""

import torch

QWEN3_06B = {
    "name": "qwen3-0.6b",
    "repo_id": "rasbt/qwen3-from-scratch",
    "vocab_size": 151_936,
    "max_seq_len": 40_960,
    "dim": 1024,               # model / embedding dimension
    "n_heads": 16,             # query heads
    "n_kv_heads": 8,           # key-value heads (grouped-query attention)
    "n_layers": 28,
    "ffn_dim": 3072,           # SwiGLU intermediate size
    "head_dim": 128,           # per-head dimension (independent of dim)
    "rope_theta": 1_000_000.0,
    "norm_eps": 1e-6,
    "qk_norm": True,           # apply RMSNorm to Q and K projections
    "dtype": torch.bfloat16,
}

LLAMA_32_1B = {
    "name": "llama-3.2-1b",
    "repo_id": "meta-llama/Llama-3.2-1B",
    "vocab_size": 128_256,
    "max_seq_len": 131_072,
    "dim": 2048,
    "n_heads": 32,
    "n_kv_heads": 8,
    "n_layers": 16,
    "ffn_dim": 8192,
    "head_dim": 64,            # dim // n_heads
    "rope_theta": 500_000.0,
    "norm_eps": 1e-5,
    "qk_norm": False,
    "dtype": torch.bfloat16,
}

# Registry for easy lookup by name
MODELS = {
    "qwen3-0.6b": QWEN3_06B,
    "llama-3.2-1b": LLAMA_32_1B,
}
