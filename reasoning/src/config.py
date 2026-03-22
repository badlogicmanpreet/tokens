"""Model architecture configurations.

Each config is a plain dict describing a transformer LLM variant.
Keeping configs separate from model code makes it easy to swap
architectures or experiment with smaller/larger models.
"""

import torch

# ---- Default model for all chapters ----

QWEN3_17B = {
    "name": "qwen3-1.7b",
    "repo_id": "Qwen/Qwen3-1.7B",
    "vocab_size": 151_936,
    "max_seq_len": 40_960,
    "dim": 2048,
    "n_heads": 16,
    "n_kv_heads": 8,
    "n_layers": 28,
    "ffn_dim": 6144,
    "head_dim": 128,
    "rope_theta": 1_000_000.0,
    "norm_eps": 1e-6,
    "qk_norm": True,
    "dtype": torch.bfloat16,
}

DEFAULT = QWEN3_17B

# ---- Other models (kept for reference) ----

QWEN3_06B = {
    "name": "qwen3-0.6b",
    "repo_id": "rasbt/qwen3-from-scratch",
    "vocab_size": 151_936,
    "max_seq_len": 40_960,
    "dim": 1024,
    "n_heads": 16,
    "n_kv_heads": 8,
    "n_layers": 28,
    "ffn_dim": 3072,
    "head_dim": 128,
    "rope_theta": 1_000_000.0,
    "norm_eps": 1e-6,
    "qk_norm": True,
    "dtype": torch.bfloat16,
}

QWEN25_15B = {
    "name": "qwen2.5-1.5b",
    "repo_id": "Qwen/Qwen2.5-1.5B",
    "vocab_size": 151_936,
    "max_seq_len": 131_072,
    "dim": 1536,
    "n_heads": 12,
    "n_kv_heads": 2,
    "n_layers": 28,
    "ffn_dim": 8960,
    "head_dim": 128,
    "rope_theta": 1_000_000.0,
    "norm_eps": 1e-6,
    "qk_norm": False,
    "attn_bias": True,
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
    "head_dim": 64,
    "rope_theta": 500_000.0,
    "norm_eps": 1e-5,
    "qk_norm": False,
    "dtype": torch.bfloat16,
}

# Registry for easy lookup by name
MODELS = {
    "qwen3-1.7b": QWEN3_17B,
    "qwen3-0.6b": QWEN3_06B,
    "qwen2.5-1.5b": QWEN25_15B,
    "llama-3.2-1b": LLAMA_32_1B,
}
