"""Transformer language model (Qwen3-compatible architecture).

Implements a decoder-only transformer with:
  - RMSNorm (pre-norm architecture)
  - Grouped-Query Attention (GQA) with optional QK-norm
  - SwiGLU feed-forward blocks
  - Rotary Position Embeddings (RoPE)

The model is fully compatible with Qwen3 pre-trained weights
while being written independently for clarity and hackability.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import build_rope_table, apply_rope
from .cache import KVCache


# -----------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-Mean-Square normalization (no mean centering, no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.scale).to(orig)


# -----------------------------------------------------------------------
# Attention
# -----------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head attention with grouped key-value heads (GQA).

    GQA reduces memory by sharing each KV head across multiple query
    heads, controlled by the ratio n_heads / n_kv_heads.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                 head_dim: int, qk_norm: bool, eps: float, dtype):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.repeats = n_heads // n_kv_heads   # how many Q heads share one KV head
        self.out_dim = n_heads * head_dim

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False, dtype=dtype)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False, dtype=dtype)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False, dtype=dtype)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False, dtype=dtype)

        self.q_norm = RMSNorm(head_dim, eps) if qk_norm else None
        self.k_norm = RMSNorm(head_dim, eps) if qk_norm else None

    def forward(self, x, cos, sin, mask, kv_cache=None, pos_offset=0, store_cache=False):
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Per-head normalization before rotary encoding
        if self.q_norm is not None:
            q, k = self.q_norm(q), self.k_norm(k)

        # Inject position information
        q = apply_rope(q, cos, sin, offset=pos_offset)
        k = apply_rope(k, cos, sin, offset=pos_offset)

        # Concatenate with cached KV from previous steps
        new_cache = None
        if store_cache:
            if kv_cache is not None:
                k = torch.cat([kv_cache[0], k], dim=2)
                v = torch.cat([kv_cache[1], v], dim=2)
            new_cache = (k, v)

        # Expand KV heads to match query head count
        k = k.repeat_interleave(self.repeats, dim=1)
        v = v.repeat_interleave(self.repeats, dim=1)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, self.out_dim)

        return self.wo(out), new_cache


# -----------------------------------------------------------------------
# Feed-forward
# -----------------------------------------------------------------------

class SwiGLU(nn.Module):
    """Gated feed-forward with SiLU activation (SwiGLU variant).

    Uses three projections: gate, up, and down — where the gate
    output is activated with SiLU then element-wise multiplied
    with the up projection before the final down projection.
    """

    def __init__(self, dim: int, ffn_dim: int, dtype):
        super().__init__()
        self.gate = nn.Linear(dim, ffn_dim, bias=False, dtype=dtype)
        self.up   = nn.Linear(dim, ffn_dim, bias=False, dtype=dtype)
        self.down = nn.Linear(ffn_dim, dim, bias=False, dtype=dtype)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


# -----------------------------------------------------------------------
# Transformer block
# -----------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer layer: norm -> attention -> residual -> norm -> ffn -> residual."""

    def __init__(self, cfg: dict):
        super().__init__()
        d, eps, dt = cfg["dim"], cfg["norm_eps"], cfg["dtype"]

        self.attn_norm = RMSNorm(d, eps)
        self.attn = Attention(
            dim=d, n_heads=cfg["n_heads"], n_kv_heads=cfg["n_kv_heads"],
            head_dim=cfg["head_dim"], qk_norm=cfg["qk_norm"], eps=eps, dtype=dt,
        )
        self.ffn_norm = RMSNorm(d, eps)
        self.ffn = SwiGLU(d, cfg["ffn_dim"], dt)

    def forward(self, x, cos, sin, mask, kv_cache=None, pos_offset=0, store_cache=False):
        h, new_kv = self.attn(
            self.attn_norm(x), cos, sin, mask,
            kv_cache=kv_cache, pos_offset=pos_offset, store_cache=store_cache,
        )
        x = x + h
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_kv


# -----------------------------------------------------------------------
# Full model
# -----------------------------------------------------------------------

class TransformerLM(nn.Module):
    """Decoder-only transformer language model.

    Converts token IDs -> embeddings -> N transformer blocks -> logits.
    Supports optional KV caching for efficient autoregressive decoding.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        dt = cfg["dtype"]

        self.embedding = nn.Embedding(cfg["vocab_size"], cfg["dim"], dtype=dt)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.norm = RMSNorm(cfg["dim"], cfg["norm_eps"])
        self.head = nn.Linear(cfg["dim"], cfg["vocab_size"], bias=False, dtype=dt)

        # Precompute rotary tables once
        cos, sin = build_rope_table(cfg["head_dim"], cfg["max_seq_len"], cfg["rope_theta"])
        self.register_buffer("_rope_cos", cos, persistent=False)
        self.register_buffer("_rope_sin", sin, persistent=False)

        # Tracks current position for cached generation
        self._pos = 0

    # ----- public --------------------------------------------------------

    def forward(self, token_ids: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """Run a forward pass and return logits.

        Args:
            token_ids: (batch, seq_len) integer tensor
            cache: optional KVCache; when provided, enables incremental decoding

        Returns:
            logits of shape (batch, seq_len, vocab_size)
        """
        B, T = token_ids.shape
        x = self.embedding(token_ids)
        use_cache = cache is not None

        # Build causal attention mask
        if not use_cache:
            mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
            pos_offset = 0
        else:
            pos_offset = self._pos
            total = pos_offset + T
            full = torch.triu(torch.ones(total, total, dtype=torch.bool, device=x.device), diagonal=1)
            mask = full[pos_offset:total, :total]
            self._pos = total

        mask = mask.unsqueeze(0).unsqueeze(0)  # broadcast over batch & heads

        cos = self._rope_cos.to(x.device)
        sin = self._rope_sin.to(x.device)

        for i, layer in enumerate(self.layers):
            layer_kv = cache[i] if use_cache else None
            x, new_kv = layer(x, cos, sin, mask, kv_cache=layer_kv,
                              pos_offset=pos_offset, store_cache=use_cache)
            if use_cache:
                cache[i] = new_kv

        x = self.norm(x)
        return self.head(x.to(self.cfg["dtype"]))

    def reset_cache_state(self):
        """Reset internal position counter (call before each new generation)."""
        self._pos = 0
