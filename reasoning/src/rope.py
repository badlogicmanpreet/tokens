"""Rotary Position Embeddings (RoPE).

RoPE encodes position information directly into the query and key
vectors of attention, enabling the model to understand token ordering
without explicit position embeddings added to the input.

The implementation uses the "split-halves" layout: given a head
vector [x0, x1, ..., x_{d-1}], we treat the first half and second
half as pairs and rotate them using sinusoidal frequencies derived
from the position index.
"""

import torch


def build_rope_table(head_dim: int, max_seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine tables for rotary embeddings.

    Returns:
        (cos_table, sin_table) each of shape (max_seq_len, head_dim)
    """
    half = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(positions, freqs)            # (seq, half)
    angles = torch.cat([angles, angles], dim=-1)      # (seq, head_dim)
    return angles.cos(), angles.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    offset: int = 0,
) -> torch.Tensor:
    """Rotate query or key tensor using precomputed cos/sin tables.

    Args:
        x: (batch, heads, seq_len, head_dim)
        cos, sin: precomputed tables from build_rope_table
        offset: starting position (used with KV cache)
    """
    seq_len = x.shape[2]
    c = cos[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)
    s = sin[offset : offset + seq_len].unsqueeze(0).unsqueeze(0)

    half = x.shape[-1] // 2
    x_left, x_right = x[..., :half], x[..., half:]
    rotated = torch.cat([-x_right, x_left], dim=-1)
    return (x * c + rotated * s).to(x.dtype)
