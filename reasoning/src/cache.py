"""Key-Value cache for efficient autoregressive generation.

During text generation, each new token only needs to attend to all
previous tokens. Without caching, we'd redundantly recompute the
key and value projections for the entire history at every step.

The KV cache stores these projections layer-by-layer so that each
generation step only computes K/V for the single new token and
concatenates it with the cached history.
"""

from __future__ import annotations


class KVCache:
    """Per-layer key-value cache backed by a simple list of tensor pairs."""

    __slots__ = ("_layers",)

    def __init__(self, n_layers: int):
        self._layers: list[tuple | None] = [None] * n_layers

    # -- access -----------------------------------------------------------

    def __getitem__(self, layer: int):
        """Return cached (keys, values) for *layer*, or None."""
        return self._layers[layer]

    def __setitem__(self, layer: int, kv_pair):
        """Store (keys, values) for *layer*."""
        self._layers[layer] = kv_pair

    # -- lifecycle --------------------------------------------------------

    def clear(self):
        for i in range(len(self._layers)):
            self._layers[i] = None
