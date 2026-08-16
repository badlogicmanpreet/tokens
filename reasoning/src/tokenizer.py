"""Tokenizer wrapper for BPE-based LLMs.

Wraps the HuggingFace `tokenizers` library to handle:
  - Encoding text to token IDs (including special tokens)
  - Decoding token IDs back to text
  - Auto-detection of EOS token for different model families
"""

import re
from pathlib import Path

from tokenizers import Tokenizer as HFTokenizer


class Tokenizer:
    """BPE tokenizer with special-token awareness."""

    def __init__(self, path: str | Path):
        """Load a tokenizer from a JSON file.

        Automatically detects special tokens from the vocabulary
        and picks the right EOS token for the model family.
        """
        self._tok = HFTokenizer.from_file(str(path))
        vocab = self._tok.get_vocab()

        # Collect all special tokens present in the vocab (angle-bracket style)
        self._special = {
            tok: tok_id for tok, tok_id in vocab.items()
            if tok.startswith("<|") and tok.endswith("|>")
            or tok.startswith("<") and tok.endswith(">") and len(tok) > 3
        }

        # Build regex for splitting text around special tokens
        if self._special:
            self._split_re = re.compile(
                "(" + "|".join(re.escape(t) for t in self._special) + ")"
            )
        else:
            self._split_re = None

        # Auto-detect EOS based on what tokens are available.
        #   Llama 3:  <|end_of_text|> (128001)
        #   Qwen3:    <|im_end|> (151645) is the actual stop token used at
        #             inference; <|endoftext|> (151643) is the padding/pretrain
        #             terminator. Treat BOTH as stop tokens (matches Qwen3's
        #             generation_config eos_token_id: [151645, 151643]).
        self.eos_ids = {
            tid for tid in (
                self._special.get("<|end_of_text|>"),   # Llama 3
                self._special.get("<|eot_id|>"),        # Llama 3 instruct
                self._special.get("<|im_end|>"),        # Qwen chat turns
                self._special.get("<|endoftext|>"),     # Qwen3 / GPT-style
            ) if tid is not None
        } or {self._tok.get_vocab_size() - 1}           # fallback: last token
        # Back-compat single id (used for padding); prefer eos_ids for stopping.
        self.eos_id = (
            self._special.get("<|end_of_text|>")
            or self._special.get("<|endoftext|>")
            or self._tok.get_vocab_size() - 1
        )

        self.bos_id = self._special.get("<|begin_of_text|>")  # Llama 3 (None for Qwen)

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        """Convert text to token IDs, correctly handling special tokens."""
        if text in self._special:
            return [self._special[text]]

        if self._split_re is None:
            return self._tok.encode(text).ids

        ids: list[int] = []
        for segment in self._split_re.split(text):
            if not segment:
                continue
            if segment in self._special:
                ids.append(self._special[segment])
            else:
                ids.extend(self._tok.encode(segment).ids)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back to text."""
        return self._tok.decode(ids, skip_special_tokens=False)
