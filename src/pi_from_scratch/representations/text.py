import hashlib
import re

import torch


class HashTokenizer:
    """A deterministic teaching tokenizer; replace it with the VLM tokenizer in M2."""

    PAD = 0

    def __init__(self, vocab_size: int, max_tokens: int):
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        self.vocab_size = vocab_size
        self.max_tokens = max_tokens

    def _token_id(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return 1 + int.from_bytes(digest) % (self.vocab_size - 1)

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        words = re.findall(r"[\w']+|[^\w\s]", text.lower(), flags=re.UNICODE)
        ids = [self._token_id(word) for word in words[: self.max_tokens]]
        mask = [True] * len(ids)
        padding = self.max_tokens - len(ids)
        ids.extend([self.PAD] * padding)
        mask.extend([False] * padding)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)
