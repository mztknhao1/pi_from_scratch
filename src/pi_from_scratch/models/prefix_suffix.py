"""Prefix/suffix attention and a small two-expert Transformer for teaching π₀."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class Pi0AttentionLayout:
    """Masks for image/language prefix, state token, and action-token block."""

    prefix_mask: Tensor
    suffix_mask: Tensor
    input_mask: Tensor
    block_starts: Tensor
    attention_mask: Tensor


def make_blockwise_attention_mask(input_mask: Tensor, block_starts: Tensor) -> Tensor:
    """Return ``[B, N, N]`` where true means query may attend to key.

    ``block_starts`` follows openpi's ``mask_ar`` convention: true starts a new
    causal block; false keeps a token in the previous token's block. Tokens may
    attend to all valid tokens in their own block and in earlier blocks.
    """
    if input_mask.ndim != 2 or input_mask.dtype != torch.bool:
        raise ValueError("input_mask must be bool with shape [batch, tokens]")
    if block_starts.ndim == 1:
        if block_starts.shape[0] != input_mask.shape[1]:
            raise ValueError("one-dimensional block_starts must have shape [tokens]")
        block_starts = block_starts[None].expand_as(input_mask)
    elif block_starts.shape != input_mask.shape:
        raise ValueError("block_starts must have shape [tokens] or [batch, tokens]")
    if block_starts.dtype != torch.bool or block_starts.device != input_mask.device:
        raise ValueError("block_starts must be bool on the same device as input_mask")

    block_ids = torch.cumsum(block_starts.to(torch.int64), dim=1)
    block_visible = block_ids[:, None, :] <= block_ids[:, :, None]
    valid_pairs = input_mask[:, None, :] & input_mask[:, :, None]
    return block_visible & valid_pairs


def make_pi0_attention_layout(prefix_mask: Tensor, action_mask: Tensor) -> Pi0AttentionLayout:
    """Build the π₀ mask: bidirectional prefix, state block, action block."""
    if prefix_mask.ndim != 2 or prefix_mask.dtype != torch.bool:
        raise ValueError("prefix_mask must be bool with shape [batch, prefix_tokens]")
    if action_mask.ndim != 2 or action_mask.dtype != torch.bool:
        raise ValueError("action_mask must be bool with shape [batch, action_horizon]")
    if prefix_mask.shape[0] != action_mask.shape[0]:
        raise ValueError("prefix_mask and action_mask must have the same batch size")
    if prefix_mask.device != action_mask.device:
        raise ValueError("prefix_mask and action_mask must be on the same device")
    if prefix_mask.shape[1] < 1 or action_mask.shape[1] < 1:
        raise ValueError("prefix and action sequences must be non-empty")

    state_mask = torch.ones(
        (prefix_mask.shape[0], 1), dtype=torch.bool, device=prefix_mask.device
    )
    suffix_mask = torch.cat((state_mask, action_mask), dim=1)
    input_mask = torch.cat((prefix_mask, suffix_mask), dim=1)

    prefix_starts = torch.zeros(prefix_mask.shape[1], dtype=torch.bool, device=prefix_mask.device)
    state_start = torch.ones(1, dtype=torch.bool, device=prefix_mask.device)
    action_starts = torch.zeros(action_mask.shape[1], dtype=torch.bool, device=prefix_mask.device)
    action_starts[0] = True
    block_starts = torch.cat((prefix_starts, state_start, action_starts), dim=0)
    attention_mask = make_blockwise_attention_mask(input_mask, block_starts)
    return Pi0AttentionLayout(
        prefix_mask=prefix_mask,
        suffix_mask=suffix_mask,
        input_mask=input_mask,
        block_starts=block_starts,
        attention_mask=attention_mask,
    )


class TwoExpertTransformerBlock(nn.Module):
    """One attention layer with separate parameters for prefix and suffix tokens."""

    def __init__(self, width: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads

        self.prefix_norm1 = nn.LayerNorm(width)
        self.suffix_norm1 = nn.LayerNorm(width)
        self.prefix_qkv = nn.Linear(width, 3 * width)
        self.suffix_qkv = nn.Linear(width, 3 * width)
        self.prefix_out = nn.Linear(width, width)
        self.suffix_out = nn.Linear(width, width)

        self.prefix_norm2 = nn.LayerNorm(width)
        self.suffix_norm2 = nn.LayerNorm(width)
        self.prefix_mlp = self._make_mlp(width, dropout)
        self.suffix_mlp = self._make_mlp(width, dropout)
        self.attention_dropout = nn.Dropout(dropout)

    @staticmethod
    def _make_mlp(width: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * width, width),
            nn.Dropout(dropout),
        )

    def _split_heads(self, value: Tensor) -> Tensor:
        batch, tokens, _ = value.shape
        return value.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, value: Tensor) -> Tensor:
        batch, _, tokens, _ = value.shape
        return value.transpose(1, 2).reshape(batch, tokens, self.width)

    def forward(
        self,
        prefix: Tensor,
        suffix: Tensor,
        layout: Pi0AttentionLayout,
    ) -> tuple[Tensor, Tensor]:
        if prefix.ndim != 3 or suffix.ndim != 3:
            raise ValueError("prefix and suffix must have shape [batch, tokens, width]")
        if prefix.shape[0] != suffix.shape[0] or prefix.shape[2] != self.width:
            raise ValueError("prefix and suffix must share batch size and configured width")
        if suffix.shape[2] != self.width:
            raise ValueError("suffix width does not match the configured width")
        if layout.prefix_mask.shape != prefix.shape[:2]:
            raise ValueError("layout prefix mask does not match prefix tokens")
        if layout.suffix_mask.shape != suffix.shape[:2]:
            raise ValueError("layout suffix mask does not match suffix tokens")

        prefix_q, prefix_k, prefix_v = self.prefix_qkv(self.prefix_norm1(prefix)).chunk(3, -1)
        suffix_q, suffix_k, suffix_v = self.suffix_qkv(self.suffix_norm1(suffix)).chunk(3, -1)
        query = self._split_heads(torch.cat((prefix_q, suffix_q), dim=1))
        key = self._split_heads(torch.cat((prefix_k, suffix_k), dim=1))
        value = self._split_heads(torch.cat((prefix_v, suffix_v), dim=1))

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        allowed = layout.attention_mask[:, None]
        has_visible_key = allowed.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        scores = torch.where(has_visible_key, scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1)
        weights = self.attention_dropout(weights) * has_visible_key.to(weights.dtype)
        context = self._merge_heads(torch.matmul(weights, value))

        prefix_len = prefix.shape[1]
        prefix_context, suffix_context = context[:, :prefix_len], context[:, prefix_len:]
        prefix = prefix + self.prefix_out(prefix_context)
        suffix = suffix + self.suffix_out(suffix_context)
        prefix = prefix + self.prefix_mlp(self.prefix_norm2(prefix))
        suffix = suffix + self.suffix_mlp(self.suffix_norm2(suffix))

        prefix = prefix * layout.prefix_mask.unsqueeze(-1).to(prefix.dtype)
        suffix = suffix * layout.suffix_mask.unsqueeze(-1).to(suffix.dtype)
        return prefix, suffix


class TwoExpertTransformer(nn.Module):
    """A stack of two-expert blocks with shared attention connectivity."""

    def __init__(self, width: int, num_heads: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            TwoExpertTransformerBlock(width, num_heads, dropout) for _ in range(num_layers)
        )
        self.prefix_norm = nn.LayerNorm(width)
        self.suffix_norm = nn.LayerNorm(width)

    def forward(
        self,
        prefix: Tensor,
        suffix: Tensor,
        layout: Pi0AttentionLayout,
    ) -> tuple[Tensor, Tensor]:
        for layer in self.layers:
            prefix, suffix = layer(prefix, suffix, layout)
        return self.prefix_norm(prefix), self.suffix_norm(suffix)
