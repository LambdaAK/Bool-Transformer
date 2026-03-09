"""
GPT-style decoder for generating boolean expressions.
Uses TransformerEncoder with causal masking (decoder-only).
"""
import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ExpressionGPT(nn.Module):
    """
    Decoder-only transformer for autoregressive expression generation.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        max_length: int = 64,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_length, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask: lower triangle = 0, upper = -inf."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask.masked_fill(mask, float("-inf"))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) 1 = valid, 0 = pad

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).float()

        x = self.embedding(input_ids)
        x = self.pos_encoding(x)

        seq_len = x.size(1)
        causal_mask = self._causal_mask(seq_len, x.device)
        key_padding_mask = (attention_mask == 0)  # True = mask out

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        logits = self.lm_head(x)
        return logits
