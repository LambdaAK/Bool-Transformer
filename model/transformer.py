"""
Small encoder-only transformer for boolean expression evaluation.
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
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class BooleanTransformer(nn.Module):
    """
    Encoder-only transformer for binary classification of boolean expressions.
    """

    def __init__(
        self,
        vocab_size: int = 10,
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

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_length, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) token IDs
            attention_mask: (batch, seq_len) 1 for valid, 0 for pad. If None, inferred from pad_id.

        Returns:
            logits: (batch, 2)
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_id).float()

        # Create causal-style mask for transformer: 0 = attend, -inf = mask
        # TransformerEncoder expects (seq_len, seq_len) or (batch, nhead, seq_len, seq_len)
        # For encoder we use key_padding_mask: (batch, seq_len) where True = mask out
        key_padding_mask = attention_mask == 0  # True where we should mask

        x = self.embedding(input_ids)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)

        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (batch, seq_len, d_model)

        # Mean pool over non-padding positions
        mask_expanded = attention_mask.unsqueeze(-1).expand(x.size())
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

        logits = self.classifier(x)  # (batch, 2)
        return logits
