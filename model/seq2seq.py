"""
Encoder-decoder transformer for boolean expression simplification (seq2seq).
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
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class SimplifierTransformer(nn.Module):
    """
    Encoder-decoder transformer for seq2seq boolean expression simplification.
    Encoder reads the complex expression, decoder generates the simplified one.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 256,
        max_length: int = 64,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.max_length = max_length

        self.src_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_length, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Causal mask for decoder: position i cannot attend to j > i."""
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        return mask

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            src_ids: (batch, src_len) - complex expression tokens
            tgt_ids: (batch, tgt_len) - decoder input (BOS + simple tokens, no EOS)
            src_key_padding_mask: (batch, src_len) True = mask out
            tgt_key_padding_mask: (batch, tgt_len) True = mask out

        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        if src_key_padding_mask is None:
            src_key_padding_mask = (src_ids == self.pad_id)
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = (tgt_ids == self.pad_id)

        src = self.src_embedding(src_ids)
        src = self.pos_encoding(src)

        tgt = self.tgt_embedding(tgt_ids)
        tgt = self.pos_encoding(tgt)

        tgt_len = tgt.size(1)
        tgt_mask = self._causal_mask(tgt_len, tgt.device)

        out = self.transformer(
            src=src,
            tgt=tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )
        logits = self.lm_head(out)
        return logits
