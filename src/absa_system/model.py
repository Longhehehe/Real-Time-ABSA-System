"""Aspect-conditioned, evidence-aware multi-polarity encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math

import torch
import torch.nn as nn

from .schema import ASPECTS, POLARITIES


@dataclass
class ABSAOutput:
    mention_logits: torch.Tensor
    sentiment_logits: torch.Tensor
    mention_attention_logits: torch.Tensor
    polarity_attention_logits: torch.Tensor
    mention_attention: torch.Tensor
    polarity_attention: torch.Tensor


class AspectEvidenceModel(nn.Module):
    """Pretrained encoder with aspect/polarity-specific token retrieval.

    Unlike the legacy CLS-only classifier, every aspect receives its own query.
    Positive, negative and neutral use separate evidence queries, allowing the
    model to preserve positive+negative evidence for the same aspect.
    """

    def __init__(
        self,
        *,
        backbone_name: str = "vinai/phobert-base",
        dropout: float = 0.2,
        encoder: nn.Module | None = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if encoder is None:
            from transformers import AutoModel

            encoder = AutoModel.from_pretrained(
                backbone_name,
                local_files_only=local_files_only,
            )
        self.encoder = encoder
        self.backbone_name = backbone_name
        hidden_size = int(self.encoder.config.hidden_size)
        self.hidden_size = hidden_size
        self.num_aspects = len(ASPECTS)
        self.num_polarities = len(POLARITIES)

        self.aspect_queries = nn.Parameter(
            torch.empty(self.num_aspects, hidden_size)
        )
        self.polarity_queries = nn.Parameter(
            torch.empty(self.num_polarities, hidden_size)
        )
        self.token_key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.token_value = nn.Linear(hidden_size, hidden_size, bias=False)
        self.mention_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.polarity_query = nn.Linear(hidden_size, hidden_size, bias=False)

        self.mention_fusion = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )
        self.polarity_fusion = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
        )
        self.mention_classifier = nn.Linear(hidden_size, 1)
        self.sentiment_classifier = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.aspect_queries, mean=0.0, std=0.02)
        nn.init.normal_(self.polarity_queries, mean=0.0, std=0.02)

    def enable_gradient_checkpointing(self) -> None:
        method = getattr(self.encoder, "gradient_checkpointing_enable", None)
        if callable(method):
            method()

    @staticmethod
    def _masked_attention(
        logits: torch.Tensor,
        content_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = content_mask.to(dtype=torch.bool)
        while mask.ndim < logits.ndim:
            mask = mask.unsqueeze(1)
        masked_logits = logits.masked_fill(~mask, -1e4)
        attention = torch.softmax(masked_logits, dim=-1)
        attention = attention * mask.to(attention.dtype)
        normalizer = attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attention = attention / normalizer
        return masked_logits, attention

    @staticmethod
    def _fuse(global_state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        expanded = global_state
        while expanded.ndim < context.ndim:
            expanded = expanded.unsqueeze(1)
        expanded = expanded.expand_as(context)
        return torch.cat(
            [
                context,
                expanded,
                context * expanded,
                torch.abs(context - expanded),
            ],
            dim=-1,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        content_mask: torch.Tensor | None = None,
    ) -> ABSAOutput:
        encoded = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        token_states = self.dropout(encoded.last_hidden_state)
        global_state = token_states[:, 0, :]
        if content_mask is None:
            content_mask = attention_mask.clone()
            content_mask[:, 0] = 0

        keys = self.token_key(token_states)
        values = self.token_value(token_states)
        mention_queries = self.mention_query(self.aspect_queries)
        mention_scores = torch.einsum(
            "ah,blh->bal",
            mention_queries,
            keys,
        ) / math.sqrt(self.hidden_size)
        mention_attention_logits, mention_attention = self._masked_attention(
            mention_scores, content_mask
        )
        mention_context = torch.einsum(
            "bal,blh->bah",
            mention_attention,
            values,
        )
        mention_hidden = self.mention_fusion(
            self._fuse(global_state, mention_context)
        )
        mention_logits = self.mention_classifier(mention_hidden).squeeze(-1)

        combined_queries = (
            self.aspect_queries[:, None, :]
            + self.polarity_queries[None, :, :]
        )
        combined_queries = self.polarity_query(combined_queries)
        polarity_scores = torch.einsum(
            "aph,blh->bapl",
            combined_queries,
            keys,
        ) / math.sqrt(self.hidden_size)
        polarity_attention_logits, polarity_attention = self._masked_attention(
            polarity_scores, content_mask
        )
        polarity_context = torch.einsum(
            "bapl,blh->baph",
            polarity_attention,
            values,
        )
        polarity_hidden = self.polarity_fusion(
            self._fuse(global_state, polarity_context)
        )
        sentiment_logits = self.sentiment_classifier(
            polarity_hidden
        ).squeeze(-1)
        return ABSAOutput(
            mention_logits=mention_logits,
            sentiment_logits=sentiment_logits,
            mention_attention_logits=mention_attention_logits,
            polarity_attention_logits=polarity_attention_logits,
            mention_attention=mention_attention,
            polarity_attention=polarity_attention,
        )

    def export_config(self) -> dict[str, Any]:
        return {
            "architecture": self.__class__.__name__,
            "backbone_name": self.backbone_name,
            "hidden_size": self.hidden_size,
            "aspects": list(ASPECTS),
            "polarities": list(POLARITIES),
        }
