"""Training objectives for structured multi-polarity ABSA."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import ABSAOutput


@dataclass
class LossBreakdown:
    total: torch.Tensor
    mention: torch.Tensor
    sentiment: torch.Tensor
    evidence: torch.Tensor
    consistency: torch.Tensor
    neutral_exclusivity: torch.Tensor

    def detached(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach().cpu()),
            "mention": float(self.mention.detach().cpu()),
            "sentiment": float(self.sentiment.detach().cpu()),
            "evidence": float(self.evidence.detach().cpu()),
            "consistency": float(self.consistency.detach().cpu()),
            "neutral_exclusivity": float(
                self.neutral_exclusivity.detach().cpu()
            ),
        }


def focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float,
    mask: torch.Tensor | None = None,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=pos_weight,
    )
    probability = torch.sigmoid(logits)
    pt = probability * targets + (1.0 - probability) * (1.0 - targets)
    loss = ((1.0 - pt) ** gamma) * bce
    if mask is None:
        return loss.mean()
    expanded_mask = mask
    while expanded_mask.ndim < loss.ndim:
        expanded_mask = expanded_mask.unsqueeze(-1)
    expanded_mask = expanded_mask.expand_as(loss).to(loss.dtype)
    return (loss * expanded_mask).sum() / expanded_mask.sum().clamp_min(1.0)


def attention_evidence_loss(
    attention: torch.Tensor,
    evidence_mask: torch.Tensor,
    available: torch.Tensor,
) -> torch.Tensor:
    target = evidence_mask / evidence_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    token_loss = -(target * torch.log(attention.clamp_min(1e-8))).sum(dim=-1)
    available = available.to(token_loss.dtype)
    return (token_loss * available).sum() / available.sum().clamp_min(1.0)


def compute_absa_loss(
    output: ABSAOutput,
    batch: dict[str, torch.Tensor],
    *,
    mention_weight: float = 2.0,
    sentiment_weight: float = 5.0,
    evidence_weight: float = 0.2,
    consistency_weight: float = 0.1,
    neutral_exclusivity_weight: float = 0.05,
    focal_gamma: float = 2.0,
    mention_pos_weight: torch.Tensor | None = None,
    sentiment_pos_weight: torch.Tensor | None = None,
) -> LossBreakdown:
    mention_targets = batch["mention_labels"].to(output.mention_logits.dtype)
    sentiment_targets = batch["sentiment_labels"].to(
        output.sentiment_logits.dtype
    )
    mention_loss = focal_bce_with_logits(
        output.mention_logits,
        mention_targets,
        gamma=focal_gamma,
        pos_weight=mention_pos_weight,
    )
    sentiment_loss = focal_bce_with_logits(
        output.sentiment_logits,
        sentiment_targets,
        gamma=focal_gamma,
        mask=mention_targets,
        pos_weight=sentiment_pos_weight,
    )

    evidence_loss = output.mention_logits.new_zeros(())
    if evidence_weight > 0 and "mention_evidence_mask" in batch:
        mention_evidence = attention_evidence_loss(
            output.mention_attention,
            batch["mention_evidence_mask"].to(output.mention_attention.dtype),
            batch["mention_evidence_available"].to(output.mention_attention.dtype),
        )
        polarity_evidence = attention_evidence_loss(
            output.polarity_attention,
            batch["polarity_evidence_mask"].to(
                output.polarity_attention.dtype
            ),
            batch["polarity_evidence_available"].to(
                output.polarity_attention.dtype
            ),
        )
        evidence_loss = 0.5 * (mention_evidence + polarity_evidence)

    mention_probability = torch.sigmoid(output.mention_logits).unsqueeze(-1)
    sentiment_probability = torch.sigmoid(output.sentiment_logits)
    consistency_loss = torch.relu(
        sentiment_probability.max(dim=-1, keepdim=True).values
        - mention_probability
    ).mean()

    neutral_probability = sentiment_probability[:, :, 2]
    polar_probability = torch.maximum(
        sentiment_probability[:, :, 0],
        sentiment_probability[:, :, 1],
    )
    neutral_exclusivity = (
        neutral_probability * polar_probability * mention_targets
    ).sum() / mention_targets.sum().clamp_min(1.0)

    total = (
        mention_weight * mention_loss
        + sentiment_weight * sentiment_loss
        + evidence_weight * evidence_loss
        + consistency_weight * consistency_loss
        + neutral_exclusivity_weight * neutral_exclusivity
    )
    return LossBreakdown(
        total=total,
        mention=mention_loss,
        sentiment=sentiment_loss,
        evidence=evidence_loss,
        consistency=consistency_loss,
        neutral_exclusivity=neutral_exclusivity,
    )
