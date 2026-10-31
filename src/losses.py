"""Training objectives for the controlled M1 versus M2 comparison."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_kl_divergence(first_logits: torch.Tensor, second_logits: torch.Tensor) -> torch.Tensor:
    """Return symmetric KL divergence between two categorical prediction distributions."""
    first_log_probabilities = F.log_softmax(first_logits.float(), dim=-1)
    second_log_probabilities = F.log_softmax(second_logits.float(), dim=-1)
    first_probabilities = first_log_probabilities.exp()
    second_probabilities = second_log_probabilities.exp()
    first_to_second = F.kl_div(
        first_log_probabilities, second_probabilities, reduction="batchmean"
    )
    second_to_first = F.kl_div(
        second_log_probabilities, first_probabilities, reduction="batchmean"
    )
    return 0.5 * (first_to_second + second_to_first)


def training_objective(
    first_logits: torch.Tensor,
    labels: torch.Tensor,
    second_logits: torch.Tensor | None = None,
    rdrop_alpha: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute CE for M1 or CE + alpha * symmetric KL for M2.

    Returns total loss, mean classification loss, and consistency loss. The
    consistency term is exactly zero for M1, making the controlled difference
    explicit and testable.
    """
    first_ce = F.cross_entropy(first_logits.float(), labels)
    if second_logits is None:
        zero = first_ce.detach().new_zeros(())
        return first_ce, first_ce.detach(), zero

    second_ce = F.cross_entropy(second_logits.float(), labels)
    classification_loss = 0.5 * (first_ce + second_ce)
    consistency_loss = symmetric_kl_divergence(first_logits, second_logits)
    total_loss = classification_loss + rdrop_alpha * consistency_loss
    return total_loss, classification_loss.detach(), consistency_loss.detach()
