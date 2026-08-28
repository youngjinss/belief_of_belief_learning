"""Tensor helpers used by the ToMnet model.

Moved out of the per-experiment utils.py. `calculate_sr_loss_kl_divergence` was
byte-identical across exp5..exp8; `spatialize_action` came in with exp7 and is
taken here in its exp8 form. Neither is used anywhere outside the model.
"""

import torch
import torch.nn.functional as F

def calculate_sr_loss_kl_divergence(sr_pred, sr_target):
    """
    Calculate SR loss using KL divergence for probability distributions
    Vectorized version for efficiency (adapted from experiment 5)

    Args:
        sr_pred: Predicted SR maps (batch_size, 3, height, width) - already normalized by softmax
        sr_target: Target SR maps (batch_size, 3, height, width) - raw values, need normalization

    Returns:
        sr_loss: KL divergence loss averaged over discount factors
    """
    batch_size, n_gammas, height, width = sr_pred.shape

    # Vectorized reshape: (batch_size, 3, height*width)
    sr_pred_flat = sr_pred.view(batch_size, n_gammas, -1)
    sr_target_flat = sr_target.view(batch_size, n_gammas, -1)

    # SR predictions are already normalized by softmax in the model
    # Normalize SR targets to probability distributions (sum=1 across spatial locations)
    sr_target_flat = sr_target_flat / (sr_target_flat.sum(dim=2, keepdim=True) + 1e-8)

    # Add small epsilon to avoid log(0)
    sr_pred_flat_safe = sr_pred_flat + 1e-8
    sr_target_flat_safe = sr_target_flat + 1e-8

    # Vectorized KL divergence computation for all gammas at once
    # KL(target || pred) = sum(target * log(target/pred))
    kl_loss = torch.nn.functional.kl_div(
        sr_pred_flat_safe.log(),
        sr_target_flat_safe,
        reduction="none",  # Keep batch and gamma dimensions
    )

    # Sum over spatial dimension, then average over batch and gamma
    kl_loss = kl_loss.sum(dim=2)  # (batch_size, n_gammas)
    kl_loss = kl_loss.mean()  # Average over batch and gamma dimensions

    return kl_loss

def spatialize_action(
    action_indices: torch.Tensor, height: int, width: int, action_space: int = 7
) -> torch.Tensor:
    """
    Convert action indices to spatial representation

    Args:
        action_indices: (batch_size,) - action indices for each sample
        height, width: spatial dimensions
        action_space: Number of possible actions (default: 7)

    Returns:
        Spatialized actions: (batch_size, 1, height, width)
    """
    batch_size = action_indices.size(0)
    device = action_indices.device

    # Create spatial action maps
    action_maps = torch.zeros(
        batch_size, 1, height, width, device=device, dtype=torch.float32
    )

    # Vectorized approach for better performance
    if action_space > 1:
        action_values = action_indices
    else:
        action_values = torch.zeros_like(action_indices, dtype=torch.float32)

    # Broadcast to spatial dimensions
    action_maps[:, 0, :, :] = action_values.view(-1, 1, 1)

    return action_maps
