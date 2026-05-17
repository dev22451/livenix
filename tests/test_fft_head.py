"""
Unit tests for FFTHead (T2.1).

Tests:
  1. Forward pass shape: FFTHead(in_channels=128) on (2,128,4,4) → (2,1,4,4).
  2. compute_target on (2,3,128,128) → shape (2,1,4,4), values in [0,1].
  3. loss(pred, target) returns a scalar tensor with gradients.
  4. Full pipeline: forward + compute_target + loss + backward all succeed
     without NaN/Inf.
"""

from __future__ import annotations

import torch
import pytest

from livenix.models.heads import FFTHead


# ---------------------------------------------------------------------------
# Test 1: forward pass output shape
# ---------------------------------------------------------------------------

def test_forward_shape() -> None:
    """FFTHead.forward on a (2, 128, 4, 4) feature map returns (2, 1, 4, 4)."""
    head = FFTHead(in_channels=128, hidden=128, target_size=(4, 4))
    feat = torch.randn(2, 128, 4, 4)
    pred = head(feat)
    assert pred.shape == (2, 1, 4, 4), (
        f"Expected output shape (2, 1, 4, 4), got {pred.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2: compute_target shape and value range
# ---------------------------------------------------------------------------

def test_compute_target_shape_and_range() -> None:
    """compute_target on (2, 3, 128, 128) → (2, 1, 4, 4) with values in [0, 1]."""
    image = torch.randn(2, 3, 128, 128)
    target = FFTHead.compute_target(image, target_size=(4, 4))
    assert target.shape == (2, 1, 4, 4), (
        f"Expected target shape (2, 1, 4, 4), got {target.shape}"
    )
    assert target.min().item() >= 0.0 - 1e-6, (
        f"Target values below 0: min={target.min().item()}"
    )
    assert target.max().item() <= 1.0 + 1e-6, (
        f"Target values above 1: max={target.max().item()}"
    )


# ---------------------------------------------------------------------------
# Test 3: loss returns scalar tensor with gradient
# ---------------------------------------------------------------------------

def test_loss_scalar_with_grad() -> None:
    """FFTHead.loss returns a scalar tensor and gradient flows through pred."""
    head = FFTHead(in_channels=128, hidden=128, target_size=(4, 4))
    # Use a leaf tensor so .grad is populated after backward.
    logits = torch.randn(2, 1, 4, 4, requires_grad=True)
    pred = torch.sigmoid(logits)  # non-leaf; gradient flows back to logits
    target = torch.rand(2, 1, 4, 4)  # already in [0, 1]

    loss = head.loss(pred, target)

    assert loss.shape == torch.Size([]), (
        f"Expected scalar (shape []), got {loss.shape}"
    )
    assert loss.requires_grad, "Loss tensor must require grad"

    # Verify backward runs without error and gradient reaches the leaf tensor.
    loss.backward()
    assert logits.grad is not None, "Gradient was not computed for logits leaf"


# ---------------------------------------------------------------------------
# Test 4: full training pipeline — forward + compute_target + loss + backward
# ---------------------------------------------------------------------------

def test_full_pipeline_no_nan_inf() -> None:
    """Full pipeline: feat → pred → loss + backward, no NaN/Inf."""
    head = FFTHead(in_channels=192, hidden=128, target_size=(4, 4))

    feat = torch.randn(2, 192, 4, 4)
    image = torch.randn(2, 3, 128, 128)

    pred = head(feat)
    target = FFTHead.compute_target(image, target_size=(4, 4))
    loss = head.loss(pred, target)

    # Sanity: no NaN or Inf anywhere
    assert not torch.isnan(pred).any(), "NaN in pred"
    assert not torch.isinf(pred).any(), "Inf in pred"
    assert not torch.isnan(target).any(), "NaN in target"
    assert not torch.isinf(target).any(), "Inf in target"
    assert not torch.isnan(loss), f"NaN in loss: {loss.item()}"
    assert not torch.isinf(loss), f"Inf in loss: {loss.item()}"

    # Backward must not raise
    loss.backward()

    # Verify at least one parameter got a gradient
    grads = [p.grad for p in head.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients were computed for any parameter"

    # No NaN/Inf in any gradient
    for g in grads:
        assert not torch.isnan(g).any(), "NaN in gradient"
        assert not torch.isinf(g).any(), "Inf in gradient"
