"""
Unit tests for PatchContrastiveHead (T2.3).

Tests:
  1. test_forward_shape: head on (4, 192, 4, 4) → (64, 128), per-row norm ≈ 1.
  2. test_loss_scalar_with_grad: forward + loss is scalar; backward reaches projector.
  3. test_loss_decreases_under_overfit: 50 SGD steps on synthetic 2-class data,
     final loss < 0.5 * initial loss.
  4. test_handles_single_class_in_batch: all-same-class labels → finite, non-NaN.
  5. test_temperature_affects_loss: τ=0.07 vs τ=0.7 produce different loss values.
"""

from __future__ import annotations

import torch
import pytest

from livenix.models.heads import PatchContrastiveHead


# ---------------------------------------------------------------------------
# Test 1: forward pass shape and L2 normalization
# ---------------------------------------------------------------------------

def test_forward_shape() -> None:
    """PatchContrastiveHead.forward on (4, 192, 4, 4) → (64, 128); per-row norm ≈ 1."""
    head = PatchContrastiveHead(in_channels=192, proj_dim=128)
    feat = torch.randn(4, 192, 4, 4)

    emb = head(feat)

    # Shape: B=4, H=4, W=4 → N = 4*4*4 = 64; proj_dim = 128
    assert emb.shape == (64, 128), (
        f"Expected shape (64, 128), got {emb.shape}"
    )

    # Every row must be unit-norm after F.normalize
    norms = emb.norm(dim=-1)  # (64,)
    assert torch.allclose(norms, torch.ones(64), atol=1e-5), (
        f"Per-row L2 norms deviate from 1.0: min={norms.min().item():.6f}, "
        f"max={norms.max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 2: loss returns a scalar with gradient; backward reaches projector
# ---------------------------------------------------------------------------

def test_loss_scalar_with_grad() -> None:
    """forward + loss with 4-image batch is a scalar; grad flows to projector weights."""
    head = PatchContrastiveHead(in_channels=192, proj_dim=128)
    feat = torch.randn(4, 192, 4, 4)
    labels = torch.tensor([0, 1, 2, 0])  # classes: real, print, replay, real

    emb = head(feat)                               # (64, 128)
    loss = head.loss(emb, labels, num_patches_per_sample=16)

    # Must be a scalar
    assert loss.shape == torch.Size([]), (
        f"Expected scalar loss (shape []), got {loss.shape}"
    )
    assert loss.requires_grad, "Loss tensor must require grad"
    assert not torch.isnan(loss), f"Loss is NaN: {loss.item()}"
    assert not torch.isinf(loss), f"Loss is Inf: {loss.item()}"

    # Backward must not raise and must reach the projector conv weights
    loss.backward()

    conv_weight = list(head.projector.parameters())[0]
    assert conv_weight.grad is not None, (
        "No gradient reached the projector Conv2d weights"
    )
    assert not torch.isnan(conv_weight.grad).any(), "NaN in projector gradient"
    assert not torch.isinf(conv_weight.grad).any(), "Inf in projector gradient"


# ---------------------------------------------------------------------------
# Test 3: loss decreases under overfit (well-formed contrastive objective)
# ---------------------------------------------------------------------------

def test_loss_decreases_under_overfit() -> None:
    """50 SGD steps on 2-class synthetic data: final loss < 0.5 * initial loss."""
    torch.manual_seed(42)

    # 8 patches, 2 classes (4 patches each).  We bypass the CNN projector and
    # work directly with the loss() method using a learnable embedding table
    # so we can overfit cleanly without needing a spatial feature map.
    proj_dim = 32
    n_patches = 8
    labels_per_sample = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])  # 8 "images", 1 patch each

    # Learnable raw embeddings (we will normalize inside the loop)
    raw_emb = torch.randn(n_patches, proj_dim, requires_grad=True)

    optimizer = torch.optim.SGD([raw_emb], lr=0.5)

    # We instantiate a PatchContrastiveHead only for its .loss() method;
    # temperature set deliberately low to produce non-trivial gradients.
    head = PatchContrastiveHead(in_channels=1, proj_dim=proj_dim, temperature=0.07)

    import torch.nn.functional as F

    initial_loss: float | None = None

    for step in range(50):
        optimizer.zero_grad()
        emb = F.normalize(raw_emb, dim=-1)
        loss = head.loss(emb, labels_per_sample, num_patches_per_sample=1)
        if initial_loss is None:
            initial_loss = loss.item()
        loss.backward()
        optimizer.step()

    final_loss = loss.item()

    assert initial_loss is not None
    assert final_loss < 0.5 * initial_loss, (
        f"Loss did not decrease sufficiently under overfit: "
        f"initial={initial_loss:.4f}, final={final_loss:.4f} "
        f"(expected final < {0.5 * initial_loss:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 4: degenerate case — all patches belong to the same class
# ---------------------------------------------------------------------------

def test_handles_single_class_in_batch() -> None:
    """labels=[0,0,0,0] (single class): loss is finite, not NaN — all patches are positives."""
    head = PatchContrastiveHead(in_channels=192, proj_dim=128)
    feat = torch.randn(4, 192, 4, 4)
    labels = torch.tensor([0, 0, 0, 0])  # all real

    emb = head(feat)
    loss = head.loss(emb, labels, num_patches_per_sample=16)

    assert not torch.isnan(loss), f"Loss is NaN for single-class batch: {loss.item()}"
    assert not torch.isinf(loss), f"Loss is Inf for single-class batch: {loss.item()}"
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"

    # Backward should still succeed (all anchors have positives)
    loss.backward()


# ---------------------------------------------------------------------------
# Test 5: temperature parameter actually affects the loss value
# ---------------------------------------------------------------------------

def test_temperature_affects_loss() -> None:
    """Same input + labels with τ=0.07 vs τ=0.7 must produce different loss values."""
    torch.manual_seed(0)
    feat = torch.randn(4, 192, 4, 4)
    labels = torch.tensor([0, 1, 2, 0])

    head_cold = PatchContrastiveHead(in_channels=192, proj_dim=128, temperature=0.07)
    head_warm = PatchContrastiveHead(in_channels=192, proj_dim=128, temperature=0.70)

    # Use the same projector weights so the only difference is temperature
    with torch.no_grad():
        for p_cold, p_warm in zip(head_cold.projector.parameters(),
                                   head_warm.projector.parameters()):
            p_warm.copy_(p_cold)

    # Run both in eval mode so BN behaves identically
    head_cold.eval()
    head_warm.eval()

    with torch.no_grad():
        emb_cold = head_cold(feat)
        loss_cold = head_cold.loss(emb_cold, labels, num_patches_per_sample=16)

        emb_warm = head_warm(feat)
        loss_warm = head_warm.loss(emb_warm, labels, num_patches_per_sample=16)

    assert not torch.isclose(loss_cold, loss_warm, atol=1e-6), (
        f"Expected different losses for τ=0.07 vs τ=0.7, "
        f"but got loss_cold={loss_cold.item():.6f}, loss_warm={loss_warm.item():.6f}"
    )
