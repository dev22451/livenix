"""Unit tests for SigLIPDistillHead (T2.2).

The actual SigLIP-So400m teacher is NOT used here. Teacher embeddings are
mocked as random tensors of the correct shape, keeping these tests
dependency-free and CI-fast. Teacher wiring is T2.11's scope.
"""

import torch
import pytest

from livenix.models.heads import SigLIPDistillHead


def test_forward_shape():
    """head(randn(2, 192, 4, 4)) → shape (2, 1152), L2 norm ≈ 1.0 per row."""
    head = SigLIPDistillHead(in_channels=192, teacher_dim=1152)
    feat = torch.randn(2, 192, 4, 4)
    out = head(feat)

    assert out.shape == (2, 1152), f"Expected (2, 1152), got {out.shape}"

    # L2 norm of each row should be ≈ 1.0
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5), (
        f"Expected unit norms, got {norms}"
    )


def test_loss_scalar_with_grad():
    """Loss is scalar and gradient flows to projection weights."""
    head = SigLIPDistillHead(in_channels=64, teacher_dim=1152)
    feat = torch.randn(3, 64, 4, 4)
    teacher_emb = torch.randn(3, 1152)

    student_emb = head(feat)
    loss = head.loss(student_emb, teacher_emb)

    # Must be a scalar
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"

    # Gradient must flow back to projection weights
    loss.backward()
    grad = head.projection.weight.grad
    assert grad is not None, "No gradient on projection.weight"
    assert grad.abs().sum().item() > 0, "Gradient is all zeros"


def test_teacher_dim_configurable():
    """SigLIPDistillHead(in_channels=64, teacher_dim=512) → output shape (B, 512)."""
    head = SigLIPDistillHead(in_channels=64, teacher_dim=512)
    feat = torch.randn(4, 64, 8, 8)
    out = head(feat)

    assert out.shape == (4, 512), f"Expected (4, 512), got {out.shape}"

    # Sanity: norms still ≈ 1
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_loss_zero_when_aligned():
    """Loss ≈ 0 when student_emb already equals the normalized teacher_emb."""
    head = SigLIPDistillHead(in_channels=128, teacher_dim=1152)
    feat = torch.randn(2, 128, 4, 4)

    # student_emb is the L2-normalized output of the head
    student_emb = head(feat)

    # Use the student output as the "teacher" embedding — after normalization
    # inside loss(), it will be identical to student_emb, so MSE = 0.
    teacher_emb = student_emb.detach().clone()

    loss = head.loss(student_emb, teacher_emb)
    assert loss.item() < 1e-6, f"Expected loss ≈ 0, got {loss.item()}"
