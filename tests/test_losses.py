"""
Unit tests for livenix.train.losses.

Coverage:
  FocalLoss:         shape, overfit, gamma sensitivity
  AsymmetricLoss:    shape, numerical stability
  CosineMarginLoss:  shape, overfit, m=0 equivalence
  CombinedLoss:      main-only, all-aux, partial-aux, cosine-margin dispatch
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import pytest

from livenix.train.losses import (
    AsymmetricLoss,
    CombinedLoss,
    CosineMarginLoss,
    FocalLoss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(B: int = 16, C: int = 3, seed: int = 42):
    """Return random logits (B, C) and integer targets (B,)."""
    torch.manual_seed(seed)
    logits = torch.randn(B, C)
    targets = torch.randint(0, C, (B,))
    return logits, targets


def _overfit_linear(
    loss_fn: nn.Module,
    input_dim: int,
    num_classes: int,
    n_samples: int = 50,
    n_epochs: int = 300,
    lr: float = 0.1,
    use_features: bool = False,
    seed: int = 0,
) -> float:
    """Train a single linear layer with ``loss_fn`` and return final accuracy.

    If ``use_features`` is True, the layer output is passed as "features" to a
    CosineMarginLoss; otherwise logits are passed through any loss that accepts them.
    """
    torch.manual_seed(seed)
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, num_classes, (n_samples,))

    linear = nn.Linear(input_dim, input_dim if use_features else num_classes)
    optimizer = optim.SGD(
        list(linear.parameters()) + list(loss_fn.parameters()),
        lr=lr,
        momentum=0.9,
    )

    for _ in range(n_epochs):
        optimizer.zero_grad()
        out = linear(X)
        if use_features:
            loss = loss_fn(out, y)
        else:
            loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        out = linear(X)
        if use_features:
            # For CosineMarginLoss: cosine similarity → argmax
            feat_norm = nn.functional.normalize(out, dim=1)
            w_norm = nn.functional.normalize(loss_fn.weight, dim=1)
            cos_sim = feat_norm @ w_norm.t()
            preds = cos_sim.argmax(dim=1)
        else:
            preds = out.argmax(dim=1)
    accuracy = (preds == y).float().mean().item()
    return accuracy


# ---------------------------------------------------------------------------
# FocalLoss
# ---------------------------------------------------------------------------

class TestFocalLoss:

    def test_focal_shape(self):
        """FocalLoss on (16, 3) logits + (16,) targets → scalar."""
        logits, targets = _make_batch(B=16, C=3)
        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, targets)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
        assert loss.item() > 0, "Loss should be positive"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_focal_overfit(self):
        """FocalLoss: training a linear classifier on 50 samples → ≥ 95% accuracy."""
        loss_fn = FocalLoss(gamma=2.0)
        acc = _overfit_linear(
            loss_fn,
            input_dim=16,
            num_classes=3,
            n_samples=50,
            n_epochs=600,
            lr=0.1,
        )
        assert acc >= 0.95, f"Expected overfit accuracy ≥ 0.95, got {acc:.3f}"

    def test_focal_gamma_changes_loss(self):
        """γ=0 (CE) and γ=2 (focal) should produce different loss values."""
        logits, targets = _make_batch(B=32, C=3, seed=7)
        loss_gamma0 = FocalLoss(gamma=0.0)(logits, targets)
        loss_gamma2 = FocalLoss(gamma=2.0)(logits, targets)
        assert not torch.allclose(loss_gamma0, loss_gamma2), (
            "FocalLoss with γ=0 and γ=2 should give different values"
        )


# ---------------------------------------------------------------------------
# AsymmetricLoss
# ---------------------------------------------------------------------------

class TestAsymmetricLoss:

    def test_asym_shape(self):
        """AsymmetricLoss on (16, 3) logits + (16,) targets → scalar."""
        logits, targets = _make_batch(B=16, C=3)
        loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
        loss = loss_fn(logits, targets)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_asym_finite_with_extreme_logits(self):
        """AsymmetricLoss should not produce NaN/Inf with very large/small logits."""
        B, C = 16, 3
        torch.manual_seed(99)
        # Mix of very large positive and very large negative logits
        logits = torch.cat([
            torch.full((B // 2, C), 1e4),
            torch.full((B // 2, C), -1e4),
        ], dim=0)
        targets = torch.randint(0, C, (B,))
        loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
        loss = loss_fn(logits, targets)
        assert torch.isfinite(loss), f"Expected finite loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# CosineMarginLoss
# ---------------------------------------------------------------------------

class TestCosineMarginLoss:

    def test_cosine_margin_shape(self):
        """CosineMarginLoss on features (16, 64) + targets (16,) → scalar."""
        torch.manual_seed(0)
        features = torch.randn(16, 64)
        targets = torch.randint(0, 3, (16,))
        loss_fn = CosineMarginLoss(in_features=64, num_classes=3, margin=0.3, scale=30.0)
        loss = loss_fn(features, targets)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_cosine_margin_overfit(self):
        """CosineMarginLoss: training on 50 samples → ≥ 95% accuracy."""
        loss_fn = CosineMarginLoss(in_features=32, num_classes=3, margin=0.3, scale=30.0)
        acc = _overfit_linear(
            loss_fn,
            input_dim=32,
            num_classes=3,
            n_samples=50,
            n_epochs=500,
            lr=0.05,
            use_features=True,
        )
        assert acc >= 0.95, f"Expected overfit accuracy ≥ 0.95, got {acc:.3f}"

    def test_cosine_margin_no_margin_recovers_cosine_softmax(self):
        """With m=0 and s=1 CosineMarginLoss should equal plain cosine-softmax CE.

        Cosine-softmax CE:
            logits[i,j] = normalize(features[i]) · normalize(weight[j])
            loss = CE(logits * scale, targets)   with scale=1, no margin.
        """
        torch.manual_seed(42)
        B, D, C = 8, 16, 3
        features = torch.randn(B, D)
        targets = torch.randint(0, C, (B,))

        loss_fn = CosineMarginLoss(in_features=D, num_classes=C, margin=0.0, scale=1.0)

        loss_module = loss_fn(features, targets)

        # Manual reference: same weight matrix, no margin, scale=1
        feat_norm = nn.functional.normalize(features, dim=1)
        w_norm = nn.functional.normalize(loss_fn.weight.detach(), dim=1)
        cos_sim = feat_norm @ w_norm.t()
        loss_manual = nn.functional.cross_entropy(cos_sim, targets)

        assert torch.allclose(loss_module, loss_manual, atol=1e-5), (
            f"m=0, s=1 CosineMarginLoss ({loss_module.item():.6f}) should equal "
            f"manual cosine CE ({loss_manual.item():.6f})"
        )


# ---------------------------------------------------------------------------
# CombinedLoss
# ---------------------------------------------------------------------------

class TestCombinedLoss:

    def _make_combined(self, use_focal: bool = True) -> CombinedLoss:
        main_fn = FocalLoss(gamma=2.0) if use_focal else AsymmetricLoss()
        return CombinedLoss(main_fn)

    def test_combined_main_only(self):
        """No aux losses → total == main."""
        logits, targets = _make_batch(B=8, C=3)
        combined = self._make_combined()
        result = combined(
            main_inputs={"logits": logits, "targets": targets},
            aux_losses=None,
        )
        assert set(result.keys()) == {"main", "siglip", "fft", "patch", "total"}
        assert torch.allclose(result["total"], result["main"]), (
            "With no aux losses, total should equal main"
        )

    def test_combined_with_aux(self):
        """All 3 aux losses provided → total == main + 0.5*siglip + 0.2*fft + 0.3*patch."""
        logits, targets = _make_batch(B=8, C=3, seed=1)
        combined = self._make_combined()

        torch.manual_seed(5)
        siglip_val = torch.tensor(0.4)
        fft_val    = torch.tensor(0.6)
        patch_val  = torch.tensor(0.8)

        result = combined(
            main_inputs={"logits": logits, "targets": targets},
            aux_losses={"siglip": siglip_val, "fft": fft_val, "patch": patch_val},
        )

        expected_total = (
            result["main"]
            + 0.5 * siglip_val
            + 0.2 * fft_val
            + 0.3 * patch_val
        )
        assert torch.allclose(result["total"], expected_total, atol=1e-6), (
            f"total {result['total'].item():.6f} != expected {expected_total.item():.6f}"
        )

    def test_combined_partial_aux(self):
        """Only siglip + patch provided → total == main + 0.5*siglip + 0.3*patch (no fft)."""
        logits, targets = _make_batch(B=8, C=3, seed=2)
        combined = self._make_combined()

        siglip_val = torch.tensor(0.3)
        patch_val  = torch.tensor(0.5)

        result = combined(
            main_inputs={"logits": logits, "targets": targets},
            aux_losses={"siglip": siglip_val, "patch": patch_val},
        )

        expected_total = result["main"] + 0.5 * siglip_val + 0.3 * patch_val
        assert torch.allclose(result["total"], expected_total, atol=1e-6), (
            f"total {result['total'].item():.6f} != expected {expected_total.item():.6f}"
        )
        # fft term should be zero
        assert result["fft"].item() == 0.0, "fft term should be 0 when not provided"

    def test_combined_with_cosine_margin(self):
        """main_inputs has 'features', main_loss_fn is CosineMarginLoss → dispatches correctly."""
        D = 32
        torch.manual_seed(10)
        features = torch.randn(8, D)
        targets  = torch.randint(0, 3, (8,))

        cosine_fn = CosineMarginLoss(in_features=D, num_classes=3)
        combined  = CombinedLoss(cosine_fn)

        result = combined(
            main_inputs={"features": features, "targets": targets},
            aux_losses=None,
        )
        assert torch.isfinite(result["total"]), "total should be finite"
        assert torch.allclose(result["total"], result["main"]), (
            "No aux → total == main"
        )
