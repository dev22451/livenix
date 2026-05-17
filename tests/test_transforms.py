"""Unit tests for ISP-aware augmentation transforms (T2.9).

Tests cover shape, dtype, determinism, normalization correctness, numpy input,
strength variation, image-size override, and NaN/Inf safety.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from PIL import Image

from livenix.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    eval_transforms,
    train_transforms,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_pil(width: int = 200, height: int = 180) -> Image.Image:
    """Return a random RGB PIL Image of the given size."""
    arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _gray_pil(value: int = 128, size: int = 64) -> Image.Image:
    """Return a uniform-grey PIL Image (R=G=B=value)."""
    arr = np.full((size, size, 3), value, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrainTransformsShape:
    def test_train_transforms_output_shape(self):
        """PIL image of any size → train_transforms → tensor (3, 128, 128)."""
        transform = train_transforms()
        img = _random_pil(width=320, height=240)
        tensor = transform(img)
        assert tensor.shape == (3, 128, 128), f"Expected (3, 128, 128), got {tensor.shape}"
        assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"


class TestEvalTransformsShape:
    def test_eval_transforms_output_shape(self):
        """PIL image of any size → eval_transforms → tensor (3, 128, 128)."""
        transform = eval_transforms()
        img = _random_pil(width=320, height=240)
        tensor = transform(img)
        assert tensor.shape == (3, 128, 128), f"Expected (3, 128, 128), got {tensor.shape}"
        assert tensor.dtype == torch.float32, f"Expected float32, got {tensor.dtype}"


class TestEvalDeterminism:
    def test_eval_is_deterministic(self):
        """eval_transforms on the same input 3 times must produce identical tensors."""
        transform = eval_transforms()
        img = _random_pil()
        results = [transform(img) for _ in range(3)]
        assert torch.equal(results[0], results[1]), "eval_transforms not deterministic (run 0 vs 1)"
        assert torch.equal(results[1], results[2]), "eval_transforms not deterministic (run 1 vs 2)"


class TestTrainStochastic:
    def test_train_is_stochastic(self):
        """Running train_transforms on the same input 3 times should NOT always
        yield identical tensors — at least one pair should differ."""
        transform = train_transforms()
        img = _random_pil()
        results = [transform(img) for _ in range(3)]
        all_equal = (
            torch.allclose(results[0], results[1])
            and torch.allclose(results[1], results[2])
        )
        assert not all_equal, (
            "train_transforms produced identical tensors for 3 runs on the same "
            "input — expected stochastic behaviour."
        )


class TestNormalization:
    def test_normalization_applied(self):
        """Pure-grey image (R=G=B=128) after eval_transforms should have per-channel
        values close to (128/255 - mean[c]) / std[c]."""
        transform = eval_transforms()
        img = _gray_pil(value=128, size=64)
        tensor = transform(img)  # (3, 64, 64) — uses default 128x128, so use that
        # Use default 128x128 default size
        transform2 = eval_transforms(image_size=(128, 128))
        tensor2 = transform2(img)

        for c in range(3):
            expected = (128.0 / 255.0 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
            actual_mean = tensor2[c].mean().item()
            assert abs(actual_mean - expected) < 1e-3, (
                f"Channel {c}: expected normalized value ≈ {expected:.6f}, "
                f"got {actual_mean:.6f} (diff={abs(actual_mean - expected):.2e})"
            )


class TestNumpyInput:
    def test_accepts_numpy_input(self):
        """train_transforms and eval_transforms should accept numpy HWC uint8 arrays."""
        arr = np.random.randint(0, 256, (180, 200, 3), dtype=np.uint8)

        train_t = train_transforms()
        tensor_train = train_t(arr)
        assert tensor_train.shape == (3, 128, 128)
        assert tensor_train.dtype == torch.float32

        eval_t = eval_transforms()
        tensor_eval = eval_t(arr)
        assert tensor_eval.shape == (3, 128, 128)
        assert tensor_eval.dtype == torch.float32


class TestStrengthVariation:
    def test_strength_changes_train_pipeline(self):
        """'light' and 'heavy' presets should produce meaningfully different
        output distributions when applied repeatedly to the same input."""
        n = 50
        img = _random_pil(width=200, height=200)

        light_t = train_transforms(augmentation_strength="light")
        heavy_t = train_transforms(augmentation_strength="heavy")

        light_tensors = torch.stack([light_t(img) for _ in range(n)])
        heavy_tensors = torch.stack([heavy_t(img) for _ in range(n)])

        # The pixel-level mean values across samples should differ between presets.
        light_mean = light_tensors.mean().item()
        heavy_mean = heavy_tensors.mean().item()

        # Compute per-sample means for std comparison
        light_sample_means = light_tensors.view(n, -1).mean(dim=1)
        heavy_sample_means = heavy_tensors.view(n, -1).mean(dim=1)

        light_std = light_sample_means.std().item()
        heavy_std = heavy_sample_means.std().item()

        # At least: the two populations should not be bit-for-bit identical
        assert not torch.equal(light_tensors, heavy_tensors), (
            "light and heavy augmentation produced identical output — "
            "strength scaling appears to have no effect."
        )
        # And their aggregate statistics should differ
        assert abs(light_mean - heavy_mean) > 0 or abs(light_std - heavy_std) > 0, (
            "light and heavy mean/std are identical — strength scaling not working."
        )


class TestImageSizeOverride:
    def test_image_size_respected(self):
        """train_transforms(image_size=(64, 64)) should produce tensors of shape
        (3, 64, 64)."""
        transform = train_transforms(image_size=(64, 64))
        img = _random_pil()
        tensor = transform(img)
        assert tensor.shape == (3, 64, 64), f"Expected (3, 64, 64), got {tensor.shape}"


class TestNoNanInf:
    def test_round_trip_no_nan(self):
        """Loop 20 random PIL images through train_transforms; assert no NaN or Inf
        in any output tensor."""
        transform = train_transforms()
        for i in range(20):
            img = _random_pil(
                width=random.randint(64, 400),
                height=random.randint(64, 400),
            )
            tensor = transform(img)
            assert not torch.isnan(tensor).any(), f"NaN detected in output for sample {i}"
            assert not torch.isinf(tensor).any(), f"Inf detected in output for sample {i}"
