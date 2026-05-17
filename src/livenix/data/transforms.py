"""ISP-aware augmentation transforms for face-PAD training.

Simulates the diversity of mobile-camera image-signal-processing pipelines —
the single largest source of domain gap between academic PAD datasets and real
phones with different ISPs.

Two public entry points:
    train_transforms()  – stochastic, all augmentations active
    eval_transforms()   – deterministic: Resize → Normalize → ToTensorV2

Both return a callable that accepts a PIL.Image OR a numpy HWC uint8 array
and returns a (3, H, W) float32 torch.Tensor.

Albumentations >= 1.4 required (already in pyproject.toml).
Tested / version-defended against albumentations 2.x API changes.
"""

from __future__ import annotations

from typing import Callable, Union

import numpy as np
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2
from packaging.version import Version as _Version

_ALBUMENTATIONS_VERSION = _Version(A.__version__)
_ALB_V2_PLUS = _ALBUMENTATIONS_VERSION >= _Version("2.0.0")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Must match LivenixModel.INPUT_SIZE — hardcoded here to avoid circular import
# risk from livenix.models.full_model.  Keep in sync if model INPUT_SIZE changes.
_DEFAULT_INPUT_SIZE = (128, 128)

# Strength-scaling factors applied to augmentation magnitudes.
_STRENGTH_SCALE = {
    "light": 0.5,
    "medium": 1.0,
    "heavy": 1.5,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pil_to_numpy(im: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a numpy HWC uint8 array (RGB)."""
    return np.asarray(im.convert("RGB"))


def _make_callable(compose: A.Compose) -> Callable:
    """Wrap an albumentations Compose so it accepts PIL or numpy and returns
    a torch.Tensor of shape (3, H, W) float32."""

    def _transform(image: Union[Image.Image, np.ndarray]) -> "torch.Tensor":
        arr = _pil_to_numpy(image) if isinstance(image, Image.Image) else image
        out = compose(image=arr)
        return out["image"]

    return _transform


def _build_gauss_noise(var_lower: float, var_upper: float, p: float) -> A.BasicTransform:
    """Build a GaussNoise transform, defending against API changes between
    albumentations versions.

    * albumentations < 2.x: ``GaussNoise(var_limit=(lo, hi), ...)``
    * albumentations >= 2.x: ``GaussNoise(std_range=(lo, hi), ...)``
      where std_range values are normalised (0–1 scale, not 0–255^2 variance).
    """
    if _ALB_V2_PLUS:
        # albumentations 2.x renamed var_limit → std_range and changed units.
        # var is pixel-scale; std = sqrt(var) / 255 to get 0-1 normalised std.
        std_lower = float(np.sqrt(var_lower)) / 255.0
        std_upper = float(np.sqrt(var_upper)) / 255.0
        return A.GaussNoise(std_range=(std_lower, std_upper), p=p)
    else:
        return A.GaussNoise(var_limit=(var_lower, var_upper), p=p)  # type: ignore[call-arg]


def _build_image_compression(q_lower: int, q_upper: int, p: float) -> A.BasicTransform:
    """Build an ImageCompression transform, defending against API changes.

    * albumentations < 2.x: ``quality_lower`` / ``quality_upper`` kwargs
    * albumentations >= 2.x: ``quality_range`` tuple
    """
    if _ALB_V2_PLUS:
        return A.ImageCompression(quality_range=(q_lower, q_upper), p=p)
    else:
        return A.ImageCompression(quality_lower=q_lower, quality_upper=q_upper, p=p)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def train_transforms(
    image_size: tuple[int, int] = _DEFAULT_INPUT_SIZE,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
    augmentation_strength: str = "medium",
) -> Callable:
    """Build a stochastic training-time transform pipeline.

    Returns a callable that takes a PIL.Image or numpy HWC uint8 array and
    returns a torch.Tensor of shape (3, H, W) float32, normalized to
    ImageNet statistics.

    Args:
        image_size: Target (H, W) after the random-resize-crop. Default 128×128.
        mean: Per-channel normalization mean.
        std:  Per-channel normalization std.
        augmentation_strength: One of ``"light"`` | ``"medium"`` | ``"heavy"``.
            Scales the probability and magnitude of each augmentation.
            - ``"light"``  → 0.5× defaults
            - ``"medium"`` → 1.0× defaults (factory default)
            - ``"heavy"``  → 1.5× defaults

    Pipeline (in order):
        1. RandomResizedCrop to ``image_size`` with scale=(0.7, 1.0)
        2. HorizontalFlip (p=0.5)
        3. ColorJitter (brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
           — simulates ISP white-balance / tone-mapping variation
        4. RandomGamma (gamma_limit=(80, 120))
        5. ImageCompression (quality≈40–95) — simulates JPEG-save at varying quality
        6. OneOf([MotionBlur, MedianBlur, GaussianBlur], p=0.3)
        7. GaussNoise (p=0.3) — simulated sensor noise
        8. Normalize(mean, std)
        9. ToTensorV2 → (3, H, W) torch.float32

    Returns:
        Callable[[PIL.Image | np.ndarray], torch.Tensor]
    """
    if augmentation_strength not in _STRENGTH_SCALE:
        raise ValueError(
            f"augmentation_strength must be one of {list(_STRENGTH_SCALE)}, "
            f"got {augmentation_strength!r}"
        )
    s = _STRENGTH_SCALE[augmentation_strength]

    h, w = image_size

    # Scale probabilities (clamped to [0, 1]) and magnitudes.
    flip_p = min(1.0, 0.5 * s)

    # ColorJitter: brightness/contrast/saturation are "max delta" style offsets
    # in albumentations (float → symmetric range, or tuple for asymmetric).
    cj_b = 0.3 * s
    cj_c = 0.3 * s
    cj_sat = 0.3 * s
    cj_hue = 0.05 * s
    cj_p = min(1.0, 0.8 * s)

    # RandomGamma: gamma_limit = (low_pct, high_pct); center 100, spread scales
    gamma_spread = int(20 * s)  # medium → (80, 120)
    gamma_lo = max(1, 100 - gamma_spread)
    gamma_hi = 100 + gamma_spread
    gamma_p = min(1.0, 0.7 * s)

    # ImageCompression: wider quality range for heavier augmentation
    q_spread = int(55 * s)  # medium → 40–95
    q_lower = max(1, 95 - q_spread)
    q_upper = 95
    jpeg_p = min(1.0, 0.6 * s)

    # Blur OneOf
    blur_p = min(1.0, 0.3 * s)

    # GaussNoise
    var_lo = 5.0 * s
    var_hi = 50.0 * s
    noise_p = min(1.0, 0.3 * s)

    transforms_list = [
        A.RandomResizedCrop(size=(h, w), scale=(0.7, 1.0), p=1.0),
        A.HorizontalFlip(p=flip_p),
        A.ColorJitter(
            brightness=cj_b,
            contrast=cj_c,
            saturation=cj_sat,
            hue=cj_hue,
            p=cj_p,
        ),
        A.RandomGamma(gamma_limit=(gamma_lo, gamma_hi), p=gamma_p),
        _build_image_compression(q_lower=q_lower, q_upper=q_upper, p=jpeg_p),
        A.OneOf(
            [
                A.MotionBlur(p=1.0),
                A.MedianBlur(p=1.0),
                A.GaussianBlur(p=1.0),
            ],
            p=blur_p,
        ),
        _build_gauss_noise(var_lower=var_lo, var_upper=var_hi, p=noise_p),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    compose = A.Compose(transforms_list)
    return _make_callable(compose)


def eval_transforms(
    image_size: tuple[int, int] = _DEFAULT_INPUT_SIZE,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> Callable:
    """Build a deterministic eval-time transform pipeline.

    Returns a callable that takes a PIL.Image or numpy HWC uint8 array and
    returns a torch.Tensor of shape (3, H, W) float32, normalized to
    ImageNet statistics.

    Args:
        image_size: Target (H, W) after resize. Default 128×128.
        mean: Per-channel normalization mean.
        std:  Per-channel normalization std.

    Pipeline (in order):
        1. Resize to ``image_size`` (no crop — preserves bounding-box content)
        2. Normalize(mean, std)
        3. ToTensorV2 → (3, H, W) torch.float32

    Returns:
        Callable[[PIL.Image | np.ndarray], torch.Tensor]
    """
    h, w = image_size
    compose = A.Compose(
        [
            A.Resize(height=h, width=w),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    )
    return _make_callable(compose)
