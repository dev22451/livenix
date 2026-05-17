"""
Livenix model heads.

- MainHead: 3-class classifier (real / print_spoof / replay_spoof). Used at
  both training and inference time. This is the ONLY head present in exported
  artifacts (ONNX/TFLite/CoreML).

- FFTHead, SigLIPDistillHead, PatchContrastiveHead: training-only auxiliary
  heads. Implemented in Week 2. Raise NotImplementedError until then.

Cosine-margin / ArcFace-style decision tightening lives in the LOSS function,
not in the head — keeps the head a plain linear so it exports cleanly.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MainHead(nn.Module):
    """3-class classifier head.

    Args:
        in_features: number of input features after global pooling.
        num_classes: 3 (real, print_spoof, replay_spoof).
        dropout: optional dropout before the linear layer.
    """

    NUM_CLASSES = 3

    def __init__(self, in_features: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x))


class FFTHead(nn.Module):
    """Auxiliary FFT-magnitude branch. Training-only, stripped at export."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError("FFTHead is implemented in Week 2.")


class SigLIPDistillHead(nn.Module):
    """Auxiliary SigLIP-So400m feature distillation. Training-only."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError("SigLIPDistillHead is implemented in Week 2.")


class PatchContrastiveHead(nn.Module):
    """Auxiliary patch-wise contrastive (NT-Xent) head. Training-only."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError("PatchContrastiveHead is implemented in Week 2.")
