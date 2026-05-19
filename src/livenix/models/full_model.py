"""
Livenix full model — backbone + main head.

This is what gets exported (ONNX → TFLite → CoreML). Auxiliary heads are
not part of this module; they are added by the training pipeline in Week 2
and stripped before export.

Class index convention (3-class output):
    0 = real
    1 = print_spoof
    2 = replay_spoof
"""

from __future__ import annotations

import torch
import torch.nn as nn

from livenix.models.backbone import LivenixBackbone
from livenix.models.heads import MainHead


CLASS_INDEX = {
    "real": 0,
    "print_spoof": 1,
    "replay_spoof": 2,
}
CLASS_NAMES = ["real", "print_spoof", "replay_spoof"]


class LivenixModel(nn.Module):
    """Inference graph: input (B, 3, 128, 128) FP32 → (B, 3) logits.

    Use a softmax outside the model to get probabilities. Softmax is excluded
    from the exported graph so callers can pick their own threshold/calibration.
    """

    INPUT_SIZE = (128, 128)
    INPUT_CHANNELS = 3

    def __init__(
        self,
        backbone_name: str = "mobilenetv4_conv_small.e2400_r224_in1k",
        pretrained_backbone: bool = False,
        cdc_theta: float = 0.7,
        dropout: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = LivenixBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained_backbone,
            cdc_theta=cdc_theta,
        )
        self.head = MainHead(
            in_features=self.backbone.out_channels,
            dropout=dropout,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Backbone returns (B, C, H, W)
        feats = self.backbone(x)
        # Plain mean over spatial dims — portable across all export targets.
        # Avoids AdaptiveAvgPool2d which sometimes serializes oddly to TFLite.
        pooled = feats.mean(dim=(2, 3))
        return self.head(pooled)
