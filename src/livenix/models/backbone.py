"""
Livenix backbone: MobileNetV4-Conv-Small with a Central Difference Convolution
(CDC) stem prepended to the first conv.

CDC stem implementation refactored as standard ops (Conv − θ·sum-Conv) so the
forward graph contains only ops supported by ONNX, TFLite, CoreML, and NCNN
export paths. No custom autograd.

References:
- CDC: Yu et al., "Searching Central Difference Convolutional Networks for
  Face Anti-Spoofing", CVPR 2020.
- MobileNetV4: Qin et al., "MobileNetV4 — Universal Models for the Mobile
  Ecosystem", 2024.

This is our own implementation. No code is taken from any third-party repo.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class CDCStem(nn.Module):
    """
    Central Difference Convolution stem.

    Equivalent to:
        y = conv(x, W) - theta * conv(x, W_summed_per_filter_then_1x1)

    Implemented with two standard nn.Conv2d ops sharing a derived weight,
    so the export graph stays portable.

    Args:
        in_channels: input channel count (3 for RGB).
        out_channels: output channel count (matches first stage of MobileNetV4
            so the backbone slots in cleanly).
        kernel_size: 3 by default.
        stride: 2 (matches MobileNetV4 conv stem stride).
        padding: 1.
        theta: CDC mixing coefficient (Yu et al. recommend 0.7).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 32,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        theta: float = 0.7,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU6(inplace=True)
        self.stride = stride
        self.theta = theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normal convolution path
        out_normal = self.conv(x)

        # Central difference path: equivalent to convolving with a kernel whose
        # sum-per-filter is multiplied with the input via 1x1 convolution.
        kernel_sum = self.conv.weight.sum(dim=(2, 3), keepdim=True)
        out_diff = F.conv2d(
            x,
            kernel_sum,
            bias=None,
            stride=self.stride,
            padding=0,
        )

        return self.act(self.bn(out_normal - self.theta * out_diff))


class LivenixBackbone(nn.Module):
    """
    CDC stem + MobileNetV4-Conv-Small feature extractor.

    The timm MobileNetV4 conv-small variant is used WITHOUT its first
    conv (replaced by our CDC stem) so the model has the CDC inductive
    bias at the input while reusing the proven MobileNetV4 body.

    Output: feature map of shape (B, C, H, W) where C is the backbone's
    last channel count and (H, W) depends on input spatial size.
    """

    def __init__(
        self,
        backbone_name: str = "mobilenetv4_conv_small.e2400_r224_in1k",
        pretrained: bool = False,
        cdc_theta: float = 0.7,
    ) -> None:
        super().__init__()
        # features_only=True returns intermediate feature maps; we use the
        # final feature map for the classification head.
        self.body = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(-1,),  # only the final feature map
        )

        # Discover the channels the stem of the body expects so our CDC stem
        # outputs the right shape, and the channels the final feature map has
        # for the head to consume.
        body_first = self._find_first_conv(self.body)
        if body_first is None:
            raise RuntimeError("Could not locate first Conv2d in timm backbone")
        stem_out_channels = body_first.out_channels

        # Build our CDC stem matching the original stem's output channels +
        # stride, then bypass the body's original stem in forward.
        self.cdc_stem = CDCStem(
            in_channels=3,
            out_channels=stem_out_channels,
            kernel_size=body_first.kernel_size[0],
            stride=body_first.stride[0],
            padding=body_first.padding[0] if isinstance(body_first.padding, tuple) else body_first.padding,
            theta=cdc_theta,
        )

        # Cache original stem so we can replace it with Identity at runtime
        # but still let the body define the structure correctly. We replace
        # the first conv module in-place with Identity.
        self._replace_first_conv_with_identity()

        # Channel count of the final feature map (for head wiring)
        self.out_channels = self.body.feature_info.channels()[-1]

    @staticmethod
    def _find_first_conv(module: nn.Module) -> nn.Conv2d | None:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                return m
        return None

    def _replace_first_conv_with_identity(self) -> None:
        """Walk the body and replace the first Conv2d module with Identity.

        We do this by traversing named_modules and using setattr on the parent.
        """
        first_conv_path = None
        for name, m in self.body.named_modules():
            if isinstance(m, nn.Conv2d):
                first_conv_path = name
                break
        if first_conv_path is None:
            return

        # Locate the parent module and the attribute name of the first conv
        parts = first_conv_path.split(".")
        parent = self.body
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], nn.Identity())

        # Same for the matching BN immediately after, if present at the same
        # parent level: MobileNetV4 stems usually pack conv+bn+act in a small
        # ConvBnAct; replacing the first Conv2d with Identity inside the
        # ConvBnAct still produces a working module because BN of zeros + act
        # is benign for downstream — but we DO want to use OUR stem's output
        # going into the body. To do that cleanly, we set the entire first
        # ConvBnAct (the parent) to Identity instead, IF the parent itself is
        # the model's first stem block.
        # For MobileNetV4 timm models, the first block is `conv_stem` (single
        # ConvBnAct). We replace that whole block with Identity to ensure our
        # CDC stem's output is what feeds the rest of the body.
        if hasattr(self.body, "conv_stem"):
            self.body.conv_stem = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run CDC stem first to produce the same shape the original stem
        # would have produced, then feed through the rest of the body.
        x = self.cdc_stem(x)
        feats = self.body(x)
        return feats[-1]  # final feature map (B, C, H, W)
