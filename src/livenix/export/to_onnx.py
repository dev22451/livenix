"""PyTorch → ONNX export."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_size: tuple[int, int, int, int] = (1, 3, 128, 128),
    opset_version: int = 17,
) -> Path:
    """Export a Livenix model to ONNX.

    Args:
        model: nn.Module producing (B, 3) logits.
        output_path: destination .onnx file path.
        input_size: NCHW input shape for tracing (B is dynamic).
        opset_version: ONNX opset (17 covers MobileNetV4 cleanly).

    Returns:
        Path to the exported ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy = torch.randn(*input_size)

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        do_constant_folding=True,
    )
    return output_path
