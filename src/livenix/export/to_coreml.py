"""PyTorch → CoreML export via coremltools 8.x.

coremltools 8.x recommends the PyTorch tracing path (not direct ONNX).
We take the original PyTorch model so the conversion has full op info.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def export_to_coreml(
    model: nn.Module,
    output_path: str | Path,
    input_size: tuple[int, int, int, int] = (1, 3, 128, 128),
    minimum_deployment_target: str = "iOS16",
) -> Path:
    """Convert a PyTorch model to a CoreML ML Program (.mlpackage).

    Args:
        model: nn.Module producing (B, 3) logits.
        output_path: destination .mlpackage directory.
        input_size: NCHW input shape used for tracing.
        minimum_deployment_target: e.g. "iOS16" or "iOS17".

    Returns:
        Path to the produced .mlpackage.
    """
    import coremltools as ct

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy = torch.randn(*input_size)
    traced = torch.jit.trace(model, dummy, strict=False)

    target_map = {
        "iOS15": ct.target.iOS15,
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
    }
    target = target_map.get(minimum_deployment_target, ct.target.iOS16)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=dummy.shape)],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        minimum_deployment_target=target,
        compute_units=ct.ComputeUnit.ALL,
        compute_precision=ct.precision.FLOAT32,
    )

    mlmodel.save(str(output_path))
    return output_path
