"""ONNX → TFLite export via onnx2tf."""

from __future__ import annotations

from pathlib import Path


def export_to_tflite(
    onnx_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Convert ONNX → TFLite (FP32).

    Args:
        onnx_path: source .onnx file.
        output_dir: directory where onnx2tf writes its outputs.

    Returns:
        Path to the produced .tflite file.
    """
    import onnx2tf

    onnx_path = Path(onnx_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(output_dir),
        output_signaturedefs=True,
        non_verbose=True,
    )

    # onnx2tf writes <name>_float32.tflite inside the output_dir.
    candidates = sorted(output_dir.glob("*_float32.tflite"))
    if not candidates:
        # Fallback to any .tflite produced
        candidates = sorted(output_dir.glob("*.tflite"))
    if not candidates:
        raise RuntimeError(f"onnx2tf produced no .tflite in {output_dir}")

    # Pick the most-recent file (in case run leaves multiple)
    tflite_path = max(candidates, key=lambda p: p.stat().st_mtime)
    return tflite_path
