"""Cross-runtime numerical parity test (QA Gate 11, preview in Gate 1).

Generates N fixed-seed random inputs, runs them through PyTorch, ONNX
(onnxruntime), TFLite (tf.lite.Interpreter), and CoreML (coremltools), then
asserts pairwise max absolute difference is within tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ParityResult:
    name: str
    max_abs_diff_vs_pytorch: float
    mean_abs_diff_vs_pytorch: float


def _run_pytorch(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t = torch.from_numpy(x)
        return model(t).numpy()


def _run_onnx(onnx_path: Path, x: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out = sess.run(None, {inp_name: x.astype(np.float32)})
    return out[0]


def _run_tflite(tflite_path: Path, x: np.ndarray) -> np.ndarray:
    try:
        import tensorflow as tf  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "tensorflow not installed — required for TFLite parity check"
        ) from e

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    outs = []
    for i in range(x.shape[0]):
        sample = x[i : i + 1].astype(np.float32)
        # onnx2tf may convert to NHWC — handle either layout
        expected_shape = input_details[0]["shape"]
        if len(expected_shape) == 4 and expected_shape[-1] == sample.shape[1]:
            # NHWC expected
            sample = np.transpose(sample, (0, 2, 3, 1))
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        outs.append(interpreter.get_tensor(output_details[0]["index"]))
    return np.concatenate(outs, axis=0)


def _run_coreml(mlpackage_path: Path, x: np.ndarray) -> np.ndarray:
    import coremltools as ct

    mlmodel = ct.models.MLModel(str(mlpackage_path))
    inp_name = list(mlmodel.input_description._fd_spec)[0]

    outs = []
    for i in range(x.shape[0]):
        sample = x[i : i + 1].astype(np.float32)
        result = mlmodel.predict({inp_name: sample})
        out_value = list(result.values())[0]
        outs.append(np.asarray(out_value).reshape(1, -1))
    return np.concatenate(outs, axis=0)


def run_parity_test(
    model: nn.Module,
    onnx_path: str | Path | None = None,
    tflite_path: str | Path | None = None,
    mlpackage_path: str | Path | None = None,
    num_samples: int = 100,
    input_shape: tuple[int, int, int] = (3, 128, 128),
    tolerance: float = 1e-2,
    seed: int = 1337,
) -> dict[str, ParityResult]:
    """Compare PyTorch outputs against each available export.

    Returns a dict of runtime name → ParityResult. Raises AssertionError
    if any runtime exceeds the tolerance.
    """
    rng = np.random.RandomState(seed)
    x = rng.randn(num_samples, *input_shape).astype(np.float32)

    pytorch_out = _run_pytorch(model, x)

    results: dict[str, ParityResult] = {}

    def _compare(name: str, other: np.ndarray) -> ParityResult:
        diff = np.abs(pytorch_out - other.reshape(pytorch_out.shape))
        return ParityResult(
            name=name,
            max_abs_diff_vs_pytorch=float(diff.max()),
            mean_abs_diff_vs_pytorch=float(diff.mean()),
        )

    if onnx_path is not None:
        results["onnx"] = _compare("onnx", _run_onnx(Path(onnx_path), x))

    if tflite_path is not None:
        results["tflite"] = _compare("tflite", _run_tflite(Path(tflite_path), x))

    if mlpackage_path is not None:
        results["coreml"] = _compare("coreml", _run_coreml(Path(mlpackage_path), x))

    # Enforce tolerance
    failed = [
        r for r in results.values() if r.max_abs_diff_vs_pytorch > tolerance
    ]
    if failed:
        details = "\n".join(
            f"  {r.name}: max={r.max_abs_diff_vs_pytorch:.4e} mean={r.mean_abs_diff_vs_pytorch:.4e}"
            for r in failed
        )
        raise AssertionError(
            f"Parity test failed (tolerance={tolerance:.0e}):\n{details}"
        )

    return results
