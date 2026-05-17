"""QA Gate 1 — pytest version of the export gate.

Lightweight smoke tests so CI / pytest can verify scaffold integrity.
Heavy ONNX → TFLite/CoreML conversion is exercised by
scripts/run_qa_gate_1.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from livenix.export import export_to_onnx
from livenix.models import LivenixModel


@pytest.fixture(scope="module")
def model() -> LivenixModel:
    return LivenixModel(pretrained_backbone=False)


def test_model_instantiates(model: LivenixModel) -> None:
    assert model is not None


def test_forward_shape(model: LivenixModel) -> None:
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, 3)


def test_forward_no_nan(model: LivenixModel) -> None:
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 128, 128))
    assert torch.isfinite(out).all().item()


def test_onnx_export(tmp_path: Path, model: LivenixModel) -> None:
    onnx_path = tmp_path / "livenix.onnx"
    export_to_onnx(model, onnx_path)
    assert onnx_path.exists()
    assert onnx_path.stat().st_size > 1024  # >1KB sanity


def test_banned_ops_absent_in_onnx(tmp_path: Path, model: LivenixModel) -> None:
    """Ensure the ONNX graph does not contain banned ops."""
    import onnx

    onnx_path = tmp_path / "livenix.onnx"
    export_to_onnx(model, onnx_path)
    graph = onnx.load(str(onnx_path)).graph

    banned = {"FFT", "STFT", "GridSample", "GroupNormalization", "If", "Loop", "Scan"}
    found_ops = {n.op_type for n in graph.node}
    overlap = found_ops & banned
    assert not overlap, f"Banned ops in ONNX graph: {overlap}"


def test_class_count(model: LivenixModel) -> None:
    """Confirm 3-class output (real / print_spoof / replay_spoof)."""
    from livenix.models.full_model import CLASS_NAMES

    assert CLASS_NAMES == ["real", "print_spoof", "replay_spoof"]
