"""QA Gate 2 script tests — synthetic CelebA-Spoof tree only."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from PIL import Image


def _load_run_qa_gate_2():
    repo = Path(__file__).resolve().parent.parent
    path = repo / "scripts" / "run_qa_gate_2.py"
    spec = importlib.util.spec_from_file_location("run_qa_gate_2", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_varied_celeba_train(root: Path, n: int = 120) -> None:
    """Minimal CelebA-Spoof layout with **distinct** RGB images per index."""
    assert n % 3 == 0
    per_class = n // 3
    # spoof_type -> livenix label: 0->0, 1->1, 7->2
    plan = [(0, 0)] * per_class + [(1, 1)] * per_class + [(7, 2)] * per_class
    for i, (spoof_type, _) in enumerate(plan):
        subject_id = f"s{i:04d}"
        subdir = "live" if spoof_type == 0 else "spoof"
        img_dir = root / "Data" / "train" / subject_id / subdir
        img_dir.mkdir(parents=True, exist_ok=True)
        stem = f"f{i:04d}"
        img_path = img_dir / f"{stem}.png"
        sidecar = img_dir / f"{stem}.txt"
        rgb = (i % 256, (i * 5) % 256, (i * 11) % 256)
        Image.new("RGB", (128, 128), rgb).save(img_path)
        sidecar.write_text("\n".join(["0"] * 40 + [str(spoof_type)]) + "\n")


def test_qa_gate_2_run_module_overfits_synthetic(tmp_path: Path):
    mod = _load_run_qa_gate_2()
    _write_varied_celeba_train(tmp_path, n=120)
    out = tmp_path / "gate2_out"
    acc, ok = mod.run_qa_gate_2(
        tmp_path,
        config_path=Path(__file__).resolve().parent.parent / "configs" / "mobile_small.yaml",
        output_dir=out,
        num_samples=100,
        epochs=45,
        seed=123,
        min_accuracy=0.99,
        batch_size=10,
        device="cpu",
    )
    assert ok, f"expected ≥99% train acc, got {acc:.4f}"


def test_qa_gate_2_errors_when_too_few_samples(tmp_path: Path):
    mod = _load_run_qa_gate_2()
    _write_varied_celeba_train(tmp_path, n=30)
    with pytest.raises(ValueError, match="only 30"):
        mod.run_qa_gate_2(
            tmp_path,
            output_dir=tmp_path / "o",
            num_samples=100,
            epochs=1,
            device="cpu",
        )
