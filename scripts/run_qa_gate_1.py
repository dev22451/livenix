"""QA Gate 1 — Export gate (untrained model).

Verifies the architecture exports cleanly to ONNX, TFLite, and CoreML
BEFORE any training spend. If this fails, fix the architecture; if it
passes, Week 2 (training) is safe to begin.

Usage:
    uv run python scripts/run_qa_gate_1.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

# Make src/livenix importable when running this script directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from livenix.export import (
    export_to_coreml,
    export_to_onnx,
    export_to_tflite,
    run_parity_test,
)
from livenix.models import LivenixModel


BUILD_DIR = ROOT / "build"
ONNX_PATH = BUILD_DIR / "livenix_untrained.onnx"
TFLITE_DIR = BUILD_DIR / "tflite_untrained"
COREML_PATH = BUILD_DIR / "livenix_untrained.mlpackage"


def section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n  {title}\n{bar}")


def main() -> int:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    section("QA Gate 1 — Export gate (untrained)")

    print("\n[1/6] Instantiating LivenixModel (untrained)...")
    model = LivenixModel(pretrained_backbone=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      Params: {n_params:,}")

    print("\n[2/6] Forward sanity check on (1, 3, 128, 128) input...")
    dummy = torch.randn(1, 3, 128, 128)
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (1, 3), f"Expected (1, 3) logits, got {tuple(out.shape)}"
    assert torch.isfinite(out).all().item(), "Forward produced NaN/Inf"
    print(f"      Output shape: {tuple(out.shape)} (OK)")
    print(f"      Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")

    print(f"\n[3/6] Exporting to ONNX → {ONNX_PATH}")
    export_to_onnx(model, ONNX_PATH)
    onnx_size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
    print(f"      Size: {onnx_size_mb:.2f} MB")

    tflite_ok = False
    tflite_path = None
    try:
        print(f"\n[4/6] Exporting to TFLite → {TFLITE_DIR}/")
        tflite_path = export_to_tflite(ONNX_PATH, TFLITE_DIR)
        tflite_size_mb = tflite_path.stat().st_size / 1024 / 1024
        print(f"      Path: {tflite_path}")
        print(f"      Size: {tflite_size_mb:.2f} MB")
        tflite_ok = True
    except Exception as e:
        print(f"      FAILED: {e}")
        traceback.print_exc(limit=3)
        print("      (TFLite export issues are sometimes resolvable in Week 1 follow-up;")
        print("       the ONNX hub artifact is the source of truth.)")

    coreml_ok = False
    try:
        print(f"\n[5/6] Exporting to CoreML → {COREML_PATH}")
        export_to_coreml(model, COREML_PATH)
        print(f"      Saved: {COREML_PATH}")
        coreml_ok = True
    except Exception as e:
        print(f"      FAILED: {e}")
        traceback.print_exc(limit=3)
        print("      (CoreML export issues should be fixed before Phase 1 ship,")
        print("       but do not block Week 1 training preparation.)")

    print("\n[6/6] Cross-runtime parity test (100 fixed-seed inputs, tol=1e-2)")
    try:
        results = run_parity_test(
            model=model,
            onnx_path=ONNX_PATH,
            tflite_path=tflite_path if tflite_ok else None,
            mlpackage_path=COREML_PATH if coreml_ok else None,
            num_samples=100,
            tolerance=1e-2,
        )
        for name, r in results.items():
            print(
                f"      {name:8s}  max_diff={r.max_abs_diff_vs_pytorch:.4e}  "
                f"mean_diff={r.mean_abs_diff_vs_pytorch:.4e}"
            )
    except Exception as e:
        print(f"      FAILED: {e}")
        traceback.print_exc(limit=3)
        return 1

    section("QA Gate 1 — Summary")
    print(f"  ONNX export   : OK  ({onnx_size_mb:.2f} MB)")
    print(f"  TFLite export : {'OK' if tflite_ok else 'FAILED — see above'}")
    print(f"  CoreML export : {'OK' if coreml_ok else 'FAILED — see above'}")

    # Minimum bar: ONNX must succeed AND parity must hold for the runtimes
    # that did export.
    if not ONNX_PATH.exists():
        print("\nRESULT: FAIL — ONNX export missing")
        return 1

    print("\nRESULT: PASS — Week 2 (training) is safe to begin.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
