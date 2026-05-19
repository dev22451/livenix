"""Load a trained checkpoint and export it to ONNX (and optionally CoreML).

Usage:
    uv run python scripts/export_checkpoint.py \
        --checkpoint runs/v0.1-binary/checkpoint_best.pth \
        --output-dir runs/v0.1-binary/export \
        --num-classes 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from livenix.export.to_onnx import export_to_onnx
from livenix.models.full_model import LivenixModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", default="runs/export")
    p.add_argument("--num-classes", type=int, default=3)
    p.add_argument("--cdc-theta", type=float, default=0.7)
    p.add_argument("--input-size", type=int, default=128)
    p.add_argument("--skip-coreml", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = LivenixModel(
        cdc_theta=args.cdc_theta,
        num_classes=args.num_classes,
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.backbone.load_state_dict(ckpt["backbone"])
    model.head.load_state_dict(ckpt["main_head"])
    model.eval()
    print(f"[ok] loaded checkpoint with {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")

    # ONNX
    onnx_path = out / "livenix.onnx"
    export_to_onnx(
        model,
        onnx_path,
        input_size=(1, 3, args.input_size, args.input_size),
    )
    print(f"[ok] ONNX -> {onnx_path}  ({onnx_path.stat().st_size/1e6:.2f} MB)")

    # CoreML (optional, may not be available on non-macOS)
    if not args.skip_coreml:
        try:
            from livenix.export.to_coreml import export_to_coreml
            coreml_path = out / "livenix.mlpackage"
            export_to_coreml(
                model,
                coreml_path,
                input_size=(1, 3, args.input_size, args.input_size),
            )
            print(f"[ok] CoreML -> {coreml_path}")
        except Exception as e:
            print(f"[skip] CoreML failed: {e}")

    # Quick smoke test of the exported ONNX
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        import numpy as np
        dummy = np.random.randn(1, 3, args.input_size, args.input_size).astype(np.float32)
        out_logits = sess.run(None, {"input": dummy})[0]
        print(f"[ok] ONNX runtime inference works, output shape={out_logits.shape}")
    except Exception as e:
        print(f"[warn] onnxruntime check failed: {e}")


if __name__ == "__main__":
    main()
