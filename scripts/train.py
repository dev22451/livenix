"""Livenix Phase 1 — full training run (T2.14).

Loads CelebA-Spoof + WMCA + HiFiMask + Deepfake-on-screen, wires the
real SigLIP-So400m teacher, runs LivenixTrainer.fit().

Usage (typical RunPod 4090):

    python scripts/train.py \\
        --config configs/mobile_small.yaml \\
        --celeba-root /workspace/data/CelebA-Spoof \\
        --wmca-root /workspace/data/WMCA \\
        --hifimask-root /workspace/data/HiFiMask \\
        --deepfake-root /workspace/data/deepfake_screen \\
        --output-dir runs/v0.1-baseline \\
        --device cuda

Datasets can be omitted; the script will skip whichever paths are missing
and print a warning. Roots can also be supplied via env vars:
    CELEBA_SPOOF_ROOT, WMCA_ROOT, HIFIMASK_ROOT, DEEPFAKE_SCREEN_ROOT.

Use --no-siglip to skip the SigLIP teacher (~1.6 GB HF download) for a
quick smoke run.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, Dataset


def _build_dataset(name: str, klass, root: Path | None, split: str, transform) -> Dataset | None:
    """Instantiate a dataset only if its root exists; else warn and return None."""
    if root is None:
        print(f"[skip] {name}: no root provided")
        return None
    if not Path(root).exists():
        print(f"[skip] {name}: root does not exist: {root}")
        return None
    ds = klass(root=root, split=split, transform=transform) if split else klass(root=root, transform=transform)
    if len(ds) == 0:
        print(f"[skip] {name}: 0 samples discovered at {root}")
        return None
    print(f"[ok]   {name:<10} split={split or 'all':<6} samples={len(ds)}")
    return ds


def _build_siglip_teacher_fn(device: torch.device):
    """Construct a callable that returns SigLIP image embeddings.

    Lazy-imports `transformers` to avoid requiring it when --no-siglip
    is set. Downloads google/siglip-so400m-patch14-384 on first call
    (~1.6 GB; cached at $HF_HOME).
    """
    try:
        from transformers import AutoModel  # noqa: F401  (verify present)
    except ImportError as e:
        raise RuntimeError(
            "transformers is required for SigLIP teacher. Install with "
            "`uv sync --extra train` or pass --no-siglip."
        ) from e

    import torch.nn.functional as F
    from transformers import AutoModel

    teacher = AutoModel.from_pretrained(
        "google/siglip-so400m-patch14-384",
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()

    # SigLIP teacher expects 384x384; our model input is 128x128.
    # Resize on-the-fly via bilinear interp, no de-normalize step
    # because the teacher's image normalization is close to ImageNet.
    @torch.no_grad()
    def siglip_teacher_fn(images: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(images, size=384, mode="bilinear", align_corners=False)
        out = teacher.get_image_features(pixel_values=x.to(teacher.dtype))
        return out.float()

    print(f"[ok]   SigLIP teacher loaded ({sum(p.numel() for p in teacher.parameters())/1e6:.0f}M params)")
    return siglip_teacher_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/mobile_small.yaml", help="TrainerConfig YAML")
    parser.add_argument("--celeba-root", default=os.environ.get("CELEBA_SPOOF_ROOT"))
    parser.add_argument("--wmca-root", default=os.environ.get("WMCA_ROOT"))
    parser.add_argument("--hifimask-root", default=os.environ.get("HIFIMASK_ROOT"))
    parser.add_argument("--deepfake-root", default=os.environ.get("DEEPFAKE_SCREEN_ROOT"))
    parser.add_argument("--output-dir", default="runs/v0.1-baseline")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-siglip", action="store_true", help="Skip SigLIP teacher (faster smoke runs)")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Cap train samples (debug)")
    args = parser.parse_args()

    # Lazy imports so --help is fast and missing deps don't crash CLI parse
    from livenix.data import (
        CelebASpoofDataset, CelebASpoofCropDataset,
        WMCADataset, HiFiMaskDataset, DeepfakeScreenDataset,
        train_transforms, eval_transforms,
    )
    from livenix.train.trainer import LivenixTrainer, TrainerConfig

    # ---- Config ----
    cfg = TrainerConfig.from_yaml(args.config)
    cfg.output_dir = args.output_dir
    cfg.device = args.device
    if args.no_siglip:
        cfg.use_siglip = False

    # Resolve device early so SigLIP teacher lands on the right one
    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)
    print(f"[cfg]  device={device}  output_dir={cfg.output_dir}  use_siglip={cfg.use_siglip}")

    # ---- Transforms ----
    image_size = tuple(cfg.input_size)
    train_tf = train_transforms(image_size=image_size, augmentation_strength="medium")
    eval_tf = eval_transforms(image_size=image_size)

    # ---- Datasets ----
    print("\n[datasets — train split]")
    train_parts: list[Dataset] = []
    # Try full CelebA-Spoof first; fall back to pre-cropped variant
    celeba_ds = _build_dataset("CelebA", CelebASpoofDataset, args.celeba_root, "train", train_tf)
    if celeba_ds is None and args.celeba_root:
        celeba_ds = _build_dataset("CelebA-Crop", CelebASpoofCropDataset, args.celeba_root, "test", train_tf)
    if celeba_ds is not None:
        train_parts.append(celeba_ds)
    for name, klass, root, split in [
        ("WMCA",     WMCADataset,           args.wmca_root,     "train"),
        ("HiFiMask", HiFiMaskDataset,       args.hifimask_root, "train"),
    ]:
        ds = _build_dataset(name, klass, root, split, train_tf)
        if ds is not None:
            train_parts.append(ds)
    # Deepfake stub has no split arg — it returns everything
    if args.deepfake_root:
        df = _build_dataset("Deepfake", DeepfakeScreenDataset, args.deepfake_root, "", train_tf)
        if df is not None:
            train_parts.append(df)

    if not train_parts:
        sys.exit("ERROR: No training datasets available. Provide at least one --*-root.")
    train_ds: Dataset = ConcatDataset(train_parts)
    if args.max_train_samples is not None:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(args.max_train_samples, len(train_ds))))
    print(f"[total] train samples: {len(train_ds)}")

    print("\n[datasets — val split]")
    val_parts: list[Dataset] = []
    celeba_val = _build_dataset("CelebA", CelebASpoofDataset, args.celeba_root, "test", eval_tf)
    if celeba_val is None and args.celeba_root:
        celeba_val = _build_dataset("CelebA-Crop", CelebASpoofCropDataset, args.celeba_root, "test", eval_tf)
    if celeba_val is not None:
        val_parts.append(celeba_val)
    for name, klass, root, split in [
        ("WMCA",     WMCADataset,           args.wmca_root,     "dev"),
        ("HiFiMask", HiFiMaskDataset,       args.hifimask_root, "test"),
    ]:
        ds = _build_dataset(name, klass, root, split, eval_tf)
        if ds is not None:
            val_parts.append(ds)
    val_ds: Dataset | None = ConcatDataset(val_parts) if val_parts else None
    print(f"[total] val samples: {len(val_ds) if val_ds else 0}")

    # ---- SigLIP teacher ----
    siglip_teacher_fn = None
    if cfg.use_siglip:
        print("\n[siglip] loading teacher (this may take a few minutes on first run)…")
        siglip_teacher_fn = _build_siglip_teacher_fn(device)
    else:
        print("\n[siglip] disabled via --no-siglip")

    # ---- Trainer ----
    print("\n[trainer] building…")
    trainer = LivenixTrainer(
        config=cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        siglip_teacher_fn=siglip_teacher_fn,
    )
    print(f"[trainer] device={trainer.device}  "
          f"main_loss={cfg.main_loss}  use_fft={cfg.use_fft}  "
          f"use_patch={cfg.use_patch_contrastive}")

    print("\n[fit] starting training…\n" + "=" * 60)
    trainer.fit()
    print("=" * 60 + "\n[fit] done.")


if __name__ == "__main__":
    main()
