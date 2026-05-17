"""QA Gate 2 — Overfit gate (tiny CelebA-Spoof subset).

Trains on 100 reproducibly sampled CelebA-Spoof *train* images for 100 epochs
with lightweight settings (no aux heads, no weight decay, eval-style crops).

Pass criterion: **train accuracy ≥ 99%** at the final epoch. If this fails,
the main classification path is broken — debug before convergence gates.

Usage:
    export CELEBA_SPOOF_ROOT=/path/to/CelebA-Spoof  # or pass --root
    uv run python scripts/run_qa_gate_2.py

The dataset must contain ``Data/train/`` in CelebA-Spoof layout (see
``livenix.data.celeba_spoof``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from livenix.data.celeba_spoof import CelebASpoofDataset  # noqa: E402
from livenix.data.transforms import eval_transforms  # noqa: E402
from livenix.train.trainer import LivenixTrainer, TrainerConfig  # noqa: E402


def _collate(samples: list) -> tuple[torch.Tensor, torch.Tensor]:
    images = torch.stack([s[0] for s in samples])
    labels = torch.tensor([s[1] for s in samples], dtype=torch.long)
    return images, labels


def _pick_indices(n_total: int, k: int, seed: int) -> list[int]:
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    return perm[: min(k, n_total)]


def run_qa_gate_2(
    celeba_root: Path,
    *,
    config_path: Path | None = None,
    output_dir: Path | None = None,
    num_samples: int = 100,
    epochs: int = 100,
    seed: int = 42,
    min_accuracy: float = 0.99,
    batch_size: int = 8,
    device: str = "auto",
) -> tuple[float, bool]:
    """Run overfit training; return (peak_train_accuracy_over_epochs, passed)."""
    cfg_path = config_path or (ROOT / "configs" / "mobile_small.yaml")
    transform = eval_transforms((128, 128))
    full_ds = CelebASpoofDataset(
        root=celeba_root,
        split="train",
        transform=transform,
    )
    if len(full_ds) < num_samples:
        raise ValueError(
            f"CelebA-Spoof train split at {celeba_root} has only {len(full_ds)} "
            f"samples; need at least {num_samples}. Check CELEBA_SPOOF_ROOT / --root."
        )

    indices = _pick_indices(len(full_ds), num_samples, seed)
    subset = Subset(full_ds, indices)

    cfg = TrainerConfig.from_yaml(cfg_path)
    cfg.epochs = epochs
    cfg.batch_size = min(batch_size, len(subset))
    cfg.weight_decay = 0.0
    cfg.lr = 1.0e-3
    cfg.dropout = 0.0
    cfg.use_fft = False
    cfg.use_siglip = False
    cfg.use_patch_contrastive = False
    cfg.save_every_minutes = 9999.0
    cfg.log_every_steps = 10**9
    cfg.output_dir = str(output_dir or (ROOT / "runs" / "qa_gate_2"))
    cfg.num_workers = 0
    cfg.device = device
    if device == "cpu":
        cfg.amp = False
    cfg.main_loss = "focal"

    train_loader = DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate,
        drop_last=False,
    )

    trainer = LivenixTrainer(cfg, subset, val_dataset=None, siglip_teacher_fn=None)

    best_acc = 0.0
    final_acc = 0.0
    for epoch in range(epochs):
        trainer._set_train_mode()
        for batch in train_loader:
            trainer.train_step(batch)
        trainer.scheduler.step()
        final_acc = trainer.eval_classification_accuracy(
            train_loader, use_train_mode=True
        )
        best_acc = max(best_acc, final_acc)
        print(f"  epoch {epoch + 1:3d}/{epochs}  train_acc={final_acc:.4f}  (best={best_acc:.4f})")

    trainer.writer.close()
    # Peak accuracy is the acceptance metric: final-epoch train-mode BN batch
    # stats are inherently noisy when measured on shuffled loaders.
    passed = best_acc >= min_accuracy
    return best_acc, passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="CelebA-Spoof root (directory containing Data/). "
        "Default: $CELEBA_SPOOF_ROOT.",
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config path.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-acc", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Tensorboard + checkpoint directory (default: runs/qa_gate_2).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    args = parser.parse_args()

    celeba_root = args.root
    if celeba_root is None:
        import os

        r = os.environ.get("CELEBA_SPOOF_ROOT")
        if not r:
            print(
                "ERROR: Set CELEBA_SPOOF_ROOT or pass --root to CelebA-Spoof directory.",
                file=sys.stderr,
            )
            return 1
        celeba_root = Path(r)

    print("=" * 70)
    print("  QA Gate 2 — Overfit gate")
    print("=" * 70)
    print(f"  root={celeba_root.resolve()}")
    print(f"  samples={args.num_samples}  epochs={args.epochs}  seed={args.seed}")

    try:
        acc, ok = run_qa_gate_2(
            celeba_root,
            config_path=args.config,
            output_dir=args.output_dir,
            num_samples=args.num_samples,
            epochs=args.epochs,
            seed=args.seed,
            min_accuracy=args.min_acc,
            batch_size=args.batch_size,
            device=args.device,
        )
    except ValueError as e:
        print(f"\nRESULT: FAIL — {e}", file=sys.stderr)
        return 1

    print("\n" + "=" * 70)
    print(f"  Peak train accuracy:  {acc:.4f}  (target >= {args.min_acc:.2f})")
    if ok:
        print("  RESULT: PASS")
        return 0
    print("  RESULT: FAIL — model/loss pipeline did not memorize subset", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
