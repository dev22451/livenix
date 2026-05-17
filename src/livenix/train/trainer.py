"""Livenix Trainer (plain PyTorch, single-GPU)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from livenix.models.backbone import LivenixBackbone
from livenix.models.heads import (
    FFTHead, MainHead, PatchContrastiveHead, SigLIPDistillHead,
)
from livenix.train.losses import (
    AsymmetricLoss, CombinedLoss, CosineMarginLoss, FocalLoss,
)


@dataclass
class TrainerConfig:
    # Model
    backbone_name: str = "mobilenetv4_conv_small.e2400_r224_in1k"
    pretrained_backbone: bool = False
    cdc_theta: float = 0.7
    dropout: float = 0.1
    input_size: tuple = (128, 128)
    num_classes: int = 3

    # Aux heads
    use_fft: bool = True
    use_siglip: bool = True
    use_patch_contrastive: bool = True
    siglip_teacher_dim: int = 1152

    # Loss
    main_loss: str = "focal"  # "focal" | "asymmetric" | "cosine_margin"
    focal_gamma: float = 2.0
    cosine_margin: float = 0.3
    cosine_scale: float = 30.0
    aux_weights: dict = field(default_factory=lambda: {"siglip": 0.5, "fft": 0.2, "patch": 0.3})

    # Optim
    optimizer: str = "adamw"   # only adamw supported in v0.1
    lr: float = 3.0e-4
    weight_decay: float = 0.05
    scheduler: str = "cosine"  # only cosine supported in v0.1

    # Training
    epochs: int = 30
    batch_size: int = 64
    num_workers: int = 4
    grad_clip_norm: float = 1.0

    # Checkpointing & logging
    output_dir: str = "runs/default"
    save_every_minutes: float = 30.0
    log_every_steps: int = 10

    # Device
    device: str = "auto"  # "auto" | "cpu" | "cuda"
    amp: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainerConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        # Flatten our YAML structure into kwargs.
        # The YAML uses nested sections (model:, training:, export:); we
        # pull the relevant keys.
        flat = {}
        if "model" in raw:
            m = raw["model"]
            for k in ("backbone_name", "pretrained_backbone", "cdc_theta",
                     "dropout", "input_size", "num_classes"):
                if k in m:
                    flat[k] = m[k]
        if "training" in raw:
            t = raw["training"]
            for k in ("epochs", "batch_size", "lr", "optimizer", "scheduler",
                     "weight_decay", "focal_gamma", "cosine_margin",
                     "aux_weights"):
                if k in t:
                    flat[k] = t[k]
        # Allow top-level overrides too
        for k in cls.__dataclass_fields__:
            if k in raw:
                flat[k] = raw[k]
        # Convert input_size from list -> tuple
        if "input_size" in flat and isinstance(flat["input_size"], list):
            flat["input_size"] = tuple(flat["input_size"])
        # Normalize aux_weights keys: YAML may use siglip_distill/fft_branch/patch_contrastive
        # CombinedLoss expects siglip/fft/patch
        if "aux_weights" in flat:
            raw_weights = flat["aux_weights"]
            normalized = {}
            _key_map = {
                "siglip_distill": "siglip",
                "fft_branch": "fft",
                "patch_contrastive": "patch",
            }
            for k, v in raw_weights.items():
                normalized_key = _key_map.get(k, k)
                normalized[normalized_key] = v
            flat["aux_weights"] = normalized
        return cls(**flat)


class LivenixTrainer:
    """End-to-end training orchestrator for Livenix v0.1 models.

    Wires:
      LivenixBackbone -> { MainHead, FFTHead (opt), SigLIPDistillHead (opt),
                           PatchContrastiveHead (opt) }
      CombinedLoss(main_loss_fn, weights) with aux losses provided per batch

    Usage:
        cfg = TrainerConfig.from_yaml("configs/mobile_small.yaml")
        trainer = LivenixTrainer(cfg, train_dataset, val_dataset,
                                 siglip_teacher_fn=mock_or_real_teacher)
        trainer.fit()
    """

    def __init__(
        self,
        config: TrainerConfig,
        train_dataset: Dataset,
        val_dataset: Dataset | None = None,
        siglip_teacher_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.cfg = config
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.siglip_teacher_fn = siglip_teacher_fn

        # Device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        # Build model
        self.backbone = LivenixBackbone(
            backbone_name=config.backbone_name,
            pretrained=config.pretrained_backbone,
            cdc_theta=config.cdc_theta,
        ).to(self.device)

        feat_dim = self.backbone.out_channels

        if config.main_loss == "cosine_margin":
            self.main_head = None  # CosineMarginLoss replaces it
            self.main_loss_fn = CosineMarginLoss(
                in_features=feat_dim,
                num_classes=config.num_classes,
                margin=config.cosine_margin,
                scale=config.cosine_scale,
            ).to(self.device)
            self.main_uses_features = True
        elif config.main_loss == "asymmetric":
            self.main_head = MainHead(feat_dim, dropout=config.dropout).to(self.device)
            self.main_loss_fn = AsymmetricLoss().to(self.device)
            self.main_uses_features = False
        else:  # focal (default)
            self.main_head = MainHead(feat_dim, dropout=config.dropout).to(self.device)
            self.main_loss_fn = FocalLoss(gamma=config.focal_gamma).to(self.device)
            self.main_uses_features = False

        # Aux heads (instantiated based on flags)
        self.fft_head = (
            FFTHead(in_channels=feat_dim).to(self.device) if config.use_fft else None
        )
        self.siglip_head = (
            SigLIPDistillHead(
                in_channels=feat_dim,
                teacher_dim=config.siglip_teacher_dim,
            ).to(self.device)
            if (config.use_siglip and siglip_teacher_fn is not None) else None
        )
        self.patch_head = (
            PatchContrastiveHead(in_channels=feat_dim).to(self.device)
            if config.use_patch_contrastive else None
        )

        # Combined loss orchestrator
        self.combined_loss = CombinedLoss(
            main_loss_fn=self.main_loss_fn,
            weights=config.aux_weights,
        )

        # Optimizer
        params = list(self.backbone.parameters())
        if self.main_head is not None:
            params += list(self.main_head.parameters())
        if isinstance(self.main_loss_fn, CosineMarginLoss):
            params += list(self.main_loss_fn.parameters())
        if self.fft_head:
            params += list(self.fft_head.parameters())
        if self.siglip_head:
            params += list(self.siglip_head.parameters())
        if self.patch_head:
            params += list(self.patch_head.parameters())
        self.optimizer = AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs)

        # Mixed precision
        self.scaler = torch.amp.GradScaler(enabled=(config.amp and self.device.type == "cuda"))

        # Logging & checkpoints
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.output_dir / "tb"))
        self._last_save_time = time.time()
        self.best_val_loss = float("inf")
        self.global_step = 0

    def _forward_batch(self, images: torch.Tensor, labels: torch.Tensor) -> dict:
        """Run one forward pass, return loss components."""
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        # Backbone produces (B, C, H, W)
        feat_map = self.backbone(images)
        # Pool for main head
        pooled = feat_map.mean(dim=(2, 3))

        # Main head input
        if self.main_uses_features:
            main_inputs = {"features": pooled, "targets": labels}
        else:
            logits = self.main_head(pooled)
            main_inputs = {"logits": logits, "targets": labels}

        # Aux losses
        aux_losses = {}
        if self.fft_head is not None:
            fft_pred = self.fft_head(feat_map)
            fft_target = FFTHead.compute_target(images, target_size=fft_pred.shape[-2:])
            aux_losses["fft"] = self.fft_head.loss(fft_pred, fft_target)
        if self.siglip_head is not None and self.siglip_teacher_fn is not None:
            student_emb = self.siglip_head(feat_map)
            with torch.no_grad():
                teacher_emb = self.siglip_teacher_fn(images)
            aux_losses["siglip"] = self.siglip_head.loss(student_emb, teacher_emb)
        if self.patch_head is not None:
            patch_emb = self.patch_head(feat_map)
            num_patches = feat_map.shape[-1] * feat_map.shape[-2]
            aux_losses["patch"] = self.patch_head.loss(patch_emb, labels, num_patches)

        # Combine
        out = self.combined_loss(main_inputs, aux_losses)
        return out  # dict with "main", "fft", "siglip", "patch", "total"

    def train_step(self, batch) -> dict:
        # batch shape conventions: (images, labels) or (images, labels, attack_types)
        images, labels = batch[0], batch[1]

        self.optimizer.zero_grad()
        with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
            loss_dict = self._forward_batch(images, labels)
            total = loss_dict["total"]

        if self.scaler.is_enabled():
            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self._all_params(), self.cfg.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(self._all_params(), self.cfg.grad_clip_norm)
            self.optimizer.step()

        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    @torch.no_grad()
    def val_step(self, batch) -> dict:
        images, labels = batch[0], batch[1]
        loss_dict = self._forward_batch(images, labels)
        return {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}

    @torch.no_grad()
    def eval_classification_accuracy(
        self, loader: DataLoader, *, use_train_mode: bool = False
    ) -> float:
        """Mean accuracy of argmax main-head predictions vs labels.

        Supported only when ``main_head`` is used (focal / asymmetric).
        Cosine-margin mode has no separate logits head — do not use here.

        Args:
            use_train_mode: If True, run the backbone/head in train mode so
                BatchNorm uses batch statistics (appropriate for small-set
                overfit checks where eval-mode running stats are misleading).
        """
        if self.main_head is None:
            raise ValueError(
                "eval_classification_accuracy requires MainHead (focal/asymmetric); "
                "cosine_margin mode is not supported."
            )
        if use_train_mode:
            self._set_train_mode()
        else:
            self._set_eval_mode()
        correct = 0
        total = 0
        for batch in loader:
            images, labels = batch[0], batch[1]
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            feat_map = self.backbone(images)
            pooled = feat_map.mean(dim=(2, 3))
            logits = self.main_head(pooled)
            pred = logits.argmax(dim=-1)
            correct += (pred == labels).sum().item()
            total += int(labels.numel())
        return correct / max(total, 1)

    def fit(self):
        """Run training for cfg.epochs epochs."""
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=_default_collate,
            drop_last=True,
        )
        val_loader = None
        if self.val_ds is not None and len(self.val_ds) > 0:
            val_loader = DataLoader(
                self.val_ds,
                batch_size=self.cfg.batch_size,
                shuffle=False,
                num_workers=self.cfg.num_workers,
                collate_fn=_default_collate,
            )

        for epoch in range(self.cfg.epochs):
            self._set_train_mode()
            for batch in train_loader:
                metrics = self.train_step(batch)
                self.global_step += 1
                if self.global_step % self.cfg.log_every_steps == 0:
                    self._log_metrics(metrics, prefix="train")
                # time-based checkpoint
                if (time.time() - self._last_save_time) >= (self.cfg.save_every_minutes * 60):
                    self._save_checkpoint(f"checkpoint_step_{self.global_step}.pth")
                    self._last_save_time = time.time()

            if val_loader is not None:
                self._set_eval_mode()
                val_losses = []
                for batch in val_loader:
                    val_losses.append(self.val_step(batch))
                if val_losses:
                    avg = {
                        k: sum(d[k] for d in val_losses) / len(val_losses)
                        for k in val_losses[0]
                    }
                    self._log_metrics(avg, prefix="val", step=self.global_step)
                    if avg["total"] < self.best_val_loss:
                        self.best_val_loss = avg["total"]
                        self._save_checkpoint("checkpoint_best.pth")

            self.scheduler.step()

        self._save_checkpoint("checkpoint_final.pth")
        self.writer.close()

    # --- helpers ---

    def _all_params(self):
        for m in [self.backbone, self.main_head, self.main_loss_fn,
                  self.fft_head, self.siglip_head, self.patch_head]:
            if m is None:
                continue
            yield from m.parameters()

    def _set_train_mode(self):
        self.backbone.train()
        for m in (self.main_head, self.fft_head, self.siglip_head, self.patch_head):
            if m is not None:
                m.train()

    def _set_eval_mode(self):
        self.backbone.eval()
        for m in (self.main_head, self.fft_head, self.siglip_head, self.patch_head):
            if m is not None:
                m.eval()

    def _log_metrics(self, metrics: dict, prefix: str, step: int | None = None):
        s = step if step is not None else self.global_step
        for k, v in metrics.items():
            self.writer.add_scalar(f"{prefix}/{k}", v, s)

    def _save_checkpoint(self, name: str):
        path = self.output_dir / name
        state = {
            "backbone": self.backbone.state_dict(),
            "config": self.cfg.__dict__,
            "global_step": self.global_step,
        }
        if self.main_head is not None:
            state["main_head"] = self.main_head.state_dict()
        if isinstance(self.main_loss_fn, CosineMarginLoss):
            state["main_loss_fn"] = self.main_loss_fn.state_dict()
        torch.save(state, path)


def _default_collate(samples):
    """Collate handling 3-tuple (img, label, attack_type) or 4-tuple."""
    # Each sample is (img_tensor, label, attack_type_str_or_int, [optional attrs])
    images = torch.stack([s[0] for s in samples])
    labels = torch.tensor([s[1] for s in samples], dtype=torch.long)
    return images, labels
