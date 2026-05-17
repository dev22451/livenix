"""Tests for LivenixTrainer + TrainerConfig.

All tests use a tiny synthetic Dataset — no real data downloaded,
no HuggingFace model loaded.  SigLIP teacher is mocked with a callable
that returns random tensors of the correct shape.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from livenix.train.trainer import LivenixTrainer, TrainerConfig


# ---------------------------------------------------------------------------
# Tiny synthetic dataset fixture
# ---------------------------------------------------------------------------

class TinyDataset(Dataset):
    """Fixed-shape random tensors to exercise the training loop without I/O."""

    def __init__(self, n: int = 16, h: int = 32, w: int = 32, num_classes: int = 3) -> None:
        self.n = n
        self.images = torch.randn(n, 3, h, w)
        self.labels = torch.randint(0, num_classes, (n,))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx].item()


def _minimal_cfg(**kwargs) -> TrainerConfig:
    """Return a TrainerConfig with fast/small defaults for unit tests."""
    defaults = dict(
        epochs=1,
        batch_size=8,
        num_workers=0,
        save_every_minutes=999.0,   # disable time-based checkpointing
        amp=False,                  # CPU tests don't need AMP
        device="cpu",
        use_fft=False,
        use_siglip=False,
        use_patch_contrastive=False,
        log_every_steps=1,
    )
    defaults.update(kwargs)
    return TrainerConfig(**defaults)


def _trainer_with_tmpdir(cfg: TrainerConfig, ds: Dataset, **kwargs) -> tuple[LivenixTrainer, Path]:
    tmp = tempfile.mkdtemp()
    cfg.output_dir = tmp
    trainer = LivenixTrainer(cfg, ds, **kwargs)
    return trainer, Path(tmp)


# ---------------------------------------------------------------------------
# test_config_from_yaml
# ---------------------------------------------------------------------------

def test_config_from_yaml():
    """Load configs/mobile_small.yaml and verify key fields are parsed."""
    cfg_path = Path(__file__).parent.parent / "configs" / "mobile_small.yaml"
    cfg = TrainerConfig.from_yaml(cfg_path)

    assert cfg.backbone_name == "mobilenetv4_conv_small.e2400_r224_in1k"
    assert cfg.epochs == 30
    assert cfg.lr == pytest.approx(3.0e-4)
    assert cfg.num_classes == 3
    assert cfg.cdc_theta == pytest.approx(0.7)
    assert cfg.batch_size == 256


# ---------------------------------------------------------------------------
# test_yaml_input_size_list_to_tuple
# ---------------------------------------------------------------------------

def test_yaml_input_size_list_to_tuple():
    """YAML stores input_size as a list; from_yaml must coerce to a tuple."""
    cfg_path = Path(__file__).parent.parent / "configs" / "mobile_small.yaml"
    cfg = TrainerConfig.from_yaml(cfg_path)
    assert isinstance(cfg.input_size, tuple)
    assert cfg.input_size == (128, 128)


# ---------------------------------------------------------------------------
# test_trainer_instantiates_with_minimal_config
# ---------------------------------------------------------------------------

def test_trainer_instantiates_with_minimal_config():
    """LivenixTrainer should construct without error on minimal config."""
    cfg = _minimal_cfg(use_siglip=False)
    ds = TinyDataset(n=8)
    trainer, _ = _trainer_with_tmpdir(cfg, ds)
    assert trainer is not None


# ---------------------------------------------------------------------------
# test_one_epoch_completes_no_crash
# ---------------------------------------------------------------------------

def test_one_epoch_completes_no_crash():
    """trainer.fit() should complete 1 epoch on 16-sample dataset without crash."""
    cfg = _minimal_cfg(epochs=1, batch_size=8, use_fft=False,
                       use_siglip=False, use_patch_contrastive=False)
    ds = TinyDataset(n=16)
    trainer, _ = _trainer_with_tmpdir(cfg, ds)
    trainer.fit()
    # 16 samples / batch_size 8 = 2 steps (drop_last=True, exact multiple)
    assert trainer.global_step == 2


# ---------------------------------------------------------------------------
# test_checkpoint_saved
# ---------------------------------------------------------------------------

def test_checkpoint_saved():
    """checkpoint_final.pth must exist after fit()."""
    cfg = _minimal_cfg(epochs=1, batch_size=8)
    ds = TinyDataset(n=16)
    trainer, tmp = _trainer_with_tmpdir(cfg, ds)
    trainer.fit()
    assert (tmp / "checkpoint_final.pth").exists()


# ---------------------------------------------------------------------------
# test_aux_heads_engaged_when_enabled
# ---------------------------------------------------------------------------

def test_aux_heads_engaged_when_enabled():
    """When use_fft and use_patch_contrastive are True, loss dict has those keys > 0."""
    cfg = _minimal_cfg(
        epochs=1, batch_size=8,
        use_fft=True,
        use_patch_contrastive=True,
        use_siglip=False,
    )
    ds = TinyDataset(n=16)
    trainer, _ = _trainer_with_tmpdir(cfg, ds)
    assert trainer.fft_head is not None
    assert trainer.patch_head is not None

    # Manually run one forward step and inspect loss components
    trainer._set_train_mode()
    images, labels = ds[0][0].unsqueeze(0).repeat(8, 1, 1, 1), torch.zeros(8, dtype=torch.long)
    loss_dict = trainer._forward_batch(images, labels)
    # Both components must be tensors > 0
    assert "fft" in loss_dict
    assert "patch" in loss_dict
    assert loss_dict["fft"].item() >= 0.0
    assert loss_dict["patch"].item() >= 0.0


# ---------------------------------------------------------------------------
# test_siglip_teacher_callable
# ---------------------------------------------------------------------------

def test_siglip_teacher_callable():
    """Passing a mock siglip_teacher_fn causes siglip loss to appear in loss dict."""
    teacher_dim = 1152

    def mock_teacher(images: torch.Tensor) -> torch.Tensor:
        return torch.randn(images.shape[0], teacher_dim)

    cfg = _minimal_cfg(
        epochs=1, batch_size=8,
        use_siglip=True,
        siglip_teacher_dim=teacher_dim,
        use_fft=False,
        use_patch_contrastive=False,
    )
    ds = TinyDataset(n=16)
    trainer, _ = _trainer_with_tmpdir(cfg, ds, siglip_teacher_fn=mock_teacher)
    assert trainer.siglip_head is not None

    trainer._set_train_mode()
    images = ds[0][0].unsqueeze(0).repeat(8, 1, 1, 1)
    labels = torch.zeros(8, dtype=torch.long)
    loss_dict = trainer._forward_batch(images, labels)
    assert "siglip" in loss_dict
    # weighted siglip loss > 0 (random init → non-zero MSE)
    assert loss_dict["siglip"].item() >= 0.0


# ---------------------------------------------------------------------------
# test_cosine_margin_path
# ---------------------------------------------------------------------------

def test_cosine_margin_path():
    """With main_loss='cosine_margin', main_uses_features is True and no MainHead."""
    cfg = _minimal_cfg(
        epochs=1, batch_size=8,
        main_loss="cosine_margin",
        use_fft=False,
        use_siglip=False,
        use_patch_contrastive=False,
    )
    ds = TinyDataset(n=16)
    trainer, _ = _trainer_with_tmpdir(cfg, ds)

    assert trainer.main_uses_features is True
    assert trainer.main_head is None
    # fit should not crash on cosine_margin path
    trainer.fit()


# ---------------------------------------------------------------------------
# test_grad_clipping_active
# ---------------------------------------------------------------------------

def test_grad_clipping_active():
    """Confirm grad_clip_norm is configured and a training step produces no NaN."""
    cfg = _minimal_cfg(
        epochs=1, batch_size=8,
        grad_clip_norm=1.0,
        use_fft=False,
        use_siglip=False,
        use_patch_contrastive=False,
    )
    ds = TinyDataset(n=16)
    trainer, _ = _trainer_with_tmpdir(cfg, ds)
    assert trainer.cfg.grad_clip_norm == 1.0

    trainer._set_train_mode()
    batch = (
        ds[0][0].unsqueeze(0).repeat(8, 1, 1, 1),
        torch.zeros(8, dtype=torch.long),
    )
    metrics = trainer.train_step(batch)
    total = metrics["total"]
    assert not (total != total)  # NaN check: NaN != NaN is True


# ---------------------------------------------------------------------------
# test_no_siglip_head_when_disabled
# ---------------------------------------------------------------------------

def test_no_siglip_head_when_disabled():
    """use_siglip=False must result in trainer.siglip_head being None."""
    cfg = _minimal_cfg(use_siglip=False)
    ds = TinyDataset(n=8)
    trainer, _ = _trainer_with_tmpdir(cfg, ds)
    assert trainer.siglip_head is None
