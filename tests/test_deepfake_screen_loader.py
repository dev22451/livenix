"""Tests for DeepfakeScreenDataset."""
from pathlib import Path

import pytest
import torch
from PIL import Image

from livenix.data import (
    DeepfakeScreenDataset,
    DEEPFAKE_SCREEN_LABEL,
    DEEPFAKE_SCREEN_ATTACK_TYPE,
)


def _write_fake_png(path: Path, size=(16, 16)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(128, 128, 128)).save(path)
    return path


def test_constants():
    assert DEEPFAKE_SCREEN_LABEL == 2
    assert DEEPFAKE_SCREEN_ATTACK_TYPE == "deepfake_on_screen"


def test_empty_root_is_empty_dataset(tmp_path):
    ds = DeepfakeScreenDataset(tmp_path / "nonexistent")
    assert len(ds) == 0


def test_loads_flat_frames(tmp_path):
    _write_fake_png(tmp_path / "frame_0.png")
    _write_fake_png(tmp_path / "frame_1.png")
    ds = DeepfakeScreenDataset(tmp_path)
    assert len(ds) == 2
    img, label, attack_type = ds[0]
    assert isinstance(img, Image.Image)
    assert label == 2
    assert attack_type == "deepfake_on_screen"


def test_loads_subdir_frames(tmp_path):
    _write_fake_png(tmp_path / "session1" / "a.png")
    _write_fake_png(tmp_path / "session2" / "b.png")
    ds = DeepfakeScreenDataset(tmp_path)
    assert len(ds) == 2


def test_transform_applied(tmp_path):
    _write_fake_png(tmp_path / "f.png")

    def to_tensor(im):
        return torch.zeros(3, 16, 16)

    ds = DeepfakeScreenDataset(tmp_path, transform=to_tensor)
    img, _, _ = ds[0]
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, 16, 16)
