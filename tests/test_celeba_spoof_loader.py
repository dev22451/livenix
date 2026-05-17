"""Unit tests for CelebASpoofDataset.

All tests use tmp_path fixtures; no real CelebA-Spoof download is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from livenix.data.celeba_spoof import (
    CELEBA_SPOOF_TO_LIVENIX_LABEL,
    CelebASpoofDataset,
    write_fake_sample,
)


# ---------------------------------------------------------------------------
# test_label_mapping_table_matches_spec
# ---------------------------------------------------------------------------

def test_label_mapping_table_matches_spec():
    expected = {
        0:  0,    # Live face        -> real
        1:  1,    # Photo            -> print_spoof
        2:  1,    # Poster           -> print_spoof
        3:  1,    # A4               -> print_spoof
        4:  1,    # Face Mask (paper)-> print_spoof
        5:  1,    # Upper Body Mask  -> print_spoof
        6:  1,    # Region Mask      -> print_spoof
        7:  2,    # PC               -> replay_spoof
        8:  2,    # Pad              -> replay_spoof
        9:  2,    # Phone            -> replay_spoof
        10: None, # 3D Mask          -> EXCLUDED
    }
    assert CELEBA_SPOOF_TO_LIVENIX_LABEL == expected


# ---------------------------------------------------------------------------
# test_empty_root_is_empty_dataset
# ---------------------------------------------------------------------------

def test_empty_root_is_empty_dataset(tmp_path: Path):
    nonexistent = tmp_path / "does_not_exist"
    ds = CelebASpoofDataset(root=nonexistent, split="train")
    assert len(ds) == 0
    # Iterating over an empty dataset should not crash
    items = list(ds)
    assert items == []


# ---------------------------------------------------------------------------
# test_loads_live_sample
# ---------------------------------------------------------------------------

def test_loads_live_sample(tmp_path: Path):
    write_fake_sample(
        root=tmp_path,
        subject_id="subject_001",
        split="train",
        subdir="live",
        filename_stem="frame_001",
        spoof_type=0,
    )
    ds = CelebASpoofDataset(root=tmp_path, split="train")
    assert len(ds) == 1
    img, label, attack_type = ds[0]
    assert label == 0, "Live sample must map to label 0 (real)"
    assert attack_type == 0, "Live sample attack_type must be 0"
    assert isinstance(img, Image.Image), "__getitem__ without transform returns PIL Image"


# ---------------------------------------------------------------------------
# test_loads_print_spoof_sample
# ---------------------------------------------------------------------------

def test_loads_print_spoof_sample(tmp_path: Path):
    write_fake_sample(
        root=tmp_path,
        subject_id="subject_002",
        split="train",
        subdir="spoof",
        filename_stem="a4_001",
        spoof_type=3,  # A4 -> print_spoof
    )
    ds = CelebASpoofDataset(root=tmp_path, split="train")
    assert len(ds) == 1
    img, label, attack_type = ds[0]
    assert label == 1, "A4 spoof must map to label 1 (print_spoof)"
    assert attack_type == 3


# ---------------------------------------------------------------------------
# test_loads_replay_spoof_sample
# ---------------------------------------------------------------------------

def test_loads_replay_spoof_sample(tmp_path: Path):
    write_fake_sample(
        root=tmp_path,
        subject_id="subject_003",
        split="train",
        subdir="spoof",
        filename_stem="phone_001",
        spoof_type=9,  # Phone -> replay_spoof
    )
    ds = CelebASpoofDataset(root=tmp_path, split="train")
    assert len(ds) == 1
    img, label, attack_type = ds[0]
    assert label == 2, "Phone spoof must map to label 2 (replay_spoof)"
    assert attack_type == 9


# ---------------------------------------------------------------------------
# test_excludes_3d_mask
# ---------------------------------------------------------------------------

def test_excludes_3d_mask(tmp_path: Path):
    # 3D mask sample — should be excluded
    write_fake_sample(
        root=tmp_path,
        subject_id="subject_004",
        split="train",
        subdir="spoof",
        filename_stem="mask3d_001",
        spoof_type=10,  # 3D Mask -> EXCLUDED
    )
    # Live sample — should be included
    write_fake_sample(
        root=tmp_path,
        subject_id="subject_004",
        split="train",
        subdir="live",
        filename_stem="live_001",
        spoof_type=0,
    )
    ds = CelebASpoofDataset(root=tmp_path, split="train")
    assert len(ds) == 1, "Only the live sample should be present; 3D mask must be excluded"
    _, label, attack_type = ds[0]
    assert label == 0
    assert attack_type == 0


# ---------------------------------------------------------------------------
# test_skips_missing_sidecar
# ---------------------------------------------------------------------------

def test_skips_missing_sidecar(tmp_path: Path):
    # Create a PNG with no .txt sidecar
    img_dir = tmp_path / "Data" / "train" / "subject_005" / "spoof"
    img_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 32), color=(0, 0, 0))
    img.save(img_dir / "no_sidecar.png")
    # No .txt written

    ds = CelebASpoofDataset(root=tmp_path, split="train")
    assert len(ds) == 0, "Image without sidecar .txt must be silently skipped"


# ---------------------------------------------------------------------------
# test_skips_malformed_sidecar
# ---------------------------------------------------------------------------

def test_skips_malformed_sidecar(tmp_path: Path):
    # Create a PNG with a .txt whose last line is not an integer
    img_dir = tmp_path / "Data" / "train" / "subject_006" / "spoof"
    img_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (32, 32), color=(0, 0, 0))
    img_path = img_dir / "bad_label.png"
    img.save(img_path)
    sidecar = img_dir / "bad_label.txt"
    lines = ["0"] * 40 + ["not_an_int"]
    sidecar.write_text("\n".join(lines) + "\n")

    ds = CelebASpoofDataset(root=tmp_path, split="train")
    assert len(ds) == 0, "Image with non-integer last sidecar line must be silently skipped"


# ---------------------------------------------------------------------------
# test_transform_applied
# ---------------------------------------------------------------------------

def test_transform_applied(tmp_path: Path):
    write_fake_sample(
        root=tmp_path,
        subject_id="subject_007",
        split="train",
        subdir="live",
        filename_stem="live_001",
        spoof_type=0,
    )

    fixed_tensor = torch.ones(3, 32, 32)

    def fixed_transform(pil_img: Image.Image) -> torch.Tensor:
        return fixed_tensor

    ds = CelebASpoofDataset(root=tmp_path, split="train", transform=fixed_transform)
    img, label, attack_type = ds[0]
    assert isinstance(img, torch.Tensor), "__getitem__ with transform must return a Tensor"
    assert torch.equal(img, fixed_tensor)


# ---------------------------------------------------------------------------
# test_split_validation
# ---------------------------------------------------------------------------

def test_split_validation(tmp_path: Path):
    with pytest.raises(ValueError, match="split must be one of"):
        CelebASpoofDataset(root=tmp_path, split="invalid")


# ---------------------------------------------------------------------------
# test_train_test_isolation
# ---------------------------------------------------------------------------

def test_train_test_isolation(tmp_path: Path):
    write_fake_sample(
        root=tmp_path,
        subject_id="subject_010",
        split="train",
        subdir="live",
        filename_stem="live_train",
        spoof_type=0,
    )
    write_fake_sample(
        root=tmp_path,
        subject_id="subject_011",
        split="test",
        subdir="live",
        filename_stem="live_test",
        spoof_type=0,
    )

    train_ds = CelebASpoofDataset(root=tmp_path, split="train")
    test_ds = CelebASpoofDataset(root=tmp_path, split="test")

    assert len(train_ds) == 1, "train split must have exactly 1 sample"
    assert len(test_ds) == 1, "test split must have exactly 1 sample"


# ---------------------------------------------------------------------------
# test_include_attributes
# ---------------------------------------------------------------------------

def test_include_attributes(tmp_path: Path):
    write_fake_sample(
        root=tmp_path,
        subject_id="subject_012",
        split="train",
        subdir="live",
        filename_stem="live_001",
        spoof_type=0,
    )
    ds = CelebASpoofDataset(root=tmp_path, split="train", include_attributes=True)
    assert len(ds) == 1
    result = ds[0]
    assert len(result) == 4, "__getitem__ with include_attributes=True must return 4-tuple"
    img, label, attack_type, attrs = result
    assert isinstance(attrs, torch.Tensor), "attrs must be a FloatTensor"
    assert attrs.shape == (40,), "attrs must have shape (40,)"
    assert attrs.dtype == torch.float32
