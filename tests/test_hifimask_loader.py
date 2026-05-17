"""Tests for the HiFiMask dataset loader (T2.7).

All tests use tmp_path — no real HiFiMask data required.
"""

from __future__ import annotations

import pytest
import torch
from torchvision import transforms

from livenix.data.hifimask import (
    HIFIMASK_TO_LIVENIX_LABEL,
    HiFiMaskDataset,
    write_fake_hifimask_sample,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dataset(tmp_path, **kwargs) -> HiFiMaskDataset:
    """Create a HiFiMaskDataset rooted at tmp_path with extra kwargs."""
    return HiFiMaskDataset(root=tmp_path, **kwargs)


# ---------------------------------------------------------------------------
# 1. test_empty_root_is_empty_dataset
# ---------------------------------------------------------------------------


def test_empty_root_is_empty_dataset(tmp_path):
    """A root with no images/ directory yields an empty dataset."""
    ds = _make_dataset(tmp_path)
    assert len(ds) == 0


# ---------------------------------------------------------------------------
# 2. test_label_mapping_table_matches_spec
# ---------------------------------------------------------------------------


def test_label_mapping_table_matches_spec():
    """HIFIMASK_TO_LIVENIX_LABEL must contain the exact spec-mandated entries."""
    expected = {
        "bonafide":    0,
        "real":        0,
        "transparent": 1,
        "plaster":     1,
        "resin":       1,
        "latex":       1,
        "silicone":    1,
        "paper":       1,
    }
    assert HIFIMASK_TO_LIVENIX_LABEL == expected


# ---------------------------------------------------------------------------
# 3. test_loads_bonafide
# ---------------------------------------------------------------------------


def test_loads_bonafide(tmp_path):
    """bonafide material -> label 0, attack_subtype == 'bonafide'."""
    write_fake_hifimask_sample(tmp_path, "session_001", "subject_001", "bonafide")
    ds = _make_dataset(tmp_path)
    assert len(ds) == 1
    _img, label, subtype = ds[0]
    assert label == 0
    assert subtype == "bonafide"


# ---------------------------------------------------------------------------
# 4. test_loads_silicone
# ---------------------------------------------------------------------------


def test_loads_silicone(tmp_path):
    """silicone material -> label 1, attack_subtype == 'silicone'."""
    write_fake_hifimask_sample(tmp_path, "session_001", "subject_001", "silicone")
    ds = _make_dataset(tmp_path)
    assert len(ds) == 1
    _img, label, subtype = ds[0]
    assert label == 1
    assert subtype == "silicone"


# ---------------------------------------------------------------------------
# 5. test_loads_plaster
# ---------------------------------------------------------------------------


def test_loads_plaster(tmp_path):
    """plaster material -> label 1, attack_subtype == 'plaster'."""
    write_fake_hifimask_sample(tmp_path, "session_001", "subject_001", "plaster")
    ds = _make_dataset(tmp_path)
    assert len(ds) == 1
    _img, label, subtype = ds[0]
    assert label == 1
    assert subtype == "plaster"


# ---------------------------------------------------------------------------
# 6. test_loads_resin
# ---------------------------------------------------------------------------


def test_loads_resin(tmp_path):
    """resin material -> label 1, attack_subtype == 'resin'."""
    write_fake_hifimask_sample(tmp_path, "session_001", "subject_001", "resin")
    ds = _make_dataset(tmp_path)
    assert len(ds) == 1
    _img, label, subtype = ds[0]
    assert label == 1
    assert subtype == "resin"


# ---------------------------------------------------------------------------
# 7. test_loads_transparent
# ---------------------------------------------------------------------------


def test_loads_transparent(tmp_path):
    """transparent material -> label 1."""
    write_fake_hifimask_sample(tmp_path, "session_001", "subject_001", "transparent")
    ds = _make_dataset(tmp_path)
    assert len(ds) == 1
    _img, label, subtype = ds[0]
    assert label == 1
    assert subtype == "transparent"


# ---------------------------------------------------------------------------
# 8. test_loads_real_alias
# ---------------------------------------------------------------------------


def test_loads_real_alias(tmp_path):
    """'real' is an alias for bonafide -> label 0."""
    write_fake_hifimask_sample(tmp_path, "session_001", "subject_001", "real")
    ds = _make_dataset(tmp_path)
    assert len(ds) == 1
    _img, label, subtype = ds[0]
    assert label == 0
    assert subtype == "real"


# ---------------------------------------------------------------------------
# 9. test_skips_unknown_material
# ---------------------------------------------------------------------------


def test_skips_unknown_material(tmp_path):
    """A material not in the mapping ('cardboard') is skipped silently."""
    write_fake_hifimask_sample(tmp_path, "session_001", "subject_001", "cardboard")
    ds = _make_dataset(tmp_path)
    assert len(ds) == 0


# ---------------------------------------------------------------------------
# 10. test_case_insensitive
# ---------------------------------------------------------------------------


def test_case_insensitive(tmp_path):
    """Uppercase directory name 'SILICONE' -> label 1 (case-insensitive lookup)."""
    write_fake_hifimask_sample(tmp_path, "session_001", "subject_001", "SILICONE")
    ds = _make_dataset(tmp_path)
    assert len(ds) == 1
    _img, label, subtype = ds[0]
    assert label == 1
    # subtype is lowercased
    assert subtype == "silicone"


# ---------------------------------------------------------------------------
# 11. test_protocol_filter
# ---------------------------------------------------------------------------


def test_protocol_filter(tmp_path):
    """Protocol filter: 3 samples written; only 2 listed in protocol file."""
    # Write 3 samples across different subjects
    p1 = write_fake_hifimask_sample(
        tmp_path, "session_001", "subject_001", "bonafide",
        filename_stem="frame_0",
        protocol="protocol_1", split="train",
    )
    p2 = write_fake_hifimask_sample(
        tmp_path, "session_001", "subject_002", "silicone",
        filename_stem="frame_0",
        protocol="protocol_1", split="train",
    )
    # Third sample: NOT added to protocol file
    write_fake_hifimask_sample(
        tmp_path, "session_001", "subject_003", "plaster",
        filename_stem="frame_0",
        # no protocol/split -> not added to protocol file
    )

    # With protocol filter: only the 2 listed samples should appear
    ds = HiFiMaskDataset(
        root=tmp_path, split="train", protocol="protocol_1"
    )
    assert len(ds) == 2

    returned_paths = {str(sample[0]) for sample in ds.samples}
    assert str(p1) in returned_paths
    assert str(p2) in returned_paths


# ---------------------------------------------------------------------------
# 12. test_no_protocol_no_filter
# ---------------------------------------------------------------------------


def test_no_protocol_no_filter(tmp_path):
    """With no protocol set, all samples are included (no filtering)."""
    write_fake_hifimask_sample(tmp_path, "session_001", "subject_001", "bonafide")
    write_fake_hifimask_sample(tmp_path, "session_001", "subject_002", "silicone")
    write_fake_hifimask_sample(tmp_path, "session_002", "subject_003", "plaster")

    # No protocols/ directory exists — should still return all 3 samples
    ds = _make_dataset(tmp_path)
    assert len(ds) == 3


# ---------------------------------------------------------------------------
# 13. test_split_validation
# ---------------------------------------------------------------------------


def test_split_validation(tmp_path):
    """Passing an invalid split raises ValueError."""
    with pytest.raises(ValueError, match="split"):
        HiFiMaskDataset(root=tmp_path, split="bogus")


# ---------------------------------------------------------------------------
# 14. test_protocol_validation
# ---------------------------------------------------------------------------


def test_protocol_validation(tmp_path):
    """Passing an invalid protocol raises ValueError."""
    with pytest.raises(ValueError, match="protocol"):
        HiFiMaskDataset(root=tmp_path, protocol="bogus")


# ---------------------------------------------------------------------------
# 15. test_transform_applied
# ---------------------------------------------------------------------------


def test_transform_applied(tmp_path):
    """When a transform is provided, __getitem__ returns a Tensor, not PIL."""
    write_fake_hifimask_sample(tmp_path, "session_001", "subject_001", "bonafide")
    transform = transforms.ToTensor()
    ds = HiFiMaskDataset(root=tmp_path, transform=transform)
    assert len(ds) == 1
    img, label, subtype = ds[0]
    assert isinstance(img, torch.Tensor)
    # ToTensor produces (C, H, W) float in [0, 1]
    assert img.dtype == torch.float32
    assert img.shape[0] == 3  # RGB channels
    assert label == 0
    assert subtype == "bonafide"
