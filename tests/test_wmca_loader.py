"""Unit tests for WMCADataset.

All tests use tmp_path fixtures; no real WMCA download is required.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from livenix.data.wmca import (
    WMCA_TO_LIVENIX_LABEL,
    WMCADataset,
    write_fake_wmca_sample,
)


# ---------------------------------------------------------------------------
# test_empty_root_is_empty_dataset
# ---------------------------------------------------------------------------

def test_empty_root_is_empty_dataset(tmp_path: Path):
    nonexistent = tmp_path / "does_not_exist"
    ds = WMCADataset(root=nonexistent, split="train")
    assert len(ds) == 0
    items = list(ds)
    assert items == []


# ---------------------------------------------------------------------------
# test_label_mapping_table_matches_spec
# ---------------------------------------------------------------------------

def test_label_mapping_table_matches_spec():
    expected: dict[str, int | None] = {
        "bonafide":     0,
        "print":        1,
        "papermask":    1,
        "replay":       2,
        "glasses":      None,
        "fakehead":     None,
        "rigidmask":    None,
        "flexiblemask": None,
    }
    assert WMCA_TO_LIVENIX_LABEL == expected


# ---------------------------------------------------------------------------
# test_loads_bonafide
# ---------------------------------------------------------------------------

def test_loads_bonafide(tmp_path: Path):
    write_fake_wmca_sample(
        root=tmp_path,
        session_id="s1",
        attack_type="bonafide",
        subject_id="sub01",
    )
    ds = WMCADataset(root=tmp_path, split="train")
    assert len(ds) == 1
    img, label, attack_type_str = ds[0]
    assert label == 0, "bonafide must map to label 0 (real)"
    assert attack_type_str == "bonafide"
    assert isinstance(img, Image.Image), "__getitem__ without transform returns PIL Image"


# ---------------------------------------------------------------------------
# test_loads_print
# ---------------------------------------------------------------------------

def test_loads_print(tmp_path: Path):
    write_fake_wmca_sample(
        root=tmp_path,
        session_id="s1",
        attack_type="print",
        subject_id="sub01",
    )
    ds = WMCADataset(root=tmp_path, split="train")
    assert len(ds) == 1
    img, label, attack_type_str = ds[0]
    assert label == 1, "print must map to label 1 (print_spoof)"
    assert attack_type_str == "print"


# ---------------------------------------------------------------------------
# test_loads_replay
# ---------------------------------------------------------------------------

def test_loads_replay(tmp_path: Path):
    write_fake_wmca_sample(
        root=tmp_path,
        session_id="s1",
        attack_type="replay",
        subject_id="sub01",
    )
    ds = WMCADataset(root=tmp_path, split="train")
    assert len(ds) == 1
    _, label, attack_type_str = ds[0]
    assert label == 2, "replay must map to label 2 (replay_spoof)"
    assert attack_type_str == "replay"


# ---------------------------------------------------------------------------
# test_loads_papermask
# ---------------------------------------------------------------------------

def test_loads_papermask(tmp_path: Path):
    write_fake_wmca_sample(
        root=tmp_path,
        session_id="s1",
        attack_type="papermask",
        subject_id="sub01",
    )
    ds = WMCADataset(root=tmp_path, split="train")
    assert len(ds) == 1
    _, label, attack_type_str = ds[0]
    assert label == 1, "papermask must map to label 1 (print_spoof)"
    assert attack_type_str == "papermask"


# ---------------------------------------------------------------------------
# test_excludes_flexiblemask
# ---------------------------------------------------------------------------

def test_excludes_flexiblemask(tmp_path: Path):
    write_fake_wmca_sample(
        root=tmp_path,
        session_id="s1",
        attack_type="bonafide",
        subject_id="sub01",
        frame_idx=0,
    )
    write_fake_wmca_sample(
        root=tmp_path,
        session_id="s2",
        attack_type="flexiblemask",
        subject_id="sub02",
        frame_idx=0,
    )
    ds = WMCADataset(root=tmp_path, split="train")
    assert len(ds) == 1, "flexiblemask must be excluded; only bonafide should remain"
    _, label, attack_type_str = ds[0]
    assert label == 0
    assert attack_type_str == "bonafide"


# ---------------------------------------------------------------------------
# test_excludes_rigidmask_glasses_fakehead
# ---------------------------------------------------------------------------

def test_excludes_rigidmask_glasses_fakehead(tmp_path: Path):
    write_fake_wmca_sample(
        root=tmp_path,
        session_id="s1",
        attack_type="bonafide",
        subject_id="sub01",
        frame_idx=0,
    )
    write_fake_wmca_sample(
        root=tmp_path,
        session_id="s2",
        attack_type="rigidmask",
        subject_id="sub02",
        frame_idx=0,
    )
    write_fake_wmca_sample(
        root=tmp_path,
        session_id="s3",
        attack_type="glasses",
        subject_id="sub03",
        frame_idx=0,
    )
    write_fake_wmca_sample(
        root=tmp_path,
        session_id="s4",
        attack_type="fakehead",
        subject_id="sub04",
        frame_idx=0,
    )
    ds = WMCADataset(root=tmp_path, split="train")
    assert len(ds) == 1, (
        "rigidmask, glasses, fakehead must all be excluded; only bonafide should remain"
    )
    _, label, attack_type_str = ds[0]
    assert label == 0
    assert attack_type_str == "bonafide"


# ---------------------------------------------------------------------------
# test_filename_parsing_robust_to_extra_tokens
# ---------------------------------------------------------------------------

def test_filename_parsing_robust_to_extra_tokens(tmp_path: Path):
    """Filename with extra underscore tokens should still parse the second token."""
    images_dir = tmp_path / "preprocessed-images"
    images_dir.mkdir(parents=True, exist_ok=True)
    # Non-standard but second token is "print"
    img_path = images_dir / "s1_print_sub01_0001_extra_token.png"
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(img_path)

    ds = WMCADataset(root=tmp_path, split="train")
    assert len(ds) == 1
    _, label, attack_type_str = ds[0]
    assert label == 1
    assert attack_type_str == "print"


# ---------------------------------------------------------------------------
# test_filename_with_too_few_tokens_skipped
# ---------------------------------------------------------------------------

def test_filename_with_too_few_tokens_skipped(tmp_path: Path):
    """Filenames with no underscores (< 2 tokens) must be silently skipped."""
    images_dir = tmp_path / "preprocessed-images"
    images_dir.mkdir(parents=True, exist_ok=True)
    img_path = images_dir / "noattack.png"
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(img_path)

    ds = WMCADataset(root=tmp_path, split="train")
    assert len(ds) == 0, "Filename with no underscores must be silently skipped"


# ---------------------------------------------------------------------------
# test_case_insensitive_attack_lookup
# ---------------------------------------------------------------------------

def test_case_insensitive_attack_lookup(tmp_path: Path):
    """Uppercase attack type in filename should still resolve correctly."""
    images_dir = tmp_path / "preprocessed-images"
    images_dir.mkdir(parents=True, exist_ok=True)
    img_path = images_dir / "s1_PRINT_sub01_0000.png"
    Image.new("RGB", (32, 32), color=(0, 0, 0)).save(img_path)

    ds = WMCADataset(root=tmp_path, split="train")
    assert len(ds) == 1, "Case-insensitive lookup must find PRINT as print"
    _, label, attack_type_str = ds[0]
    assert label == 1
    assert attack_type_str == "print"


# ---------------------------------------------------------------------------
# test_protocol_csv_filter
# ---------------------------------------------------------------------------

def test_protocol_csv_filter(tmp_path: Path):
    """Protocol CSV filtering: only listed filenames included for the split."""
    # Write two print samples; add only the first to protocol CSV
    path_a = write_fake_wmca_sample(
        root=tmp_path,
        session_id="s1",
        attack_type="print",
        subject_id="sub01",
        frame_idx=0,
        protocol_split="train",  # appears in CSV
    )
    path_b = write_fake_wmca_sample(
        root=tmp_path,
        session_id="s2",
        attack_type="print",
        subject_id="sub02",
        frame_idx=0,
        # NOT added to CSV
    )

    # With protocol CSV present: only the one in CSV is included
    ds_filtered = WMCADataset(root=tmp_path, split="train")
    assert len(ds_filtered) == 1, (
        "With protocol CSV, only the listed sample should be included"
    )
    _, label, _ = ds_filtered[0]
    assert label == 1

    # Remove the CSV and verify both are included (fallback to all)
    csv_path = tmp_path / "protocols" / "grandtest_train.csv"
    csv_path.unlink()
    ds_all = WMCADataset(root=tmp_path, split="train")
    assert len(ds_all) == 2, (
        "Without protocol CSV, all valid samples should be returned"
    )


# ---------------------------------------------------------------------------
# test_split_validation
# ---------------------------------------------------------------------------

def test_split_validation(tmp_path: Path):
    with pytest.raises(ValueError, match="split must be one of"):
        WMCADataset(root=tmp_path, split="bogus")


# ---------------------------------------------------------------------------
# test_transform_applied
# ---------------------------------------------------------------------------

def test_transform_applied(tmp_path: Path):
    write_fake_wmca_sample(
        root=tmp_path,
        session_id="s1",
        attack_type="bonafide",
        subject_id="sub01",
    )

    fixed_tensor = torch.ones(3, 32, 32)

    def fixed_transform(pil_img: Image.Image) -> torch.Tensor:
        return fixed_tensor

    ds = WMCADataset(root=tmp_path, split="train", transform=fixed_transform)
    img, label, attack_type_str = ds[0]
    assert isinstance(img, torch.Tensor), "__getitem__ with transform must return a Tensor"
    assert torch.equal(img, fixed_tensor)
