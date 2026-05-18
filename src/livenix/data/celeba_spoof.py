"""CelebA-Spoof dataset loader for Livenix.

Maps CelebA-Spoof's 11 fine-grained spoof types to the Livenix 3-class
scheme (real / print_spoof / replay_spoof).

Mapping rationale:
  0       -> 0 (real)
  1-6     -> 1 (print_spoof:   photo/poster/A4/paper-masks)
  7-9     -> 2 (replay_spoof:  PC/pad/phone screens)
  10      -> EXCLUDED (3D mask — see CLAIMS.md, Phase 1 does not claim
                        silicone defense)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor
from torch.utils.data import Dataset


# Module-level mapping so it can be imported and inspected independently.
# None means "exclude this sample entirely from the dataset".
CELEBA_SPOOF_TO_LIVENIX_LABEL: dict[int, int | None] = {
    0:  0,    # Live face          -> real
    1:  1,    # Photo              -> print_spoof
    2:  1,    # Poster             -> print_spoof
    3:  1,    # A4                 -> print_spoof
    4:  1,    # Face Mask (paper)  -> print_spoof
    5:  1,    # Upper Body Mask    -> print_spoof
    6:  1,    # Region Mask        -> print_spoof
    7:  2,    # PC (monitor)       -> replay_spoof
    8:  2,    # Pad (tablet)       -> replay_spoof
    9:  2,    # Phone              -> replay_spoof
    10: None, # 3D Mask            -> EXCLUDED in Phase 1
}


class CelebASpoofDataset(Dataset):
    """CelebA-Spoof loader mapping fine-grained spoof types to the
    Livenix 3-class scheme (real / print_spoof / replay_spoof).

    3D mask samples (CelebA-Spoof type 10) are EXCLUDED in Phase 1 because
    we do not claim silicone defense (see CLAIMS.md). They will be revisited
    in Phase 2 when HiFiMask paid licensing lands.

    Args:
        root: Path to CelebA-Spoof root directory (containing Data/).
            If the directory does not exist or contains no valid samples,
            the dataset is empty (len == 0) and getitem raises IndexError.
        split: 'train' or 'test'.
        transform: Optional callable applied to PIL image -> Tensor.
            If None, returns the raw PIL Image so tests can inspect it.
        include_attributes: If True, also yields the 40-dim attribute vector.
            Default False to keep the per-sample tuple small for training.

    Returns from __getitem__:
        tuple (image, label, attack_type_int)
        where:
            image: PIL Image OR Tensor (depending on transform)
            label: int in {0, 1, 2} (Livenix 3-class)
            attack_type_int: int in {0..9} — the ORIGINAL CelebA-Spoof spoof
                type ID, useful for per-attack confusion-matrix reporting.

        For real samples, attack_type_int == 0.

        If include_attributes=True, returns 4-tuple:
            (image, label, attack_type_int, attrs)
        where attrs is a FloatTensor of shape (40,).
    """

    SUBDIRS = ("live", "spoof")
    ALLOWED_SPLITS = {"train", "test"}

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Callable | None = None,
        include_attributes: bool = False,
    ) -> None:
        super().__init__()
        if split not in self.ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {self.ALLOWED_SPLITS}")
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.include_attributes = include_attributes
        self.samples: list[tuple[Path, int, int]] = self._discover_samples()

    def _discover_samples(self) -> list[tuple[Path, int, int]]:
        """Walk <root>/Data/<split>/<subject>/{live,spoof}/*.png.

        For each image:
          - read sidecar .txt (same stem)
          - parse the last line as the spoof type ID
          - look up Livenix label via CELEBA_SPOOF_TO_LIVENIX_LABEL
          - if None (3D mask), skip
          - otherwise add tuple (image_path, livenix_label, original_spoof_type)

        Robustness:
          - silently skip images without sidecar .txt
          - silently skip images whose sidecar .txt has non-integer last line
          - silently skip whole directory if root/Data doesn't exist
        """
        data_dir = self.root / "Data" / self.split
        if not data_dir.is_dir():
            return []

        samples: list[tuple[Path, int, int]] = []

        for subject_dir in sorted(data_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            for subdir in self.SUBDIRS:
                subdir_path = subject_dir / subdir
                if not subdir_path.is_dir():
                    continue
                for img_path in sorted(subdir_path.glob("*.png")):
                    sidecar = img_path.with_suffix(".txt")
                    if not sidecar.is_file():
                        continue
                    spoof_type = _parse_spoof_type(sidecar)
                    if spoof_type is None:
                        continue
                    livenix_label = CELEBA_SPOOF_TO_LIVENIX_LABEL.get(spoof_type)
                    if livenix_label is None:
                        continue
                    samples.append((img_path, livenix_label, spoof_type))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"index {idx} out of range for dataset of size {len(self.samples)}")
        img_path, label, attack_type = self.samples[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        if self.include_attributes:
            attrs = self._load_attributes(img_path)
            return im, label, attack_type, attrs
        return im, label, attack_type

    def _load_attributes(self, img_path: Path) -> Tensor:
        """Read sidecar .txt's first 40 lines as 0/1 attribute floats.

        Returns (40,) FloatTensor. If file is malformed, returns zeros.
        """
        sidecar = img_path.with_suffix(".txt")
        try:
            lines = sidecar.read_text().splitlines()
            attr_values = []
            for line in lines[:40]:
                stripped = line.strip()
                if stripped == "":
                    attr_values.append(0.0)
                else:
                    attr_values.append(float(stripped))
            # Pad with zeros if fewer than 40 attribute lines
            while len(attr_values) < 40:
                attr_values.append(0.0)
            return torch.tensor(attr_values[:40], dtype=torch.float32)
        except Exception:
            return torch.zeros(40, dtype=torch.float32)


def _parse_spoof_type(sidecar: Path) -> int | None:
    """Parse the last non-empty line of a sidecar .txt as an integer spoof type.

    Returns None if parsing fails.
    """
    try:
        text = sidecar.read_text()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None
        return int(lines[-1])
    except (ValueError, OSError):
        return None


class CelebASpoofCropDataset(Dataset):
    """Loader for the pre-cropped CelebA-Spoof variant (immada/celeba-spoof-crop).

    Structure: <root>/CelebA_Spoof/<split>/live/*.jpg  (label 0 — real)
               <root>/CelebA_Spoof/<split>/spoof/*.jpg (label 1 — print_spoof)

    Only two classes available (no replay distinction) — sufficient for a
    first baseline training run. Use CelebASpoofDataset for full 3-class.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples: list[tuple[Path, int]] = self._discover()

    def _discover(self) -> list[tuple[Path, int]]:
        base = self.root / "CelebA_Spoof" / self.split
        if not base.is_dir():
            # fallback: try split directly under root
            base = self.root / self.split
        if not base.is_dir():
            return []
        samples = []
        for label, subdir in ((0, "live"), (1, "spoof")):
            d = base / subdir
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in self.IMG_EXTS:
                    samples.append((p, label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        attempts = 0
        n = len(self.samples)
        while attempts < n:
            img_path, label = self.samples[(idx + attempts) % n]
            try:
                with Image.open(img_path) as im:
                    im = im.convert("RGB")
                if self.transform is not None:
                    im = self.transform(im)
                return im, label, label
            except (UnidentifiedImageError, OSError):
                attempts += 1
        raise RuntimeError("CelebASpoofCropDataset: all samples unreadable")


def write_fake_sample(
    root: Path,
    subject_id: str,
    split: str,
    subdir: str,
    filename_stem: str,
    spoof_type: int,
    image_size: tuple[int, int] = (32, 32),
) -> Path:
    """Write a single fake CelebA-Spoof sample (a tiny PNG + its sidecar .txt).

    The .txt has 40 attribute lines (all 0) plus the spoof type line.
    Returns the path to the written .png file.

    Args:
        root: CelebA-Spoof root directory (the directory containing Data/).
        subject_id: Subject directory name (e.g. "0000001").
        split: 'train' or 'test'.
        subdir: 'live' or 'spoof'.
        filename_stem: Filename without extension (e.g. "frame_001").
        spoof_type: CelebA-Spoof spoof type integer (0-10).
        image_size: (width, height) for the generated PNG.

    Returns:
        Path to the written .png file.
    """
    img_dir = root / "Data" / split / subject_id / subdir
    img_dir.mkdir(parents=True, exist_ok=True)

    img_path = img_dir / f"{filename_stem}.png"
    sidecar_path = img_dir / f"{filename_stem}.txt"

    # Write a minimal RGB PNG
    img = Image.new("RGB", image_size, color=(128, 64, 192))
    img.save(img_path)

    # Write sidecar: 40 attribute lines (all 0) + spoof type line
    lines = ["0"] * 40 + [str(spoof_type)]
    sidecar_path.write_text("\n".join(lines) + "\n")

    return img_path
