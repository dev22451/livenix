"""WMCA (Wide Multi-Channel presentation Attack) dataset loader for Livenix.

Maps WMCA's 8 attack types to the Livenix 3-class scheme
(real / print_spoof / replay_spoof).

Mapping rationale (see CLAIMS.md for Phase 1 exclusion policy):
  bonafide    -> 0 (real)
  print       -> 1 (print_spoof)
  papermask   -> 1 (print_spoof — paper mask is print-class)
  replay      -> 2 (replay_spoof)
  glasses     -> EXCLUDED (partial occlusion — out of scope Phase 1)
  fakehead    -> EXCLUDED (mannequin head — out of scope Phase 1)
  rigidmask   -> EXCLUDED (3D rigid mask — out of scope, Phase 2)
  flexiblemask-> EXCLUDED (silicone mask — out of scope, Phase 2)

Reference: Mostaani et al. 2020 (IDIAP), WMCA public preprocessed release.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


# Module-level mapping so it can be imported and inspected independently.
# Keys are LOWERCASE attack type strings. None means "exclude this sample".
WMCA_TO_LIVENIX_LABEL: dict[str, int | None] = {
    "bonafide":     0,     # Live face        -> real
    "print":        1,     # Print attack     -> print_spoof
    "papermask":    1,     # Paper mask       -> print_spoof
    "replay":       2,     # Replay attack    -> replay_spoof
    "glasses":      None,  # Partial occlusion -> EXCLUDED Phase 1
    "fakehead":     None,  # Mannequin head   -> EXCLUDED Phase 1
    "rigidmask":    None,  # 3D rigid mask    -> EXCLUDED Phase 1
    "flexiblemask": None,  # Silicone mask    -> EXCLUDED Phase 1
}


class WMCADataset(Dataset):
    """WMCA loader mapping attack types to the Livenix 3-class scheme.

    Phase 1 scope (see CLAIMS.md):
      INCLUDE: bonafide, print, replay, papermask
      EXCLUDE: glasses, fakehead, rigidmask, flexiblemask

    Args:
        root: Path to WMCA root (containing preprocessed-images/ and
            optionally protocols/).
        split: 'train' | 'dev' | 'eval' (matches WMCA grandtest splits).
            When protocols/ is present, filter by the corresponding CSV.
            When absent, ignore split (all samples returned).
        transform: optional PIL -> Tensor transform.

    Returns from __getitem__:
        (image, livenix_label, attack_type_str)
        where attack_type_str is the ORIGINAL lowercased WMCA attack name
        ("bonafide", "print", "replay", "papermask").
    """

    ALLOWED_SPLITS = {"train", "dev", "eval"}

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        if split not in self.ALLOWED_SPLITS:
            raise ValueError(f"split must be one of {self.ALLOWED_SPLITS}")
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples: list[tuple[Path, int, str]] = self._discover_samples()

    def _load_protocol_filenames(self) -> set[str] | None:
        """Load the set of filenames from protocols/grandtest_<split>.csv.

        Returns None if the protocol CSV does not exist (caller includes all).
        Returns a set of bare filenames (e.g. "s1_bonafide_s1_0001.png")
        if the CSV is present.
        """
        csv_path = self.root / "protocols" / f"grandtest_{self.split}.csv"
        if not csv_path.is_file():
            return None

        allowed: set[str] = set()
        try:
            with csv_path.open(newline="") as fh:
                reader = csv.reader(fh)
                for row in reader:
                    if not row:
                        continue
                    filename = row[0].strip()
                    if filename:
                        # Store just the basename to match against discovered files
                        allowed.add(Path(filename).name)
        except OSError:
            return None
        return allowed

    def _discover_samples(self) -> list[tuple[Path, int, str]]:
        """Walk preprocessed-images/, parse filename for attack type, filter.

        For each .png:
          1. Parse attack token from filename: filename.split("_")[1].lower()
             (second underscore-token in the standard WMCA naming convention)
          2. Look up via WMCA_TO_LIVENIX_LABEL (case-insensitive)
          3. Skip if None (excluded type)
          4. Skip if not in protocol CSV for the given split
             (only when protocols/<split>.csv exists; otherwise include all)

        Robustness:
          - missing root -> empty
          - missing preprocessed-images/ -> empty
          - filename with fewer than 2 underscore tokens -> skip
          - unknown attack token (not in our mapping) -> skip
        """
        images_dir = self.root / "preprocessed-images"
        if not images_dir.is_dir():
            return []

        # Load protocol CSV if available; None means "no filter"
        allowed_filenames = self._load_protocol_filenames()

        samples: list[tuple[Path, int, str]] = []

        for img_path in sorted(images_dir.glob("*.png")):
            # Protocol CSV filter
            if allowed_filenames is not None and img_path.name not in allowed_filenames:
                continue

            # Parse attack type from second underscore-separated token
            parts = img_path.stem.split("_")
            if len(parts) < 2:
                continue  # too few tokens — skip silently

            attack_token = parts[1].lower()

            # Look up in mapping (case-insensitive via lowercased key)
            livenix_label = WMCA_TO_LIVENIX_LABEL.get(attack_token)
            if livenix_label is None:
                # Either excluded type or unknown token
                if attack_token not in WMCA_TO_LIVENIX_LABEL:
                    continue  # unknown token — skip silently
                # Known but excluded (maps to None)
                continue

            samples.append((img_path, livenix_label, attack_token))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(
                f"index {idx} out of range for dataset of size {len(self.samples)}"
            )
        img_path, label, attack_type_str = self.samples[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        return im, label, attack_type_str


def write_fake_wmca_sample(
    root: Path,
    session_id: str,
    attack_type: str,
    subject_id: str,
    frame_idx: int = 0,
    image_size: tuple[int, int] = (32, 32),
    protocol_split: str | None = None,
) -> Path:
    """Write a single fake WMCA frame. Returns path to the written .png.

    Creates a PNG at:
        <root>/preprocessed-images/<session_id>_<attack_type>_<subject_id>_<frame_idx:04d>.png

    If protocol_split is set, also appends a row to:
        <root>/protocols/grandtest_<protocol_split>.csv

    Args:
        root: WMCA root directory (the directory that will contain
            preprocessed-images/ and optionally protocols/).
        session_id: Session identifier (e.g. "s1").
        attack_type: WMCA attack type string (e.g. "bonafide", "print").
        subject_id: Subject identifier (e.g. "sub01").
        frame_idx: Frame index (zero-padded to 4 digits).
        image_size: (width, height) for the generated PNG.
        protocol_split: If set ('train', 'dev', 'eval'), append the filename
            to protocols/grandtest_<protocol_split>.csv.

    Returns:
        Path to the written .png file.
    """
    images_dir = root / "preprocessed-images"
    images_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{session_id}_{attack_type}_{subject_id}_{frame_idx:04d}.png"
    img_path = images_dir / filename

    # Write a minimal RGB PNG
    img = Image.new("RGB", image_size, color=(128, 64, 192))
    img.save(img_path)

    if protocol_split is not None:
        protocols_dir = root / "protocols"
        protocols_dir.mkdir(parents=True, exist_ok=True)
        csv_path = protocols_dir / f"grandtest_{protocol_split}.csv"
        with csv_path.open("a") as fh:
            fh.write(f"{filename},{attack_type}\n")

    return img_path
