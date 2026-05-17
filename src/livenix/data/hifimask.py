"""CASIA-SURF HiFiMask dataset loader for Livenix.

Maps HiFiMask's mask materials to the Livenix 3-class scheme
(real / print_spoof / replay_spoof).

Mapping rationale (Phase 1 design decision — see CLAIMS.md):
  bonafide    -> 0 (real)
  real        -> 0 (alias used in some release variants)
  transparent -> 1 (print_spoof — physical mask, same physics as print)
  plaster     -> 1 (print_spoof)
  resin       -> 1 (print_spoof)
  latex       -> 1 (print_spoof)
  silicone    -> 1 (print_spoof — SILENT silicone training, not publicly claimed)
  paper       -> 1 (print_spoof)

All mask materials map to label 1 (print_spoof) because:
  - Physical objects placed in front of the camera lack real-skin specular
    response, subsurface scattering, and micro-texture (same physics class
    as print attacks).
  - Mapping to replay_spoof (label 2) would be wrong — no screen involved.
  - A 4th "mask" class is out of scope for Phase 1.

The ORIGINAL material name is preserved as attack_subtype so QA Gate 3 can
report per-material confusion separately.

Phase 1 silent-silicone note: HiFiMask is 10% of the training mix.  We never
claim silicone-mask defense (see CLAIMS.md). The model gains silent resilience;
Phase 2 will either earn the public claim with bigger model + multi-frame, or
drop silicone from the mix.

Reference: Liu et al., "3D High-Fidelity Mask Face Presentation Attack
Detection Challenge," ICCVW 2021 / CASIA-SURF HiFiMask 2022 release.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


# Module-level mapping so it can be imported and inspected independently.
# Keys are LOWERCASE material strings. All mask materials map to label 1.
# Phase 1 design note: every mask material maps to print_spoof (label 1).
# We never claim silicone defense (see CLAIMS.md). Original material kept
# as `attack_subtype` for per-material QA Gate 3 reporting.
HIFIMASK_TO_LIVENIX_LABEL: dict[str, int] = {
    "bonafide":     0,
    "real":         0,   # alias some releases use
    "transparent":  1,
    "plaster":      1,
    "resin":        1,
    "latex":        1,
    "silicone":     1,
    "paper":        1,
}


class HiFiMaskDataset(Dataset):
    """HiFiMask 3D-mask dataset loader.

    All mask materials map to Livenix label 1 (print_spoof) for SILENT
    silicone training. Original material is preserved as `attack_subtype`.

    Args:
        root: HiFiMask root containing images/.
        split: 'train' | 'test' (optional — applied when protocol files
            are present; ignored otherwise).
        protocol: 'protocol_1' | 'protocol_2' | 'protocol_3' (optional —
            HiFiMask defines 3 cross-domain protocols). Default: None
            (no protocol filtering; all samples).
        transform: optional PIL -> Tensor.

    Returns from __getitem__:
        (image, livenix_label, attack_subtype_str)
        where attack_subtype_str is the lowercased material name
        ('bonafide', 'silicone', 'plaster', ...).
    """

    ALLOWED_SPLITS = {"train", "test"}
    ALLOWED_PROTOCOLS = {"protocol_1", "protocol_2", "protocol_3"}

    def __init__(
        self,
        root: str | Path,
        split: str | None = None,
        protocol: str | None = None,
        transform: Callable | None = None,
    ) -> None:
        super().__init__()
        if split is not None and split not in self.ALLOWED_SPLITS:
            raise ValueError(
                f"split must be one of {self.ALLOWED_SPLITS} or None, got {split!r}"
            )
        if protocol is not None and protocol not in self.ALLOWED_PROTOCOLS:
            raise ValueError(
                f"protocol must be one of {self.ALLOWED_PROTOCOLS} or None, got {protocol!r}"
            )
        self.root = Path(root)
        self.split = split
        self.protocol = protocol
        self.transform = transform
        self.samples: list[tuple[Path, int, str]] = self._discover_samples()

    def _load_protocol_paths(self) -> set[str] | None:
        """Load the set of relative image paths from a protocol split file.

        File location: <root>/protocols/<protocol>_<split>.txt
        Each line: <relative_image_path> <label>

        Returns None if:
          - protocol or split is not set
          - protocol file does not exist

        Returns a set of relative path strings (as written in the file) when
        the file exists and can be parsed.
        """
        if self.protocol is None or self.split is None:
            return None

        proto_file = self.root / "protocols" / f"{self.protocol}_{self.split}.txt"
        if not proto_file.is_file():
            return None

        allowed: set[str] = set()
        try:
            for line in proto_file.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                # Lines are: <relative_path> <label>  (space-separated)
                parts = line.split()
                if parts:
                    allowed.add(parts[0])
        except OSError:
            return None
        return allowed

    def _discover_samples(self) -> list[tuple[Path, int, str]]:
        """Walk images/<session>/<subject>/<material>/*.png.

        For each image:
          - extract material from the parent directory name
          - lowercase + look up in HIFIMASK_TO_LIVENIX_LABEL
          - skip if unknown material
          - filter by protocol file if both protocol AND split set

        Robustness:
          - missing root -> empty
          - missing images/ -> empty
          - unknown material directory -> skipped
          - protocol file missing -> no filtering (include all)
        """
        images_dir = self.root / "images"
        if not images_dir.is_dir():
            return []

        # Load protocol allowed-paths set if applicable; None = no filter
        allowed_paths = self._load_protocol_paths()

        samples: list[tuple[Path, int, str]] = []

        # Walk: images/<session_id>/<subject_id>/<material>/*.png
        for session_dir in sorted(images_dir.iterdir()):
            if not session_dir.is_dir():
                continue
            for subject_dir in sorted(session_dir.iterdir()):
                if not subject_dir.is_dir():
                    continue
                for material_dir in sorted(subject_dir.iterdir()):
                    if not material_dir.is_dir():
                        continue
                    material = material_dir.name.lower()
                    livenix_label = HIFIMASK_TO_LIVENIX_LABEL.get(material)
                    if livenix_label is None:
                        # Unknown material — skip silently
                        continue
                    for img_path in sorted(material_dir.glob("*.png")):
                        if allowed_paths is not None:
                            # Protocol filter: check relative path from root
                            rel = img_path.relative_to(self.root).as_posix()
                            if rel not in allowed_paths:
                                continue
                        samples.append((img_path, livenix_label, material))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(
                f"index {idx} out of range for dataset of size {len(self.samples)}"
            )
        img_path, label, attack_subtype_str = self.samples[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        return im, label, attack_subtype_str


def write_fake_hifimask_sample(
    root: Path,
    session_id: str,
    subject_id: str,
    material: str,
    filename_stem: str = "frame_0",
    image_size: tuple[int, int] = (32, 32),
    protocol: str | None = None,
    split: str | None = None,
) -> Path:
    """Write a fake HiFiMask frame. Returns path to the written .png.

    Creates a PNG at:
        <root>/images/<session_id>/<subject_id>/<material>/<filename_stem>.png

    If protocol and split are both set, also appends a row to:
        <root>/protocols/<protocol>_<split>.txt
    The row format matches protocol files:
        images/<session_id>/<subject_id>/<material>/<filename_stem>.png <label>

    Args:
        root: HiFiMask root directory (will contain images/ and optionally
            protocols/).
        session_id: Session identifier (e.g. "session_001").
        subject_id: Subject identifier (e.g. "subject_001").
        material: Material/category name (e.g. "bonafide", "silicone").
        filename_stem: Filename without extension (e.g. "frame_0").
        image_size: (width, height) for the generated PNG.
        protocol: If set ('protocol_1', 'protocol_2', 'protocol_3'), add an
            entry to the corresponding protocol split file.
        split: If set ('train', 'test'), combined with protocol to determine
            the protocol file name.

    Returns:
        Path to the written .png file.
    """
    img_dir = root / "images" / session_id / subject_id / material
    img_dir.mkdir(parents=True, exist_ok=True)

    img_path = img_dir / f"{filename_stem}.png"

    # Write a minimal RGB PNG
    img = Image.new("RGB", image_size, color=(128, 64, 192))
    img.save(img_path)

    if protocol is not None and split is not None:
        protocols_dir = root / "protocols"
        protocols_dir.mkdir(parents=True, exist_ok=True)
        proto_file = protocols_dir / f"{protocol}_{split}.txt"

        # Determine the label for this material
        label = HIFIMASK_TO_LIVENIX_LABEL.get(material.lower(), 1)

        # Relative path from root (POSIX style)
        rel = img_path.relative_to(root).as_posix()
        with proto_file.open("a") as fh:
            fh.write(f"{rel} {label}\n")

    return img_path
