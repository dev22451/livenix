"""Deepfake-on-screen dataset loader (Phase 1 stub).

Loads in-house collected frames of deepfake video output shown on a screen
and re-captured by another camera. All samples are labeled as label 2
(replay_spoof) since the model detects the SCREEN-REPLAY physics (moiré,
specular reflection, color gamut) — not the deepfake semantic content.

Data collection procedure (for whoever stages the directory):
  1. Run a deepfake video tool (e.g. DeepFaceLive, Roop) producing a real-time
     face-swap output stream.
  2. Display that output on a phone or tablet screen.
  3. Use a SECOND camera (laptop/phone) to capture frames of the screen.
  4. Extract frames at 1-2 FPS using ffmpeg.
  5. Save .png frames into the data root directory (flat structure OK).

For Phase 1 prototype we do not need a large set; a few hundred frames is
enough to teach the model that deepfake-on-screen still triggers the
replay-detection pathway.

Layout expected:
  <root>/
    *.png             (any flat structure; subdirs allowed but not required)

Empty directory → empty dataset (no crash).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch.utils.data
from PIL import Image


# All samples in this loader are labeled replay_spoof. We don't have a
# mapping dict because there's no per-sample type information — all
# samples are by construction screen-replay-of-deepfake.
DEEPFAKE_SCREEN_LABEL = 2  # replay_spoof
DEEPFAKE_SCREEN_ATTACK_TYPE = "deepfake_on_screen"


class DeepfakeScreenDataset(torch.utils.data.Dataset):
    """Deepfake-on-screen replay loader (Phase 1 stub).

    Args:
        root: Path to a directory containing .png frames captured from a
            deepfake-displayed screen. Missing or empty → len == 0.
        transform: optional PIL → Tensor.

    Returns from __getitem__:
        (image, livenix_label, attack_type_str)
        where livenix_label == 2 (replay_spoof) and
        attack_type_str == "deepfake_on_screen" for all samples.
    """

    EXTENSIONS = (".png", ".jpg", ".jpeg")

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
    ):
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.samples: list[Path] = self._discover_samples()

    def _discover_samples(self) -> list[Path]:
        """Recursively find image files under root.

        Returns an empty list if root does not exist or contains no images.
        """
        if not self.root.exists() or not self.root.is_dir():
            return []
        paths: list[Path] = []
        for ext in self.EXTENSIONS:
            paths.extend(self.root.rglob(f"*{ext}"))
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path = self.samples[idx]
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        if self.transform is not None:
            im = self.transform(im)
        return im, DEEPFAKE_SCREEN_LABEL, DEEPFAKE_SCREEN_ATTACK_TYPE
