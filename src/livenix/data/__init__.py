"""Data loaders and preprocessing. Implemented in Week 2."""

from livenix.data.celeba_spoof import CelebASpoofDataset, CelebASpoofCropDataset, CELEBA_SPOOF_TO_LIVENIX_LABEL
from livenix.data.deepfake_screen import (
    DeepfakeScreenDataset,
    DEEPFAKE_SCREEN_LABEL,
    DEEPFAKE_SCREEN_ATTACK_TYPE,
)
from livenix.data.wmca import WMCADataset, WMCA_TO_LIVENIX_LABEL
from livenix.data.hifimask import HiFiMaskDataset, HIFIMASK_TO_LIVENIX_LABEL
from livenix.data.transforms import train_transforms, eval_transforms
from livenix.data.splits import (
    SplitResult,
    ContaminationReport,
    make_identity_disjoint_split,
    audit_split_contamination,
    cosine_sim_matrix,
)

__all__ = [
    "CelebASpoofDataset", "CelebASpoofCropDataset", "CELEBA_SPOOF_TO_LIVENIX_LABEL",
    "DeepfakeScreenDataset", "DEEPFAKE_SCREEN_LABEL", "DEEPFAKE_SCREEN_ATTACK_TYPE",
    "WMCADataset", "WMCA_TO_LIVENIX_LABEL",
    "HiFiMaskDataset", "HIFIMASK_TO_LIVENIX_LABEL",
    "train_transforms", "eval_transforms",
    "SplitResult",
    "ContaminationReport",
    "make_identity_disjoint_split",
    "audit_split_contamination",
    "cosine_sim_matrix",
]
