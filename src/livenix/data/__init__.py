"""Data loaders and preprocessing. Implemented in Week 2."""

from livenix.data.celeba_spoof import CelebASpoofDataset, CELEBA_SPOOF_TO_LIVENIX_LABEL
from livenix.data.wmca import WMCADataset, WMCA_TO_LIVENIX_LABEL

__all__ = [
    "CelebASpoofDataset", "CELEBA_SPOOF_TO_LIVENIX_LABEL",
    "WMCADataset", "WMCA_TO_LIVENIX_LABEL",
]
