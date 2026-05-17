"""
Livenix model heads.

- MainHead: 3-class classifier (real / print_spoof / replay_spoof). Used at
  both training and inference time. This is the ONLY head present in exported
  artifacts (ONNX/TFLite/CoreML).

- FFTHead, SigLIPDistillHead, PatchContrastiveHead: training-only auxiliary
  heads. Implemented in Week 2. Raise NotImplementedError until then.

Cosine-margin / ArcFace-style decision tightening lives in the LOSS function,
not in the head — keeps the head a plain linear so it exports cleanly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MainHead(nn.Module):
    """3-class classifier head.

    Args:
        in_features: number of input features after global pooling.
        num_classes: 3 (real, print_spoof, replay_spoof).
        dropout: optional dropout before the linear layer.
    """

    NUM_CLASSES = 3

    def __init__(self, in_features: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, self.NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x))


class FFTHead(nn.Module):
    """Auxiliary FFT-magnitude prediction head. Training-only, stripped at export.

    Purpose
    -------
    Screen-replay attacks introduce moiré patterns — spatial-frequency
    artifacts that arise from the interference between the camera's Bayer
    sensor grid and the display's pixel grid. These patterns are strong cues
    in frequency space but weak in the spatial domain. By supervising the
    backbone's feature map to predict the log-magnitude spectrum of the input
    image, we encourage the backbone to encode frequency-domain information
    that is otherwise under-represented in a purely spatial classification
    pipeline.

    Design (follows Yu et al. CDCN / Minivision FTGenerator pattern,
    clean reimplementation):
    - Three Conv-BN-ReLU blocks: in_channels → hidden → hidden//2 → 1
    - All 3×3 convolutions with padding=1 (same spatial size)
    - Final sigmoid activation: output in [0, 1] to match BCE target
    - Input feature map is adaptively pooled to `target_size` before the
      convolution stack so the head is resolution-agnostic.

    Training coupling
    -----------------
    During training the caller:
      1. Feeds the backbone feature map into ``forward(feat_map)`` to get
         the predicted spectrum map ``pred``.
      2. Calls ``FFTHead.compute_target(image)`` on the raw input batch to
         get the normalized ground-truth spectrum ``target``.
      3. Calls ``head.loss(pred, target)`` to compute scalar BCE.
      4. Scales the result by 0.2 before adding to the combined loss (see
         ARCHITECTURE.md, Loss section).

    IMPORTANT: This module is NOT referenced by LivenixModel.forward(). It is
    instantiated and used only by the training loop. The exported ONNX graph
    therefore contains zero FFT ops, preserving cross-platform portability
    across ONNX Runtime, TFLite, CoreML, and NCNN.

    Args:
        in_channels: Number of channels in the backbone feature map (e.g. 960
            for MobileNetV4-Conv-Small final feature map).
        hidden: Width of the first hidden conv layer. The second hidden layer
            uses ``hidden // 2`` channels. Default: 128.
        target_size: Spatial (H, W) of the output prediction and the computed
            target spectrum. Kept small (default 4×4) to stay computationally
            cheap and match the coarse frequency resolution sufficient for
            moiré detection. Default: (4, 4).
    """

    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        target_size: tuple[int, int] = (4, 4),
    ) -> None:
        super().__init__()
        self.target_size = target_size

        # Three Conv-BN-ReLU blocks:
        #   Block 1: in_channels → hidden
        #   Block 2: hidden → hidden // 2
        #   Block 3: hidden // 2 → 1  (final, sigmoid instead of ReLU)
        mid = hidden // 2

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden, mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        # Final block: 1-channel output + sigmoid to bound to [0, 1]
        self.block3 = nn.Sequential(
            nn.Conv2d(mid, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_map: Tensor) -> Tensor:
        """Predict a magnitude-spectrum-like map from a backbone feature map.

        Args:
            feat_map: Backbone feature map of shape (B, C, H, W).

        Returns:
            Predicted spectrum map of shape (B, 1, target_size[0], target_size[1]),
            with all values in [0, 1].
        """
        # Pool to target_size first so the conv stack operates at a fixed
        # spatial resolution regardless of the input feature map size.
        x = F.adaptive_avg_pool2d(feat_map, self.target_size)
        x = self.block1(x)
        x = self.block2(x)
        return self.block3(x)

    @staticmethod
    def compute_target(
        image: Tensor,
        target_size: tuple[int, int] = (4, 4),
    ) -> Tensor:
        """Derive a normalized log-magnitude FFT spectrum from the input image.

        The target is computed entirely in Python/PyTorch and is used only
        during training. No FFT ops enter the exported inference graph.

        Pipeline:
          1. Grayscale via channel mean → (B, 1, H, W).
          2. ``torch.fft.rfft2`` (real-input 2D FFT) → complex (B, 1, H, W//2+1).
          3. ``torch.abs(...)`` → real magnitude.
          4. ``torch.log1p(magnitude)`` to compress the very wide dynamic range
             (bright DC component vs. weak high-frequency tails).
          5. Adaptive average pool to ``target_size``.
          6. Per-sample min-max normalization to [0, 1] so the BCE target is
             well-conditioned independent of absolute magnitude scale.

        Args:
            image: Input batch of shape (B, 3, H, W), RGB, normalized to
                ImageNet mean/std (floats, not uint8).
            target_size: Spatial size of the returned target. Must match the
                ``target_size`` used when constructing the FFTHead.

        Returns:
            Spectrum target of shape (B, 1, target_size[0], target_size[1]),
            per-sample normalized to [0, 1].  Gradients are not propagated
            (the target is treated as a fixed label).
        """
        with torch.no_grad():
            # Step 1: grayscale — mean over the channel dimension
            # (B, 3, H, W) → (B, 1, H, W)
            gray = image.mean(dim=1, keepdim=True)

            # Step 2: 2D real-input FFT.  rfft2 exploits conjugate symmetry
            # of real inputs, returning only the non-redundant half of the
            # frequency plane: shape (B, 1, H, W//2 + 1), complex64.
            spectrum = torch.fft.rfft2(gray)

            # Step 3: magnitude
            mag = torch.abs(spectrum)  # (B, 1, H, W//2 + 1)

            # Step 4: log-compress — log(1 + |X|) keeps the gradient-friendly
            # [0, inf) range while dramatically compressing the DC spike.
            log_mag = torch.log1p(mag)

            # Step 5: pool to target_size
            pooled = F.adaptive_avg_pool2d(log_mag, target_size)
            # pooled: (B, 1, target_size[0], target_size[1])

            # Step 6: per-sample min-max normalization to [0, 1].
            # Reshape to (B, -1) for clean vectorized min/max computation.
            B = pooled.shape[0]
            flat = pooled.view(B, -1)
            mn = flat.min(dim=1, keepdim=True).values  # (B, 1)
            mx = flat.max(dim=1, keepdim=True).values  # (B, 1)
            # Avoid division by zero if the spectrum is perfectly flat (edge case).
            denom = (mx - mn).clamp(min=1e-8)
            normalized = (flat - mn) / denom
            return normalized.view_as(pooled)

    def loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """Binary cross-entropy between predicted and target spectrum maps.

        Both tensors are already in [0, 1] (pred from sigmoid, target from
        min-max normalization), so we use the numerically stable
        ``F.binary_cross_entropy`` directly.

        The caller is responsible for weighting this loss (0.2× per the
        CombinedLoss spec in ARCHITECTURE.md) before adding it to the total.

        Args:
            pred: Predicted map, shape (B, 1, H, W), values in [0, 1].
            target: Ground-truth map from ``compute_target``, same shape,
                values in [0, 1].

        Returns:
            Scalar tensor (mean BCE over the batch).
        """
        return F.binary_cross_entropy(pred, target)


class SigLIPDistillHead(nn.Module):
    """1x1 projection from backbone feature dim → SigLIP teacher embedding dim.

    Purpose
    -------
    SigLIP-So400m (Google, Apache 2.0) has been pretrained on 4B image-text
    pairs and encodes rich semantic priors — especially useful for distinguishing
    naturalistic human faces from printed/replayed/AIGC variants that deviate
    subtly from scene statistics.

    By supervising the backbone's spatial feature map to match the frozen
    SigLIP teacher's image embedding, we transfer these priors without
    increasing inference cost: the teacher is run only during training, and
    this module is stripped at export.

    Design
    ------
    A single 1×1 convolution maps from the backbone's channel count to the
    teacher's embedding dimension. After projection the spatial grid is
    averaged to a single vector that is L2-normalized, mirroring the
    normalized-embedding style used in SigLIP's own contrastive training.

    Loss is MSE between the normalized student vector and the L2-normalized
    teacher embedding. MSE on normalized vectors is proportional to
    ``2 - 2 * cos(θ)``, so it effectively minimizes the cosine distance while
    remaining differentiable and numerically stable — no temperature
    hyper-parameter needed compared to InfoNCE.  The caller weights this loss
    at 0.5× before adding it to the combined objective (see ARCHITECTURE.md,
    Loss section).

    Teacher integration
    -------------------
    The actual SigLIP-So400m teacher (1.6 GB HuggingFace checkpoint) is
    wired in by the training loop in **T2.11**. The trainer runs the teacher
    under ``torch.no_grad()`` in bf16 to produce ``teacher_emb: Tensor``
    of shape ``(B, teacher_dim)``, then calls ``head.loss(student_emb,
    teacher_emb)``.  This module does **not** load, import, or reference the
    teacher in any way — keeping T2.2 dependency-free and CI-fast.

    Stripped at export
    ------------------
    ``SigLIPDistillHead`` is instantiated only by the training pipeline.
    ``LivenixModel.forward()`` does not reference it; the ONNX/TFLite/CoreML
    graphs therefore contain zero teacher-distillation ops.

    Args:
        in_channels: backbone feature-map channel count (e.g. 960 for
            MobileNetV4-Conv-Small final stage).
        teacher_dim: teacher embedding dim. Default 1152 matches
            SigLIP-So400m-patch14-384.
    """

    TEACHER_NAME = "google/siglip-so400m-patch14-384"  # documented, not loaded here
    DEFAULT_TEACHER_DIM = 1152

    def __init__(self, in_channels: int, teacher_dim: int = 1152) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, teacher_dim, kernel_size=1, bias=False)
        self.teacher_dim = teacher_dim

    def forward(self, feat_map: Tensor) -> Tensor:
        """Project (B, C, H, W) backbone features → (B, teacher_dim) student embedding.

        Pipeline:
          1. 1×1 conv maps channel dim: C → teacher_dim, spatial unchanged.
          2. Spatial mean pools the H×W grid to a single vector per sample.
          3. L2 normalization aligns the student vector with the teacher's
             normalized embedding space, making the MSE loss cosine-aware.

        Args:
            feat_map: Backbone feature map, shape (B, in_channels, H, W).

        Returns:
            L2-normalized student embedding, shape (B, teacher_dim).
            Norm is ≈ 1.0 per row (up to floating-point precision).
        """
        x = self.projection(feat_map)   # (B, teacher_dim, H, W)
        x = x.mean(dim=(2, 3))          # (B, teacher_dim)
        return F.normalize(x, dim=-1)   # L2-normalized for cosine-friendly loss

    def loss(self, student_emb: Tensor, teacher_emb: Tensor) -> Tensor:
        """MSE between L2-normalized student and teacher embeddings.

        Both embeddings are normalized before the MSE is computed, so the
        loss value lies in ``[0, 4]`` (0 = identical direction, 4 = antipodal).
        In practice it converges well into the ``[0, 0.1]`` range.

        The trainer (T2.11) is responsible for:
          - Running the frozen SigLIP teacher under ``torch.no_grad()`` to
            obtain ``teacher_emb``.
          - Scaling the returned scalar by 0.5 before adding to the combined
            loss (ARCHITECTURE.md Loss section).

        This module does NOT load or run the teacher itself.

        Args:
            student_emb: Output of ``forward()``, shape (B, teacher_dim),
                already L2-normalized.
            teacher_emb: Teacher image embedding from T2.11, shape
                (B, teacher_dim). May be unnormalized — this method applies
                its own normalization so the caller does not need to pre-normalize.

        Returns:
            Scalar MSE loss tensor with gradient flowing back through
            ``student_emb`` into ``self.projection``.
        """
        teacher_norm = F.normalize(teacher_emb, dim=-1)
        return F.mse_loss(student_emb, teacher_norm)


class PatchContrastiveHead(nn.Module):
    """Auxiliary patch-wise contrastive (NT-Xent) head. Training-only."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError("PatchContrastiveHead is implemented in Week 2.")
