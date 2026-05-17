"""
Livenix training losses.

This module provides the four loss components used by the training stack:

  - FocalLoss      — multi-class focal CE for the 3-class main head (Lin et al. 2017).
  - AsymmetricLoss — decoupled positive/negative focal loss (Ben-Baruch et al. 2021).
  - CosineMarginLoss — ArcFace-style cosine-margin softmax (Deng et al. 2019).
  - CombinedLoss   — orchestrator: main loss + weighted aux losses.

Aux head losses (FFT BCE, SigLIP MSE, patch NT-Xent) live on their respective head
modules (``FFTHead.loss``, ``SigLIPDistillHead.loss``, ``PatchContrastiveHead.loss``).
The trainer computes those and passes them as pre-computed scalars to ``CombinedLoss``,
keeping this module fully decoupled from the head implementations.

Combined-loss weights (per ARCHITECTURE.md, Loss section):
    L = main + 0.5 * siglip_distill + 0.2 * fft + 0.3 * patch_contrastive
"""

from __future__ import annotations

import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FocalLoss(nn.Module):
    """Multi-class focal loss for the 3-class main head (Lin et al., 2017).

        FL(p_t) = - α_t * (1 - p_t) ** γ * log(p_t)

    Reduces the relative weight of well-classified examples, concentrating
    training on hard examples.  Useful for the real/spoof class imbalance
    typical in face anti-spoofing datasets.

    Args:
        gamma: Focusing parameter γ ≥ 0.  γ = 0 recovers cross-entropy.
            Default: 2.0 (as in Lin et al. 2017).
        alpha: Per-class weights of shape ``(num_classes,)``.  If ``None``
            all classes are weighted equally (no reweighting).  Pass a Tensor
            on the correct device; this module does **not** move it.

    Inputs:
        logits:  ``(B, C)`` raw (pre-softmax) class scores.
        targets: ``(B,)`` integer class indices in ``[0, C)``.

    Output:
        Scalar mean focal loss over the batch.

    References:
        Lin, T.-Y. et al. (2017). "Focal Loss for Dense Object Detection."
        ICCV 2017. https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Union[Tensor, None] = None,
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError(f"gamma must be >= 0, got {gamma}")
        self.gamma = gamma
        # Store as buffer so it moves with the module (e.g. .cuda()) but is
        # not treated as a learnable parameter.
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha: Union[Tensor, None] = None  # type: ignore[assignment]

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute focal loss.

        Args:
            logits:  ``(B, C)`` pre-softmax class scores.
            targets: ``(B,)`` long integer class indices.

        Returns:
            Scalar mean focal loss.
        """
        # log_softmax → numerically stable log-probabilities (B, C)
        log_probs = F.log_softmax(logits, dim=1)

        # Gather the log-probability of the ground-truth class for each sample.
        # log_pt: (B,)
        log_pt = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)

        # p_t = exp(log_pt) — probability of the correct class
        pt = log_pt.exp()

        # Focal modulation: (1 - p_t)^γ
        focal_weight = (1.0 - pt) ** self.gamma

        # Apply per-class alpha weights if provided
        if self.alpha is not None:
            # alpha_t: (B,) — weight for each sample's target class
            alpha_t = self.alpha.gather(0, targets)
            loss = -alpha_t * focal_weight * log_pt
        else:
            loss = -focal_weight * log_pt

        return loss.mean()


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for imbalanced classification (Ben-Baruch et al., 2021).

    Decouples the focusing parameter into separate values for positives and
    negatives, allowing stronger down-weighting of easy negative examples.

    For multi-class problems the loss is applied per-class in a one-vs-rest
    manner after softmax, mirroring the original multi-label formulation.

    For each sample *i* and class *c*:
        p_c  = softmax(logits)[i, c]
        y_c  = 1 if targets[i] == c else 0    (one-hot)
        p_c' = max(0, p_c - clip)             (only for negative y_c = 0)

        L_+ = -(1 - p_c)^{γ_pos} * log(p_c)              [positive]
        L_- = -(p_c')^{γ_neg}    * log(1 - p_c')          [negative]
        L   = y_c * L_+  +  (1 - y_c) * L_-

    The final scalar is the mean over all (sample, class) pairs.

    Args:
        gamma_neg: Focusing parameter for hard negatives.  Default: 4.0.
        gamma_pos: Focusing parameter for positives.  Default: 1.0.
        clip:      Probability shift applied to negative predictions before
            computing the negative loss.  Values close to 0 are shifted
            to 0, effectively discarding very easy negatives.  Default: 0.05.

    Inputs:
        logits:  ``(B, C)`` raw (pre-softmax) class scores.
        targets: ``(B,)`` integer class indices in ``[0, C)``.

    Output:
        Scalar mean asymmetric loss over the batch.

    References:
        Ben-Baruch, E. et al. (2021). "Asymmetric Loss For Multi-Label
        Classification." ICCV 2021. https://arxiv.org/abs/2009.14119
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute asymmetric loss.

        Args:
            logits:  ``(B, C)`` pre-softmax class scores.
            targets: ``(B,)`` long integer class indices.

        Returns:
            Scalar mean asymmetric loss.
        """
        B, C = logits.shape

        # Probabilities via softmax: (B, C), values in (0, 1)
        probs = F.softmax(logits, dim=1)

        # One-hot encoding of targets: (B, C)
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        # --- Positive branch (y = 1) ---
        # p_pos: probabilities of true classes, clamp for numerical safety
        p_pos = probs.clamp(min=1e-8)
        focal_pos = (1.0 - p_pos) ** self.gamma_pos
        loss_pos = -focal_pos * torch.log(p_pos)  # (B, C)

        # --- Negative branch (y = 0) ---
        # Clip negative probabilities: subtract clip and clamp to [0, 1]
        p_neg_shifted = (probs - self.clip).clamp(min=0.0, max=1.0)
        # Probability of "not the class" after shifting: 1 - p_neg_shifted
        p_neg_complement = (1.0 - p_neg_shifted).clamp(min=1e-8)
        focal_neg = (p_neg_shifted) ** self.gamma_neg
        loss_neg = -focal_neg * torch.log(p_neg_complement)  # (B, C)

        # Combine: positive samples use L_+, negative samples use L_-
        loss = one_hot * loss_pos + (1.0 - one_hot) * loss_neg  # (B, C)

        return loss.mean()


class CosineMarginLoss(nn.Module):
    """ArcFace-style cosine-margin softmax loss (Deng et al., 2019).

    Adds a fixed angular margin ``m`` to the cosine similarity of the target
    class before computing cross-entropy with temperature scaling ``s``:

        cos(θ_y + m)  for the target class y
        cos(θ_j)      for all other classes j ≠ y

        L = CE( s * [cos(θ_1), ..., cos(θ_y + m), ..., cos(θ_C)] )

    This tightens the decision boundary without adding inference cost: the
    ``weight`` matrix replaces the plain ``nn.Linear`` in ``MainHead``.

    This module REPLACES the standard ``nn.Linear → CE`` pipeline for the
    main head.  It owns its own learnable weight matrix and accepts raw
    (un-normalized) feature vectors.

    Args:
        in_features: Feature dimension D (must match the backbone output dim
            passed to ``MainHead``).
        num_classes: Number of output classes C.  Default: 3 (real /
            print_spoof / replay_spoof).
        margin: Angular margin ``m`` in radians.  Default: 0.3.
        scale: Temperature scale ``s`` (hyperspherical radius).  Default: 30.0.

    Inputs:
        features: ``(B, D)`` un-normalized feature vectors.
        targets:  ``(B,)`` long integer class indices.

    Output:
        Scalar CE loss after margin perturbation.

    Note:
        This is an **alternative** to the plain ``MainHead`` CE path.
        ``CombinedLoss`` dispatches to this module when ``main_inputs``
        contains ``"features"`` and ``main_loss_fn`` is an instance of
        ``CosineMarginLoss``.

    References:
        Deng, J. et al. (2019). "ArcFace: Additive Angular Margin Loss for
        Deep Face Recognition." CVPR 2019. https://arxiv.org/abs/1801.07698
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int = 3,
        margin: float = 0.3,
        scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Learnable weight matrix: each row is the class prototype on the unit hypersphere.
        # Initialized with Kaiming uniform to break symmetry.
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Pre-compute and cache cos(m) and sin(m) for the margin addition.
        self.register_buffer("cos_m", torch.tensor(math.cos(margin)))
        self.register_buffer("sin_m", torch.tensor(math.sin(margin)))
        # Threshold: cos(π - m) — used to handle edge cases where θ + m > π
        self.register_buffer("threshold", torch.tensor(math.cos(math.pi - margin)))

    def forward(self, features: Tensor, targets: Tensor) -> Tensor:
        """Compute ArcFace cosine-margin loss.

        Args:
            features: ``(B, D)`` un-normalized feature vectors.
            targets:  ``(B,)`` long integer class indices.

        Returns:
            Scalar CE loss with angular margin applied to the target class.
        """
        # L2-normalize both features and weight prototypes → cosine similarity
        feat_norm = F.normalize(features, dim=1)           # (B, D)
        weight_norm = F.normalize(self.weight, dim=1)      # (C, D)

        # Cosine similarity matrix: cos(θ_{i,j})  shape (B, C)
        cos_theta = feat_norm @ weight_norm.t()
        # Clamp for numerical stability before arccos/trig operations
        cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # sin(θ) = sqrt(1 - cos²(θ))
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)

        # cos(θ + m) = cos(θ)·cos(m) - sin(θ)·sin(m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # For large θ (θ > π - m), cos(θ + m) would be larger than cos(θ),
        # which would flip the gradient.  Fall back to cos(θ) - sin(m)*m in
        # that regime (standard ArcFace stabilization).
        cos_theta_m = torch.where(
            cos_theta > self.threshold,
            cos_theta_m,
            cos_theta - self.sin_m * self.margin,
        )

        # One-hot mask: apply margin only to the target class
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Blend: target class gets cos(θ + m), others keep cos(θ)
        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta

        # Scale and compute CE
        logits = self.scale * logits
        return F.cross_entropy(logits, targets)


class CombinedLoss(nn.Module):
    """Orchestrator that combines the main classification loss with weighted aux losses.

    Per ARCHITECTURE.md (Loss section):

        L = main_loss + 0.5 * siglip_distill_loss
                      + 0.2 * fft_loss
                      + 0.3 * patch_contrastive_loss

    Aux losses are passed in as **pre-computed scalar tensors** — the trainer
    computes them by calling each auxiliary head's ``.loss()`` method, then
    passes the results here.  This keeps ``CombinedLoss`` fully decoupled from
    head implementations and makes it easy to ablate individual aux heads.

    Main-loss dispatch:
        - If ``main_inputs`` contains ``"features"`` *and* ``main_loss_fn`` is
          a ``CosineMarginLoss`` instance, ``features`` are passed directly to
          the module (ArcFace path).
        - Otherwise ``main_inputs["logits"]`` must be present and is passed to
          ``main_loss_fn(logits, targets)`` (FocalLoss / AsymmetricLoss path).

    Args:
        main_loss_fn: A ``nn.Module`` with signature
            ``forward(logits_or_features, targets) -> scalar``.
            Typically ``FocalLoss``, ``AsymmetricLoss``, or ``CosineMarginLoss``.
        weights: Optional override dict with keys ``"siglip"``, ``"fft"``,
            ``"patch"`` mapping to their scalar weights.  Any key not provided
            uses the default from ARCHITECTURE.md.

    Example::

        focal = FocalLoss(gamma=2.0)
        combined = CombinedLoss(focal)

        # Training step:
        main_out = model(images)               # logits (B, 3)
        siglip_loss = siglip_head.loss(...)    # scalar
        fft_loss    = fft_head.loss(...)       # scalar
        patch_loss  = patch_head.loss(...)     # scalar

        result = combined(
            main_inputs={"logits": main_out, "targets": labels},
            aux_losses={"siglip": siglip_loss, "fft": fft_loss, "patch": patch_loss},
        )
        result["total"].backward()
    """

    _DEFAULT_WEIGHTS = {"siglip": 0.5, "fft": 0.2, "patch": 0.3}

    def __init__(
        self,
        main_loss_fn: nn.Module,
        weights: Union[dict, None] = None,
    ) -> None:
        super().__init__()
        self.main_loss_fn = main_loss_fn
        # Merge provided weights with defaults (provided values override)
        self.weights = {**self._DEFAULT_WEIGHTS, **(weights or {})}

    def forward(
        self,
        main_inputs: dict,
        aux_losses: Union[dict, None] = None,
    ) -> dict:
        """Compute and combine all losses.

        Args:
            main_inputs: Dict that must contain ``"targets"`` and one of:
                - ``"logits"``  — for FocalLoss / AsymmetricLoss path.
                - ``"features"`` — for CosineMarginLoss path (dispatched only
                  when ``main_loss_fn`` is a ``CosineMarginLoss`` instance).
            aux_losses: Dict with any subset of keys ``"siglip"``, ``"fft"``,
                ``"patch"``, each mapping to a pre-computed scalar tensor.
                Missing keys contribute 0 to the total.  Pass ``None`` to
                skip all aux losses.

        Returns:
            Dict with keys:
                ``"main"``   — scalar main loss.
                ``"siglip"`` — weighted siglip loss (or 0).
                ``"fft"``    — weighted fft loss (or 0).
                ``"patch"``  — weighted patch contrastive loss (or 0).
                ``"total"``  — sum of all components.

        Raises:
            ValueError: If ``main_inputs`` lacks both ``"logits"`` and a
                valid ``"features"`` dispatch.
        """
        targets = main_inputs["targets"]

        # --- Main loss dispatch ---
        if "features" in main_inputs and isinstance(self.main_loss_fn, CosineMarginLoss):
            main_loss = self.main_loss_fn(main_inputs["features"], targets)
        elif "logits" in main_inputs:
            main_loss = self.main_loss_fn(main_inputs["logits"], targets)
        else:
            raise ValueError(
                "main_inputs must contain 'logits', or 'features' when "
                "main_loss_fn is CosineMarginLoss.  "
                f"Got keys: {list(main_inputs.keys())}"
            )

        # --- Aux losses ---
        aux = aux_losses or {}
        device = main_loss.device

        def _weighted(key: str) -> Tensor:
            if key in aux:
                return self.weights[key] * aux[key]
            return torch.tensor(0.0, device=device)

        siglip_term = _weighted("siglip")
        fft_term = _weighted("fft")
        patch_term = _weighted("patch")

        total = main_loss + siglip_term + fft_term + patch_term

        return {
            "main": main_loss,
            "siglip": siglip_term,
            "fft": fft_term,
            "patch": patch_term,
            "total": total,
        }
