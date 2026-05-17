"""Identity-disjoint split + contamination audit utilities.

Used by QA Gate 12 (contamination audit) and by every dataset loader's
train/val/test split logic.

Design principle: identity NEVER overlaps across splits. CelebA-Spoof and
WMCA both have subject IDs in their on-disk paths; we use those as identities
in production. Tests use synthetic identity strings.

The contamination audit verifies the splits using a callable embedding
function (FaceNet in production; a deterministic mock in tests). This
abstraction keeps the audit testable without model downloads.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Generic, TypeVar

import torch
import torch.nn.functional as F


T = TypeVar("T")


@dataclass
class SplitResult(Generic[T]):
    train: list[T]
    val: list[T]
    test: list[T]


@dataclass
class ContaminationReport:
    pairs_flagged: int          # total pairs whose cosine sim >= threshold
    by_split: dict[str, int]    # "train_val", "train_test", "val_test" → count
    examples: list[tuple]       # up to 5 (sample_a, sample_b, similarity) tuples
    threshold: float


def make_identity_disjoint_split(
    samples: list[T],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    identity_fn: Callable[[T], str] = None,
    seed: int = 0,
) -> SplitResult[T]:
    """Split samples into train/val/test with NO identity overlap.

    Algorithm:
      1. Bucket samples by identity_fn(sample) → dict[identity, list[sample]]
      2. Shuffle the unique identities (seeded RNG)
      3. Walk identities; assign each ENTIRE BUCKET to train | val | test
         based on cumulative target fractions
      4. Final split has train_frac of identities (not samples — but for
         balanced datasets these are close)

    Args:
        samples: list of arbitrary samples
        train_frac: fraction of IDENTITIES in train (default 0.7)
        val_frac:   fraction of IDENTITIES in val (default 0.15)
                    (test gets the remainder = 1 - train_frac - val_frac)
        identity_fn: callable mapping sample -> identity string. REQUIRED.
        seed: RNG seed for reproducibility

    Returns:
        SplitResult with .train / .val / .test lists.

    Raises:
        ValueError if identity_fn is None
        ValueError if train_frac + val_frac >= 1.0
    """
    if identity_fn is None:
        raise ValueError("identity_fn is required and must not be None")
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac ({train_frac}) + val_frac ({val_frac}) must be < 1.0, "
            f"got {train_frac + val_frac}"
        )

    if not samples:
        return SplitResult(train=[], val=[], test=[])

    # 1. Bucket samples by identity
    buckets: dict[str, list] = {}
    for sample in samples:
        identity = identity_fn(sample)
        buckets.setdefault(identity, []).append(sample)

    # 2. Shuffle the unique identities with seeded RNG
    identities = sorted(buckets.keys())  # sort first for determinism
    rng = random.Random(seed)
    rng.shuffle(identities)

    n_identities = len(identities)
    n_train = math.floor(train_frac * n_identities)
    n_val = math.floor(val_frac * n_identities)
    # test gets the remainder (at least 1 if possible)
    # Ensure we don't exceed n_identities
    if n_train + n_val >= n_identities:
        # Squeeze val down if needed to leave at least some test
        n_val = max(0, n_identities - n_train - 1)

    train_ids = identities[:n_train]
    val_ids = identities[n_train:n_train + n_val]
    test_ids = identities[n_train + n_val:]

    train: list = []
    val: list = []
    test: list = []

    for identity in train_ids:
        train.extend(buckets[identity])
    for identity in val_ids:
        val.extend(buckets[identity])
    for identity in test_ids:
        test.extend(buckets[identity])

    return SplitResult(train=train, val=val, test=test)


def audit_split_contamination(
    train: list[T],
    val: list[T],
    test: list[T],
    embedding_fn: Callable[[T], torch.Tensor],
    threshold: float = 0.6,
    max_examples: int = 5,
) -> ContaminationReport:
    """Audit for identity leakage across splits via embedding similarity.

    For each cross-split pair (train×val, train×test, val×test), compute
    cosine similarity between every sample-pair's embeddings. Flag any
    pair with sim >= threshold as a potential identity overlap.

    Args:
        train, val, test: split sample lists
        embedding_fn: callable mapping sample -> (D,) torch.Tensor. The
            function is responsible for L2-normalization or not — we
            normalize internally. In production: wrap a FaceNet model.
            In tests: mock that returns a deterministic embedding per ID.
        threshold: cosine similarity threshold (0.6 catches most same-ID
            pairs; lower = stricter)
        max_examples: how many flagged pairs to surface in the report

    Returns:
        ContaminationReport with counts and example flagged pairs.
    """
    # Vectorize: compute embeddings for all samples in each split as one
    # tensor, then compute split_a @ split_b.T for each pair, then mask.

    def _embed_split(split_samples: list) -> torch.Tensor | None:
        if not split_samples:
            return None
        embeddings = [embedding_fn(s) for s in split_samples]
        return torch.stack(embeddings)  # (N, D)

    train_emb = _embed_split(train)
    val_emb = _embed_split(val)
    test_emb = _embed_split(test)

    by_split: dict[str, int] = {"train_val": 0, "train_test": 0, "val_test": 0}
    all_examples: list[tuple] = []

    pairs = [
        ("train_val", train, train_emb, val, val_emb),
        ("train_test", train, train_emb, test, test_emb),
        ("val_test", val, val_emb, test, test_emb),
    ]

    for pair_name, samples_a, emb_a, samples_b, emb_b in pairs:
        if emb_a is None or emb_b is None:
            continue

        sim_matrix = cosine_sim_matrix(emb_a, emb_b)  # (N, M)
        mask = sim_matrix >= threshold
        count = int(mask.sum().item())
        by_split[pair_name] = count

        if count > 0 and len(all_examples) < max_examples:
            # Collect flagged (i, j) pairs
            flagged_indices = mask.nonzero(as_tuple=False)  # (K, 2)
            for idx in flagged_indices:
                if len(all_examples) >= max_examples:
                    break
                i, j = int(idx[0].item()), int(idx[1].item())
                sim_val = float(sim_matrix[i, j].item())
                all_examples.append((samples_a[i], samples_b[j], sim_val))

    total_flagged = sum(by_split.values())

    return ContaminationReport(
        pairs_flagged=total_flagged,
        by_split=by_split,
        examples=all_examples[:max_examples],
        threshold=threshold,
    )


def cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute the cosine similarity matrix between two batches of vectors.

    Args:
        a: (N, D), b: (M, D) — float tensors
    Returns:
        (N, M) tensor of cosine similarities in [-1, 1]
    """
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return a @ b.T
