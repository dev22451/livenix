"""Tests for identity-disjoint split + contamination audit (QA Gate 12)."""

from __future__ import annotations

import math

import pytest
import torch

from livenix.data.splits import (
    SplitResult,
    ContaminationReport,
    make_identity_disjoint_split,
    audit_split_contamination,
    cosine_sim_matrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_samples(n_ids: int, samples_per_id: int) -> list[str]:
    """Return a flat list of '<id>_<idx>' strings."""
    return [f"id_{i}_{j}" for i in range(n_ids) for j in range(samples_per_id)]


def _id_fn(sample: str) -> str:
    """Extract identity from '<id_idx>_<sample_idx>' strings."""
    # Format: "id_<N>_<M>" — identity is "id_<N>"
    parts = sample.rsplit("_", 1)
    return parts[0]


def _split_identities(samples: list[str]) -> set[str]:
    return {_id_fn(s) for s in samples}


def _make_mock_embedding_fn(dim: int = 16):
    """Return a mock embedding_fn.

    - Samples sharing the SAME identity produce the SAME embedding.
    - Samples with DIFFERENT identities produce orthogonal embeddings
      (one-hot into a dim-sized space, cycling through dims).
    """
    identity_to_vec: dict[str, torch.Tensor] = {}
    counter = [0]

    def embedding_fn(sample: str) -> torch.Tensor:
        identity = _id_fn(sample)
        if identity not in identity_to_vec:
            idx = counter[0] % dim
            counter[0] += 1
            vec = torch.zeros(dim)
            vec[idx] = 1.0
            identity_to_vec[identity] = vec
        return identity_to_vec[identity]

    return embedding_fn


# ---------------------------------------------------------------------------
# make_identity_disjoint_split tests
# ---------------------------------------------------------------------------

def test_make_split_basic():
    """100 samples, 20 IDs (5 per ID); split 0.6/0.2/0.2 — no ID overlap."""
    samples = _make_samples(n_ids=20, samples_per_id=5)
    result = make_identity_disjoint_split(
        samples, train_frac=0.6, val_frac=0.2, identity_fn=_id_fn, seed=42
    )

    train_ids = _split_identities(result.train)
    val_ids = _split_identities(result.val)
    test_ids = _split_identities(result.test)

    # No overlap
    assert train_ids & val_ids == set(), "train and val share identities"
    assert train_ids & test_ids == set(), "train and test share identities"
    assert val_ids & test_ids == set(), "val and test share identities"

    # Approximate fractions (±2 identities tolerance)
    assert abs(len(train_ids) - 12) <= 2
    assert abs(len(val_ids) - 4) <= 2

    # All samples accounted for
    all_out = result.train + result.val + result.test
    assert set(all_out) == set(samples)


def test_make_split_deterministic_with_seed():
    """Same seed → identical splits; different seed → different splits."""
    samples = _make_samples(n_ids=20, samples_per_id=5)

    r1 = make_identity_disjoint_split(samples, identity_fn=_id_fn, seed=7)
    r2 = make_identity_disjoint_split(samples, identity_fn=_id_fn, seed=7)
    r3 = make_identity_disjoint_split(samples, identity_fn=_id_fn, seed=99)

    assert r1.train == r2.train
    assert r1.val == r2.val
    assert r1.test == r2.test

    # Different seed should (very likely) produce different splits
    assert r1.train != r3.train or r1.val != r3.val or r1.test != r3.test


def test_make_split_raises_on_missing_identity_fn():
    """identity_fn=None → ValueError."""
    samples = _make_samples(n_ids=5, samples_per_id=3)
    with pytest.raises(ValueError, match="identity_fn"):
        make_identity_disjoint_split(samples, identity_fn=None)


def test_make_split_raises_on_invalid_fractions():
    """train_frac=0.8 + val_frac=0.3 → ValueError (sum >= 1.0)."""
    samples = _make_samples(n_ids=10, samples_per_id=3)
    with pytest.raises(ValueError):
        make_identity_disjoint_split(
            samples, train_frac=0.8, val_frac=0.3, identity_fn=_id_fn
        )


def test_make_split_handles_empty():
    """Empty sample list → SplitResult([], [], [])."""
    result = make_identity_disjoint_split([], identity_fn=_id_fn)
    assert result.train == []
    assert result.val == []
    assert result.test == []


def test_make_split_handles_single_identity():
    """All 10 samples share one identity — they all land in exactly ONE split."""
    samples = [f"id_0_{j}" for j in range(10)]
    result = make_identity_disjoint_split(
        samples, train_frac=0.7, val_frac=0.15, identity_fn=_id_fn, seed=0
    )

    non_empty = [s for s in [result.train, result.val, result.test] if s]
    assert len(non_empty) == 1, "Samples should all be in exactly one split"
    assert len(non_empty[0]) == 10


# ---------------------------------------------------------------------------
# cosine_sim_matrix tests
# ---------------------------------------------------------------------------

def test_cosine_sim_matrix_shape():
    """(3, 8) @ (5, 8).T → (3, 5)."""
    a = torch.randn(3, 8)
    b = torch.randn(5, 8)
    sim = cosine_sim_matrix(a, b)
    assert sim.shape == (3, 5)


def test_cosine_sim_matrix_self_similarity_is_one():
    """cosine_sim_matrix(x, x).diagonal() ≈ 1.0."""
    x = torch.randn(6, 16)
    sim = cosine_sim_matrix(x, x)
    diag = sim.diagonal()
    assert torch.allclose(diag, torch.ones(6), atol=1e-5)


# ---------------------------------------------------------------------------
# audit_split_contamination tests
# ---------------------------------------------------------------------------

def test_audit_no_contamination_when_split_disjoint():
    """Identity-disjoint split with orthogonal mock embeddings → 0 flagged."""
    samples = _make_samples(n_ids=10, samples_per_id=5)
    result = make_identity_disjoint_split(
        samples, train_frac=0.6, val_frac=0.2, identity_fn=_id_fn, seed=0
    )
    embedding_fn = _make_mock_embedding_fn(dim=64)

    report = audit_split_contamination(
        result.train, result.val, result.test,
        embedding_fn=embedding_fn,
        threshold=0.6,
    )

    assert report.pairs_flagged == 0
    assert report.by_split["train_val"] == 0
    assert report.by_split["train_test"] == 0
    assert report.by_split["val_test"] == 0
    assert report.examples == []


def test_audit_flags_contamination_when_seeded_overlap():
    """Manually overlap train/val identities → pairs_flagged > 0."""
    # Build train and val that SHARE "id_0"
    train = ["id_0_0", "id_0_1", "id_1_0", "id_1_1"]
    val = ["id_0_2", "id_2_0"]   # id_0 also in train → contamination
    test = ["id_3_0", "id_4_0"]

    embedding_fn = _make_mock_embedding_fn(dim=64)
    # Pre-register id_0 so both train and val get the SAME vector
    _ = embedding_fn("id_0_0")   # registers id_0

    report = audit_split_contamination(
        train, val, test,
        embedding_fn=embedding_fn,
        threshold=0.6,
    )

    assert report.pairs_flagged > 0
    assert report.by_split["train_val"] > 0
    assert len(report.examples) > 0


def test_audit_threshold_respected():
    """Embeddings with similarity exactly 0.5: threshold=0.6 → 0; threshold=0.4 → flagged."""
    # Two embeddings at 60-degree angle (cos=0.5)
    dim = 4
    a_vec = torch.tensor([1.0, 0.0, 0.0, 0.0])
    b_vec = torch.tensor([0.5, math.sqrt(0.75), 0.0, 0.0])  # cos(a,b)=0.5

    # train has one sample, val has one sample — always cross-split
    train = ["train_0"]
    val = ["val_0"]
    test: list[str] = []

    call_map = {"train_0": a_vec, "val_0": b_vec}
    embedding_fn = lambda s: call_map[s]  # noqa: E731

    # threshold=0.6 → 0.5 < 0.6 → not flagged
    report_strict = audit_split_contamination(
        train, val, test, embedding_fn=embedding_fn, threshold=0.6
    )
    assert report_strict.pairs_flagged == 0

    # threshold=0.4 → 0.5 >= 0.4 → flagged
    report_loose = audit_split_contamination(
        train, val, test, embedding_fn=embedding_fn, threshold=0.4
    )
    assert report_loose.pairs_flagged > 0


def test_audit_examples_capped():
    """100+ flagged pairs but max_examples=5 → len(examples) == 5."""
    # Make 20 samples in train and 10 in val, ALL sharing the same identity
    # → all 200 pairs are contaminated; examples should be capped at 5
    identity = "id_shared"
    train = [f"id_shared_{i}" for i in range(20)]
    val = [f"id_shared_{j}" for j in range(20, 30)]
    test: list[str] = []

    shared_vec = torch.zeros(16)
    shared_vec[0] = 1.0
    embedding_fn = lambda s: shared_vec  # noqa: E731

    report = audit_split_contamination(
        train, val, test,
        embedding_fn=embedding_fn,
        threshold=0.6,
        max_examples=5,
    )

    assert report.pairs_flagged == 200   # 20×10
    assert len(report.examples) == 5
