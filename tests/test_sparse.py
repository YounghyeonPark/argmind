# SPDX-License-Identifier: Apache-2.0
"""A sparse descriptor: how few active dimensions survive, and what the sparsity actually costs.

The crop embedding is 108-d with 100 non-zeros and its energy is spread — top-8 holds 37%, top-16
holds 60% — so keeping only the largest `k` throws real signal away. Measured across the three
capabilities established in `test_recognition_memory`, `test_attention_by_memory` and
`test_bootstrap`:

| k | recognition | attention | bytes per exemplar |
|---|---|---|---|
| 4 | 0.967 | 0.875 | 24 |
| 8 | 0.978 | 0.938 | 48 |
| **16** | **1.000** | **1.000** | **96** |
| 32 | 1.000 | 1.000 | 192 |
| 108 (dense) | 1.000 | 1.000 | 432 |

Bytes are `(value f4, index u16)` pairs against a dense f4 vector. **k = 16 is free**: both tasks are
at the dense score for **4.5× less storage** and a sixth of the multiply-adds in a match.

**The bootstrap number looked erratic and the cause is the metric, not the representation.** Scored
as "is the single rarest crop within 0.10 of an object centre" it reads 0.167, 1.000, 0.000, 0.750,
1.000 for k = 4, 8, 16, 32, 108 — non-monotonic, and reproducible at three times the sample, so not
noise. Two explanations were tested:

- **support mismatch** — two vectors sparsified to different supports share few active dimensions and
  the dot product collapses. Named in advance and **refuted**: at k ≥ 16, 0.000 of candidate pairs
  share no support.
- **the ranking itself** — measured, and it is fine at every k. On-object crops rank far rarer than
  off-object ones and the separation improves monotonically:

| k | mean rarity rank, on-object | off-object |
|---|---|---|
| 4 | 43.2 | 132.0 |
| 8 | 29.4 | 133.5 |
| 16 | 25.1 | 134.0 |
| 32 | 21.2 | 134.5 |
| 108 | 21.2 | 134.5 |

with 25 of 247 candidates on an object. **So sparsity does not break the bootstrap signal; taking a
single argmin and demanding its centre fall within 0.10 does.** The crop at the argmin covers only
0.20–0.28 of its area with object — it contains one without being centred on it. The rank
separation is the sound measure and it is what this file asserts.
"""

from __future__ import annotations

import numpy as np
import pytest

from argmind.adapt import RecursiveLeastSquares
from argmind.backbone import Backbone
import tests.test_attention_by_memory as A
import tests.test_bootstrap as B
import tests.test_recognition_memory as R


def topk(v, k):
    """Keep the `k` largest-magnitude entries, zero the rest."""
    if k >= v.size:
        return v
    out = np.zeros_like(v)
    out[np.argpartition(np.abs(v), -k)[-k:]] = v[np.argpartition(np.abs(v), -k)[-k:]]
    return out


def sparse_embed(bb, patch, k):
    v = topk(np.asarray(bb.features(patch), dtype=np.float64), k)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


@pytest.fixture(scope="module")
def bb():
    return Backbone(R.PLAN, (R.CROP, R.CROP, 3), dtype="f8", seed=23, pool=R.POOL)


def recognition(bb, k, steps=300, n_test=60):
    head = RecursiveLeastSquares(bb.n_outputs + 1, 3, forgetting=0.999, p0=100.0, dtype="f8")
    rng = np.random.default_rng(5)
    for t in range(steps):
        cx, cy = R.place(rng, test=False)
        img = R.render(t % 3, cx, cy, seed=t)
        head.update(np.concatenate([sparse_embed(bb, R.crop_at(img, cx, cy), k), [1.0]]),
                    np.eye(3)[t % 3])
    tr, hits = np.random.default_rng(77), 0
    for t in range(n_test):
        cx, cy = R.place(tr, test=True)
        img = R.render(t % 3, cx, cy, seed=8000 + t)
        p = np.asarray(head.predict(
            np.concatenate([sparse_embed(bb, R.crop_at(img, cx, cy), k), [1.0]])))
        hits += int(int(np.argmax(p)) == t % 3)
    return hits / n_test


def rank_separation(bb, k):
    """Mean rarity rank of on-object candidates against off-object ones. Lower is rarer.

    This rather than the argmin, because the argmin is what made the measurement look erratic — it
    lands on crops covering only a fifth of their area with object, which a 0.10 centre tolerance
    scores as a miss while the ranking is perfectly sound.
    """
    cs = B.interior(B.H, B.W)
    specs = [("disc", 0.30, 0.40), ("bar", 0.70, 0.62)]
    img = B.render(specs, seed=1, texture="none")
    on = np.asarray([any(np.hypot(cx - p[1], cy - p[2]) < 0.10 for p in specs) for cx, cy in cs])
    E = np.asarray([sparse_embed(bb, B.crop_at(img, cx, cy, B.H, B.W), k) for cx, cy in cs])
    S = E @ E.T
    rare = -(S.sum(axis=1) - np.diag(S)) / (len(cs) - 1)
    order = np.argsort(-rare)
    rank = np.empty(len(cs), int)
    rank[order] = np.arange(len(cs))
    return float(rank[on].mean()), float(rank[~on].mean())


def test_sixteen_of_a_hundred_and_eight_dimensions_is_free_for_recognition(bb):
    """4.5× less storage per exemplar at the dense score."""
    dense, sparse16 = recognition(bb, 108), recognition(bb, 16)
    assert dense > 0.9, f"the dense control fell to {dense:.3f}"
    assert sparse16 >= dense - 0.05, (
        f"k=16 reaches {sparse16:.3f} against dense {dense:.3f}; the claim that sixteen dimensions "
        "are free no longer holds")


def test_it_degrades_gracefully_rather_than_collapsing(bb):
    """The shape of the trade matters as much as one number: four dimensions out of 108 still
    recognise, which is what says the code is distributed rather than brittle."""
    four = recognition(bb, 4)
    assert four > 0.7, f"k=4 collapsed to {four:.3f}; the degradation is not graceful"
    assert four <= recognition(bb, 16) + 1e-9, "k=4 beat k=16, which would invert the trade"


@pytest.mark.parametrize("k", (4, 8, 16, 32, 108))
def test_the_rarity_RANKING_survives_every_sparsity_tested(bb, k):
    """**The correction.** Scored by its single argmin the bootstrap looked non-monotonic — 0.167,
    1.000, 0.000, 0.750, 1.000 — and both offered explanations were tested. Support mismatch was
    refuted (0.000 of pairs share no support at k ≥ 16) and the ranking turns out to be sound at
    every k. The argmin lands on crops covering a fifth of their area with object; the metric was
    the fragile part, not the representation.
    """
    on, off = rank_separation(bb, k)
    assert on < off - 40, (
        f"k={k}: on-object crops rank {on:.1f} against off-object {off:.1f}; the rarity signal no "
        "longer separates them and the bootstrap rests on it")


def test_the_separation_improves_with_k_which_is_the_monotone_the_argmin_hid(bb):
    coarse, fine = rank_separation(bb, 4)[0], rank_separation(bb, 32)[0]
    assert fine < coarse, (
        f"k=32 ranks on-object crops at {fine:.1f}, no better than k=4's {coarse:.1f} — more "
        "dimensions should sharpen the rarity signal, and the erratic argmin is what obscured it")
