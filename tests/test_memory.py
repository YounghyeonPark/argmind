# SPDX-License-Identifier: Apache-2.0
"""`ExemplarMemory` — the measured recognition loop, made into something this library can carry.

Everything in `tests/test_recognition_memory.py`, `test_attention_by_memory.py`, `test_bootstrap.py`
and `test_sparse.py` is a **measurement**: it establishes that a crop descriptor from an untrained
backbone recognises objects, steers attention, bootstraps without labels, and survives at 16 of 108
dimensions. None of it went through a contract. This does.

**What the contract actually demands here**, and why each is not a formality:

| requirement | why | measured |
|---|---|---|
| one flat contiguous state block | persistence is a copy and the C ABI needs no serialisation | 8 slots × 108 features × k=16 → **1,056 B** |
| fixed capacity | the target tier has no heap; the demo's `ObjectMemory` grows a Python list and is expressible nowhere but a host | a full memory replaces, never grows |
| `state` / `load_state` / `n_features` / `n_outputs` / `dtype` | what `persist.pack` reads | goes through `Store` unchanged |

**It is deliberately not an `Adapter`.** `update(x, y)` presumes a target and an exemplar has none —
it is admitted, not fitted. Declaring the method with a `y` no caller could supply would be the
half-built shape this project refuses. `n_outputs` **is** declared, as the capacity, because that is
what it would emit read as a layer and because it makes the record's shape check mean something: an
8-slot record is refused by a 16-slot memory, measured below.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from argmind.memory import ExemplarMemory
from argmind.persist import Store


def descriptors(n, dim=108, seed=0):
    g = np.random.default_rng(seed)
    return [g.standard_normal(dim) for _ in range(n)]


def test_the_state_is_ONE_flat_contiguous_block_and_the_arithmetic_is_stated():
    m = ExemplarMemory(capacity=8, n_features=108, k=16, threshold=0.8)
    assert m.state.ndim == 1 and m.state.flags["C_CONTIGUOUS"]
    assert m.state_size == 2 * 8 * 16 + 8 == 264, "the layout changed; the byte figure moves with it"
    assert m.state.nbytes == 1056, f"{m.state.nbytes} B — the 8 KB core tier budget is stated on this"
    # Live storage, not a snapshot: `Stack` diverges here and it is worth knowing which is which.
    m.state[0] = 7.0
    assert m.state[0] == 7.0


def test_it_recognises_what_it_has_seen_and_admits_what_it_has_not():
    m = ExemplarMemory(capacity=8, n_features=108, k=16, threshold=0.8)
    a, b = descriptors(2, seed=1)
    first = m.observe(a)
    assert not first.matched and m.occupied == 1, "the first sighting cannot be a match"
    again = m.observe(a * 1.01)
    assert again.matched and again.slot == first.slot and m.occupied == 1, (
        "the same object was admitted twice instead of being recognised")
    other = m.observe(b)
    assert not other.matched and other.slot != first.slot and m.occupied == 2


def test_an_EMPTY_memory_reports_no_match_rather_than_a_best_one():
    """Empty slots are skipped rather than scoring zero. Letting them compete would make the first
    query return a confident nearest neighbour to nothing."""
    m = ExemplarMemory(capacity=4, n_features=108, k=16, threshold=0.8)
    r = m.look_up(descriptors(1)[0])
    assert not r.matched and r.slot == -1 and r.score == 0.0


def test_a_FULL_memory_replaces_and_never_grows():
    """The property that makes it expressible on a target at all."""
    m = ExemplarMemory(capacity=4, n_features=108, k=16, threshold=0.95)
    before = m.state.size
    for d in descriptors(20, seed=2):
        m.observe(d)
    assert m.occupied == 4, f"occupancy reached {m.occupied} on a capacity-4 memory"
    assert m.state.size == before, "the state block grew"


def test_the_least_recently_MATCHED_slot_is_the_one_replaced():
    """Replacement has to be defined without a heap-allocated queue, so an age scalar per slot does
    it — and a slot that keeps being recognised keeps its place."""
    m = ExemplarMemory(capacity=3, n_features=108, k=16, threshold=0.9)
    a, b, c, d = descriptors(4, seed=3)
    for x in (a, b, c):
        m.observe(x)
    kept = m.look_up(a)                       # touching `a` makes it the most recent
    assert kept.matched
    m.observe(d)                              # forces a replacement
    assert m.look_up(a).matched, "the slot that was just recognised was the one evicted"


def test_it_goes_through_Store_unchanged_and_a_wrong_capacity_is_REFUSED():
    m = ExemplarMemory(capacity=8, n_features=108, k=16, threshold=0.8)
    seen = descriptors(5, seed=4)
    for d in seen:
        m.observe(d)

    with tempfile.TemporaryDirectory() as t:
        store = Store(Path(t) / "mem")
        store.save(m, fingerprint=0xBEEF)

        restored = ExemplarMemory(capacity=8, n_features=108, k=16, threshold=0.8)
        assert store.load(restored, fingerprint=0xBEEF) is not None
        assert restored.occupied == 5
        assert all(restored.look_up(d).matched for d in seen), (
            "the memory came back but no longer recognises what it held")

        wider = ExemplarMemory(capacity=16, n_features=108, k=16, threshold=0.8)
        with pytest.raises(ValueError, match="108x8"):
            store.load(wider, fingerprint=0xBEEF)


def test_it_does_NOT_claim_to_be_an_Adapter():
    """Recorded as a decision rather than left to be discovered. An exemplar is admitted, not
    fitted; a `y` in the signature would be one no caller could supply."""
    from argmind.adapt import Adapter

    m = ExemplarMemory(capacity=4, n_features=108, k=16, threshold=0.8)
    assert not isinstance(m, Adapter)
    assert not hasattr(m, "update"), (
        "ExemplarMemory has grown an update(); if it is genuinely an Adapter now, this file's "
        "reasoning about why it is not should change with it")
    assert not hasattr(m, "predict")


@pytest.mark.parametrize("bad", [
    dict(capacity=0, n_features=8, k=2, threshold=0.5),
    dict(capacity=4, n_features=8, k=0, threshold=0.5),
    dict(capacity=4, n_features=8, k=9, threshold=0.5),
    dict(capacity=4, n_features=8, k=2, threshold=1.5),
])
def test_shapes_and_thresholds_it_cannot_use_are_refused_at_construction(bad):
    with pytest.raises(ValueError):
        ExemplarMemory(**bad)


def test_it_discriminates_REAL_descriptors_which_is_what_says_the_right_k_are_kept():
    """**Added because a mutant survived.** Every test above uses random Gaussian descriptors, where
    any consistent choice of `k` dimensions self-matches exactly — so storing the *smallest* `k`
    instead of the largest passed them all. Real crop embeddings carry structure, and keeping the
    wrong half of it destroys discrimination while leaving self-matching intact.

    Uses the same fixture the recognition measurements were made on: one object stored, a
    different-shaped object of the **same colour** queried against it.
    """
    from argmind.backbone import Backbone
    from tests.test_recognition_memory import CROP, PLAN, POOL, SAME, crop_at, render

    bb = Backbone(PLAN, (CROP, CROP, 3), dtype="f8", seed=23, pool=POOL)
    m = ExemplarMemory(capacity=4, n_features=bb.n_outputs, k=16, threshold=0.9)

    def describe(shape_i, cx, cy, seed):
        return np.asarray(bb.features(crop_at(render(shape_i, cx, cy, seed=seed), cx, cy)))

    disc = describe(0, 0.40, 0.50, 1)
    m.admit(disc)

    same = m.look_up(describe(0, 0.44, 0.52, 2))          # the same shape, moved a little
    other = m.look_up(describe(2, 0.40, 0.50, 3))         # a bar, same colour, same place

    assert same.score > other.score + 0.05, (
        f"a disc scores {same.score:.3f} against a stored disc and a bar scores {other.score:.3f}; "
        "the memory is not discriminating shape, which is what the descriptor was measured to carry")
    assert same.matched, f"the same shape scored only {same.score:.3f} and did not match"
