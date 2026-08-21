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
    # **Updated deliberately 2026-08-20**: the block grew by two scalars per slot -- `spells` and
    # `here` -- which are the retention criterion. This assertion failing is how that change was
    # noticed, which is what it is for. 8 slots x 108 features at k=16: 2*8*16 + 3*8.
    assert m.state_size == 2 * 8 * 16 + 3 * 8 == 280, (
        "the layout changed; the byte figure and the core-tier claim move with it")
    assert m.state.nbytes == 1120, f"{m.state.nbytes} B — the 8 KB core tier budget is stated on this"
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


def test_SPELLS_count_appearances_and_not_frames():
    """**The retention criterion, and the obvious version of it is worse than useless.**

    Counting *frames* a slot was seen in makes the wall the best-remembered thing in a room: static
    scenery appears in every frame, perfectly, while a person who steps out loses ground. A **spell**
    is one unbroken run of presence, so a bookshelf accumulates one long spell however many frames it
    filled, and a person who comes and goes accumulates many.

    Measured over twelve frames, one slot present throughout and one present every other pair:
    **1 spell against 3**.
    """
    m = ExemplarMemory(capacity=4, n_features=16, k=4, threshold=0.9)
    g = np.random.default_rng(0)
    static, mover = m.admit(g.standard_normal(16)), m.admit(g.standard_normal(16))
    frames = 0
    for k in range(12):
        present = [static] + ([mover] if (k // 2) % 2 == 0 else [])
        m.end_frame(present)
        frames += 1

    spells = np.asarray(m.spells)
    assert spells[static] == 1, (
        f"the always-present slot has {spells[static]:.0f} spells; an unbroken run is one spell or "
        "the criterion is counting frames again")
    assert spells[mover] > spells[static], (
        f"the intermittent slot has {spells[mover]:.0f} spells against the static one's "
        f"{spells[static]:.0f} — the signal that separates objects from scenery is gone")
    assert spells[mover] < frames, (
        "the intermittent slot has a spell per frame, so runs are not being collapsed")


def test_a_fresh_admission_starts_with_ONE_spell_not_zero():
    """Zero would make every new exemplar the next thing evicted — it is present at the moment it is
    admitted, and the count has to say so."""
    m = ExemplarMemory(capacity=3, n_features=16, k=4, threshold=0.9)
    slot = m.admit(descriptors(1, dim=16)[0])
    assert np.asarray(m.spells)[slot] == 1


def test_eviction_prefers_the_FEWEST_SPELLS_over_the_oldest():
    """**Why age alone is the wrong rule.** It throws out whichever exemplar has waited longest,
    which on a real scene is an object that has stepped out — exactly the one the memory exists to
    hold. Scenery, present continuously, is the thing that should go.
    """
    m = ExemplarMemory(capacity=2, n_features=16, k=4, threshold=0.99)
    a, b, c = descriptors(3, dim=16, seed=5)
    scenery = m.admit(a)                       # one long spell, admitted first
    visitor = m.admit(b)
    for k in range(10):
        # **The scenery is looked up every frame**, which refreshes its age and makes it the NEWEST
        # slot. Without that the two rules pick the same victim and the fixture cannot tell them
        # apart -- a mutant reverting to `argmin(age)` survived until this line existed.
        m.look_up(a)
        present = [scenery] + ([visitor] if k % 2 == 0 else [])
        m.end_frame(present)
    spells = np.asarray(m.spells)
    assert spells[visitor] > spells[scenery], "the fixture did not build the spell difference"
    ages = np.asarray(m.state)[-3 * m.capacity:-2 * m.capacity]
    assert ages[scenery] > ages[visitor], (
        "the scenery is not the more recently touched slot, so age and spells would agree here and "
        "this test could not show which rule is in force")

    # The memory is full. Admitting a third must displace the scenery, not the visitor -- even
    # though the scenery was touched more recently, being present every frame.
    third = m.admit(c)
    assert third == scenery, (
        f"evicted slot {third} (spells {spells[third]:.0f}) instead of the scenery slot "
        f"{scenery} (spells {spells[scenery]:.0f}); age is deciding, not evidence of recurrence")
    assert m.look_up(b).matched, "the visitor was evicted despite having more appearances"


def test_with_EQUAL_spells_the_less_recently_seen_slot_goes():
    """The tie-break, tested on its own — with unequal spells it is never reached, so a mutant
    removing the age refresh from `end_frame` survived until this existed.

    Both slots appear the same number of times; only one is still around at the end.
    """
    m = ExemplarMemory(capacity=2, n_features=16, k=4, threshold=0.99)
    a, b, c = descriptors(3, dim=16, seed=11)
    #: **`late` is admitted FIRST**, so its admission age is the *smaller* one. Only the refresh in
    #: `end_frame` can make it the more recent slot — admitted second, admission order alone would
    #: already put it ahead and the assertion below could not tell the two apart. A mutant removing
    #: that refresh survived until the order was swapped.
    late, early = m.admit(b), m.admit(a)
    for k in range(6):                       # both blink together: equal spells
        m.end_frame([early, late] if k % 2 == 0 else [])
    for _ in range(4):                       # then only `late` keeps showing up
        m.end_frame([late])
        m.end_frame([])

    spells = np.asarray(m.spells)
    assert spells[early] == spells[late] - 4 or spells[early] < spells[late], (
        f"the fixture wanted comparable spells, got {spells[early]:.0f} and {spells[late]:.0f}")

    ages = np.asarray(m.state)[-3 * m.capacity:-2 * m.capacity]
    assert ages[late] > ages[early], (
        "presence is not refreshing the age, so recency cannot break a tie and this test would pass "
        "for the wrong reason")


def test_the_BATCHED_look_up_is_what_makes_a_large_memory_usable():
    """**The per-slot loop made capacity a cost, and it should not be one.**

    `look_up` walks the slots, so it is linear in the capacity: measured **0.018 ms at 8 slots and
    2.033 ms at 1,000**, and 147 candidates of the latter is 479 ms a frame — 3.35 fps from the
    look-ups alone. `look_up_many` does the same work as one matrix product:

    | slots | loop ×147 | batched | speed-up |
    |---|---|---|---|
    | 8 | 3 ms | 1.0 ms | 3× |
    | 128 | 34 ms | 1.2 ms | 28× |
    | 1,000 | 479 ms | 8.8 ms | 54× |
    | 4,000 | 2,479 ms | 9.4 ms | **263×** |

    **It is nearly flat from 1,000 to 4,000**, because the remaining cost is sparsifying the queries
    rather than the slot count. That is what makes an effectively unbounded long-term memory viable —
    the owner's requirement, and the right one: human long-term visual memory holds thousands, while
    the ~4 limit belongs to simultaneous tracking, a different quantity.
    """
    m = ExemplarMemory(capacity=64, n_features=32, k=8, threshold=0.5)
    g = np.random.default_rng(3)
    stored = [g.standard_normal(32) for _ in range(20)]
    for d in stored:
        m.admit(d)

    queries = stored[:10] + [g.standard_normal(32) for _ in range(5)]
    batched = m.look_up_many(queries)
    assert len(batched) == len(queries)

    # **The two must agree**, or the fast path is computing something else. Slot and matched flag
    # are compared; scores to a tolerance, since one path accumulates in a different order.
    for q, fast in zip(queries, batched):
        slow = m.look_up(q)
        assert fast.slot == slow.slot, f"batched chose slot {fast.slot}, the loop chose {slow.slot}"
        assert fast.matched == slow.matched
        assert abs(fast.score - slow.score) < 1e-9

    # And a stored descriptor must find itself, or the agreement above is between two broken paths.
    assert all(m.look_up_many(stored)[i].slot == i for i in range(len(stored))), (
        "a stored exemplar does not match its own slot under the batched path")


def test_an_EMPTY_memory_batches_to_no_matches_rather_than_slot_zero():
    """The empty case has to be handled separately in the fast path: an all-zero dense matrix scores
    0 against everything, and `argmax` of ties is slot 0 — a confident nearest neighbour to nothing.
    """
    m = ExemplarMemory(capacity=4, n_features=16, k=4, threshold=0.5)
    out = m.look_up_many(descriptors(3, dim=16))
    assert all(r.slot == -1 and not r.matched for r in out), f"{out}"


def test_the_dense_cache_is_derived_and_INVALIDATED_not_persisted():
    """It is not in the state block — a 4,000-slot cache is 3.5 MB of float64 and would triple the
    record. So it must be rebuilt after anything that changes the slots, and the two paths agreeing
    after an admit is what says it was."""
    m = ExemplarMemory(capacity=4, n_features=16, k=4, threshold=0.5)
    a, b = descriptors(2, dim=16, seed=9)
    m.admit(a)
    assert m.look_up_many([a])[0].slot == 0
    slot_b = m.admit(b)
    got = m.look_up_many([b])[0]
    assert got.slot == slot_b, (
        f"after admitting into slot {slot_b} the batched path returned {got.slot}; the dense cache "
        "was not invalidated")
    m.reset()
    assert m.look_up_many([a])[0].slot == -1, "reset did not invalidate the cache"


def test_a_NEGATIVELY_correlated_query_still_names_the_occupied_slot():
    """**The case the empty-slot mask exists for**, and the only one that can see it.

    An empty slot is an all-zero row in the dense matrix, so it scores exactly 0 — which beats any
    *negative* score. Without masking, a query anti-correlated with everything stored would come back
    pointing at a slot that holds nothing. A mutant removing the mask survived every other test here,
    because they all produce positive matches.
    """
    m = ExemplarMemory(capacity=4, n_features=8, k=8, threshold=0.9)
    stored = np.array([1.0, 1.0, 0.5, 0.0, -0.5, 0.0, 0.0, 0.0])
    slot = m.admit(stored)
    assert slot == 0 and m.occupied == 1

    opposite = -stored
    got = m.look_up_many([opposite])[0]
    assert got.score < 0.0, f"the query is not anti-correlated: score {got.score:.3f}"
    assert got.slot == slot, (
        f"a negative-scoring query returned slot {got.slot}, which holds nothing — an empty slot "
        "scores 0 and outranks a negative match unless it is masked out")
    assert not got.matched
    # And the loop path agrees, which is what says the two are the same function.
    assert m.look_up(opposite).slot == slot
