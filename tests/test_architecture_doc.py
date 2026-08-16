# SPDX-License-Identifier: Apache-2.0
"""`docs/ARCHITECTURE.md` is checked against the running code, figure by figure.

**Why this exists at all.** A hand-copied table in this repository has gone stale three times, which
is why `cpp/target/SIZES.md` is generated and gated. `ARCHITECTURE.md` is not generated — its prose
carries reasoning a generator would flatten — so the weaker half is done instead: every number in it
is recomputed here and compared. A stale figure fails the suite.

**And the failure mode this guards against is specific.** `SIZES.md` once instructed readers to
re-run it with a `--regen-sizes` option that did not exist. A document promising a tool, or quoting
a shape the code no longer produces, is the same failure arriving by another door — so this file
also asserts that the doc does *not* claim a generator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from argmind.backbone import Backbone
from demo.pointer import IN_SHAPE, MAP_H, MAP_W, PLAN, POOL, build, map_memory_width

DOC = Path(__file__).resolve().parent.parent / "docs" / "ARCHITECTURE.md"


@pytest.fixture(scope="module")
def text():
    return DOC.read_text(encoding="utf-8", errors="replace")


@pytest.fixture(scope="module")
def regions(text):
    """The doc states each figure **twice** — once in the forward diagram, once in the parameter
    table — so it is checked in both places separately.

    A whole-document `in` check is a presence test, not a consistency test: a mutant that staled the
    weight count in the diagram and left the table correct **survived**, because the string was
    still somewhere in the file. A diagram disagreeing with its own table is exactly the failure
    this document invites, so the two regions are split and compared independently.
    """
    diagram = text.split("## 2.")[0]
    table = text.split("## 3.")[1].split("## 4.")[0]
    assert len(diagram) > 400 and len(table) > 200, "the section headings moved; re-split"
    return {"diagram": diagram, "table": table}


@pytest.fixture(scope="module")
def facts():
    """Recompute every figure the document states, from the code it describes."""
    bb = Backbone(PLAN, IN_SHAPE, pool=POOL, seed=0)
    h, w, c = IN_SHAPE
    shapes, weights = [f"{h} x {w} x {c}"], []
    for e in PLAN:
        if e[0] == "conv":
            _k, co, k, s = e
            weights.append(k * k * c * co + co)
            h, w, c = -(-h // s), -(-w // s), co
        elif e[0] == "dw":
            _k, k, s = e
            weights.append(k * k * c + c)
            h, w = -(-h // s), -(-w // s)
        elif e[0] == "pw":
            _k, co = e
            weights.append(c * co + co)
            c = co
        else:
            raise AssertionError(f"ARCHITECTURE.md does not describe layer {e!r}")
        shapes.append(f"{h} x {w} x {c}")

    out = {}
    for mode in ("points", "map"):
        _b, _h, guard, *_ = build(3, "backbone", output=mode)
        out[mode] = (guard.n_features, guard.n_outputs, guard.state.size)

    return {
        "shapes": shapes,
        "layer_weights": weights,
        "backbone_params": int(bb.state.size),
        "embedding": int(bb.n_outputs),
        "peak": int(bb.peak_activation_values()),
        "memory": int(map_memory_width()),
        "grid": (MAP_H, MAP_W),
        "heads": out,
    }


def test_the_layer_shapes_in_the_diagram_are_the_ones_the_code_produces(text, facts):
    """The forward figure's tensor column. If a plan entry changes, every line below it moves."""
    for shape in facts["shapes"]:
        assert shape in text, f"the diagram does not carry the tensor shape {shape!r}"
    assert f"3*3*16 = {facts['embedding']}" in text or f"= {facts['embedding']}" in text, (
        f"the pooled embedding is {facts['embedding']} and the diagram does not say so")


def test_the_weight_counts_are_per_layer_and_they_add_up(text, facts, regions):
    """Each layer's own count, and the total, **in both the diagram and the table**. A total can be
    right while the split is wrong, and one region can be right while the other is stale."""
    for n in facts["layer_weights"]:
        for where, body in regions.items():
            assert f"{n:,}" in body or str(n) in body, (
                f"the {where} does not report a layer of {n} weights")
    assert sum(facts["layer_weights"]) == facts["backbone_params"], (
        f"this test's own arithmetic is wrong: {sum(facts['layer_weights'])} against the backbone's "
        f"{facts['backbone_params']}")
    assert f"{facts['backbone_params']:,}" in text, (
        f"the doc does not state the backbone's {facts['backbone_params']} weights")


def test_the_head_shapes_and_parameter_table_match_both_modes(text, facts, regions):
    for mode, (n, m, size) in facts["heads"].items():
        assert f"{n} -> {m}" in text, f"the {mode} head shape {n} -> {m} is not in the doc"
        assert f"{size:,}" in text, f"the {mode} head's {size} parameters are not in the doc"

    # The map head is the one figure the diagram and the table both carry; they must agree.
    map_size = facts["heads"]["map"][2]
    for where, body in regions.items():
        assert f"{map_size:,}" in body, (
            f"the {where} does not carry the map head's {map_size:,} parameters — the diagram and "
            "the table have drifted apart")

    for mode, (_n, _m, size) in facts["heads"].items():
        total = size + facts["backbone_params"]
        assert f"{total:,}" in text, f"the {mode} total of {total} is not in the doc"
        pct = round(100.0 * size / total)
        assert f"{pct}%" in text, (
            f"the doc does not carry the {mode} head's share, {pct}% of the stack")


def test_the_basis_width_and_the_map_geometry_are_stated(text, facts):
    n_map = facts["heads"]["map"][0]
    h, w = facts["grid"]
    assert f"{facts['embedding']} frame + {facts['memory']} memory + 1 constant" in text, (
        "the basis composition in the diagram does not match "
        f"{facts['embedding']} + {facts['memory']} + 1 = {n_map}")
    assert facts["embedding"] + facts["memory"] + 1 == n_map, (
        "this test's own arithmetic is wrong: the basis parts do not sum to the head's width")
    assert f"{h} x {w} = {h * w}" in text, f"the output grid {h}x{w} is not in the doc"


def test_the_peak_activation_figure_is_the_one_the_backbone_reports(text, facts):
    """The number a target tier actually has to hold, and the one most likely to be quoted. It is
    *not* the weight count, and the doc says so — that distinction is the reason it is here."""
    assert f"{facts['peak']:,}" in text, (
        f"the doc does not state the peak activation cost of {facts['peak']:,} values")


def test_the_doc_does_not_promise_a_generator_it_does_not_have(text):
    """`SIZES.md` once told readers to re-run it with `--regen-sizes`, which did not exist. That is
    the failure this assertion prevents from recurring in a second document."""
    assert "There is no generator." in text, (
        "ARCHITECTURE.md no longer states that it is checked rather than generated — if a generator "
        "was added, this assertion is where the claim was pinned and should change with it")
    for invented in ("--regen-arch", "docs.archcheck", "--regen-architecture"):
        assert invented not in text, (
            f"the doc references {invented!r}; if that tool now exists, delete this line, and if it "
            "does not, delete the reference")
