# SPDX-License-Identifier: Apache-2.0
"""The C++ streams and aligner against the Python reference.

Two halves that fail differently, so they are driven separately.

The **ring** has to agree on which pushes it accepted, what it overwrote, and what
timestamps it still holds. The reference raises on a non-monotonic timestamp and the C++
returns false — different mechanisms, and they must refuse the same samples, because from
the first disagreement the two are comparing different histories.

The **aligner** has to agree on which sample each policy picks and how stale it says that
sample was. This is the part worth testing hardest: the whole abstraction exists because the
choice of policy changes the answer, so a port that quietly collapsed two policies into one
would still look correct on any single-rate fixture.

One arithmetic detail is carried across deliberately and pinned here: interpolation blends in
**float64 and then narrows**, even for an f32 stream, because the reference does. Blending in
the stream's own type would be a defensible choice and a different one.
"""

import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import _toolbuild

from argmind.align import Aligner
from argmind.stream import Stream

REPO = Path(__file__).resolve().parent.parent
CPP = REPO / "cpp"

# The explicit `4x` before the double is not decoration. Python's `struct` with `<` packs
# without padding; a C++ struct does not, and the double after two uint32s lands at offset 24
# there and would land at 20 here. The first version of this file omitted it, and the tool
# read a capacity of zero out of the misaligned header and died on a modulo — loudly, which
# was lucky. Every size below is asserted against the tool's own static_asserts.
_JOB = struct.Struct("<4sHBBHHII4xd")    # 32 bytes; see cpp/tools/scheck.cpp
_SPEC = struct.Struct("<HHI")            # 8
_PUSH = struct.Struct("<IId")            # 16
_PICK = struct.Struct("<BBB5xd")         # 16
assert (_JOB.size, _SPEC.size, _PUSH.size, _PICK.size) == (32, 8, 16, 16), (
    "these must match the static_asserts in cpp/tools/scheck.cpp")

POLICIES = {"hold_last": 0, "nearest": 1, "interpolate": 2, "drop_if_stale": 3}


def _bits(x: float) -> int:
    """A float's bit pattern, so -0.0 and 0.0 compare unequal and a NaN payload shows."""
    return struct.unpack("<Q", struct.pack("<d", float(x)))[0]


def _compiler() -> str | None:
    for name in ("g++", "clang++"):
        if shutil.which(name):
            return name
    return None


@pytest.fixture(scope="session", params=_toolbuild.ARCHES)
def scheck(request, tmp_path_factory) -> Path:
    """The parity tool, built for the host and for the declared target.

    Both, on every case: a review measured a QEMU invocation at 157 ms and showed
    that the hand-picked subset which used to stand in for the target passed with
    the host binaries substituted, and passed again with the target FPU set to
    round-toward-zero. See tests/_toolbuild.py.
    """
    work = tmp_path_factory.mktemp("cpp_s_" + request.param)
    return _toolbuild.build_tool("scheck", request.param, work)


def run_cpp(scheck: Path, tmp_path: Path, specs, pushes, queries=(), *,
            policy="hold_last", max_stale=0.0, dtype="f8"):
    """specs: [(dim, capacity)]. pushes: [(stream, t, values, valid)]."""
    dt = np.dtype(dtype)
    mode = 1 if queries else 0
    job = tmp_path / "job.bin"
    out = tmp_path / "out.bin"
    with open(job, "wb") as fh:
        fh.write(_JOB.pack(b"AMSS", 1, dt.itemsize, mode, POLICIES[policy], len(specs),
                           len(pushes), len(queries), float(max_stale)))
        for dim, cap in specs:
            fh.write(_SPEC.pack(dim, cap, 0))
        for sid, t, values, valid in pushes:
            fh.write(_PUSH.pack(sid, 1 if valid else 0, float(t)))
            fh.write(np.ascontiguousarray(values, dtype=dt).tobytes())
        for t in queries:
            fh.write(struct.pack("<d", float(t)))

    done = subprocess.run([str(scheck), str(job), str(out)], capture_output=True,
                          text=True)
    assert done.returncode == 0, f"scheck failed: {done.stdout}\n{done.stderr}"

    raw = out.read_bytes()
    off = 0
    accepted = list(raw[off:off + len(pushes)])
    off += len(pushes)

    rings = []
    for _ in specs:
        count, dropped = struct.unpack_from("<II", raw, off)
        off += 8
        times = list(struct.unpack_from(f"<{count}d", raw, off))
        off += 8 * count
        rings.append({"count": count, "dropped": dropped, "times": times})

    picks = []
    for _ in queries:
        row = []
        for dim, _cap in specs:
            present, valid, refused, staleness = _PICK.unpack_from(raw, off)
            off += _PICK.size
            values = np.frombuffer(raw, dtype=dt, count=dim, offset=off)
            off += dim * dt.itemsize
            row.append({"present": bool(present), "valid": bool(valid),
                        "refused": bool(refused), "staleness": staleness,
                        "values": np.array(values)})
        picks.append(row)
    assert off == len(raw), f"unread bytes in the result: {len(raw) - off}"
    return {"accepted": accepted, "rings": rings, "picks": picks}


def run_python(specs, pushes, queries=(), *, policy="hold_last", max_stale=0.0,
               dtype="f8"):
    streams = [Stream(f"s{i}", dim=dim, capacity=cap, dtype=dtype)
               for i, (dim, cap) in enumerate(specs)]
    accepted = []
    for sid, t, values, valid in pushes:
        try:
            streams[sid].push(t=float(t), values=values, valid=valid)
            accepted.append(1)
        except ValueError:
            accepted.append(0)

    rings = [{"count": len(s), "dropped": s.dropped, "times": list(s.times())}
             for s in streams]

    aligner = Aligner("s0", policy, max_stale if max_stale > 0 else None)
    picks = []
    for t in queries:
        row = []
        for s in streams:
            got = aligner._pick(s, float(t))
            if got is None:
                row.append({"present": False, "valid": False, "staleness": 0.0,
                            "refused": False, "values": np.zeros(s.dim, dtype=dtype)})
            else:
                sample, gap = got
                refused = (policy == "drop_if_stale" and max_stale > 0
                           and abs(gap) > max_stale)
                row.append({"present": True, "valid": sample.valid, "staleness": gap,
                            "refused": refused, "values": np.array(sample.values)})
        picks.append(row)
    return {"accepted": accepted, "rings": rings, "picks": picks}


def compare(scheck, tmp_path, specs, pushes, queries=(), **kw):
    cpp = run_cpp(scheck, tmp_path, specs, pushes, queries, **kw)
    py = run_python(specs, pushes, queries, **kw)

    assert cpp["accepted"] == py["accepted"], (
        "the two accepted different pushes, so every comparison after this is against a "
        f"different history: first differs at index "
        f"{next(i for i, (a, b) in enumerate(zip(cpp['accepted'], py['accepted'])) if a != b)}")
    for i, (c, p) in enumerate(zip(cpp["rings"], py["rings"])):
        assert c["count"] == p["count"], f"stream {i}: count {c['count']} vs {p['count']}"
        assert c["dropped"] == p["dropped"], (
            f"stream {i}: dropped {c['dropped']} vs {p['dropped']}")
        assert [_bits(x) for x in c["times"]] == [_bits(x) for x in p["times"]], (
            f"stream {i}: retained timestamps differ")

    for q, (crow, prow) in enumerate(zip(cpp["picks"], py["picks"])):
        for i, (c, p) in enumerate(zip(crow, prow)):
            assert c["present"] == p["present"], f"query {q} stream {i}: present"
            if not p["present"]:
                continue
            assert c["valid"] == p["valid"], f"query {q} stream {i}: valid"
            assert c["refused"] == p["refused"], (
                f"query {q} stream {i}: the stale-drop decision differs "
                f"(staleness {c['staleness']})")
            # Bit comparison, not `approx` or `array_equal`: both of those treat -0.0 as
            # equal to 0.0, and DESIGN.md claimed this comparator distinguished them. It
            # did not until now — the byte-exact version lived only in a scratchpad script.
            assert _bits(c["staleness"]) == _bits(p["staleness"]), (
                f"query {q} stream {i}: staleness {c['staleness']!r} vs "
                f"{p['staleness']!r} (differ in bits, which -0.0 vs 0.0 would)")
            assert c["values"].tobytes() == p["values"].tobytes(), (
                f"query {q} stream {i}: values {c['values']} vs {p['values']}")
    return cpp, py


# ---------------------------------------------------------------------------
# The ring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_the_ring_overwrites_and_counts_identically(scheck, tmp_path, dtype):
    specs = [(1, 4)]
    pushes = [(0, 0.1 * (i + 1), [float(i)], True) for i in range(10)]
    cpp, _ = compare(scheck, tmp_path, specs, pushes, dtype=dtype)
    assert cpp["rings"][0]["count"] == 4, "the ring did not stay bounded"
    assert cpp["rings"][0]["dropped"] == 6


def test_both_refuse_the_same_non_monotonic_pushes(scheck, tmp_path):
    """The reference raises and the C++ returns false. Different mechanisms, same refusal —
    and it has to be the same set, because a sample one side kept and the other dropped
    makes every later frame a comparison between different histories."""
    pushes = [
        (0, 1.0, [1.0], True),
        (0, 0.9, [2.0], True),      # backwards
        (0, 1.0, [3.0], True),      # equal is also refused
        (0, 1.5, [4.0], True),
        (0, float("nan"), [5.0], True),
        (0, 2.0, [6.0], True),
    ]
    cpp, py = compare(scheck, tmp_path, [(1, 8)], pushes)
    assert cpp["accepted"] == [1, 0, 0, 1, 0, 1], cpp["accepted"]
    assert cpp["rings"][0]["times"] == [1.0, 1.5, 2.0]


def test_a_wrapped_ring_still_reports_times_in_order(scheck, tmp_path):
    pushes = [(0, float(i + 1), [float(i)], True) for i in range(7)]
    cpp, _ = compare(scheck, tmp_path, [(1, 3)], pushes)
    assert cpp["rings"][0]["times"] == [5.0, 6.0, 7.0]


def test_multi_dimensional_streams_round_trip(scheck, tmp_path):
    rng = np.random.default_rng(3)
    specs = [(1, 8), (3, 8), (6, 16)]
    pushes = []
    for i in range(12):
        for sid, (dim, _cap) in enumerate(specs):
            pushes.append((sid, 0.1 * (i + 1) + sid * 1e-3,
                           rng.standard_normal(dim), True))
    compare(scheck, tmp_path, specs, pushes)


# ---------------------------------------------------------------------------
# The policies, which exist because they disagree
# ---------------------------------------------------------------------------

def straddling_fixture():
    """Fast reference at 100 Hz, slow aiding source at 10 Hz, deliberately offset so the
    slow stream *straddles* the query instants. Without that there is nothing to
    interpolate between, and a test would be measuring the refusal to extrapolate instead
    of the interpolation."""
    specs = [(1, 64), (1, 16)]
    pushes = [(0, i * 0.01, [float(i)], True) for i in range(1, 21)]
    pushes.append((1, 0.007, [100.0], True))
    pushes.append((1, 0.107, [200.0], True))
    pushes.sort(key=lambda p: p[1])
    return specs, pushes


@pytest.mark.parametrize("policy", ["hold_last", "nearest", "interpolate",
                                    "drop_if_stale"])
@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_every_policy_picks_the_same_sample(scheck, tmp_path, policy, dtype):
    specs, pushes = straddling_fixture()
    queries = [0.01 * k for k in range(1, 21)]
    compare(scheck, tmp_path, specs, pushes, queries, policy=policy, dtype=dtype)


def test_the_policies_disagree_with_each_other(scheck, tmp_path):
    """If every policy returned the same answer the abstraction would be pointless, and a
    port that collapsed two of them would pass every other test in this file."""
    specs, pushes = straddling_fixture()
    got = {}
    for policy in ("hold_last", "nearest", "interpolate"):
        cpp, _ = compare(scheck, tmp_path, specs, pushes, [0.05], policy=policy)
        got[policy] = float(cpp["picks"][0][1]["values"][0])
    assert len({round(v, 9) for v in got.values()}) > 1, (
        f"all policies agreed ({got}); the alignment choice is not being applied")


def test_hold_last_never_looks_ahead_and_nearest_does(scheck, tmp_path):
    specs, pushes = straddling_fixture()
    queries = [0.01 * k for k in range(1, 21)]

    hold, _ = compare(scheck, tmp_path, specs, pushes, queries, policy="hold_last")
    assert all(row[1]["staleness"] <= 0.0 for row in hold["picks"] if row[1]["present"]), (
        "hold_last used a sample from after the reference instant")

    near, _ = compare(scheck, tmp_path, specs, pushes, queries, policy="nearest")
    assert any(row[1]["staleness"] > 0.0 for row in near["picks"] if row[1]["present"]), (
        "nearest never picked a future sample on a fixture built to force it — if this "
        "holds, the policies are no longer distinguishable")


def test_interpolation_blends_in_float64_even_for_a_float32_stream(scheck, tmp_path):
    """Carried across deliberately: the reference widens to float64, blends, and narrows
    back. Blending in float32 gives a different last bit for most weights, so this is a real
    choice rather than a formality — and it is the kind of difference that would surface as
    a mysterious few-ulp disagreement much later."""
    specs = [(1, 8), (1, 8)]
    pushes = [(1, 0.0, [1.0], True), (0, 0.5, [0.0], True), (1, 1.0, [3.0], True)]
    pushes.sort(key=lambda p: p[1])
    cpp, py = compare(scheck, tmp_path, specs, pushes, [0.3], policy="interpolate",
                      dtype="f4")

    got = float(cpp["picks"][0][1]["values"][0])
    wide = np.float32((1.0 - 0.3) * np.float64(1.0) + 0.3 * np.float64(3.0))
    narrow = np.float32(np.float32(1.0 - 0.3) * np.float32(1.0) +
                        np.float32(0.3) * np.float32(3.0))
    assert got == pytest.approx(float(wide), abs=0.0)
    if wide != narrow:
        assert got != pytest.approx(float(narrow), abs=0.0), (
            "the blend was done in float32; the reference widens first")


def test_interpolation_refuses_to_extrapolate(scheck, tmp_path):
    """Past the end of a stream, interpolation is a guess. Both sides yield nothing rather
    than inventing a value."""
    specs, pushes = straddling_fixture()
    cpp, _ = compare(scheck, tmp_path, specs, pushes, [0.30], policy="interpolate")
    assert cpp["picks"][0][1]["present"] is False


def test_a_dead_stream_reports_growing_staleness(scheck, tmp_path):
    """The failure this guards: a sensor stops, hold_last keeps contributing its final
    value forever, and the fused estimate drifts with nothing to show why."""
    specs = [(1, 64), (1, 8)]
    pushes = [(1, 0.001, [7.0], True)]
    pushes += [(0, i * 0.01, [float(i)], True) for i in range(1, 51)]
    pushes.sort(key=lambda p: p[1])
    queries = [i * 0.01 for i in range(1, 51)]
    cpp, _ = compare(scheck, tmp_path, specs, pushes, queries, policy="hold_last")
    assert cpp["picks"][-1][1]["staleness"] < -0.4, "staleness did not accumulate"


def test_no_pick_before_the_slow_stream_has_any_data(scheck, tmp_path):
    """Alignment cannot invent a sample that has not arrived."""
    specs = [(1, 8), (1, 8)]
    pushes = [(0, 0.01, [1.0], True)]
    cpp, _ = compare(scheck, tmp_path, specs, pushes, [0.01])
    assert cpp["picks"][0][0]["present"] is True
    assert cpp["picks"][0][1]["present"] is False


def test_validity_is_carried_through_untouched(scheck, tmp_path):
    specs = [(1, 8), (1, 8)]
    pushes = [(1, 0.005, [1.0], False), (0, 0.01, [1.0], True)]
    cpp, _ = compare(scheck, tmp_path, specs, pushes, [0.01])
    assert cpp["picks"][0][0]["valid"] is True
    assert cpp["picks"][0][1]["valid"] is False


def test_interpolation_requires_both_endpoints_valid(scheck, tmp_path):
    """The reference ANDs the two endpoints' validity into the blended sample. A port that
    took only the first would report a good reading built partly from a bad one."""
    specs = [(1, 8), (1, 8)]
    pushes = [(1, 0.0, [1.0], True), (0, 0.5, [0.0], True), (1, 1.0, [3.0], False)]
    pushes.sort(key=lambda p: p[1])
    cpp, _ = compare(scheck, tmp_path, specs, pushes, [0.3], policy="interpolate")
    assert cpp["picks"][0][1]["present"] is True
    assert cpp["picks"][0][1]["valid"] is False


def test_a_query_exactly_on_a_sample_is_not_interpolated(scheck, tmp_path):
    """A degenerate bracket — the same sample on both sides — must return that sample
    rather than dividing by a zero interval."""
    specs = [(1, 8), (1, 8)]
    pushes = [(1, 0.0, [1.0], True), (0, 0.5, [0.0], True), (1, 1.0, [3.0], True)]
    pushes.sort(key=lambda p: p[1])
    cpp, _ = compare(scheck, tmp_path, specs, pushes, [1.0], policy="interpolate")
    assert cpp["picks"][0][1]["present"] is True
    assert float(cpp["picks"][0][1]["values"][0]) == 3.0
    assert cpp["picks"][0][1]["staleness"] == 0.0


@pytest.mark.parametrize("policy", ["hold_last", "nearest", "interpolate",
                                    "drop_if_stale"])
def test_the_reference_names_the_non_causal_policies(policy):
    """Which policies may consult the future, on the Python side only.

    An earlier version of this claimed "the two implementations must agree" and then
    compared `NON_CAUSAL` against a tuple written out by hand in this file — no C++ was
    consulted, so changing `argmind::is_causal` could not fail it. The C++ table is pinned
    by `cpp/tests/test_core.cpp` instead. Two independent assertions of the same table is
    not a comparison, and this docstring now says which one it is.
    """
    from argmind.align import NON_CAUSAL
    expected_non_causal = policy in ("nearest", "interpolate")
    assert (policy in NON_CAUSAL) == expected_non_causal


# ---------------------------------------------------------------------------
# The sweeps, checked in so the discrimination figures are reproducible
# ---------------------------------------------------------------------------
#
# An audit found the numbers in DESIGN.md came from scratchpad scripts that no longer exist,
# and singled out the discrimination count as the one that must be checked in: it is the
# negative control — the only evidence the sweep can produce a disagreement at all — and it
# was also the number that was partly wrong, because `hold_last` and `drop_if_stale` share a
# branch in `pick()` and cannot differ there.


def sweep_fixture(n=4):
    """Four streams at different rates and widths, deliberately offset so that the slow ones
    straddle most query instants — otherwise the sweep measures the refusal to extrapolate
    rather than the policies."""
    specs = [(1, 64), (2, 32), (3, 16), (6, 16)][:n]
    pushes = []
    rng = np.random.default_rng(17)
    for sid, (dim, _cap) in enumerate(specs):
        period = 0.01 * (sid + 1) ** 2
        t = 0.003 * (sid + 1)
        while t < 1.0:
            pushes.append((sid, t, rng.standard_normal(dim), True))
            t += period
    pushes.sort(key=lambda p: p[1])
    return specs, pushes


def test_the_policies_disagree_across_a_wide_sweep(scheck, tmp_path):
    """How often the choice of policy changes the answer, measured rather than asserted.

    `hold_last` and `drop_if_stale` are excluded from the count: `pick()` handles them in one
    branch and never consults the staleness limit, so they are identical by construction and
    counting them would inflate the figure. What distinguishes `drop_if_stale` is the
    stale-drop decision, which `compare()` now checks separately on every pick.
    """
    specs, pushes = sweep_fixture()
    queries = [0.004 + 0.002 * k for k in range(300)]

    values = {}
    for policy in ("hold_last", "nearest", "interpolate"):
        cpp, _ = compare(scheck, tmp_path, specs, pushes, queries, policy=policy)
        values[policy] = [
            tuple(float(v) for v in row[1]["values"]) if row[1]["present"] else None
            for row in cpp["picks"]]

    differing = sum(1 for k in range(len(queries))
                    if len({values[p][k] for p in values}) > 1)
    assert differing > len(queries) // 2, (
        f"the three distinguishable policies returned the same sample at "
        f"{len(queries) - differing} of {len(queries)} instants; a port that collapsed two "
        "of them would pass this suite")
    # 292 of 300, measured by running this. Written down after the fact rather than
    # predicted: the first version of this line guessed 267 and was wrong, which is the
    # third unmeasured count this port has produced and the reason the figures in DESIGN.md
    # now come from tests rather than from prose.
    assert differing == 292, (
        f"the discrimination count moved to {differing}; that is not a failure by itself, "
        "but it is the sweep's negative control and a change means the fixture changed")


def test_the_stale_drop_decision_is_compared_and_actually_fires(scheck, tmp_path):
    """`drop_if_stale` differs from `hold_last` in exactly one decision, and until this test
    that decision was never compared between the two implementations."""
    specs, pushes = sweep_fixture(n=2)
    queries = [0.004 + 0.002 * k for k in range(300)]
    cpp, _ = compare(scheck, tmp_path, specs, pushes, queries, policy="drop_if_stale",
                     max_stale=0.02)
    refused = sum(1 for row in cpp["picks"] for pick in row if pick["refused"])
    assert refused > 0, (
        "no pick exceeded the staleness limit, so the decision under test never fired")
    assert refused < len(queries) * len(specs), "every pick was refused; the limit is useless"


def test_interpolating_in_float32_would_change_the_answer(scheck, tmp_path):
    """The blend widens to float64 and narrows back because the reference does. This counts
    how many of the tested weights would land on a different float32 had it not — so the
    agreement elsewhere is a measurement rather than a formality."""
    specs = [(1, 8), (1, 128)]
    pushes = [(1, 0.0, [1.0], True), (1, 1.0, [3.0], True), (0, 0.5, [0.0], True)]
    pushes.sort(key=lambda p: p[1])
    queries = [k / 500.0 for k in range(1, 500)]
    cpp, _ = compare(scheck, tmp_path, specs, pushes, queries, policy="interpolate",
                     dtype="f4")

    would_differ = 0
    for k, row in enumerate(cpp["picks"]):
        if not row[1]["present"]:
            continue
        w = queries[k]
        wide = np.float32((1.0 - w) * 1.0 + w * 3.0)
        narrow = np.float32(np.float32(1.0 - np.float32(w)) * np.float32(1.0) +
                            np.float32(w) * np.float32(3.0))
        assert row[1]["values"][0] == wide, f"weight {w}: not the widened blend"
        if wide != narrow:
            would_differ += 1
    assert would_differ > 50, (
        f"only {would_differ} of {len(queries)} weights would differ under an f32 blend; "
        "this check has too little grip to be evidence")
