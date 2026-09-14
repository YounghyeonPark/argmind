# SPDX-License-Identifier: Apache-2.0
"""The committed expectation data still matches what the shipped Python produces.

`native/expectations/*.json` is generated from this Python and checked into the repository, because
**the day the Python goes the golden data has to already be data.** Everything the Rust suite knows
about what `converse.py`, `loop.py`, `llm.py` and the eight harnesses did is in those files; a
gitignored copy would leave a fresh clone with a Rust suite that cannot run at all.

While both implementations exist the files have to stay current, and the failure this catches is a
quiet one: a change to the shipped Python that the Rust matches anyway, because the Rust is being
compared against a stale snapshot of the old behaviour. The Rust tests cannot see that — they read
the file, and the file agrees with them.

**This test goes with the Python.** Afterwards the JSONs are frozen data and there is nothing left
to compare them against, which is the point of committing them.
"""
from __future__ import annotations

import hashlib
import pathlib
import runpy
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
HERE = ROOT / "native" / "expectations"


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_every_generator_has_its_committed_json():
    """A generator whose output is not committed is a Rust test nobody can run after the port."""
    generators = {p.stem for p in HERE.glob("*.py")} - {"all"}
    committed = {p.stem for p in HERE.glob("*.json")}
    assert generators, "the expectation generators have vanished"
    missing = sorted(generators - committed)
    assert not missing, f"no committed JSON for {missing} — run python native/expectations/all.py"
    #: And the other way: a JSON with no generator is data nothing can re-derive. That is fine
    #: *after* the Python goes and a mistake before it.
    orphans = sorted(committed - generators)
    assert not orphans, f"{orphans} has no generator"


def test_the_committed_expectations_are_what_the_python_produces_today():
    """Regenerate every one of them and require the bytes back unchanged.

    Run in a subprocess: the generators import `argmind` and `measure` modules and set module-level
    state, and a stale import from an earlier test in the same session would make this compare the
    Python against itself from ten minutes ago.
    """
    before = {p.name: sha(p) for p in sorted(HERE.glob("*.json"))}
    assert len(before) >= 10, f"only {len(before)} expectation files — several are not committed"

    done = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "all.py")],
                          cwd=str(ROOT), capture_output=True)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")[-3000:]

    after = {p.name: sha(p) for p in sorted(HERE.glob("*.json"))}
    moved = sorted(k for k in before if before[k] != after.get(k))
    assert not moved, (
        f"the shipped Python no longer produces the committed expectations: {moved}. "
        f"Either a change was made and the Rust is being checked against a stale snapshot, or the "
        f"files simply need regenerating — run python native/expectations/all.py and read the diff "
        f"before committing it.")
    assert sorted(after) == sorted(before), "a regeneration added or removed a file"


def test_nothing_in_the_expectations_depends_on_a_modification_time():
    """Twice in a row, byte for byte.

    **A checkout rewrites every mtime**, so an expectation that carries one disagrees with itself on
    the next clone — and the disagreement looks like a port defect rather than a fixture defect.
    `sending.py` sorts by name for exactly this reason and records no `newest_run`.
    """
    first = {p.name: sha(p) for p in sorted(HERE.glob("*.json"))}
    done = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "all.py")],
                          cwd=str(ROOT), capture_output=True)
    assert done.returncode == 0, done.stderr.decode("utf-8", "replace")[-3000:]
    second = {p.name: sha(p) for p in sorted(HERE.glob("*.json"))}
    assert first == second, "a generator is not deterministic"

    text = (HERE / "sending.json").read_text(encoding="utf-8")
    assert "newest_run" not in text, "sending.json is carrying a modification time again"


@pytest.mark.parametrize("name", sorted(p.name for p in HERE.glob("*.json")))
def test_each_expectation_is_committed_rather_than_ignored(name):
    """`git check-ignore` rather than reading `.gitignore`, so the answer is git's own."""
    done = subprocess.run(["git", "check-ignore", "-q", f"native/expectations/{name}"],
                          cwd=str(ROOT), capture_output=True)
    #: 1 means "not ignored", which is what this wants. 0 means git would leave it out of the
    #: repository, and the Rust suite would be unrunnable the moment the Python went.
    assert done.returncode == 1, f"native/expectations/{name} is gitignored"
