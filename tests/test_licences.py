# SPDX-License-Identifier: Apache-2.0
"""Only things with no open-source licence problem, held by a test rather than remembered.

The owner's rule, 2026-09-14: **오픈소스라이센스에 문제 없는것만 사용할것.** A rule nothing enforces
is a rule nobody keeps, and this project has the receipt — the `NOTICE` it shipped until today
attributed OSQP, CVXPY and SciPy for a recursive-least-squares library that was removed on
2026-09-09, and named none of the five things actually imported. Nobody noticed for five days,
because nothing looked.

So: every dependency declared in `pyproject.toml` or in a `Cargo.toml` under `native/` must appear
in `docs/licences.toml` with a licence from its `allowed` list. **Adding a dependency fails the
suite until a line is added there**, which is the whole mechanism — the cost of a new dependency is
paid at the moment it is taken on, by the person taking it on, rather than at the moment somebody
audits.

What this does **not** check, said plainly rather than implied:

* **Transitive dependencies.** `windows-rs` pulls a dozen crates and `transformers` pulls more.
  This checks what this project chose, not what those chose. `cargo tree` and `pip download` are
  the tools for that and neither runs offline in a test.
* **That a project's declared licence is true.** It reads what each declares. A file that says MIT
  and contains somebody else's GPL code is not something a test can see.
* **Model weights and services.** They are recorded in the `[external]` section and are not
  dependencies: nothing here redistributes them.

Needs no network, no model and no installed package — it reads two kinds of text file.
"""

from __future__ import annotations

import pathlib
import re
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs" / "licences.toml"


def record() -> dict:
    return tomllib.loads(RECORD.read_text(encoding="utf-8"))


def recorded_names(d: dict, *, shipped_only: bool = False) -> dict[str, str]:
    """Every dependency the record names, flattened to `name -> licence`.

    `shipped_only` drops `[python.dev]`. A NOTICE lists what the distributed work depends on; the
    test runner and the linter are neither included nor required by anybody installing this, and
    naming them there would make the file less true rather than more complete. They are still
    recorded and still have to be on the allowlist.
    """
    out: dict[str, str] = {}
    for section in ("python", "rust"):
        block = d.get(section, {})
        #: `[python.voice]` and friends are nested a level deeper than `[rust]`.
        groups = list(block.items()) if section == "python" else [(section, block)]
        for group_name, group in groups:
            if shipped_only and group_name == "dev":
                continue
            for name, meta in group.items():
                out[name.lower().replace("_", "-")] = meta["licence"]
    return out


def declared_python() -> set[str]:
    p = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    wanted = list(p.get("dependencies", []))
    for extra in p.get("optional-dependencies", {}).values():
        wanted += extra
    #: `comtypes>=1.4; sys_platform == 'win32'` -> `comtypes`
    return {re.split(r"[<>=!;\[ ]", spec.strip(), maxsplit=1)[0].lower() for spec in wanted if spec.strip()}


def declared_rust() -> set[str]:
    out: set[str] = set()
    for manifest in sorted((ROOT / "native").glob("*/Cargo.toml")):
        d = tomllib.loads(manifest.read_text(encoding="utf-8"))
        for table in ("dependencies", "dev-dependencies"):
            for name, spec in d.get(table, {}).items():
                #: A path dependency is this project's own crate, not a third party.
                if isinstance(spec, dict) and "path" in spec:
                    continue
                out.add(name.lower().replace("_", "-"))
    return out


def test_every_dependency_is_recorded_with_a_licence():
    """**The mechanism.** A new dependency has to be written down before the suite goes green."""
    d = record()
    known = recorded_names(d)
    missing = sorted((declared_python() | declared_rust()) - set(known))
    assert not missing, (
        f"{missing} are declared as dependencies and are not in {RECORD.name}. Add each with the "
        "licence its project declares — the rule is that only things with no open-source licence "
        "problem are used, and this is where that is written down")


def test_every_recorded_licence_is_on_the_allowlist():
    """Recording a dependency is not enough; the licence has to be one this project accepts."""
    d = record()
    allowed = set(d["allowed"])
    bad = {name: lic for name, lic in recorded_names(d).items() if lic not in allowed}
    assert not bad, (
        f"{bad} carry licences that are not on the allowlist. If one of them should be allowed, "
        f"adding it to `allowed` in {RECORD.name} is the decision — make it deliberately, in a "
        "commit that says why")


def test_the_allowlist_has_no_copyleft_and_the_refusals_say_why():
    """**Protects the allowlist itself**, which is otherwise a list anybody can widen quietly.

    The refusal list is not decoration: it is where the reasoning lives, so a future reader does
    not have to re-derive why AGPL or the CUDA EULA were turned down. A refusal with no reason is
    an opinion; one with a reason is a decision.
    """
    d = record()
    forbidden = ("GPL", "AGPL", "LGPL", "CC-BY-NC", "EULA", "Proprietary", "NonCommercial")
    slipped = [lic for lic in d["allowed"] if any(f.lower() in lic.lower() for f in forbidden)]
    assert not slipped, f"{slipped} reached the allowlist"
    assert d["refused"], "the refusals are where the reasoning lives"
    for licence, why in d["refused"].items():
        assert len(why.strip()) > 20, f"{licence} is refused without a reason"


def test_the_NOTICE_names_what_is_actually_used():
    """**The defect this file exists because of.**

    Until today `NOTICE` attributed OSQP, CVXPY and SciPy — a removed library's dependencies — and
    named none of the five things this tree imports. It shipped with the package, so it was not an
    internal note: it was a false statement to anybody who installed this.

    Checked by name against the record rather than by eye, and both directions: a dependency
    missing from `NOTICE` fails, and so does a `NOTICE` still naming something long gone.
    """
    notice = (ROOT / "NOTICE").read_text(encoding="utf-8")
    for name in recorded_names(record(), shipped_only=True):
        #: `pillow` is listed as `Pillow`, `torch` as `PyTorch` — match case-insensitively on the
        #: stem, which is what a reader would look for.
        stem = name.split("-")[0]
        assert re.search(re.escape(stem), notice, re.IGNORECASE), (
            f"{name} is a dependency and NOTICE does not mention it")
    gone = ("OSQP", "CVXPY", "perf_model", "PHASE0_FINDINGS", "taped out", "silicon")
    still_there = [g for g in gone if g.lower() in notice.lower()]
    assert not still_there, (
        f"NOTICE still names {still_there} from the library removed on 2026-09-09")
