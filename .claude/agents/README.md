# Sub-agent team

**Five agents, all of them build.** Each owns a part of the tree, and between them they cover it
with no gaps and no overlaps.

| agent | owns | model |
|---|---|---|
| `architect` | the shape — `contract.py`, `loop.py`, `llm.py`, the AST rules, where a new thing goes | opus |
| `conversation` | `converse.py`, `intent.py` — what it says, how it reads a turn, repair | opus |
| `environment` | `argmind/env/` — capabilities behind Eyes and Hands | opus |
| `app` | `argmind/app.py`, `tests/test_app.py` — the window a person touches | opus |
| `measurement` | `measure/` — the instruments every decision here rests on | opus |

**Architecture leads.** `architect` drew the boundaries the other four work inside, and it is the
only one that may change a contract. When something does not obviously belong anywhere, that is an
architecture question and not a coding one.

## How this list was arrived at, including two wrong versions

**Version one** was the previous project's team with the dead entries removed: `evidence-auditor`,
`numerics-reviewer`, `mutation-tester`. `numerics-reviewer` answered precision and conditioning
questions for a recursive-least-squares library that had been deleted the day before.

**Version two** replaced the contents and kept the silhouette: `harness-auditor`, `prompt-referee`,
`windows-reviewer`, `evidence-auditor`, `mutation-tester`. All five verified and **none built
anything**. The owner said so in five words. Two causes, both mine: the previous team was also all
reviewers and I did not question the shape, and "work done by hand at least twice" was tallied from
a session where one hand did all the building, so only the checking looked delegable.

**Version three** — this one. The owner: *"테스트는 나중에 해도 되니 개발위주의 에이전트 구성을
할것. 아키텍쳐 디자인이 매우 중요함."* The rule that chose the earlier lists is set aside
deliberately; the team is organised by **what it owns**, not by how often the work has recurred.

## The checking did not disappear, it moved inside

`prompt-referee`, `evidence-auditor` and `mutation-tester` are gone from the roster. Two of their
three jobs are now clauses in the builders' own briefs, because on this project measuring is part of
building rather than a review that follows it:

- **`conversation` runs both harnesses on every change it makes.** Prompt edits here are 0 for 5 and
  three were nearly recorded as improvements; that gate could not be optional.
- **`environment` owes every capability a docstring with the measurement that justified it.** A
  capability without one is a guess with a docstring.
- **`measurement` builds the instruments and knows the seven ways they have lied here** — three
  destroyed baselines, two confounded runs, two displays that disagreed with the verdict beside
  them.
- **`architect` must find the reading where the contract does not change**, before changing it.

**Mutation testing is deferred, not judged.** "Can this suite fail" is a real question and the last
run of it caught a test passing for a reason nobody had chosen. `git show` at any commit before
2026-09-10 brings the agent back when tests come round again.

**The cost of dropping a dedicated auditor is real and stated rather than waved at.** Four
conclusions were withdrawn in two days and **three of them flattered me**. A standard carried by the
person doing the work is weaker than one carried by somebody else, and this project's own history
says so. The mitigation is that every brief above names the specific mistakes that agent's area has
produced — not the standards in the abstract, which is what gets skipped under momentum.

## Boundaries, because the split is the point

    architect      argmind/core/contract.py, loop.py, llm.py, tests/test_core.py
    conversation   argmind/core/converse.py, intent.py
    environment    argmind/env/
    app            argmind/app.py, tests/test_app.py
    measurement    measure/, the scratchpad

Each brief says what its agent may **not** touch. `environment` and `app` never edit `core/`;
`measurement` never edits `argmind/`; `conversation` hands anything about `loop.py`'s shape to
`architect`. **An agent that finds it needs something outside its area names it and stops** — that
hand-off is the design working, not a failure.

**Never let an agent run a git command that writes to the working tree.** `checkout`, `restore`,
`stash` and `reset` all discard uncommitted work, and this repository usually has some. One run
reverted an uncommitted fix to HEAD while verifying a mutant and recovered only because it had read
the file first.

**Never let an agent click anything it did not launch itself.** A GUI test here once drove a window
with `SendInput`, Windows refused the foreground change, and a line of Korean went into somebody's
remote desktop session.

Their reports are not shown to the user directly, so relay what matters. A sub-agent's conclusion is
a claim about the repository, and claims get verified.

## What the removed agents found, kept because it is the argument for having a team at all

- `embedded-reviewer` found a comment that misdescribed the very layout it existed to justify, and a
  32-bit overflow that **could not be reproduced on the 64-bit host** — host figures are not target
  figures, applied to behaviour.
- `parity-verifier` confirmed byte identity and found a real bug in the *reference*: after a reboot
  the first write landed on the record it had just loaded, destroying the newest copy rather than
  the spare — the exact failure two slots exist to prevent. Its figures were later withdrawn and
  re-derived as checked-in tests, because a number from a harness that no longer exists is not
  evidence. **If an agent's figures are worth quoting, check the harness in.**
- `mutation-tester` found that `select_newest` was compiled but never called by any binary the suite
  built, and — on 2026-09-10 — that a test pinning role-aware lookup passed because a *second* role
  check in a tiebreak was quietly carrying the case alone.
- `evidence-auditor`, run on the prose describing all of the above, **falsified the summary of it**.
  Run twice, it found that the fix the first run asked for had no test of its own.

**They are also a dependency.** Two were once launched, both terminated on a spend limit before
reporting, and the figures were then measured by the same hand that wrote the code and presented as
if reviewed. The numbers were real; the independence was not.
