# Sub-agent team

Five agents, each covering a task this project has actually had to do repeatedly, with the
mistakes it actually made encoded in them. They are checked into the repository for the same
reason `.githooks/` is: the discipline is part of the project, not part of one machine.

| agent | when | model | writes? |
|---|---|---|---|
| `evidence-auditor` | before a commit stating a number; after editing DESIGN.md, CLAUDE.md or docs/findings/; when a result looks good | opus | no |
| `numerics-reviewer` | before declaring a precision floor or tolerance; when an estimator misbehaves | opus | scratchpad only |
| `mutation-tester` | after writing a module and its tests; before claiming a suite protects something | sonnet | restores what it edits |
| `parity-verifier` | after any change to `cpp/` or the Python adapters; when a tolerance is under pressure | opus | yes |
| `embedded-reviewer` | after any change under `cpp/`; before adding an ABI entry point | sonnet | no |

## Why these five

Each one exists because the work was done by hand at least twice and the second time was
not easier than the first:

- **evidence-auditor** — the project's standards are its distinguishing feature and they are
  the thing most easily skipped under momentum. Three guessed thresholds, two stale
  documentation claims and one comment describing behaviour the code did not have got
  through in the work so far. All were caught, none by review.
- **numerics-reviewer** — every substantive numerical claim here needed an experiment, and
  the first version of the experiment was wrong three times out of three: confounded
  variables in the fp16 sweep, a trend line fitted to a converging series, and an instrument
  that perturbed what it measured.
- **mutation-tester** — done five times, mechanically identical each time, seventy-five
  mutants. Four survivors across those runs, each of which exposed a real gap in a suite that
  looked complete.
- **parity-verifier** — the Python-to-C++ axis is now the main verification axis, and the
  metric alone took two corrections before it measured the right thing.
- **embedded-reviewer** — the target contract is easy to violate invisibly on a host, where
  everything compiles and nothing is measured on the device.

## Deliberately not included

A **doc-keeper**. Keeping DESIGN.md and CLAUDE.md true is real recurring work, but it is
`evidence-auditor`'s skill applied to prose — "does this claim match the repository" is the
same question as "does this comment match the code". Two agents with the same job means one
gets ignored.

A **benchmark runner**. `argmind/bench.py` already is one: `--record` and `--compare` with
measured noise, exiting non-zero. An agent wrapping a CLI that already reports honestly adds
a layer without adding judgement.

A **general code reviewer**. `/code-review` and `/simplify` cover that, and a project-specific
agent competing with them would fragment where findings come from.

## Using them

The agents overlap by design at one point only: after a change under `cpp/`, run
`embedded-reviewer` and `parity-verifier` together — one asks whether it still fits the
target, the other whether it still computes the right thing, and those are independent
questions.

Their reports are not shown to the user directly, so relay what matters. And treat their
findings the way this project treats any other claim: a sub-agent's conclusion is a claim
about the repository, and claims get verified. That is not a formality — the first real run
of this team produced a report whose cumulative mutant count was wrong, alongside findings
that were all correct.

**Never let an agent run a git command that writes to the working tree.** `checkout`,
`restore`, `stash` and `reset` all discard uncommitted work, and this repository usually has
some. One run reverted an uncommitted fix to HEAD while verifying a mutant and recovered only
because it had read the file first. `mutation-tester` now says so in its own instructions;
say it again in the prompt when a task involves editing sources.

## What the first run found

Worth recording, because it is the argument for the team existing:

- `embedded-reviewer` found a comment that misdescribed the very layout it existed to
  justify, and a 32-bit overflow that **cannot be reproduced on the 64-bit host** — the
  project's rule that host figures are not target figures, applied to behaviour.
- `parity-verifier` confirmed byte identity and found a real bug in the *reference*: after
  a reboot the first write landed on the record it had just loaded, destroying the newest
  copy rather than the spare — the exact failure two slots exist to prevent. Its sweep
  figures were later withdrawn from DESIGN.md and re-derived as checked-in tests
  (`tests/test_cpp_persist_parity.py`: 840 configurations, 624 bit flips, 78 truncation
  lengths, 400 fuzz blobs), because a number from a harness that no longer exists is not
  evidence. **If an agent's figures are worth quoting, check the harness in.**
- `mutation-tester` found that `select_newest` was compiled but never called by any binary
  the suite built, and that a defensive clamp added to the harness had made the library's
  own clamp unreachable.
- `evidence-auditor`, run on the prose describing all of the above, **falsified the summary
  of it** — it built a second surviving mutant in minutes and found ephemeral figures, a
  header with no unit-test coverage, a function with no caller, and four assertions that
  could not fail. Run twice, it then found that the fix the first run asked for had no test
  of its own. It is the agent to run before a commit that states a number, not after.
