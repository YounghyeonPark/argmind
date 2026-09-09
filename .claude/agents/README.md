# Sub-agent team

Three agents, each covering a task this project has actually had to do repeatedly, with the
mistakes it actually made encoded in them. They are checked into the repository for the same
reason `.githooks/` is: the discipline is part of the project, not part of one machine.

**There were five.** `parity-verifier` compared Python against a C++ core and `embedded-reviewer`
reviewed changes against a Cortex-M target contract; both were removed on 2026-09-09 with the port
they existed to check. What they found is kept below, because the argument for having a team at all
is what they caught.

| agent | when | model | writes? |
|---|---|---|---|
| `evidence-auditor` | before a commit stating a number; after editing CLAUDE.md, REBUILD.md or docs/findings/; when a result looks good | opus | no |
| `numerics-reviewer` | before declaring a precision floor or tolerance; when an estimator misbehaves | opus | scratchpad only |
| `mutation-tester` | after writing a module and its tests; before claiming a suite protects something | sonnet | restores what it edits |

## Why these three

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

## Deliberately not included

A **doc-keeper**. Keeping CLAUDE.md true is real recurring work, but it is
`evidence-auditor`'s skill applied to prose — "does this claim match the repository" is the
same question as "does this comment match the code". Two agents with the same job means one
gets ignored.

A **benchmark runner**. There is no longer a benchmark harness to wrap — and when there was, the
reason still applies: an agent wrapping a CLI that already reports honestly adds a layer without
adding judgement.

A **general code reviewer**. `/code-review` and `/simplify` cover that, and a project-specific
agent competing with them would fragment where findings come from.

## Using them

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

Worth recording, because it is the argument for the team existing. The first two entries are from
the agents that were removed with the C++ port; they are kept because **what they caught is the
reason the remaining three are checked in**, and neither finding depends on the code still being
here.

- `embedded-reviewer` found a comment that misdescribed the very layout it existed to
  justify, and a 32-bit overflow that **cannot be reproduced on the 64-bit host** — the
  project's rule that host figures are not target figures, applied to behaviour.
- `parity-verifier` confirmed byte identity and found a real bug in the *reference*: after
  a reboot the first write landed on the record it had just loaded, destroying the newest
  copy rather than the spare — the exact failure two slots exist to prevent. Its sweep
  figures were later withdrawn from DESIGN.md and re-derived as checked-in tests
  (840 configurations, 624 bit flips, 78 truncation lengths, 400 fuzz blobs), because a
  number from a harness that no longer exists is not evidence. **If an agent's figures are
  worth quoting, check the harness in** — and note that those checked-in tests have since
  been removed too, so the sentence now applies to itself: the record is at
  `before-removal-2026-09-09`.
- `mutation-tester` found that `select_newest` was compiled but never called by any binary
  the suite built, and that a defensive clamp added to the harness had made the library's
  own clamp unreachable.
- `evidence-auditor`, run on the prose describing all of the above, **falsified the summary
  of it** — it built a second surviving mutant in minutes and found ephemeral figures, a
  header with no unit-test coverage, a function with no caller, and four assertions that
  could not fail. Run twice, it then found that the fix the first run asked for had no test
  of its own. It is the agent to run before a commit that states a number, not after.
