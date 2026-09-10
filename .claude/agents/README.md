# Sub-agent team

Five agents, each covering work this project has actually had to do repeatedly, with the mistakes it
actually made encoded in them. They are checked into the repository for the same reason `.githooks/`
is: the discipline is part of the project, not part of one machine.

| agent | when | model | writes? |
|---|---|---|---|
| `harness-auditor` | before running a new measurement; before quoting a number; when a result is convenient | opus | no |
| `prompt-referee` | after **any** edit to `converse.py`, the widget list, the model or its options | opus | runs measurements |
| `windows-reviewer` | after any change under `argmind/env/`; before trusting a coordinate | opus | no |
| `evidence-auditor` | before a commit stating a number; after editing CLAUDE.md or `docs/findings/` | opus | no |
| `mutation-tester` | after writing a module and its tests; before claiming a suite protects something | sonnet | restores what it edits |

## Why these five, and how the list was chosen

The rule is the one this file has always used: **work done by hand at least twice, mechanically
identical each time.** Counted over the two days that rebuilt this project into an agent app:

| what recurred | times |
|---|---|
| a measurement destroyed its own baseline, or lost the conditions the number depended on | 7 |
| a prompt change judged on one of its two harnesses | 5 |
| a Windows / COM / DPI / z-order defect | 6 |
| a conclusion of mine withdrawn | 4 |
| a test found to be passing for a reason nobody chose | 1 |
| a precision floor, tolerance or estimator question | **0** |

That last row is why `numerics-reviewer` is gone. It answered questions about conditioning,
cancellation and precision floors, and **there is no estimator in this repository any more** — the
recursive-least-squares library it served was removed on 2026-09-09. It was a good agent for a
project that no longer exists.

- **harness-auditor** — the most-earned of the five. Three baselines destroyed by filenames that did
  not carry the model or the prompt, two confounded results from a flag that defaulted to the wrong
  configuration, and twice a display that disagreed with the verdict beside it. Every one was caught
  by luck, and every one would otherwise have entered the record as a finding.
- **prompt-referee** — prompt edits here are **0 for 5**, and three of the five were nearly recorded
  as improvements because only the flattering half was measured. Choosing and talking are one
  measurement; this agent refuses to report half of it.
- **windows-reviewer** — takes the slot `embedded-reviewer` held, for the platform that actually
  exists. Elements dying mid-read, NULL COM pointers that are not `None`, a DPI space that agrees
  only on this desk, an always-on-top window eating its own clicks.
- **evidence-auditor** — unchanged in job and updated in examples. The standards are the project's
  distinguishing feature and the thing most easily skipped under momentum.
- **mutation-tester** — the only one whose remit did not change, because "can this suite fail" is not
  a question about which code is here.

## Deliberately not included

A **conversation reviewer**, for whether the agent is pleasant to talk to. It is the owner's stated
priority — *"중요한건 상호작용 및 의사소통"* — and it is the one thing here with no objective
handle. `measure/talking.py` and `measure/repair.py` cover the parts that can be counted; the rest
would be an agent with an opinion, and this project has enough of those.

A **doc-keeper**. Keeping CLAUDE.md true is real recurring work, but it is `evidence-auditor`'s
skill applied to prose. Two agents with the same job means one gets ignored.

A **general code reviewer**. `/code-review` covers it, and a project-specific rival would fragment
where findings come from.

## Using them

`prompt-referee` and `harness-auditor` overlap at one point on purpose: one asks whether the
*measuring code* is sound, the other *runs* it and reports the trade. After a change to
`converse.py`, the second is the one you want; before trusting a harness you have just written, the
first.

Their reports are not shown to the user directly, so relay what matters. And treat their findings
the way this project treats any other claim: a sub-agent's conclusion is a claim about the
repository, and claims get verified. The first real run of the previous team produced a report whose
cumulative mutant count was wrong alongside findings that were all correct.

**Never let an agent run a git command that writes to the working tree.** `checkout`, `restore`,
`stash` and `reset` all discard uncommitted work, and this repository usually has some. One run
reverted an uncommitted fix to HEAD while verifying a mutant and recovered only because it had read
the file first.

**Never let an agent click anything it did not launch itself.** A GUI test here once drove a window
with `SendInput`, Windows refused the foreground change, and a line of Korean went into somebody's
remote desktop session.

## What the removed agents found, kept because it is the argument for having a team

- `embedded-reviewer` found a comment that misdescribed the very layout it existed to justify, and a
  32-bit overflow that **could not be reproduced on the 64-bit host** — the rule that host figures
  are not target figures, applied to behaviour.
- `parity-verifier` confirmed byte identity and found a real bug in the *reference*: after a reboot
  the first write landed on the record it had just loaded, destroying the newest copy rather than
  the spare — the exact failure two slots exist to prevent. Its sweep figures were later withdrawn
  and re-derived as checked-in tests, because a number from a harness that no longer exists is not
  evidence. **If an agent's figures are worth quoting, check the harness in** — and note that those
  tests have since been removed too, so the sentence applies to itself: the record is at
  `before-removal-2026-09-09`.
- `mutation-tester` found that `select_newest` was compiled but never called by any binary the suite
  built, and that a defensive clamp in the harness had made the library's own clamp unreachable.
- `evidence-auditor`, run on the prose describing all of the above, **falsified the summary of it** —
  it built a second surviving mutant in minutes and found ephemeral figures, a header with no
  unit-test coverage, a function with no caller, and four assertions that could not fail. Run twice,
  it then found that the fix the first run asked for had no test of its own. **It is the agent to
  run before a commit that states a number, not after.**

**They are also a dependency.** Two were once launched, both terminated on a spend limit before
reporting, and the figures were measured by the same hand that wrote the code and presented as if
reviewed. The numbers were real; the independence was not.
