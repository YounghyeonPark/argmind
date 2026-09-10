# Sub-agent team

Five agents, each covering work this project has actually had to do repeatedly, with the mistakes it
actually made encoded in them. Checked into the repository for the same reason `.githooks/` is: the
discipline is part of the project, not part of one machine.

| agent | when | builds? | model |
|---|---|---|---|
| `measurement` | to get a number nobody has; before believing one | writes `measure/` | opus |
| `environment` | to give the agent something it cannot see or do; after any change under `argmind/env/` | writes `argmind/env/` | opus |
| `prompt-referee` | after **any** edit to `converse.py`, the widget list, the model or its options | runs measurements | opus |
| `evidence-auditor` | before a commit stating a number; after editing CLAUDE.md or `docs/findings/` | no | opus |
| `mutation-tester` | after writing a module and its tests; before claiming a suite protects something | restores what it edits | sonnet |

## Two build, three check — and the first attempt at this list had five checkers

**Worth recording because the mistake is instructive.** The team was rebuilt on 2026-09-10 for the
project this became, and the first version was `harness-auditor`, `prompt-referee`,
`windows-reviewer`, `evidence-auditor`, `mutation-tester`. **All five verified. None built
anything.** The owner said so in five words and was right.

The cause was inheriting the previous team's silhouette. Those five were also all reviewers, and
changing the contents without questioning the shape is the same error as leaving the old team in
place — which is what the rebuild was for.

There was a bias in the counting too. "Work done by hand at least twice" was tallied from a session
where the building was all done by one hand, so only the *checking* looked delegable. Counted
properly, the construction recurred as much:

| built by hand | times |
|---|---|
| a measurement harness — `pointing`, `clicking`, `ancestry`, `obstruction`, `pointer_theft`, `talking`, `repair` | **7** |
| a capability in `argmind/env/` — `read`, `at`, `attending_title`, `pointing_is_exact`, control patterns | **5** |
| the app and its window | 1 — not yet earned by the rule |

So the two agents that own an area own it **fully**: `measurement` writes harnesses and audits them,
`environment` adds capabilities and reviews them. Reviewing without building is half a job, and the
rules that make a harness trustworthy are the same rules you would check it against.

**The checkers still outnumber the builders, and that is chosen rather than left over.** Four
conclusions were withdrawn in two days and three of them were flattering; the typing was never the
bottleneck. If the balance is wrong, the honest candidate for a third builder is the app and its
conversation — built once so far, which the rule this file uses does not yet justify.

## Why these five

- **measurement** — seven harnesses in two days, mechanically the same job each time. Three of them
  destroyed a baseline, two produced a confounded result, and two printed a display that disagreed
  with the verdict beside it. Every one caught by luck: a latency figure that looked wrong, a goal
  showing "wrong" beside the answer that was right.
- **environment** — five capabilities in two days, and eight distinct platform failures behind them:
  elements dying mid-read, a NULL COM pointer that is not `None`, a DPI space that agrees only on
  this desk, an always-on-top window eating its own clicks.
- **prompt-referee** — prompt edits here are **0 for 5**, and three were nearly recorded as
  improvements because only the flattering half was measured. Choosing and talking are one
  measurement in two halves; this refuses to report half of it.
- **evidence-auditor** — the standards are this project's distinguishing feature and the thing most
  easily skipped under momentum.
- **mutation-tester** — the only remit that did not change. "Can this suite fail" is not a question
  about which code is here.

**`numerics-reviewer` is gone.** It answered precision, conditioning and cancellation questions, and
there is no estimator in this repository any more — the library it served was removed on 2026-09-09.
A good agent for a project that no longer exists. Over the two days that rebuilt this, precision
questions came up **zero** times.

## Deliberately not included

A **conversation reviewer**, for whether the agent is pleasant to talk to. It is the owner's stated
priority — *"중요한건 상호작용 및 의사소통"* — and the one thing here with no objective handle.
`measure/talking.py` and `measure/repair.py` cover what can be counted; the rest would be an agent
with an opinion, and this project has enough of those.

A **doc-keeper**. Keeping CLAUDE.md true is real recurring work, but it is `evidence-auditor`'s skill
applied to prose. Two agents with the same job means one gets ignored.

A **general code reviewer**. `/code-review` covers it, and a project-specific rival would fragment
where findings come from.

## Using them

`measurement` and `prompt-referee` overlap at one point on purpose: the first owns the harness, the
second owns the *verdict* when a change touches what the model sees. After editing `converse.py`,
the second. When a question has no number yet, the first.

`environment` and `measurement` hand work to each other: a harness that needs a capability the
library lacks says which and stops, and a capability with no measurement behind it is a guess with a
docstring.

Their reports are not shown to the user directly, so relay what matters. And treat their findings
the way this project treats any other claim: a sub-agent's conclusion is a claim about the
repository, and claims get verified.

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
  evidence. **If an agent's figures are worth quoting, check the harness in** — and those tests have
  since been removed too, so the sentence applies to itself: the record is at
  `before-removal-2026-09-09`.
- `mutation-tester` found that `select_newest` was compiled but never called by any binary the suite
  built, and that a defensive clamp in the harness had made the library's own clamp unreachable.
- `evidence-auditor`, run on the prose describing all of the above, **falsified the summary of it**.
  Run twice, it then found that the fix the first run asked for had no test of its own. **It is the
  agent to run before a commit that states a number, not after.**

**They are also a dependency.** Two were once launched, both terminated on a spend limit before
reporting, and the figures were measured by the same hand that wrote the code and presented as if
reviewed. The numbers were real; the independence was not.
