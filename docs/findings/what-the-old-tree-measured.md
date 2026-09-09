# What the removed tree measured, and where the rest of it is

Between 2026-08-08 and 2026-09-09 this repository was a recursive-least-squares library for
on-device adaptation, ported to C++ for Cortex-M, with a classical-perception demo and an agent app
on top. All of it was removed on 2026-09-09 (`REBUILD.md` step 5, after step 4 measured the rebuilt
core's choosing accuracy at 18 of 28).

**The full record is intact and one command away.** It was not summarised away; it was left where it
belongs, attached to the code it describes:

    git show before-removal-2026-09-09:docs/findings/scope-and-adapters.md
    git show before-removal-2026-09-09:docs/findings/vocabulary-and-labelling.md
    git show before-removal-2026-09-09:docs/findings/map-output.md
    git show before-removal-2026-09-09:docs/findings/objectives.md
    git show before-removal-2026-09-09:docs/findings/platform-and-tiers.md
    git show before-removal-2026-09-09:docs/findings/purpose-and-decisions.md
    git show before-removal-2026-09-09:docs/findings/runtime-library-and-cpp.md
    git show before-removal-2026-09-09:docs/findings/agent-app.md
    git show before-removal-2026-09-09:docs/design/status-log.md
    git show before-removal-2026-09-09:DESIGN.md
    git show --stat before-removal-2026-09-09                     # the whole tree

`argmind-v1` (`5e52f61`) is the same tree before the rebuilt core was added.
`accelerator-final` is an earlier, unrelated hardware-accelerator study, removed 2026-08-08.

This file holds only the results that **still bear on work done here**, because a result that
depends on code nobody can run is history and a result about how measurement goes wrong is a tool.

## Results that transfer

**Precision floors are a property of the algorithm, not of the platform.** Three distinct floors
appeared across one workload set, so no single precision policy works everywhere. Re-derived for RLS
rather than inherited across algorithms, which is the borrowed-assumption error: **fp16 fails the
RLS covariance recursion even at condition number 1**, and fp32 survives without holding
positive-definiteness exactly. Both halves had tests.

**A point cannot stand in for a field.** Building a whole-frame foreground mask from one clicked
point asserts that every other cell is background, and the same demo separately measured that a
difference map with two movers puts its centroid between them. The fix in both cases was a refusal,
not a conversion.

**Global pooling cannot tell shapes apart, and the standard remedy does not escape it.** Pooled
descriptors are shift-invariant at 0.9988 and separate a disc from a bar by a margin of 0.004; the
3×3 grid that discriminates at 0.099 is invariant only to 0.911. Soft bilinear binning — the
textbook answer to that quantisation — matches hard 3×3's invariance with a *smaller* margin.
**Discrimination comes from encoding where inside a crop the content sits**, which is exactly what a
shift disturbs, so the two cannot both be had from pooling.

**Self-training on your own hardened output declines monotonically.** A head trained on its own
thresholded predictions fell 0.938 → 0.839 against a frozen control and went on to 0.717 over 4,000
steps, still declining, while a teacher ceiling read 0.943. The gap is the cost of the *labels*, not
of updating, and it is not the zero-fixed-point collapse a degeneracy detector would catch: the mask
spread into the wrong cells while its area moved *toward* the truth. **A system cannot delete its own
supervisor by imitating itself.**

**A criterion can pass for the wrong reason and look better as conditions get worse.** A retention
test read `back <= first`, which is silent about *which* item came back. Under a real camera load the
target was admitted **zero times in 25** while the criterion passed **23 of 25**, because a
background exemplar cleared the threshold and the criterion rode that same slot twice. The harder
load made the number go up. Whole categorical claims were withdrawn over this.

**On a question where the seed spread exceeds the effect, one seed is not evidence.** A 19-fold
collapse read 1.07 ± 1.37 over ten seeds; a margin that halved at seed 1 tripled at seed 23. Two
published conclusions of mine were withdrawn as single draws. The tidiness that fooled me was a
monotone collapse, not a round number.

**Not-significant is not zero, and sometimes the design could not have reached significance.** A
refinement benefit read +3 of 25, McNemar p = 0.375 — and with five discordant pairs *no outcome*
could have reached 0.05, so that row reported the design's sensitivity rather than the effect.

## How the record itself went wrong, twice

**A `>=` floor is blind to a count rising.** A C++ check count drifted 4,652 → 5,653 unnoticed
because both gates were floors. The repair was to couple the prose to the number with a test that
reads the documents, not to raise the floor.

**Collapsing several lapses into one fixable cause is the self-flattering shape.** An audit found
that "the cause is one mechanism" was wrong — one figure had the floor cause, another had no gate at
all, and a third had a test asserting the right value with prose that never followed. Three unrelated
failures, one comfortable story.

## Sub-agent independence was claimed and not had

An embedded review and an independent parity verification were both launched and both terminated on
a spend limit before reporting; the figures were then measured directly and presented in the same
section. The numbers were real; **the independence was not**, since the mutation list, the sweep and
the memory figures were designed by the same hand that wrote the code. That is the arrangement under
which a flattering result survives, and it is worth recognising the next time a review does not come
back.
