# argmind

**A local-model agent you talk to while you work at your PC.** Open-source model, on your machine,
no cloud call. It answers, and it acts only when acting is what you asked for.

    python -m argmind look                  what is on screen right now, and how long reading it took
    python -m argmind talk                  a conversation. It clicks only with --live
    python -m argmind say "이 창 닫아줘"       one exchange and exit
    python -m argmind watch --for build     it speaks when something happens, unasked

Windows for now; the core knows nothing about Windows. Needs [Ollama](https://ollama.com) with a
small text model (`ollama pull qwen2.5:7b`) and `comtypes`.

## The one rule

> **The deterministic side says WHERE. The probabilistic side says WHAT.
> A model never emits a coordinate.**

Measured on this machine before any of it was written. Asking a vision model for a pixel took
**24–82 s** and on a dense screen never answered at all; Windows UI Automation answers the same
question exactly in **41–266 ms**; a text model chooses among named things in well under a second.
The vision model could *see* — its reasoning log read every 14-pixel filename correctly — it simply
could not do arithmetic on what it saw.

So the OS supplies the names and the rectangles, and the model only ever picks a name.

## What it is

    argmind/core/    Widget, Eyes, Hands, the model, and the loop. No platform anywhere in it
    argmind/env/     one environment: UI Automation for eyes, SendInput for hands
    measure/         the goal set, the captured screens, and every answer behind the numbers below

Four separable pieces, because **acting is not a stage that always runs, it is something decided**,
and a thought can start without anybody speaking:

    observe()          read the world
    poll()             anything a Watch cares about?  -> triggers
    decide(trigger)    one model call                 -> what to say, and 0..n things to do
    perform(intent)    each effect re-resolved against the world AS IT IS NOW, or refused

The re-resolution is not caution. Over the second a model call takes, a widget that was listed is
**gone 13 times in 14**, so an effect stores a *name* and never a rectangle: clicking a remembered
rectangle is clicking where something used to be.

## What it gets right, measured

**18 of 28 answerable goals, and 8 of 9 correct declines** — 37 goals against 4 real windows of
12 to 152 widgets, labelled before anything ran, three repeats, the worst counted. Nine goals have
no right answer so the measurement can fail.

**Not good enough to click unsupervised; good enough to propose.** An effect the agent decided on its
own, without standing permission, is **downgraded to speech** rather than refused — it says *"the
build finished, want me to open it?"*. `--allow-unprompted` is a separate word from `--live`.

`docs/findings/choosing-accuracy.md` has the conditions, the failures one by one, and the
hypothesis of mine that the shuffle control refuted.

## History

Until 2026-09-09 this name meant a recursive-least-squares library for on-device adaptation, ported
to C++ for Cortex-M. That tree was removed rather than left unused; it is intact under the tags
`before-removal-2026-09-09` and `argmind-v1`, and `REBUILD.md` says why.
`docs/findings/what-the-old-tree-measured.md` keeps the results that still bear on work here.

## Licence

Apache-2.0 — see `LICENSE` and `NOTICE`. Author: Younghyeon Park. A side project, not a product,
with no support commitment.
