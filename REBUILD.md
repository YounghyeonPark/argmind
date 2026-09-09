# Rebuilding argmind — the plan, before anything is touched

**This file is not committed and nothing has been changed.** It exists to be read and argued with.
The last attempt went straight to execution — 58,567 lines deleted in one commit — and was reverted,
correctly: the decisions inside it were mine, made silently, and a commit statistic is not a place to
discover them.

---

## The fact that drives everything

**The new direction imports nothing from the current library. Not one module of twelve.**

| current module | lines | does the new direction use it? |
|---|---|---|
| `adapt.py` (RLS, NLMS, MLP) | 1,353 | no — nothing learns online any more |
| `guard.py` | 704 | no — a guard exists to protect a learner |
| `bench.py` | 931 | no |
| `backbone.py` | 660 | no — no learned descriptor |
| `pipeline.py` | 500 | no |
| `recognise.py` | 470 | no — this is the line measured into the ground |
| `persist.py` | 423 | no — nothing to persist yet |
| `memory.py` (ExemplarMemory) | 273 | no — the cache it served is gone |
| `align.py`, `stream.py` | 477 | no — the screen is read synchronously |
| `stack.py` | 216 | no |

That is not a judgement on the work. **None of it was ever measured failing** — the runtime library,
the C++ core, the parity harnesses and the QEMU verification on M4 and M55 are sound. They are simply
not what an agent that reads widgets from the OS and asks a local model which one is needed for.

## Why rebuild at all — the measurement

| asked of | question | measured on this machine |
|---|---|---|
| `qwen3-vl:8b` | *where is this button, in pixels?* | **24–82 s**; on a dense screen **no answer at all** |
| Windows UI Automation | *what is on screen and where?* | **41–266 ms**, exact, in the UI's own language |
| `qwen2.5:7b` | *which of these named things did they mean?* | **67–85 ms** to first token |
| the whole loop, warm | read → choose → click | **602 ms** end to end |

The vision model was not failing to *see*: its own reasoning log read every 14-pixel filename
correctly. It failed at arithmetic on what it saw — 13,000 characters computing
`row index × row height`, never finishing.

So the rule the rebuild is organised around, which a test enforces by walking the AST:

> **The deterministic side says WHERE. The probabilistic side says WHAT.
> A model never emits a coordinate.**

And the reason the old perception line is not worth continuing, in one sentence: **it was squeezing
meaning out of pixels in a domain that offers no ground truth.** Object identity retained 0–2 of 25
under load, every threshold from 0.60 to 0.98 gave 0/25, and no pooling escaped the
invariance-versus-discrimination frontier. A PC has ground truth and it is free.

---

## Decision 1 — what happens to the library and the C++ core

**Three options. Last time I took (a) without asking.**

**(a) Remove from the tree, keep at the tag.** 46 KB of source and 13,005 lines of C++ leave; a
`git show argmind-v1:cpp/...` brings any of it back. The argument is this project's own, written in
its CLAUDE.md about the accelerator project: *forty files of a different project tax every search
from here on.* With 155 files and 58,902 lines, every grep, every test run and every context window
carries work the new direction does not use.

**(b) Keep it in the tree, unused.** Nothing is lost, nothing has to be decided today, and it can be
picked up if the direction changes again. The cost is the tax above, paid every day, plus 1,217
tests that must keep passing or be marked as not-run.

**(c) Keep a subset.** The honest subset is *empty* — the table above shows nothing is imported. Any
"we might want this later" list is a guess, and this project's standard is that a guess is not a
reason.

**My recommendation is (a)**, and the reason is not tidiness: the 1,217-test suite is the thing that
made every number in this repository trustworthy, and a suite that covers code nothing uses stops
being a gate and becomes a cost. But it deletes work that took a month and was done well, so it is
your call and not mine.

## Decision 2 — three environments, and how far the rule reaches — **ANSWERED**

Three were named: PC, phone, robot. **The phone comes later; the PC is first.** The robot follows the
phone.

The question that matters is not which to build but **whether the rule survives the move**, because
it was derived from PC measurements and generalising it unmeasured is the error this project keeps
catching.

| environment | the source of WHERE | status |
|---|---|---|
| **PC** | UI Automation | **measured** — names + exact rectangles, 41–266 ms, in the UI's own language |
| **phone** | `AccessibilityService` / `uiautomator` | **unmeasured.** An emulator was booted to check and shut down when the phone was deferred |
| **robot** | *nothing* | **measured failing** — v1's camera work retained 0–2 of 25 under load |

**The first two share a spine, and not by luck.** Every GUI platform ships an accessibility tree —
Windows UIA, Android `AccessibilityService`, iOS XCUITest — because screen readers require one. So
"names and rectangles, free and exact" is a property of *GUIs*, not a quirk of Windows. That is an
argument, not a measurement, and it is labelled as one: the phone number is not in hand.

**The robot is different in kind.** A room has no accessibility tree. But the reason v1 failed there
deserves stating precisely, because "cameras don't work" is the wrong lesson: it failed with an
**untrained descriptor and a hand-set threshold**, and every threshold from 0.60 to 0.98 gave 0/25. A
trained detector or a grounding model is a different instrument. **It is also unmeasured here**, so
the honest position is that the robot's WHERE source is *unknown*, not *impossible*.

### The shape: one core, an interface per environment — **decided**

I argued for deferring the abstraction. That was wrong for a reason worth writing down: **the
interface is not a guess.** Both accessibility trees return the same four things, so the shape is
determined by what the platforms already provide, not by what a future might want.

| | Windows UIA | Android `uiautomator` |
|---|---|---|
| what it is called | `Name` | `text` / `content-desc` |
| what kind of thing | `LocalizedControlType` | `class` |
| where it is | `BoundingRectangle` | `bounds` |
| is it visible | `IsOffscreen` | `visible-to-user` |

So the core is written against two small protocols and nothing else:

    Eyes    foreground() -> list[Widget]        Widget = (name, role, rect, window)
    Hands   click(widget) / type_text(s) / press(key)

    argmind/
      core/      widget, choose, llm, loop     knows no platform at all
      env/
        windows.py    UIA + SendInput           built now, measured
        android.py    uiautomator + input tap   later
        robot.py      -- see below

**Everything above the protocols already satisfies this today**: `loop.py` calls `foreground()`,
`choose.py` reads `.name` and `.role`, `hands.py` reads `.centre`. Not one of them knows what UIA is.
The split is therefore cheap now and churn later, which is the argument for doing it from the first
commit.

### Where the third environment does not fit, and saying so rather than pretending

**A room has no accessibility tree, so `Eyes` cannot be implemented for a robot the way it is for a
screen.** Anything that filled it would be a *detector* — a thing that guesses names and boxes — and
the core would then be receiving probabilistic output through an interface whose whole contract is
that it is exact. That is the rule inverted, quietly, at the seam:

> The deterministic side says WHERE. The probabilistic side says WHAT.

A robot backend has to answer WHERE from somewhere, and today nobody here knows where. **The
protocols are shared with the phone honestly and with the robot provisionally**, and if the robot
turns out to need a different core, that is a finding and not a failure of the design. v1 spent a
month proving that a room does not hand you ground truth; the second-cheapest lesson available is not
to bury that in a `robot.py` that implements `Eyes` badly.

**One fork to be aware of when the phone does arrive**: a phone cannot run an 8B model well, so
either this machine drives it over ADB, or a much smaller model runs on it. That decision changes the
structure, and it is deferred with the phone rather than guessed at now.

## The goal — **set**

> **A real-time conversational agent app for the PC. Highest priority. Everything else waits.**

### What "conversational" adds that is not built and not measured

The skeleton was a command executor: `do "close this window"` picks a widget, clicks it, prints a
line. **It never says anything back.** That is not conversation, and the gap is structural rather
than cosmetic — the loop has no reply step and the model is asked only one kind of question.

So the model's job becomes two questions at once, and **one call, not two**, because a round trip is
the thing being spent:

    hear ──▶ one call ──▶ {"say": "..."}                 answered, nothing touched
                       └─ {"click": <id>, "say": "..."}  acted, and said what it did

**The latency budget says this is affordable.** Conversation is text, and text on this machine was
measured at **67–85 ms to first token** — the expensive part was ever only the vision, which this
design removed. A reply that begins inside a tenth of a second is what "real-time" can mean here.

**Voice or text is an open sub-decision.** Text costs nothing and works today. Voice needs a capture
library — none is installed, which the previous app recorded as *absent rather than unfinished* —
and the cascade is known: Silero VAD, a small STT, a small TTS. It is deferred rather than assumed.

### The criterion — proposed, since asking twice produced none

Not asking a third time. Here is one in a shape that can fail; change the numbers or the wording, but
**something like this has to exist before step 2**, because every phase of this project that drifted
drifted from having ambition before a criterion:

> **A month from now this works if**, on my real screen, I can say a thing in my own words and it
> replies within one second and does the right thing **nine times in ten**, across twenty different
> requests I did not choose in advance.
>
> **It has failed if** the right-thing rate is below that, or a reply routinely takes more than a
> second, or I find myself phrasing requests to suit it.

Every clause is measurable. The last one is there because it is the way this kind of system usually
fails while looking like it works: the person adapts, and the score stays high.

**Two of the three numbers are already in hand** — reply latency 67–85 ms, action latency 0.2 s of
UIA. **The rate is not**, and the first live attempt at it went one right, one wrong. That is step 4,
and it is deliberately before anything is deleted.

---

## The path, once those are answered

Staged, so each step is reviewable and revertible on its own — which is the part that was missing
last time.

| step | commit | undo |
|---|---|---|
| 1 | tag the current state (`argmind-v1` already exists at `5e52f61`) | — |
| 2 | `core/` — `Widget`, the two protocols, `converse`, `llm`, `intent`, `loop`, and tests against fakes. **No platform code at all**, so it runs anywhere | `git revert` |
| 3 | `env/windows.py` — UIA and SendInput behind the protocols, plus the CLI | `git revert` |
| 4 | measure choosing accuracy on tens of goals — the number that is missing | — |
| 5 | *only then*, remove the old tree if decision 1 says so | `git revert` |

Steps 2 and 3 are **committed** at `1a87686` — 36 core tests, no display, no model, no network.
**Step 4 is measured**, and it is the number the plan said would decide whether the rebuild is worth
doing.

### Step 4: 18 of 28 answerable goals right, 8 of 9 correct declines

37 goals against 4 real captured screens (12, 21, 123 and 152 widgets), labelled before anything ran,
3 repeats each, the worst counted. Per-run 19 / 18 / 18, so it is not a single draw. Nine goals have
no right answer, so the measurement can fail. `docs/findings/choosing-accuracy.md` carries the whole
thing; the short version:

**It is not good enough to click unsupervised, and it is good enough to propose.** Eight wrong
answers in 37 goals, one of them where the right response was to do nothing, is a bad click most
working hours — and 18 right with 8 of 9 declines is a long way from chance on lists of 12 to 152.
That is the same place the permission default landed for safety reasons, which is worth noting
because the two could have disagreed.

**A hypothesis of mine was refuted by its own control.** Six of seven wrong answers sat 1–10 places
from a correct one, which looked like the vision model's arithmetic failure in a new guise and
pointed at a fix I liked. Shuffling each screen's order expands the median gap from 4 places (3.3% of
the list) to 33 (26.8%): the errors were **semantic** neighbours, and a toolbar puts related controls
next to each other, so the two were confounded. Shuffling also costs accuracy — 18 → 15 right, 8 → 13
wrong — which says the interface's own order carries information and the list must not be re-sorted.

**Asking for a name instead of an index is safer and less capable**: wrong 8 → 3, missed 3 → 12,
right 18 → 15. A real trade, not an improvement; nothing was changed on it.

### What step 2 turned out to be, once "sequential" was refused

The first draft welded four things into a pipeline: hear → look → decide → act. That was wrong, and
the reason it was wrong is that **acting is not a stage that always runs, it is something decided**.
A thought can also start without anybody speaking. So the core is four separable pieces:

    observe()          read the world                41–266 ms measured, needs no model
    poll()             anything a watch cares about?  → 0..n triggers
    decide(trigger)    one model call                 → an Intent: what to say, 0..n to do
    perform(intent)    each effect re-resolved against the world AS IT IS NOW, or refused

**`Trigger → Intent → Effects`**, where a `Trigger` is either `Said` (a person spoke) or `Perceived`
(a `Watch` noticed something), and an `Intent` carries a sentence and between zero and eight effects.
Zero is the common case and the important one.

**A watch has a floor, and that is measured rather than tuned.** Perception changes in 13 seconds out
of 14; a model call costs 0.3–2 s. "Think whenever the screen changes" is a model that never stops
and an agent that never listens, so `Watch.gap` defaults to 3 s and is part of the design.

### The brake the second half needed

Every brake built for the first half — `live`, a rate limit, a guard before each action — assumes
*"you asked, it does one thing"*. None of them covers *"it decided, repeatedly, while you were
reading something else"*. So:

**an unprompted effect without standing permission is downgraded to speech, not refused.** It becomes
*"the build finished — want me to open it?"*, and the person answers. `Session.may_act_unprompted` is
off by default and `--allow-unprompted` is a separate CLI word from `--live`, because they are
different risks. `report()["proposed_only"]` counts what the default cost.

### A seventh way the harness lied, found here rather than measured

`Reply.json()` was `re.search(r"\{.*?\}")` — correct for exactly as long as a reply contained no
nested object, and wrong the moment the answer grew a list of things to do. It matched up to the
*inner* brace, failed to parse, returned `None`, and **every action silently disappeared while the
model was answering perfectly**. Same shape as the other six: a plausible zero produced by the
harness, invisible from the outside. Pinned by `test_a_NESTED_reply_is_read_whole`.

Step 2 before step 3 on purpose: if the core cannot be written and tested without naming a platform,
the split is not real, and finding that out costs one commit rather than a restructure.

Step 4 before step 5 on purpose. **Choosing accuracy is unmeasured**, and the second live attempt
already got one wrong: *"open the file menu"* picked the document rather than the File menu. If that
number comes back poor, the rebuild is not worth doing and nothing has been deleted.

## The safety net, verified

    argmind-v1                    5e52f61  the current state, everything intact
    rebuild-attempt-2026-09-08    efa8a09  the reverted rebuild, recoverable
    accelerator-final                      the 2026-08-08 project, untouched

Nothing has been pushed. The remote still ends at `c524df8`.

---

## What carries regardless of every decision above

Not code. These:

- **The standards.** Measured distinguished from estimated; a model that cannot fail is not evidence;
  a uniformly tidy result means the measurement is broken; corrections that favour you get the
  harder look. They earned their place again this week.
- **Six ways a local-model harness lies**, each met on this machine, each producing a plausible zero:
  `format: "json"` silences the model; too small a `num_predict` truncates it mid-thought; the answer
  is on `message.content`, not `response`; `num_ctx` defaults to 4096 and a 1440p screenshot plus a
  reply is 4,096 tokens; instructing the coordinate convention starts a 13,000-character argument
  with itself; and prefill measured warm is 25 ms against 0.9–5.5 s cold. **Three would have killed a
  direction that works, one would have kept a direction that does not, and every one pointed the way
  I was already leaning.** A seventh joined them during the rebuild — a non-greedy brace match that
  silently dropped every action as soon as the reply contained a nested object.
- **The rule**, and the measurements behind it.
