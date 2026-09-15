# Porting the core — what speed does not decide, 2026-09-13

> 컴파일 및 빌드시간은 중요하지 않음. 코어는 pc 모바일 로봇등 다양한 환경을 고려했을때 가장 빠른
> 퍼포먼스를 낼수있도록 구현해야 함.

The direction is right and the reason under it is not available. **Speed does not select the core's
language, because on the PC there is no measurable speed to win**, and the environment the direction
names third — the robot — is one this repository has already excluded twice, for reasons no language
changes. What is left is one real argument, it is about portability rather than performance, and it
depends on a fork nobody has decided yet.

This is the accounting. Nothing was ported and no source was touched. `python -m pytest`: **148
passed in 16.61 s**, before and unchanged.

## The answer, first

1. **Do not port yet.** Not because a native core is wrong, but because its only live justification
   is an Android phone running the core on the device, and whether the phone runs the core at all is
   undecided. That decision costs an afternoon of measurement; the port costs the rest of `measure/`
   being re-pointed. Decide in that order.
2. **Speed is not a reason.** Measured below: the whole Python side of one turn is 0.032 ms against
   a 610 ms model call, and the cheapest honest proxy for the FFI seam a port would add — copying one
   screen across it — is **0.095 ms, three times the work it would replace**.
3. **The robot is answered, not open.** It fails the WHERE rule and, since today, the WHETHER rule,
   independently. A native core would be a fast core that still cannot host a robot. `contract.py`
   does not change; the robot wants a different core and this one should stop naming it as a target.
4. **If a port happens, PyO3 is the right mechanism** — it is the only option on the table that keeps
   `measure/` driving the shipped core rather than a Python twin. The bill it comes with is itemised
   below and it is not small: two AST rules lose their home in `pytest`, four hashing call sites and
   one whole-tree A/B mechanism assume the library is text on disk, and `--prompt-check`'s twelve
   mutation rows assume the library's source can be edited in memory and never written.

## What one turn actually costs

Measured 2026-09-13 on this machine, Python 3.14.5, no model and no display. n=200 per row after 20
warm-up iterations, the densest captured screen (`measure/screens/system-information.json`, 152
widgets), a twelve-line history, and a canned three-effect reply. The script is not kept; every row
is one call in a loop around `time.perf_counter` and is re-derivable in ten minutes.

| | median | p95 | max |
|---|---|---|---|
| `converse()` whole — render, `Reply.json()`, build the `Intent` | **0.032 ms** | 0.035 | 0.039 |
| &nbsp;&nbsp;of which `describe()`, the 152-line widget list | 0.019 | 0.021 | 0.044 |
| &nbsp;&nbsp;of which `Reply.json()` | 0.008 | 0.008 | 0.011 |
| `Session.exchange()` cold — observe, decide, no history | 0.040 | 0.041 | 0.045 |
| `Session.exchange()` warm — twelve turns of history carried | **0.071 ms** | 0.089 | 0.114 |

The rendered prompt was 6,093 characters. This re-derives the owner's 0.03 ms independently, on a
prompt of nearly the same size, and adds the warm number, which is the one a shipped turn pays.

**Against what the turn waits for** — model 610 ms, transcription 234, UIA 150, synthesis 25, and
then ~3,100 ms of the person listening — 0.071 ms is **0.007% of the machine's second and 0.0017% of
the turn**. A core that took *zero* time would save 0.07 ms: nothing the person can perceive, nothing
any instrument in `measure/` can resolve, and below the 0.024 ms spread of this measurement's own
median-to-max.

### The seam costs more than the work

A PyO3 core has to be handed the screen. These are Python-side proxies for that cost, same conditions
— **not measured PyO3 crossings**, because no extension was built; each is a lower bound on the
corresponding shape of seam.

| | median |
|---|---|
| 152 `Widget`s to a list of tuples (the cheapest thing PyO3 can extract) | 0.005 ms |
| 152 tuples back to `Widget` objects | 0.057 ms |
| 152 `Widget`s to JSON and back (a serialising seam) | **0.095 ms** |
| a trivial C-extension call, for scale | 32 ns |

`Session` reads the screen up to three times a turn — once to ask, once to re-resolve the target
immediately before acting, once to confirm — so a serialising seam costs ~0.29 ms a turn to save
0.07 ms. **A Rust core makes the PC slower**, by an amount too small to matter in either direction.
That is the whole speed argument, in both directions, and it is finished.

Estimated, not measured: the tuple form (0.005 ms) is what a competent PyO3 boundary would use, so a
real port lands at roughly break-even on the PC. Break-even is the ceiling here, not the floor.

## What is core, and what only lives in `core/`

Lines split with `ast`, counting a docstring line as prose:

| | code | docstring | comment | prose share |
|---|---|---|---|---|
| `contract.py` | 34 | 54 | 35 | 61% |
| `intent.py` | 84 | 66 | 6 | 36% |
| `converse.py` | 87 | 233 | 43 | 72% |
| `loop.py` | 202 | 327 | 70 | 61% |
| `llm.py` | 159 | 178 | 82 | 58% |
| **total** | **575** | 886 | 237 | **60%** |

**1,868 lines is not the size of the port. 575 is, and the portable part is 407.** Sixty per cent of
`core/` is the reasoning, and the reasoning is this project's artefact — it survives a port as
comments, but see *What is lost*.

**`llm.py` is an environment that happens to live in `core/`.** It is an HTTP client for one server
on one loopback port, plus `Reply`, `_Balanced` and `_looping`, which are that server's streaming
behaviour. A phone may not run Ollama at all; if it does not, every line of `llm.py` is dead there.
So the portable core is `contract` + `intent` + `converse` + `loop` = **407 code lines**, and the
model is a third seam beside `Eyes` and `Hands`.

**And the reading where nothing changes is the right one today.** By `contract.py`'s own criterion —
a capability belongs in the protocol when *the core calls it* — a `Speaks` protocol would qualify,
since `converse()` calls `model.ask()`. It is absent because `llm.py` sits inside `core/`, so the
concrete class was always in reach and duck typing sufficed; every fake in `tests/` and the
`_Capture` in `measure/choosing.py` already exercise the seam. Declaring it now would be a contract
change that moves no behaviour and forks nothing. **Do not move the file and do not add the protocol
until something needs it.** What would need it is precisely a port: a Rust crate has to name the
trait it calls, so a port *converts an implicit seam into an explicit one*. That is a genuine
architectural gain, and it is worth roughly an afternoon in Python if it is ever wanted on its own.

**One thing verified while counting: `converse()` never reads a rectangle.** `describe()` uses
`w.role` and `w.name`; `_effect()` uses `listed[index].name` and `.role`. The prompt side of the WHERE
rule is already structural, not merely tested — which matters below.

## The robot: answered, not open

Two exclusions, and they are independent. Neither is about language.

**WHERE.** A room has no deterministic source of it. Anything filling `Eyes` for a robot is a
detector guessing names and boxes, so the core would receive probabilistic output through an
interface whose entire contract is that it is exact — the rule inverted at the seam. v1 spent a month
establishing that a room hands you no ground truth.

**WHETHER**, added 2026-09-13 by refusing spoken confirmation. `Session.perform` is reached from
exactly one place, a person pressing 실행, and a button press is an event the OS attributes to a
physical device. A robot has no such device in the loop: its confirmation channel is speech, which is
the exact thing that was refused, on the grounds that a transcript is a recogniser's opinion about a
waveform. And a robot's actions are the ones least able to afford it — a wrong click has no undo, a
knocked-over glass has less.

So: **a native core would be a fast core that still cannot host a robot.** The three ways of making
it fit, assessed:

**(A) Widen the contract — let `Eyes` carry uncertainty.** Rejected. Every consumer of `Eyes`
(`Session.find`, `perform`, the index-to-name mapping in `converse`) would have to handle *maybe a
thing called X, about here, confidence c*, and a confidence invites a threshold nobody can set — the
same argument that keeps `confidence` off `Heard`, where it would be `None` on the only recogniser in
sight. The PC and the phone would report 1.0 for ever, so two environments of three would carry
machinery they never exercise, and the exact path would be degraded to serve a path that does not
exist. This is the rule inverted with extra steps.

**(B) A second core.** Accepted, as a classification rather than as work. A robot is a different
problem, not a second implementation of this one, so the instruments split cleanly — a robot's
measurements are robot measurements, and nothing is shared as code until something measures that it
should be. What transfers is ideas: the four separable pieces, `Trigger` / `Effect` / `Intent`, one
call per thought, an unprompted effect downgraded to speech.

**(C) An explicit `Detector` seam with its uncertainty in the type.** Rejected *now*, and worth
saying why, because it is the tempting one. It would not weaken `Eyes`: it is a different protocol
returning a different type, and no core module would call it. But that is the objection — it would be
a protocol in `core/` that nothing in `core/` calls, which is exactly what `contract.py` refuses for
`attending_title()` and what `voice.py` refuses for `Ears`. And the interesting part of it is not the
type, it is the conversion of a guess into a commitment, which *is* the unsolved robot problem.
**What would earn it**: a working robot environment, outside this repository, whose detector has been
measured — a recall and a precision with their denominators and their conditions — and a
demonstration that some component consumes that uncertainty and does something different at two
different values of it. Absent that, a `Detector` protocol is a guess about a shape, written into the
one file this project does not let people guess in.

**Decision: `contract.py` does not change, and the classification does.** The robot moves from *the
open architectural question* to *a finding: out of scope for this core*. That is not a failure, and
it should stop appearing in lists of what the core targets — while it is on that list it will go on
being offered as a reason for changes that do not help it.

## PyO3, assessed

The candidate is right about the thing that matters most. **The worst outcome available is two cores
with the instruments pointed at the one nobody runs**, and PyO3 is the only mechanism on the table
that avoids it. `measure/` is 7,489 lines and 46 result files, larger than the core, and every number
in `CLAUDE.md` is a number because that code can be re-run. It will stay Python.

The seam itself is affordable (0.005–0.095 ms a turn against 610 ms). What is not free is everything
in `tests/` and `measure/` that assumes the library is Python.

### The architecture rules, audited one at a time

Nine rules enforce the architecture by reading source. Under a Rust core reached through PyO3:

| rule | where | survives? |
|---|---|---|
| the core names no platform | `test_core.py` | **yes, rewritten** — a text scan; the comment detector becomes `//`, `///`, `//!`, which is easier to get right than Python docstrings |
| no core module asks a model for `x` and `y` | `test_core.py` | **no** — walks Python string constants; needs a Rust parser |
| the core never hands a model an image | `test_core.py` | **yes, unchanged** — behavioural, asserted inside `FakeModel`, and a PyO3 `converse()` still calls a Python fake |
| exactly one `perform` call site in `app.py` | `test_app.py` | **yes, unchanged** — walks the caller, which stays Python; `session.perform(intent)` is still an attribute call |
| only `app.py` and `__main__.py` reach an action | `test_voice.py` | **yes, unchanged** — walks non-core Python |
| the CLI and the voice seam never meet | `test_voice.py` | **yes, unchanged** — a text scan of `__main__.py` |
| the voice seam imports nothing from the core | `test_voice.py` | **yes, unchanged** |
| the core grew no ears and no mouth | `test_voice.py` | **no** — walks Python identifiers in `core/`; needs a Rust parser |
| `--prompt-check`'s harness audit | `measure/choosing.py` | **yes, unchanged** — walks `measure/*.py` |

**Seven of nine survive, and the two that do not are the two load-bearing ones.** That is the answer
to the question nobody had answered, and it is not "the AST tests die" — it is narrower and worse. The
rules that walk *callers* are untouched; the rules that walk *the core itself* are the casualties, and
those are the ones that hold the rule the whole design hangs on.

Three ways to re-home them, and they are not equivalent:

* **`syn` in `cargo test`.** Correct, and the natural place. The cost is that `python -m pytest` — the
  command in the git hook, the command every standard in `CLAUDE.md` is quoted against — no longer
  runs the rule. A rule enforced by a second command is a rule enforced when somebody remembers.
* **A `pytest` test that shells out to `cargo test`.** Keeps one command. Costs the suite's best
  property: *148 tests, none needing a display, a model or a network* becomes *…or a Rust toolchain*,
  and the failure mode on a machine without cargo has to be a loud failure and never a skip. A skipped
  architecture rule is the shape of defect `measure/sending.py` exists to prevent, one directory over.
* **Make the rule unnecessary by construction.** For the coordinate rule this is genuinely available,
  and it is the one architectural thing a native core buys: give the renderer a type with no rectangle
  in it — `Listing { name, role }` — and *the model cannot be shown a coordinate* becomes a compile
  error rather than a test. `converse()` already never reads `.rect`, so this narrows a type without
  changing behaviour. **It is not a complete replacement**: a Rust core could still write the words
  *"give me x and y"* into a prompt about a scrollbar, and only a parser walk catches that. So it
  reduces the coordinate rule to a residue; it does not retire it. The ears-and-mouth rule has no
  constructive form at all — it is a rule about what is *absent*.

Recommendation if the port happens: the constructive type **and** `syn` in `cargo test` **and** a
loud-failing pytest shim, in that order of preference for what each is trusted to catch.

### Two rules do not walk source at all, and they are the quieter risk

`tests/test_core.py:391` and `measure/repair.py:354` both read the channel list off
`inspect.signature(Session.remember_doing)`, so that a fifth channel arrives *failing* the past-tense
rule rather than quietly exempt from it. That is runtime reflection on a Python function. PyO3
generates `__text_signature__` for many shapes, and its coverage of keyword-only parameters with
defaults is not something to assume. **If `Session` becomes a `#[pyclass]`, check this first**: it is
the one breakage that would present as a test still passing while covering nothing.

### What `measure/` assumes about the library being text

Four places, all specific:

* `measure/sending.py:295` `loop_now()` hashes `_loop.__file__`, and `deferral()` refuses to count a
  recorded conversation run as coverage unless it matches. Under a compiled core that path is a
  `.pyd`. **Measured today: three builds of a byte-identical two-line Rust source, same rustc 1.98.1,
  same flags, same directory, produced three different SHA-256s.** So an artefact hash forks on every
  rebuild — the `p`-scheme's over-triggering failure returning in a form that cannot be fixed by
  writing better prose. Fixable by hashing the `.rs` sources instead, which requires them present at
  measurement time: true in this repository, false in an installed wheel.
* `measure/reoffer.py:229` runs an A/B over **two `argmind/` trees** via `--argmind PATH` and
  `sys.path`, checking that `argmind/core/loop.py` `is_file()`. Under a compiled core an arm is a
  build, and two builds of the same extension module name cannot both be live in one process. Each
  arm needs its own crate name or its own process.
* `measure/reoffer.py:1059` `tree_prints()` rglobs `*.py` under the imported package, and `--compare`
  prints *"argmind/ files differing: NONE — same tree both arms"* out of it. With the core compiled,
  that line would print NONE **while the arms differ** — an instrument reporting a clean result it has
  not earned, which by this directory's own standard is the worst failure available.
* `measure/choosing.py:456`, `:1021` and `_mutant_module()` at `:845` read `converse.py`'s **source
  text** off disk, mutate it with regexes, and `exec` the result as an in-memory module.
  `prompt_record(module=...)` takes the module as an argument specifically so this can be done
  *"without writing a byte into `argmind/`"*.

That last one is the expensive one and it deserves precision. `--prompt-check` ran clean today:
**12 rows, all live, PASS, shipped tag `r322a4c`** — three rows that must not fork (a 203-line
docstring replaced by one line, a comment, a blank line) and nine that must (the reply-shape phrase,
the field ordering, a word of `SYSTEM`, a word of `UNPROMPTED`, the widget-line format, the
history-line format, `MOST_AT_ONCE`, `MAX_LISTED`, the offer-conditioned line), plus the
demonstration that the superseded `q` was blind to the last of them. **Every one of those twelve rows
is a textual mutation of a Python source string.** Under Rust each becomes a mutated crate, a build,
and a distinctly named extension module in a fresh interpreter, and each row's regex has to be
rewritten against Rust syntax. Build time is not the objection — the owner has said it does not
matter, and it does not. The objection is that a check whose mechanism becomes twelve times more
awkward is a check that erodes, and this directory's written history is that an eroded check produces
a wrong conclusion within days.

## Staging, and what verifies each step

Byte-exact equivalence exists for exactly one of the four files. Say so plainly rather than let the
one that has it stand in for the rest.

**Stage 0 — build the two missing equivalence tests, in Python, before any port.** Both are worth
doing on their own merits; a port merely makes them mandatory.

* *For `loop.py`: a golden transcript.* Feed a scripted sequence of `(trigger, screen, canned reply)`
  and assert the resulting history lines and `Exchange` fields byte for byte. **This does not exist.**
  What stands in for it today is `sending.loop_now()`, a hash of the file — which by construction
  cannot survive a rewrite in any language, since the file is the thing being replaced. A golden
  transcript is the only form of the test a port can pass, and writing it in Python first means the
  Python core proves it before the Rust core is asked to.
* *For `llm.py`: a corpus of recorded replies.* `tests/test_llm_stream.py` (350 lines) already fakes a
  server, so the harness exists; what does not exist is real output to replay through it. Every
  harness stores aggregates and **no per-reply record survives** — `converse.py`'s docstring withdrew
  two tallies for exactly this reason. So truncation, `_Balanced` and `_looping` would be ported
  against synthetic streams only. That is a gap, it is nameable, and the fix is one field in the
  harnesses' JSON.

**Stage 1 — the prompt renderer.** `describe()` and `converse()`'s assembly. Verified by
`prompt_record()`: a renderer that produces `r322a4c` over the six probes has provably not changed
the prompt, and every recorded baseline stays comparable. Strengthen it for a port with a differential
run over all four captured screens × asked/unprompted × three history shapes, byte-compared — six
probes pin the branches, not the inputs. **This stage needs no FFI at all if `converse()` stays Python
and calls a Rust `render()`**, which is why it is the right spike: no seam, no marshalling of
`Widget`s, one string out.

**Stage 2 — reply parsing and `_effect`.** Blocked on Stage 0's corpus. Verified by differential
against the Python implementation over that corpus plus the malformed cases `tests/test_core.py`
already enumerates.

**Stage 3 — `loop.py`.** Blocked on Stage 0's golden transcript. This is the stage that introduces the
real seam: three `Eyes` reads and one model call per turn cross the boundary. It is also the stage
that breaks `loop_now()`, `tree_prints()` and possibly `inspect.signature`. Budget those as work, not
as surprises.

**Stage 4 — `llm.py`: do not.** It is one platform's HTTP client. Porting it moves the file that
*must* differ per platform into the crate that is supposed to be shared.

**Before all of them — Stage −1, and it is the one that decides whether any of this happens.** The
phone's shape is undecided: either this machine drives it over ADB, or a much smaller model runs on
it. **If the ADB fork wins, the core never runs on the phone, and the portability argument — the only
surviving argument — evaporates entirely.** The measurement that settles it is cheap: time an
`adb exec-out uiautomator dump` round trip from this machine against the 150 ms `Eyes` budget the PC
already meets, and establish whether any model the phone can hold answers a choosing fixture at all.
An afternoon, and it either kills the port or gives it its first real reason.

## What is lost

Concretely, so that none of it arrives as a surprise:

1. **Two architecture rules lose their home in `python -m pytest`** — the coordinate rule and the
   ears-and-mouth rule. Re-homing them costs either a second command or a toolchain dependency inside
   the suite. These are how this project's rules are enforced rather than remembered.
2. **`--prompt-check`'s twelve mutation rows** become twelve builds with bespoke Rust-syntax
   mutations, and the mechanism that renders a mutant *without writing a byte into `argmind/`* has no
   cheap equivalent.
3. **Four hashing call sites and one whole-tree A/B mechanism** stop describing the library, one of
   them silently — `tree_prints()` printing NONE while the arms differ.
4. **The cost of changing a prompt sentence.** Not the seconds: the mechanism. Today an edit is a
   string, and `prompt_record()` renders six probes against a mutated copy of the module in
   milliseconds without touching the tree. This project edits prompts constantly and measures each
   one; five sentences were added to `SYSTEM` over its life and every one of them slid the model,
   which is only known because each was cheap to try and cheap to un-try.
5. **The zero-dependency property.** `argmind/core/` imports `json`, `time`, `urllib`, `base64`, `io`
   and `typing` and nothing else — verified. `comtypes` at 5 MiB is the whole deterministic half. A
   crate has a `Cargo.lock`, and pyo3 is its first entry: *no third party at all* becomes *no third
   party except the one that makes it callable*.
6. **One language becomes two**, permanently. `measure/` at 7,489 lines and `tests/` at 2,410 stay
   Python whatever happens to the 575.
7. **The prose stops being coupled to the code by anything checkable.** Sixty per cent of `core/` is
   reasoning, and several tests exist precisely to keep prose honest — the four offer-label spellings
   asserted against what `remember_doing()` actually writes, the past-tense rule read off a live
   signature. Those couplings are Python-shaped. The comments would survive a port; the machinery that
   catches them going stale would have to be rebuilt one at a time.

## What would refute this record

* **The ADB measurement comes back bad** — the phone cannot be driven remotely inside the `Eyes`
  budget, and some model it can hold does answer a choosing fixture. Then the core runs on the device,
  Python's presence there has to be costed for real, and portability becomes the strongest argument
  on the table.
* **A turn stops being dominated by the model.** Every number here is a ratio against 610 ms. A 50 ms
  model, or a batching design that amortises the call, would move 0.071 ms from invisible to merely
  small — still not decisive, and the threshold at which it becomes decisive is a measurement somebody
  would have to make rather than assert.
* **A screen an order of magnitude denser than 152 widgets.** `describe()` is linear, so 1,520 widgets
  is ~0.19 ms — computed, not measured, and still invisible. `MAX_LISTED = 200` would bind first, and
  that is a different conversation.
* **`inspect.signature` works cleanly through PyO3 for keyword-only parameters with defaults.** That
  removes the quietest risk above and is ten minutes to check.
* **A measured PyO3 crossing that beats the tuple proxy.** 0.005 ms is a Python-side lower bound and a
  real boundary could be cheaper in ways this estimate cannot see. It cannot be cheaper than free, and
  free still only buys 0.07 ms.

## Stage 1 was run, and it passed — 2026-09-13

The record above named the prompt renderer as the right first spike and named `r322a4c` as what
would verify it. That was executed the same day. `native/prompt-spike/` is a Rust port of
`describe()` and `converse()`'s assembly — no HTTP, no session, no effects — rendering the same six
probes and hashing the same record.

    python record  5,676 bytes  ->  r322a4c
    rust   record  5,676 bytes  ->  r322a4c        identical, and identical in length

**The test can fail.** Changing one character of the Rust prompt — `On screen now:` to
`On screen now :` — moves it to `r801e06` and lengthens the record to 5,682 bytes. So the match is
a result and not a tautology, which matters because a fingerprint that cannot fork is the defect
this scheme replaced.

It carries `prompt_record()`'s own refusal too: if any two of the six probes render identically the
spike asserts rather than hashing, because two identical renders mean a branch the print claims to
cover has stopped being reached and the tag would go on looking healthy.

**What this settles and what it does not.** It settles the risk that mattered most — this project
has measured prompt edits as *never free*, 0 for 5, so a port that changed a space would be an
unmeasured prompt change wearing a port's clothes, and that risk is now detectable in under a
second with no model and no fixtures. It settles nothing about `loop.py` or `llm.py`, which is
exactly the asymmetry the staging section names: one of four files has byte-exact equivalence and
the other three do not.

### Two of the record's own numbers, re-derived independently

* **Build reproducibility, confirmed and it is worse than a caveat.** Three builds of a
  byte-identical Rust source, same rustc 1.98.1, same flags, gave three different SHA-256s of the
  artifact: `11c76bd3…`, `fe79b636…`, `146e8fdf…`. So any fingerprint over a compiled core forks on
  every rebuild — the `p`-scheme over-trigger returning in a form no amount of prose discipline can
  fix, since there is no comment to avoid writing.
* **The seam proxy, re-measured, is worse than reported.** 152 widgets across the boundary:
  0.005 ms as tuples, 0.062 ms rebuilding objects, **0.144 ms via JSON** — against 0.032 ms for all
  the Python `converse()` does. The record said 0.095 ms for the JSON path; on this run it is
  higher. Either way the direction holds and hardens: **the crossing costs more than the work it
  replaces**, and `Session` reads the screen up to three times a turn.

`adb` is not installed on this machine, so the one measurement the record calls decisive — the
Android path, and whether the phone runs the core at all or is driven from the PC — could not be
attempted today.

## The port is finished, and the Python is gone — 2026-09-15

`python-before-removal-2026-09-15` is the last commit with it. Every module has a successor, every
harness has a successor, and **what the Python produced is committed as data** — `native/
expectations/*.json` and `native/golden/` — so the equivalence checks outlived the implementation
they were taken from. The recorded results and captured screens stay in `measure/`, which is now
data and no code.

### The stated payoff did not arrive

This record argued that a native core would turn rules held by AST tests into rules held by the
compiler, and named the example: a `Listing { name, role }` type making *the model is never shown a
coordinate* structural. **`Listing` was never built.** The core still passes `Widget`, which carries
a `rect`, and the rule is held by `native/argmind-rules/` reading `converse.rs` — text, not syntax,
which is *weaker* than the `ast` walk it replaced.

What the port did buy is real and is somewhere else:

* **`Reach`** makes *never touch anything this harness did not start* a type rather than care, and
  widening it for `obstruction` meant adding a variant that carries both process ids rather than
  editing a condition.
* **`Proposals`** makes a stale button structurally unable to confirm a later action — numbers are
  never reused, so the defect cannot be reintroduced by an off-by-one.
* **A job object** makes *the recogniser dies with argmind* the operating system's promise. `Drop`
  could not do it: `std::process::exit` skips destructors and Task Manager asks nobody.
* **`Did` and `Answered`** make "refused", "landed", "gone" and "declined" different values rather
  than different comments, which is what let `obstruction` predict four outcomes and print them
  beside what happened.

None of those is the benefit that was argued. Recorded plainly because **a rewrite justified by a
payoff that did not materialise should say so in the document that justified it.**

### Deleting it found three things the port had missed

All three were found by asking what would be lost, rather than by assuming nothing would — and none
would have shown up as a failing test, because the tests that covered them were the ones being
deleted.

* **The licence rule and the core's blindness were held only by Python.** `tests/test_licences.py`
  walked `native/**/Cargo.toml` as well as `pyproject.toml`; deleting it would have taken the check
  on the Rust dependencies with it. On its first run the ported version failed, correctly: the
  record still said whisper.cpp was *"not yet a dependency"* while argmind was compiling and
  running it.
* **The transcript guard had no successor at all.** Not a test — a capability. `argmind/env/
  whisper.py` catches Whisper's degenerate loops, 7 of 7 reproduced, and rejects none of the
  strings that must survive; the Rust `hear()` was returning whatever came off the socket. Given
  that silence is transcribed as words **120 times out of 120**, that is the difference between a
  guard and none.
* **"The room stays quiet unless somebody asked" was a comment in two places.** `think` speaks and
  `notice` does not, and nothing checked it.

### What is worth taking from that

**A port is finished when somebody tries to delete the original and writes down what breaks**, not
when the new tests pass. Every one of the three had a green suite on both sides.

Three of this project's rules are now held by reading the source rather than by the compiler, in
`native/argmind-rules/`, and each was checked by breaking it: eleven mutants, eleven caught. That
is the honest state — not what this record predicted, and better than nothing enforcing them.
