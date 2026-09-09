# The PC app — what building a window measured, 2026-09-09

`python -m argmind app`. A small always-on-top window: a transcript, an input box, and a row with a
button for every action it wants to take. `argmind/app.py`, `tests/test_app.py`.

Three measurements shaped it and none of them was a design preference.

## The agent's window is on the screen the agent reads

The moment you click into the input box, `GetForegroundWindow` returns **the agent**. An app that
reads the foreground window answers questions about its own text box, and this is not a corner case
— it is what happens every single time somebody types.

So `WindowsEyes(ignore_pid=...)` answers about the last window that was in front and was somebody
else's, and the header always names it.

**The first version of that fix was wrong and running the app is what showed it.** It seeded the
remembered window during setup, before the app's own window existed. That is a race, and it lost:
the app opened with `0` remembered and answered *"I can't see any windows or widgets open here"*
twice in a row while a browser with 141 widgets sat directly behind it — **with no way to recover,
because the app kept focus, so `attending()` never saw anybody else again.**

`attending()` now falls back to walking top-level windows in z-order and taking the first visible,
titled one that is not ours. `looked_at_itself` counts how often the substitution was used at all,
because the substitution is a **guess about what "this window" means** and a guess nobody counts is
a guess nobody can check.

## Does the cursor go where it is sent, and is the centre on the widget?

`measure/pointing.py`. Two separate questions, and the recorded gap was that **the click half of
end-to-end accuracy had never been measured at all**.

**(b) The write path is exact here — 40 of 40 points came back from `GetCursorPos` unchanged.** This
process is DPI-unaware (`GetProcessDpiAwareness` returns 0) and UI Automation reports physical
pixels, so on a scaled display the two spaces disagree and every click is offset by the scaling
factor. This desk is at 96 DPI, which is a property of one machine and not a guarantee.
`WindowsHands.pointing_is_exact()` therefore runs once at startup and the app **switches clicking
off** if it fails, rather than finding out by trying.

**(a) A widget's own rectangle centre resolves back to that widget 73% of the time**, to a name that
contains or is contained by it 14%, and to something else 12.8% (n = 304, one real window).

**That last figure is an upper bound on how often a click misses, not the miss rate.** A click lands
on a pixel; a parent element answering UIA's hit test does not mean the pixel belongs to something
else. The clearest case is window caption buttons: `Close`, `Minimize` and `Maximize` all resolve to
`application`, and the pixel is nonetheless exactly the close button. So `WindowsEyes.at()` is shown
to the person as a note beside the confirm button and **never used to refuse** — a gate here would
block closing a window, which is the most common thing anybody asks for.

## The first two questions cost 4.7 s and 5.9 s; the rest cost 0.3

Measured through the app's own path on a 12-widget window, five questions in a row:

    1   4,715 ms    ttft 4,381 ms
    2   5,940 ms    ttft 5,178 ms
    3     331 ms    ttft   119 ms
    4     329 ms    ttft    94 ms
    5     608 ms    ttft   126 ms

**Two calls to warm, not one** — which is why the app's warm-up makes two and a test asserts the
count rather than trusting a comment. The steady state matches the 0.44 s a call measured in the
choosing harness, so the app is not adding cost; it was paying a cold load in front of the person.

The app now warms at startup and shows a running counter while it thinks, because a four-second wait
with nothing moving reads as a hang. The cost is 5–6 GB held from launch — a real cost on a machine
that ran out of memory a week ago — so `--cold` turns it off and `Local.release()` still runs on the
way out.

## The shape the accuracy number forces

Choosing is right 18 times in 28. Eight wrong answers in thirty-seven goals is a bad click most
working hours, and a click has no undo. So **every action is proposed and a person presses the
button**: `Session.perform` is called from exactly one place in `argmind/app.py`, and
`test_the_session_permits_unprompted_action_ONLY_because_a_person_gates_it` walks the AST to keep it
that way.

That also settles a flag that would otherwise look alarming. The app builds its `Session` with
`may_act_unprompted=True` — the opposite of the library default — and it is **stronger, not weaker**:
a person confirming each action per-action is a better brake than a session-wide flag. Leaving it
off would have turned everything the agent noticed into speech and made the button a lie.

Between proposing and confirming, the screen moves: 13 times in 14 over the second a call takes. So
the row carries a name, the name is resolved when the button is pressed, and if the thing has gone
by then the row says so instead of clicking where it used to be.

## The click was measured at last, and it found a defect in the core (2026-09-09)

`measure/clicking.py`. Character Map, launched and closed by the script, three goals with outcomes
written before running, live clicks. The foreground window's process id is checked immediately
before every click, because a click goes to a pixel and anything that came to the front would
receive it.

| goal | chose | result |
|---|---|---|
| 고급 옵션 켜줘 | `Advanced view` | **worked** — `+ Character set :, Go to Unicode :, Group by :, Search` |
| 글꼴 목록 열어줘 | `Font :` | **nothing** — not one name appeared or disappeared |
| 이 창 최소화해 | `Minimize` | **worked** — `IsIconic` true, `- Close, Maximize, Minimize, System` |

**Two of three did the expected thing, and the agent chose the right widget three times out of
three.** So the chain completes: a person confirms, the click lands, the thing they wanted happens.
That had never once been observed before this.

### WITHDRAWN: "`Widget.centre` clicked the inert part of the right control"

That was written here, put in CLAUDE.md and committed in `00979fd`, and it is **wrong**. The
reasoning was that a combo box's rectangle is centred on its text field while the arrow that opens
the list is at the edge, so an exact click on the right control could still be inert. Plausible,
consistent with every number in hand, and not what happened.

### What actually happened: the name lost its role on the way back

Character Map lays out a **`text` label reading `Font :` immediately before a `combo box` also called
`Font :`**. The model chose the combo box. The effect stored the name. `Session.find()` resolved the
name again at the moment of acting — and returned **the label**, because the label comes first.

The click landed on a piece of text, which does nothing when clicked. That is indistinguishable from
a click that missed, and it is why the first explanation survived a whole round of work.

    index 8   text        'Font :'   rect (34, 107, 53, 20)     <- what find() returned
    index 9   combo box   'Font :'   rect (86, 106, 321, 30)    <- what the model chose

**The decision to store a name rather than a rectangle is still right** — a rectangle goes stale
thirteen times in fourteen. The key was simply incomplete. `Effect` now carries the role, `find()`
matches on both, and two tests pin it: one for the two-roles case, and one for the case a mutation
exposed, where the chosen thing **moves** and a same-named label of another role ends up nearer to
where it used to be.

### The app's own window was blocking the clicks, and it now steps aside

`measure/obstruction.py`. A borderless always-on-top window is placed exactly over a check box, and
the check box is then acted on:

    patterns off, covered       via click    did nothing
    patterns on,  covered       via click    did nothing
    patterns off, stood aside   via click    WORKED

**A synthetic click goes to whatever window is topmost at that pixel.** The app is 430x560,
always-on-top, in the bottom-right corner — so any widget underneath it was unreachable and the
click landed on argmind itself. That is not an exotic case; it is a corner of every screen the app
runs on.

The fix is fifteen lines and no COM: before performing, the agent asks the window to drop `-topmost`
and lower itself, and to come back afterwards. It is a handshake on a `threading.Event` rather than
a sleep, because the window lives on the other thread and guessing how long a repaint takes is how
this kind of fix works everywhere except where it matters. The wait is capped at a second, and
timing out means the click happens with the window still in place — the behaviour from before this
existed, not a new failure.

### The control patterns are unearned, measured twice — removal recommended, not yet done

Two independent measurements and no benefit in either:

| | with patterns | without |
|---|---|---|
| the three-goal set | 3 of 3 | 3 of 3 |
| a covered widget | did nothing (fell back to a click) | did nothing |

The obstruction row is the one that settles it. The patterns were kept on the argument that they do
not aim at a pixel — but **`_element_for` resolves the element *by point*,** so an obstruction hands
it the obstruction's own element and it falls back to a click anyway. The mechanism inherits the
exact problem it was supposed to be immune to.

Against zero measured benefit: about ninety lines, a second COM client, an ancestor walk on every
click, four distinct ways to fall back, and one real crash it introduced — `ElementFromPoint` under a
cover produced a **NULL COM pointer, which comtypes does not return as `None`**, so `element is None`
missed it and reading a property raised `ValueError`.

**Recommended for removal and still present**, because it was built to an explicit instruction and
undoing that is the owner's call rather than mine. If a case ever appears that needs patterns, a
tree search from the target window rather than a point lookup is where it would start — and `Widget`
carries a window *title*, not a handle, so that is not a small change.

**What is not established: that patterns are useless.** Three goals on one application is a smoke
test. A control whose clickable point is not its centre — a very wide list item, a menu that needs a
hover — was never in the set.

`WindowsHands` operated a control through `Invoke`, `ExpandCollapse`, `Toggle` or `SelectionItem`
when it could resolve the element, falling back to a synthetic click otherwise, and counted which.
That is the deterministic side doing the acting — the same argument that put WHERE on the
environment, and it did not survive being measured. The verdict is in the section above; the code
is still in place pending a decision.

One thing the work did settle: `LegacyIAccessible.DoDefaultAction()` is **not** a universal
fallback. Every control in that window advertises `LegacyIAccessible`, and calling it on the combo
box raised `COMError` while `Expand` succeeded.

### Four ways to fall back and no record of which

`_pattern_for` returned `None` for four different reasons — no element at the point, no ancestor
matching within five hops, the element died, or no pattern offered — and recorded none of them. A
run that fell back to a click on a control that provably supports `ExpandCollapse` could not be
diagnosed at all. **The same mistake as keeping a boolean where the delta was needed, made twice in
one day.** The reason is now carried in the action record, and it is what identified the real defect:
*"the element offers none of invoke, expand, toggle or select"* on a combo box that offers
`ExpandCollapse` could only mean the element was not the combo box.

### Two faults in the measurement, both mine, both found by running it twice

**The minimise expectation was wrong about Windows, not about the agent.** It read the window
afterwards and expected no widgets; a minimised window still answers UI Automation with part of its
tree. The click had worked the whole time and the first run scored it a failure. Now it asks
`IsIconic`.

**It read one window before and every window after.** A combo box's drop-down is its own top-level
window, so reading only the launched one reports "nothing happened" for a list that did open — and
the first run could not tell that apart from a click that landed on an inert area, which is the real
defect. The comparison is now symmetric, and **the delta is printed every time**: a boolean cannot be
diagnosed after the fact, and that is what made the first run's second row unreadable.

## What is not here

The window itself has no tests, on purpose rather than by neglect: it draws what arrives on a queue
and sends what is clicked, and a test of it would be a test of tkinter. Everything above it —
proposing, refusing, warming, the vanished target, the attending title — is a message on that queue,
and those are tested against fakes with no display, no model, no COM.

**The agent still sometimes answers a Korean question in English.** Two attempts to fix that in the
system prompt both made choosing worse and were withdrawn; `choosing-accuracy.md` has the numbers.
It is a real defect, left in place rather than traded for a worse click.

**One unintended side effect, recorded because it happened on the owner's machine.** An early attempt
to test the window drove it with synthetic keystrokes from another process; Windows refused the
foreground change, and the text went to whatever window actually had focus — a line of Korean typed
into a Chrome Remote Desktop session. Harmless, and the reason the app is now driven through its own
queue in `scratchpad/drive_app.py` instead. **Do not test a GUI with SendInput.**
