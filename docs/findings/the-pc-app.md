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

### `Widget.centre` clicked the inert part of the right control

The failure is not the agent's. A combo box's bounding rectangle is centred on its **text field**;
the part that opens the list is the arrow at the right edge. The click landed exactly where it was
aimed, on exactly the widget the model chose, and did nothing.

**This is sharper than the 73% figure from `pointing.py`.** That measured how often a centre resolves
to a *different element*. Here the centre was on the correct element and the click was still
inert — so "the centre is on the widget" is not sufficient, and `contract.py` calls `centre` "the one
place a name becomes a coordinate".

The repair that follows from the project's own rule: **UI Automation knows how to operate its own
controls.** `InvokePattern`, `ExpandCollapsePattern` and `TogglePattern` activate a control without
anybody guessing a pixel — which is the deterministic side doing the acting, the same argument that
put WHERE on the environment in the first place. Not all controls expose a pattern, so a synthetic
click has to stay as the fallback, and **which one was used has to be counted** rather than assumed.
Unmeasured as of this writing: whether invoking differs from clicking in ways that matter (focus, for
one, which a real click sets and an invoke may not).

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
