# Eyes on Linux — AT-SPI2, 2026-09-17

The rule everything hangs on is that **the deterministic side says WHERE**. On Windows that is UI
Automation. The question for a port was whether Linux has an equivalent that answers in the same
shape — a name, a role, and a rectangle in screen pixels — or only something that approximates it.

It has one, and it answers in 116 ms.

## Measured before a line was written

`gtk3-widget-factory`, walked through `busctl` by hand:

```text
toggle button   Menu      (1352, 23, 36, 46)
radio button    Page 1    (551, 23, 121, 46)
combo box       Left      (36, 300, 121, 34)

nodes: 121   named: 63   with a rectangle: 120   buttons: 24
```

That is `Widget { name, role, rect, window }` with nothing missing and nothing invented. **The
contract survives the port**, which was the thing worth knowing before committing to anything.

## Two things the probing found

**Toolkits publish nothing until an assistive technology says it is listening.** With
`org.a11y.Status.IsEnabled` false, `gtk3-widget-factory` offers one node and no tree; with it true,
121. So argmind sets it — and that is **a real side effect on every other application on the
desktop**, not just the one being looked at. It is the same bargain a screen reader makes and
argmind is the same kind of program, but it is switched on when eyes are opened rather than at
startup, so a build that never looks at anything never touches it.

**Off-screen widgets report `(-2147483648, -2147483648, 1, 1)`** rather than failing. `i32::MIN`.
Menu items inside a closed combo box do it. Handing those to the core would put widgets at
coordinates no display has, so `sane()` drops them. Found by looking at output, not by defending
against an imagined case.

**And GTK4 published nothing here** where GTK3 published everything — `zenity` 4.2 offered its
application node and no children, with `IsEnabled` on and the app started afterwards. Unexplained,
and recorded rather than guessed at: it may be Wayland, it may be `zenity`, it may be how GTK4
gates its bridge. It matters because it bounds how much of a modern desktop argmind could read, and
it is the first thing to re-check on a real Linux machine.

## What it does

146 widgets of `gtk3-widget-factory` in **116 ms**, against UI Automation's 41–266 ms for the same
job on Windows. Same order, and the comparison is fair: both are the whole active window.

```text
보고 있는 창: "gtk3-widget-factory"
146개, 116 ms
toggle button   Menu                       (1352, 23, 36, 46)
button          Close                      (1401, 31, 34, 30)
radio button    Page 1                     (551, 23, 121, 46)
combo box       emblem-default-symbolic    (419, 256, 157, 34)
이름 있는 것 68개, 이름 없는 것 78개
```

The walk is breadth-first and bounded at twelve deep and four hundred widgets. Breadth-first
matters: a depth-first walk of a window whose first child is a document spends the whole budget
inside it, and the controls a person means are near the top.

## The hit test was wrong, and running it is what said so

`at_point` is the one question the `Eyes` trait does not carry — the page uses it to say what a
proposed click would land on. The first version called `GetAccessibleAtPoint` once and believed the
answer. Asked for the middle of a button called `Menu`, it said `("", "panel")`.

**`GetAccessibleAtPoint` returns a direct child of the node it is called on, not the deepest thing
at that point.** Followed down now, bounded, with one look inside a nameless node because a
button's label is often a named child of it. It answers `("Menu", "toggle button")`.

## Hands, through the same layer

**Measured first, and it decided the design.** A GTK button offers exactly one action, named
`Click` and described as *Clicks the button*, and `DoAction` answers `true`. So the path Windows
prefers has an exact counterpart here — `patterns.rs` there records
`Did::Operated { how, live }`, *no pixel was aimed at and the cursor did not move*, which the
pointer measurement found is worth 3 of 3 — and the D-Bus connection the eyes already open is all
it costs. **No new dependency for hands at all.**

Verified against the widget's own state rather than against hope, because a push button that worked
and one that did nothing look identical from outside:

```text
dry     Operated { how: "atspi-action", live: false }    0x43000900 -> 0x43000900
armed   Operated { how: "atspi-action", live: true }     0x43000900 -> 0x43000910
```

Bit 4 — `CHECKED` — flipped. The toggle toggled.

`find()` re-resolves by name, role and rectangle immediately before acting, which is the invariant
everything rests on: a listed widget is gone 13 times in 14 over the second a call takes. Name and
role alone are still not a key, and the tie-break among copies settles nothing — **wrong here in
exactly the way it is wrong on Windows**, and recorded rather than papered over.

## Two bugs that cost two runs each, and what they were really about

**Announcing twice empties the tree you were announcing to.** `Bus::open()` wrote
`org.a11y.Status.IsEnabled` on every connection, and one run opens three — eyes, hands, and the
state read. Every write makes the toolkit bridges re-register, and a re-registering bridge has
nothing to say. It presented as *the hands cannot find a widget the eyes just listed*, which is a
long way from the cause. Now: read first, write only if false, once per process, and wait 400 ms
for the bridges to publish.

**`ACTIVE` needs a window manager and WSLg has not got one.** The frame reported `0x43200100` —
visible, not active — while plainly on screen. GTK sets `ACTIVE` from the window manager's focus,
and inside WSLg there is no window manager: Windows is one level up. So `attended()` prefers a
frame that claims to be active and otherwise takes the largest on screen, **and says which it
did**. A guess labelled as a guess is usable; a guess presented as a reading is not.

That second one is also a limit on this proving ground, and worth saying plainly: **WSLg cannot
answer "which window is in front"**, so the part of the Windows design that exists precisely to
answer that — `Attending`, the watcher, the newest-not-in-front rule — has no counterpart here that
can be tested. It needs a real desktop.

Reading the enum from memory caused the smaller of the two: the first `state_of` called bit 8
`CHECKED`, which the frame's own `0x43200100` disproves. It returns the whole state word now, and
comparing before with after needs no opinion about the enum at all.

## What is still not written: typing

**AT-SPI no longer synthesises input.** `DeviceEventController` used to generate key events and this
at-spi2-core does not expose it — checked by introspection, not assumed. So `type_text` and `press`
refuse, and name what it would take.

**On Linux that is a permission question before it is a coding one.**

- **X11** lets any client inject input with XTEST. `libxdo` is installed and the work is small.
- **Wayland** deliberately does not. The sanctioned route is `xdg-desktop-portal`'s RemoteDesktop
  interface, where the person grants a session in a dialog belonging to the *system*.

**And that last point changes something argmind promised.** Its four-way permission remembers a
decision in `decisions.json` and does not ask again; on Wayland the portal decides whether to
re-ask, and revocation happens in the system rather than in argmind's settings panel. So *항상
허용* cannot be honoured from inside the application there, and *항상거부하면 세팅창에서 바꿀수
있게 할것* is only half true. The honest model is **two gates** — argmind's, and the system's where
there is one — and the settings panel manages only the first.

Three platforms hold the gate in three different places, which is worth having written down
somewhere: on **Windows** `SendInput` needs no permission at all, so argmind's dialog is the only
gate and is a discipline it imposes on itself; on **X11** XTEST is open to any client and there is
no gate; on **macOS** the OS grants Accessibility once, in System Settings; on **Wayland** the
portal grants per session. Only the first two of those are measured here — the macOS and Wayland
rows are knowledge, not measurement, and this machine has no portal installed to check against.

Which of the two to support, and what the portal actually gives, is not something to decide from a
WSLg session that has no window manager of its own. Nothing is written until it is decided against
a real desktop.

## The dependency, and why the house style lost again

`dbus` 0.9, Apache-2.0 OR MIT. **6 crates**, against `zbus`'s 85 and `atspi`'s 93 — resolved and
counted, not guessed. It binds `libdbus-1`, which this tree already requires on Linux because the
tray speaks StatusNotifierItem over it, so it adds no system dependency either.

Hand-writing the wire protocol was considered and refused, which is the opposite of the usual
answer here — this tree hand-writes WASAPI, Media Foundation, SAPI and raw HTTP/1.1. D-Bus is SASL
and then a binary marshaller with alignment and endianness rules, and **the failure mode of getting
it subtly wrong is a widget tree that is quietly wrong.** This project treats a silently wrong
answer as worse than a crash, and the GPU K-quant that reported RTF 0.123 while generating nonsense
is why.

## A hole it found in the licence rule

`outside_dependencies` read only `[dependencies]`. `dbus` went in under
`[target.'cfg(target_os = "linux")'.dependencies]` and **the rule did not see it** — a crate could
have been added that way and cost nothing, which is the one thing that rule exists to prevent.

It was noticed because the *other* half of the check complained that the record named a crate
nothing declared. The record was right and the reader was wrong, which is the pleasanter of the two
ways round. Fixed, with a test of its own — whose own first version was wrong, because it indented
its sample manifest and indented lines are skipped on purpose.
