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

## What is not written: hands

**On Linux that is a permission question before it is a coding one.**

- **X11** lets any client inject input with XTEST. `libxdo` is installed and the work is small.
- **Wayland** deliberately does not. The sanctioned route is `xdg-desktop-portal`'s RemoteDesktop
  interface, where the person grants a session — so what changes is the permission model, not the
  implementation, and argmind's four-way permission has somewhere to go.

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
