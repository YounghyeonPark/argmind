# A desktop of its own — 2026-11-08

The owner, 2026-11-07: *사용자의 컴퓨터 사용과 충돌하지 않게 별도의 샌드박스 창에서 작업을 수행하는것
고려.* A task runs for minutes, and the person is at the same machine. The pane answers that for the
web; this is about everything else.

## What this machine has, and the two routes that fell

**Windows 11 Home**: no Windows Sandbox, no Hyper-V machines.

**A Linux screen works and has nothing to run.** `Xvfb` in WSL, its own D-Bus and accessibility bus,
and `argmind-env-linux` read 148 widgets and flipped a toggle there (`eyes-on-linux.md`, *A screen
nobody is using*). But the program a person would want on it is an office suite, and **LibreOffice
from Ubuntu brings `libpoppler`, GPL-2/GPL-3** — the same rule that purged `libimagequant` from the
sandbox. Its 79 packages were installed into `argmind-sandbox`, their copyright files read, and all
of it purged again, the X server too: the sandbox's package list is byte for byte what it was, and
`import docx, openpyxl` still works. The editors that would be left are GPL as well.

## The route that stands: a second Windows desktop

`CreateDesktopW` makes a desktop in the same window station; a program started with
`STARTUPINFO.lpDesktop` set draws there; a thread that calls `SetThreadDesktop` can enumerate its
windows and read them with UI Automation. **Nothing is shown and no input can reach it** —
`SendInput` goes to the desktop the person is on — so everything must be done through control
patterns, which is the path `patterns.rs` already measured as leaving the pointer alone.

`cargo run -p argmind-env-windows --example hidden-desktop`, Character Map, **three runs, identical**:

| | |
|---|---|
| read | 26 elements, 39–44 ms |
| `Toggle` on *Advanced view* | 0 → 1, read back |
| `SetValue` on the copy box | `숨은화면` |
| `Invoke` on *Select* | the box became `!숨은화면` — the button did its job |
| `PrintWindow` | 490×571, 96% not black — it can be watched |
| foreground window | unchanged |
| pointer, physical | (1186, 1066) before and after |

`PROBE_WORD=1`, once, with Word not running: Word started **on that desktop**, 137 elements in 415 ms
(the ribbon, *Save*, the tabs), closed through `WindowPattern.Close` and left no process. **Its title
read *Word (Unlicensed Product)*** — on this machine it will open documents and may not edit them.

**A pointer that seemed to move, and did not.** The first probe was PowerShell over .NET's managed
UI Automation, and it read the pointer at (1186, 1066) before and (1779, 1599) after — even with
every press switched off. 1779/1186 and 1599/1066 are both 1.5, this screen's scale: loading UI
Automation made the process DPI-aware between the two reads. Read with `GetPhysicalCursorPos`, it
does not move. The same managed client also failed `Toggle` and `Invoke` there (*Operation is not
valid*, *Hot key is already registered*), which the native client — the one argmind uses — did not.

## What it does not settle

- **A program that runs once per person joins the copy already open**, on whichever desktop that
  is — Word started while the person has Word open would put its window on *their* desktop, and the
  person opening Word while argmind's is running would put *theirs* on argmind's. Only checked from
  one side: the probe started Word because it was not running.
- **Keys.** There is no input there at all, so `Enter` and `Ctrl+S` are `Invoke` on a button or
  nothing. Text goes in through `ValuePattern`, which replaces the whole value.
- `ElementFromPoint`, which `patterns.rs` starts from, points into the person's desktop; a hidden
  desktop's widgets have to be found from the window down.
