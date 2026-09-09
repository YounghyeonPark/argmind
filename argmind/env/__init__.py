# SPDX-License-Identifier: Apache-2.0
"""`argmind.env` — the environments. One file each, and the only place a platform is named.

`argmind/core/` is written against `Eyes` and `Hands` and a test walks its source to keep it that
way, so everything platform-specific arrives here.

| | |
|---|---|
| `windows.py` | UI Automation + `SendInput`. **Built and measured**: 41–266 ms for a window's widgets |
| *android.py* | `uiautomator` + `input tap`. **Later** — the shape is the same, the numbers are not in hand |
| *robot.py* | **Does not fit, and is not going to be forced.** A room has no accessibility tree; anything filling `Eyes` there would be a detector guessing names and boxes, which puts probabilistic output through an interface whose whole contract is that it is exact |

Imported lazily by the caller rather than re-exported here: importing this package must not require
`comtypes`, or the core stops being testable on a machine that has none.
"""

__all__: list[str] = []
