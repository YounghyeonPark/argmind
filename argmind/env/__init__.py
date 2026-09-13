# SPDX-License-Identifier: Apache-2.0
"""`argmind.env` — the environments. One file each, and the only place a platform is named.

`argmind/core/` is written against `Eyes` and `Hands` and a test walks its source to keep it that
way, so everything platform-specific arrives here.

| | |
|---|---|
| `windows.py` | UI Automation + `SendInput`. **Built and measured**: 41–266 ms for a window's widgets |
| *android.py* | `uiautomator` + `input tap`. **Later** — the shape is the same, the numbers are not in hand |
| *robot.py* | **Does not fit, and is not going to be forced.** A room has no accessibility tree; anything filling `Eyes` there would be a detector guessing names and boxes, which puts probabilistic output through an interface whose whole contract is that it is exact |

**Voice is a different seam and so it is different files.** `Eyes` and `Hands` are how the agent
touches the machine's other applications; a microphone and a speaker are how it touches the person,
and `argmind/voice.py` — not `core/contract.py` — is the protocol they implement. They are split
because only one of them is a Windows facility:

| | |
|---|---|
| `sapi.py` | `Speaker` over SAPI. **The voice is picked per utterance**: an English voice given pure Korean emits 0 bytes, 79 of 79 strings, and given Korean with a number in it emits the number alone |
| `whisper.py` | `Listener` over `sounddevice` and `openai/whisper-large-v3-turbo`. Transcription **median 234 ms**; a repetition guard, and no pace cut-off until a person's audio exists |

Imported lazily by the caller rather than re-exported here: importing this package must not require
`comtypes`, or the core stops being testable on a machine that has none.
"""

__all__: list[str] = []
