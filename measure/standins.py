# SPDX-License-Identifier: Apache-2.0
"""The two stubs every fixture harness stands a `Session` on: `Fixed` eyes, `Nothing` hands.

Nothing here is a measurement. It exists so there is **one** definition of "a screen that does not
change" and "hands that touch nothing", and so that the definition does not live inside a harness.

**It was in `measure/repair.py` until 2026-09-13**, and `measure/conversation.py` imported it from
there. That made the shared file import the leaf: the moment `repair.py` wanted anything back out of
`conversation.py` — `Recorder`, `recent_block`, `drive`, all of which it had hand-copied — the two
formed a cycle that resolved differently depending on which module was named on the command line.
`python -m measure.repair` would have worked and `python -m measure.conversation` would have raised
`ImportError` on `Recorder`, which is the kind of breakage that looks like a bad merge rather than a
bad import graph. Moved rather than duplicated.
"""

from __future__ import annotations


class Fixed:
    """Eyes that always show the same screen. Nothing is clicked, so nothing changes.

    **This is a real condition of every number taken through it**, and the flattering direction:
    the shipped `Eyes` reads a live window where a listed widget is gone 13 times in 14 over the
    second a model call takes, so `Session.find()`'s staleness path is never reached here and
    `Session.vanished` is always 0. A harness on `Fixed` measures the decision, never the race.
    """

    def __init__(self, widgets) -> None:
        self.widgets = list(widgets)

    def foreground(self):
        return list(self.widgets)

    def timed_foreground(self):
        #: A plausible non-zero, not a real reading. Nothing should quote it, and nothing does —
        #: the harnesses time the model call themselves with `perf_counter`.
        return list(self.widgets), 0.001


class Nothing:
    """Hands that accept every instruction and do nothing.

    `live = False` is the flag an interface reads to know it must not claim anything happened. Each
    method returns `{}` rather than `None` so a caller that stores the result stores a mapping in
    both the real and the stub case, and never a `NoneType` that only shows up under fixtures.
    """

    live = False

    def click(self, widget, button="left"):
        return {}

    def type_text(self, text):
        return {}

    def press(self, key):
        return {}
