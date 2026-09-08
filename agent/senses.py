# SPDX-License-Identifier: Apache-2.0
"""What the agent can see and hear, and what it cannot.

The owner's I/O environment: *"pc화면, 웹캠, 마이크, 키보드, 마우스 입력"*. Three of those are here.
**Audio input is absent, not unfinished**, and the distinction is this project's own — `mss`,
`sounddevice`, `soundcard` and `pyaudio` are all missing from this interpreter, so a microphone sense
would be a stub pretending to be a sensor. It raises with the reason instead.

Everything is a small object with a `read()`, so the loop can be driven by a test that owns no
display and no camera. `demo/pointer.py` learned that the hard way: *a test that could not run
without a camera is a demo that rots.*

**Measured on this machine.** A full-screen grab of 2560×1440 costs **34 ms**, which is the loop's
ceiling at about 29 Hz and is an order more than everything downstream of it — describing the frame
is ~1 ms. So the screen is the budget, and nothing here tries to be clever below it.
"""

from __future__ import annotations

import time

import numpy as np

__all__ = ["ScreenSense", "WebcamSense", "MicrophoneSense", "KeyboardMouseSense", "MISSING_AUDIO"]

MISSING_AUDIO = ("no audio capture library is installed in this interpreter (tried sounddevice, "
                 "soundcard, pyaudio). This is absent rather than unfinished: a microphone sense "
                 "that returned silence would be worse than one that says it is not here.")


class ScreenSense:
    """The PC screen, as an HxWx3 array in [0, 1].

    `PIL.ImageGrab` rather than `mss`, because `mss` is not installed and `PIL` is. The grab is
    full-screen and then decimated by the caller — grabbing a region would be faster and is not done,
    because which region matters is exactly what the agent is supposed to work out.
    """

    def __init__(self) -> None:
        from PIL import ImageGrab                      # noqa: PLC0415 - optional at import time

        self._grab = ImageGrab.grab
        probe = self._grab()
        self.size = probe.size                          # (width, height)

    def read(self) -> np.ndarray:
        im = self._grab().convert("RGB")
        return np.asarray(im, dtype=np.float32) / np.float32(255.0)


class WebcamSense:
    """The webcam, as an HxWx3 array in [0, 1], BGR→RGB.

    Opened once and held. A camera that cannot be opened raises here rather than returning empty
    frames forever, which is the failure mode that makes a loop look alive while seeing nothing.
    """

    def __init__(self, index: int = 0) -> None:
        import cv2                                      # noqa: PLC0415

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(int(index))
        if not self._cap.isOpened():
            raise RuntimeError(f"webcam {index} would not open")

    def read(self) -> np.ndarray:
        ok, frame = self._cap.read()
        if not ok:
            raise RuntimeError("the webcam stopped returning frames")
        return np.asarray(frame[:, :, ::-1], dtype=np.float32) / np.float32(255.0)

    def close(self) -> None:
        self._cap.release()


class MicrophoneSense:
    """Not built. See `MISSING_AUDIO`."""

    def __init__(self, *_a, **_k) -> None:
        raise NotImplementedError(MISSING_AUDIO)


class KeyboardMouseSense:
    """What the human is doing: cursor position, buttons, and modifier keys.

    Windows only, through `user32`, with no dependency. This is a *sense* and not a hook: it samples
    the state when asked rather than intercepting events, because an agent that swallowed the user's
    input would be a keylogger with extra steps.
    """

    def __init__(self) -> None:
        import ctypes                                   # noqa: PLC0415

        self._ct = ctypes
        self._u = ctypes.windll.user32

        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        self._POINT = POINT

    def read(self) -> dict:
        pt = self._POINT()
        self._u.GetCursorPos(self._ct.byref(pt))
        down = lambda vk: bool(self._u.GetAsyncKeyState(vk) & 0x8000)    # noqa: E731
        return {
            "cursor": (int(pt.x), int(pt.y)),
            "left": down(0x01), "right": down(0x02), "middle": down(0x04),
            "shift": down(0x10), "ctrl": down(0x11), "alt": down(0x12),
            "t": time.perf_counter(),
        }
