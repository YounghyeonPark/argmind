# `demo/` — argmind driven by a webcam, on a PC

**Not part of the library.** Nothing under `argmind/` or `cpp/` imports anything here, and
this directory has dependencies the library must never acquire — OpenCV for the camera and
the window. `python -m pytest` at the repository root does not run it. That separation is the
point: argmind is a library that allocates nothing and talks to no hardware, and a demo that
blurred that line would be arguing against the design it is demonstrating.

```
pip install -r demo/requirements.txt
python -m demo.pointer            # from the repository root
```

## What it shows, and why each part is here

A dot follows your hand. That is not the interesting part — a frozen model can do that. What
a frozen model **cannot** do is every line of this table, and each one exercises a different
piece of the library:

| what you see | what it is |
|---|---|
| you hold the button on yourself and drag, and it starts following | online adaptation from labels that arrive when *you* supply them |
| the camera runs at ~30 Hz and your clicks do not | `Aligner` — two independent clocks, which is what it was built for |
| move the laptop or change the lighting and it re-converges | drift tracking, the original reason this library exists |
| **cover the camera** and it freezes instead of being destroyed | `Guard` — and it says which check tripped |
| uncover it and it thaws | bounded rollback and explicit recovery |
| quit and restart: it still works | `persist`, two slots with a CRC and a configuration fingerprint |

Press `h` for the keys.

## Measured

| | |
|---|---|
| backbone + pipeline, host, numpy | **931 fps** — a 30 Hz camera is not the constraint |
| position error after ~400 taught frames | **0.035**, against **0.24** for guessing the centre |
| head state | **1,160 B** (NLMS, 145 features, two outputs) |
| backbone weights | **4,960 B** (3 layers, 3x3 grid, 144 features) |
| covered camera, frame std | **0.000** against 0.194-0.197 for every legitimate scene |

`demo/test_pointer.py` drives all of it headlessly with synthetic frames and runs in the root
suite — no camera and no OpenCV, because `cv2` is imported inside `main()`. **A demo nobody
runs rots.**

## What is honest about it

**The backbone is untrained, so the accuracy is poor.** Its weights are random. Measured on a
synthetic moving blob, a shallow random backbone predicts position to about **0.23 of the
target's own standard deviation** — the dot follows you roughly, not precisely. Importing
trained weights would fix that and is a separate piece of work; the file format is already
there, since `Backbone` keeps its weights in one flat block that `persist` carries.

**The architecture is shallow and grid-pooled on purpose, and both were measured.** Random
convolutional features work shallow and carry nothing deep: predicting brightness reads 0.009
at one layer and **1.119 at twenty-seven**, which is worse than a constant. And global average
pooling discards *where* anything was, so a position task reads 1.076 globally against 0.231 on
a 3×3 grid. A deep globally-pooled stack — which is what a real MobileNet is — would make this
demo impossible with untrained weights.

**The demo's job is the library's behaviour, not the backbone's accuracy.** All six rows above
are visible whether the dot tracks well or badly.

**The guard's external check watches the raw frame's standard deviation, and two other
statistics were tried first and rejected.** Distance from a baseline embedding saturates — a
covered camera reads 1.000 against a dim room's 0.877, which is not a detector — and the
embedding's norm overlaps outright, 0.604 covered against 0.578 for a dim room. Frame standard
deviation separates completely: **0.000 covered against 0.194-0.197 for every legitimate
scene**. It is also the right *kind* of statistic for `set_external`, because the head
genuinely cannot see it: a flat grey frame produces a perfectly plausible embedding that the
head fits happily.

**You teach by dragging, not clicking, and that is a measurement rather than a preference.**
NLMS at 145 features reaches a radial error of 0.058 after 300 updates and gets nowhere after
50 — and nobody clicks fifty times.

## Keys

| key | |
|---|---|
| hold and drag | teach: "I am here now", one label per frame while held |
| `f` | freeze by hand |
| `t` | thaw |
| `r` | reset the head and forget everything |
| `s` | save to `demo/state/`, `l` load |
| `q` | quit (it saves) |
