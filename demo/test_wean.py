# SPDX-License-Identifier: Apache-2.0
"""The wean experiment: can the classical labeller be deleted? Measured — **no**, and by how much.

The 2026-08-14 decision says argmind must do the learning and inferencing itself, and that
`demo/label_sources.py`'s 1,087 lines of classical perception are scaffolding to be replaced. What
this repository had been calling "the wean result" is **free-running IoU**, which is a *frozen-model*
evaluation: the teacher stops and so does learning. That is not the question. Deleting the labeller
means the head must keep learning from a signal it produces itself.

**Axes fixed before running**: regime in {frozen after withdrawal, self-trained on its own hardened
output, teacher throughout as the ceiling}; scene `jump_every=1`, where persistence collapses to
0.126 so a coasting head cannot fake the score.

**Pass criteria, stated before the numbers existed**: self-trained IoU >= frozen, or self-training
is harmful rather than useful; and the self-produced foreground fraction must stay near the truth's,
the known failure mode being degenerate collapse — this repository already records that zero is a
fixed point of a self-fed map.

| regime | IoU mean | IoU last quarter |
|---|---|---|
| teacher throughout (ceiling) | 0.943 | 0.941 |
| frozen at withdrawal | 0.938 | 0.931 |
| **self-trained** | **0.839** | **0.804** |

**Criterion 1 fails decisively, and it keeps failing.** Over 4,000 steps of self-training the IoU
falls monotonically and is still falling at the end:

    self    0.851 0.802 0.775 0.761 0.748 0.736 0.719 0.717
    frozen  0.940 0.935 0.934 0.940 0.937 0.934 0.933 0.937

**Criterion 2 passes, which makes the finding sharper rather than softer.** It is *not* the
zero-fixed-point collapse recorded elsewhere in this project: the thresholded foreground fraction
moves 0.084 → 0.102 *toward* the truth's 0.129 while accuracy falls. The mask is **spreading into
the wrong cells**, not vanishing — so a degeneracy detector watching for collapse would not see this.

**What it means for the decision.** The head perceives: frozen, it holds 0.937 from the frame on a
scene where persistence scores 0.126. What it cannot do is supply its own training signal — every
round of self-labelling costs accuracy, steadily and without a floor in sight. So the labeller
cannot be deleted by self-training on the head's own output, and something that is not the head's
own output is what the 2026-08-14 decision actually requires.
"""

from __future__ import annotations

import numpy as np

from argmind.adapt import RecursiveLeastSquares
from demo.map_head import Pyramid, map_iou
from demo.pointer import MAP_H, MAP_W, build, map_basis, map_teacher
from demo.test_map_output import frame_features
from demo.test_map_perception import scene

THRESH = 0.4          #: the demo's own display threshold for calling a cell foreground


def wean(regime, *, t1=300, n=1000, jump_every=1):
    """Returns `(iou_mean, iou_last_quarter, foreground_fraction, true_foreground_fraction)`."""
    _bb, _h, _g, features, *_rest = build(1, "raw", output="map")
    head = RecursiveLeastSquares(features.dim + 1, MAP_H * MAP_W,
                                 forgetting=1.0, p0=100.0, dtype="f8")
    pyr, iou, fg, fg_true = Pyramid(MAP_H, MAP_W), [], [], []
    for k in range(n):
        img, truth = scene(k, jump_every=jump_every, seed=k)
        x = np.concatenate([map_basis(frame_features(img), pyr), [1.0]])
        target = map_teacher(truth)
        pred = np.clip(np.asarray(head.predict(x), dtype=np.float64), 0.0, 1.0)
        if k < t1:
            head.update(x, target)
        else:
            iou.append(map_iou(pred, target))
            fg.append(float((pred >= THRESH).mean()))
            fg_true.append(float((target > 0).mean()))
            if regime == "teacher":
                head.update(x, target)
            elif regime == "self":
                head.update(x, (pred >= THRESH).astype(np.float64))
        pyr.observe(pred if k >= t1 else target)
    q = len(iou) // 4
    return (float(np.mean(iou)), float(np.mean(iou[-q:])),
            float(np.mean(fg)), float(np.mean(fg_true)))


def test_self_training_on_its_own_labels_makes_the_head_WORSE():
    """**The wean result, and it is negative.** Criterion 1 was that self-training must be at least
    as good as freezing. It is not, and the gap widens with time."""
    frozen_mean, frozen_last, _f, _t = wean("frozen")
    self_mean, self_last, _f2, _t2 = wean("self")

    assert frozen_mean > 0.85, (
        f"the frozen control reads {frozen_mean:.3f}; it used to hold 0.94, and without a head that "
        "perceives at all this comparison says nothing")
    assert self_mean < frozen_mean, (
        f"self-training ({self_mean:.3f}) is no longer worse than freezing ({frozen_mean:.3f}) — "
        "if that became true the wean experiment has changed outcome and the 2026-08-14 decision "
        "is closer than this file says")
    assert self_last < self_mean, (
        f"self-training's last quarter ({self_last:.3f}) is not below its own mean ({self_mean:.3f}); "
        "the finding is that it degrades *over time*, not merely that it starts lower")


def test_the_ceiling_control_shows_the_scene_is_learnable_at_all():
    """A negative result needs the positive control beside it, or 'self-training fails' and 'nothing
    works here' are indistinguishable."""
    teacher_mean, _l, _f, _t = wean("teacher")
    frozen_mean, _l2, _f2, _t2 = wean("frozen")
    assert teacher_mean > 0.85, f"the teacher ceiling is only {teacher_mean:.3f}"
    assert abs(teacher_mean - frozen_mean) < 0.1, (
        f"the teacher ({teacher_mean:.3f}) and frozen ({frozen_mean:.3f}) controls have separated; "
        "continued teaching used to buy almost nothing, and that is what makes the self-training "
        "loss attributable to the self-labels rather than to updating at all")


def test_it_is_NOT_the_zero_fixed_point_collapse_this_project_records_elsewhere():
    """Criterion 2, and it passing is what makes the finding precise. The mask does not vanish; it
    spreads into the wrong cells while its area moves *toward* the truth. A degeneracy detector
    watching for collapse would miss this entirely."""
    _m, _l, fg_self, fg_true = wean("self")
    assert fg_self > 0.02, (
        f"the self-trained foreground fraction is {fg_self:.3f} — it HAS collapsed toward zero, "
        "which is a different failure from the one recorded, and the docstring should change with it")
    assert fg_self < fg_true * 1.5, f"foreground exploded to {fg_self:.3f} against a true {fg_true:.3f}"
