# SPDX-License-Identifier: Apache-2.0
"""The two environments behind the table in `demo/policy_health.py`.

Neither is a benchmark and neither is part of the library. They exist so that the figures the
module quotes can be regenerated, after an audit here found that quoted sweep numbers had come
from a run leaving no trace.

Both use `argmind` exactly as a caller would: a frozen random convolutional backbone from
`argmind.backbone` produces the features, and `RecursiveLeastSquares` learns a linear Q per
action on top. RLS on the bootstrapped target `r + gamma * max_a' Q(s', a')` is least-squares
TD, so no library change is involved -- that is the point.

part tracking (`track_*`)
    Three blobs in a rigid triangle plus an independently drifting bright distractor. Three
    trackers each see a local crop and the other trackers' positions relative to their own. The
    objective is abstract: salient, distinct, rigid -- **no part is named**, because a frozen
    backbone cannot know which blob is an eye. `W_RIGID` weights the rigidity term and is the
    swept axis.

panning (`pan_*`)
    One agent on a wide frame sees only a crop and chooses where the crop goes next. Reward is
    the novelty of what it observes: how far the embedding moved. Grounded and non-circular --
    it is never told where the blob is -- and it is maximised by panning back and forth, which
    is the failure the module's docstring reports.
"""

from __future__ import annotations

import itertools

import numpy as np

from argmind.adapt import RecursiveLeastSquares
from argmind.backbone import Backbone

# ---------------------------------------------------------------- part tracking

H, W = 72, 96
N_PART = 3
VIEW = 24
STEP = 3
#: stay, left, right, up, down, as (dy, dx). Index 0 is the do-nothing action.
ACTIONS = ((0, 0), (0, -STEP), (0, STEP), (-STEP, 0), (STEP, 0))
#: Which action undoes which; -1 for none. `demo.policy_health.action_stats` takes this.
OPPOSITE = (-1, 2, 1, 4, 3)
GAMMA = 0.9
SEP = 12.0
W_SAL = W_DIST = 1.0
START = ((20.0, 20.0), (76.0, 20.0), (48.0, 52.0))


def scene(t: int, seed: int = 0):
    """Three blobs in a rigid triangle, translating and rotating slowly -- a 'face' with no
    labels on it -- plus a distractor that is bright but not part of the rigid set. An objective
    that only rewards brightness will happily sit on the distractor; `rigid` is what refuses
    it."""
    g = np.random.default_rng(seed)
    cx, cy = 0.5 + 0.12 * np.sin(t * 0.05), 0.5 + 0.08 * np.cos(t * 0.04)
    ang = 0.3 * np.sin(t * 0.03)
    pts = []
    for dx, dy in ((-0.10, -0.06), (0.10, -0.06), (0.0, 0.09)):
        rx = dx * np.cos(ang) - dy * np.sin(ang)
        ry = dx * np.sin(ang) + dy * np.cos(ang)
        pts.append(((cx + rx) * W, (cy + ry) * H))
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.full((H, W, 3), 0.30)
    for px, py in pts:
        img += np.exp(-(((xx - px) ** 2 + (yy - py) ** 2) / (2 * 4.5 ** 2)))[:, :, None] * 0.8
    dxp, dyp = (0.15 + 0.5 * ((t * 0.011) % 1.0)) * W, 0.85 * H
    img += np.exp(-(((xx - dxp) ** 2 + (yy - dyp) ** 2) / (2 * 4.5 ** 2)))[:, :, None] * 0.8
    return (img + g.standard_normal(img.shape) * 0.015).astype("f4"), pts, (dxp, dyp)


def salient(img, p, r: int = 5) -> float:
    """Local contrast. The caller can compute it and it needs no semantics."""
    x, y = int(p[0]), int(p[1])
    patch = img[max(0, y - r):min(H, y + r + 1), max(0, x - r):min(W, x + r + 1), 0]
    return float(patch.mean() - img[:, :, 0].mean())


def clamp(p):
    return (float(np.clip(p[0], VIEW // 2, W - VIEW // 2)),
            float(np.clip(p[1], VIEW // 2, H - VIEW // 2)))


def crop(img, p):
    x, y = int(p[0]), int(p[1])
    return img[y - VIEW // 2:y + VIEW // 2, x - VIEW // 2:x + VIEW // 2, :]


def pair_d(pos):
    return {(i, j): float(np.linalg.norm(np.array(pos[i]) - np.array(pos[j])))
            for i, j in itertools.combinations(range(N_PART), 2)}


def reward(img, pos, i: int, prev_d, w_rigid: float) -> float:
    """Tracker `i`'s own share, so nothing global has to be credited. `prev_d` holds the
    pairwise distances one step ago, which is what makes `rigid` a per-step quantity rather
    than a whole-episode statistic."""
    r = W_SAL * salient(img, pos[i])
    for j in range(N_PART):
        if j == i:
            continue
        d = float(np.linalg.norm(np.array(pos[i]) - np.array(pos[j])))
        r += W_DIST * min(d, SEP) / SEP / (N_PART - 1)
        if prev_d is not None:
            key = (min(i, j), max(i, j))
            r -= w_rigid * abs(d - prev_d[key]) / max(prev_d[key], 1e-6) / (N_PART - 1)
    return r


def track_state(bb, img, pos, i: int, scale: float):
    """The crop's embedding says WHAT IS UNDER ME; the relative offsets say WHERE THE OTHERS
    ARE FROM HERE. Both are translation-invariant, which forces the policy to be. Adding the
    tracker's own absolute position breaks that invariance and was measured to make the result
    *worse* (905 against 1008), so it is not included."""
    emb = bb.features(crop(img, pos[i])) * scale
    rel = []
    for j in range(N_PART):
        if j == i:
            continue
        rel += [(pos[j][0] - pos[i][0]) / W, (pos[j][1] - pos[i][1]) / H]
    return np.concatenate([emb, rel, [1.0]])


def onehot(s, a: int, n_s: int, n_actions: int = len(ACTIONS)):
    """A separate linear map per action: the standard linear-state / tabular-action form, which
    keeps the head linear and therefore keeps RLS applicable."""
    v = np.zeros(n_actions * n_s)
    v[a * n_s:(a + 1) * n_s] = s
    return v


def track_backbone():
    bb = Backbone([("conv", 16, 5, 3)], (VIEW, VIEW, 3), dtype="f4", seed=1, pool=2)
    cal = [crop(scene(k, seed=k)[0], (48, 36)) for k in range(30)]
    scale = 1.0 / max(float(np.array([bb.features(c) for c in cal]).std()), 1e-9)
    return bb, scale, bb.n_outputs + 2 * (N_PART - 1) + 1


def track_run(bb, scale, qs, w_rigid: float, n: int = 400, eps: float = 0.0,
              learn: bool = False, zero_reward: bool = False, seed: int = 0,
              collect: bool = False):
    """One episode. Returns the total reward and, if asked, the action stream per tracker."""
    g = np.random.default_rng(seed)
    pos = [clamp(p) for p in START]
    prev = [None] * N_PART
    prev_d = None
    total = 0.0
    n_s = None
    streams = [[] for _ in range(N_PART)]
    for k in range(n):
        img, _parts, _dis = scene(k, seed=k)
        states, acts = [], []
        for i in range(N_PART):
            s = track_state(bb, img, pos, i, scale)
            n_s = len(s)
            states.append(s)
            vals = [float(qs[i].predict(onehot(s, j, n_s))[0]) for j in range(len(ACTIONS))]
            a = int(g.integers(len(ACTIONS))) if g.random() < eps else int(np.argmax(vals))
            acts.append(a)
        rs = [reward(img, pos, i, prev_d, w_rigid) for i in range(N_PART)]
        total += float(np.sum(rs))
        if zero_reward:
            rs = [0.0] * N_PART
        if learn:
            for i in range(N_PART):
                if prev[i] is None:
                    continue
                nxt = max(float(qs[i].predict(onehot(states[i], j, n_s))[0])
                          for j in range(len(ACTIONS)))
                qs[i].update(prev[i], [rs[i] + GAMMA * nxt])
        for i in range(N_PART):
            prev[i] = onehot(states[i], acts[i], n_s)
            if collect:
                streams[i].append(acts[i])
        prev_d = pair_d(pos)
        pos = [clamp((pos[i][0] + ACTIONS[acts[i]][1],
                      pos[i][1] + ACTIONS[acts[i]][0])) for i in range(N_PART)]
    return (total, streams) if collect else (total, None)


def track_train(bb, scale, n_s: int, w_rigid: float, seed: int, zero: bool = False,
                episodes: int = 10):
    qs = [RecursiveLeastSquares(len(ACTIONS) * n_s, 1, forgetting=0.999, p0=10.0, dtype="f8")
          for _ in range(N_PART)]
    for ep in range(episodes):
        track_run(bb, scale, qs, w_rigid, eps=max(0.05, 0.6 * (0.75 ** ep)), learn=True,
                  zero_reward=zero, seed=seed * 50 + ep)
    return qs


# --------------------------------------------------------------------- panning

FRAME_H, FRAME_W = 96, 160
VIEW_H, VIEW_W = 48, 64
PAN_STEP = 16
PAN_ACTIONS = ((0, 0), (0, -PAN_STEP), (0, PAN_STEP), (-PAN_STEP, 0), (PAN_STEP, 0))


def world(n: int, seed: int = 0):
    """A blob on a smooth bouncing path across the wide frame."""
    g = np.random.default_rng(seed)
    x, y = 0.5, 0.5
    vx, vy = g.choice([-1, 1]) * 0.018, g.choice([-1, 1]) * 0.013
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
    out, pos = [], []
    for _ in range(n):
        x, y = x + vx, y + vy
        if not 0.12 < x < 0.88:
            vx, x = -vx, min(max(x, 0.12), 0.88)
        if not 0.12 < y < 0.88:
            vy, y = -vy, min(max(y, 0.12), 0.88)
        d = ((xx - x * FRAME_W) ** 2 + (yy - y * FRAME_H) ** 2) / (2 * 11.0 ** 2)
        img = np.full((FRAME_H, FRAME_W, 3), 0.35) + np.exp(-d)[:, :, None] * 0.95
        out.append((img + g.standard_normal(img.shape) * 0.02).astype("f4"))
        pos.append((x * FRAME_W, y * FRAME_H))
    return out, pos


def pan_crop(frame, cy, cx):
    cy = int(min(max(cy, VIEW_H // 2), FRAME_H - VIEW_H // 2))
    cx = int(min(max(cx, VIEW_W // 2), FRAME_W - VIEW_W // 2))
    return frame[cy - VIEW_H // 2:cy + VIEW_H // 2, cx - VIEW_W // 2:cx + VIEW_W // 2, :], cy, cx


def pan_backbone():
    bb = Backbone([("conv", 24, 5, 4)], (VIEW_H, VIEW_W, 3), dtype="f4", seed=1, pool=3)
    cal, _ = world(40, seed=99)
    sample = np.array([bb.features(pan_crop(f, FRAME_H // 2, FRAME_W // 2)[0]) for f in cal])
    return bb, 1.0 / max(float(sample.std()), 1e-9), bb.n_outputs + 1


def pan_run(bb, scale, frames, positions, policy: str, q=None, eps: float = 0.0,
            learn: bool = False, zero_reward: bool = False, seed: int = 0):
    """One pass over the world. Returns the total reward and the action stream."""
    g = np.random.default_rng(seed)
    cy, cx = FRAME_H // 2, FRAME_W // 2
    prev_emb = prev_feat = None
    total = 0.0
    stream = []
    n_s = bb.n_outputs + 1
    for k, frame in enumerate(frames):
        view, cy, cx = pan_crop(frame, cy, cx)
        emb = bb.features(view) * scale
        s = np.concatenate([emb, [1.0]])
        r = 0.0 if prev_emb is None else float(np.linalg.norm(emb - prev_emb))
        if zero_reward:
            r = 0.0
        total += r
        if policy == "random":
            a = int(g.integers(len(PAN_ACTIONS)))
        elif policy == "oracle":
            px, py = positions[k]
            a = int(np.argmin([abs(cx + dx - px) + abs(cy + dy - py)
                               for dy, dx in PAN_ACTIONS]))
        else:
            vals = [float(q.predict(onehot(s, j, n_s))[0]) for j in range(len(PAN_ACTIONS))]
            a = int(g.integers(len(PAN_ACTIONS))) if g.random() < eps else int(np.argmax(vals))
        if learn and prev_feat is not None:
            nxt = max(float(q.predict(onehot(s, j, n_s))[0]) for j in range(len(PAN_ACTIONS)))
            q.update(prev_feat, [r + GAMMA * nxt])
        prev_feat, prev_emb = onehot(s, a, n_s), emb
        stream.append(a)
        dy, dx = PAN_ACTIONS[a]
        cy, cx = cy + dy, cx + dx
    return total, stream
