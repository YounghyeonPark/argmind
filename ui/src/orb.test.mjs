// The sphere's arithmetic, without a browser.
//
// **A canvas cannot be asserted about and the projection can.** What this checks is the part that
// would be wrong silently: points landing outside the canvas, a sphere that is not round, a state
// that does not change the shape, and a jolt that never decays. None of that is visible in a
// screenshot taken at the wrong moment, and all of it is arithmetic.
//
//     node ui/src/orb.test.mjs
//
// It runs against the **compiled** `dist/orb.js`, so a stale build fails here rather than in the
// browser. There is no test framework: this project takes no dependency it does not need, and the
// whole of one is `assert` and a count.

import assert from "node:assert/strict";
import { orb } from "../dist/orb.js";

let drawn = [];
let frames = [];

/// Just enough of a canvas: it records the arcs and answers the two questions the orb asks it.
function fakeCanvas(w, h) {
  const paint = {
    globalAlpha: 1,
    fillStyle: "",
    clearRect() { drawn = []; },
    beginPath() { this._at = null; },
    arc(x, y, r) { this._at = { x, y, r, alpha: this.globalAlpha }; },
    fill() { if (this._at) { drawn.push(this._at); } },
  };
  return {
    width: 0, height: 0,
    getContext: () => paint,
    getBoundingClientRect: () => ({ width: w, height: h }),
  };
}

globalThis.window = {
  devicePixelRatio: 1,
  matchMedia: () => ({ matches: false }),
};
globalThis.performance = { now: () => now };
globalThis.requestAnimationFrame = (fn) => { frames.push(fn); };

let now = 0;
function step(ms) {
  now += ms;
  const pending = frames;
  frames = [];
  for (const fn of pending) {
    fn(now);
  }
}

let checked = 0;
function ok(what, condition) {
  assert.ok(condition, what);
  checked += 1;
}

// ---- it draws, and inside the box ---------------------------------------------------------------

const canvas = fakeCanvas(400, 300);
const eye = orb(canvas);
step(16);

ok("it drew something", drawn.length > 100);
ok("the canvas was sized for the device", canvas.width === 400 && canvas.height === 300);
ok("every point is inside the canvas",
   drawn.every((p) => p.x >= 0 && p.x <= 400 && p.y >= 0 && p.y <= 300));
ok("nothing is drawn at zero size", drawn.every((p) => p.r > 0));
ok("alpha stays in range", drawn.every((p) => p.alpha > 0 && p.alpha <= 1));

// **It is round.** The widest and tallest extents agree to within a few per cent, which a
// projection with a sign error or a missing divide would not.
const xs = drawn.map((p) => p.x);
const ys = drawn.map((p) => p.y);
const wide = Math.max(...xs) - Math.min(...xs);
const tall = Math.max(...ys) - Math.min(...ys);
ok(`round: ${wide.toFixed(1)} by ${tall.toFixed(1)}`, Math.abs(wide - tall) / wide < 0.08);
// And it is centred on the canvas rather than drifting to a corner.
ok("centred", Math.abs((Math.max(...xs) + Math.min(...xs)) / 2 - 200) < 6
   && Math.abs((Math.max(...ys) + Math.min(...ys)) / 2 - 150) < 6);

// **Depth reads as depth**: the near points are bigger than the far ones, and drawn later.
const half = Math.floor(drawn.length / 2);
const near = drawn.slice(half).reduce((a, p) => a + p.r, 0) / (drawn.length - half);
const far = drawn.slice(0, half).reduce((a, p) => a + p.r, 0) / half;
ok(`near is drawn larger than far (${near.toFixed(2)} vs ${far.toFixed(2)})`, near > far * 1.4);

// ---- the states are different shapes -------------------------------------------------------------

function extent() {
  step(16);
  const x = drawn.map((p) => p.x);
  return Math.max(...x) - Math.min(...x);
}

function height() {
  const y = drawn.map((p) => p.y);
  return Math.max(...y) - Math.min(...y);
}

eye.doing("resting");
const restingWide = extent();
const restingTall = height();

// **Listening is drawn in**, and that is the point of it: a shape that expanded would read as the
// agent putting something out rather than taking it in.
eye.doing("listening");
extent();
ok(`listening is squeezed (${height().toFixed(1)} < ${restingTall.toFixed(1)})`,
   height() < restingTall * 0.95);

// **Gone is collapsed**, so a page whose stream died does not look like a page that is thinking.
eye.doing("gone");
extent();
ok("gone is smaller still", height() < restingTall * 0.8);

// ---- a jolt swells and then returns ---------------------------------------------------------------

eye.doing("resting");
extent();
const before = extent();
eye.jolt("said");
// **Sampled at the peak of the ring**, which is half way through it — and the ring never changes
// sign, so any moment inside it would do. An earlier version of the orb rippled through zero and
// this sample happened to land on the far side of it, which is how that was caught.
step(16);
step(530);
const during = drawn.map((p) => p.x);
const swollen = Math.max(...during) - Math.min(...during);
ok(`a jolt swells it (${swollen.toFixed(1)} vs ${before.toFixed(1)})`, swollen > before * 1.03);

// And **it comes back**, rather than staying big — the ring is a second, so well past it the shape
// is the resting one again to within the breathing.
for (let i = 0; i < 40; i += 1) {
  step(50);
}
const after = extent();
ok(`it settles back (${after.toFixed(1)} vs ${restingWide.toFixed(1)})`,
   Math.abs(after - restingWide) / restingWide < 0.1);

// **A refusal goes the other way.** "It pressed the button and nothing moved" must not look like
// success, and inward is how that is said without a word.
eye.jolt("refused");
step(16);
step(400);
const pulled = drawn.map((p) => p.x);
ok("a refusal draws it in", Math.max(...pulled) - Math.min(...pulled) < before);

// ---- it costs nothing while nothing happens --------------------------------------------------------

// **Resting, it asks for about ten frames a second, not sixty.** argmind runs all day beside other
// windows; a canvas spinning at 60 fps forever is a battery cost for an idle program.
for (let i = 0; i < 40; i += 1) {
  step(50);
}
eye.doing("resting");
let painted = 0;
for (let i = 0; i < 60; i += 1) {
  drawn = [];
  step(16);
  if (drawn.length > 0) {
    painted += 1;
  }
}
// 60 animation frames over 960 ms; at ten a second that is about ten paints.
ok(`resting paints ${painted} of 60 offered frames`, painted > 4 && painted < 20);

// Thinking takes what it is offered.
eye.doing("thinking");
painted = 0;
for (let i = 0; i < 20; i += 1) {
  drawn = [];
  step(16);
  if (drawn.length > 0) {
    painted += 1;
  }
}
ok(`thinking paints every frame (${painted} of 20)`, painted === 20);

// **Stopped is stopped**, so a page that navigated away leaves nothing running.
eye.stop();
drawn = [];
for (let i = 0; i < 5; i += 1) {
  step(16);
}
ok("stop() ends it", drawn.length === 0);

// ---- reduced motion ---------------------------------------------------------------------------

// **The motion is a preference and the state is not.** With motion off there is no spin and no
// breathing, and the shape still says which state it is in.
globalThis.window.matchMedia = () => ({ matches: true });
frames = [];
const quiet = orb(fakeCanvas(400, 300));
step(16);
const firstShot = drawn.map((p) => `${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(" ");
step(500);
const laterShot = drawn.map((p) => `${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(" ");
ok("with reduced motion it does not move", firstShot === laterShot);
quiet.doing("listening");
step(16);
const listeningShot = drawn.map((p) => `${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(" ");
ok("but the state still changes the shape", listeningShot !== laterShot);
quiet.stop();

console.log(`  orb: ${checked} checks ok`);
