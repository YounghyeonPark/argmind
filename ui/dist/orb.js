// The sphere in the middle of the page — what argmind is doing, as a shape rather than a word.
//
// **Why this exists.** Everything the agent does already arrives as an event, and every one of them
// currently lands as a line of text or a CSS class: `busy` toggles the words 생각 중 in the corner.
// A person working beside it should be able to tell *thinking* from *listening* from *nothing at
// all* out of the corner of their eye, without reading. That is the whole claim; it is not
// decoration and it adds no information the page did not already have.
//
// **No dependency, and that is the licence rule rather than a preference.** A point cloud on a
// sphere, a rotation, a perspective divide and two sinusoids is about a hundred lines of arithmetic
// — a 3D library would be a dependency to audit for something `<canvas>` already does.
//
// **It must not cost anything while somebody works.** argmind runs all day beside other windows, so
// a canvas spinning at 60 fps forever would be a battery cost for an idle program. Resting, this
// draws about ten frames a second; busy, it takes the frames the browser offers; disconnected, it
// stops. `prefers-reduced-motion` turns the motion off entirely and leaves the shape, which still
// carries the state.
/// Points spread evenly over a sphere — the Fibonacci lattice, because a latitude/longitude grid
/// bunches at the poles and the bunching is exactly what the eye picks out as an artefact.
function shell(count) {
    const out = [];
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < count; i += 1) {
        const y = 1 - (i / (count - 1)) * 2;
        const ring = Math.sqrt(Math.max(0, 1 - y * y));
        const turn = golden * i;
        out.push({ x: Math.cos(turn) * ring, y, z: Math.sin(turn) * ring });
    }
    return out;
}
/// How far out a point sits, as a function of where it is and what time it is.
//
// Two sinusoids over fixed directions rather than a noise function: it is smooth, it is seamless on
// a sphere because it is a function of the point itself, and it is four multiplies.
function swell(p, t, speed, scale) {
    const a = Math.sin(p.y * 3.1 + t * speed);
    const b = Math.sin((p.x + p.z) * 2.3 - t * speed * 0.7);
    return (a + b * 0.6) * scale;
}
const HOW = {
    //: spin: turns per second. wave: how fast the surface travels. swell: how far it moves.
    resting: { spin: 0.035, wave: 0.55, swell: 0.035, squeeze: 1, fps: 10 },
    //: Faster and rougher — the one state a person most wants to read at a glance, because it is the
    //: one where nothing else on the page changes for a second or two.
    thinking: { spin: 0.16, wave: 2.4, swell: 0.1, squeeze: 1, fps: 0 },
    //: Drawn inward and tight. Listening is the agent *taking* something in, and an expanding shape
    //: would read as the opposite.
    listening: { spin: 0.05, wave: 1.2, swell: 0.055, squeeze: 0.84, fps: 0 },
    //: Collapsed and still. The stream is gone and the page cannot know anything.
    gone: { spin: 0, wave: 0, swell: 0, squeeze: 0.55, fps: 0 },
};
/// How hard each thing hits, and how long it rings. Seconds.
const JOLTS = {
    //: A reply. The biggest ordinary event and the one the person is waiting for.
    said: { push: 0.3, ring: 1.1 },
    //: A transcript arrived. Smaller — it is the beginning of a turn, not the end of one.
    heard: { push: 0.18, ring: 0.7 },
    //: Something the agent noticed by itself. **Sharp and short**, because it is an interruption and
    //: should read as one.
    noticed: { push: 0.38, ring: 0.55 },
    //: A confirmed action went out.
    acted: { push: 0.34, ring: 0.9 },
    //: It refused, or the target had gone. Inward, so it does not read as success.
    refused: { push: -0.26, ring: 0.8 },
};
export function orb(canvas) {
    const paint = canvas.getContext("2d");
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let points = shell(520);
    let doing = "resting";
    let visible = 1;
    // Each jolt is a push and the moment it landed; they add, so two events in a second do not
    // cancel one another out into stillness.
    let jolts = [];
    let running = true;
    let orbiting = false;
    let last = 0;
    let turned = 0;
    function size() {
        const box = canvas.getBoundingClientRect();
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = Math.max(1, Math.round(box.width * dpr));
        const h = Math.max(1, Math.round(box.height * dpr));
        if (canvas.width !== w || canvas.height !== h) {
            canvas.width = w;
            canvas.height = h;
            // Fewer points in a small window: the density that reads as a sphere at 400px reads as a
            // smudge at 150.
            points = shell(Math.max(200, Math.min(700, Math.round((w * h) / 900))));
        }
        return { w, h };
    }
    function draw(now) {
        if (!paint) {
            return;
        }
        const { w, h } = size();
        const how = HOW[doing];
        const seconds = now / 1000;
        //: Spent jolts are dropped rather than kept at zero, so the resting check below can be a
        //: length test and the frame budget actually falls back to ten a second.
        jolts = jolts.filter((j) => seconds - j.at < j.ring);
        let kick = 0;
        for (const j of jolts) {
            const age = (seconds - j.at) / j.ring;
            //: A half-cycle of a sine, so it rises and returns to exactly zero rather than snapping,
            //: times a ripple that **never crosses zero**. The first version multiplied by `cos(age * 9)`
            //: and that reverses a fifth of the way through — so a reply spent most of its ring pulling
            //: *inward*, which is the shape a refusal is supposed to have. Outward and inward are the
            //: only two things this says without words; they cannot be allowed to trade places.
            kick += j.push * Math.sin(age * Math.PI) * (0.75 + 0.25 * Math.cos(age * 12));
        }
        if (!still) {
            turned += how.spin * (now - last) * 0.006;
        }
        last = now;
        paint.clearRect(0, 0, w, h);
        const mid = { x: w / 2, y: h / 2 };
        const span = Math.min(w, h) * 0.34;
        const cos = Math.cos(turned);
        const sin = Math.sin(turned);
        //: A fixed tilt, so the pole is never pointed straight at the viewer — a sphere seen exactly
        //: down its axis shows the lattice as rings and stops looking like a sphere.
        const tiltC = Math.cos(0.45);
        const tiltS = Math.sin(0.45);
        const on = [];
        for (const p of points) {
            const wave = still ? 0 : swell(p, seconds, how.wave, how.swell);
            const r = 1 + wave + kick;
            //: Squeezed towards the middle rather than scaled: listening should look *held in*, and a
            //: uniformly smaller sphere would just look further away.
            const x = p.x * r;
            const y = p.y * r * how.squeeze;
            const z = p.z * r;
            const rx = x * cos - z * sin;
            const rz = x * sin + z * cos;
            const ry = y * tiltC - rz * tiltS;
            const rz2 = y * tiltS + rz * tiltC;
            //: A gentle perspective divide. Enough to tell front from back, not enough to distort.
            const near = 1 / (2.6 - rz2);
            on.push({ sx: mid.x + rx * span * near * 2.6, sy: mid.y + ry * span * near * 2.6,
                depth: rz2 });
        }
        //: **The ring, while a task runs** — forty points on a circle a third wider than the sphere,
        //: tilted the other way and turning on its own clock, mixed into the depth sort so its far half
        //: passes behind the sphere.
        if (orbiting) {
            const round = still ? 0 : seconds * 0.35;
            for (let i = 0; i < 40; i += 1) {
                const a = (i / 40) * Math.PI * 2 + round;
                const x = Math.cos(a) * 1.38;
                const z = Math.sin(a) * 1.38;
                const y = x * 0.32;
                const ry = y * tiltC + z * tiltS;
                const rz2 = -y * tiltS + z * tiltC;
                const near = 1 / (2.6 - rz2);
                on.push({ sx: mid.x + x * span * near * 2.6, sy: mid.y + ry * span * near * 2.6, depth: rz2, ring: true });
            }
        }
        //: Back to front, so the far half is drawn under the near half rather than over it.
        on.sort((a, b) => a.depth - b.depth);
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        for (const q of on) {
            //: Depth does the work colour usually would: the far side is small and faint, the near side
            //: solid. One ink, so it sits in the page's palette rather than beside it.
            const front = (q.depth + 1) / 2;
            const alpha = (q.ring ? 0.12 + front * 0.35 : 0.1 + front * 0.5) * visible;
            const dot = (q.ring ? 0.8 + front * 1.2 : 0.6 + front * 1.5) * dpr;
            paint.globalAlpha = alpha;
            paint.fillStyle = doing === "gone" ? "#6b6862" : "#2c4a6e";
            paint.beginPath();
            paint.arc(q.sx, q.sy, dot, 0, Math.PI * 2);
            paint.fill();
        }
        paint.globalAlpha = 1;
    }
    //: The frame budget. `fps: 0` means *take what the browser offers*; anything else waits.
    let next = 0;
    function tick(now) {
        if (!running) {
            return;
        }
        const how = HOW[doing];
        //: A turning ring at ten frames a second stutters, so it asks for twenty-four — still well
        //: short of the sixty a busy sphere takes.
        const fps = orbiting && how.fps > 0 ? 24 : how.fps;
        const quiet = fps > 0 && jolts.length === 0;
        if (!quiet || now >= next) {
            next = now + (quiet ? 1000 / fps : 0);
            draw(now);
        }
        requestAnimationFrame(tick);
    }
    last = performance.now();
    requestAnimationFrame(tick);
    return {
        doing(what) {
            doing = what;
        },
        jolt(what) {
            const j = JOLTS[what];
            jolts.push({ push: j.push, ring: j.ring, at: performance.now() / 1000 });
            //: A cap, because a burst of events should look like one big event and not queue up into a
            //: minute of ringing.
            if (jolts.length > 6) {
                jolts.shift();
            }
        },
        presence(of) {
            visible = Math.max(0, Math.min(1, of));
        },
        working(on) {
            orbiting = on;
        },
        stop() {
            running = false;
        },
    };
}
