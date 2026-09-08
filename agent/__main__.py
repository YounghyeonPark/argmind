# SPDX-License-Identifier: Apache-2.0
"""`python -m agent` — run the loop, or just measure whether it would help you.

**The default does nothing to your machine.** It watches the screen, keys each frame, and reports the
hit rate the cache *would* have had. That is the question worth answering before anything else: this
whole design rests on your screen recurring, and 20.4 % of pairs recurring on *one* capture of *one*
machine is not a promise about yours.

    python -m agent --measure 60              # watch for 60 s, report the hit rate. No key, no spend.
    python -m agent --goal "..." --steps 20   # run with a brain, still dry-run
    python -m agent --goal "..." --live       # and now it moves the mouse

`--live` is a separate word from `--goal` on purpose.
"""

from __future__ import annotations

import argparse
import sys
import time

from .act import Actuator
from .brain import ScriptedBrain
from .cache import ActionCache
from .loop import AgentLoop
from .senses import ScreenSense
from .situation import SituationSense


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agent", description=__doc__)
    p.add_argument("--measure", type=float, metavar="SECONDS",
                   help="watch the screen for SECONDS and report the hit rate the cache would have "
                        "had. Touches nothing, needs no key, costs nothing. This is the default "
                        "because the design rests on your screen repeating and that is your number, "
                        "not the one measured here")
    p.add_argument("--goal", type=str, help="what the agent should try to do")
    p.add_argument("--steps", type=int, default=10, help="how many cycles to run (default 10)")
    p.add_argument("--live", action="store_true",
                   help="actually move the mouse and type. Off by default: this drives whatever "
                        "window has focus and there is no undo")
    p.add_argument("--rate", type=float, default=4.0, metavar="N",
                   help="most actions per second (default 4). A brake, not a throttle")
    p.add_argument("--threshold", type=float, default=0.995, metavar="C",
                   help="cosine a situation must reach to count as one already seen (default "
                        "0.995). Lower hits more often and clicks in the wrong place more often")
    p.add_argument("--capacity", type=int, default=512, metavar="N",
                   help="how many situations to remember (default 512)")
    p.add_argument("--cache", type=str, metavar="PATH",
                   help="load this cache at start and save it at the end, so repetition pays off "
                        "across runs rather than only within one")
    p.add_argument("--hz", type=float, default=2.0, metavar="F",
                   help="observations per second while measuring (default 2)")
    return p


def measure(args) -> int:
    """Watch, key every frame, and report what a cache would have done. Acts on nothing."""
    screen, sense = ScreenSense(), SituationSense()
    cache = ActionCache(sense.n_features, capacity=args.capacity, threshold=args.threshold)
    print(f"  watching {screen.size[0]}x{screen.size[1]} for {args.measure:.0f} s at {args.hz:g} Hz."
          f"  Nothing is clicked or typed.", flush=True)

    #: **Two numbers, because one of them flatters.** A hit on the frame half a second ago is a
    #: still screen, not a repeated task -- the first version of this printed 94.6 % for an idle
    #: terminal, which is true and means nothing. A hit against an entry stored more than `gap`
    #: seconds ago is the one that corresponds to coming back to something.
    gap = 10.0
    stored_at: dict[int, float] = {}
    t0, n, recurrent = time.perf_counter(), 0, 0
    while time.perf_counter() - t0 < args.measure:
        now = time.perf_counter()
        situation = sense.describe(screen.read(), t=now)
        hit = cache.lookup(situation.key)
        if hit is None:
            stored_at[cache.store(situation.key, {"do": "wait", "seconds": 0.0})] = now
        elif now - stored_at.get(hit.slot, now) > gap:
            recurrent += 1
        n += 1
        time.sleep(max(0.0, 1.0 / args.hz - 0.04))

    s = cache.stats()
    print(f"\n  {n} observations, {s['entries']} distinct situations remembered")
    print(f"  any hit         {s['hit_rate']:>7.1%}   includes the screen simply not changing")
    print(f"  RECURRENT hit   {recurrent / max(n, 1):>7.1%}   matched something first seen over "
          f"{gap:.0f} s ago")
    print("\n  The second number is the one that matters: the share of frames where the agent would"
          "\n  have come back to a situation it had already decided about. The first counts a still"
          "\n  screen as a saving, which it is not. Both are upper bounds -- a cached decision also"
          "\n  has to still be the RIGHT one, and neither mode can tell you that.")
    if args.cache:
        cache.save(args.cache)
        print(f"  saved to {args.cache}.keys.npy / .values.json")
    return 0


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.measure or not args.goal:
        if not args.measure:
            args.measure = 30.0
            print("  no --goal given, so measuring instead. --help shows the rest.\n")
        return measure(args)

    sense = SituationSense()
    cache = ActionCache(sense.n_features, capacity=args.capacity, threshold=args.threshold)
    if args.cache:
        try:
            cache.load(args.cache)
            print(f"  loaded {cache.stats()['entries']} remembered situations from {args.cache}")
        except FileNotFoundError:
            print(f"  {args.cache} not there yet; starting empty")

    try:
        from .brain import ClaudeBrain

        brain = ClaudeBrain()
        print("  brain: ClaudeBrain (UNTESTED — this path has never been executed here)")
    except Exception as exc:                                   # noqa: BLE001 - reported, not hidden
        brain = ScriptedBrain(default=[])
        print(f"  brain: none ({type(exc).__name__}: {exc}).\n"
              "  Running with a brain that decides nothing, so this measures the cache and the "
              "loop and not the model.")

    loop = AgentLoop(screen=ScreenSense(), situation=sense, cache=cache, brain=brain,
                     actuator=Actuator(live=args.live, max_per_second=args.rate))
    print(f"  {'LIVE — this will move the mouse' if args.live else 'dry run — nothing is touched'}")
    try:
        loop.run(args.goal, args.steps)
    except KeyboardInterrupt:
        print("\n  stopped")
    for key, value in loop.report().items():
        print(f"    {key:<24} {value}")
    if args.cache:
        cache.save(args.cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())
