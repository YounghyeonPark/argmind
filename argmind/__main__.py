# SPDX-License-Identifier: Apache-2.0
"""`python -m argmind` — look at what the environment sees, talk to it, or let it notice.

    python -m argmind look              what is on screen right now, and how long reading it took
    python -m argmind talk              a conversation. It answers; it clicks only with --live
    python -m argmind say "닫아줘"       one exchange and exit
    python -m argmind watch --for build  it speaks when something happens, unasked

`look` is the default because it costs nothing, touches nothing, needs no model, and answers the
question that decides whether any of this works on *your* screen: **if the thing you want is not in
that list, no model will find it.** That is what `Session.blind` counts once the conversation starts.

`--live` is a separate word from anything you say, on purpose. **`--allow-unprompted` is a second
one**, because they are different risks: `--live` means "do what I asked", and without the second
flag anything the agent decides *by itself* comes out as a sentence rather than a click. `watch`
runs without either and is worth running that way first — it costs a model call every time it
notices something, and how often that is on your desktop is not a thing to guess.
"""

from __future__ import annotations

import argparse
import sys
import time

from .core import Appeared, Session, WindowChanged
from .core.llm import Local

#: **Windows hands Python a cp1252 stdout and widget names are not cp1252.** The first run of this
#: CLI died on a Korean window title — which would be a silly failure for a design whose whole point
#: is that the names come back in the interface's own language.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):                      # a pipe that will not be reconfigured
        pass

#: A small text model. It is asked a text question and never for a coordinate, which is why the fast
#: one is the right one here: measured 67-85 ms to first token against 2.7 s for the vision model.
DEFAULT_MODEL = "qwen2.5:7b"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m argmind", description=__doc__)
    p.add_argument("mode", nargs="?", default="look", choices=("look", "talk", "say", "watch"))
    p.add_argument("text", nargs="?", default=None, help="what to say, for `say`")
    p.add_argument("--live", action="store_true",
                   help="actually click. Off by default: this drives whatever window has focus and "
                        "there is no undo")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"local model (default {DEFAULT_MODEL})")
    p.add_argument("--limit", type=int, default=200, metavar="N",
                   help="most widgets to list for the model (default 200)")
    p.add_argument("--rate", type=float, default=4.0, metavar="N",
                   help="most actions per second (default 4). A brake, not a throttle")
    p.add_argument("--allow-unprompted", action="store_true",
                   help="let it act on what it noticed. Without this, anything it decides by itself "
                        "is said rather than done")
    p.add_argument("--for", dest="needle", default=None, metavar="TEXT",
                   help="for `watch`: notice things whose name contains TEXT. Without it, notice "
                        "the foreground window changing")
    p.add_argument("--every", type=float, default=1.0, metavar="S",
                   help="for `watch`: how often to look (default 1 s). Each watch has its own "
                        "floor on how often it may start a thought")
    return p


def _environment(args):
    from .env.windows import WindowsEyes, WindowsHands          # noqa: PLC0415

    return WindowsEyes(), WindowsHands(live=args.live, max_per_second=args.rate)


def look(args) -> int:
    eyes, _ = _environment(args)
    widgets, took = eyes.timed_foreground()
    if not widgets:
        print("  nothing named in the foreground window. That is a real answer, not a failure: "
              "canvas content, video, games and custom-drawn interfaces expose no widgets.")
        return 0
    print(f"  {widgets[0].window[:96]!r}")
    print(f"  {len(widgets)} named widgets, read in {took * 1000:.0f} ms\n")
    print(f"  {'#':>4}  {'role':<16} {'position':>18}  name")
    for i, w in enumerate(widgets[:args.limit]):
        x, y, ww, hh = w.rect
        print(f"  {i:>4}  {w.role[:16]:<16} {f'{x},{y} {ww}x{hh}':>18}  {w.name[:60]}")
    if len(widgets) > args.limit:
        print(f"  ... and {len(widgets) - args.limit} more")
    return 0


def _session(args, watches=None) -> Session:
    eyes, hands = _environment(args)
    return Session(eyes=eyes, model=Local(args.model), hands=hands, limit=args.limit,
                   watches=watches, may_act_unprompted=args.allow_unprompted)


def _show(ex) -> None:
    if ex.said:
        print(f"  > {ex.said}")
    for done in ex.did:
        if done.target is not None:
            x, y = done.target.centre
            where = f"{done.target.role} {done.target.name[:40]!r} at ({x}, {y})"
        else:
            where = repr(done.effect.text[:40])
        confirmed = "" if done.confirmed is None else f", confirmed={done.confirmed}"
        print(f"    [{done.effect.kind} {where}{confirmed}]")
    #: Printed, not swallowed. It decided to do something and was not allowed to, and hiding that
    #: would make the permission look like the agent having no opinion.
    for effect in ex.proposed:
        print(f"    [would {effect.kind} {(effect.target or effect.text)[:40]!r} — "
              f"unasked, so it only said so. --allow-unprompted lets it]")
    for effect in ex.refused:
        print(f"    [{effect.kind} {(effect.target or effect.text)[:40]!r} refused — gone by the "
              f"time it was to be touched]")
    print(f"    saw {ex.n_widgets:>3} widgets | look {ex.look_s * 1000:>5.0f} ms | "
          f"think {ex.think_s * 1000:>5.0f} ms | total {ex.total_s * 1000:>5.0f} ms")


def talk(args) -> int:
    session = _session(args)
    print(f"  {args.model} | "
          f"{'LIVE — it will click' if args.live else 'dry run — it will not touch anything'}")
    print("  say something, or an empty line to leave.\n")
    while True:
        try:
            heard = input("  you: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not heard:
            break
        _show(session.exchange(heard))
        print()
    for key, value in session.report().items():
        print(f"    {key:<20} {value}")
    return _released(session, 0)


def say(args) -> int:
    if not args.text:
        print('  say what: python -m argmind say "close this window"')
        return 2
    session = _session(args)
    _show(session.exchange(args.text))
    return _released(session, 0)


def watch(args) -> int:
    """Let it notice. **The half that can act while nobody is looking**, which is why it prints its
    poll rate and its permission before it starts."""
    watches = [Appeared(args.needle)] if args.needle else [WindowChanged()]
    session = _session(args, watches)
    for_what = f"{args.needle!r} appearing" if args.needle else "the window changing"
    permission = ("it may ACT on what it notices" if args.allow_unprompted
                  else "it will only SAY what it notices")
    print(f"  {args.model} | watching for {for_what} every {args.every:.1f} s")
    print(f"  {permission} | ctrl-c to stop")
    print()
    try:
        while True:
            for trigger in session.poll():
                print(f"  · {trigger.text}")
                _show(session.think(trigger))
                print()
            time.sleep(args.every)
    except KeyboardInterrupt:
        print()
    for key, value in session.report().items():
        print(f"    {key:<20} {value}")
    return _released(session, 0)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    return {"look": look, "talk": talk, "say": say, "watch": watch}[args.mode](args)


def _released(session, code: int) -> int:
    """Unload the model on the way out. 5-6 GB held by a process nobody is talking to is not free,
    and this project left exactly that on a machine for fifty-four minutes on a day it ran out of
    memory."""
    session.model.release()
    return code


if __name__ == "__main__":
    sys.exit(main())
