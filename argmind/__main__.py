# SPDX-License-Identifier: Apache-2.0
"""`python -m argmind` — look at what the environment sees, talk to it, or let it notice.

    python -m argmind app               the window. Leave it running while you work
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

#: **qwen3:8b with its thinking switched off.** Measured over the same 37 goals, three repeats:
#:
#:     model                right/28  correct/37   a call
#:     qwen2.5:7b              18        26        0.45 s
#:     qwen3:8b, no thinking   23        30        0.45 s
#:     qwen2.5:14b             25        32        2.39 s
#:     qwen3:8b, thinking      27        34        3.75 s
#:
#: The first row to the second is a free upgrade — same latency, four more correct decisions — and
#: it is the only change measured on this project that cost nothing. The last row is the most
#: accurate thing available here, 27 of 28 against a criterion of nine in ten, and at 3.75 s it is
#: not a conversation. `--think on` is that trade, one flag away.
#:
#: **`think=False` is sent explicitly**, because leaving it out costs `qwen3:8b` 1,112 characters of
#: reasoning to pick a name out of a list it can already see. Models that have never heard of the
#: field accept it: `qwen2.5:7b` answered in 0.11 s with it set.
DEFAULT_MODEL = "qwen3:8b"
#: Reasoning off. `None` here would omit the field and let a thinking model think.
DEFAULT_THINK = False


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m argmind", description=__doc__)
    p.add_argument("mode", nargs="?", default="look",
                   choices=("app", "look", "talk", "say", "watch"))
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
    p.add_argument("--dry", action="store_true",
                   help="for `app`: propose and confirm as usual, but touch nothing")
    p.add_argument("--think", action="store_true",
                   help="let a reasoning model reason. 27 of 28 goals right instead of 23, and "
                        "3.75 s a reply instead of 0.45 — accuracy against conversation")
    p.add_argument("--cold", action="store_true",
                   help="for `app`: do not load the model until the first question. Saves 5-6 GB "
                        "until you use it and costs about 10 s when you do")
    p.add_argument("--notice", action="store_true",
                   help="for `app`: also speak when the window you are in changes. Off by "
                        "default — it costs a model call every time it fires")
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
    return Session(eyes=eyes, model=Local(args.model, think=DEFAULT_THINK), hands=hands,
                   limit=args.limit,
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


def app(args) -> int:
    """The window. Everything else here is a CLI over the same session; this is the thing you leave
    running."""
    from .app import launch                                              # noqa: PLC0415

    return launch(model=args.model, limit=args.limit, rate=args.rate,
                  dry=args.dry, notice=args.notice, warm=not args.cold,
                  think=True if args.think else DEFAULT_THINK)


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
    return {"app": app, "look": look, "talk": talk, "say": say,
            "watch": watch}[args.mode](args)


def _released(session, code: int) -> int:
    """Unload the model on the way out. 5-6 GB held by a process nobody is talking to is not free,
    and this project left exactly that on a machine for fifty-four minutes on a day it ran out of
    memory."""
    session.model.release()
    return code


if __name__ == "__main__":
    sys.exit(main())
