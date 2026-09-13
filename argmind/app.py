# SPDX-License-Identifier: Apache-2.0
"""The PC app — a small window you talk to while you work.

    python -m argmind app

**Three things decide the whole shape of this file, and each of them was measured.**

**The agent's window is on the screen the agent reads.** The moment you click into the input box,
`GetForegroundWindow` is *this* app, and an agent that answers about its own text box is useless.
So the eyes are built with `ignore_pid` and answer about the last window that was in front and was
somebody else's — and the header says which one, always, because a substitution nobody can see is a
substitution nobody can correct.

**Choosing is right 18 times in 28, so a person confirms every action.** That is not timidity about
a good number; it is what the number means. Eight wrong answers in thirty-seven goals is a bad click
most working hours, and a click has no undo. Each proposed effect appears as a row with what it will
do and a button. **`Session.perform` is called from exactly one place in this file — the button.**

**The world moves under every decision.** A widget that was listed is gone 13 times in 14 over the
second a call takes, so the effect carries a name and the name is looked up again when the button is
pressed, not when the model chose. If the thing has gone by then the row says so instead of clicking
where it used to be.

## Why the model lives on a thread

Every call into UI Automation is COM, and COM objects belong to the thread that made them. So the
whole `Session` — eyes, hands, model — lives in `Agent`, and the window never touches it: two queues
carry text one way and events the other. That also keeps a 0.4-1 s model call off the event loop,
which is the difference between an app and a frozen rectangle.

## The check before the first click

`WindowsHands.pointing_is_exact()` runs once at startup. UI Automation reports physical pixels and a
DPI-unaware process can have its cursor calls virtualised; where those disagree every click is
offset and lands on whatever happens to be there. Measured 40 of 40 exact on this display, which is
a property of one desk. If it fails here, the app says so and refuses to click at all.
"""

from __future__ import annotations

import collections
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import font as tkfont

from .core import Said, Session, WindowChanged
from .core.intent import Intent
from .core.llm import Local

__all__ = ["Agent", "Window", "run"]

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

#: The talk key. Held, not toggled — push-to-talk is `voice.py`'s design decision and the reason
#: there is no `MIN_GAP` here, no echo cancellation and no way to mistake a phone call in the room
#: for a turn. **F9 because the entry box must keep every key a person types**, and a Tk binding
#: only fires when this window has focus — so you have to be looking at argmind to talk to it. That
#: is a real limit of doing it without a system-wide hook, and the status bar says so.
TALK_KEY = "F9"
#: How long the inbox waits while a transcript is coming. Transcription is 234 ms median, p90 300,
#: max 645 — measured on synthesised speech, so a floor.
QUICK = 0.05
#: How long to keep waiting after the key comes up before saying nothing was heard. Extended while
#: the listener still has audio in hand.
AWAIT_SPEECH = 3.0

INK = {
    "bg": "#fbfbfa", "panel": "#f2f1ee", "line": "#dedcd6", "text": "#23211d",
    "you": "#2f5d8a", "agent": "#23211d", "act": "#8a5a1a", "noticed": "#6b6862",
    "note": "#8a2f2f", "dim": "#8c8880",
}


class Proposal:
    """One effect waiting for a person to press a button.

    Holds the effect rather than a resolved widget, because the widget it would touch is looked up
    again at the moment the button is pressed. Between proposing and confirming, the screen moved.
    """

    __slots__ = ("token", "effect", "label", "hint")

    def __init__(self, token: int, effect, label: str, hint: str = "") -> None:
        self.token = token
        self.effect = effect
        self.label = label
        #: What UI Automation says is at that point right now. A **second opinion, not a gate** —
        #: a widget's own centre resolves back to itself 73% of the time, and a caption button
        #: resolving to "application" is UIA's hit test disagreeing with a pixel that is correct.
        self.hint = hint


class Agent(threading.Thread):
    """Owns the session and every COM object in the process. Speaks only through queues."""

    def __init__(self, *, model: str, limit: int, rate: float, dry: bool,
                 watches, every: float = 0.7, build=None, warm: bool = True,
                 think=DEFAULT_THINK, build_voice=None,
                 may_speak_unprompted: bool = False) -> None:
        super().__init__(daemon=True, name="argmind-agent")
        self.inbox: queue.Queue = queue.Queue()
        self.outbox: queue.Queue = queue.Queue()
        self.model_name = model
        self.think = think
        self.limit = limit
        self.rate = rate
        self.dry = dry
        self.watch_for = watches
        self.every = every
        #: **Measured, and the reason this exists**: the first question costs 4,715 ms and the
        #: second 5,940 ms, of which 4.4 s and 5.2 s are time to first token — the weights loading.
        #: The third onwards are 331, 329 and 608 ms. Two calls to warm, not one. Paying that before
        #: the person types costs 5-6 GB held from launch, which is a real cost on a machine that
        #: ran out of memory a week ago, so `--cold` turns it off and `release()` still runs on exit.
        self.warm = bool(warm)
        self.session: Session | None = None
        self.alive = True
        self._pending: dict[int, Proposal] = {}
        self._next = 1
        #: `None` rather than `""`, so the **first** answer is sent even when it is empty. An
        #: agent that has not yet seen anybody else's window has to say so.
        self._title: str | None = None
        self._can_point = False
        #: How the environment is made, injectable so **the whole message protocol can be tested
        #: without a display, a model or COM** — which is the only way the app's logic gets a test
        #: at all, since a window cannot be asserted about.
        self.build = build or self._windows
        self._title_of = lambda: ""
        self._what_is_at = lambda _x, _y: ("", "")

        # ---- voice, and it is off until somebody switches it on ----------------------------------
        #: **Not inverted, unlike `may_act_unprompted`.** The app inverts that one because there is
        #: a stronger gate downstream — a person pressing 실행. There is no downstream gate on a
        #: sentence: you cannot un-hear one.
        self.may_speak_unprompted = bool(may_speak_unprompted)
        #: Built lazily and separately from eyes/hands/model, because the listener holds 1,544 MiB
        #: of weights the moment it is first pressed and nobody should pay that for launching a
        #: window. Injectable for the same reason `build` is: `test_app.py` runs with no devices.
        self.build_voice = build_voice or self._voices
        self.listener = None
        self.speaker = None
        #: Hold durations, for the measurement that does not exist yet. Whisper transcribes silence
        #: as words **120 of 120**, so a stray press is a turn; rejecting it needs a minimum hold or
        #: a no-speech threshold, and both need numbers only real recorded speech can set. Recording
        #: what the presses actually were is how that session produces a number instead of a guess.
        self.holds: collections.deque = collections.deque(maxlen=128)
        self._held_from: float | None = None
        #: While a transcript is on its way, the inbox wait drops from `every` to `QUICK`. A spoken
        #: turn should not spend 0.7 s of the person's patience sitting in a poll interval.
        self._awaiting = 0.0

    # ---- what the window calls ------------------------------------------------------------------

    def ask(self, text: str) -> None:
        self.inbox.put(("ask", text))

    def run_proposal(self, token: int) -> None:
        self.inbox.put(("run", token))

    def drop(self, token: int) -> None:
        self.inbox.put(("drop", token))

    def quit(self) -> None:
        self.alive = False
        self.inbox.put(("quit", None))

    def use_voice(self, on: bool) -> None:
        """Open or close the microphone and the voice. Queued — it is slow and nothing waits on it.

        There is no command-line flag for this and there must not be: `tests/test_voice.py` forbids
        the word in `__main__.py`, because the CLI performs actions with no confirm button and a
        microphone there would let a mis-transcription click something. The switch lives in the
        window for the same reason the 실행 button does.
        """
        self.inbox.put(("voice", bool(on)))

    def talk(self, down: bool) -> None:
        """The talk key. **Called on the GUI thread, deliberately not through the inbox.**

        Both calls it makes are cheap signals into worker threads that these objects own — `stop()`
        measured at 0.069 ms, `hold()` is a queue put — and both are useless if they wait. The inbox
        is only read between turns, so a queued `hold(True)` pressed while the model is thinking
        would start capturing after the answer, losing the first word of an interruption, and a
        queued `stop()` would arrive after the sentence being interrupted had finished.
        """
        speaker, listener = self.speaker, self.listener
        #: **This is barge-in, and it is the whole of it.** The agent's speech is a courtesy and the
        #: person's is the input, so there is no arbitration and no queue: cut mid-word, no resume.
        if down and speaker is not None:
            speaker.stop()
        if listener is None:
            return
        if down:
            self._held_from = time.perf_counter()
        else:
            started = self._held_from
            self._held_from = None
            if started is not None:
                self.holds.append(round(time.perf_counter() - started, 3))
            self._awaiting = time.monotonic() + AWAIT_SPEECH
        listener.hold(down)

    # ---- the thread -----------------------------------------------------------------------------

    def run(self) -> None:                                                # pragma: no cover - GUI
        try:
            self._setup()
        except Exception as exc:                                          # noqa: BLE001
            #: `ready` clears the transcript, so the reason has to arrive after it, not before.
            self._say("ready", False)
            self._say("note", f"{type(exc).__name__}: {exc}")
            return
        while self.alive:
            try:
                wait = QUICK if time.monotonic() < self._awaiting else self.every
                what, payload = self.inbox.get(timeout=wait)
            except queue.Empty:
                self._idle()
                continue
            if what == "quit":
                break
            handler = {"ask": self._ask, "run": self._run, "drop": self._drop,
                       "voice": self._voice}.get(what)
            if handler is None:
                continue
            try:
                handler(payload)
            except Exception as exc:                                      # noqa: BLE001
                #: A failure here must not take the thread down: the app would go quiet and look
                #: like it was thinking forever.
                self._say("note", f"{type(exc).__name__}: {exc}")
                self._say("busy", False)
        self._teardown()

    def _windows(self):
        from .env.windows import WindowsEyes, WindowsHands                # noqa: PLC0415

        return (WindowsEyes(ignore_pid=os.getpid()),
                WindowsHands(live=not self.dry, max_per_second=self.rate),
                Local(self.model_name, think=self.think))

    def _voices(self):
        """The Windows pair. Imported here, so a machine with no torch still opens the window."""
        from .env.sapi import SapiSpeaker                                 # noqa: PLC0415
        from .env.whisper import WhisperListener                          # noqa: PLC0415

        #: `pace_limit` stays unset. `whisper.py` names the audio that would set it and says why
        #: that audio does not exist yet; a guessed cut-off silently drops real turns.
        return WhisperListener(), SapiSpeaker()

    def _voice(self, on: bool) -> None:
        if bool(on) == (self.listener is not None or self.speaker is not None):
            return
        if not on:
            self._shut_voice()
            self._say("voice", (False, ""))
            return
        try:
            self.listener, self.speaker = self.build_voice()
        except Exception as exc:                                          # noqa: BLE001
            #: A missing microphone, no Korean voice, no CUDA — all of them arrive here, and none of
            #: them may take the window down. The person keeps a working agent they can type at.
            self._shut_voice()
            self._say("voice", (False, f"{type(exc).__name__}: {exc}"))
            return
        korean = getattr(self.speaker, "has_korean", lambda: True)()
        self._say("voice", (True, "" if korean else
                            "한국어 음성이 없어 한국어 대답은 소리가 나지 않습니다"))

    def _shut_voice(self) -> None:
        for part in (self.listener, self.speaker):
            try:
                if part is not None:
                    part.release()
            except Exception:                                             # noqa: BLE001, S110
                pass
        self.listener = self.speaker = None
        self._held_from = None
        self._awaiting = 0.0

    def _setup(self) -> None:
        eyes, hands, model = self.build()
        #: **Three things an interface wants that the core does not**, resolved once here rather
        #: than guarded at every call site. `contract.py` says why they are optional: a core that
        #: only answers questions needs none of them, and an environment that cannot provide them
        #: is still a complete environment.
        self._title_of = getattr(eyes, "attending_title", lambda: "")
        self._what_is_at = getattr(eyes, "at", lambda _x, _y: ("", ""))
        #: **Seed the remembered window before ours exists.** The agent thread starts before the
        #: window is built, so right now the foreground is whatever the person was using; a moment
        #: later it is us, and `attending()` would have nothing to fall back to. Racy by nature and
        #: strictly better than starting blind.
        getattr(eyes, "attending", lambda: None)()
        self._can_point = getattr(hands, "pointing_is_exact", lambda: True)()
        if not self._can_point and not self.dry:
            #: Refusing is the honest response. A click offset by a scaling factor lands on whatever
            #: is there, and "whatever is there" is not something to find out by trying.
            hands.live = False
            self._say("note", "the cursor does not go where it is sent on this display — "
                              "UI Automation and SetCursorPos disagree, so clicking is switched "
                              "off. Everything else works.")
        self.session = Session(eyes=eyes, model=model, hands=hands,
                               limit=self.limit, watches=self.watch_for,
                               #: **The human is the permission.** Nothing in this file calls
                               #: `perform` except the confirm button, so an unprompted effect is
                               #: gated by a person rather than by this flag — which is stronger
                               #: than the flag, not weaker. Leaving it off would silently turn
                               #: every noticed action into speech and the button into a lie.
                               may_act_unprompted=True)
        self._say("ready", True)
        if self.warm:
            self._warm()

    def _warm(self) -> None:
        """Two throwaway calls, because one does not finish warming."""
        self._say("busy", True)
        t0 = time.perf_counter()
        try:
            for _ in range(2):
                self.session.model.ask("Reply with the single word: ready")
        except Exception as exc:                                          # noqa: BLE001
            self._say("note", f"모델을 열지 못했습니다 — {type(exc).__name__}: {exc}")
            self._say("busy", False)
            return
        self._say("warmed", time.perf_counter() - t0)
        self._say("busy", False)

    def _teardown(self) -> None:
        self._shut_voice()
        if self.session is not None:
            try:
                self.session.model.release()
            except Exception:                                             # noqa: BLE001, S110
                pass

    # ---- the work -------------------------------------------------------------------------------

    def _idle(self) -> None:
        if self._listen():
            return
        title = self._title_of()
        if title != self._title:
            self._title = title
            self._say("attending", title)
        if not self.watch_for:
            return
        for trigger in self.session.poll():
            self._say("noticed", trigger.text)
            self._think(trigger)

    def _listen(self) -> bool:
        """Take a finished utterance if there is one. Returns whether it handled the tick.

        A spoken turn is `Said(text)` — **the same trigger a typed one makes**. `Trigger.asked` is
        what decides whether an effect is downgraded to speech, so a separate kind for speech would
        make every spoken request unprompted and invert the permission model by a spelling.
        """
        if self.listener is None:
            return False
        heard = self.listener.heard()
        if heard is not None:
            self._awaiting = 0.0
            #: **Verbatim, as the person's line.** If what it thought it heard is not written down,
            #: a misunderstanding leaves no evidence in the room and there is nothing for
            #: *"아니 그거 말고"* to point at.
            self._say("heard", heard.text)
            self._think(Said(heard.text))
            return True
        if not self._awaiting:
            return False
        if getattr(self.listener, "pending", lambda: 0)():
            #: Still transcribing. Do not give up on the person mid-sentence.
            self._awaiting = time.monotonic() + AWAIT_SPEECH
            return False
        if time.monotonic() >= self._awaiting:
            self._awaiting = 0.0
            self._say("unheard", None)
        return False

    def _ask(self, text: str) -> None:
        self._think(Said(text))

    def _think(self, trigger) -> None:
        self._say("busy", True)
        t0 = time.perf_counter()
        self.session.observe()
        seen = len(self.session.seen)
        intent = self.session.decide(trigger)
        took = time.perf_counter() - t0

        self.session.history.append(("them" if trigger.asked else "noticed", trigger.text))
        if intent.say:
            self.session.history.append(("you", intent.say))
            self._say("say", intent.say)
            self._speak(intent.say, asked=trigger.asked)
        elif not intent.effects:
            self._say("say", "…")
        for effect in intent.effects:
            self._propose(effect)
        self._say("stat", {"widgets": seen, "took": took,
                           "unprompted": not trigger.asked})
        self._say("busy", False)

    def _speak(self, text: str, *, asked: bool) -> None:
        """Say it aloud, if there is a voice and if it was spoken to.

        **A median reply is ~3.1 s of Korean speech and a p90 is 6.5 s** — longer than the model
        call, the transcription and the synthesis put together — which is why `say()` returns at
        once and why the talk key can cut it off.
        """
        if self.speaker is None or not text.strip():
            return
        if not asked and not self.may_speak_unprompted:
            #: A `Watch` firing makes noise in the room on its own. Off by default and **not**
            #: inverted the way `may_act_unprompted` is: a proposal waits behind a button, a
            #: sentence does not wait behind anything.
            return
        try:
            self.speaker.say(text)
        except Exception as exc:                                          # noqa: BLE001
            self._say("note", f"음성 출력 실패 — {type(exc).__name__}: {exc}")

    def _propose(self, effect) -> None:
        token, self._next = self._next, self._next + 1
        if effect.kind == "click":
            found = self.session.find(effect.target, effect.role)
            if found is None:
                self._say("gone", (token, effect.target))
                return
            name, role = self._what_is_at(*found.centre)
            hint = ""
            if name != found.name and name not in found.name and found.name not in name:
                hint = f"그 지점에는 {role or '?'} {name or '(이름 없음)'}"
            label = f"클릭  {found.role} “{found.name}”"
        elif effect.kind == "type":
            label, hint = f"입력  “{effect.text}”", ""
        else:
            label, hint = f"키  {effect.text}", ""
        proposal = Proposal(token, effect, label, hint)
        self._pending[token] = proposal
        #: **In the conversation, not only on the screen.** An offer the person then argues with —
        #: "not that one" — needs to be something the next turn can see.
        self.session.remember_doing(offered=[effect])
        self._say("propose", proposal)

    def _run(self, token: int) -> None:
        proposal = self._pending.pop(token, None)
        if proposal is None:
            return
        #: **Step out of the way first.** Measured 2026-09-09: a synthetic click goes to whatever
        #: window is topmost at that pixel, and this app is always-on-top in the corner of the
        #: screen — so a widget underneath it was unreachable, and the click landed on argmind. The
        #: window lowers itself, the click happens, and it comes back. A handshake rather than a
        #: sleep, because the window is on the other thread and guessing how long it takes to
        #: repaint is how this kind of fix works everywhere except where it matters.
        self._stand_aside(True)
        try:
            #: **The only call to `perform` in this file.** Everything above proposes; this runs.
            did, _proposed, refused = self.session.perform(
                Intent("", [proposal.effect], proposal.effect.because))
        finally:
            self._stand_aside(False)
        self.session.remember_doing(did=did, refused=refused)
        if refused:
            self._say("gone", (token, proposal.effect.target or proposal.effect.text))
        for done in did:
            self._say("ran", (token, proposal.label, done.confirmed))

    def _stand_aside(self, away: bool) -> None:
        ready = threading.Event()
        self._say("aside", (away, ready))
        #: A second is generous for lowering a window and far short of hanging the agent if the
        #: interface has gone: the click then happens with the window still in place, which is the
        #: behaviour before this existed rather than a new failure.
        ready.wait(1.0)

    def _drop(self, token: int) -> None:
        proposal = self._pending.pop(token, None)
        if proposal is not None:
            #: Pressing "no" is a thing the person said, and until now it was said to the screen
            #: and not to the agent — the next turn had no idea it had just been turned down.
            self.session.remember_doing(turned_down=[proposal.effect])
            self._say("dropped", proposal.token)

    def _say(self, kind: str, payload=None) -> None:
        self.outbox.put((kind, payload))


class Window:                                                             # pragma: no cover - GUI
    """The window. Touches no COM, holds no session, and never blocks on the model."""

    def __init__(self, agent: Agent, *, dry: bool) -> None:
        self.agent = agent
        self.dry = dry
        self.rows: dict[int, tk.Frame] = {}
        self.busy = False
        self.since = 0.0
        #: Off until somebody switches it on. The listener holds 1,544 MiB the moment it is first
        #: pressed and the speaker makes noise in a room the app cannot see, so neither is something
        #: to start by surprise.
        self.voice_on = False
        #: Windows repeats KeyPress while a key is held; KeyRelease comes only on the real release.
        self._held = False

        self.root = tk.Tk()
        self.root.title("argmind")
        self.root.configure(bg=INK["bg"])
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._place(430, 560)

        family = "Malgun Gothic" if "Malgun Gothic" in tkfont.families() else "TkDefaultFont"
        self.body = tkfont.Font(family=family, size=10)
        self.small = tkfont.Font(family=family, size=8)

        self._header()
        self._transcript()
        self.actions = tk.Frame(self.root, bg=INK["bg"])
        self.actions.grid(row=2, column=0, sticky="ew", padx=8)
        self._input()
        self._status()
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        #: **Bound on the toplevel, so it fires wherever the focus is inside this window** — the
        #: entry box included, which is why the key is F9 and not a character. It does not fire when
        #: another application has focus: a system-wide hook is out of scope and sits badly with an
        #: app whose whole discipline is not stealing focus. So: you look at argmind to talk to it.
        self.root.bind(f"<KeyPress-{TALK_KEY}>", self._talk_down)
        self.root.bind(f"<KeyRelease-{TALK_KEY}>", self._talk_up)

        self.write("연결 중…", "note")
        self.root.after(60, self._drain)

    # ---- construction ---------------------------------------------------------------------------

    def _place(self, w: int, h: int) -> None:
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        #: 110 clears a default Windows taskbar plus the window frame, which is ~40 px taller
        #: than the content and had the app sitting over the clock.
        self.root.geometry(f"{w}x{h}+{sw - w - 24}+{sh - h - 110}")
        self.root.minsize(360, 320)

    def _header(self) -> None:
        bar = tk.Frame(self.root, bg=INK["panel"])
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_columnconfigure(1, weight=1)
        tk.Label(bar, text="argmind", bg=INK["panel"], fg=INK["text"],
                 font=(self.body.actual("family"), 10, "bold")).grid(row=0, column=0, padx=(10, 6),
                                                                    pady=6)
        #: **Always visible, never optional.** This is the window the agent will answer about, and
        #: it is not the window with focus once you click in here.
        self.attending = tk.Label(bar, text="…", bg=INK["panel"], fg=INK["dim"], anchor="w")
        self.attending.grid(row=0, column=1, sticky="ew", padx=(0, 10))

    def _transcript(self) -> None:
        frame = tk.Frame(self.root, bg=INK["bg"])
        frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(8, 4))
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        self.text = tk.Text(frame, wrap="word", bd=0, highlightthickness=0, bg=INK["bg"],
                            fg=INK["text"], font=self.body, spacing1=2, spacing3=4, padx=2)
        self.text.grid(row=0, column=0, sticky="nsew")
        bar = tk.Scrollbar(frame, command=self.text.yview, width=10)
        bar.grid(row=0, column=1, sticky="ns")
        self.text.configure(yscrollcommand=bar.set, state="disabled")
        for tag, colour in (("you", INK["you"]), ("agent", INK["agent"]), ("act", INK["act"]),
                            ("noticed", INK["noticed"]), ("note", INK["note"])):
            self.text.tag_configure(tag, foreground=colour)
        self.text.tag_configure("noticed", font=(self.body.actual("family"), 9, "italic"))
        self.text.tag_configure("act", font=(self.body.actual("family"), 9))

    def _input(self) -> None:
        frame = tk.Frame(self.root, bg=INK["bg"])
        frame.grid(row=3, column=0, sticky="ew", padx=8, pady=(6, 4))
        frame.grid_columnconfigure(0, weight=1)
        self.entry = tk.Entry(frame, font=self.body, bd=1, relief="solid",
                              highlightthickness=0, bg="white", fg=INK["text"])
        self.entry.grid(row=0, column=0, sticky="ew", ipady=5)
        self.entry.bind("<Return>", lambda _e: self.send())
        self.send_button = tk.Button(frame, text="보내기", command=self.send, bd=0, padx=12,
                                     bg=INK["panel"], fg=INK["text"], font=self.small,
                                     activebackground=INK["line"])
        self.send_button.grid(row=0, column=1, sticky="ns", padx=(6, 0))
        self.voice_button = tk.Button(frame, text="음성 켜기", command=self.toggle_voice, bd=0,
                                      padx=10, bg=INK["panel"], fg=INK["dim"], font=self.small,
                                      activebackground=INK["line"])
        self.voice_button.grid(row=0, column=2, sticky="ns", padx=(6, 0))
        self.entry.focus_set()

    def _status(self) -> None:
        self.status = tk.Label(self.root, text="", bg=INK["bg"], fg=INK["dim"], anchor="w",
                               font=self.small)
        self.status.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 8))

    # ---- talking --------------------------------------------------------------------------------

    def write(self, line: str, tag: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", line + "\n", tag)
        self.text.configure(state="disabled")
        self.text.see("end")

    def toggle_voice(self) -> None:
        self.voice_button.configure(text="여는 중…", state="disabled")
        self.agent.use_voice(not self.voice_on)

    def _talk_down(self, _e=None):
        if not self.voice_on or self._held:
            return "break"
        self._held = True
        self.agent.talk(True)
        self.status.configure(text=f"듣는 중 — {TALK_KEY} 를 누르고 계세요")
        return "break"

    def _talk_up(self, _e=None):
        if not self._held:
            return "break"
        self._held = False
        self.agent.talk(False)
        self.status.configure(text="옮겨 적는 중…")
        return "break"

    def send(self) -> None:
        text = self.entry.get().strip()
        if not text or self.busy:
            return
        self.entry.delete(0, "end")
        self.write(f"나: {text}", "you")
        self.agent.ask(text)

    def propose(self, proposal: Proposal) -> None:
        """A row with what would happen and a button. **The only path to an action.**"""
        row = tk.Frame(self.actions, bg=INK["panel"], highlightbackground=INK["line"],
                       highlightthickness=1)
        row.pack(fill="x", pady=2)
        row.grid_columnconfigure(0, weight=1)
        text = proposal.label + (f"\n{proposal.hint}" if proposal.hint else "")
        tk.Label(row, text=text, bg=INK["panel"], fg=INK["act"], anchor="w", justify="left",
                 font=self.small).grid(row=0, column=0, sticky="ew", padx=8, pady=5)
        tk.Button(row, text="실행", bd=0, padx=10, font=self.small, bg="#e6ddc9", fg=INK["act"],
                  command=lambda: self.agent.run_proposal(proposal.token)).grid(row=0, column=1,
                                                                               padx=(0, 4))
        tk.Button(row, text="아니오", bd=0, padx=8, font=self.small, bg=INK["panel"],
                  fg=INK["dim"],
                  command=lambda: self.agent.drop(proposal.token)).grid(row=0, column=2,
                                                                       padx=(0, 6))
        self.rows[proposal.token] = row

    def _tick(self) -> None:
        if not self.busy:
            return
        self.status.configure(text=f"생각 중… {time.monotonic() - self.since:.1f}s")
        self.root.after(100, self._tick)

    def clear_row(self, token: int) -> None:
        row = self.rows.pop(token, None)
        if row is not None:
            row.destroy()

    # ---- the pump -------------------------------------------------------------------------------

    def _drain(self) -> None:
        while True:
            try:
                kind, payload = self.agent.outbox.get_nowait()
            except queue.Empty:
                break
            self._event(kind, payload)
        self.root.after(60, self._drain)

    def _event(self, kind: str, payload) -> None:
        if kind == "attending":
            self.attending.configure(
                text=payload[:52] or "아직 다른 창을 못 봤습니다 — 한 번 전환해 주세요")
        elif kind == "say":
            self.write(payload, "agent")
        elif kind == "noticed":
            self.write(f"· {payload}", "noticed")
        elif kind == "heard":
            #: Verbatim, and marked as heard rather than typed, so a wrong transcript is visible as
            #: a wrong transcript instead of as the agent misunderstanding a sentence you never said.
            self.write(f"나(음성): {payload}", "you")
            self.status.configure(text="")
        elif kind == "unheard":
            #: Whisper answers silence with words 120 times out of 120, so the listener's guards
            #: throw some transcripts away. Saying nothing here would look like the app hanging.
            self.write("  (아무 말도 알아듣지 못했습니다)", "note")
            self.status.configure(text="")
        elif kind == "voice":
            on, why = payload
            self.voice_on = bool(on)
            self.voice_button.configure(text="음성 끄기" if on else "음성 켜기", state="normal",
                                        fg=INK["text"] if on else INK["dim"])
            if on:
                self.write(f"음성이 켜졌습니다. {TALK_KEY} 를 누르고 말하세요 — 이 창에 "
                           f"포커스가 있어야 하고, 누르면 하던 말을 멈춥니다.", "noticed")
            if why:
                self.write(f"  {why}", "note")
        elif kind == "propose":
            self.propose(payload)
        elif kind == "ran":
            #: Only this row goes. "Click the box, type the name, press enter" is one intention and
            #: three rows, and running the first must not silently drop the other two.
            token, label, confirmed = payload
            mark = {True: "화면이 바뀜", False: "화면이 그대로 — 빗나갔을 수 있음",
                    None: "드라이런"}[confirmed]
            self.write(f"  ✓ {label}  ({mark})", "act")
            self.clear_row(token)
        elif kind == "dropped":
            self.clear_row(payload)
        elif kind == "gone":
            token, what = payload
            self.write(f"  ✗ “{what}” 은(는) 누를 시점에 이미 사라졌습니다", "act")
            self.clear_row(token)
        elif kind == "busy":
            self.busy = bool(payload)
            self.send_button.configure(state="disabled" if self.busy else "normal",
                                       text="…" if self.busy else "보내기")
            if self.busy:
                #: **A four-second wait with no moving part reads as a hang.** The first question
                #: after the app opens pays the model's cold load, measured at 5.8-8.4 s, and the
                #: app cannot make that shorter — it can only stop it looking broken.
                self.since = time.monotonic()
                self._tick()
        elif kind == "stat":
            self.status.configure(
                text=f"위젯 {payload['widgets']}개 · 생각 {payload['took'] * 1000:.0f} ms"
                     + ("  · 스스로 알아챈 것" if payload["unprompted"] else ""))
        elif kind == "ready":
            self._ready(bool(payload))
        elif kind == "warmed":
            self.status.configure(text=f"모델 준비됨 ({payload:.1f}s) · 이제 첫 질문도 빠릅니다")
        elif kind == "aside":
            away, ready = payload
            try:
                self.root.attributes("-topmost", not away)
                if away:
                    self.root.lower()
                else:
                    self.root.lift()
            finally:
                #: Set even if the window refuses to move, so the agent is never left waiting on a
                #: cosmetic step.
                ready.set()
        elif kind == "note":
            self.write(payload, "note")

    def _ready(self, ok: bool) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        if not ok:
            self.write("시작하지 못했습니다. Ollama가 켜져 있는지, comtypes가 설치돼 있는지 "
                       "확인해 주세요.", "note")
            return
        self.write("무엇이든 물어보세요. 위에 적힌 창에 대해 답합니다.", "noticed")
        self.write("행동은 제안만 하고, 실행은 당신이 누릅니다"
                   + (" (지금은 드라이런 — 아무것도 건드리지 않습니다)." if self.dry else "."),
                   "noticed")

    def close(self) -> None:
        self.agent.quit()
        self.agent.join(timeout=3.0)
        self.root.destroy()

    def loop(self) -> None:
        self.root.mainloop()


def launch(*, model: str = DEFAULT_MODEL, limit: int = 200, rate: float = 4.0,
           dry: bool = False, notice: bool = False, warm: bool = True,
           think=DEFAULT_THINK) -> int:                                   # pragma: no cover - GUI
    """Start the agent thread and the window, and block until the window closes."""
    agent = Agent(model=model, limit=limit, rate=rate, dry=dry, warm=warm, think=think,
                  watches=[WindowChanged()] if notice else [])
    agent.start()
    Window(agent, dry=dry).loop()
    return 0


if __name__ == "__main__":
    sys.exit(launch())
