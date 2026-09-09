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

#: A small text model. It is asked a text question and never for a coordinate.
DEFAULT_MODEL = "qwen2.5:7b"

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
                 watches, every: float = 0.7, build=None, warm: bool = True) -> None:
        super().__init__(daemon=True, name="argmind-agent")
        self.inbox: queue.Queue = queue.Queue()
        self.outbox: queue.Queue = queue.Queue()
        self.model_name = model
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
                what, payload = self.inbox.get(timeout=self.every)
            except queue.Empty:
                self._idle()
                continue
            if what == "quit":
                break
            handler = {"ask": self._ask, "run": self._run, "drop": self._drop}.get(what)
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
                Local(self.model_name))

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
        if self.session is not None:
            try:
                self.session.model.release()
            except Exception:                                             # noqa: BLE001, S110
                pass

    # ---- the work -------------------------------------------------------------------------------

    def _idle(self) -> None:
        title = self._title_of()
        if title != self._title:
            self._title = title
            self._say("attending", title)
        if not self.watch_for:
            return
        for trigger in self.session.poll():
            self._say("noticed", trigger.text)
            self._think(trigger)

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
        elif not intent.effects:
            self._say("say", "…")
        for effect in intent.effects:
            self._propose(effect)
        self._say("stat", {"widgets": seen, "took": took,
                           "unprompted": not trigger.asked})
        self._say("busy", False)

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
           dry: bool = False, notice: bool = False,
           warm: bool = True) -> int:                                     # pragma: no cover - GUI
    """Start the agent thread and the window, and block until the window closes."""
    agent = Agent(model=model, limit=limit, rate=rate, dry=dry, warm=warm,
                  watches=[WindowChanged()] if notice else [])
    agent.start()
    Window(agent, dry=dry).loop()
    return 0


if __name__ == "__main__":
    sys.exit(launch())
