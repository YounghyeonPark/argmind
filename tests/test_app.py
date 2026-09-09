# SPDX-License-Identifier: Apache-2.0
"""The app's logic, tested without a window.

**A window cannot be asserted about, and the interesting part is not the window.** What matters is
the protocol between the agent thread and whatever draws it: what it proposes, what it refuses, and
above all **that nothing is ever performed except through a confirmation**. Every one of those is a
message, and messages are testable.

So `Agent` takes a `build` that returns `(eyes, hands, model)` and these tests pass fakes: no
display, no Ollama, no COM, no Windows. The `Window` class is left untested on purpose rather than
by neglect — it draws what arrives and sends what is clicked, and a test of it would be a test of
tkinter.
"""

from __future__ import annotations

import queue
import time

from argmind.app import Agent
from argmind.core import Appeared
from tests.test_core import FakeEyes, FakeHands, FakeModel, SCREEN, says, w


class Watched(FakeEyes):
    """`FakeEyes` plus the two optional extras an interface asks for."""

    def __init__(self, *screens, title="Their Window", at=("", "")) -> None:
        super().__init__(*screens)
        self._title = title
        self._at = at

    def attending_title(self):
        return self._title

    def at(self, x, y):
        return self._at


def agent(model, *screens, dry=False, watches=(), eyes=None, **kw):
    made = eyes or Watched(*(screens or (SCREEN,)), **kw)
    hands = FakeHands(live=not dry)
    #: Warming is off here and tested on its own. It costs two model calls, which would eat the
    #: canned answers these tests depend on — and a fixture quietly consuming the thing under test
    #: is how a suite starts measuring itself.
    a = Agent(model="fake", limit=200, rate=1000.0, dry=dry, watches=list(watches), every=0.02,
              warm=False, build=lambda: (made, hands, model))
    a.eyes_for_test, a.hands_for_test = made, hands
    return a


def drain(a, want, timeout=4.0):
    """Run the agent until `want` messages of interest have arrived, or give up.

    Started and stopped inside the test rather than left running, because a daemon thread that
    outlives its assertions is how a suite starts passing for reasons nobody chose.
    """
    got = []
    a.start()
    try:
        end = time.time() + timeout
        while time.time() < end and len(got) < want:
            try:
                got.append(a.outbox.get(timeout=0.05))
            except queue.Empty:
                continue
    finally:
        a.quit()
        a.join(timeout=2.0)
    return got


def kinds(messages):
    return [k for k, _p in messages]


def only(messages, kind):
    return [p for k, p in messages if k == kind]


# ---- the thing the whole app exists to guarantee ------------------------------------------------

def test_asking_PROPOSES_and_does_not_act():
    """**The measurement made this the shape.** Choosing is right 18 times in 28, so eight wrong
    answers in thirty-seven goals is a bad click most working hours and a click has no undo.

    Nothing may reach the hands without a person pressing something.
    """
    a = agent(FakeModel(says("닫을게요", {"click": 0})))
    a.ask("이 창 닫아줘")
    messages = drain(a, 6)
    assert only(messages, "say") == ["닫을게요"]
    assert len(only(messages, "propose")) == 1
    assert a.hands_for_test.done == [], "it acted without being confirmed"


def test_CONFIRMING_is_what_acts():
    """The other half, so the test above cannot pass by an agent that never works."""
    a = agent(FakeModel(says("닫을게요", {"click": 0})))
    a.ask("닫아줘")
    a.start()
    proposal = None
    end = time.time() + 4.0
    while time.time() < end and proposal is None:
        try:
            kind, payload = a.outbox.get(timeout=0.05)
        except queue.Empty:
            continue
        if kind == "propose":
            proposal = payload
    assert proposal is not None, "it never proposed anything"
    a.run_proposal(proposal.token)
    ran = []
    end = time.time() + 4.0
    while time.time() < end and not ran:
        try:
            kind, payload = a.outbox.get(timeout=0.05)
        except queue.Empty:
            continue
        if kind == "ran":
            ran.append(payload)
    a.quit()
    a.join(timeout=2.0)
    assert ran and ran[0][1].startswith("클릭")
    assert [d[0] for d in a.hands_for_test.done] == ["click"]


def test_saying_NO_leaves_the_hands_alone():
    a = agent(FakeModel(says("닫을게요", {"click": 0})))
    a.ask("닫아줘")
    messages = drain(a, 6)
    proposals = only(messages, "propose")
    assert proposals
    a.drop(proposals[0].token)
    assert a.hands_for_test.done == []


def test_a_target_that_went_away_is_reported_rather_than_clicked():
    """The 13-in-14 measurement reaches the interface too: between proposing and confirming, the
    screen moved. The row has to say so instead of clicking where the thing used to be."""
    gone = [x for x in SCREEN if x.name != "닫기"]
    a = agent(FakeModel(says("닫을게요", {"click": 0})), SCREEN, gone)
    a.ask("닫아줘")
    messages = drain(a, 6)
    assert only(messages, "propose") == [], "it offered a button for something already gone"
    assert only(messages, "gone"), "it went quiet instead of saying the thing had gone"


# ---- what it says about itself ------------------------------------------------------------------

def test_it_reports_WHICH_WINDOW_it_is_answering_about():
    """**Not decoration.** Once you click into the app, the foreground window is the app, and the
    answer comes from a remembered window instead. A substitution nobody can see is one nobody can
    correct, so the title is a message the interface must always have."""
    a = agent(FakeModel(says("네")), eyes=Watched(SCREEN, title="Their Editor"))
    messages = drain(a, 3)
    assert "Their Editor" in only(messages, "attending")


def test_a_hint_appears_only_when_the_point_disagrees():
    """`at()` is a second opinion, not a gate: a widget's own centre resolves back to itself 73% of
    the time and a caption button resolving to "application" is a correct pixel with a disagreeing
    hit test. So a mismatch is shown to the person and never acted on."""
    quiet = agent(FakeModel(says("ok", {"click": 0})), eyes=Watched(SCREEN, at=("닫기", "button")))
    quiet.ask("닫아줘")
    assert only(drain(quiet, 6), "propose")[0].hint == ""

    loud = agent(FakeModel(says("ok", {"click": 0})), eyes=Watched(SCREEN, at=("문서", "text")))
    loud.ask("닫아줘")
    proposal = only(drain(loud, 6), "propose")[0]
    assert "문서" in proposal.hint and proposal.effect.target == "닫기", (
        "the hint must describe the disagreement without changing what would be clicked")


def test_a_broken_environment_says_so_instead_of_going_quiet():
    """A thread that dies during setup leaves a window that looks like it is thinking forever."""
    def explode():
        raise RuntimeError("no UI Automation here")

    a = Agent(model="fake", limit=10, rate=1000.0, dry=True, watches=[], every=0.02,
              warm=False, build=explode)
    messages = drain(a, 2)
    assert only(messages, "ready") == [False]
    assert any("no UI Automation here" in note for note in only(messages, "note"))
    assert kinds(messages).index("ready") < kinds(messages).index("note"), (
        "the reason must arrive after `ready`, which clears the transcript, or it is wiped")


def test_one_bad_turn_does_not_take_the_thread_down():
    """The agent must survive a failure the way a conversation does — by saying something."""
    class Angry(FakeModel):
        def ask(self, prompt, images=None):
            raise ValueError("the model fell over")

    a = agent(Angry())
    a.ask("안녕")
    messages = drain(a, 4)
    assert any("the model fell over" in note for note in only(messages, "note"))
    assert only(messages, "busy")[-1] is False, "it left the send button disabled forever"


def test_WARMING_happens_before_the_person_types_and_costs_two_calls():
    """**Measured, and the reason warming exists**: the first question costs 4,715 ms and the second
    5,940 ms — 4.4 s and 5.2 s of that is time to first token, the weights loading. The third
    onwards are 331, 329 and 608 ms. **One call does not finish warming**, which is why this is two
    and not one, and why a test asserts the count rather than trusting the comment.
    """
    #: Three answers, because the fake hands them out in order and the first two are the warm-up's.
    #: A real `Local` has no queue; this is the fixture's shape, not the app's.
    model = FakeModel(says("워밍1"), says("워밍2"), says("첫 대답"))
    made, hands = Watched(SCREEN), FakeHands(live=False)
    a = Agent(model="fake", limit=200, rate=1000.0, dry=True, watches=[], every=0.02,
              warm=True, build=lambda: (made, hands, model))
    a.ask("안녕")
    messages = drain(a, 8)
    assert only(messages, "warmed"), "it never warmed the model"
    assert len(model.prompts) == 3, (
        f"{len(model.prompts)} calls; warming is two and the question is the third")
    #: And the warming must not have eaten the answer the person asked for.
    assert only(messages, "say") == ["첫 대답"]


# ---- noticing ------------------------------------------------------------------------------------

def test_something_NOTICED_becomes_a_proposal_and_still_needs_the_button():
    """**The permission argument, held under test.** The session is built with
    `may_act_unprompted=True`, which on its own would be alarming — it is correct only because
    `perform` is unreachable in this app except through a confirmation. If a refactor ever makes an
    unprompted effect run by itself, this is what fails.
    """
    built = SCREEN + [w("Build succeeded")]
    a = agent(FakeModel(says("빌드가 끝났어요", {"click": 4})), SCREEN, built, built,
              watches=[Appeared("build", gap=0.0)])
    messages = drain(a, 8, timeout=5.0)
    assert only(messages, "noticed"), "the watch never fired"
    assert a.hands_for_test.done == [], "it acted on its own judgement without being asked"


def test_the_session_permits_unprompted_action_ONLY_because_a_person_gates_it():
    """Stated as an assertion so the reasoning cannot rot silently: the flag is on, and the reason
    it is safe is that exactly one place calls `perform`."""
    import ast
    import io
    import pathlib

    source = pathlib.Path(__file__).resolve().parents[1] / "argmind" / "app.py"
    tree = ast.parse(io.open(source, encoding="utf-8").read())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "perform"]
    assert len(calls) == 1, (
        f"{len(calls)} places call perform(); the app's whole safety argument is that there is one, "
        "reached only by a button")
