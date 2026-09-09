# SPDX-License-Identifier: Apache-2.0
"""The core, tested without naming a platform — which is the point of the core existing.

**If any test here needed Windows, the split would be imaginary.** Everything runs against a fake
environment and a fake model: no display, no OS automation, no local model, no network. That is not
convenience — a suite that needs a live model measures the model's mood on the day it ran, and a
suite that needs a display cannot run where it matters most.

The rule the whole design rests on is checked against the source, by walking the AST, because a rule
that lives in a docstring is a rule that gets quietly undone:

> **The deterministic side says WHERE. The probabilistic side says WHAT.
> A model never emits a coordinate.**
"""

from __future__ import annotations

import ast
import io
import pathlib

import pytest

from argmind.core import Appeared, Perceived, Session, Widget, WindowChanged, converse, describe
from argmind.core.llm import Reply

CORE = pathlib.Path(__file__).resolve().parents[1] / "argmind" / "core"


# ---- fakes ------------------------------------------------------------------------------------

class FakeEyes:
    """A surface that can show something different on each read, because it does.

    **A thought reads up to three times** — once to ask the question, once to find the chosen thing
    again immediately before clicking it, and once to see whether the click changed anything — and
    the screen is measured to differ between those reads **thirteen times in fourteen** over the
    second a model call takes. A fake that returns the same list every time cannot express the case
    the design exists for.

    `screens` is one list per read; the last one repeats once they run out.
    """

    def __init__(self, *screens) -> None:
        self.screens = [list(s) for s in screens] or [[]]
        self.reads = 0

    def foreground(self):
        seen = self.screens[min(self.reads, len(self.screens) - 1)]
        self.reads += 1
        return list(seen)

    def timed_foreground(self):
        return self.foreground(), 0.001


class FakeHands:
    def __init__(self, live=False) -> None:
        self.live = live
        self.done = []

    def click(self, widget, button="left"):
        self.done.append(("click", widget.name, widget.centre))
        return {"do": "click", "on": widget.name}

    def type_text(self, text):
        self.done.append(("type", text))
        return {"do": "type"}

    def press(self, key):
        self.done.append(("key", key))
        return {"do": "key"}


class FakeModel:
    """Answers with canned strings in order, and records every prompt it was given."""

    def __init__(self, *texts) -> None:
        self.texts = list(texts)
        self.prompts = []

    def ask(self, prompt, images=None):
        self.prompts.append(prompt)
        assert images is None, "the core must never hand a model an image"
        text = self.texts.pop(0) if self.texts else '{"say": "", "do": []}'
        return Reply(text, 0.001, 0.002, 100, 10, "stop")


def w(name, role="button", rect=(10, 20, 40, 30), window="Win"):
    return Widget(name, role, rect, window)


SCREEN = [w("닫기"), w("Share"), w("Mute (m)"), w("File", role="menu item", rect=(35, 0, 36, 35))]


def says(text, *actions):
    """The model's answer, in the shape the prompt asks for."""
    import json
    return json.dumps({"say": text, "do": list(actions)}, ensure_ascii=False)


def session(model, *screens, live=False, **kw):
    """`session(model)` shows `SCREEN` on every read; extra positional lists are the later reads."""
    return Session(eyes=FakeEyes(*(screens or (SCREEN,))), model=model, hands=FakeHands(live), **kw)


# ---- the rule -------------------------------------------------------------------------------

def test_the_core_NAMES_NO_PLATFORM():
    """The split is real only if the core can be written without a platform in it. A false positive
    here is easier to argue with than a slow drift back into Windows-specific code."""
    banned = ("uiautomation", "comtypes", "win32", "ctypes", "windll", "sendinput",
              "adb", "uiautomator", "hwnd")
    offenders = []
    for path in sorted(CORE.glob("*.py")):
        for n, line in enumerate(io.open(path, encoding="utf-8"), 1):
            low = line.lower()
            #: The contract's docstring COMPARES the platforms to justify the interface, which is
            #: description and not dependence. Only code lines are checked.
            if line.lstrip().startswith(("#", '"', "'", "|")) or "*" in line:
                continue
            for word in banned:
                if word in low:
                    offenders.append(f"{path.name}:{n}: {line.strip()[:70]}")
    assert not offenders, f"the core has grown a platform dependency: {offenders}"


def test_NO_core_module_asks_a_model_for_a_coordinate():
    """Measured: asking a vision model for a pixel cost 24-82 s and on a dense screen returned
    nothing, while the environment returns the same thing exactly in 0.2 s. A prompt that asks for
    x and y is that trade being silently undone."""
    offenders = []
    for path in sorted(CORE.glob("*.py")):
        tree = ast.parse(io.open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                low = node.value.lower()
                if '"x"' in low and '"y"' in low:
                    offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, (
        f"a coordinate is being requested from a model at {offenders}; the environment supplies it")


def test_the_core_never_hands_a_model_an_IMAGE():
    """The other half of the same trade, and `FakeModel` asserts it on every call — this test exists
    so the assertion is exercised rather than merely present."""
    s = session(FakeModel(says("ok", {"click": 1})))
    s.exchange("share it")
    assert s.model.prompts, "the model was never called, so the assertion above proved nothing"


# ---- conversation ----------------------------------------------------------------------------

def test_it_can_ANSWER_without_touching_anything():
    """The thing the previous skeleton could not do. A command executor always acts; something you
    talk to has to be able to just reply."""
    s = session(FakeModel(says("세 개 열려 있어요")))
    ex = s.exchange("뭐가 열려 있지?")
    assert ex.said == "세 개 열려 있어요"
    assert ex.did == [] and s.hands.done == []
    assert s.report()["spoke_only"] == 1


def test_it_can_answer_AND_act_in_one_call():
    """One round trip, because latency is the budget: two calls would double the only expensive
    thing in the loop."""
    s = session(FakeModel(says("닫을게요", {"click": 0})))
    ex = s.exchange("이 창 닫아줘")
    assert ex.said == "닫을게요" and ex.did[0].target.name == "닫기"
    assert s.hands.done == [("click", "닫기", (30, 35))]
    assert len(s.model.prompts) == 1, "the reply and the action must come from one call"


def test_ONE_THOUGHT_can_produce_SEVERAL_effects():
    """"Click the box, type the name, press enter" is one intention and three effects. The previous
    shape allowed exactly one click per utterance, which made that three round trips and three
    chances for the screen to move underneath."""
    s = session(FakeModel(says("검색할게요", {"click": 1}, {"type": "argmind"}, {"key": "enter"})))
    ex = s.exchange("share에 argmind 치고 엔터")
    assert [d.effect.kind for d in ex.did] == ["click", "type", "key"]
    assert s.hands.done == [("click", "Share", (30, 35)), ("type", "argmind"), ("key", "enter")]


@pytest.mark.parametrize("text", [says("x", {"click": 99}), says("x", {"click": -1}),
                                  says("x", {"click": "Share"}), says("x", {"type": ""}),
                                  'not json at all', ''])
def test_an_UNUSABLE_choice_is_refused_rather_than_repaired(text):
    """An out-of-range number is a hallucination, not a near miss. Clamping it turns a model that
    invented an index into a click on whatever sits at the edge of the list."""
    ex = session(FakeModel(text)).exchange("do it")
    assert ex.did == []


@pytest.mark.parametrize("text, effects", [
    ('{"say": "ok", "do": [{"click": 0}]}', 1),
    ('{"say": "ok", "do": [{"click": 0}, {"key": "enter"}]}', 2),
    ('sure, here you go: {"say": "ok", "do": [{"click": 0}]} hope that helps', 1),
    ('{"say": "press } to close", "do": [{"click": 0}]}', 1),
])
def test_a_NESTED_reply_is_read_whole(text, effects):
    r"""**The seventh pitfall, and it was found by this restructure rather than by a measurement.**
    The reply extractor was `re.search(r"\{.*?\}")` — correct for exactly as long as the answer
    contained no nested object, and wrong the moment it grew a list of things to do. It matched up to
    the *inner* brace, failed to parse, and returned nothing. Every action silently disappeared while
    the model was answering perfectly.

    That is the same shape as the other six: a plausible zero produced by the harness. It is pinned
    here because nothing about the failure was visible — the agent simply never did anything.
    """
    ex = session(FakeModel(text)).exchange("do it")
    assert len(ex.did) == effects


def test_NOTHING_MATCHED_is_counted_because_it_decides_what_to_build_next():
    """`blind` is the number that would justify anything slower and cleverer than this. Estimating
    it later is how the previous project spent a month on a mechanism nothing needed."""
    s = session(FakeModel(says("그건 화면에 없어요")))
    s.exchange("피자 주문해줘")
    assert s.blind == 1 and s.report()["nothing_matched"] == 1


def test_a_surface_with_NO_widgets_is_told_apart_from_a_failure():
    """Canvas, video, games and custom-drawn UI expose nothing. The prompt has to say that rather
    than present an empty list as if the screen were blank — and it must not count as blind, since
    there was nothing to match against."""
    s = session(FakeModel(says("여긴 아무것도 안 보여요")), [])
    s.exchange("클릭해줘")
    assert "no widgets" in s.model.prompts[0]
    assert s.blind == 0, "an empty surface is not the model failing to find something"


def test_the_conversation_is_CARRIED_and_bounded():
    """A reply that ignores what was just said is not a conversation; a prompt that carries all of it
    pushes the widget list out of a context that has to stay small to stay fast."""
    s = session(FakeModel(*[says(f"r{i}") for i in range(6)]))
    s.remember = 2
    for i in range(6):
        s.exchange(f"line {i}")
    last = s.model.prompts[-1]
    assert "Recently:" in last and "line 5" not in last.split("They said:")[0]
    assert last.count("them:") + last.count("you:") == 2, "history is bounded by `remember`"


def test_the_widget_list_keeps_the_INTERFACE_S_OWN_WORDS():
    """Measured: the model matched an English request to the Korean widget "닫기". That is the case a
    keyword matcher fails, and the reason a model is here at all — so the names are passed through
    untouched."""
    listed = describe(SCREEN)
    assert '0: button "닫기"' in listed
    assert listed.splitlines()[3] == '3: menu item "File"'


# ---- noticing, and the permission that goes with it -------------------------------------------

def test_the_model_is_TOLD_when_nobody_asked():
    """Being asked is a reason to act; noticing something is usually a reason to mention it. The
    prompt has to carry which one this is, or the two get the same answer."""
    s = session(FakeModel(says("빌드 끝났네요")))
    s.think(Perceived("'Build succeeded' appeared"))
    prompt = s.model.prompts[0]
    assert "Nobody asked you anything" in prompt and "You noticed:" in prompt

    s2 = session(FakeModel(says("네")))
    s2.exchange("안녕")
    assert "Nobody asked you anything" not in s2.model.prompts[0]


def test_an_UNPROMPTED_action_is_downgraded_to_SPEECH_and_not_refused():
    """**The brake this restructure needed.** The three built for "you asked, it does one thing" — a
    `live` flag, a rate limit, a guard before each action — none of them covers "it decided,
    repeatedly, while you were reading something else".

    Refusing silently would be worse than useless: the agent noticed something and said nothing. So
    the effect becomes a sentence, and the person answers it.
    """
    s = session(FakeModel(says("빌드 끝났어요. 열까요?", {"click": 1})), live=True)
    ex = s.think(Perceived("'Build succeeded' appeared"))
    assert s.hands.done == [], "it acted on its own judgement without being allowed to"
    assert ex.did == [] and len(ex.proposed) == 1
    assert ex.said, "it was stopped from acting AND said nothing, which is the useless case"
    assert s.report()["proposed_only"] == 1


def test_STANDING_PERMISSION_is_what_lets_an_unprompted_action_run():
    """The permission is a property of the session, not of the utterance — which is the whole point:
    it is granted once, deliberately, by the person, and not inferred from a single "yes"."""
    s = session(FakeModel(says("열게요", {"click": 1})), live=True, may_act_unprompted=True)
    ex = s.think(Perceived("'Build succeeded' appeared"))
    assert [d.effect.kind for d in ex.did] == ["click"] and ex.proposed == []
    assert s.hands.done[0][1] == "Share"


def test_being_ASKED_is_never_downgraded():
    """The other side of the same switch, so the test cannot pass by refusing everything."""
    s = session(FakeModel(says("닫을게요", {"click": 0})), live=True)
    ex = s.exchange("닫아줘")
    assert len(ex.did) == 1 and ex.proposed == []


# ---- watches ----------------------------------------------------------------------------------

def test_the_FIRST_poll_reports_nothing_because_there_is_no_BEFORE():
    """An empty "before" would make the entire screen newly appeared, and the agent would open its
    mouth about every widget on the desktop the moment it started."""
    s = session(FakeModel(), watches=[WindowChanged(gap=0.0)])
    assert s.poll() == []
    assert s.seen, "it reported nothing because it never looked, which is a different bug"


def test_a_watch_turns_a_CHANGE_into_a_trigger():
    other = [w("Send", window="Mail")]
    s = session(FakeModel(), SCREEN, other, watches=[WindowChanged(gap=0.0)])
    assert s.poll() == []
    fired = s.poll()
    assert len(fired) == 1 and not fired[0].asked and "Mail" in fired[0].text


def test_a_watch_will_NOT_fire_again_inside_its_floor():
    """**Measured, not tuned**: something on screen changes in 13 seconds out of 14 and a model call
    costs 0.3-2 s. Without a floor the agent is permanently mid-thought and deaf, so the floor is
    part of the design and belongs under test like anything else load-bearing."""
    built = SCREEN + [w("Build succeeded")]
    s = session(FakeModel(), SCREEN, built, SCREEN, built, watches=[Appeared("build")])
    assert s.poll() == []                                  # no before
    assert len(s.poll()) == 1, "it did not notice the thing appearing at all"
    assert s.poll() == []                                  # it went away; Appeared says nothing
    assert s.poll() == [], "it fired twice within its floor and would saturate the model"


def test_a_watch_notices_something_GOING_AWAY_too():
    """Appearing and vanishing are different questions, and the second is the one a person actually
    asks: "tell me when that download dialog is gone"."""
    from argmind.core import Vanished                                    # noqa: PLC0415

    gone = [x for x in SCREEN if x.name != "Mute (m)"]
    s = session(FakeModel(), SCREEN, gone, watches=[Vanished("mute", gap=0.0)])
    assert s.poll() == []
    fired = s.poll()
    assert len(fired) == 1 and "Mute (m)" in fired[0].text


def test_a_watch_reports_WHAT_happened_and_not_merely_THAT_it_did():
    """The trigger text goes into the prompt verbatim. "something changed" is not something a model
    can answer usefully."""
    built = SCREEN + [w("Build succeeded")]
    s = session(FakeModel(), SCREEN, built, watches=[Appeared("build", gap=0.0)])
    s.poll()
    assert "Build succeeded" in s.poll()[0].text


# ---- the environment contract ----------------------------------------------------------------

def test_a_widget_is_the_ONE_PLACE_a_name_becomes_a_coordinate():
    assert w("x", rect=(0, 0, 100, 50)).centre == (50, 25)
    assert w("x", rect=(-2560, 0, 46, 40)).centre == (-2537, 20), (
        "a negative origin is a monitor to the left, not an error to clamp")


def test_confirmation_is_NONE_in_a_dry_run_and_not_False():
    """Nothing was clicked, so nothing should have changed. `False` would read as a failure that
    never happened."""
    ex = session(FakeModel(says("ok", {"click": 0}))).exchange("close it")
    assert ex.did[0].confirmed is None


def test_a_target_that_VANISHED_before_the_click_is_refused():
    """**The measurement that forces the whole shape**: over the second a model call takes, thirteen
    times in fourteen something that was on screen when the question was asked is gone when the
    answer arrives. The remembered rectangle is then where the thing *used to be*, and whatever
    occupies it now was never chosen.

    So the choice is re-resolved by name against a fresh reading, and a miss is a refusal.
    """
    gone = [x for x in SCREEN if x.name != "닫기"]
    s = session(FakeModel(says("닫을게요", {"click": 0})), SCREEN, gone, live=True)
    ex = s.exchange("close it")
    assert ex.did == [], "it clicked a rectangle whose widget had already gone"
    assert len(ex.refused) == 1 and s.hands.done == []
    assert s.report()["gone_before_click"] == 1


def test_a_name_shared_by_TWO_ROLES_resolves_to_the_one_that_was_chosen():
    """**Measured on a real window, and it cost a wrong diagnosis before it was found.**

    Character Map lays out a `text` label reading `Font :` immediately before a `combo box` also
    called `Font :`. The model chose the combo box, the effect stored the name, the name was
    resolved again at the moment of acting, and it came back the **label** — because the label is
    first in the list. The click landed on a piece of text and did nothing, which is
    indistinguishable from a click that missed, and was written up as a different defect entirely.

    A name is the right key against a screen that moves; a name **alone** is not a key at all.
    """
    screen = [w("Font :", role="text", rect=(34, 107, 53, 20)),
              w("Font :", role="combo box", rect=(86, 106, 321, 30))]
    s = session(FakeModel(says("열게요", {"click": 1})), screen, screen, screen, live=True)
    ex = s.exchange("글꼴 목록 열어줘")
    assert ex.did, "it refused a target that was there"
    assert ex.did[0].target.role == "combo box", (
        "it resolved the label rather than the thing the model chose")
    assert s.hands.done[0][2] == (246, 121), "it clicked the label's centre"


def test_the_role_survives_the_chosen_thing_MOVING_past_its_namesake():
    """The role has to filter the candidates, not merely pick between equals.

    **A mutation found this gap.** With the role dropped from the candidate list, the case above
    still passed, because the tiebreak asks where a thing of *that role* was last seen and the right
    answer was nearest to it. Only when the chosen thing **moves** — a list that scrolled, which the
    design says happens constantly — does a same-named label of another role end up nearer to where
    it used to be, and win.

    Here the label never moves and the list item scrolls 120 px away; the label is then 100 px from
    where the item was and the item is 120 px. Without the role in the filter, the label wins.
    """
    label = w("Item", role="text", rect=(0, 100, 40, 20))
    seen = [label, w("Item", role="list item", rect=(0, 200, 40, 20))]
    scrolled = [label, w("Item", role="list item", rect=(0, 320, 40, 20))]
    s = session(FakeModel(says("열게요", {"click": 1})), seen, scrolled, scrolled, live=True)
    ex = s.exchange("그거 눌러줘")
    assert ex.did and ex.did[0].target.role == "list item", (
        "it followed the nearest namesake instead of the thing that was chosen")
    assert s.hands.done[0][2] == (20, 330)


def test_an_effect_that_names_NO_role_still_resolves():
    """The role is half of a key, not a requirement. An effect built without one — by an older
    caller, or by a test — must not start refusing everything."""
    from argmind.core.intent import Effect                                # noqa: PLC0415

    s = session(FakeModel(), SCREEN, SCREEN, SCREEN, live=True)
    s.observe()
    assert s.find("닫기") is not None
    assert Effect("click", target="닫기").role == ""


def test_a_target_that_MOVED_is_clicked_where_it_is_now():
    """The other half, and the reason the match is on the name rather than the rectangle: a list that
    scrolled or a window that resized moves everything, and the thing the person asked for is still
    there — somewhere else."""
    moved = [w("닫기", rect=(999, 500, 40, 30))] + SCREEN[1:]
    s = session(FakeModel(says("닫을게요", {"click": 0})), SCREEN, moved, moved, live=True)
    ex = s.exchange("close it")
    assert ex.did[0].target.centre == (1019, 515), (
        "it clicked the old rectangle; the name follows the world and the rectangle does not")
    assert s.hands.done[0][2] == (1019, 515)


def test_confirmation_asks_the_WIDGET_LAYER_and_not_pixels():
    """The clicked thing being gone *after* the click, or the window having changed, is exact.
    Comparing screenshots would be a guess about whether the right thing moved."""
    gone = [x for x in SCREEN if x.name != "닫기"]
    #: present when asked, still present when re-found, gone after the click — a close that worked.
    s = session(FakeModel(says("ok", {"click": 0})), SCREEN, SCREEN, gone, live=True)
    assert s.exchange("close it").did[0].confirmed is True

    s2 = session(FakeModel(says("ok", {"click": 0})), SCREEN, SCREEN, SCREEN, live=True)
    assert s2.exchange("close it").did[0].confirmed is False, (
        "the button is still exactly where it was, which usually means the click missed")


def test_the_report_separates_LOOKING_from_THINKING_from_ACTING():
    """"It is fast" is a claim about which part. Timed separately so a regression lands on the piece
    that caused it rather than on the total."""
    s = session(FakeModel(says("ok", {"click": 1})))
    ex = s.exchange("share it")
    assert ex.total_s == pytest.approx(ex.look_s + ex.think_s + ex.act_s)
    r = s.report()
    assert r["thoughts"] == 1 and r["acted"] == 1 and r["spoke_only"] == 0
    assert r["asked"] == 1 and r["noticed"] == 0


def test_converse_returns_an_INTENT_even_when_there_is_nothing_to_say():
    """An empty intent is a real answer and has to be distinguishable from a failure, so it is
    falsy but never `None`."""
    intent = converse(FakeModel(''), SCREEN, Perceived("something"))
    assert intent is not None and not intent
    assert intent.trigger is not None, "an intent without its cause cannot be permission-checked"
