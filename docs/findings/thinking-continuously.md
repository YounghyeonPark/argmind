# Thinking continuously

**The owner, 2026-09-17:** 입력정보는 전부(화면, 카메라, 음성, 키입력 등등) 읽고 생각을 계속
진행해야 함 — and, earlier the same day, 그냥 지켜보는것만 업데이트 하는거면 의미없고 계속 자기
생각을 반복해야함.

## What was actually missing

A turn's prompt was the system sentence, twelve turns of history, the screen now, an unanswered
offer, and the dials. **Not one of those is something argmind concluded.** `Session` held `seen`,
`history` and four counters; history holds *what was said*, never *what it worked out*. Ask it the
same question twice and the second answer is not built on the first — it is built on a transcript
of the first.

So the honest answer to *argmind는 어떤 생각을 함* was: **it does not think, it answers.**

## What was built

A `Note` with three fields, revised on a timer, rendered into the prompt above the screen, and shown
in the page. Two of the fields the model writes; **the third it is not allowed to.**

| | |
|---|---|
| believes | what it takes to be true about what the person is doing |
| waiting | what it is expecting — the field that gives the next tick a subject |
| tried | what argmind actually did, **read off its own record** |

The tick is a **separate model call whose answer shape has no `say` and no `do` in it.** That is what
makes a timer safe here: there is no path from a tick to the screen or to the speaker. It runs every
30 s, and only while 지켜보기 is on, because they are the same thing in substance — argmind spending
model calls on a screen nobody pointed at.

**A blank note renders nothing.** The six probes still hash to `r322a4c` at 5,676 bytes, the tag
recorded from the Python. Same discipline as the dials, and the same reason: prompt edits here are
0 for 5 and the ones that paid were slots that were absent until there was something to put in them.

## Three things went wrong in the first run, and each is worth keeping

150 seconds of somebody moving between an editor, two browsers and a file manager.

**1. It invented what it had done.** Asked for all three fields, from a session in which argmind had
done precisely nothing, it answered *이미 메모리 사용량이 높은 탭들을 목록으로 정리해두고, 사용자가
필요할 때 쉽게 접근할 수 있도록 했다.* It had not, and it carried that for three revisions.

A fabricated `tried` is worse than an empty one: the field exists to stop argmind offering the same
thing twice, so a false entry suppresses an offer that was never made. And it is the wrong side of
this project's line — *the deterministic side says WHERE, the probabilistic side says WHAT.* **What
argmind did is not a judgement, it is a record**, written into the history by `remember_doing`. So
`what_it_did` reads it and the model is asked for two fields. Re-run: `이미 한 것` empty for 100
seconds, correctly.

**2. It anchored.** Four revisions in, a file manager on `C:` was still being read as *exploring
files related to Younghyeon Park* — the old belief bending new evidence to fit. The prompt now says
so in as many words: 화면이 바뀌었으면 앞의 메모를 이어붙이지 말고 버려라. Re-run: three different
activities, three independent beliefs, no *still*.

**3. It wrote in English.** Which is this morning's measurement again — the language of an added
block decides the language of the reply, 28 of 44 → 19 with an English one and → 42 with a Korean
one. The note is read by a person in a panel with Korean labels *and* goes back into a prompt that
has to come out in their language. Both halves point the same way, so the whole thinking prompt and
the rendered block are Korean, and the block's three labels are the page's three labels exactly.

**What is still not good: `waiting`.** In the first run it paraphrased the belief with *may* — *User
may proceed with the branch renaming* — which is not an expectation. Asking for it by name and
concretely improved it to *사용자가 특정 기사나 링크를 클릭할지 기다리고 있다*, which is a real
expectation but a generic one. Nothing yet measures whether a `waiting` is ever *met*, and that is
the measurement this field needs.

## And the watch was reading argmind itself

Found by running it. `attending.rs` had written the prediction down and left it:

> It assumes the person is in the interface when they act. That holds for a typed turn — they typed
> it — and it would not hold for a turn started by a `Watch` firing while they work elsewhere.
> **Nothing here fires on a watch yet; when something does, this is the thing to revisit.**

Watching became the default and a tick was added, so things fire on a watch constantly, and the
assumption broke exactly where it said it would. The old rule was *the newest window that is not the
one in front*, so while the person worked in Chrome, having come from argmind, the newest handle
that was not Chrome **was argmind**. Over those 150 seconds, **4 of 8 notices read `the window
changed to 'argmind'`** — a model call each, spent reading argmind's own transcript.

The rule is now `app.py`'s original and sharper one, available again since the page moved into a
window this process owns: **the newest window that is not ours.** Both halves of the old rule had to
go — keeping the *in front* exclusion and adding a process filter answers with the window *before*
the one the person is using, which is right for a typed turn and wrong for every watch.

**Prediction written before the change: zero notices naming argmind. The re-run gave zero.**

The rule is now pure — `attended_where` takes ownership as a function — because `ours` is a
process-id call and a test against the real one could only use handles belonging to nobody, which is
precisely the case the rule does not turn on.

## What this costs

Two model calls a minute while 지켜보기 is on, on top of whatever the watch spends, at 0.61 s median.
It is the reason the tick is tied to the same switch rather than having one of its own.

## The other inputs, one at a time

The request was **화면, 카메라, 음성, 키입력 등등**. They are in very different states and saying so
is more use than a uniform answer:

* **화면** — done, and now on by default. Read every 0.7 s while watching; a thought at most every
  3 s; a note every 30 s.
* **음성** — blocked on one measurement, and the harness for it is written. Always-on listening needs
  a gate in front of the recogniser because Whisper says words for silence 120 times out of 120, and
  the threshold for that gate needs a recording of this room that does not exist yet.
  `argmind-measure --bin listening` makes it in 30 seconds and needs somebody to talk.
* **카메라** — **not blocked on effort, blocked on there being nothing to send a frame to.** The
  model is `qwen3:8b`, which is text. A vision model was measured at 24–82 s for a single pixel
  coordinate and on a dense screen never answered, which is why the whole architecture says the
  model never emits one. A camera frame has no path into this prompt at all until a vision model is
  chosen, and that is a decision — which one, its licence, and the VRAM it takes beside the two
  servers already running.
* **키입력** — implementable and **deliberately not implemented.** Reading every keystroke means a
  global keyboard hook, which is a keylogger: it captures passwords, card numbers and messages to
  other people, and it would sit in a program that sends text to a model. What argmind plausibly
  wants from the keyboard is *activity* — that the person is typing, in which application, how fast,
  whether they stopped — and none of that needs the characters. That is a different feature with a
  different permission, and it is the owner's call which one was meant.
