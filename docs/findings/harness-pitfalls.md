# Ten ways this harness produced a confident wrong number

Moved out of `CLAUDE.md` on 2026-09-10 to stop paying for it on every turn. **This is the list to
check the next time a measurement here reads zero, or reads something convenient.**

Every one was met on this machine. Every one was invisible from the outside — the agent simply did
nothing, or did something plausible. **Three would have killed a direction that works and one would
have kept a direction that does not**, and each pointed the way the person reading it was already
leaning.

## Six from the local model, 2026-09-08/09

1. **`format: "json"` silences the model completely** — 366 ms, empty output. The shape is asked for
   in the prompt and pulled out of the text instead.
2. **Too small a `num_predict` truncates it mid-thought.** `done_reason` says `length` and the
   content is empty, which reads as a refusal.
3. **The answer is on `message.content`.** Reading `response`, the other endpoint's field, returns
   nothing at all.
4. **`num_ctx` defaults to 4096** and a 1440p screenshot plus a reply is about 4,096 tokens. The
   recorded conclusion "cannot work at native resolution" was this.
5. **Instructing the coordinate convention** starts a 13,000-character argument with itself. The
   convention is *observed*, not instructed — the model emits 0–1000 normalised coordinates
   unprompted, and a run scored "0 of 18, below chance" was reading those as pixels.
6. **Prefill measured warm is 25 ms against 0.9–5.5 s cold.** A warm-cache figure was published and
   withdrawn.

## Four more from the rebuild, 2026-09-09/10

7. **`Reply.json()` was a non-greedy brace match**, correct only while a reply contained no nested
   object. The moment the answer grew a list of things to do it stopped at the inner brace, failed
   to parse, and **every action silently disappeared while the model answered perfectly**. Pinned by
   `test_a_NESTED_reply_is_read_whole`.
8. **A prompt that said "the exact name of one thing above"** over lines reading `- button "Copy"`
   got `button "Copy"` back, and **five correct answers scored zero**.
9. **A shuffle seeded with `hash()` of a string**, which Python randomises per process, so the first
   analysis scored against an order the model was never shown and reported a median gap of 27 that
   meant nothing.
10. **A thinking model's cost is invisible if the thinking is discarded.** `llm.py` throws away
    `message.thinking` — correctly, it is not an answer — and that hid 201 output tokens and 1,112
    characters of reasoning behind a twelve-token reply, at 2.95 s to first token. The length is
    recorded as `Reply.thought` now; `Local(think=False)` switches it off, and models that have
    never heard of the field accept it.

## And seven ways the measuring code itself has lied

These are not about the model. `.claude/agents/measurement.md` carries them as a working brief.

1. **A filename that does not carry everything the number depends on.** Three baselines destroyed —
   the protocol was in the name and not the model, then not the prompt, and **the third was
   destroyed by the fix for the second**.
2. **Conditions not written into the result.** Four rows of a model comparison were measured under
   one reply shape and a fifth under another, and nothing in the files said so.
3. **A default that silently selects the other configuration.** `--think` defaulted to unset, so a
   reasoning model reasoned and a run meant to measure a prompt measured the prompt *and* the
   thinking. **The latency told on it and nothing else would have.**
4. **A display that disagrees with the verdict beside it.** A result taken as the worst of three
   repeats printed beside repeat one's choice, so a goal that chose the accepted answer showed as
   wrong — **twice, the second hours after the first was criticised**.
5. **A boolean where the delta was needed.** A click that landed and did nothing looked identical to
   a click that missed, and two causes stayed indistinguishable through a whole round of work. A
   function with four failure paths returning `None` for all of them is the same defect.
6. **A harness that can only fail one way.** `choosing.py` carries nine goals with no right answer;
   `talking.py` is its opposite half. A model that never acts scores perfectly on one and uselessly
   on the other.
7. **An asymmetric comparison.** One window read before and every window read after, so widgets that
   were always there counted as newly appeared.
