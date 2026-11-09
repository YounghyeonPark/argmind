//! One call: what to say, and what if anything to do.
//!
//! The prompt this renders is fingerprinted by `measure/choosing.py` as `r322a4c` — a hash of the
//! rendered text over six probes. **A renderer that produces that tag has provably not changed a
//! character of what the model is sent**, which matters because prompt edits in this project have
//! been measured as never free: five in a row each raised one number and lowered another.

use crate::contract::Widget;
use crate::intent::{Effect, Intent, Trigger};
use crate::llm::first_object;
use crate::mind::Mind;
use crate::thinking::Note;

pub const MAX_LISTED: usize = 200;
/// **Derived, not chosen.** 27 of 28 real requests want one effect or two; spurious ones come in
/// threes, fours and an eight. Three is the documented sequence — click, type, enter.
pub const MOST_AT_ONCE: usize = 3;

pub const SYSTEM: &str = r#"You are a helper the person talks to while they work at their PC.

You are shown the things currently on their screen, each with a number. Reply as a person would —
briefly, in the language they used.

If they asked you to do something and the listed things can do it, name the numbers. If they were
just talking, or the thing they want is not listed, say so and do nothing: a wrong click is worse
than an honest "I can't see that here"."#;

/// What an unprompted turn is told, and **the line that made argmind narrate.**
///
/// It used to read *Mentioning it is usually enough.* That was written about the action slot — *do
/// not act, say something instead* — and the model read it as an instruction to mention. Measured
/// 2026-09-17 over eight window changes: **it spoke 8 times out of 8, and 6 of those were the same
/// sentence with the window name swapped.** With watching on by default that is most of what
/// argmind says, and the owner's transcript is four of them in a row.
///
/// ## The edit that was tried, and what it cost — **reverted**
///
/// Making silence the stated default — *Most of the time there is nothing worth saying: leave "say"
/// empty* — was measured at `r9d5af7` against this text at `r322a4c`:
///
/// | | spoke | wanted to act | same shape |
/// |---|---|---|---|
/// | this text | 8 of 8 | **0 of 8** | 6 of 8 |
/// | telling it to be silent | 8 of 8 | **2 of 8** | 4 of 8 |
///
/// **It bought no silence at all and started wanting to click things.** The action count was put in
/// as the control precisely because a prompt edit can buy quiet by acting instead; here it bought
/// neither. Reverted, and **prompt edits in this project are now 0 for 6.**
///
/// What that leaves: the model fills `say` whichever way it is asked, so the lever is probably not
/// the prompt. `noticing`'s own `beyond_the_name` is mechanical and already separates *it added
/// something* from *it restated the trigger*, which is the shape of a guard on the way out — the
/// same answer `parroted` reached after being told not to echo did not work either.
pub const UNPROMPTED: &str = "Nobody asked you anything — you noticed this yourself. Mentioning it is usually\nenough. Only propose doing something if it is clearly wanted, and expect to be asked first.";

pub const STILL_OPEN: &str = "Still open, from earlier:";
pub const STILL_OPEN_TAIL: &str =
    "What they say next may be a yes to that, a no, or nothing to do with it.";

pub const OFFER_OPEN: &str = "offered";
pub const OFFER_SETTLED: [&str; 3] = ["did", "could not", "they said no to"];

/// The list as the model sees it. Names pass through untouched — not lowercased, not translated:
/// the model matched an English request to the Korean widget `닫기` without help, which is the case
/// a keyword matcher fails and the reason a model is here at all.
pub fn describe_screen(widgets: &[Widget], limit: usize) -> String {
    widgets
        .iter()
        .take(limit)
        .enumerate()
        .map(|(i, w)| format!("{}: {} \"{}\"", i, w.role, w.name))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The offers nobody has answered yet, oldest first, in the words the history holds them in.
/// Closes on **any** settling label rather than matching the effect — matching would mean parsing
/// the history's prose back into an effect.
pub fn still_open(history: &[(String, String)]) -> Vec<String> {
    let mut open: Vec<String> = Vec::new();
    for (who, text) in history {
        if who == OFFER_OPEN {
            open.push(text.clone());
        } else if OFFER_SETTLED.contains(&who.as_str()) {
            open.clear();
        }
    }
    open
}

/// The prompt, exactly. Verified against `r322a4c` — **and `r322a4c` is what a resting [`Mind`]
/// still renders**, which is the whole safety of having added one. See [`crate::mind`].
pub fn render(widgets: &[Widget], trigger: &Trigger, history: &[(String, String)],
              limit: usize, mind: &Mind) -> String {
    render_with(widgets, trigger, history, limit, mind, &Note::blank())
}

/// The same, plus what it had already worked out. **A blank [`Note`] renders nothing**, so this is
/// `render` — see [`crate::thinking`] for why that guarantee is the price of adding a slot at all.
pub fn render_with(widgets: &[Widget], trigger: &Trigger, history: &[(String, String)],
                   limit: usize, mind: &Mind, note: &Note) -> String {
    let listed: Vec<&Widget> = widgets.iter().take(limit).collect();
    let mut lines: Vec<String> = vec![SYSTEM.to_string()];
    if !trigger.asked {
        lines.push(UNPROMPTED.to_string());
    }
    lines.push(String::new());
    if !history.is_empty() {
        lines.push("Recently:".to_string());
        for (who, text) in history {
            lines.push(format!("  {who}: {text}"));
        }
        lines.push(String::new());
    }
    //: **What it was thinking before this moment**, above the screen rather than below it: the
    //: screen is the largest block in the prompt and a slot underneath it is a slot behind two
    //: hundred widget names. Placement is unargued, like the dials' — what is measured is only
    //: that an empty note changes nothing.
    if let Some(block) = note.block() {
        lines.push(block);
        lines.push(String::new());
    }
    lines.push("On screen now:".to_string());
    lines.push(if listed.is_empty() {
        "  (nothing named — this surface exposes no widgets)".to_string()
    } else {
        let owned: Vec<Widget> = listed.iter().map(|w| (*w).clone()).collect();
        describe_screen(&owned, limit)
    });

    //: **A slot fed by conversation state, not a sentence.** Present only when an offer is open,
    //: recomputed every turn, and its tail names all three readings on purpose — a line leaning
    //: one way is what slid the five prompt edits before it.
    let pending = if trigger.asked { still_open(history) } else { Vec::new() };
    if !pending.is_empty() {
        lines.push(String::new());
        lines.push(STILL_OPEN.to_string());
        for text in &pending {
            lines.push(format!("  {text}"));
        }
        lines.push(STILL_OPEN_TAIL.to_string());
    }

    //: **What the person set in the settings, and nothing whatsoever when they set nothing.**
    //: Placed here, next to the trigger, for the only reason this project has a number for: a
    //: stated rule is obeyed in 36 of 36 probes at a distance of eight lines or less. That is a
    //: measurement about rules and this is a manner, so the placement is **unargued** — the last
    //: time a placement was argued from prompt distance the argument was withdrawn by the run that
    //: confirmed the change. What would settle it is `saying` and `talking` run with the slot at
    //: the top and here; what is measured below is the prior question, whether it does anything.
    if let Some(aim) = mind.line() {
        lines.push(String::new());
        lines.push(aim);
    }

    lines.push(String::new());
    lines.push(if trigger.asked {
        format!("They said: {}", trigger.text)
    } else {
        format!("You noticed: {}", trigger.text)
    });
    lines.push(String::new());
    lines.push(
        r#"Reply with only JSON: {"say": "<what you say back>", "do": [ ... ]} where each entry of"#
            .to_string());
    lines.push(
        r#""do" is {"click": <number>}, {"type": "<text to type>"} or {"key": "<enter|tab|escape>"}."#
            .to_string());
    lines.push(r#"Leave "do" empty if nothing should happen."#.to_string());
    lines.join("\n")
}

/// One entry of `do`, validated against what is actually on screen.
///
/// **The number becomes a name and a role here and never travels further as an index.** That one
/// line is the staleness fix; the role is half of the key, and leaving it out was a real defect —
/// a window with a label and a combo box both called `Font :` resolved to the label and the click
/// did nothing.
pub fn effect_of(item: &serde_json::Value, listed: &[Widget], trigger: &Trigger) -> Option<Effect> {
    let obj = item.as_object()?;
    let because = Some(trigger.asked);
    if let Some(v) = obj.get("click") {
        let index = match v {
            serde_json::Value::Number(n) => n.as_i64()?,
            serde_json::Value::String(s) => s.parse::<i64>().ok()?,
            _ => return None,
        };
        if index < 0 || index as usize >= listed.len() {
            return None;
        }
        let w = &listed[index as usize];
        return Some(Effect { kind: "click".into(), target: w.name.clone(), role: w.role.clone(),
                             text: String::new(), because_asked: because , which: None});
    }
    if let Some(v) = obj.get("type") {
        let s = v.as_str()?;
        if s.is_empty() {
            return None;
        }
        return Some(Effect { kind: "type".into(), target: String::new(), role: String::new(),
                             text: s.to_string(), because_asked: because , which: None});
    }
    if let Some(v) = obj.get("key") {
        let s = v.as_str()?;
        return Some(Effect { kind: "key".into(), target: String::new(), role: String::new(),
                             text: s.to_lowercase(), because_asked: because , which: None});
    }
    None
}

/// **Did it just say their own words back at them with a question mark?**
///
/// The owner, 2026-09-17: 머해 → 머해?, then 왜 따라해 → 머해?, then 왜 반복해 → 머해?. 바본가.
///
/// Reproduced with every dial at rest, so it is not the dials, and the model emits it itself —
/// `{"say": "머해?", "do": []}` off the wire, so it is not the reader either. **It is what this
/// model does when it cannot parse the input**: `뭐해` is answered properly and `머해`, the
/// colloquial spelling of the same two syllables, is mirrored. Then the mirror is in the history
/// and the next two turns get it again, which is what the screenshot shows.
///
/// **Telling it not to do this does not work** — given *상대가 한 말을 그대로 되돌려 말하지 말 것*
/// in the prompt it replied `머해`, dropping the question mark and keeping the parrot. So this is
/// caught on the way out, the way `argmind-env-whisper` catches a degenerate transcript rather
/// than asking Whisper nicely.
///
/// ## Why the rule is this narrow
///
/// **Only *their words plus a question mark*.** Exact equality is excluded on purpose: `안녕` is a
/// perfectly good answer to `안녕`, and `네` twice running is a conversation, not a fault. Turning
/// what somebody said into a question back at them is the one shape with no honest reading.
pub fn parroted(say: &str, trigger: &Trigger) -> bool {
    let said = trigger.text.trim();
    let back = say.trim();
    if said.is_empty() || back.len() <= said.len() {
        return false;
    }
    //: `?` and the full-width `？` a Korean keyboard can produce, and nothing else — a reply that
    //: adds a *word* is a reply.
    let Some(rest) = back.strip_prefix(said) else { return false };
    rest.chars().all(|c| c == '?' || c == '？' || c.is_whitespace())
}

/// How much of the shorter of two sentences is one run of text they share, 0.0..=1.0.
///
/// **The first attempt removed the trigger's words from the reply and compared what was left.** It
/// does not work: a watch says *the window changed to 'Taskbar'* and the reply says *태스크바로
/// 넘어왔어요* — the name is **translated**, so the trigger's letters are not in the reply to remove.
/// Nor does comparing the tail, which was the attempt before that and worked only while the name
/// happened to come first.
///
/// What survives translation is that the two remarks *are the same sentence*. The longest run they
/// share needs no vocabulary, no name extraction and no assumption about word order:
///
/// | | shared run |
/// |---|---|
/// | *태스크바로 넘어왔어요. 도움이 필요하시면 언제든 말씀해주세요!* against the same for another window | most of it |
/// | the same against *지우개가 22번 버튼입니다.* | a few characters |
pub fn overlap(a: &str, b: &str) -> f64 {
    let (x, y): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    if x.is_empty() || y.is_empty() {
        return 0.0;
    }
    //: One row at a time: these are sentences, and a full table of a screenful would be the only
    //: thing here that ever allocated seriously.
    let mut prev = vec![0usize; y.len() + 1];
    let mut best = 0usize;
    for i in 1..=x.len() {
        let mut row = vec![0usize; y.len() + 1];
        for j in 1..=y.len() {
            if x[i - 1] == y[j - 1] {
                row[j] = prev[j - 1] + 1;
                best = best.max(row[j]);
            }
        }
        prev = row;
    }
    best as f64 / x.len().min(y.len()) as f64
}

/// How much two remarks may share before they are the same remark.
///
/// **0.6, chosen with the margin shown rather than tuned.** The pair that started this shares about
/// four fifths of the shorter sentence; two remarks that are actually about different things share
/// a few characters of a dozen. Nothing between 0.4 and 0.8 would change the verdict on either, and
/// the tests pin both sides so a later move has to break one of them.
pub const SAME_REMARK: f64 = 0.6;

/// **Has argmind already said this, in all but the name?**
///
/// Measured 2026-09-17: over eight window changes it spoke **8 times out of 8**, and **6 of those
/// were one sentence with the window name swapped** — *Chrome 데스크탑으로 넘어왔어요. 도움이
/// 필요하시면 언제든 말씀해주세요!*, then the same for the taskbar, then for the next window. A
/// person alt-tabs all day, so with watching on that was most of what argmind said.
///
/// **Telling it not to did not work.** Making silence the stated default in `UNPROMPTED` bought no
/// silence at all and started it wanting to click things, 0 of 8 to 2 of 8. This is the other
/// answer, the one that worked for the parrot: catch it on the way out.
///
/// **The first time is said and the repeats are not.** A window change genuinely worth remarking on
/// is remarked on once; the rule is against the fourth identical remark, not against the first.
pub fn said_before(say: &str, lately: &[String]) -> bool {
    let now = say.trim();
    //: **Too short to judge on.** Two four-character remarks overlap by accident, and this rule is
    //: about repetition rather than about brevity.
    if now.chars().count() < 8 {
        return false;
    }
    lately.iter().any(|before| overlap(now, before.trim()) >= SAME_REMARK)
}

/// What a reply means, given what was on screen. An empty `Intent` is a real answer.
pub fn read_reply(reply: &str, listed: &[Widget], trigger: &Trigger) -> Intent {
    let got = first_object(reply).unwrap_or(serde_json::json!({}));
    let say = got.get("say").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
    let wanted = got.get("do").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    //: **Reported, not erased.** The words stay exactly as the model produced them so a transcript
    //: or a harness can still see what happened; an interface that wants to say something else in
    //: their place has the flag to know it should.
    let parroted = parroted(&say, trigger);
    if wanted.len() > MOST_AT_ONCE {
        //: The whole batch goes, not the first three. A reply asking for eight things has
        //: misunderstood, and doing three of them is worse than saying so.
        return Intent { say, effects: Vec::new(), parroted, sure: None };
    }
    let effects = wanted.iter().filter_map(|item| effect_of(item, listed, trigger)).collect();
    Intent { say, effects, parroted, sure: None }
}

/// **How sure the model was of the action part of a reply** — the product of the probabilities of
/// every token after `"do":`, which is the action given what was already said. The reply shape puts
/// `say` first, so this is *how sure of the click, having decided what to say*.
///
/// `None` when there is no action — `null`, `[]`, or no `do` at all — or no probabilities. Pure, so
/// it can be checked without a model.
/// **Does this reply ask before doing?** *찾아볼까요?* with a click beside it is two answers to one
/// question — the words say *shall I* and the action slot says *I did*. The owner's request for a
/// trip plan was answered that way and the click was a Git button in the editor. A question ending
/// in *까요?* means the action waits for the person, however 확인 없이 실행 is set.
pub fn asks_first(say: &str) -> bool {
    let s = say.trim().trim_end_matches(|c: char| c == '!' || c.is_whitespace());
    s.ends_with('?') && s.contains("까요")
}

/// **A request nobody can read** — its letters replaced by `?` or `�` on the way in. On
/// 2026-11-09 a sentence sent through a shell that was not speaking UTF-8 arrived as `?? 40mm ??
/// 20mm ?? 10mm ??? 3D? ??? part.stl? ????`; the turn answered *상자를 만들어 드릴게요* and, with 확인
/// 없이 실행 on, pressed *Synchronize Changes* in the editor and *Bookmarks* in Chrome. A garbled
/// sentence is not a request, so it is asked again and nothing is done.
///
/// A `?` counts only in a run that **starts a word** — a garbled syllable — so *뭐해???* and *왜?
/// 왜?* are questions, not damage. Unreadable when at least three such marks make up at least a
/// quarter of what was written, or when three replacement characters `�` are in it at all.
pub fn unreadable(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let written = chars.iter().filter(|c| !c.is_whitespace()).count();
    let replaced = chars.iter().filter(|&&c| c == '\u{FFFD}').count();
    let mut lost = 0;
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '?' && (i == 0 || chars[i - 1].is_whitespace()) {
            let start = i;
            while i < chars.len() && chars[i] == '?' {
                i += 1;
            }
            lost += i - start;
        } else {
            i += 1;
        }
    }
    replaced >= 3 || (lost >= 3 && lost * 4 >= written)
}

pub fn action_sureness(tokens: &[(String, f64)]) -> Option<f64> {
    if tokens.is_empty() {
        return None;
    }
    let mut text = String::new();
    let mut starts = Vec::with_capacity(tokens.len());
    for (t, _) in tokens {
        starts.push(text.len());
        text.push_str(t);
    }
    let key = text.find("\"do\"")?;
    let colon = key + text[key..].find(':')? + 1;
    let value = text[colon..].trim_start();
    if value.starts_with("null") || value.starts_with("[]") || value.is_empty() {
        return None;
    }
    let mut logp = 0.0;
    let mut any = false;
    for (i, (t, lp)) in tokens.iter().enumerate() {
        //: Every token that ends after the colon — a token straddling it carries the value's start.
        if starts[i] + t.len() > colon && !t.trim().is_empty() {
            logp += lp;
            any = true;
        }
    }
    any.then(|| logp.exp())
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_garbled_request_is_not_a_request() {
        //: Word for word what arrived on 2026-11-09.
        assert!(super::unreadable("?? 40mm ?? 20mm ?? 10mm ??? 3D? ??? part.stl? ????"));
        assert!(super::unreadable("??? ?? ????"));
        assert!(super::unreadable("\u{FFFD}\u{FFFD} 40mm \u{FFFD}\u{FFFD}\u{FFFD}"));
        //: Questions, however many marks.
        for ok in ["뭐해???", "왜? 왜? 왜? 왜?", "이거 뭐야? 저건?", "what?", "가로 40mm 세로 20mm 상자 만들어줘",
                   "? 이건 뭐지", "", "??"] {
            assert!(!super::unreadable(ok), "{ok}");
        }
    }

    #[test]
    fn a_reply_that_asks_holds_its_action() {
        for q in ["먼저 항공권과 호텔을 찾아볼까요?", "검색해 볼까요? ", "이걸로 할까요?!"] {
            assert!(super::asks_first(q), "{q}");
        }
        for s in ["검색했습니다", "무엇을 찾으세요?", "열었어요. 더 볼까요 말까요는 나중에", ""] {
            assert!(!super::asks_first(s), "{s}");
        }
    }
}
