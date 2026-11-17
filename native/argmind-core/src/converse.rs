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
    //: **The placeholder in the language set**, the line nearest the reply — and at rest the
    //: same characters as ever. See `mind::Language`.
    lines.push(format!(
        r#"Reply with only JSON: {{"say": "{}", "do": [ ... ]}} where each entry of"#,
        mind.say_slot().unwrap_or("<what you say back>")));
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
        return Intent { say, effects: Vec::new(), parroted, sure: None, rewritten_from: None };
    }
    let effects = wanted.iter().filter_map(|item| effect_of(item, listed, trigger)).collect();
    Intent { say, effects, parroted, sure: None, rewritten_from: None }
}

/// **Which script a piece of text is written in**: Hangul is Korean, kana is Japanese, Han with
/// neither is Chinese, Latin letters with none of those is English, and two of Hangul and kana
/// together is `"mixed"` — *メモ장이 열렸습니다*. `None` when it has no letters at all (an emoji, a
/// number), which is in no language and so in every one.
pub fn written_in(text: &str) -> Option<&'static str> {
    let any = |lo: char, hi: char| text.chars().any(|c| (lo..=hi).contains(&c));
    let hangul = any('\u{AC00}', '\u{D7A3}') || any('\u{1100}', '\u{11FF}') || any('\u{3130}', '\u{318F}');
    let kana = any('\u{3040}', '\u{30FF}');
    let han = any('\u{4E00}', '\u{9FFF}');
    let latin = text.chars().any(|c| c.is_ascii_alphabetic());
    match (hangul, kana, han, latin) {
        (true, true, _, _) => Some("mixed"),
        (true, false, _, _) => Some("ko"),
        (false, true, _, _) => Some("ja"),
        (false, false, true, _) => Some("zh"),
        (false, false, false, true) => Some("en"),
        _ => None,
    }
}

/// **Is this reply in the language the person set?** Read with the screen's own labels taken out,
/// because *メモ장を開きます* names a button that is called 메모장 and is Japanese in every word it
/// chose. **A Japanese reply may hold Han** — every one does — so it is read as Japanese when it
/// has kana and no Hangul, and a reply of Han alone (*了解。*) is read as Chinese and costs one
/// rewrite. Unset or unknown codes are always in it: nothing was asked for.
pub fn in_language(say: &str, listed: &[Widget], code: &str) -> bool {
    if crate::mind::language(code).is_none() {
        return true;
    }
    let mut bare = say.to_string();
    let mut names: Vec<&str> = listed.iter().map(|w| w.name.as_str()).filter(|n| !n.trim().is_empty()).collect();
    //: Longest first, so *Visual Studio Code* goes before a *Code* inside it.
    names.sort_by_key(|n| std::cmp::Reverse(n.chars().count()));
    for n in names {
        bare = bare.replace(n, " ");
    }
    match written_in(&bare) {
        None => true,
        Some(script) => script == code,
    }
}

/// **The one extra call, for a reply that came back in another language** — the language's own
/// sentence, written in it, and the reply after it. Only ever made when [`in_language`] says no.
pub fn rewrite_prompt(say: &str, code: &str) -> Option<String> {
    crate::mind::language(code).map(|l| format!("{}\n\n{}", l.rewrite, say.trim()))
}

/// **What came back from a rewrite, if it is one.** Taken only when it is in the language asked
/// for, is not empty, and is not much longer than what it rewrites (a model that answers its own
/// words rather than rewriting them writes more) — otherwise `None`, and the reply stays the
/// model's own, in the wrong language but not invented. Quotes around the whole of it go.
pub fn read_rewrite(reply: &str, was: &str, listed: &[Widget], code: &str) -> Option<String> {
    let mut t = reply.trim();
    for (open, close) in [('「', '」'), ('『', '』'), ('"', '"'), ('“', '”'), ('\'', '\'')] {
        if t.chars().count() >= 2 && t.starts_with(open) && t.ends_with(close) {
            t = t[open.len_utf8()..t.len() - close.len_utf8()].trim();
        }
    }
    let longest = 3 * was.chars().count() + 20;
    (!t.is_empty() && t.chars().count() <= longest && in_language(t, listed, code)).then(|| t.to_string())
}

/// **Words that are not a turn's reply, said in `code`** — [`crate::session::Session::say_in`], here
/// so `--bin tongues` measures the function the application calls. At most one model call, and
/// only for words not in `code` already.
pub fn say_in(model: &mut dyn crate::contract::Speaks, text: &str, code: &str) -> String {
    if text.trim().is_empty() || in_language(text, &[], code) {
        return text.to_string();
    }
    let Some(ask) = rewrite_prompt(text, code) else { return text.to_string() };
    let again = model.ask(&ask);
    match read_rewrite(&again, text, &[], code) {
        Some(said) if keeps_the_facts(text, &said) => said,
        _ => text.to_string(),
    }
}

/// **Did a rewrite keep what the words reported?** Every run of digits (grouping commas read through,
/// so *369,700* and *369700* are one) and every file name in `was` must be in `now`. Words are
/// free to change; a number or a name is a fact about the work.
pub fn keeps_the_facts(was: &str, now: &str) -> bool {
    let digits = |t: &str| -> Vec<String> {
        let plain: String = t.chars().filter(|c| *c != ',').collect();
        plain.split(|c: char| !c.is_ascii_digit()).filter(|d| !d.is_empty()).map(str::to_string).collect()
    };
    let have = digits(now);
    digits(was).iter().all(|d| have.contains(d)) && named_files(was).iter().all(|f| now.contains(f.as_str()))
}

/// **File names, in any script** — *ideas.md*, and *수익_방법_목록.md* with the particle after it
/// (*…목록.md에*): a name whose stem is Korean is still the name of a file, and a rewrite into English
/// that calls it *income_methods_list.md* has named a file that is not there. The extension is the
/// ASCII letters and digits after the last dot, starting with a letter, one to five of them.
pub fn named_files(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for word in text.split(|c: char| c.is_whitespace() || "「」『』()（）[]\"'“”‘’,，、:：;；".contains(c)) {
        let Some(dot) = word.rfind('.') else { continue };
        let stem = word[..dot].trim_start_matches(|c: char| c == '`' || c == '*');
        let ext: String = word[dot + 1..].chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
        if !stem.is_empty() && !stem.ends_with('.') && (1..=5).contains(&ext.len())
            && ext.starts_with(|c: char| c.is_ascii_alphabetic()) {
            let name = format!("{stem}.{ext}");
            if !out.contains(&name) {
                out.push(name);
            }
        }
    }
    out
}

/// **Does this reply ask before doing?** *찾아볼까요?* with a click beside it is two answers to one
/// question — the words say *shall I* and the action slot says *I did*. The owner's request for a
/// trip plan was answered that way and the click was a Git button in the editor. A question ending
/// in *까요?* means the action waits for the person, however 확인 없이 실행 is set.
///
/// **And the other ways of offering — since 2026-11-09.** argmind's own judgement about Google
/// Drive's iPhone dialog was said as *기기 추가하시겠어요?* and the click went ahead beside it, with
/// 스스로 실행 on; only the dialog having gone stopped it. *…시겠어요?*, *…시겠습니까?*, *…할래요?*
/// and the English *shall I / would you like* are offers too. A question that offers nothing —
/// *무엇을 찾으세요?* — still does not hold an action, since a reply that did what was asked and asks
/// what next is not waiting for a yes.
pub fn asks_first(say: &str) -> bool {
    const OFFERS: [&str; 4] = ["까요", "시겠어요", "시겠습니까", "시겠나요"];
    const OFFERS_EN: [&str; 6] = ["shall i", "should i", "would you like", "do you want me", "want me to", "may i"];
    //: **Japanese and Chinese, since 2026-11-17**, with the languages a person can set: *…ましょうか？*
    //: (*開きましょうか* — shall I open it), *…てもいいですか* (may I), *…ませんか* (shall we — which
    //: also catches *知りませんか？*, a row where none was needed, the side chosen), and *要我 / 需要我
    //: / 要不要 / 我来 / 我帮你*. Not *…ますか？* or *好吗*, which ask *what will you do* and *how are
    //: you* as often as they offer.
    const OFFERS_JA: [&str; 4] = ["ましょうか", "てもいいですか", "てもよろしいですか", "ませんか"];
    const OFFERS_ZH: [&str; 5] = ["要我", "需要我", "要不要", "我来", "我帮你"];
    let s = say.trim().trim_end_matches(|c: char| c == '!' || c == '！' || c.is_whitespace());
    let low = s.to_lowercase();
    //: *…ㄹ래요?* — 할래요, 열래요, 닫을래요 — read off the syllable's final ㄹ, since *그래요?* ends
    //: the same two letters and offers nothing.
    let chars: Vec<char> = s.chars().collect();
    let will = chars.windows(3).any(|w| w[1] == '래' && w[2] == '요'
        && ('가'..='힣').contains(&w[0]) && (w[0] as u32 - '가' as u32) % 28 == 8);
    //: A full-width mark is how Chinese and Japanese end a question, and a Korean keyboard makes it too.
    (s.ends_with('?') || s.ends_with('？'))
        && (will || OFFERS.iter().chain(&OFFERS_JA).chain(&OFFERS_ZH).any(|o| s.contains(o))
            || OFFERS_EN.iter().any(|o| low.contains(o)))
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

/// **A bare go-ahead** — *실행해줘*, *응*, *해줘* — that names nothing to do. On its own it is not a
/// request, it is a yes to whatever was last offered.
pub fn goes_ahead(text: &str) -> bool {
    let t: String = text.split_whitespace().collect::<String>()
        .trim_end_matches(['.', '!', '~', '?']).to_string();
    const YES: [&str; 22] = ["실행해줘", "실행해", "실행", "해줘", "해", "응", "네", "예", "그래", "좋아", "진행해",
                             "진행해줘", "진행", "ㅇㅇ", "오케이", "ok", "OK", "그렇게해줘", "그렇게해", "부탁해",
                             "응해줘", "네해주세요"];
    YES.contains(&t.as_str()) || t == "해주세요" || t == "실행해주세요" || t == "진행해주세요"
        || goes_ahead_in_other_tongues(text)
}

/// **A yes in English, Chinese and Japanese** — since 2026-11-17, the same kind of word as the
/// Korean: a go-ahead that names nothing. Read whole, spaces and the closing mark taken off, so
/// *go ahead!* is one and *go ahead and open Notepad* is not. *いいよ* is left out: it says *no
/// thanks* as often as *go on*.
fn goes_ahead_in_other_tongues(text: &str) -> bool {
    use crate::tongue::{lower, reads_as};
    let t: String = lower(text).split_whitespace().collect::<Vec<_>>().join(" ")
        .trim_end_matches(['.', '!', '~', '?', '。', '！', '？', '～']).trim().to_string();
    if reads_as(text, "en") {
        const YES: [&str; 22] = ["yes", "yeah", "yep", "okay", "sure", "go ahead", "do it", "please do", "yes please",
                                 "run it", "go", "go on", "proceed", "please", "sounds good", "let's do it",
                                 "go for it", "yes, do it", "yes do it", "ok, do it", "ok do it", "carry on"];
        return YES.contains(&t.as_str());
    }
    let t: String = t.split_whitespace().collect();
    if reads_as(text, "zh") {
        const YES: [&str; 18] = ["好", "好的", "好啊", "可以", "行", "执行", "执行吧", "去吧", "做吧", "是的", "嗯",
                                 "对", "开始吧", "继续", "请执行", "去做吧", "来吧", "好，做吧"];
        return YES.contains(&t.as_str());
    }
    if reads_as(text, "ja") {
        const YES: [&str; 15] = ["はい", "うん", "お願い", "お願いします", "やって", "やってください", "実行して",
                                 "実行", "どうぞ", "ええ", "オッケー", "進めて", "頼む", "よろしく", "はい、お願いします"];
        return YES.contains(&t.as_str());
    }
    false
}

/// **What a go-ahead says yes to**: the person's request just before argmind's last reply, when
/// that reply asked (*…할까요?*). The owner, 2026-11-09: *작업 화면에서 문자표를 열어서…* was answered
/// *작업 화면에서 실행할까요?*, then *실행해줘* started a task whose whole request was *실행해줘* —
/// it found an old `box.py` in the work folder and made the box again. `None` when the last reply
/// asked nothing, so a yes with nothing behind it starts nothing.
pub fn offer_answered(history: &[(String, String)]) -> Option<String> {
    let mut lines = history.iter().rev();
    let (_, reply) = lines.by_ref().find(|(who, _)| who == "you")?;
    if !asks_first(reply) {
        return None;
    }
    lines.take_while(|(who, _)| who != "you").find(|(who, _)| who == "them").map(|(_, t)| t.clone())
}

/// **How sure the model was of the action part of a reply** — the product of the probabilities of
/// every token after `"do":`, which is the action given what was already said. The reply shape puts
/// `say` first, so this is *how sure of the click, having decided what to say*.
///
/// `None` when there is no action — `null`, `[]`, or no `do` at all — or no probabilities. Pure, so
/// it can be checked without a model.
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
    fn a_go_ahead_answers_the_offer_before_it() {
        for yes in ["실행해줘", "응", "해 줘", "진행해!", "네.", "실행해주세요"] {
            assert!(super::goes_ahead(yes), "{yes}");
        }
        for not in ["메모장 실행해줘", "해줘야 할 게 뭐야", "응 그런데 그거 말고", "고마워"] {
            assert!(!super::goes_ahead(not), "{not}");
        }
        let h = |pairs: &[(&str, &str)]| pairs.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect::<Vec<_>>();
        //: The owner's case, word for word.
        let asked = h(&[("them", "작업 화면에서 문자표(charmap)를 열어서 ★ 문자를 골라 '복사할 문자' 칸에 넣어줘"),
                        ("you", "문자표(charmap)를 열고 ★ 문자를 '복사할 문자' 칸에 복사해드릴게요. 작업 화면에서 실행할까요?")]);
        assert_eq!(super::offer_answered(&asked).as_deref(),
                   Some("작업 화면에서 문자표(charmap)를 열어서 ★ 문자를 골라 '복사할 문자' 칸에 넣어줘"));
        //: Lines in between — what was offered on screen — do not hide the request.
        let between = h(&[("them", "검색해줘"), ("you", "찾아볼까요?"), ("offered", "click …")]);
        assert_eq!(super::offer_answered(&between).as_deref(), Some("검색해줘"));
        //: A reply that asked nothing, or no history, offers nothing.
        assert_eq!(super::offer_answered(&h(&[("them", "안녕"), ("you", "안녕하세요!")])), None);
        assert_eq!(super::offer_answered(&[]), None);
        //: A noticed remark that asked, with nobody's request before it since the last reply.
        assert_eq!(super::offer_answered(&h(&[("you", "도와드릴까요?"), ("noticed", "창이 바뀜"), ("you", "업데이트할까요?")])), None);
    }

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
        for q in ["먼저 항공권과 호텔을 찾아볼까요?", "검색해 볼까요? ", "이걸로 할까요?!",
                  //: argmind's own words beside a click it went ahead with, 2026-11-09.
                  "Google Drive에 'Apple iPhone' 기기 추가 창이 보입니다. 기기 추가하시겠어요?",
                  "저장하시겠습니까?", "지금 닫을래요?", "같이 열래요?", "Shall I open it?", "Would you like me to add the device?",
                  "メモ帳を開きましょうか？", "検索してもいいですか?", "一緒に見ませんか？", "要我帮你打开记事本吗？",
                  "需要我搜索一下吗?", "要不要打开计算器？", "열까요？"] {
            assert!(super::asks_first(q), "{q}");
        }
        for s in ["검색했습니다", "무엇을 찾으세요?", "열었어요. 더 볼까요 말까요는 나중에", "",
                  "기기를 추가하시겠어요", "괜찮으세요?", "그래요?", "What next?", "I opened it. Anything else?",
                  "何をしますか？", "メモ帳を開きました。", "你好吗？", "我打开了记事本。", "开吗"] {
            assert!(!super::asks_first(s), "{s}");
        }
    }
}
