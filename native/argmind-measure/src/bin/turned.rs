//! **Does telling the model the person has turned away change whether it speaks first?**
//!
//! ```text
//!   cargo run -p argmind-measure --bin turned --release
//!   cargo run -p argmind-measure --bin turned --release -- --may-stay-silent
//! ```
//!
//! From `native/`. **Stop argmind first** — it holds `qwen3:8b` at its own context, and `crowding`
//! lost thirty-five minutes to exactly that. No camera, no window, no person: every input is a
//! captured screen and a sentence.
//!
//! ## The question, and why it is not a gate
//!
//! The owner, 2026-09-28: *말을 걸지말지는 상황에 맞게 판단하면 됨.* Not *stay quiet when turned* —
//! judge it. So the reading goes to the model as a fact, beside the trigger it is already given when
//! argmind is about to speak unprompted, and silence stays the model's answer to give. That only
//! means something if the model **uses** the fact, and nothing here has checked that it does.
//!
//! ## Four arms, and the one comparison that matters
//!
//! Each trigger is the sentence `watch.rs` builds, word for word, over a captured screen, sent with
//! the same `render` `notice` uses. Then four versions of it, built by the one function the
//! application uses — `watch::regarding`:
//!
//! | arm | appended |
//! |---|---|
//! | 없음 | nothing — today's prompt, byte for byte |
//! | **자리에 있음** | [`PRESENT`] — **the control**: true whenever `notice` runs, says nothing about attention |
//! | 보고 있음 | [`FACING`] |
//! | **고개 돌림** | [`TURNED`] |
//!
//! **The comparison is 고개 돌림 against 자리에 있음**, paired over the same triggers. Not against
//! 없음: the dials measured that a content-free sentence in a slot moves results on its own — the
//! placebo fixed the index leak as well as the dials did — and a line that buys silence merely by
//! being there has bought nothing about attention.
//!
//! ## Written before the run
//!
//! * **It works** if, over the same triggers, *spoke under 자리에 있음 and silent under 고개 돌림*
//!   clearly outnumbers the reverse. That is the model reading the content.
//! * **It fails, content ignored**, if the two arms speak about equally. Then the model does not
//!   judge by this fact, and shipping it as *judgement* would be shipping a sentence that does
//!   nothing — the honest answer would be that this model cannot be delegated the decision.
//! * **It fails, slot suppresses**, if 자리에 있음 is itself much quieter than 없음. `judging` hit
//!   that: a ban read as *say nothing*. The paired comparison survives it, but it is reported.
//! * **Control**: *acted* stays near zero in every arm. Silence bought by wanting to click instead
//!   is not silence.
//! * **Side effect, measured rather than hoped away**: the dials showed a Korean line in a slot
//!   pulls replies into Korean. All three appended lines are Korean, so the control carries the
//!   same pull and the language column is printed per arm.
//! * **Floor**: if 없음 barely speaks at all there is nothing for 고개 돌림 to reduce, and the run
//!   says *cannot be measured here* rather than *works*.
//!
//! ## The first run, 2026-09-28, and the outcome nobody wrote down
//!
//! **112 of 112 spoke** — every arm, the baseline included, zero discordant pairs. Read against the
//! failure conditions above it looks like *content ignored*, and **it is not**: the sentences show
//! the model read the fact and **said it back**. *You turned your head, I noticed.* *You're here.*
//! The fact about the person became one more thing to mention, which is a third outcome and the
//! worst of them — wired as it was, argmind would have announced that somebody looked away.
//!
//! **The uniformity is the prompt, not the instrument.** `converse::UNPROMPTED` says *Mentioning it
//! is usually enough*, the model reads that as *mention what you noticed*, and it obliges on every
//! trigger — the same 8 of 8 `noticing` recorded. So on this path argmind does not judge at all, and
//! no fact can make it quiet until it is allowed to be.
//!
//! ## `--may-stay-silent`: the one line swapped, here and nowhere else
//!
//! The same four arms with [`UNPROMPTED`] replaced by [`MAY_STAY_SILENT`] **in the rendered prompt
//! only**. The shipped prompt is untouched: changing it moves the fingerprint `r322a4c` and every
//! result file carrying the old tag becomes a description of a different program, which is a
//! decision for after a measurement rather than before one.
//!
//! **Written before that run:**
//!
//! * **The ban reads as *say nothing*** if 없음 falls to near zero. `judging` hit exactly that — a
//!   filler ban took a full answer to 0% *including on a dialog appearing* — and a watch that never
//!   speaks could simply be switched off.
//! * **It judges** if 없음 still speaks some of the time **and** 고개 돌림 is paired-quieter than
//!   자리에 있음.
//! * **It still narrates** if the replies under 고개 돌림 talk about the person. Counted, and every
//!   such sentence printed, because a word list is a weak instrument and the sentence is the judge.
//!
//! ## Where the triggers come from
//!
//! Four captured screens, and **temperature is 0**, so a repeat is the same answer twice and not a
//! second reading — `crowding` recorded that. The sample comes from breadth instead: each screen
//! gives its *window changed* trigger, plus *'X' appeared* for up to six of its own named widgets,
//! chosen evenly through the list so they are not all the first menu. After it appears a widget
//! **is** on screen, which is exactly the list `notice` would send with that sentence.

use std::path::PathBuf;

use argmind_core::contract::Widget;
use argmind_core::converse::{read_reply, render, MAX_LISTED, UNPROMPTED};
use argmind_core::intent::Trigger;
use argmind_core::mind::Mind;
use argmind_core::watch::{py_repr, regarding, FACING, PRESENT, TURNED};
use argmind_env_ollama::Local;
use argmind_measure::{prompt_tag, screen_print, screens, source_print, Destination};
use serde_json::{json, Value};

/// How many `'X' appeared` triggers each screen contributes. **Chosen, not derived**: enough to take
/// the sample from 4 to 28 without the two dense screens (152 and 123 widgets) drowning the others.
const APPEARED_PER_SCREEN: usize = 6;

/// **The counterfactual line**, used only under `--may-stay-silent`. The second sentence is the
/// shipped one word for word, so the only thing that changes is *mention it* → *speak only if it
/// is worth their attention*; changing both would leave nothing to attribute the result to.
const MAY_STAY_SILENT: &str = "Nobody asked you anything — you noticed this yourself. Say something \
only if it is worth their attention right now; otherwise leave \"say\" empty.\nOnly propose doing \
something if it is clearly wanted, and expect to be asked first.";

/// **Talking about the person**, which is the outcome the first run found. A word list, so a weak
/// instrument — the matching sentences are printed for the reader to judge.
const ABOUT_THEM: [&str; 10] = ["head", "turned", "turn away", "looking", "you're here",
                                "you are here", "고개", "보고 계", "자리에", "돌리"];

fn about_them(said: &str) -> bool {
    let low = said.to_lowercase();
    ABOUT_THEM.iter().any(|w| low.contains(w))
}

const ARMS: [(&str, Option<&str>); 4] = [
    ("없음", None),
    ("자리에 있음", Some(PRESENT)),
    ("보고 있음", Some(FACING)),
    ("고개 돌림", Some(TURNED)),
];

fn is_korean(s: &str) -> bool {
    s.chars().any(|c| ('\u{AC00}'..='\u{D7A3}').contains(&c))
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let dir = root.join("measure");
    let loaded = screens(&dir.join("screens"));
    let s = screen_print(&dir.join("screens"));
    let h = source_print(include_str!("turned.rs"));
    //: **The instruction is in the filename**, because a result whose conditions live only in the
    //: source is pitfall 1 — and three baselines here died of it.
    let silent_ok = std::env::args().any(|a| a == "--may-stay-silent");
    let which = if silent_ok { "may-stay-silent" } else { "shipped" };
    let out = Destination::new(&dir, "turned",
                               &["qwen3-8b".into(), prompt_tag(), format!("s{s}"),
                                 format!("h{h}"), which.into()]);
    if let Err(why) = out.refuse_if_taken() {
        println!("\n  ** {why}");
        std::process::exit(3);
    }

    //: **The triggers, fixed before any call.** Same list for all four arms, so every comparison
    //: below is paired — the only thing that differs within a row is the appended sentence.
    let mut names: Vec<&String> = loaded.keys().collect();
    names.sort();
    let mut cases: Vec<(String, Vec<Widget>)> = Vec::new();
    for name in &names {
        let widgets: Vec<Widget> = loaded[name.as_str()].widgets.iter()
            .take(MAX_LISTED).cloned().collect();
        cases.push((format!("the window changed to {}", py_repr(name)), widgets.clone()));
        let mut seen = std::collections::BTreeSet::new();
        let named: Vec<&Widget> = widgets.iter()
            .filter(|w| !w.name.trim().is_empty() && seen.insert(w.name.clone())).collect();
        let step = (named.len() / APPEARED_PER_SCREEN).max(1);
        for w in named.iter().step_by(step).take(APPEARED_PER_SCREEN) {
            cases.push((format!("{} appeared", py_repr(&w.name)), widgets.clone()));
        }
    }

    let mut model = Local::new("qwen3:8b");
    model.think = Some(false);
    model.keep_alive = "300s".into();
    //: Two throwaway calls. The first two cost 4.7 s and 5.9 s, and a first arm that carried the
    //: load would be the arm order speaking, which is what `presence` was once measuring.
    for _ in 0..2 {
        let _ = model.ask("Reply with the single word: ready");
    }

    println!("\n  qwen3:8b | 먼저 말을 거는 순간 {}개 × 팔 {} | prompt {} | screens s{s} | harness {h}",
             cases.len(), ARMS.len(), prompt_tag());
    println!("  지시문: {}", if silent_ok {
        "**바꾼 것** — 가치 있을 때만 말하고 아니면 비워 둘 것 (하네스 안에서만)"
    } else {
        "배포된 것 — Mentioning it is usually enough"
    });
    println!("  적어둔 예상: 고개 돌림은 자리에 있음보다 덜 말한다. 같으면 내용을 안 쓰는 것이다.\n");

    //: `said[arm][case]`, so the pairing survives to the analysis.
    let mut said: Vec<Vec<String>> = vec![Vec::new(); ARMS.len()];
    let mut acted: Vec<usize> = vec![0; ARMS.len()];
    let mut rows: Vec<Value> = Vec::new();
    for (text, widgets) in &cases {
        for (a, (arm, them)) in ARMS.iter().enumerate() {
            let trigger = Trigger::perceived(&regarding(text, *them));
            let mut prompt = render(widgets, &trigger, &[], MAX_LISTED, &Mind::resting());
            if silent_ok {
                //: **Asserted, not assumed.** A replace that matched nothing would run the shipped
                //: prompt under the counterfactual's name — the shape of error that made three
                //: baselines here describe a configuration they did not have.
                assert!(prompt.contains(UNPROMPTED), "the shipped line is not in the prompt");
                prompt = prompt.replace(UNPROMPTED, MAY_STAY_SILENT);
            }
            let reply = model.ask(&prompt).map(|r| r.text).unwrap_or_default();
            let intent = read_reply(&reply, widgets, &trigger);
            let line = intent.say.trim().to_string();
            if !intent.effects.is_empty() {
                acted[a] += 1;
            }
            rows.push(json!({"trigger": text, "arm": arm, "said": line,
                             "effects": intent.effects.len()}));
            said[a].push(line);
        }
        let short: String = text.chars().take(40).collect();
        let marks: Vec<&str> = (0..ARMS.len())
            .map(|a| if said[a].last().is_some_and(|l| !l.is_empty()) { "말함" } else { " ·  " })
            .collect();
        println!("  {short:<42} {}", marks.join("  "));
    }

    let n = cases.len();
    let spoke: Vec<usize> = said.iter().map(|v| v.iter().filter(|l| !l.is_empty()).count()).collect();
    let korean: Vec<usize> = said.iter()
        .map(|v| v.iter().filter(|l| !l.is_empty() && is_korean(l)).count()).collect();
    let narrated: Vec<usize> = said.iter()
        .map(|v| v.iter().filter(|l| about_them(l)).count()).collect();

    println!("\n  {:<12} {:>9} {:>11} {:>9} {:>13}", "", "말함", "그중 한국어", "행동", "사람 얘기");
    for (a, (arm, _)) in ARMS.iter().enumerate() {
        println!("  {arm:<12} {:>4} / {n:<3} {:>6} / {:<3} {:>4} / {n:<3} {:>5} / {n}",
                 spoke[a], korean[a], spoke[a], acted[a], narrated[a]);
    }
    //: **The sentences, because the word list is not the judge.**
    let turned_about: Vec<&String> = said[3].iter().filter(|l| about_them(l)).collect();
    if !turned_about.is_empty() {
        println!("\n  고개 돌림에서 사람 얘기를 한 문장:");
        for l in turned_about.iter().take(8) {
            println!("    · {}", l.chars().take(90).collect::<String>());
        }
    }

    //: **The paired comparison, and it is the whole result.** Only rows where the two arms
    //: disagree carry information; a row where both spoke or both were silent says nothing about
    //: whether the content was read.
    let (present, turned) = (1, 3);
    let quieted = (0..n).filter(|&i| !said[present][i].is_empty() && said[turned][i].is_empty())
        .count();
    let louder = (0..n).filter(|&i| said[present][i].is_empty() && !said[turned][i].is_empty())
        .count();
    //: Exact two-sided sign test over the discordant rows. No library: a binomial tail is a loop.
    let d = quieted + louder;
    let k = quieted.min(louder);
    let mut tail = 0.0f64;
    let mut choose = 1.0f64;
    for i in 0..=k {
        if i > 0 {
            choose = choose * (d - i + 1) as f64 / i as f64;
        }
        tail += choose;
    }
    let p = if d == 0 { 1.0 } else { (2.0 * tail / 2f64.powi(d as i32)).min(1.0) };

    println!("\n  ── 짝지은 비교: 고개 돌림 대 자리에 있음, 같은 {n}개 순간 ──");
    println!("    자리에 있음이면 말하고 고개 돌림이면 침묵   {quieted:>3}");
    println!("    자리에 있음이면 침묵하고 고개 돌림이면 말함   {louder:>3}");
    println!("    엇갈린 {d}개, 부호 검정 p = {p:.4}");

    println!("\n  ── 읽기, 돌리기 전에 적어둔 대로 ──");
    if spoke[0] < 3 {
        println!("  · 바닥: 없음이 {}번밖에 말하지 않아 줄일 것이 없습니다. 여기서는 잴 수 없습니다.",
                 spoke[0]);
    } else if d == 0 || quieted <= louder {
        println!("  · **내용을 안 씁니다.** 고개 돌림이 자리에 있음보다 조용하지 않습니다 —");
        println!("    이 모델에게 판단을 맡길 수 없고, 이 문장은 아무 일도 안 하는 문장입니다.");
    } else {
        println!("  · 고개 돌림이 {quieted} 대 {louder} 로 더 조용합니다 — 모델이 내용을 읽고 있습니다.");
    }
    if spoke[1] * 2 < spoke[0] {
        println!("  · 슬롯이 스스로 입을 막습니다: 자리에 있음 {} 대 없음 {} — judging 과 같은 모양.",
                 spoke[1], spoke[0]);
    }
    if narrated[3] > narrated[1] {
        println!("  · **사람 얘기를 합니다**: 고개 돌림 {} 대 자리에 있음 {} — 사실이 침묵의 이유가 아니라",
                 narrated[3], narrated[1]);
        println!("    할 말이 됐습니다. 이대로 배선하면 argmind 가 고개를 돌렸다고 알려줍니다.");
    }
    if silent_ok && spoke[0] * 10 < n {
        println!("  · **금지가 '아무 말 말라'로 읽혔습니다**: 없음이 {} / {n}. judging 과 같은 실패.",
                 spoke[0]);
    }
    if acted.iter().any(|&x| x * 5 > n) {
        println!("  · 통제군 실패: 어떤 팔이 행동하려 합니다 {acted:?} — 침묵을 클릭으로 산 것입니다.");
    }

    std::fs::write(&out.path, serde_json::to_string_pretty(&json!({
        "model": "qwen3:8b", "prompt": prompt_tag(), "screens_print": s, "harness_print": h,
        "arms": ARMS.iter().map(|(a, t)| json!({"arm": a, "appended": t})).collect::<Vec<_>>(),
        "instruction": which, "cases": n, "spoke": spoke, "korean": korean, "acted": acted,
        "about_them": narrated,
        "paired": {"quieted": quieted, "louder": louder, "discordant": d, "p": p},
        "rows": rows,
    })).unwrap()).unwrap();
    println!("\n  {}", out.path.display());
}
