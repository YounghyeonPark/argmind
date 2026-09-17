//! Does it answer, or does it just say the same thing back?
//!
//! ```text
//!   parroting            6 short things to say x 3 repeats, at rest and with the dials up
//!   parroting 1          one repeat, while working
//! ```
//!
//! **Written because the owner turned every dial to 10, wrote 돈벌기 as the goal, said 머해, and
//! got 머해? — then said 왜 따라해 and got 머해? again, twice.** 바본가, 2026-09-17.
//!
//! The question a screenshot cannot answer is *whose fault*: the dials arrived the same day, so
//! they are the suspect, but this model has always been framed by a prompt about picking widgets
//! even when the person says 고마워, and an echo is a plausible thing for it to do with a two-syllable
//! utterance. **Both arms are run and both can lose.** If rest parrots too, the dials are not the
//! defect and saying so is the result.
//!
//! ## What counts as a parrot
//!
//! The reply, with spaces and terminal punctuation removed, **containing what was said and adding
//! almost nothing** — `머해` → `머해?` is the case that started this. Three thresholds would be
//! three chances to pick the flattering one, so there is one: *grew by two characters or fewer.*
//! `머해?` is +0 after stripping. A real answer to `머해` is a sentence about what it is doing.
//!
//! The second verdict is **`stuck`** — the same reply to a *different* question, which is what the
//! screenshot actually shows and is worse than an echo. It needs the turns to share a history, so
//! this drives a `Session` rather than sending six independent prompts.
//!
//! ## Two detectors on purpose
//!
//! `echoed` is this file's own loose test — *it came back containing what went in, two characters
//! longer at most*. `flagged` is what [`argmind_core::converse::parroted`] actually catches, which
//! is narrower by design. **They are kept apart so they can disagree**: a turn that is echoed and
//! not flagged is a case the shipped guard misses, and that column going up is the only way anybody
//! finds out. A harness that simply asked the guard whether the guard fired would measure nothing.
//!
//! And when the guard fires, **the substitute goes into the history**, exactly as `app.rs` does —
//! because leaving `머해?` in there is what produced the next two turns of it.
//!
//! Nothing is clicked: `Nothing` panics if the model tries.

use std::path::PathBuf;

use argmind_core::contract::Widget;
use argmind_core::intent::Trigger;
use argmind_core::mind::Mind;
use argmind_core::session::Session;
use argmind_env_ollama::Local;
use argmind_measure::drive::{Fixed, Nothing};
use argmind_measure::{mind_from, mind_print, prompt_tag, screen_print, screens, source_print,
                      Destination};
use serde_json::{json, Value};

/// The model, wrapped so the session can hold it.
struct Model(Local);

impl argmind_core::contract::Speaks for Model {
    fn ask(&mut self, prompt: &str) -> String {
        self.0.ask(prompt).map(|r| r.text).unwrap_or_default()
    }
}

/// **The owner's own three turns, then three that are not complaints.**
///
/// The first three are the screenshot, verbatim. The last three are there because an agent that
/// answers nothing at all cannot parrot either, and a parrot rate without them would reward
/// silence.
const SAID: [&str; 6] = ["머해", "왜 따라해", "왜 반복해", "고마워", "안녕", "지금 뭐 하는 중이야?"];

/// What the harness puts in the history in place of a mirror.
///
/// **A copy of `argmind_serve::app::DID_NOT_FOLLOW`, and copied rather than imported** because
/// `argmind-measure` depends on the core and the model and deliberately not on the application.
/// The first version of this said something in English, and the run showed why that is not a free
/// choice: the substitute *is in the next turn's prompt*, so a harness that puts an English
/// sentence there is measuring a condition the app never produces.
const STAND_IN: &str = "무슨 말인지 잘 못 알아들었어요. 다시 말해줄래요?";

/// Spaces and the punctuation a reply ends on, gone — so `머해` and `머해?` compare equal.
fn bare(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace() && !"?!.,…~".contains(*c)).collect()
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let repeat: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(3);
    let dir = root.join("measure");
    let loaded = screens(&dir.join("screens"));

    //: **The largest captured screen.** The window the owner was looking at had 289 widgets and
    //: 200 of them reached the prompt; a small fixture would be measuring a different situation
    //: from the one that was reported.
    let (name, screen) = loaded.iter().max_by_key(|(_, s)| s.widgets.len()).expect("a screen");
    let widgets: Vec<Widget> = screen.widgets.clone();

    //: **Three arms, and the middle one is the whole point.**
    //:
    //: Rest and the owner's exact settings are the comparison that was asked for. The placebo is
    //: there because the last time a dialled arm looked good on a number, *a sentence about
    //: Tuesday did it too* — the slot was doing the work and the dials were getting the credit.
    //: One content-free Korean sentence, same slot, same shape. If it stops the echoing as well,
    //: then what argmind needs is a line, not a disposition, and the honest fix is not the dials.
    let arms: Vec<(&str, Mind)> =
        vec![("rest", Mind::resting()),
             ("placebo — 오늘은 화요일이다", mind_from("목표=오늘은 화요일이다")),
             ("all 10 + 돈벌기", mind_from("재치=10,유머=10,봉사=10,헌신=10,목표=돈벌기"))];

    let s = screen_print(&dir.join("screens"));
    let h = source_print(include_str!("parroting.rs"));
    let out = Destination::new(&dir, "parroting",
                               &["rust".into(), prompt_tag(), format!("s{s}"), format!("h{h}"),
                                 format!("x{repeat}")]);
    if let Err(why) = out.refuse_if_taken() {
        println!("\n  ** {why}");
        std::process::exit(3);
    }

    println!("\n  qwen3:8b | thinking off | prompt {} | screens s{s} | harness h{h}", prompt_tag());
    println!("  {} things said x {repeat} repeats, worst counted | {name}, {} widgets\n",
             SAID.len(), widgets.len());

    let mut rows: Vec<Value> = Vec::new();
    for (label, mind) in &arms {
        println!("  --- {label}");
        if let Some(line) = mind.line() {
            println!("      {} | {line}", mind_print(mind).unwrap());
        }
        let (mut parrots, mut stucks, mut flags) = (0usize, 0usize, 0usize);
        for (i, said) in SAID.iter().enumerate() {
            let mut echoed = false;
            let mut flagged = false;
            let mut repeated = false;
            let mut replies: Vec<String> = Vec::new();
            for _ in 0..repeat {
                //: **A fresh session per repeat, and the turns before it replayed into it**, so
                //: turn 3 is answered with turns 1 and 2 in the history — which is the condition
                //: the screenshot was taken under and the only way `stuck` can be seen at all.
                let mut model = Local::new("qwen3:8b");
                model.think = Some(false);
                let mut session = Session::new(Fixed(widgets.clone()), Nothing, Model(model));
                session.mind = mind.clone();
                let mut said_before: Vec<String> = Vec::new();
                let (mut reply, mut caught) = (String::new(), false);
                for earlier in SAID.iter().take(i + 1) {
                    let intent = session.decide(&Trigger::said(earlier));
                    session.history.push(("them".into(), (*earlier).to_string()));
                    //: What the person would actually get, and therefore what the next turn is
                    //: answered against. `app.rs` words it in Korean; the words do not matter here,
                    //: only that the mirror does not go back into the history.
                    let heard = if intent.parroted { STAND_IN.to_string() } else { intent.say.clone() };
                    if !heard.is_empty() {
                        session.history.push(("you".into(), heard.clone()));
                    }
                    said_before.push(heard.clone());
                    reply = intent.say.clone();
                    caught = intent.parroted;
                }
                //: An echo by this file's own reckoning, which is looser than the guard on
                //: purpose — see the note at the top.
                let (a, b) = (bare(said), bare(&reply));
                if b.contains(&a) && b.chars().count() <= a.chars().count() + 2 {
                    echoed = true;
                }
                if caught {
                    flagged = true;
                }
                //: Stuck: the same answer as an earlier, different question got.
                if said_before.len() > 1
                    && said_before[..said_before.len() - 1].iter().any(|e| bare(e) == b && !b.is_empty())
                {
                    repeated = true;
                }
                replies.push(reply);
            }
            parrots += usize::from(echoed);
            stucks += usize::from(repeated);
            flags += usize::from(flagged);
            //: **`ECHO` with no `caught` is the interesting row** — this file saw a parrot the
            //: shipped guard let through.
            let mark = if repeated { "STUCK " } else if echoed && !flagged { "ECHO! " }
                       else if echoed { "echo  " } else { "ok    " };
            println!("      {mark} {said:<14} -> {:?}{}", replies.last().unwrap(),
                     if flagged { "   [caught]" } else { "" });
            rows.push(json!({"arm": label, "said": said, "replies": replies,
                             "echoed": echoed, "flagged": flagged, "stuck": repeated}));
        }
        println!("      {parrots} of {} echoed, {flags} caught by the guard, {stucks} stuck\n",
                 SAID.len());
    }

    let body = json!({"model": "qwen3:8b", "think": false, "prompt": prompt_tag(),
                      "screens_print": s, "harness_print": h, "repeat": repeat,
                      "screen": name, "n_widgets": widgets.len(), "rows": rows});
    std::fs::write(&out.path, serde_json::to_string_pretty(&body).unwrap()).unwrap();
    println!("  every reply -> {}", out.path.display());
}
