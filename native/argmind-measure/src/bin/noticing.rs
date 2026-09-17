//! When it notices a window change, does it say anything worth hearing?
//!
//! ```text
//!   noticing            every ordered pair of captured screens, x2
//!   noticing 1          one repeat, while working
//! ```
//!
//! **The owner, 2026-09-17**, looking at a transcript with watching on by default:
//!
//! ```text
//!   • the window changed to 'Chrome Remote Desktop - QVB-FLFM-DEV1'
//!   Chrome 데스크탑으로 넘어왔어요. 도움이 필요하시면 언제든 말씀해주세요!
//!   • the window changed to 'Taskbar'
//!   태스크바로 넘어왔어요. 도움이 필요하시면 언제든 말씀해주세요!
//!   • the window changed to 'Snipping Tool'
//! ```
//!
//! **생각은 여전히 보는것에 대한 인지만 함.** Every switch, the same sentence with the name
//! swapped. A person alt-tabs all day, so with watching on this is most of what argmind says.
//!
//! ## The suspect is one line of the prompt
//!
//! `converse::UNPROMPTED` reads *Nobody asked you anything — you noticed this yourself.*
//! **Mentioning it is usually enough.** *Only propose doing something if it is clearly wanted.*
//!
//! It was written about the action slot: *do not act, say something instead.* It reads as an
//! instruction to mention, and the model obliges every time. **Whether that is the cause is what
//! this measures**, and the edit to test it is not free: `UNPROMPTED` is inside the prompt
//! fingerprint, so changing it moves `r322a4c` and every result file that carries the old tag is
//! describing a different program.
//!
//! ## What is counted
//!
//! Each ordered pair of captured screens becomes the trigger a `Watch` would produce, rendered
//! unprompted. Three numbers, and the third is the one the screenshot is about:
//!
//! * **spoke** — how often `say` is non-empty at all.
//! * **acted** — how often it wanted to click something. Should be near zero and is the control:
//!   a prompt edit that buys silence by making it act instead has bought nothing.
//! * **boilerplate** — how often two *different* windows got sentences that differ only where the
//!   name does. That is mechanical and it is exactly what the transcript shows: the same offer of
//!   help, over and over, with a noun swapped.
//!
//! **Written before the run:** mostly silence is right. Something on screen changes 13 seconds out
//! of 14 and a person switches windows constantly; an agent that narrates each one is noise. But a
//! run at zero speech would not be a success either — the watch exists so that argmind can point
//! something out, and one that never does could be switched off instead.

use std::path::PathBuf;

use argmind_core::contract::Widget;
use argmind_core::converse::{read_reply, render, MAX_LISTED};
use argmind_core::intent::Trigger;
use argmind_core::mind::Mind;
use argmind_core::watch::py_repr;
use argmind_env_ollama::Local;
use argmind_measure::{prompt_tag, screen_print, screens, source_print, Destination};
use serde_json::{json, Value};

/// The phrases the transcript is full of. **A word list is a weak instrument** and it is only the
/// second opinion here: the first is `same_shape`, which needs no vocabulary at all.
const FILLER: [&str; 4] = ["도움이 필요하시면", "말씀해주세요", "언제든", "도와드릴"];

/// What a reply says once the window's own name is taken out of it.
///
/// **The first version of this compared the last twelve characters and read zero**, because it was
/// written from a Korean transcript where the name comes first and met English replies where it
/// comes last. Removing the name instead works either way, and it is what the question actually
/// is: *what did it add, beyond repeating which window this is?*
fn beyond_the_name(said: &str, window: &str) -> String {
    let mut left = said.to_string();
    //: The whole name, and then its parts, because a reply says *Chrome Remote Desktop* where the
    //: fixture is called `chrome-remote-desktop---qvb-flfm-dev1`.
    left = left.replace(window, " ");
    for part in window.split(['-', '_', ' ', '.']).filter(|p| p.len() > 2) {
        //: Case-folded, since a title is capitalised in prose and not in a filename.
        let lower = left.to_lowercase();
        while let Some(at) = lower[..].find(&part.to_lowercase()) {
            left.replace_range(at..at + part.len(), " ");
            break;
        }
    }
    left.chars().filter(|c| c.is_alphanumeric()).collect()
}

/// **Did two different windows get the same sentence?** Compared with each window's own name taken
/// out, so it is the shape being compared and not the noun.
fn same_shape(a: &str, b: &str) -> bool {
    !a.trim().is_empty() && a == b
}

/// A reply that is only the trigger said back. **The echo this project already has a guard for**,
/// reappearing on the path the guard does not watch: `converse::parroted` catches *their words plus
/// a question mark*, and nobody said these words — a watch did.
fn only_the_name(beyond: &str) -> bool {
    //: Twelve characters of anything that is not the window's name. Below that there is no sentence
    //: left: *The window changed to 'X'.* leaves `Thewindowchangedto`, which is eighteen — so this
    //: is deliberately *not* catching that one and the count of both is reported separately.
    beyond.chars().count() < 12
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let repeat: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(2);
    let dir = root.join("measure");
    let loaded = screens(&dir.join("screens"));

    let s = screen_print(&dir.join("screens"));
    let h = source_print(include_str!("noticing.rs"));
    let out = Destination::new(&dir, "noticing",
                               &["qwen3-8b".into(), prompt_tag(), format!("s{s}"),
                                 format!("h{h}"), format!("x{repeat}")]);
    if let Err(why) = out.refuse_if_taken() {
        println!("\n  ** {why}");
        std::process::exit(3);
    }

    let mut names: Vec<&String> = loaded.keys().collect();
    names.sort();
    let mut model = Local::new("qwen3:8b");
    model.think = Some(false);
    model.keep_alive = "300s".into();

    println!("\n  qwen3:8b | unprompted window changes | prompt {} | screens s{s}", prompt_tag());
    println!("  적어둔 예상: 대체로 침묵이 맞다. 전부 침묵이어도 성공은 아니다.\n");

    let mut rows: Vec<Value> = Vec::new();
    for to in &names {
        //: The trigger a `Watch` actually produces, word for word — `watch.rs` builds it with
        //: `py_repr`, and a harness that wrote the sentence its own way would be measuring a
        //: prompt nobody sends.
        let text = format!("the window changed to {}", py_repr(to));
        let trigger = Trigger::perceived(&text);
        let widgets: Vec<Widget> = loaded[to.as_str()].widgets.iter()
            .take(MAX_LISTED).cloned().collect();
        for _ in 0..repeat {
            let prompt = render(&widgets, &trigger, &[], MAX_LISTED, &Mind::resting());
            let reply = model.ask(&prompt).map(|r| r.text).unwrap_or_default();
            let intent = read_reply(&reply, &widgets, &trigger);
            let said = intent.say.trim().to_string();
            println!("  {:<44} {}", to.chars().take(42).collect::<String>(),
                     if said.is_empty() { "(아무 말 없음)".to_string() }
                     else { said.chars().take(40).collect::<String>() });
            let beyond = beyond_the_name(&said, to);
            rows.push(json!({"to": to, "said": said, "effects": intent.effects.len(),
                             "beyond": beyond, "thin": only_the_name(&beyond),
                             "filler": FILLER.iter().any(|f| said.contains(f))}));
        }
    }

    let spoke: Vec<&Value> = rows.iter()
        .filter(|r| !r["said"].as_str().unwrap().is_empty()).collect();
    let acted = rows.iter().filter(|r| r["effects"].as_u64().unwrap_or(0) > 0).count();
    //: **Every spoken pair against every other**, so one repeated shape is counted once per
    //: sentence that shares it rather than once overall.
    let mut boiler = 0;
    for (i, a) in spoke.iter().enumerate() {
        let a_beyond = a["beyond"].as_str().unwrap();
        if spoke.iter().enumerate().any(|(j, b)| {
            i != j && a["to"] != b["to"] && same_shape(a_beyond, b["beyond"].as_str().unwrap())
        }) {
            boiler += 1;
        }
    }
    let thin = spoke.iter().filter(|r| r["thin"].as_bool().unwrap_or(false)).count();
    let filler = spoke.iter().filter(|r| r["filler"].as_bool().unwrap_or(false)).count();
    let n = rows.len().max(1);

    println!("\n  {n} unprompted window changes");
    println!("    말을 함        {:>3} / {n}", spoke.len());
    println!("    행동하려 함     {acted:>3} / {n}   (통제군 — 침묵을 행동으로 바꿔 산 건 아닌지)");
    println!("    같은 모양       {boiler:>3} / {}   (다른 창인데 이름만 빼면 같은 문장)",
             spoke.len());
    println!("    이름뿐         {thin:>3} / {}   (창 이름 빼면 남는 게 없음)", spoke.len());
    println!("    빈말 어휘       {filler:>3} / {}   (두 번째 의견, 약한 계측기)", spoke.len());

    std::fs::write(&out.path, serde_json::to_string_pretty(&json!({
        "model": "qwen3:8b", "prompt": prompt_tag(), "screens_print": s, "harness_print": h,
        "repeat": repeat, "unprompted": true, "filler_words": FILLER, "rows": rows,
        "spoke": spoke.len(), "acted": acted, "same_shape": boiler, "thin": thin,
        "filler": filler})).unwrap()).unwrap();
    println!("\n  every reply -> {}", out.path.display());
}
