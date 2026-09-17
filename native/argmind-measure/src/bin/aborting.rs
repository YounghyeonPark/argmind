//! When a reader walks away, does the card go free?
//!
//! ```text
//!   aborting            4 rounds of three arms
//!   aborting 1          one round, while working
//! ```
//!
//! **The question this settles.** Running senses beside the conversation costs the conversation
//! about three times its median and, once in ten, a whole look — 10.81 s, measured in
//! `overlapping`. Declining to *start* a look while somebody is being answered fixes the median.
//! Only abandoning a look **already in flight** can fix that tail, and abandoning is nothing more
//! than dropping the socket. So: **does ollama stop generating for a reader that has gone, or does
//! it finish the job into a closed pipe?**
//!
//! If it stops, the tail is fixable and `Local::ask_while` is the fix. If it does not, the third
//! rule can only ever be *do not start*, one answer in four stays slow while a sense is running,
//! and the honest thing is to say so rather than ship a cancel button that cancels nothing.
//!
//! ## Three arms, and the two controls are the measurement
//!
//! Each round starts a look and then times three short answers. What differs is when the answers
//! begin and what happened to the look:
//!
//! | arm | the look | the answers begin |
//! |---|---|---|
//! | **after** | runs to completion | once it is done |
//! | **abandoned** | dropped after 2 s | immediately after the drop |
//! | **during** | left running | 2 s in, while it works |
//!
//! **`after` and `during` are what make `abandoned` readable.** On its own a number for `abandoned`
//! says nothing: it has to be compared with a free card and a busy one, and those two are the arms.
//! `abandoned` near `after` means the drop freed the card. `abandoned` near `during` means the
//! server kept going and the drop bought nothing.
//!
//! Written before the run, so it cannot be fitted afterwards: **I do not know which.** Ollama
//! streams over HTTP with `Connection: close`, and whether it notices a vanished reader between
//! tokens or only when it next writes a large buffer is not something this repository has measured.
//!
//! No camera is opened; the picture is drawn, as in `describing`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use argmind_env_ollama::Local;
use argmind_measure::{test_card, Destination};
use serde_json::json;

pub const SEES: &str = "qwen3-vl:8b";
pub const SPEAKS: &str = "qwen3:8b";

/// How long to let a look run before walking away from it. **Two seconds** — long enough that the
/// model is certainly generating, short enough that a look of about 8 s has most of its work left.
pub const DROP_AFTER: f64 = 2.0;

const ASK: [&str; 3] = ["안녕", "고마워", "뭐 하고 있어?"];

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Three short answers, timed, one after another.
fn answers(model: &Local) -> Vec<f64> {
    ASK.iter().map(|q| {
        let t0 = std::time::Instant::now();
        let _ = model.ask(q);
        t0.elapsed().as_secs_f64()
    }).collect()
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let rounds: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(4);
    let dir = root.join("measure");

    let out = Destination::new(&dir, "aborting",
                               &[SEES.replace(':', "-"), SPEAKS.replace(':', "-"),
                                 format!("x{rounds}")]);
    if let Err(why) = out.refuse_if_taken() {
        println!("\n  ** {why}");
        std::process::exit(3);
    }

    let card = test_card(320, 180);
    let speaking = Local { keep_alive: "300s".into(), think: Some(false), ..Local::new(SPEAKS) };
    let look = "이 사진에 무엇이 있는지 한 문장으로 말하라. 색과 배치를 포함하라.";

    println!("\n  {SEES} looking, {SPEAKS} answering | {rounds} rounds | no camera");
    println!("  a look is dropped after {DROP_AFTER:.0}s in the middle arm\n");
    for _ in 0..2 {
        let _ = speaking.ask("Reply with the single word: ready");
    }

    let mut rows = Vec::new();
    println!("  {:>5}  {:>12}  {:>12}  {:>12}", "round", "after", "abandoned", "during");
    for round in 0..rounds {
        //: --- after: the look finishes, then the answers. A free card.
        let seeing = Local { keep_alive: "300s".into(), think: Some(false), ..Local::new(SEES) };
        let _ = seeing.look_at(look, &card);
        let after = answers(&speaking);

        //: --- abandoned: drop the socket two seconds in, then answer at once.
        let carry_on = Arc::new(AtomicBool::new(true));
        let abandoned_reason = {
            let (go, card, look) = (Arc::clone(&carry_on), card.clone(), look.to_string());
            let handle = std::thread::spawn(move || {
                let seeing = Local { keep_alive: "300s".into(), think: Some(false),
                                     ..Local::new(SEES) };
                seeing.ask_while(&look, &[argmind_env_ollama::base64(&card)],
                                 &|| go.load(Ordering::SeqCst), &mut |_| {})
                    .map(|r| r.reason).unwrap_or_else(|e| format!("(실패: {e})"))
            });
            std::thread::sleep(std::time::Duration::from_secs_f64(DROP_AFTER));
            carry_on.store(false, Ordering::SeqCst);
            //: **Joined before the answers start**, so the socket is provably shut rather than
            //: shutting — otherwise this arm would be measuring a race.
            handle.join().unwrap_or_else(|_| "(panicked)".into())
        };
        let abandoned = answers(&speaking);

        //: --- during: a look left running, answers two seconds in. A busy card.
        let running = Arc::new(AtomicBool::new(true));
        let looking = {
            let (go, card, look) = (Arc::clone(&running), card.clone(), look.to_string());
            std::thread::spawn(move || {
                let seeing = Local { keep_alive: "300s".into(), think: Some(false),
                                     ..Local::new(SEES) };
                let _ = seeing.ask_while(&look, &[argmind_env_ollama::base64(&card)],
                                         &|| go.load(Ordering::SeqCst), &mut |_| {});
            })
        };
        std::thread::sleep(std::time::Duration::from_secs_f64(DROP_AFTER));
        let during = answers(&speaking);
        running.store(false, Ordering::SeqCst);
        let _ = looking.join();

        println!("  {round:>5}  {:>11.2}s  {:>11.2}s  {:>11.2}s",
                 median(after.clone()), median(abandoned.clone()), median(during.clone()));
        rows.push(json!({"round": round, "after_s": after, "abandoned_s": abandoned,
                         "during_s": during, "abandoned_reason": abandoned_reason}));
    }

    let all = |k: &str| -> Vec<f64> {
        rows.iter().flat_map(|r| r[k].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
            .collect()
    };
    let (a, b, c) = (median(all("after_s")), median(all("abandoned_s")), median(all("during_s")));
    println!("\n  over every answer:");
    println!("    after a look finished   {a:.2}s");
    println!("    after abandoning one    {b:.2}s");
    println!("    while one runs          {c:.2}s");

    //: **Read against both controls, not against a threshold.** Halfway between a free card and a
    //: busy one is the interesting answer and a single cutoff would hide it.
    let gap = (c - a).abs();
    println!();
    if gap < 0.3 {
        println!("  **This round says nothing.** A busy card and a free one answered the same, so\n  \
                  the two controls did not separate and the middle arm cannot be read against them.");
    } else if (b - a).abs() < gap / 3.0 {
        println!("  **Dropping the socket frees the card.** Abandoning lands with a free card, so\n  \
                  abandoning a look in flight is worth building.");
    } else if (c - b).abs() < gap / 3.0 {
        println!("  **Dropping the socket buys nothing.** Abandoning lands with a busy card: the\n  \
                  server keeps generating for a reader that has gone. The third rule can only be\n  \
                  *do not start*, and the tail stays.");
    } else {
        println!("  **Partly.** Abandoning lands between the two, so the server gives the card up\n  \
                  but not at once. The number to build against is {b:.2}s, not {a:.2}s.");
    }

    std::fs::write(&out.path, serde_json::to_string_pretty(&json!({
        "sees": SEES, "speaks": SPEAKS, "rounds": rounds, "drop_after_s": DROP_AFTER,
        "asked": ASK, "rows": rows,
        "median": {"after": a, "abandoned": b, "during": c}})).unwrap()).unwrap();
    println!("\n  every answer -> {}", out.path.display());
}
