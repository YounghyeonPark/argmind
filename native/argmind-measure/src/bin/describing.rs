//! Can a picture become a sentence fast enough to be worth having?
//!
//! ```text
//!   describing            4 rounds, alternating the vision model and the text one
//!   describing 1          one round, while working
//! ```
//!
//! **The owner, 2026-09-17:** 카메라를 fast 하게 vtt해서 text description을 llm에 넣으면
//! 해석할수 있을까?
//!
//! ## Why the old number does not answer it
//!
//! This project's reason for keeping vision out is a measurement — *a vision model took 24–82 s for
//! a pixel and on a dense screen never answered* — and **it does not transfer.** That was asking for
//! a coordinate among two hundred small labels. A caption of a webcam frame is a different task: no
//! precision, no coordinates, one scene. Citing it here would be using a number outside the
//! conditions it carries, which this repository has a rule against.
//!
//! It also does not break the rule everything hangs on. *The deterministic side says WHERE, the
//! probabilistic side says WHAT* — a description is a WHAT. `camera.rs` already says the same thing
//! from the other side: it hands out a scalar and never a region, because *where* the difference is
//! would be a coordinate from a detector.
//!
//! ## Two guesses, both wrong, and what it actually is
//!
//! **The first guess was video memory.** `qwen3:8b` holds 6.3 GB of a 16 GB card and the neural
//! voice holds more, so a second 6.4 GB model looked like it would evict the first and make every
//! glance cost two model loads. **It does not.** Both sit on the GPU at once — 12.7 GB of 16 — and
//! the answer after a look takes **0.2 s**, which is a text model that never moved.
//!
//! **The second guess was resolution**, and this file was built to sweep it. `prompt_tokens` is
//! **1110 at every size**, from 640×360 down to 80×46: ollama normalises the image to a fixed patch
//! budget and the source resolution buys nothing. The sweep is kept because that is the finding.
//!
//! **It is the model thinking.** Time tracks output tokens at 16–18 ms each, and `thought` is
//! 467–1690 characters of reasoning about a test card. `qwen3-vl:8b` on ollama **ignores
//! `think: false`** — measured three ways: with the field false, with the field absent, and with
//! `/no_think` in the prompt, which made it 25% *worse*. So roughly 85% of a look is the model
//! thinking about four coloured squares, and there is no switch here that stops it.
//!
//! Every size is timed in the shape argmind would use it — describe, then answer — and the first
//! round of each is reported apart from the rest, because a cold load and a steady-state cost are
//! different claims.
//!
//! ## No camera is opened
//!
//! The picture is drawn here, in code — see [`test_card`]. That is deliberate: the question at this
//! stage is *how long does a picture take to cross this pipeline*, which a synthetic image answers
//! exactly as well as a photograph, and it needs nobody's room. **Whether the description is any
//! good is the next measurement and it is not this one** — it needs a real camera, a person
//! deciding what the right answer was, and a permission that does not exist yet.

use std::path::PathBuf;

use argmind_env_ollama::Local;
use argmind_measure::{test_card, Destination};
use serde_json::json;

/// The vision model, already on this machine and **Apache-2.0**, which the licence rule requires.
pub const SEES: &str = "qwen3-vl:8b";
/// The one that answers, and the one a look would be evicting.
pub const SPEAKS: &str = "qwen3:8b";

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let rounds: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(4);
    let dir = root.join("measure");

    let out = Destination::new(&dir, "describing",
                               &[SEES.replace(':', "-"), SPEAKS.replace(':', "-"),
                                 "sizes".into(), format!("x{rounds}")]);
    if let Err(why) = out.refuse_if_taken() {
        println!("\n  ** {why}");
        std::process::exit(3);
    }

    //: **Four sizes an octave apart**, so a cost that scales with pixels and one that does not are
    //: told apart by the shape of the column rather than by two points and a hope. The smallest is
    //: near `camera.rs`'s own 64×36, which is what a presence check already gets by.
    let sizes: [(usize, usize); 4] = [(640, 360), (320, 180), (160, 90), (80, 46)];

    println!("\n  {SEES} describing, {SPEAKS} answering | {rounds} rounds at each size");
    println!("  a drawn test card | no camera is opened\n");

    let seeing = Local { keep_alive: "300s".into(), think: Some(false), ..Local::new(SEES) };
    let speaking = Local { keep_alive: "300s".into(), think: Some(false), ..Local::new(SPEAKS) };

    let mut rows = Vec::new();
    println!("  {:>9}  {:>5}  {:>9}  {:>8}  {:>6}  {:>5}   {}",
             "size", "round", "describe", "answer", "in", "out", "thought");
    for (w, h) in sizes {
        let card = test_card(w, h);
        for round in 0..rounds {
            let t0 = std::time::Instant::now();
            //: **Tokens in and out, kept.** The first sweep was non-monotone in size — 80x46 slower
            //: than 160x90 — and without these there is no telling an image that cost more to read
            //: from a caption that simply ran on longer. A time with no denominator is the thing
            //: this directory exists to refuse.
            let got = seeing.look_at(
                "이 사진에 무엇이 있는지 한 문장으로 말하라. 색과 배치를 포함하라.", &card);
            let describe_s = t0.elapsed().as_secs_f64();
            let (into, outof, thought) = got.as_ref().ok()
                .map(|r| (r.prompt_tokens, r.out_tokens, r.thought)).unwrap_or((None, None, 0));
            let seen = got.map(|r| r.text).unwrap_or_else(|e| format!("(실패: {e})"));

            //: **The answer immediately after**, because a description is only worth its time if
            //: something can be said off it, and that second call is where an eviction would show.
            let t1 = std::time::Instant::now();
            let answered = speaking.ask(&format!(
                "카메라가 본 것: {seen}\n\n이 사람이 자리에 있는 것 같나? 한 문장으로."))
                .map(|r| r.text).unwrap_or_else(|e| format!("(실패: {e})"));
            let answer_s = t1.elapsed().as_secs_f64();

            let one_line: String = seen.replace('\n', " ").chars().take(22).collect();
            println!("  {:>4}x{:<4}  {round:>5}  {describe_s:>8.1}s  {answer_s:>7.1}s  {:>6}  \
                      {:>5}  {thought:>5}   {one_line}", w, h,
                     into.map(|n| n.to_string()).unwrap_or_else(|| "?".into()),
                     outof.map(|n| n.to_string()).unwrap_or_else(|| "?".into()));
            rows.push(json!({"w": w, "h": h, "png_bytes": card.len(), "round": round,
                             "describe_s": describe_s, "answer_s": answer_s,
                             "prompt_tokens": into, "out_tokens": outof, "thought_chars": thought,
                             "described": seen, "answered": answered}));
        }
    }

    //: **The first round of each size apart from the rest.** A cold load and a steady-state cost
    //: are different claims and one mean over both is neither.
    println!("\n  {:>9}  {:>10}  {:>12}  {}", "size", "first", "after, mean", "worst");
    for (w, h) in sizes {
        let at: Vec<&serde_json::Value> = rows.iter()
            .filter(|r| r["w"] == w && r["h"] == h).collect();
        let whole = |r: &serde_json::Value|
            r["describe_s"].as_f64().unwrap() + r["answer_s"].as_f64().unwrap();
        let after: Vec<f64> = at.iter().skip(1).map(|r| whole(r)).collect();
        //: **One round is one round.** A mean and a worst over nothing printed as `NaN` and as
        //: `-1.8e308`, which is a table saying something false rather than saying nothing.
        if after.is_empty() {
            println!("  {:>4}x{:<4}  {:>9.1}s  {:>12}  {:>7}", w, h, whole(at[0]),
                     "(one round)", "");
            continue;
        }
        let mean = after.iter().sum::<f64>() / after.len() as f64;
        let worst = after.iter().cloned().fold(f64::MIN, f64::max);
        println!("  {:>4}x{:<4}  {:>9.1}s  {:>11.1}s  {:>6.1}s", w, h, whole(at[0]), mean, worst);
    }
    println!("\n  **That is the price of one camera glance**, a look and the answer off it.");

    std::fs::write(&out.path, serde_json::to_string_pretty(&json!({
        "sees": SEES, "speaks": SPEAKS, "rounds": rounds,
        "sizes": sizes.iter().map(|(w, h)| json!([w, h])).collect::<Vec<_>>(),
        "camera": false, "rows": rows})).unwrap()).unwrap();
    println!("\n  every round -> {}", out.path.display());
}
