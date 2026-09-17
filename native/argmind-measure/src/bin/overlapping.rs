//! Does running a sense beside the answer make the answer slower?
//!
//! ```text
//!   overlapping            10 answers alone, then 10 with a look running beside them
//!   overlapping 4          four of each, while working
//! ```
//!
//! **The owner, 2026-09-17:** 각각의 인지를 병렬로 구동하고, 계속 리프레시하면 생각하는 부분이
//! 그때그때 처리된걸 읽어오면 더 빠르지 않을까.
//!
//! It is the right shape and for the camera it is close to mandatory: a look costs about 8 s, and
//! on the one thread that owns the session that is 8 s in which nothing answers anybody. A sense
//! that refreshes on its own and leaves its latest reading somewhere the thinking can pick up is
//! how that stops being true.
//!
//! **But nothing gets faster by being parallel.** A look still takes 8 s; it stops *blocking*. The
//! thing that could actually change is the number the owner cares about most — *중요한건 상호작용
//! 및 의사소통임* — because a look and an answer are two jobs on one graphics card, and a queue is
//! a queue whichever thread put the work in it.
//!
//! So the question this measures is not *is parallel faster*. It is **what does a sense running
//! beside the conversation cost the conversation**, which is the half of the proposal that can lose.
//!
//! ## What it does
//!
//! Two arms, same text prompts, same model, worst and median reported:
//!
//! * **alone** — the answer with nothing else running. This is the baseline the shipped agent has.
//! * **beside a look** — the same answers while a second thread keeps `qwen3-vl:8b` busy on a drawn
//!   picture, back to back, which is the heaviest version of the proposal.
//!
//! A third fact falls out for free: whether the two overlap at all. If ollama serialises them, the
//! answers do not merely slow down, they wait for whole looks, and the difference is unmissable.
//!
//! **No camera is opened.** The picture is drawn, as in `describing`.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use argmind_env_ollama::Local;
use argmind_measure::Destination;
use serde_json::json;

/// The same two models `describing` used, and the same reason: both are already here and the
/// vision one is Apache-2.0.
pub const SEES: &str = "qwen3-vl:8b";
pub const SPEAKS: &str = "qwen3:8b";

/// Eight things to answer, short enough that the reply is not the cost.
const ASK: [&str; 8] = ["안녕", "지금 몇 시야?", "고마워", "뭐 하고 있어?",
                        "오늘 날씨 어때?", "잠깐만", "잘 했어", "이거 어떻게 해?"];

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// One arm: `rounds` answers, timed.
fn answers(model: &Local, rounds: usize) -> Vec<f64> {
    (0..rounds).map(|i| {
        let t0 = std::time::Instant::now();
        let _ = model.ask(ASK[i % ASK.len()]);
        t0.elapsed().as_secs_f64()
    }).collect()
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let rounds: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(10);
    let dir = root.join("measure");

    let out = Destination::new(&dir, "overlapping",
                               &[SEES.replace(':', "-"), SPEAKS.replace(':', "-"),
                                 format!("x{rounds}")]);
    if let Err(why) = out.refuse_if_taken() {
        println!("\n  ** {why}");
        std::process::exit(3);
    }

    let speaking = Local { keep_alive: "300s".into(), think: Some(false), ..Local::new(SPEAKS) };
    println!("\n  {SPEAKS} answering, {SEES} looking | {rounds} answers per arm | no camera\n");

    //: Warm, because a cold load in the first arm would be charged to the arm rather than to the
    //: load — the same two throwaway calls every harness here makes.
    for _ in 0..2 {
        let _ = speaking.ask("Reply with the single word: ready");
    }

    let alone = answers(&speaking, rounds);
    println!("  alone            median {:.2}s   worst {:.2}s",
             median(alone.clone()), alone.iter().cloned().fold(0.0, f64::max));

    //: **A second thread keeping the vision model busy back to back.** The heaviest form of the
    //: proposal: not a look every thirty seconds but a look always in flight.
    let stop = Arc::new(AtomicBool::new(false));
    let looks = Arc::new(AtomicU64::new(0));
    let looking = {
        let (stop, looks) = (Arc::clone(&stop), Arc::clone(&looks));
        std::thread::spawn(move || {
            let card = argmind_measure::test_card(320, 180);
            let seeing = Local { keep_alive: "300s".into(), think: Some(false),
                                 ..Local::new(SEES) };
            while !stop.load(Ordering::SeqCst) {
                let _ = seeing.look_at("이 사진에 무엇이 있는지 한 문장으로 말하라.", &card);
                looks.fetch_add(1, Ordering::SeqCst);
            }
        })
    };
    //: Let one look get properly under way, so the first answer of this arm meets a busy card
    //: rather than an idle one.
    std::thread::sleep(std::time::Duration::from_secs(3));

    let beside = answers(&speaking, rounds);
    stop.store(true, Ordering::SeqCst);
    let _ = looking.join();

    println!("  beside a look    median {:.2}s   worst {:.2}s   ({} looks finished meanwhile)",
             median(beside.clone()), beside.iter().cloned().fold(0.0, f64::max),
             looks.load(Ordering::SeqCst));

    let (m0, m1) = (median(alone.clone()), median(beside.clone()));
    println!("\n  the answer costs {:.1}x as much beside a look", m1 / m0.max(1e-9));
    //: **The unmissable case, named before the run.** If ollama serialises the two models, an
    //: answer does not slow down a little, it waits for a whole look — so a median anywhere near
    //: the 8 s a look takes means they are not overlapping at all.
    if m1 > 4.0 {
        println!("  **They are not overlapping.** An answer is waiting out a whole look, which is\n  \
                  ollama queueing rather than two jobs sharing a card.");
    }

    std::fs::write(&out.path, serde_json::to_string_pretty(&json!({
        "sees": SEES, "speaks": SPEAKS, "rounds": rounds, "asked": ASK,
        "alone_s": alone, "beside_s": beside,
        "looks_finished": looks.load(Ordering::SeqCst)})).unwrap()).unwrap();
    println!("\n  every answer -> {}", out.path.display());
}
