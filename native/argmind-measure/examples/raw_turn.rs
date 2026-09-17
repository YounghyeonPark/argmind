//! One turn, printed whole: the prompt that went out and the bytes that came back.
//!
//! ```text
//!   cargo run -p argmind-measure --example raw_turn -- "머해"
//!   cargo run -p argmind-measure --example raw_turn -- "머해" "재치=10,목표=돈벌기"
//! ```
//!
//! **For looking at a defect, not for measuring one.** Nothing is scored and nothing is written to
//! `measure/`; it exists because the first question about a bad reply is always whether the model
//! said it or the reader mangled it — which in this repository has been the reader five times.

use std::path::PathBuf;

use argmind_core::contract::Widget;
use argmind_core::converse::{read_reply, render, MAX_LISTED};
use argmind_core::intent::Trigger;
use argmind_core::mind::Mind;
use argmind_env_ollama::Local;
use argmind_measure::{mind_from, screens};

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let said = std::env::args().nth(1).unwrap_or_else(|| "머해".into());
    let spec = std::env::args().nth(2).unwrap_or_default();
    let mind = if spec.is_empty() { Mind::resting() } else { mind_from(&spec) };

    let loaded = screens(&root.join("measure").join("screens"));
    let (name, screen) = loaded.iter().max_by_key(|(_, s)| s.widgets.len()).unwrap();
    let listed: Vec<Widget> = screen.widgets.iter().take(MAX_LISTED).cloned().collect();

    let trigger = Trigger::said(&said);
    let prompt = render(&listed, &trigger, &[], MAX_LISTED, &mind);
    let mut model = Local::new("qwen3:8b");
    model.think = Some(false);
    let raw = model.ask(&prompt).map(|r| r.text).unwrap_or_default();
    let intent = read_reply(&raw, &listed, &trigger);

    println!("screen      {name}, {} widgets, {} listed", screen.widgets.len(), listed.len());
    println!("prompt      {} lines, {} bytes", prompt.lines().count(), prompt.len());
    println!("--- the last 6 lines of the prompt");
    for line in prompt.lines().rev().take(6).collect::<Vec<_>>().into_iter().rev() {
        println!("  {line}");
    }
    println!("--- raw reply\n{raw}");
    println!("--- read as\n  say     {:?}\n  effects {}", intent.say, intent.effects.len());
}
