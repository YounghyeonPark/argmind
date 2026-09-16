//! Run the voice's build from end to end, **as if somebody had said yes.**
//!
//!     cargo run -p argmind-serve --example buildvoice        # build if it is not built
//!     cargo run -p argmind-serve --example buildvoice fresh  # delete the shed first
//!
//! Takes several minutes, downloads about a gigabyte if the weights are not already there, and
//! leaves a server listening until this process ends. **Makes no sound** — it starts the
//! synthesiser and does not ask it for anything.
//!
//! ## Why this is an example and not a test
//!
//! It compiles somebody else's program. The suite is required to need no network, no compiler and
//! no GPU, and `tests/speaker.rs` checks everything about this that can be checked without one.
//! What is left over is the half a suite cannot reach, and the honest place for it is a thing a
//! person runs on purpose.
//!
//! ## Why it does not ask
//!
//! `argmind setup` is the path with the window in it, and the window steals the foreground. That
//! is the right behaviour for a person sitting at the machine and the wrong one for a measurement
//! — running it from a session that is driving something else interrupts that session, which has
//! happened here twice. **Running this file is the yes.** It says so above the call.

use std::time::Instant;

use argmind_serve::{fix, speaker};

fn main() {
    let fresh = std::env::args().any(|a| a == "fresh");
    println!();
    println!("  소스:   {} @ {}", speaker::REPO, &speaker::COMMIT[..12]);
    println!("  받는 곳: {}", speaker::shed().display());
    println!("  Vulkan: {}", speaker::sdk().map(|p| p.display().to_string())
        .unwrap_or_else(|| "없음 — 여기서 멈춥니다".into()));
    let missing = speaker::missing_tools();
    if !missing.is_empty() {
        println!("  빌드 도구 없음: {}\n", missing.join(", "));
        return;
    }
    if speaker::sdk().is_none() {
        return;
    }
    for (what, at) in [("talker", speaker::talker_at()), ("codec", speaker::codec_at())] {
        println!("  {what}: {}", if at.is_file() { "있음" } else { "없음 — 내려받습니다" });
    }

    if fresh {
        //: **The only way to exercise the clone and the submodules.** Everything else about the
        //: sequence runs against a tree that is already there.
        println!("\n  기존 트리를 지웁니다 — 처음부터 받습니다");
        let _ = std::fs::remove_dir_all(speaker::shed());
    }
    println!("  이미 빌드됨: {}\n", speaker::ready());

    //: **This line is the approval.** Everywhere else in this program `apply` is reached only
    //: through a window and an answer; here, running the file is the answer.
    let offer = fix::Fix::BuildSpeaker {
        repo: speaker::REPO.into(),
        commit: speaker::COMMIT.into(),
        why: "측정".into(),
        minutes: speaker::MINUTES,
        built: speaker::ready(),
    };
    println!("{}\n", fix::describe(&offer).lines()
        .map(|l| format!("  {l}")).collect::<Vec<_>>().join("\n"));

    let began = Instant::now();
    let said = fix::apply(&offer, &mut |line| {
        println!("    [{:>5.0}s] {line}", began.elapsed().as_secs_f64());
    });
    let took = began.elapsed().as_secs_f64();
    match said {
        Ok(good) => println!("\n  {good} — {:.0}분 {:.0}초", took / 60.0, took % 60.0),
        Err(why) => {
            println!("\n  실패 ({:.0}초): {why}", took);
            return;
        }
    }

    //: **What the server reports about itself**, rather than what this file believes. The alias is
    //: the one thing the two ends have to agree on by hand.
    println!("  응답함: {}", speaker::answering());
    println!("  argmind가 띄운 것: {}", speaker::ours());
    println!("  빌드된 커밋: {:?}", speaker::built_at());
    println!("\n  Enter를 누르면 서버를 끕니다.");
    let mut line = String::new();
    let _ = std::io::stdin().read_line(&mut line);
}
