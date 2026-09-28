//! **A spoken *"응"* cannot run an action**, and that is structural rather than remembered.
//!
//! The deterministic side says WHERE, and WHETHER. A button press is an event Windows attributes to
//! a physical device; a transcript is a recogniser's opinion about a waveform. Putting one in the
//! other's slot is the same failure as handing a robot's detector to `Eyes` — and it is one edit
//! away at any time, because the edit looks like a convenience.
//!
//! Four claims, and each is checked where it cannot be sidestepped:
//!
//! * **The seam cannot name a `Session`**, so it cannot perform through one however it is later
//!   edited. A property, not an enumeration of today's call sites.
//! * **The core grew no ears and no mouth.** `Eyes` and `Hands` are in the contract because the
//!   core calls them; nothing in the core could call a `Listener`.
//! * **Only three files outside the core reach an action at all**, listed with the reason each is
//!   allowed. `one_place_acts.rs` counts calls *inside `app.rs`* and a voice front end in a new
//!   file would satisfy that by not being in that file; this is the same claim made across the
//!   tree.
//! * **The command line and the voice never meet.** The CLI performs behind `--live` rather than
//!   behind a button, which is right for a developer tool driven from a keyboard and is exactly
//!   what must not be handed a microphone.
//!
//! Ported from `tests/test_voice.py`, which held all four by walking Python that is going away.

use argmind_rules::{code_lines, outside_dependencies, root};

fn read(rel: &str) -> String {
    std::fs::read_to_string(root().join("native").join(rel))
        .unwrap_or_else(|e| panic!("{rel} should be there ({e})"))
}

/// **The seam cannot reach an action, because it cannot name a session.**
#[test]
fn the_seam_cannot_name_a_session() {
    let seam = read("argmind-voice/src/lib.rs");
    let leaked: Vec<String> = code_lines(&seam).into_iter()
        .filter(|(_, l)| {
            let code = l.split("//").next().unwrap_or("");
            ["Session", "argmind_core", "perform", "Effect", "Intent"]
                .iter().any(|w| code.contains(w))
        })
        .map(|(n, l)| format!("lib.rs:{n}: {}", l.trim()))
        .collect();
    assert!(leaked.is_empty(),
            "the voice seam names {leaked:#?} — speech may start a thought, never finish one");

    //: **And it depends on nothing at all**, which is the same claim made about the build. A seam
    //: that could reach the core through a dependency is a seam that will.
    let manifest = read("argmind-voice/Cargo.toml");
    assert!(outside_dependencies(&manifest).is_empty(),
            "the voice seam has taken a dependency: {:?}", outside_dependencies(&manifest));
    assert!(!manifest.contains("argmind-core"),
            "the voice seam depends on the core; it is then one import from a Session");
}

/// **The core grew no ears and no mouth.**
///
/// Being I/O is not the criterion for the contract; being called by the core is. A turn goes in as
/// a string and comes out as one, and both ends of a voice channel attach outside that.
#[test]
fn the_core_has_no_ears_and_no_mouth() {
    let core = root().join("native").join("argmind-core").join("src");
    let mut offenders = Vec::new();
    for path in argmind_rules::rust_files(&core) {
        let text = std::fs::read_to_string(&path).expect("a core file reads");
        let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        for (n, line) in code_lines(&text) {
            let code = line.split("//").next().unwrap_or("");
            //: `Speaks` is the *model* trait and is not a mouth — the core asks a model a question.
            //: Matching on `Speak` would catch it, which is why these are whole names.
            for word in ["Listener", "Speaker", "Heard", "argmind_voice", "Microphone"] {
                if code.contains(word) {
                    offenders.push(format!("{name}:{n}: {}", line.trim()));
                }
            }
        }
    }
    assert!(offenders.is_empty(), "the core has grown a voice: {offenders:#?}");
}

/// **Only these reach an action, and each is here for a stated reason.**
#[test]
fn nothing_outside_the_core_performs_except_the_three_that_may() {
    //: `app.rs` — one call. **It used to be behind 실행 and since 2026-09-22 it is behind 실행
    //: *or* a standing 확인 없이 실행**, the owner's decision after watching the buttons pile up.
    //: What the one call still buys is that there is one place to read: `Proposals` hands out the
    //: token either way, the core still downgrades anything nobody asked for, and the second
    //: opinion becomes a gate when no person is reading it. `argmind-serve/tests/acting.rs`.
    //: `argmind.rs` — the command line, behind `--live`, driven from a keyboard by a developer.
    //: `clicking.rs` — the harness whose entire subject is whether a confirmed click does the
    //: thing, launched against an application it started itself and dry unless told otherwise.
    const MAY: [&str; 3] = ["argmind-serve/src/app.rs",
                            "argmind-serve/src/bin/argmind.rs",
                            "argmind-measure/src/bin/clicking.rs"];
    let mut offenders = Vec::new();
    let native = root().join("native");
    let mut stack = vec![native.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().filter_map(Result::ok) {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                //: Not `target/`, and **not this crate** — a rules file that quotes the call it is
                //: looking for would otherwise report itself.
                if name != "target" && name != "argmind-rules" {
                    stack.push(path);
                }
                continue;
            }
            if path.extension().is_none_or(|e| e != "rs") {
                continue;
            }
            let rel = path.strip_prefix(&native).unwrap_or(&path)
                .to_string_lossy().replace('\\', "/");
            //: **The crate that defines it, including its own tests** — widened on 2026-09-28 and
            //: said out loud rather than slipped in. `argmind-core/src/` was already skipped
            //: wholesale because that is where `perform` lives and the core is the thing that
            //: performs; a test binary in the same crate is not the program, is linked into no
            //: executable, and cannot be reached from a microphone or from anything else at run
            //: time. What made the widening necessary was `tests/duplicates.rs`, which pins that a
            //: click with two candidates is **refused** — coverage of a refusal, which is the last
            //: thing this rule should be making harder to test.
            //:
            //: **`argmind-serve/tests/` is still caught, and that is the half worth keeping**: a
            //: test there would reach the real hands and click whoever ran it. Checked by breaking
            //: it both ways.
            if MAY.contains(&rel.as_str())
                || rel.starts_with("argmind-core/src/")
                || rel.starts_with("argmind-core/tests/") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap_or_default();
            for (n, line) in code_lines(&text) {
                if line.split("//").next().unwrap_or("").contains(".perform(") {
                    offenders.push(format!("{rel}:{n}"));
                }
            }
        }
    }
    assert!(offenders.is_empty(),
            "{offenders:#?} reach an action outside the confirmation button. A module a \
             microphone feeds may never be one of them");
}

/// **The command line and the voice never meet.**
///
/// A `--voice` flag on `talk` would let a mis-transcription click something with no confirmation
/// anywhere, because that path performs behind `--live` and nothing else.
#[test]
fn the_command_line_has_no_microphone() {
    let cli = read("argmind-serve/src/bin/argmind.rs");
    let met: Vec<String> = code_lines(&cli).into_iter()
        .filter(|(_, l)| {
            let code = l.split("//").next().unwrap_or("");
            //: Whole names again: the CLI has a local called `heard` for a typed line, and
            //: `Speaks` for the model. Neither is a device.
            ["argmind_voice", "Microphone", "SapiSpeaker", "argmind_env_whisper"]
                .iter().any(|w| code.contains(w))
        })
        .map(|(n, l)| format!("argmind.rs:{n}: {}", l.trim()))
        .collect();
    assert!(met.is_empty(),
            "the command line has grown a voice: {met:#?}. It reaches perform() behind --live \
             with no button, so speech there would confirm an action");
}

/// **The room stays quiet unless somebody asked.**
///
/// A proposal waits behind a button; a sentence waits behind nothing. So a thought the agent had by
/// itself — a watch firing while its owner is in a meeting — appears in the page and **is not said
/// out loud**. Both places in `app.rs` say this in a comment, which is the sentence rather than the
/// fact: `think` speaks and `notice` does not, and one copied line would end it silently.
///
/// ## The one exception, added 2026-09-21, and why it is not a hole
///
/// Choosing a voice means hearing it. The neural talker is a base model with no speakers of its
/// own, so which voice comes out is a sampler seed — a number, and a number cannot be judged by
/// reading it. `Job::Voice` therefore speaks, and it is a second place in this file.
///
/// **What keeps that from being the copied line this rule exists to catch is *what* it may say.**
/// The exception is written against the argument, not the caller: a second `voice.say` is allowed
/// only if it says [`A_SAMPLE`] — one fixed constant with no model output anywhere near it. A
/// `voice.say(intent.say)` or a `voice.say(&note.believes)` added to `notice` still fails, which is
/// the failure that matters. Widening by a caller name would not have held: the job loop's nearest
/// enclosing `fn` is `hello`, several hundred lines up, so the owner this scanner reports for
/// anything inside it is an accident of layout.
#[test]
fn only_a_reply_that_was_asked_for_is_spoken_aloud() {
    let app = read("argmind-serve/src/app.rs");
    let lines: Vec<&str> = app.lines().collect();
    let (mut turns, mut samples) = (Vec::new(), Vec::new());
    for (n, line) in lines.iter().enumerate() {
        let code = line.split("//").next().unwrap_or("");
        //: **Any receiver, not one called `voice`.** Written as `voice.say(` until 2026-09-21,
        //: which is a rule about a local variable's name: breaking it with
        //: `if let Some(v) = desk.voice.as_mut() { v.say(&desk.session.note.believes) }` was waved
        //: straight through. Nothing else in this file has a `say` method, so the wider match costs
        //: nothing and stops the rule being evaded by a rename.
        if !code.contains(".say(") {
            continue;
        }
        //: **A fixed sentence, named rather than written out.** A literal here would be a second
        //: copy of the sample and could drift from the one that is said.
        if code.contains(".say(A_SAMPLE)") {
            samples.push(n + 1);
            continue;
        }
        let owner = lines[..n].iter().rev()
            .find_map(|l| l.trim_start().strip_prefix("fn ")
                      .map(|r| r.split('(').next().unwrap_or("").to_string()))
            .unwrap_or_default();
        turns.push((n + 1, owner));
    }
    assert_eq!(turns.len(), 1, "more than one place speaks a turn: {turns:?}");
    assert_eq!(turns[0].1, "think",
               "app.rs:{} speaks from {:?}; only a turn somebody started may become sound",
               turns[0].0, turns[0].1);
    //: **At most one demonstration**, so the exception cannot itself be copied.
    assert!(samples.len() <= 1, "more than one place says the sample: {samples:?}");

    //: And the unprompted path is still there to be wrong about — a rule about `notice` means
    //: nothing if `notice` has gone.
    assert!(app.contains("fn notice("), "the unprompted path has moved; recheck this rule");

    //: **The exception is checked by breaking it**, like the other four rules here. If the sample
    //: were ever anything but a constant this test would be waved through by the line above, so
    //: what `A_SAMPLE` *is* gets asserted where it is declared: `tests/voice.rs` requires one short
    //: Korean sentence, and this requires that it is a `const` rather than something computed.
    assert!(app.contains("pub const A_SAMPLE: &str = \""),
            "A_SAMPLE is no longer a fixed string; the exception above rests on it being one");
}

/// The files being read are the ones meant. **A rule checked against nothing passes forever.**
#[test]
fn there_is_a_seam_and_a_command_line_to_check() {
    let seam = read("argmind-voice/src/lib.rs");
    assert!(seam.contains("trait Listener") && seam.contains("trait Speaker"),
            "argmind-voice is no longer the seam");
    let cli = read("argmind-serve/src/bin/argmind.rs");
    assert!(cli.contains(".perform("), "the CLI no longer performs — has this rule moved?");
}
