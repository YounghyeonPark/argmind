//! What the window says before anything happens.
//!
//! **An approval is only worth having if it was informed.** A dialog reading *"install
//! dependencies?"* is answered yes by everybody, because they want the program to work — and they
//! have then authorised something they could not name. These check that every sentence names the
//! thing, says where it comes from, and does not hide the two words that change what is being
//! agreed to: *administrator*, and *installs nothing*.
//!
//! Nothing here shows a window or runs anything. The text is separated from the doing precisely so
//! it can be read without a machine being changed.

use argmind_serve::fix::{describe, Fix};

fn pull() -> Fix {
    Fix::PullModel { model: "qwen3:8b".into(), about_gb: 5.0 }
}

fn page() -> Fix {
    Fix::OpenPage { url: "https://ollama.com/download".into(),
                    why: "Ollama가 있어야 argmind가 동작합니다.".into() }
}

fn elevated() -> Fix {
    Fix::RunElevated { command: "DISM /Online /Add-Capability /CapabilityName:X".into(),
                       why: "한국어 음성이 없으면 소리로 나오지 않습니다.".into() }
}

/// **Every sentence names the thing it is about**, so a person reading only the window knows what
/// they are agreeing to.
#[test]
fn the_window_names_what_it_will_do() {
    let said = describe(&pull());
    assert!(said.contains("qwen3:8b"), "{said}");
    //: The size, because five gigabytes is the part somebody would want to know before saying yes
    //: on a metered connection.
    assert!(said.contains("5 GB"), "{said}");
    //: And what it will *not* do, which is the reason this one needs no second thought.
    assert!(said.contains("설치 프로그램은 실행되지 않습니다"), "{said}");

    let said = describe(&page());
    assert!(said.contains("https://ollama.com/download"), "{said}");
    let said = describe(&elevated());
    assert!(said.contains("DISM /Online"), "{said}");
}

/// **Opening a page installs nothing, and it says so.**
///
/// This is the honest end of the road for software argmind will not fetch and run on somebody's
/// behalf — and a person who thought pressing yes had installed Ollama would go looking for it.
#[test]
fn opening_a_page_says_that_it_installs_nothing() {
    let said = describe(&page());
    assert!(said.contains("직접 설치하지는 않습니다"), "{said}");
    assert!(!page().changes_the_machine());
    //: The other two do change it, and the difference is a property of the variant rather than of
    //: the words — so a rewritten sentence cannot quietly reclassify one.
    assert!(pull().changes_the_machine());
    assert!(elevated().changes_the_machine());
}

/// **Administrator is not a detail to bury**, so the word is in the body and in the title.
#[test]
fn an_elevated_command_says_administrator_before_it_asks() {
    let said = describe(&elevated());
    assert!(said.contains("관리자 권한"), "{said}");
    //: And it warns that Windows will ask again — two windows before anything changes, which is a
    //: thing a person should be expecting rather than surprised by.
    assert!(said.contains("Windows가 한 번 더"), "{said}");
    assert!(elevated().title().contains("관리자"));
}

/// **Running somebody else's build is the sharpest line here**, and the window has to say so.
///
/// The earlier design refused to do it at all. The owner's decision is that the window asks and the
/// agent then does the whole sequence — so the least this can do is make the sentence carry the
/// part that was being objected to, in the words a person would use to object.
#[test]
fn downloading_and_running_a_stranger_is_named_as_that() {
    let fix = Fix::Install { url: "https://ollama.com/download/OllamaSetup.exe".into(),
                             file: "OllamaSetup.exe".into(),
                             why: "Ollama가 있어야 argmind가 동작합니다.".into(),
                             about_mb: 700 };
    let said = describe(&fix);
    //: Where from, how big, and that the certificate is checked — the three facts that turn "a
    //: file nobody had seen" into a file somebody agreed to.
    assert!(said.contains("ollama.com"), "{said}");
    assert!(said.contains("700 MB"), "{said}");
    assert!(said.contains("인증서"), "{said}");
    //: **And that it will be run**, in those words. A sentence about downloading that quietly also
    //: executes is the one this whole module exists to prevent.
    assert!(said.contains("실행합니다"), "{said}");
    assert!(said.contains("신뢰한다"), "{said}");
    assert!(fix.runs_a_stranger());
    assert!(fix.changes_the_machine());
}

/// **Fetching data is not running code**, and the two are different variants so the text cannot
/// blur them.
#[test]
fn fetching_a_model_says_that_nothing_is_executed() {
    let checked = Fix::Fetch { url: "https://huggingface.co/x/y".into(),
                               into: "C:/models/ggml-base.bin".into(),
                               why: "받아쓰기 모델".into(), about_mb: 141,
                               sha256: Some("60ed".repeat(16)) };
    let said = describe(&checked);
    assert!(said.contains("실행하지 않습니다"), "{said}");
    assert!(said.contains("해시를 확인"), "{said}");
    //: It fetches, so it changes the machine — but it does not run a stranger, and the two
    //: questions have two answers.
    assert!(checked.changes_the_machine());
    assert!(!checked.runs_a_stranger());

    //: **No published hash is said out loud rather than left out.** "Nobody publishes one for
    //: this" is a fact about the source, and the person deciding should have it in front of them.
    let unchecked = Fix::Fetch { url: "https://example.com/x".into(), into: "x".into(),
                                 why: "y".into(), about_mb: 1, sha256: None };
    let said = describe(&unchecked);
    assert!(said.contains("게시된 해시가 없어"), "{said}");
    assert!(!said.contains("해시를 확인하고"), "{said}");
}

/// The titles are three, and they are not interchangeable.
#[test]
fn each_kind_of_question_has_its_own_title() {
    let titles = [pull().title(), page().title(), elevated().title()];
    assert_eq!(titles.iter().collect::<std::collections::BTreeSet<_>>().len(), 3);
    //: Every one says which program is asking. A modal window with no name on it, appearing over
    //: whatever somebody was doing, is a window they have no reason to trust.
    assert!(titles.iter().all(|t| t.contains("argmind")));
}

/// Every sentence **ends in the question it is asking**, so the last thing read is the decision.
#[test]
fn the_body_ends_on_the_question() {
    for fix in [pull(), page(), elevated()] {
        let said = describe(&fix);
        assert!(said.trim_end().ends_with("까요?"), "{said}");
        //: And it is more than a line: a one-sentence dialog is the vague one this exists to avoid.
        assert!(said.lines().filter(|l| !l.trim().is_empty()).count() >= 4, "{said}");
    }
}

// ---- every variant, not the three that happened to get a helper -----------------------------------

/// One of each, **and the list is the test**.
///
/// The rules below were written against three variants named one at a time, so `StartOllama`
/// arriving on 2026-09-21 was checked by none of them — a new kind of question could have shipped
/// with a title that did not say argmind, or a body that trailed off instead of asking. Anything
/// added to `Fix` now has to appear here or the compiler complains, and then it has to pass.
fn one_of_each() -> Vec<Fix> {
    let all = vec![
        pull(), page(), elevated(),
        Fix::Install { url: "https://ollama.com/download/OllamaSetup.exe".into(),
                       file: "OllamaSetup.exe".into(),
                       why: "Ollama가 있어야 argmind가 동작합니다.".into(), about_mb: 700 },
        Fix::Fetch { url: "https://huggingface.co/x/y".into(), into: "C:/models/x.bin".into(),
                     why: "받아쓰기 모델".into(), about_mb: 141, sha256: None },
        Fix::BuildRecogniser { repo: "https://github.com/ggml-org/whisper.cpp.git".into(),
                               commit: "8a9a".repeat(10), why: "받아쓰기".into(),
                               minutes: 10, built: false },
        Fix::BuildSpeaker { repo: "https://github.com/ServeurpersoCom/qwentts.cpp.git".into(),
                            commit: "90ef".repeat(10), why: "목소리".into(),
                            minutes: 15, built: false },
        Fix::StartOllama { why: "Ollama가 설치되어 있는데 실행 중이 아닙니다.".into() },
    ];
    //: **Exhaustive by construction.** A variant added to the enum and not to the list above makes
    //: this fail to compile, which is the only way a list like this stays a list of everything.
    for fix in &all {
        match fix {
            Fix::PullModel { .. } | Fix::OpenPage { .. } | Fix::Install { .. }
            | Fix::Fetch { .. } | Fix::BuildRecogniser { .. } | Fix::BuildSpeaker { .. }
            | Fix::StartOllama { .. } | Fix::RunElevated { .. } => {}
        }
    }
    all
}

/// **Every window asks, names itself, and says more than one thing.**
#[test]
fn every_kind_of_question_is_a_question() {
    for fix in one_of_each() {
        let said = describe(&fix);
        assert!(said.trim_end().ends_with("까요?"), "{:?}\n{said}", fix.title());
        assert!(said.lines().filter(|l| !l.trim().is_empty()).count() >= 4,
                "{} is a one-line dialog:\n{said}", fix.title());
        assert!(fix.title().contains("argmind"), "{}", fix.title());
    }
}

/// **Starting something already here is not installing it**, and the window has to be the one that
/// makes that difference legible — the two live behind the same standing answer.
#[test]
fn starting_ollama_says_that_nothing_is_downloaded() {
    let start = Fix::StartOllama { why: "Ollama가 설치되어 있는데 실행 중이 아닙니다.".into() };
    let said = describe(&start);
    assert!(said.contains("이미 설치된"), "{said}");
    assert!(said.contains("내려받는 것은 없습니다"), "{said}");
    //: **And that it outlives argmind**, which is the one surprising consequence: argmind takes its
    //: own two servers down with it and deliberately does not take this one.
    assert!(said.contains("계속 실행됩니다"), "{said}");
    assert!(!said.contains("MB"), "a start should not be quoting a download size: {said}");

    //: It runs somebody else's program, so it is flagged as that — the flag is a property of the
    //: variant, and the sentence is where the mitigation goes.
    assert!(start.runs_a_stranger());
    assert!(start.changes_the_machine());

    //: **Its own title**, because a person who reads only the title must not think they are about
    //: to download 700 MB, nor the other way round.
    let install = Fix::Install { url: "https://ollama.com/download/OllamaSetup.exe".into(),
                                 file: "OllamaSetup.exe".into(), why: "x".into(), about_mb: 700 };
    assert_ne!(start.title(), install.title());
}

/// **The need picks the fix from what is actually wrong**, which is the defect this variant exists
/// for: *not answering* is not *not installed*, and offering the installer against the first is a
/// 700 MB download on every launch under a standing yes.
///
/// Machine-dependent by nature, so it asserts whichever branch this machine is on and says which.
#[test]
fn an_installed_ollama_is_offered_a_start_and_not_an_installer() {
    let need = argmind_serve::needs::by_key("qwen3:8b", "ollama").expect("ollama is not a need");
    let here = argmind_serve::ollama::awake() || argmind_serve::ollama::here();
    match (here, need.fix) {
        (true, Some(Fix::StartOllama { .. })) => {}
        (false, Some(Fix::Install { .. })) => {}
        //: Off Windows nothing is offered at all, which is `offered`'s doing and not this rule's.
        (_, None) => assert!(!argmind_serve::platform::can_carry_out_fixes()),
        (here, other) => panic!("ollama here={here} was offered {:?}", other.map(|f| f.title())),
    }
    //: **Three states, not two**, and the sentence has to tell them apart: answering, installed
    //: and stopped, or not here at all. The middle one is the whole point — *응답 없음* on its own
    //: sends somebody to ollama.com for something already on their disk.
    let want = match (need.present, here) {
        (true, _) => "응답함",
        (false, true) => "설치되어 있지만 실행 중이 아닙니다",
        (false, false) => "설치되어 있지 않습니다",
    };
    assert!(need.found.contains(want), "expected {want:?} in {:?}", need.found);
}
