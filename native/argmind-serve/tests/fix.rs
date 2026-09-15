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
