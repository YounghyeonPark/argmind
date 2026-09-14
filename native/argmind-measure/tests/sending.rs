//! The blindness bookkeeping: what a harness did not send, and which run covers it.
//!
//! Checked against the Python's own answer for the tree as it stands, and then against three
//! synthetic directories — because the real one can only be in **one** of the three states at a
//! time, and the state it is in today is the one where nothing has run the axis.

use std::path::PathBuf;

use argmind_measure::sending::{caveat, conversation_runs, deferral, norm_print, Sent};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn want() -> Value {
    let p = root().join("native").join("expectations").join("sending.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|_| {
        panic!("{} is missing. Generate it: python native/expectations/all.py", p.display())
    })).unwrap()
}

#[test]
fn the_conversation_runs_are_read_the_same_way() {
    let d = want();
    let got = conversation_runs(&root().join("measure"));
    let recorded = d["runs"].as_array().unwrap();
    assert_eq!(got.len(), recorded.len(), "a conversation result appeared or vanished");
    //: Compared as a set: both sort newest first, and two files written in the same second have no
    //: defined order between them. What matters is that each file is read to the same two fields.
    for row in recorded {
        let file = row["file"].as_str().unwrap();
        let mine = got.iter()
            .find(|r| r.path.file_name().unwrap().to_string_lossy() == file)
            .unwrap_or_else(|| panic!("{file} was not found"));
        assert_eq!(mine.prompt.as_deref(), row["prompt"].as_str(), "{file} prompt");
        assert_eq!(mine.loop_print.as_deref(), row["loop_print"].as_str(), "{file} loop_print");
    }
}

#[test]
fn the_caveat_is_the_sentence_python_prints() {
    let d = want();
    let (tag, lp) = (d["prompt"].as_str().unwrap(), d["loop_print"].as_str().unwrap());
    let dir = root().join("measure");
    assert_eq!(caveat(&dir, tag, lp), d["caveat"].as_str().unwrap());

    let got = deferral(&dir, tag, lp);
    assert_eq!(got.exercised_at_this_prompt(), d["exercised_at_this_prompt"].as_bool().unwrap());
    assert_eq!(got.exercised_at_this_tree(), d["exercised_at_this_tree"].as_bool().unwrap());
    assert_eq!(got.at_this_prompt.len(),
               d["runs_at_this_prompt"].as_array().unwrap().len());
}

/// **All three states, because the tree can only be in one.**
///
/// The middle one is the one worth protecting: a run exists at this prompt but under a different
/// `loop.py`, so the window and the wording of a history line have moved underneath it. A port that
/// collapsed it into "covered" would lose the only thing that notices — the prompt fingerprint does
/// not cover `loop.py`.
#[test]
fn the_three_states_are_three_different_sentences() {
    let base = std::env::temp_dir().join("argmind-sending-states");
    let _ = std::fs::remove_dir_all(&base);

    let write = |dir: &std::path::Path, name: &str, prompt: &str, loop_print: &str| {
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join(name), format!(
            "{{\"prompt_scheme\": \"r = …\", \"prompt\": \"{prompt}\", \
              \"loop_print\": \"{loop_print}\"}}")).unwrap();
    };

    let nothing = base.join("nothing");
    std::fs::create_dir_all(&nothing).unwrap();
    let said = caveat(&nothing, "r322a4c", "d8d312");
    assert!(said.contains("NOTHING has run the history axis"), "{said}");

    let other_loop = base.join("other-loop");
    write(&other_loop, "conversation-a.json", "322a4c", "aaaaaa");
    let said = caveat(&other_loop, "r322a4c", "d8d312");
    assert!(said.contains("under a different loop.py"), "{said}");
    assert!(said.contains("aaaaaa -> d8d312"), "the sentence has to name both: {said}");

    let covered = base.join("covered");
    write(&covered, "conversation-b.json", "322a4c", "d8d312");
    let said = caveat(&covered, "r322a4c", "d8d312");
    assert!(said.contains("has run at r322a4c + loop d8d312"), "{said}");

    //: A run at a *different prompt* is not coverage, however recent it is.
    let stale = base.join("stale");
    write(&stale, "conversation-c.json", "a62805", "d8d312");
    let said = caveat(&stale, "r322a4c", "d8d312");
    assert!(said.contains("NOTHING has run the history axis"), "{said}");

    let _ = std::fs::remove_dir_all(&base);
}

/// **Counted, not declared.** A harness that says it sends history and does not is the failure this
/// module exists for.
#[test]
fn a_sender_counts_what_it_actually_sent() {
    let mut sent = Sent::default();
    sent.turn(None);
    sent.turn(Some(&[]));
    sent.turn(Some(&[("them".into(), "hi".into())]));
    //: An **empty** history counts as without. It is not a conversation; treating it as one is how
    //: a harness comes to claim coverage it does not have.
    assert_eq!(sent, Sent { with: 1, without: 2 });
}

/// `l` is the newline-normalised hash, not the byte one: the same file checked out on Windows and
/// on Linux differs in every line ending and is the same file.
#[test]
fn the_normalised_print_ignores_line_endings() {
    let dir = std::env::temp_dir().join("argmind-norm-print");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("crlf"), b"one\r\ntwo\r\n").unwrap();
    std::fs::write(dir.join("lf"), b"one\ntwo\n").unwrap();
    std::fs::write(dir.join("other"), b"one\ntwo\nthree\n").unwrap();
    assert_eq!(norm_print(&dir.join("crlf")), norm_print(&dir.join("lf")));
    assert_ne!(norm_print(&dir.join("lf")), norm_print(&dir.join("other")));
    let _ = std::fs::remove_dir_all(&dir);
}
