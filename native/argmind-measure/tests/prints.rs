//! The fingerprints, against the Python that produced every committed result.
//!
//! A port that computes a different `screen_print` does not fail loudly — it writes a result under
//! a name nobody can compare with anything, which is the failure this project has had five times.
//!
//! **`screen_print` was edited by hand once**, on 2026-11-17: `5020f7` → `9d54a1`, when the
//! screens were redacted before the repository went public (`measure/screens/README.md`). The
//! Python that generated `measure.json` is gone, so there was nothing to regenerate it with.

use std::path::PathBuf;

use argmind_measure::{prompt_tag, screen_print, screens, spend, Destination};
use serde_json::Value;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn want() -> Value {
    let p = root().join("native").join("expectations").join("measure.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|_| {
        panic!("{} is missing. Generate it: python native/expectations/all.py", p.display())
    })).unwrap()
}

#[test]
fn the_fixtures_hash_to_what_python_records() {
    let d = want();
    let dir = root().join("measure").join("screens");
    assert_eq!(screen_print(&dir), d["screen_print"].as_str().unwrap());
    assert_eq!(prompt_tag(), d["prompt_tag"].as_str().unwrap());
}

#[test]
fn the_fixtures_load_to_the_same_widgets() {
    let d = want();
    let loaded = screens(&root().join("measure").join("screens"));
    let recorded = d["screens"].as_array().unwrap();
    assert_eq!(loaded.len(), recorded.len(), "a fixture appeared or vanished");
    for row in recorded {
        let name = row["name"].as_str().unwrap();
        let got = loaded.get(name).unwrap_or_else(|| panic!("{name} did not load"));
        assert_eq!(got.window, row["window"].as_str().unwrap(), "{name} window");
        assert_eq!(got.widgets.len(), row["widgets"].as_u64().unwrap() as usize, "{name} count");
        //: The first three of each, so an off-by-one in the order or the rectangle shows.
        for (i, w) in row["first"].as_array().unwrap().iter().enumerate() {
            let r = w["rect"].as_array().unwrap();
            assert_eq!(got.widgets[i].name, w["name"].as_str().unwrap(), "{name}[{i}] name");
            assert_eq!(got.widgets[i].role, w["role"].as_str().unwrap(), "{name}[{i}] role");
            assert_eq!(got.widgets[i].rect,
                       (r[0].as_i64().unwrap(), r[1].as_i64().unwrap(),
                        r[2].as_i64().unwrap(), r[3].as_i64().unwrap()), "{name}[{i}] rect");
        }
    }
}

#[test]
fn the_cost_record_has_the_same_shape_and_says_nothing_when_there_was_no_reply() {
    use argmind_core::llm::Reply;
    let d = want();
    let reply = Reply { text: "{\"say\": \"ok\"}".into(), prompt_tokens: Some(373),
                        out_tokens: Some(29), reason: "stop".into(), thought: 512 };
    let got = spend(Some(&reply), Some(0.478));
    for key in ["out_tokens", "reason", "thought_chars", "truncated"] {
        assert_eq!(&got[key], &d["spend"][key], "{key}");
    }
    //: **All-`None` rather than zeros.** A call that produced no reply is a fact; a zero would be a
    //: measurement of something that did not happen.
    let nothing = spend(None, None);
    for key in ["seconds", "out_tokens", "reason", "thought_chars", "truncated"] {
        assert!(nothing[key].is_null(), "{key} should be absent, not zero");
        assert!(d["spend_none"][key].is_null(), "the Python disagrees about {key}");
    }
    let stopped = Reply { text: "{}".into(), prompt_tokens: None, out_tokens: Some(41),
                          reason: "degenerate".into(), thought: 0 };
    assert_eq!(spend(Some(&stopped), Some(1.2))["truncated"], d["spend_stopped"]["truncated"]);
}

#[test]
fn a_destination_refuses_a_path_that_is_already_there() {
    let dir = root().join("measure");
    let taken = Destination::new(&dir, "replies", &[
        "qwen3-8b".into(), "off".into(), "off".into(), "r322a4c".into(), "s5020f7".into(),
        "f041650".into(), "h9a643b".into(), "cec77a6".into(), "lbc9cc0".into(), "mb240dd".into(),
        "stopsoff".into(), "x1".into()]);
    assert!(taken.path.exists(), "the committed corpus moved; this test needs a real collision");
    let refused = taken.refuse_if_taken().unwrap_err();
    assert!(refused.contains("REFUSING TO WRITE"));
    assert!(refused.contains("Do not delete it"), "the message has to stop somebody deleting it");

    let free = Destination::new(&dir, "replies", &["nothing-has-this-name".into()]);
    assert!(free.refuse_if_taken().is_ok());
}

/// **The bytes are sorted by content, not by path**, and the four committed fixtures cannot show
/// that: their path order and their content order happen to coincide, so a port that sorted by path
/// passed every test above. Found by mutation, closed here with a directory built to disagree.
///
/// The property matters because it is what makes the fingerprint invariant to a rename: a fixture
/// renamed and not changed is the same fixture, and a measurement recorded against it stays
/// comparable.
#[test]
fn the_screen_print_is_invariant_to_a_rename() {
    let dir = std::env::temp_dir().join("argmind-screen-print-order");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    //: `a.json` holds the byte-larger content and `z.json` the smaller, so path order and content
    //: order are opposites.
    let big = r#"{"window": "W", "widgets": [{"name": "z", "role": "button", "rect": [0,0,1,1]}]}"#;
    let small = r#"{"window": "W", "widgets": [{"name": "a", "role": "button", "rect": [0,0,1,1]}]}"#;
    std::fs::write(dir.join("a.json"), big).unwrap();
    std::fs::write(dir.join("z.json"), small).unwrap();
    let before = screen_print(&dir);

    //: Swap the names, change nothing else. Sorted by content this is the same screen set; sorted
    //: by path it is a different one.
    std::fs::write(dir.join("a.json"), small).unwrap();
    std::fs::write(dir.join("z.json"), big).unwrap();
    let after = screen_print(&dir);
    let _ = std::fs::remove_dir_all(&dir);

    assert_eq!(before, after,
               "the fingerprint moved when two fixtures swapped names — it is sorting by path");
}
