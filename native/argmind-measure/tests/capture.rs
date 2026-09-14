//! The fixture writer, checked against the fixtures already on disk.
//!
//! **A capture that writes the same content in different bytes is a capture that moves `s`.** Every
//! harness puts `s` — a hash of these files as they sit on disk — in its result filename, so a
//! re-captured fixture written by a different encoder would make every new result look
//! incomparable with every old one, for no reason anybody could see.
//!
//! So the four committed screens are read, re-serialised by the writer that a live capture would
//! use, and required back byte for byte. No display, no UI Automation, no desktop.

use std::path::PathBuf;

use argmind_measure::capture::{screen_json, slug, Kept};
use argmind_measure::{screen_print, screens};
use argmind_core::contract::Widget;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

#[test]
fn every_committed_fixture_is_written_back_byte_for_byte() {
    let dir = root().join("measure").join("screens");
    let loaded = screens(&dir);
    assert_eq!(loaded.len(), 4, "there are four captured screens, found {}", loaded.len());
    let mut total = 0;
    for (name, screen) in &loaded {
        let path = dir.join(format!("{name}.json"));
        let want = std::fs::read(&path).unwrap();
        let got = screen_json(&screen.window, &screen.widgets);
        assert_eq!(got.len(), want.len(),
                   "{name}: wrote {} bytes against {} on disk", got.len(), want.len());
        assert!(got == want, "{name}: the bytes differ at {:?}",
                got.iter().zip(&want).position(|(a, b)| a != b));
        //: The two properties `s` actually depends on, stated rather than implied by the equality
        //: above — so a future change that broke one says which.
        assert!(got.windows(2).any(|w| w == b"\r\n"), "{name}: written with LF");
        assert!(!got.ends_with(b"\n"), "{name}: written with a trailing newline");
        total += screen.widgets.len();
    }
    assert!(total > 250, "only {total} widgets across the fixtures");
    println!("  {} screens, {total} widgets, re-written byte for byte", loaded.len());
}

/// And the consequence, directly: rewriting every fixture leaves `s` where it was.
#[test]
fn rewriting_every_fixture_leaves_the_fingerprint_alone() {
    let dir = root().join("measure").join("screens");
    let before = screen_print(&dir);
    let copy = std::env::temp_dir().join("argmind-capture-rewrite");
    let _ = std::fs::remove_dir_all(&copy);
    std::fs::create_dir_all(&copy).unwrap();
    for (name, screen) in &screens(&dir) {
        std::fs::write(copy.join(format!("{name}.json")),
                       screen_json(&screen.window, &screen.widgets)).unwrap();
    }
    assert_eq!(screen_print(&copy), before, "a rewrite moved s, and every harness names s");
    let _ = std::fs::remove_dir_all(&copy);
}

/// A title becomes a filename, and the committed names are the proof it is the same rule.
#[test]
fn a_window_title_becomes_the_filename_it_already_has() {
    let dir = root().join("measure").join("screens");
    for (name, screen) in &screens(&dir) {
        assert_eq!(&slug(&screen.window), name,
                   "{:?} no longer slugs to the file it is in", screen.window);
    }
    //: Colons, slashes and dots all become hyphens; the case is dropped; the length is capped.
    assert_eq!(slug("Character Map"), "character-map");
    assert_eq!(slug("Untitled - Paint"), "untitled---paint");
    assert_eq!(slug("C:\\dev/argmind"), "c--dev-argmind");
    assert_eq!(slug(""), "untitled");
    assert_eq!(slug("   "), "untitled");
    //: **48 characters, counted in characters.** A Korean title is three bytes a character and
    //: slicing bytes would cut one in half and produce a filename that is not valid text.
    let long = "가".repeat(80);
    assert_eq!(slug(&long).chars().count(), 48);
    assert!(slug(&long).chars().all(|c| c == '가'));
}

/// **Keep the richest reading of a window, not the first.**
///
/// A window caught mid-paint exposes a fraction of its widgets, and keeping that would quietly make
/// every goal against it unanswerable — a failure that reads as the model choosing badly.
#[test]
fn a_window_caught_mid_paint_does_not_overwrite_a_full_reading() {
    let w = |n: &str| Widget { name: n.into(), role: "button".into(), rect: (0, 0, 10, 10),
                               window: "Paint".into() };
    let mut kept = Kept::default();
    assert!(kept.offer(vec![w("a"), w("b")]));
    //: Fewer widgets, same window: refused.
    assert!(!kept.offer(vec![w("a")]));
    assert_eq!(kept.windows["Paint"].len(), 2);
    //: More: taken.
    assert!(kept.offer(vec![w("a"), w("b"), w("c")]));
    assert_eq!(kept.windows["Paint"].len(), 3);
    //: **An equal-length reading is not richer**, so the first of two equal readings is kept and a
    //: window flickering between two states does not thrash the file.
    assert!(!kept.offer(vec![w("x"), w("y"), w("z")]));
    assert_eq!(kept.windows["Paint"][0].name, "a");
    //: A reading of nothing is not a window.
    assert!(!kept.offer(vec![]));
    assert_eq!(kept.windows.len(), 1);
}
