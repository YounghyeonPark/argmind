//! What each watch says happened, against the words the Python produces.
//!
//! A watch's description is **not a log line — it is the text of a `Trigger`**, so it goes into the
//! prompt and the model reads it. Two implementations that describe the same change differently are
//! two agents, and nothing downstream compares the wording against anything, so this is the only
//! place it can be caught.
//!
//! The floor is checked on a clock the test supplies. Nothing in the core reads a wall clock — the
//! same rule that keeps a platform out of here — which is what makes three seconds testable in no
//! time at all.

use std::path::PathBuf;

use argmind_core::contract::Widget;
use argmind_core::watch::{poll, py_repr, Looking, Watch, MIN_GAP};
use serde_json::Value;

fn want() -> Value {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .join("native").join("expectations").join("watch.json");
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|_| {
        panic!("{} is missing. Generate it: python native/expectations/all.py", p.display())
    })).unwrap()
}

fn widgets(rows: &Value) -> Vec<Widget> {
    rows.as_array().unwrap().iter()
        .map(|r| Widget::new(r[0].as_str().unwrap(), "button", (0, 0, 10, 10),
                             r[1].as_str().unwrap()))
        .collect()
}

fn watch_named(name: &str) -> Watch {
    Watch::new(match name {
        "window" => Looking::WindowChanged,
        n if n.starts_with("appeared:") => Looking::Appeared(n[9..].to_string()),
        n if n.starts_with("vanished:") => Looking::Vanished(n[9..].to_string()),
        other => panic!("no watch called {other}"),
    })
}

#[test]
fn every_watch_says_what_the_python_says() {
    let d = want();
    assert_eq!(d["min_gap"].as_f64().unwrap(), MIN_GAP);
    let rows = d["rows"].as_array().unwrap();
    assert!(rows.len() >= 40, "only {} scene/watch pairs", rows.len());
    let mut spoke = 0;
    for row in rows {
        let watch = watch_named(row["watch"].as_str().unwrap());
        let got = watch.look(&widgets(&row["before"]), &widgets(&row["after"]));
        assert_eq!(got.as_deref(), row["said"].as_str(),
                   "{} / {}", row["watch"], row["scene"]);
        if got.is_some() {
            spoke += 1;
        }
    }
    //: **A suite where every watch stays silent is a suite that tests nothing.**
    assert!(spoke >= 6, "only {spoke} of {} pairs described anything", rows.len());
    println!("  {} pairs, {spoke} of them describing something", rows.len());
}

/// **Python's quoting, exactly** — three wordings for one event, and nothing would notice which.
#[test]
fn a_name_is_quoted_the_way_python_quotes_it() {
    assert_eq!(py_repr("Character Map"), "'Character Map'");
    //: An apostrophe inside makes Python switch quotes rather than escape.
    assert_eq!(py_repr("it's here"), "\"it's here\"");
    //: Unless there is a double quote too, in which case it escapes the single.
    assert_eq!(py_repr("it's \"here\""), "'it\\'s \"here\"'");
    assert_eq!(py_repr("say \"hello\""), "'say \"hello\"'");
    //: A backslash doubles.
    assert_eq!(py_repr("C:\\Search"), "'C:\\\\Search'");
    //: **Printable non-ASCII stays as itself.** Python 3 stopped escaping it, and a watch
    //: announcing an escape sequence would tell the model nothing it can read.
    assert_eq!(py_repr("검색 결과"), "'검색 결과'");
    assert_eq!(py_repr(""), "''");
    assert_eq!(py_repr("a\nb"), "'a\\nb'");
    assert_eq!(py_repr("a\tb"), "'a\\tb'");
}

/// **The floor is not politeness.** Perception changes 13 seconds in 14 and a call costs 0.3–2 s,
/// so an unfloored watch is a model that never stops and an agent that never listens.
#[test]
fn a_watch_will_not_fire_again_inside_its_floor() {
    let a = vec![Widget::new("A", "button", (0, 0, 1, 1), "Paint")];
    let b = vec![Widget::new("A", "button", (0, 0, 1, 1), "Character Map")];
    let c = vec![Widget::new("A", "button", (0, 0, 1, 1), "Notepad")];
    let mut watch = Watch::new(Looking::WindowChanged);

    //: **The first one fires**, and it must — a floor measured from the epoch would make the agent
    //: silent for the first three seconds of its life and nobody would ever see it.
    assert!(watch.fired(&a, &b, 1000.0).is_some());
    //: Another real change, well inside the floor: refused.
    assert!(watch.fired(&b, &c, 1001.0).is_none());
    assert!(watch.fired(&b, &c, 1002.9).is_none());
    //: And past it: allowed.
    assert!(watch.fired(&b, &c, 1003.0).is_some());

    //: The floor is a floor on *firing*, not on looking — a change that happens inside it is not
    //: remembered and fired late, it is simply not this watch's turn.
    let mut watch = Watch::every(Looking::WindowChanged, 0.0);
    assert!(watch.fired(&a, &b, 0.0).is_some());
    assert!(watch.fired(&b, &c, 0.0).is_some());
}

/// A trigger from a watch is **not asked for**, and everything downstream turns on that word.
#[test]
fn what_a_watch_produces_was_never_asked_for() {
    let a = vec![Widget::new("A", "button", (0, 0, 1, 1), "Paint")];
    let b = vec![Widget::new("A", "button", (0, 0, 1, 1), "Character Map")];
    let mut watch = Watch::new(Looking::WindowChanged);
    let fired = watch.fired(&a, &b, 1.0).expect("it changed");
    assert!(!fired.asked, "an unprompted effect is only downgraded because of this field");
    assert_eq!(fired.text, "the window changed to 'Character Map'");
}

/// **The first poll has no `before` and returns nothing**, whatever happened.
#[test]
fn the_first_look_reports_nothing_rather_than_the_whole_desktop() {
    let after = vec![Widget::new("Search", "button", (0, 0, 1, 1), "Paint"),
                     Widget::new("B", "button", (0, 0, 1, 1), "Paint")];
    let mut watches = vec![Watch::new(Looking::Appeared("search".into())),
                           Watch::new(Looking::WindowChanged)];
    //: Treating an empty screen as a previous state would report every widget as newly appeared
    //: and the window as newly changed — a burst of triggers the moment the agent starts.
    assert!(poll(&mut watches, &[], &after, 1.0).is_empty());
    //: With something to compare against, they fire.
    let before = vec![Widget::new("B", "button", (0, 0, 1, 1), "Paint")];
    assert_eq!(poll(&mut watches, &before, &after, 2.0).len(), 1);
    //: And no watches at all costs one screen read and starts nothing.
    assert!(poll(&mut [], &before, &after, 3.0).is_empty());
}
