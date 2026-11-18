//! **The application's hands answer with the real hands, for every question the trait asks.**
//!
//! `argmind-serve`'s `surface::Touching` is the one `Hands` the application has: the desktop's or
//! the pane's, on one switch. `Hands` gives most of its methods a default — *cannot tell*, *cannot
//! press a point* — so an environment that has nothing to say compiles. **A wrapper that forgets one
//! compiles too, and answers the default for hands that could have answered.** Until 2026-11-19
//! `Touching` forwarded `into_a_secret` and nothing added after it: `typing_lands`, `landed` and
//! `click_inside` were the trait's defaults in the application, so the screen checks of 2026-10-21
//! never ran there and a point inside a place could not be pressed. Every test passed, because
//! each check was measured on the hands themselves.
//!
//! So this reads the trait's methods out of `contract.rs` and asks that `impl Hands for Touching`
//! defines each one — a method added to the trait later fails here until the wrapper forwards it.
//!
//! Checked by breaking it: `landed` taken out of `Touching` fails, and so does a new method added to
//! the trait and not to the wrapper.

use argmind_rules::root;

/// The `fn` names inside the block that starts at `head` — up to the first line that is exactly `}`.
fn methods_in(source: &str, head: &str) -> Vec<String> {
    let start = source.find(head).unwrap_or_else(|| panic!("`{head}` is not there any more"));
    source[start..].lines().skip(1).take_while(|l| l.trim_end() != "}")
        .filter_map(|l| {
            let t = l.trim_start();
            let t = t.split("//").next().unwrap_or("");
            t.strip_prefix("fn ").map(|r| r.split(|c: char| c == '(' || c == '<').next().unwrap_or("").to_string())
        })
        .collect()
}

#[test]
fn touching_forwards_every_method_hands_has() {
    let contract = std::fs::read_to_string(root().join("native/argmind-core/src/contract.rs")).unwrap();
    let surface = std::fs::read_to_string(root().join("native/argmind-serve/src/surface.rs")).unwrap();
    let asked = methods_in(&contract, "pub trait Hands");
    assert!(asked.len() >= 8, "read too few of the trait's methods to mean anything: {asked:?}");
    let answered = methods_in(&surface, "impl Hands for Touching");
    let missing: Vec<&String> = asked.iter().filter(|m| !answered.contains(m)).collect();
    assert!(missing.is_empty(),
            "the application's hands answer the trait's default for {missing:?} — forward them in surface.rs");
}
