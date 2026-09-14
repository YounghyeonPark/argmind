//! Does the point we would click actually land on the thing we chose?
//!
//! ```text
//!   pointing              (a) alone — reads the screen, touches nothing
//!   pointing --cursor     also (b): moves the real pointer and puts it back
//! ```
//!
//! **The other half of end-to-end accuracy.** Choosing was measured at 18 of 28 and the recorded
//! limit was that *"the click half is unmeasured, so 18/28 is a claim about choosing and not about
//! the whole loop"*. Two separate questions:
//!
//! **(a) Is a widget's centre on the widget?** Read-only. UI Automation says a control's rectangle
//! is *here*; asking what is at the middle of that rectangle can answer something else entirely —
//! an overlapping pane, a parent, nothing at all — and then a click resolves to whatever occupies
//! the point.
//!
//! **(b) Does the cursor land where it is sent?** Coordinates read from a DPI-aware process and
//! written from a DPI-unaware one can be virtualised. If the two disagree every click is offset by
//! the scaling factor and nothing lands — a failure that would look exactly like the model choosing
//! badly.
//!
//! **(b) moves the real mouse pointer**, so it is opt-in and the original position is put back in
//! every exit path. Nothing is clicked: no button is ever pressed by this file. `pointer_theft`
//! exists because yanking the pointer out from under somebody using the machine is a real cost, and
//! this measurement is not worth paying it by accident.

use std::collections::BTreeMap;
use std::path::PathBuf;

use argmind_core::contract::{Eyes, Widget};
use argmind_env_windows::{cursor_round_trip, screen_metrics, WindowsEyes};

fn main() {
    let _root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cursor = std::env::args().any(|a| a == "--cursor");
    let sample: usize = std::env::args().skip(1).filter(|a| a != "--cursor")
        .filter_map(|a| a.parse().ok()).next().unwrap_or(40);

    let (vw, vh, pw, ph) = screen_metrics();
    println!("  virtual screen {vw}x{vh}   primary {pw}x{ph}");

    let mut eyes = WindowsEyes::new().expect("UI Automation would not open");
    let widgets: Vec<Widget> = eyes.foreground();
    if widgets.is_empty() {
        println!("  nothing in the foreground window exposes a name. Put a normal window in front.");
        std::process::exit(2);
    }
    println!("  {} widgets in {:?}\n", widgets.len(),
             widgets[0].window.chars().take(60).collect::<String>());

    // ---- (a) is the centre on the widget? ----------------------------------------------------
    let mut tally: BTreeMap<&str, usize> = BTreeMap::new();
    let mut misses: Vec<(&Widget, String, String)> = Vec::new();
    for w in &widgets {
        let (cx, cy) = w.centre();
        match eyes.at_point(cx, cy) {
            None => {
                *tally.entry("no element there").or_default() += 1;
                misses.push((w, "(nothing)".into(), String::new()));
            }
            Some((name, role)) => {
                if name == w.name {
                    *tally.entry("the same thing").or_default() += 1;
                //: A parent or a child of the right control is a different failure from a
                //: completely unrelated one, and the fix is different too.
                } else if (!name.is_empty() && w.name.contains(&name))
                    || (!w.name.is_empty() && name.contains(&w.name)) {
                    *tally.entry("a containing or contained name").or_default() += 1;
                } else {
                    *tally.entry("something else").or_default() += 1;
                    misses.push((w, name, role));
                }
            }
        }
    }
    println!("  (a) what is actually at each widget's centre, over all {}:", widgets.len());
    let mut ranked: Vec<(&&str, &usize)> = tally.iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(a.1));
    for (k, v) in ranked {
        println!("      {v:>4}  ({:>4.1}%)  {k}", 100.0 * *v as f64 / widgets.len() as f64);
    }
    if !misses.is_empty() {
        println!("\n      the first ten that resolve to something else:");
        for (w, name, role) in misses.iter().take(10) {
            println!("        {:<14} {:<26} -> {:<12} {}",
                     w.role.chars().take(14).collect::<String>(),
                     w.name.chars().take(26).collect::<String>(),
                     role.chars().take(12).collect::<String>(),
                     name.chars().take(26).collect::<String>());
        }
    }

    // ---- (b) the write path ------------------------------------------------------------------
    if !cursor {
        println!("\n  (b) not run — it moves the real mouse pointer. Pass --cursor to measure the \
                  write path.");
        return;
    }
    let points: Vec<(i64, i64)> = widgets.iter().take(sample).map(Widget::centre).collect();
    let pairs = cursor_round_trip(&points);
    let off: Vec<&((i64, i64), (i64, i64))> = pairs.iter().filter(|(s, g)| s != g).collect();
    println!("\n  (b) cursor: {} of {} points came back exactly",
             pairs.len() - off.len(), pairs.len());
    match off.first() {
        Some((s, g)) => println!("      first disagreement: sent {s:?}, read back {g:?}  -> the \
                                  write path IS scaled, and every click is offset"),
        None => println!("      the write path is not scaled — SetCursorPos and UI Automation \
                          share one coordinate space"),
    }
}
