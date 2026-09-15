//! Why does a widget's centre not lead back to the widget? **Read-only, clicks nothing.**
//!
//! ```text
//!   ancestry                       Character Map, every combo box, button, check box, list, edit
//!   ancestry mspaint.exe           somewhere else
//!   ancestry charmap.exe 12        look further up
//! ```
//!
//! Hands operate a control through UI Automation when they can find the element, and they find it
//! by asking what is at the widget's centre and then walking **up** until the name and the role
//! match. Measured: that fails for the one case the whole mechanism was built for — a combo box,
//! whose centre sits on the edit control inside it.
//!
//! This prints the chain, so the reason is visible instead of guessed, and the patterns the
//! matching element actually offers — **all twenty-two of them**, because the four the hands were
//! built around were picked from a guess about which controls expose what, and a guess is not a
//! survey.
//!
//! It launches its own window and closes it. It presses nothing and clicks nothing.

use std::time::Duration;

use argmind_core::contract::Eyes;
use argmind_env_windows::{bring_to_front, windows_of, Launched, WindowsEyes};

const INTERESTING: [&str; 5] = ["combo box", "button", "check box", "list", "edit"];

fn main() {
    //: **`std::process::exit` does not run destructors**, so every refusal below has to leave
    //: through here or the launched application outlives the harness. It did, once: a run that
    //: refused left Character Map open, and the orphan held the console pipe so the shell that
    //: started it never returned. `Drop` is only a guarantee if nothing jumps over it.
    let code = run();
    std::process::exit(code);
}

fn run() -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let app = args.iter().find(|a| a.ends_with(".exe")).cloned()
        .unwrap_or_else(|| "charmap.exe".into());
    let hops: usize = args.iter().filter_map(|a| a.parse().ok()).next().unwrap_or(8);

    //: **Walk what is already in front, instead of launching.** The launch path cannot run on a
    //: machine where the shell is not the foreground process — Windows refuses the front to
    //: anybody who does not already have it — and the question this asks is about point
    //: resolution, which needs the window on top whichever way it got there. `pointing` reads the
    //: foreground window for the same reason.
    if args.iter().any(|a| a == "--foreground") {
        let mut eyes = WindowsEyes::new().expect("UI Automation would not open");
        let widgets = eyes.foreground();
        if widgets.is_empty() {
            println!("  nothing in the foreground window exposes a name.");
            return 2;
        }
        println!("  {:?}: {} widgets, walking up to {hops} hops
",
                 widgets[0].window.chars().take(50).collect::<String>(), widgets.len());
        let (found, lost) = walk(&mut eyes, &widgets, hops);
        println!("  {found} of {} resolved back to themselves within {hops} hops; {lost} did not",
                 found + lost);
        return 0;
    }

    let mut launched = Launched::start(&app).expect("could not start it");
    let Some(main_window) = launched.window(Duration::from_secs(6)) else {
        println!("  {app} opened no visible window");
        return 2;
    };
    //: **This resolves by point, so it measures whatever is on top.** Found the hard way: the first
    //: run of this port reported `0 of 12` with every chain rooted in `edit "Connected"` — a remote
    //: desktop window that was covering the coordinates. That is not the ancestor walk failing, it
    //: is the harness reading the wrong window, and the two look identical in the output. So the
    //: launched window is brought to the front and **the front is verified**; a run that cannot get
    //: it refuses rather than reporting a clean zero.
    if !bring_to_front(main_window, launched.pid, Duration::from_secs(3)) {
        println!("  {app} is not in front — Windows refuses the front to a process that does not \
                  already have it, and every point would then resolve to whatever is covering it.");
        println!("  Refusing rather than reporting a zero about the covering window. Pass \
                  --foreground to walk whatever IS in front instead.");
        return 2;
    }
    let handles = windows_of(launched.pid);
    let mut eyes = WindowsEyes::new().expect("UI Automation would not open");
    let widgets = eyes.read(handles[0]);
    println!("  {app}: {} widgets, walking up to {hops} hops\n", widgets.len());

    let (found, lost) = walk(&mut eyes, &widgets, hops);
    println!("  {found} of {} resolved back to themselves within {hops} hops; {lost} did not",
             found + lost);
    launched.close();
    0
}

/// The walk itself — **one definition**, so the launched mode and the foreground mode cannot drift
/// into measuring two slightly different things.
fn walk(eyes: &mut WindowsEyes, widgets: &[argmind_core::contract::Widget], hops: usize)
        -> (usize, usize) {
    let (mut found, mut lost) = (0, 0);
    for w in widgets.iter().filter(|w| INTERESTING.contains(&w.role.as_str())) {
        let (cx, cy) = w.centre();
        println!("  {} {:?} centre ({cx}, {cy})", w.role,
                 w.name.chars().take(28).collect::<String>());
        let chain = eyes.ancestry_at(cx, cy, hops, Some((&w.name, &w.role)));
        if chain.is_empty() {
            println!("      nothing at that point");
            lost += 1;
            println!();
            continue;
        }
        for (i, hop) in chain.iter().enumerate() {
            if hop.died {
                println!("      {i}: (died mid-read)");
                continue;
            }
            println!("      {i}: {:<16} {:?}{}", hop.role.chars().take(16).collect::<String>(),
                     hop.name.chars().take(34).collect::<String>(),
                     if hop.matched { "  <== name and role match" } else { "" });
            if hop.matched {
                println!("      supports: {}", if hop.patterns.is_empty() {
                    "(none)".to_string()
                } else {
                    hop.patterns.join(", ")
                });
            }
        }
        //: **A walk that ran out of hops is not the same as one that never started**, and the
        //: tally keeps them apart — the first says the ancestor lookup is too shallow, the second
        //: says the point resolves to nothing at all.
        if chain.iter().any(|h| h.matched) {
            found += 1;
        } else {
            lost += 1;
            println!("      no ancestor within {hops} hops matched the name and role");
        }
        println!();
    }

    (found, lost)
}
