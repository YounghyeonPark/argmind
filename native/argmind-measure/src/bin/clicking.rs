//! Does a confirmed click actually do the thing? **The last unmeasured link.**
//!
//! ```text
//!   clicking             launch Character Map, three goals, click for real
//!   clicking --dry       everything except pressing the button
//! ```
//!
//! Everything else in this chain is measured and this was not. Choosing is measured. The cursor
//! goes where it is sent. A widget's centre resolves back to that widget about seven times in ten.
//! And then nobody pressed the button: every run of the application was dry, so *"the person
//! confirms, the click lands, the thing they wanted happens"* had never once been observed.
//!
//! ## What makes this safe to run on a machine somebody is using
//!
//! **Two independent guards, and neither is a comment.** The hands are `Reach::OnlyProcess`, so
//! before every click the process owning the window under the point is looked up and anything else
//! is refused. And the foreground window is checked against the same process id before each goal —
//! a click goes to a pixel, so if anything has come to the front the click would land in it. When
//! that happens the goal is **skipped and reported as skipped**, not retried.
//!
//! Character Map is chosen because it is small, it is launched and closed by this process, and
//! nothing it does can lose somebody's work.
//!
//! ## What "it worked" means, decided before running
//!
//! A click that lands and does nothing is indistinguishable from a click that missed, so each goal
//! carries **two** checks, both written before anything ran:
//!
//! * `changed` — did the set of named things differ afterwards at all? Weak, but it separates
//!   "something happened" from "nothing happened", which is the failure this exists to catch.
//! * `expected` — the specific thing that should be true afterwards. Strong, and **it can be wrong
//!   about the application rather than about the agent**: if Character Map's advanced view exposes
//!   no control named "Search", this reports a failure that is mine. Both are printed so the two
//!   cannot be confused.
//!
//! ## This cannot be run from the session that is driving it — 2026-09-16
//!
//! **The thing being measured is the thing that breaks the measurement.** It takes the foreground
//! and presses buttons; run from a terminal inside the session that started it, that steals focus
//! from the window the person is working in, and the run is interrupted by its own first act. Three
//! attempts from inside such a session were cut off, and an earlier hypothesis that the *screen was
//! locked* was invented to explain the wreckage — the input desktop was `Default` throughout.
//!
//! Open a separate console, give it focus, run the binary, and leave the keyboard alone:
//!
//! ```text
//!   native/target/debug/clicking.exe        run from C:/dev/argmind
//! ```
//!
//! ## Measured 2026-09-16 — **2 of 3, n = 1**
//!
//! The Python scored `3 of 3` at n=3 on the same three goals and under the same condition — one
//! Character Map for all three, so the second goal meets a screen the first has already changed.
//! **This is one run against three and the model is noisy** (choosing reads 21, 23 and 24 across
//! runs of one configuration), so it is a number, not a regression.
//!
//! | goal | chosen | outcome |
//! |---|---|---|
//! | 고급 옵션 켜줘 | `Advanced view` | WORKED — `+ ["Character set :", "Go to Unicode :", "Group by :", "Search"]` |
//! | 글꼴 목록 열어줘 | `Open` | CHANGED — `+ ["DOS: Arabic", "DOS: Baltic", …]`, which is the **character set** list |
//! | 이 창 최소화해 | `Minimize` | WORKED — `- ["Close", "Maximize", "Minimize", "System"]` |
//!
//! ## What the one miss says, and it is not about the model
//!
//! **`name` and `role` together are still not a key.** In basic view Character Map has exactly one
//! button named `Open`, at `(391, 107)`, immediately beside the `Font :` combo box that spans
//! `x 86..407` — it is that combo's drop-down. The first goal turned advanced view on, which adds
//! more combo boxes; the second goal asked for the font list, the model named `Open`, and what
//! opened was the **character set** list. So advanced view has at least two buttons named `Open`,
//! and the agent resolved to the wrong one. The delta is what proves which.
//!
//! **The model could not have done better.** It is shown `name, role` pairs; three identical
//! `Open / button` entries give it no way to name the one it means. This is the same defect as the
//! `text` label reading `Font :` in front of a `combo box` of the same name — the one the `role`
//! field was added to fix — one level deeper.
//!
//! **And the tie-break does not break this tie.** [`argmind_core::session::Session::find`] picks
//! the candidate nearest to where a same-named widget was last seen, and it finds that anchor with
//! `.find()` — *the first* match in the previous reading. With duplicates the anchor is itself
//! arbitrary, so the nearest-to-it is arbitrary too. It disambiguates a control that moved; it
//! cannot disambiguate a control that has copies.

use std::collections::BTreeSet;
use std::thread::sleep;
use std::time::Duration;

use argmind_core::contract::Speaks;
use argmind_core::intent::Trigger;
use argmind_core::session::Session;
use argmind_env_ollama::Local;
use argmind_env_windows::hands::Did;
use argmind_env_windows::{foreground_is, is_minimised, pointing, Pointing, to_front, windows_of,
                          Launched, WindowsEyes};

struct Model(Local);

impl Speaks for Model {
    fn ask(&mut self, prompt: &str) -> String {
        self.0.ask(prompt).map(|r| r.text).unwrap_or_default()
    }
}

/// What should be true afterwards. **Written before the run.**
enum Expect {
    /// A name containing this appears among the process's windows.
    Appears(&'static str),
    /// Windows itself reports the launched window minimised.
    Minimised,
}

impl Expect {
    fn says(&self) -> String {
        match self {
            Expect::Appears(n) => format!("a name containing {n:?} appears"),
            Expect::Minimised => "Windows reports the window minimised".into(),
        }
    }
    fn holds(&self, before: &BTreeSet<String>, after: &BTreeSet<String>, main: isize) -> bool {
        match self {
            Expect::Appears(n) => after.difference(before)
                .any(|name| name.to_lowercase().contains(&n.to_lowercase())),
            Expect::Minimised => is_minimised(main),
        }
    }
}

const GOALS: [(&str, Expect); 3] = [
    ("고급 옵션 켜줘", Expect::Appears("search")),
    ("글꼴 목록 열어줘", Expect::Appears("arial")),
    ("이 창 최소화해", Expect::Minimised),
];

/// Every named thing across **every** window the process owns.
///
/// Not just the launched one: a combo box's drop-down is its own top-level window, so reading only
/// the main one reports "nothing happened" for a list that did open.
fn named(eyes: &mut WindowsEyes, pid: u32) -> BTreeSet<String> {
    windows_of(pid).into_iter().flat_map(|h| eyes.read(h)).map(|w| w.name).collect()
}

fn main() {
    //: **`std::process::exit` does not run destructors.** Every refusal after the launch leaves
    //: through here, or the application outlives the harness that promised to close it.
    let code = run();
    std::process::exit(code);
}

fn run() -> i32 {
    let dry = std::env::args().any(|a| a == "--dry");
    let app = std::env::args().skip(1).find(|a| a.ends_with(".exe"))
        .unwrap_or_else(|| "charmap.exe".into());

    //: **Asked before anything is clicked.** If the write path is scaled, every click is offset by
    //: the scaling factor and nothing lands — a failure that would look exactly like the model
    //: choosing badly.
    match pointing() {
        Pointing::Exact => {}
        //: **Two refusals, told apart.** One is a display setting and one is a locked screen, and
        //: sending somebody to check their DPI while the machine is merely locked wastes an
        //: afternoon.
        Pointing::Scaled => {
            println!("  the cursor does not land where it is sent on this display — refusing to click");
            return 2;
        }
        Pointing::Refused => {
            //: **What is true, and no further.** The first version of this said "the screen is
            //: locked"; measured, the input desktop was `Default` and `SetCursorPos` still failed
            //: with `ERROR_INVALID_HANDLE`, so that explanation was invented rather than found.
            println!("  the desktop will not take input — SetCursorPos is refused. Nothing here can reach the screen until it does; the cause is not established (see Pointing::Refused).");
            return 2;
        }
    }

    let mut launched = Launched::start(&app).expect("could not start it");
    let Some(main_window) = launched.window(Duration::from_secs(6)) else {
        println!("  {app} opened no visible window");
        return 2;
    };
    let pid = launched.pid;

    let eyes = WindowsEyes::new().expect("UI Automation would not open");
    //: Bound to the launched process, and dry unless this run is live.
    let hands = if dry { launched.hands() } else { launched.hands().arm() };
    let mut model = Local::new("qwen3:8b");
    model.think = Some(false);
    //: Two throwaway calls: the first two cost 4.7 s and 5.9 s and this run should not measure the
    //: weights loading.
    for _ in 0..2 {
        let _ = model.ask("Reply with the single word: ready");
    }
    let mut session = Session::new(eyes, hands, Model(model));

    println!("  {app} (pid {pid}) | qwen3:8b | {} | only this window's process\n",
             if dry { "DRY — nothing will be pressed" } else { "LIVE — it will press buttons" });

    let (mut worked, mut moved) = (0, 0);
    for (goal, expect) in &GOALS {
        to_front(main_window);
        sleep(Duration::from_millis(400));
        if !foreground_is(pid) {
            println!("  SKIPPED   {:<22} another window came to the front", trim(goal));
            continue;
        }

        session.observe();
        let before = named(&mut session.eyes, pid);
        let intent = session.decide(&Trigger::said(goal));
        let clicks: Vec<_> = intent.effects.iter().filter(|e| e.kind == "click").cloned().collect();
        let Some(click) = clicks.first() else {
            println!("  NO CLICK  {:<22} said {:?}", trim(goal), trim(&intent.say));
            continue;
        };
        //: **Re-resolved immediately before acting**, because a listed widget is gone 13 times in
        //: 14 over the second a call takes.
        let Some(target) = session.find(&click.target, &click.role) else {
            println!("  GONE      {:<22} {:?} was not there to click", trim(goal),
                     trim(&click.target));
            continue;
        };
        if !foreground_is(pid) {
            println!("  SKIPPED   {:<22} the front changed while deciding", trim(goal));
            continue;
        }

        session.perform(std::slice::from_ref(click));
        let did = session.hands.done.last().cloned();
        sleep(Duration::from_millis(900));
        let after = named(&mut session.eyes, pid);
        let changed = after != before;
        let met = expect.holds(&before, &after, main_window);
        if met {
            worked += 1;
        }
        if changed {
            moved += 1;
        }
        let mark = if met { "WORKED" } else if changed { "CHANGED" } else { "NOTHING" };
        println!("  {mark:<9} {:<22} {:?} | {}: {met}", trim(goal), trim(&target.name),
                 expect.says());
        //: **The delta, always.** A boolean cannot be diagnosed after the fact, and the first run
        //: of this measurement left two different causes indistinguishable by keeping only one.
        let gained: Vec<&String> = after.difference(&before).take(4).collect();
        let lost: Vec<&String> = before.difference(&after).take(4).collect();
        if !gained.is_empty() || !lost.is_empty() {
            println!("            + {gained:?}   - {lost:?}");
        } else if !changed {
            println!("            nothing at all appeared or disappeared");
        }
        if let Some(Did::Elsewhere { owner }) = did {
            println!("            REFUSED by the reach guard: the point belonged to {owner:?}");
        }

        session.hands.press_key("escape");
        sleep(Duration::from_millis(400));
    }

    println!("\n  {worked} of {} did the expected thing; {moved} changed the window at all",
             GOALS.len());
    //: **Passed the guard and actually went out are two numbers**, and a dry run has the first
    //: without the second. Printing one of them as the other is how a dry run comes to read as
    //: evidence that something was pressed.
    let sent = session.hands.done.iter().filter(|d| matches!(d, Did::Sent { live: true })).count();
    println!("  {} clicks passed the reach guard, {sent} actually went out;               refused as elsewhere {}, malformed {}",
             session.hands.clicked, session.hands.elsewhere, session.hands.refused);
    if session.eyes.vanished > 0 {
        println!("  ({} elements died between being listed and being read)", session.eyes.vanished);
    }
    launched.close();
    0
}

fn trim(s: &str) -> String {
    s.chars().take(22).collect()
}
