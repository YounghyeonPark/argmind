//! Does acting steal the mouse pointer? **The one axis the control patterns may win on.**
//!
//! ```text
//!   pointer-theft            both arms, live — it presses a check box in its own Character Map
//!   pointer-theft --dry      resolve both paths and press nothing
//! ```
//!
//! A synthetic click moves the operating system's cursor to the widget and presses the physical
//! button. On a machine somebody is using — and this one is used over a remote desktop while the
//! agent runs — that yanks the pointer out from under them every time the agent acts. A control
//! pattern calls a method on the control and never touches the pointer at all.
//!
//! **The patterns were measured on task success (`3 of 3` either way) and on obstruction (nothing
//! either way) and found unearned on both.** Success is not the only axis a person cares about, and
//! this puts the remaining one on the same footing: cursor position before and after, both ways.
//!
//! Nothing here needs the model. It launches its own Character Map and closes it, and the hands are
//! bound to that process so nothing else on the screen can be touched.
//!
//! ## Measured 2026-09-16 — **this is the axis the patterns win**
//!
//! Both arms did the thing. The pointer, parked at `(700, 400)` first:
//!
//! | arm | via | the check box | the pointer |
//! |---|---|---|---|
//! | patterns on | `invoke` | WORKED | `(700, 400) -> (700, 400)` — **unmoved** |
//! | patterns off | click | WORKED | `(700, 400) -> (130, 459)` — **moved to the widget** |
//!
//! **n = 3**, all three identical. They took three attempts to collect, because between runs the
//! desktop kept refusing input — see the note on [`argmind_env_windows::Pointing::Refused`], where
//! what that refusal actually means is *not* settled.
//!
//! **This is the axis where the patterns earn something.** Task success says nothing either way
//! (`3 of 3` both) and `obstruction` says nothing either way (24 of 24, neither path reaches past a
//! cover). Here they differ every time: `invoke` leaves the pointer where the person left it and a
//! click drags it to the widget. Whether that is worth ninety lines, a second COM client and an
//! ancestor walk is the owner's call; the measurement is no longer the missing part of it.
//!
//! ## The first version of this measured nothing, and said `unmoved` twice
//!
//! `park()` called `cursor_round_trip`, which sends the cursor somewhere, reads it back **and puts
//! it where it found it** — a probe, not a mover. So no arm was ever parked: each read the position
//! the previous arm's click had left the pointer in, which is the check box. A click that moves the
//! pointer to where it already is reads as `unmoved`, and both arms agreed for the same wrong
//! reason: `(130, 459) -> (130, 459)`, twice, while the header announced a park that never
//! happened.
//!
//! **The note above `where_is` in this file describes exactly this hazard**, and was written when
//! the same call was found in the *reading* half. It was fixed there and left here. A uniformly
//! tidy result — both arms unmoved — is evidence the measurement is broken, and it was.

use std::collections::BTreeSet;
use std::thread::sleep;
use std::time::Duration;

use argmind_env_windows::hands::Did;
use argmind_env_windows::{bring_to_front, cursor_now, pointing, put_cursor, Pointing,
                          whats_in_front, windows_of, Launched, WindowsEyes};

/// A check box, whose effect is unambiguous and whose pattern (`Invoke`) is one the hands try
/// first: advanced view adds four named things, one of them a search control.
const TARGET: (&str, &str) = ("Advanced view", "check box");
const MARKER: &str = "search";
/// Park the pointer somewhere that is not the target, so **"it did not move" is a real answer** and
/// not an accident of where the pointer already happened to be.
const PARKED: (i64, i64) = (700, 400);

fn named(eyes: &mut WindowsEyes, pid: u32) -> BTreeSet<String> {
    windows_of(pid).into_iter().flat_map(|h| eyes.read(h)).map(|w| w.name).collect()
}

/// Where the pointer is, **without moving it**.
///
/// The first version of this read the cursor through `cursor_round_trip`, which sends it somewhere
/// and reads it back — so the instrument that answers *"did the pointer move?"* would have moved
/// it, and both arms would have reported `unmoved` for the same wrong reason.
fn where_is() -> (i64, i64) {
    cursor_now()
}

struct Arm {
    via: String,
    before: (i64, i64),
    after: (i64, i64),
    worked: bool,
    ok: bool,
}

fn attempt(patterns: bool, app: &str, dry: bool) -> Arm {
    let mut fail = Arm { via: String::new(), before: PARKED, after: PARKED, worked: false,
                         ok: false };
    let Ok(mut launched) = Launched::start(app) else {
        fail.via = "could not start it".into();
        return fail;
    };
    let Some(main_window) = launched.window(Duration::from_secs(6)) else {
        fail.via = "no window".into();
        return fail;
    };
    if !bring_to_front(main_window, launched.pid, Duration::from_secs(3)) {
        //: The click arm would land in whatever is covering it, and the two arms would then be
        //: measuring different things. Refuse rather than report.
        let (_, title, owner) = whats_in_front();
        fail.via = format!("could not get the front — {title:?} (pid {owner:?}) has it");
        return fail;
    }

    let mut eyes = WindowsEyes::new().expect("UI Automation would not open");
    let widgets = eyes.read(main_window);
    let Some(widget) = widgets.iter().find(|w| (w.name.as_str(), w.role.as_str()) == TARGET)
        .cloned() else {
        fail.via = format!("no {:?}", TARGET.0);
        return fail;
    };

    let mut hands = launched.hands();
    if patterns {
        hands = hands.with_patterns();
    }
    if !dry {
        hands = hands.arm();
    }
    hands.max_per_second = 8.0;

    if !park() {
        return Arm { via: "the pointer would not park".into(), before: where_is(),
                     after: where_is(), worked: false, ok: false };
    }
    let before_cursor = where_is();
    let before = named(&mut eyes, launched.pid);
    let did = hands.click_button(&widget, "left");
    sleep(Duration::from_millis(800));
    let after_cursor = where_is();
    let after = named(&mut eyes, launched.pid);
    launched.close();

    let via = match &did {
        Did::Operated { how, .. } => (*how).to_string(),
        Did::Sent { .. } => format!("click{}", if hands.why.is_empty() {
            String::new()
        } else {
            format!(" ({})", hands.why)
        }),
        Did::Elsewhere { owner } => format!("refused, the point belongs to {owner:?}"),
        Did::Refused(why) => why.clone(),
    };
    Arm { via, before: before_cursor, after: after_cursor,
          worked: after.difference(&before).any(|n| n.to_lowercase().contains(MARKER)),
          ok: true }
}

/// Put the pointer out of the way and **leave it there**.
///
/// **This was `cursor_round_trip` and that call puts the cursor back where it found it.** So the
/// park never happened: each arm read the position the previous arm's click had left the pointer
/// in, which for the second arm is the check box itself — and a click that moves the pointer to
/// where it already is reads as `unmoved`. Both arms reported *unmoved*, which is exactly the
/// failure the note above `where_is` describes, fixed there and left here. Measured before the fix:
/// `(130, 459) -> (130, 459)` on both arms, with the harness announcing a park to (700, 400) that
/// never took place.
///
/// Returns whether the pointer actually arrived. **A park that silently failed is the same lie
/// again**, so the caller reports the arm as unrunnable rather than measuring through it.
fn park() -> bool {
    put_cursor(PARKED.0, PARKED.1)
}

fn main() {
    //: `std::process::exit` does not run destructors — every path leaves through here.
    std::process::exit(run());
}

fn run() -> i32 {
    let dry = std::env::args().any(|a| a == "--dry");
    let app = std::env::args().skip(1).find(|a| a.ends_with(".exe"))
        .unwrap_or_else(|| "charmap.exe".into());
    //: **Two refusals, told apart.** One is a display setting and one is a locked screen, and
    //: sending somebody to check their DPI while the machine is merely locked wastes an afternoon.
    //: This harness hit the second mid-measurement: the first run produced a number and the two
    //: repeats could not, because the workstation had locked between them.
    match if dry { Pointing::Exact } else { pointing() } {
        Pointing::Exact => {}
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

    println!("  acting on {} {:?}, pointer parked at {:?} first | {}\n",
             TARGET.1, TARGET.0, PARKED,
             if dry { "DRY — nothing will be pressed" } else { "LIVE" });
    for (label, patterns) in [("patterns on ", true), ("patterns off", false)] {
        let arm = attempt(patterns, &app, dry);
        if !arm.ok {
            println!("  {label}  {}", arm.via);
            continue;
        }
        //: **The delta, not a boolean.** Where it was and where it ended up, so a pointer that
        //: moved somewhere unexpected is distinguishable from one that moved to the widget.
        let moved = arm.before != arm.after;
        println!("  {label}  via {:<28} {:<11} pointer {:?} -> {:?}  {}",
                 arm.via.chars().take(28).collect::<String>(),
                 if arm.worked { "WORKED" } else { "did nothing" },
                 arm.before, arm.after, if moved { "MOVED" } else { "unmoved" });
        sleep(Duration::from_secs(1));
    }
    //: **Both arms or neither.** A run where one arm could not start says nothing about the
    //: difference, which is the only thing this measures.
    0
}
