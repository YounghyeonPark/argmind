//! Can argmind put text in a box without touching the keyboard?
//!
//! ```text
//!   typing                    dry — resolves and reports, writes nothing
//!   typing --live             actually puts the text in
//!   typing --live mspaint.exe some other classic Win32 application
//! ```
//!
//! **Why this is not settled by the clicking answer.** Control patterns ship for clicking since
//! 2026-09-22, because a pattern calls a method on the control and leaves the pointer where the
//! person put it — measured unmoved 3 of 3 against dragged to the widget. Typing has the same
//! collision one device over: `KEYEVENTF_UNICODE` goes to whatever holds focus, so argmind types
//! into whatever you are typing into.
//!
//! **And the obvious fix is not the same operation.** `argmind-env-linux` types through
//! `EditableText.InsertText`, which inserts *at an offset*. UI Automation has no insert anywhere —
//! `ValuePattern::SetValue` replaces the whole value and `TextPattern` only reads. So the shape
//! that could be honest is narrow: **set only into a control that is empty**, keys otherwise. That
//! is what `patterns::set_value` does and what this measures.
//!
//! ## A fresh window for every case, and the first version is why
//!
//! The first version typed four times into one box and cleared it between arms with `ctrl+a` and
//! `delete`. **`ctrl+a` is not a named key** — `KEY_NAMES` is fifteen single keys, listed rather
//! than guessed precisely so an agent cannot invent `alt+F4` — so the clear was refused, every arm
//! after the first ran on a dirty box, and the run read:
//!
//! ```text
//!   키보드        ascii   맞음   "hello\r"
//!   키보드        한글    다름   "hello안녕하세요 반갑습니다\r"      <- never cleared
//! ```
//!
//! and then reported that the narrow condition had failed. **It had not; the harness had.** So
//! there is no clearing here at all: each case gets its own application, launched and killed, and
//! an empty box is a property of a new window rather than of a step that can be refused.
//!
//! ## What it answers
//!
//! | | |
//! |---|---|
//! | **does it land** | read back out of the control, by point rather than by focus |
//! | **is it the same text** | `SetValue` takes a `BSTR`, keys take UTF-16 units. Korean is the case |
//! | **where did it go** | the widget's value *and* the focused element's, because those coming apart is the one way `set_value` could write somewhere nobody looked |
//! | **does it refuse what it must** | a box with something in it already: keys, never `SetValue` |
//!
//! ## What it will not do
//!
//! **It types only into a window it launched itself.** `Launched::hands()` is `Reach::OnlyProcess`
//! and `set_value` is given the same guard from the focused element's own process rather than from
//! a pixel — without that, a harness restricted to its own windows could still write into somebody
//! else's, and the rule would hold for one hand and not the other.
//!
//! **Notepad will not do.** On Windows 11 `System32\notepad.exe` is a launcher for a packaged
//! application, so the pid this starts owns no window and the reach guard would correctly refuse
//! the one that appears. Character Map is a classic Win32 application and `clicking` drives it for
//! the same reason.
//!
//! **It takes the foreground**, so it cannot be run from the session driving it.
//!
//! ## What it said — 2026-09-22, Character Map
//!
//! | path | ascii | Korean | a box with something in it |
//! |---|---|---|---|
//! | keys | right, 4 ms | right, 7 ms | — |
//! | `ValuePattern` | right, 6 ms | right, 14 ms | **fell back to keys**: `"먼저"` → `"먼저추가"` |
//!
//! No warnings fired: every new window started empty and the value at the box matched the value at
//! the focused element in all four cases, so the text went where it was aimed. The narrow condition
//! held — a box that already held something was appended to and not replaced. The milliseconds are
//! the call returning rather than the text appearing, and at four to fourteen they decide nothing.
//!
//! **That run also made plain that the first design answered nothing.** `set_value` wrote to
//! `GetFocusedElement`, and getting focus onto the right control means clicking it — the collision
//! the path was supposed to avoid. So it now writes **at the point of the widget argmind just
//! clicked**: `WindowsHands` remembers the click, `type_string` takes it once, and an aim older
//! than `AIM_LASTS` is no aim. No prompt edit was needed, because the target was never in the
//! reply — it was in the effect before it.
//!
//! **What that buys, stated no wider than it is.** The click still happens, so the pointer still
//! moves; what changes is where the *text* goes. Between argmind's click and its typing there are
//! 250 ms, and in that time a person can click into a box of their own — the keyboard would put
//! argmind's sentence in theirs and this puts it in the one argmind aimed at. On top of that the
//! text arrives at once rather than character by character, and no IME sits in the middle of a
//! Korean string.
//!
//! **A second column says which path actually ran**, counted by the hands rather than inferred:
//! both arms produce the same text when both work, so a fallback nobody noticed would read as the
//! pattern arm succeeding.

use std::thread::sleep;
use std::time::{Duration, Instant};

use argmind_core::contract::{Eyes, Hands, Widget};
use argmind_env_windows::launch::{foreground_is, to_front, Launched};
use argmind_env_windows::patterns::value_at;
use argmind_env_windows::{open_automation, WindowsEyes};

/// A classic Win32 application with an ordinary edit box in its default view.
const APP: &str = "charmap.exe";

/// **Korean and ASCII**, because the two paths carry text differently: `SetValue` hands over a
/// `BSTR` and the keyboard hands over UTF-16 code units one at a time. A test in English alone
/// would pass on a path that mangles the thing this program mostly says.
const SAYS: [&str; 2] = ["hello", "안녕하세요 반갑습니다"];

/// One application, opened and ready, with the box to type into.
struct Fresh {
    app: Launched,
    box_: Widget,
}

fn fresh(exe: &str, eyes: &mut WindowsEyes) -> Option<Fresh> {
    let app = Launched::start(exe).ok()?;
    let window = app.window(Duration::from_secs(8))?;
    to_front(window);
    sleep(Duration::from_millis(600));
    if !foreground_is(app.pid) {
        return None;
    }
    //: `edit` before `document`: a document is usually the whole canvas and an edit is the field
    //: somebody types into.
    let seen = eyes.foreground();
    let box_ = seen.iter().find(|w| w.role == "edit")
        .or_else(|| seen.iter().find(|w| w.role == "document"))?
        .clone();
    Some(Fresh { app, box_ })
}

fn tidy(s: &str) -> String {
    s.trim_end_matches(['\r', '\n']).to_string()
}

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let live = std::env::args().any(|a| a == "--live");
    let exe = std::env::args().skip(1).find(|a| a.ends_with(".exe"))
        .unwrap_or_else(|| APP.into());
    println!("\n  키를 누르지 않고 글자를 넣을 수 있는가 — {}",
             if live { "LIVE, 실제로 씁니다" } else { "DRY, 아무것도 쓰지 않습니다" });
    println!("  {exe} 를 경우마다 새로 띄웁니다. 앞면을 가져가고, 끝나면 닫습니다.\n");

    let Ok(automation) = open_automation() else {
        println!("  UI Automation 이 열리지 않았습니다");
        return 2;
    };
    let mut eyes = match WindowsEyes::new() {
        Ok(e) => e,
        Err(e) => {
            println!("  눈이 열리지 않았습니다 ({e})");
            return 2;
        }
    };

    println!("  {:<13} {:<7} {:>8}   {:<6} {:<38} {}",
             "경로", "글자", "걸림", "판정", "실제 경로", "상자에서 읽은 것");
    let mut rows: Vec<(&str, bool)> = Vec::new();
    for arm in ["키보드", "ValuePattern"] {
        for said in SAYS {
            let Some(mut it) = fresh(&exe, &mut eyes) else {
                println!("  {arm:<13} {:<7} 창을 띄우지 못했습니다 — 패키지 앱이면 다른 \
프로세스로 넘깁니다. System32 의 고전 Win32 앱을 인자로 주세요.",
                         if said.is_ascii() { "ascii" } else { "한글" });
                continue;
            };
            let (cx, cy) = it.box_.centre();
            let mut hands = it.app.hands();
            if arm == "ValuePattern" {
                hands = hands.with_typing_patterns();
            }
            if live {
                hands = hands.arm();
            }
            hands.click(&it.box_);
            sleep(Duration::from_millis(250));
            //: **Empty because the window is new**, and reported rather than assumed — a box that
            //: started with something in it would send the pattern arm down the fallback and the
            //: two arms would quietly be one.
            let started = tidy(&value_at(&automation, cx, cy).unwrap_or_default());
            let began = Instant::now();
            hands.type_text(said);
            let took = began.elapsed().as_secs_f64() * 1000.0;
            let used = hands.by_value;
            //: **The reason, not just the fact.** A harness that says *it fell back* and not *why*
            //: cannot diagnose itself — this run reported the fallback four times out of four and
            //: nothing about which of `set_value`'s five refusals it was.
            let why = hands.why.clone();
            sleep(Duration::from_millis(350));

            let in_box = tidy(&value_at(&automation, cx, cy).unwrap_or_default());
            let ok = !live || in_box == said;
            //: **Which path actually ran**, counted by the hands rather than inferred from the
            //: result: the two arms produce the same text when both work, so a fallback nobody
            //: noticed would read as the pattern arm succeeding.
            let how = if used > 0 {
                "패턴".to_string()
            } else if why.is_empty() {
                "키보드".to_string()
            } else {
                format!("키보드 — {why}")
            };
            rows.push((arm, ok));
            println!("  {arm:<13} {:<7} {took:>6.0} ms   {:<6} {:<38} {:?}{}",
                     if said.is_ascii() { "ascii" } else { "한글" },
                     if ok { "맞음" } else { "다름" }, how,
                     in_box.chars().take(24).collect::<String>(),
                     if started.is_empty() { "" } else { "   ** 새 창인데 비어 있지 않았음" });
            it.app.close();
            sleep(Duration::from_millis(250));
        }
    }

    //: **The case that would delete somebody's work.** A fresh window, something put in with the
    //: keyboard — which appends — and then the pattern arm on top of it.
    println!();
    match fresh(&exe, &mut eyes) {
        None => println!("  '이미 뭔가 있을 때' 를 묻지 못했습니다 — 창이 뜨지 않았습니다"),
        Some(mut it) => {
            let (cx, cy) = it.box_.centre();
            let mut keys = it.app.hands();
            if live {
                keys = keys.arm();
            }
            keys.click(&it.box_);
            sleep(Duration::from_millis(250));
            keys.type_text("먼저");
            sleep(Duration::from_millis(350));
            let before = tidy(&value_at(&automation, cx, cy).unwrap_or_default());

            let mut pattern = it.app.hands().with_typing_patterns();
            if live {
                pattern = pattern.arm();
            }
            pattern.click(&it.box_);
            sleep(Duration::from_millis(250));
            pattern.type_text("추가");
            sleep(Duration::from_millis(350));
            let after = tidy(&value_at(&automation, cx, cy).unwrap_or_default());

            if !live {
                println!("  DRY 라 '이미 뭔가 있을 때' 는 묻지 못합니다 — --live 가 필요합니다");
            } else if before.is_empty() {
                println!("  키보드가 아무것도 넣지 못해 이 질문을 못 했습니다");
            } else if after.contains(&before) {
                println!("  이미 있던 것을 지우지 않았습니다 — 키보드로 넘어갔습니다. \
{before:?} -> {after:?}");
            } else {
                println!("  **이미 있던 것이 사라졌습니다** — 좁은 조건이 지켜지지 않았습니다. \
{before:?} -> {after:?}");
            }
            it.app.close();
        }
    }

    println!();
    println!("  ── 읽기 ──");
    for arm in ["키보드", "ValuePattern"] {
        let mine: Vec<_> = rows.iter().filter(|r| r.0 == arm).collect();
        if mine.is_empty() {
            println!("  {arm:<13} 잰 것이 없습니다");
            continue;
        }
        println!("  {arm:<13} {}/{} 맞음", mine.iter().filter(|r| r.1).count(), mine.len());
    }
    if !live {
        println!("\n  DRY 였으므로 '맞음' 은 글자가 들어갔다는 뜻이 아니라 경로가 골라졌다는 \
뜻입니다. 숫자를 쓰려면 --live 로 돌리세요.");
    }
    0
}
