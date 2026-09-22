//! Can argmind put text in a box without touching the keyboard?
//!
//! ```text
//!   typing              dry — resolves and reports, writes nothing
//!   typing --live       actually puts the text in
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
//! ## The four things it has to answer, and two arms so it can
//!
//! | | |
//! |---|---|
//! | **does it land** | is the text in the box afterwards, read back out of the control |
//! | **is it the same text** | `SetValue` takes a string; keys take UTF-16 units. Korean is the case |
//! | **does it refuse the case it must** | a box with something in it already: keys, never `SetValue` |
//! | **does it cost the keyboard** | what had focus before, and what has it after |
//!
//! Both arms on the same box, one after the other, so the comparison is not against a remembered
//! number from another day.
//!
//! ## What it will not do
//!
//! **It types only into a window it launched itself.** `Launched::hands()` is `Reach::OnlyProcess`,
//! and `set_value` is given the same guard from the focused element's own process rather than from
//! a pixel — without that a harness restricted to its own windows could still write into somebody
//! else's, and the rule would hold for one hand and not the other.
//!
//! **It takes the foreground**, so it cannot be run from the session driving it. That is the same
//! constraint `clicking` has and the same reason: a program cannot steal focus from the thing that
//! would lose it and still measure what happened.

use std::thread::sleep;
use std::time::{Duration, Instant};

use argmind_core::contract::{Eyes, Hands, Widget};
use argmind_env_windows::launch::{foreground_is, to_front, Launched};
use argmind_env_windows::{open_automation, WindowsEyes};

/// Something with a plain edit box that Windows has had for twenty years.
const APP: &str = "notepad.exe";

/// **Korean and ASCII**, because the two paths carry text differently: `SetValue` hands over a
/// `BSTR` and the keyboard hands over UTF-16 code units one at a time. A test in English would pass
/// on a path that mangles the thing this program mostly says.
const SAYS: [&str; 2] = ["hello", "안녕하세요 반갑습니다"];

fn edit_in(eyes: &mut WindowsEyes, pid: u32) -> Option<Widget> {
    eyes.foreground().into_iter()
        .find(|w| w.role == "edit" || w.role == "document")
        .filter(|_| foreground_is(pid))
}

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let live = std::env::args().any(|a| a == "--live");
    println!("\n  키를 누르지 않고 글자를 넣을 수 있는가 — {}",
             if live { "LIVE, 실제로 씁니다" } else { "DRY, 아무것도 쓰지 않습니다" });
    println!("  이 하네스가 직접 띄운 창에만 씁니다. 앞면을 가져갑니다.\n");

    let launched = match Launched::start(APP) {
        Ok(l) => l,
        Err(e) => {
            println!("  {APP} 를 띄우지 못했습니다 ({e})");
            return 2;
        }
    };
    let Some(window) = launched.window(Duration::from_secs(6)) else {
        println!("  {APP} 가 창을 열지 않았습니다");
        return 2;
    };
    let pid = launched.pid;
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

    let mut rows: Vec<(&str, &str, bool, String, f64)> = Vec::new();
    for arm in ["키보드", "ValuePattern"] {
        for said in SAYS {
            to_front(window);
            sleep(Duration::from_millis(500));
            if !foreground_is(pid) {
                println!("  건너뜀  {arm:<12} 다른 창이 앞으로 왔습니다");
                continue;
            }
            let Some(box_) = edit_in(&mut eyes, pid) else {
                println!("  건너뜀  {arm:<12} 편집 상자를 찾지 못했습니다");
                continue;
            };
            //: **Emptied between arms through the same hands**, so the second arm meets the
            //: condition `set_value` requires rather than whatever the first left behind.
            let mut hands = launched.hands();
            if live {
                hands = hands.arm();
            }
            hands.click(&box_);
            sleep(Duration::from_millis(200));
            if live {
                hands.press("ctrl+a");
                hands.press("delete");
                sleep(Duration::from_millis(200));
            }

            let mut hands = launched.hands();
            if arm == "ValuePattern" {
                hands = hands.with_typing_patterns();
            }
            if live {
                hands = hands.arm();
            }
            hands.click(&box_);
            sleep(Duration::from_millis(200));
            let began = Instant::now();
            hands.type_text(said);
            let took = began.elapsed().as_secs_f64() * 1000.0;
            sleep(Duration::from_millis(300));

            let landed = argmind_env_windows::patterns::focused_value(&automation).unwrap_or_default();
            let ok = !live || landed.trim() == said;
            println!("  {arm:<12} {:<12} {:>6.0} ms  {}  읽어보니 {:?}",
                     if said.is_ascii() { "ascii" } else { "한글" }, took,
                     if ok { "맞음" } else { "다름" }, landed.chars().take(24).collect::<String>());
            rows.push((arm, said, ok, landed, took));
        }
    }

    //: **The case it has to refuse**, and the one that would delete somebody's work if it did not.
    println!();
    to_front(window);
    sleep(Duration::from_millis(400));
    if let Some(box_) = edit_in(&mut eyes, pid) {
        let mut hands = launched.hands().with_typing_patterns();
        if live {
            hands = hands.arm();
        }
        hands.click(&box_);
        sleep(Duration::from_millis(200));
        //: Whatever the last arm left in the box is what makes this the *already has something*
        //: case; if the box is empty the question has not been asked.
        let before = argmind_env_windows::patterns::focused_value(&automation).unwrap_or_default();
        hands.type_text("추가");
        sleep(Duration::from_millis(300));
        let after = argmind_env_windows::patterns::focused_value(&automation).unwrap_or_default();
        if before.is_empty() {
            println!("  상자가 비어 있어 '이미 뭔가 있을 때' 를 묻지 못했습니다");
        } else if after.contains(&before) {
            println!("  이미 있던 것을 지우지 않음 — 키보드로 넘어갔습니다 (앞 {:?})",
                     before.chars().take(16).collect::<String>());
        } else {
            println!("  **이미 있던 것이 사라졌습니다** — 좁은 조건이 지켜지지 않았습니다");
        }
    }

    println!();
    println!("  ── 읽기 ──");
    let by = |arm: &str| rows.iter().filter(|r| r.0 == arm).collect::<Vec<_>>();
    for arm in ["키보드", "ValuePattern"] {
        let mine = by(arm);
        if mine.is_empty() {
            println!("  {arm:<12} 잰 것이 없습니다");
            continue;
        }
        let right = mine.iter().filter(|r| r.2).count();
        let mean = mine.iter().map(|r| r.4).sum::<f64>() / mine.len() as f64;
        println!("  {arm:<12} {right}/{} 맞음, 평균 {mean:.0} ms", mine.len());
    }
    if !live {
        println!("\n  DRY 였으므로 '맞음' 은 글자가 들어갔다는 뜻이 아닙니다 — 경로가 골라졌다는 \
뜻입니다. 숫자를 쓰려면 --live 로 다시 돌리세요.");
    }
    //: Notepad is left with text in it on purpose: nothing here saves, and a window closed by the
    //: harness would take the evidence with it.
    println!("\n  메모장은 열어 둡니다. 저장하지 않았으니 그냥 닫으시면 됩니다.");
    0
}
