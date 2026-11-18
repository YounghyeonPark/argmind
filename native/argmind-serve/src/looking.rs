//! **Where the list is blind, a picture — and only there.** Since 2026-11-18.
//!
//! The rule this project hangs on is *the deterministic side says WHERE, the probabilistic side
//! says WHAT, and a model never emits a coordinate*. `--bin looking` measured both sides of it on
//! five drawn screens: where the list has a thing by name it is more exact than a vision model
//! (12 of 13 against 9), and the picture's misses are the dangerous kind — the row next to the
//! right one; where the list is blind — a character inside a grid listed as one pane, anything
//! inside a remote desktop — the picture found 5–6 of 6 and the list none. The owner chose: **the
//! picture only where the list is blind, always with a press.**
//!
//! So a picture of the screen reaches a model in one place — `app::look_inside`, which one turn
//! calls — and this file is everything that place decides with: when to look, what to show, how to
//! read the answer and what to keep. What comes back is held three ways:
//!
//! * **asked only when the list could not answer** — the person asked for something to be
//!   pressed, the turn chose nothing or chose a place the list cannot see into, and there is such a
//!   place on the screen ([`wants_a_look`]);
//! * **kept only where the list holds nothing smaller than a region** ([`landed`]) — a point that
//!   falls on a listed button or row is dropped, because there the list has already had its say
//!   and was the more exact of the two;
//! * **made into an effect that names the container, not the point** ([`effect`]): the container
//!   is re-resolved by name and role at the press and the point is a fraction of it, so the point
//!   moves with the container and cannot leave it — `argmind_core::intent::Inside`.
//!
//! And a person presses it: `app::propose` never runs an effect with `inside` set by itself,
//! whatever 확인 없이 실행 says. `argmind-rules` (`a_picture_points_only_where_the_list_is_blind`)
//! reads the source for that, for the one call, and for this file being the only maker of a point.
//!
//! **Off until somebody turns it on** — `looking`, a decidable key of its own — and the model is
//! `vision-model`'s, a need of its own. **Not run inside the application from here**: the pieces
//! were run on a real Character Map through these functions (`argmind-measure --example
//! pointing`); the turn that strings them together in `app.rs` was not.

use argmind_core::contract::Widget;
use argmind_core::intent::{Effect, Inside};

/// **The model that looks — the 8b, chosen by what it pressed, against what fits.**
///
/// The card first (`--bin looking --fit`, 2026-11-18, argmind running as the owner runs it): the
/// 4b stayed on beside the conversation's model *and* argmind's own work at 24 layers; the 8b fitted
/// beside the conversation (15.8 of 16.4 GB) and beside the work **put a model out on every call,
/// 9–10 s each**. Then the path itself, on the bars written first (`--arms gated`, the place cut
/// out): **the 8b met every one** — 4 of the 4 reachable, no row for an impossible request, no row
/// in the wrong place — and **the 4b missed two**, a row on the grid's middle for ★ and one on
/// 지금 설치 for a calculator that is not there. On a real Character Map (`--example pointing`) both
/// found 8 of 10, and the 4b pressed ★, which is not on the page, twice; the 8b declined it. A look
/// happens only on a request to press inside a place the list cannot see into, so the reload is
/// paid rarely, and a row pointing at the wrong thing would be paid every time.
pub const VISION_MODEL: &str = "qwen3-vl:8b-instruct";

/// **The smallest a listed thing can be and still be a place rather than a control** — chosen, not
/// measured, and **chosen twice**: 48×32 held on the drawn screens, which are at 100%, and on a real
/// Character Map at 150% it read as places every button (113×33), the caption buttons (71×44) and
/// the text labels (180×57). At 128×96 a label at 200% (~76 tall) is still not one, and Character
/// Map's grid (643×363), a canvas or a remote desktop always is. An unnamed icon in a toolbar is
/// therefore not reachable — the trial is the grid, the canvas and the remote desktop.
pub const REGION: (i64, i64) = (128, 96);

/// The widest picture sent, in pixels. `--bin looking` measured 1280×800; a wider window is
/// shrunk to this so a look costs what was measured rather than four times it.
pub const MOST_WIDE: usize = 1280;

/// **What the vision model is asked**, word for word what `--bin looking --arms gated` measured.
pub fn asking(request: &str) -> String {
    let request = request.replace('"', "'");
    format!("This is a screenshot of one window on a Windows PC. The person said: \"{request}\"\n\
             Find the one on-screen element they want clicked. Answer with JSON only: \
             {{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"what it is, in a few words\"}} with \
             coordinates from 0 to 1000 relative to the image width and height. If nothing in this \
             picture does what they asked, answer {{\"bbox_2d\": null}}.")
}

/// What came back.
#[derive(Clone, Debug, PartialEq)]
pub enum Saw {
    /// The middle of the box, in thousandths of the picture, and what the model called it.
    At { x: f64, y: f64, label: String },
    /// It said nothing there does it.
    Nothing,
    Unreadable,
}

/// The numbers after `key`, or `Some(vec![])` for a `null` there; `None` when the key is absent.
fn numbers_after(text: &str, key: &str) -> Option<Vec<f64>> {
    let at = text.find(key)?;
    let rest = text[at + key.len()..].trim_start_matches(|c: char| c == '"' || c == ':' || c.is_whitespace());
    if rest.starts_with("null") {
        return Some(vec![]);
    }
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in rest.chars() {
        if c.is_ascii_digit() || c == '.' || (c == '-' && cur.is_empty()) {
            cur.push(c);
            continue;
        }
        if !cur.is_empty() {
            if let Ok(n) = cur.parse() {
                out.push(n);
            }
            cur.clear();
        }
        if c == ']' || c == '}' || out.len() >= 4 {
            break;
        }
    }
    if !cur.is_empty() {
        if let Ok(n) = cur.parse() {
            out.push(n);
        }
    }
    Some(out)
}

/// The string after `"label"`, unescaped enough to show a person.
fn label_in(text: &str) -> String {
    let Some(at) = text.find("\"label\"") else { return String::new() };
    let rest = &text[at + 7..];
    let Some(open) = rest.find('"') else { return String::new() };
    let mut out = String::new();
    let mut escaped = false;
    for c in rest[open + 1..].chars() {
        match (escaped, c) {
            (true, c) => {
                out.push(c);
                escaped = false;
            }
            (false, '\\') => escaped = true,
            (false, '"') => break,
            (false, c) => out.push(c),
        }
    }
    out.chars().take(40).collect::<String>().trim().to_string()
}

/// **A box in 0–1000, as Qwen3-VL is trained to give it**, read as its middle.
pub fn read(reply: &str) -> Saw {
    match numbers_after(reply, "bbox_2d").or_else(|| numbers_after(reply, "point_2d")) {
        Some(n) if n.is_empty() => Saw::Nothing,
        Some(n) if n.len() >= 4 => Saw::At { x: (n[0] + n[2]) / 2.0, y: (n[1] + n[3]) / 2.0, label: label_in(reply) },
        Some(n) if n.len() == 2 => Saw::At { x: n[0], y: n[1], label: label_in(reply) },
        _ => Saw::Unreadable,
    }
}

/// Where a point in thousandths of the picture is on the screen, the picture being of `window`.
pub fn on_screen(x: f64, y: f64, window: (i64, i64, i64, i64)) -> (i64, i64) {
    let (wx, wy, ww, wh) = window;
    let (x, y) = (x.clamp(0.0, 1000.0), y.clamp(0.0, 1000.0));
    (wx + (x * ww as f64 / 1000.0) as i64, wy + (y * wh as f64 / 1000.0) as i64)
}

fn contains(r: (i64, i64, i64, i64), x: i64, y: i64) -> bool {
    x >= r.0 && x < r.0 + r.2 && y >= r.1 && y < r.1 + r.3
}

/// A listed thing big enough to be a place rather than a control.
pub fn is_region(w: &Widget) -> bool {
    w.rect.2 >= REGION.0 && w.rect.3 >= REGION.1
}

/// **A place the list cannot see into**: a region with nothing listed inside it.
pub fn blind(w: &Widget, all: &[Widget]) -> bool {
    is_region(w) && !all.iter().any(|o| o != w && {
        let (cx, cy) = o.centre();
        contains(w.rect, cx, cy) && o.rect.2 * o.rect.3 < w.rect.2 * w.rect.3
    })
}

/// What a point fell on, by the list.
#[derive(Debug, PartialEq)]
pub enum Landed<'a> {
    /// Inside a region, on nothing listed smaller than it: the list is blind here.
    Inside(&'a Widget),
    /// On a listed thing smaller than a region — a button, a row, a box. **The list's to answer**,
    /// and dropped: it was the more exact of the two wherever it had the thing.
    OnAControl(&'a Widget),
    /// On nothing listed at all, so there is no container to name and re-find at the press.
    OnNothing,
}

/// **The smallest listed thing under the point**, and whether that is a place or a control.
pub fn landed(all: &[Widget], x: i64, y: i64) -> Landed<'_> {
    let smallest = all.iter().filter(|w| contains(w.rect, x, y)).min_by_key(|w| w.rect.2 * w.rect.3);
    match smallest {
        None => Landed::OnNothing,
        Some(w) if is_region(w) => Landed::Inside(w),
        Some(w) => Landed::OnAControl(w),
    }
}

/// **The effect a point becomes**: the container by name and role, and the point as a fraction of
/// it. `text` carries what the picture showed, for the person and for the conversation's history,
/// which is told *where* and *what* and never the point.
pub fn effect(container: &Widget, x: i64, y: i64, label: &str) -> Effect {
    let (cx, cy, cw, ch) = container.rect;
    let part = |v: i64, from: i64, size: i64| (((v - from).clamp(0, size.max(1)) * 1000) / size.max(1)) as u16;
    Effect {
        kind: "click".into(), target: container.name.clone(), role: container.role.clone(),
        text: label.to_string(), because_asked: Some(true), which: None,
        inside: Some(Inside { x: part(x, cx, cw), y: part(y, cy, ch) }),
    }
}

/// **The person asked for something to be pressed** — a word rule, because whether to look at all
/// is not worth a model call. A sentence about something else never looks.
pub fn asks_to_press(text: &str) -> bool {
    const PRESS: [&str; 40] = [
        "눌러", "누르", "클릭", "선택", "골라", "고르", "열어", "닫아", "켜 줘", "켜줘", "꺼 줘", "꺼줘",
        "재생", "설치", "체크", "들어가",
        "click", "press", "tap", "select", "choose", "pick", "open", "close", "turn on", "turn off",
        "点", "按", "打开", "关闭", "选", "播放",
        "押し", "クリック", "開い", "開け", "閉じ", "選ん", "選択", "再生",
    ];
    let low = text.to_lowercase();
    PRESS.iter().any(|w| low.contains(w))
}

/// **What part of the window the picture is of**: the blind place the turn chose, else the one
/// blind place on the screen, else — with several — the whole window (`None`).
///
/// **Cut out and enlarged, because that is what found the characters**: on a real Character Map at
/// 150%, a fresh window for every request (`argmind-measure --example pointing`, 2026-11-18), ten
/// characters read **8 and 8 of 10** with the grid cut out and made 1280 wide, against **5** for the
/// window as it was and **6** for the whole window made 1280 wide.
pub fn place_to_show(effects: &[Effect], listed: &[Widget]) -> Option<Widget> {
    let places: Vec<&Widget> = listed.iter().filter(|w| blind(w, listed)).collect();
    if let Some(click) = effects.iter().find(|e| e.kind == "click") {
        if let Some(w) = places.iter().find(|w| w.name == click.target && w.role == click.role) {
            return Some((*w).clone());
        }
    }
    match places.as_slice() {
        [one] => Some((*one).clone()),
        _ => None,
    }
}

/// **Whether this turn looks at the picture.** All of: the person asked for a press; the screen
/// holds somewhere the list is blind; and the turn chose no click, or chose to click such a place
/// — the middle of a remote desktop is not the button inside it.
pub fn wants_a_look(text: &str, effects: &[Effect], listed: &[Widget]) -> bool {
    if !asks_to_press(text) {
        return false;
    }
    let places: Vec<&Widget> = listed.iter().filter(|w| blind(w, listed)).collect();
    if places.is_empty() {
        return false;
    }
    match effects.iter().find(|e| e.kind == "click") {
        None => true,
        Some(click) => places.iter().any(|w| w.name == click.target && w.role == click.role),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(name: &str, role: &str, rect: (i64, i64, i64, i64)) -> Widget {
        Widget::new(name, role, rect, "문자표")
    }

    fn charmap() -> Vec<Widget> {
        vec![
            w("문자 격자", "pane", (343, 145, 581, 262)),
            w("열기", "button", (789, 113, 46, 24)),
            w("복사(C)", "button", (757, 431, 61, 24)),
            w("복사할 문자", "edit", (443, 431, 300, 24)),
            w("글자 표시", "text", (50, 591, 180, 57)),
            w("창", "pane", (0, 0, 1280, 800)),
        ]
    }

    #[test]
    fn a_box_is_read_as_its_middle_with_its_label() {
        assert_eq!(read(r#"{"bbox_2d": [100, 200, 300, 400], "label": "지금 설치"}"#),
                   Saw::At { x: 200.0, y: 300.0, label: "지금 설치".into() });
        assert_eq!(read(r#"```json
[{"bbox_2d": [10, 20, 30, 40], "label": "★ \"star\""}]
```"#), Saw::At { x: 20.0, y: 30.0, label: "★ \"star\"".into() });
        assert_eq!(read(r#"{"bbox_2d": null}"#), Saw::Nothing);
        assert_eq!(read("I cannot see it"), Saw::Unreadable);
    }

    #[test]
    fn a_grid_is_blind_and_the_window_around_it_is_not() {
        let all = charmap();
        assert!(blind(&all[0], &all), "nothing listed inside the grid");
        assert!(!blind(&all[5], &all), "the window holds listed buttons");
        assert!(!blind(&all[1], &all), "a button is not a region");
        assert!(!blind(&all[4], &all), "nor a label at 150%, 57 px tall");
    }

    #[test]
    fn a_point_on_a_button_is_the_lists_and_one_in_the_grid_is_not() {
        let all = charmap();
        assert_eq!(landed(&all, 600, 250), Landed::Inside(&all[0]));
        assert_eq!(landed(&all, 800, 120), Landed::OnAControl(&all[1]),
                   "the list has this button by name, and was the more exact");
        assert_eq!(landed(&all, 2000, 2000), Landed::OnNothing);
        //: Inside the window but on none of its buttons: the window is a region, so the gap is kept
        //: and named by the window — the case of an icon with no name in a named toolbar.
        assert_eq!(landed(&all, 100, 700), Landed::Inside(&all[5]));
        assert_eq!(landed(&all, 100, 620), Landed::OnAControl(&all[4]));
    }

    #[test]
    fn the_effect_names_the_container_and_holds_the_point_as_a_fraction() {
        let all = charmap();
        let e = effect(&all[0], 343 + 581 / 2, 145 + 262 / 4, "★");
        assert_eq!((e.kind.as_str(), e.target.as_str(), e.role.as_str()), ("click", "문자 격자", "pane"));
        assert_eq!(e.inside, Some(Inside { x: 499, y: 248 }));
        assert_eq!(e.because_asked, Some(true));
        assert_eq!(e.text, "★");
        //: Back to a point in the same rectangle, a pixel off at most.
        let (x, y) = e.inside.unwrap().within(all[0].rect);
        assert!((x - (343 + 290)).abs() <= 1 && (y - (145 + 65)).abs() <= 1, "({x}, {y})");
    }

    #[test]
    fn a_point_in_the_picture_is_placed_on_the_window() {
        assert_eq!(on_screen(500.0, 500.0, (100, 50, 1280, 800)), (740, 450));
        assert_eq!(on_screen(-5.0, 2000.0, (0, 0, 1000, 1000)), (0, 1000));
    }

    #[test]
    fn only_a_request_to_press_looks() {
        let all = charmap();
        assert!(wants_a_look("★ 문자 눌러줘", &[], &all));
        assert!(wants_a_look("click the star", &[], &all));
        assert!(!wants_a_look("고마워", &[], &all), "talk never looks");
        assert!(!wants_a_look("인쇄해줘", &[], &all), "not a press");
        //: Nowhere blind on the screen: nothing to look for.
        let buttons = vec![w("열기", "button", (0, 0, 40, 20)), w("창", "pane", (0, 0, 40, 20))];
        assert!(!wants_a_look("열기 눌러줘", &[], &buttons));
    }

    #[test]
    fn the_picture_is_of_the_place_chosen_or_the_only_one() {
        let all = charmap();
        assert_eq!(place_to_show(&[], &all).map(|w| w.name), Some("문자 격자".to_string()));
        let mut two = all.clone();
        two.push(w("원격 화면", "pane", (2000, 0, 800, 600)));
        assert_eq!(place_to_show(&[], &two), None, "two places and nothing chosen: the whole window");
        let chose = Effect { kind: "click".into(), target: "원격 화면".into(), role: "pane".into(),
                             ..Default::default() };
        assert_eq!(place_to_show(&[chose], &two).map(|w| w.name), Some("원격 화면".to_string()));
    }

    #[test]
    fn a_turn_that_chose_a_button_does_not_look_and_one_that_chose_the_grid_does() {
        let all = charmap();
        let click = |target: &str, role: &str| Effect { kind: "click".into(), target: target.into(),
                                                       role: role.into(), ..Default::default() };
        assert!(!wants_a_look("복사 눌러줘", &[click("복사(C)", "button")], &all));
        assert!(wants_a_look("★ 눌러줘", &[click("문자 격자", "pane")], &all),
                "the middle of the grid is not the character");
    }
}
