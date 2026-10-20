//! **Where argmind is looking and pressing: the desktop, or its own browser pane.**
//!
//! One session, two surfaces, one switch. Not two sessions, because **the whole safety argument is
//! that one place performs** — `argmind-rules/tests/one_place_acts.rs` counts the calls — and a
//! second session would be a second place. It is also one conversation: what was said about a web
//! page and what was said about the desktop are the same person talking to the same agent.
//!
//! **A proposal remembers which surface it was made on**, and `act` presses there, whatever the
//! switch says by the time the button is pressed: an offer to click *메일* on Naver, confirmed after
//! the person flipped back to the desktop, must not go looking for a desktop widget called *메일*.
//!
//! **Watching is always the desktop.** `notice` is about what the person is doing; the pane is what
//! argmind is doing, and a watch on it would be argmind noticing itself — the defect the watch had
//! once already, when it read argmind's own transcript.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use argmind_core::contract::{Did, Eyes, Hands, Widget};
use argmind_env_browser::{Pane, WebEyes, WebHands, Wire};

use crate::platform::{Sight, Touch};

/// What the window hands the server: how to talk to the pane, and how to show or hide it.
#[derive(Clone)]
pub struct WebPane {
    pub wire: Arc<dyn Wire>,
    pub show: Arc<dyn Fn(bool) + Send + Sync>,
}

/// The switch, shared by the eyes, the hands and whoever flips it.
#[derive(Clone, Default)]
pub struct OnWeb(Arc<AtomicBool>);

impl OnWeb {
    pub fn get(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
    pub fn set(&self, on: bool) {
        self.0.store(on, Ordering::SeqCst)
    }

    /// Switch to `on` until the returned guard is dropped, then back to what it was.
    pub fn for_now(&self, on: bool) -> ForNow {
        let was = self.get();
        self.set(on);
        ForNow { switch: self.clone(), was }
    }
}

/// Puts the switch back when it goes out of scope, however the scope is left.
pub struct ForNow {
    switch: OnWeb,
    was: bool,
}

impl Drop for ForNow {
    fn drop(&mut self) {
        self.switch.set(self.was);
    }
}

pub struct Looking {
    /// **The desktop's eyes, by name**, for the places that must read the desktop whatever the
    /// switch says — watching, and the window a held remark was about.
    pub desk: Sight,
    pub web: Option<WebEyes>,
    pub on_web: OnWeb,
}

impl Looking {
    fn web_now(&mut self) -> Option<&mut WebEyes> {
        if self.on_web.get() { self.web.as_mut() } else { None }
    }

    /// The window in front, as a handle and a title. **On the pane, the page's title and host**,
    /// with handle 1 so *no window* (0) still means what it meant.
    pub fn window(&self) -> (isize, String) {
        match (self.on_web.get(), self.web.as_ref()) {
            (true, Some(web)) => {
                let title = web.window();
                (1, if title.is_empty() { "웹".into() } else { title })
            }
            _ => self.desk.window(),
        }
    }

    /// The hit test's second opinion. **None on the pane**: the pane's hands ask the page the same
    /// question as a gate, before every click, so a second opinion would be the same answer twice.
    pub fn at_point(&mut self, x: i64, y: i64) -> Option<(String, String)> {
        if self.on_web.get() { None } else { self.desk.inner.at_point(x, y) }
    }
}

impl Eyes for Looking {
    fn foreground(&mut self) -> Vec<Widget> {
        match self.web_now() {
            Some(web) => web.foreground(),
            None => self.desk.foreground(),
        }
    }
    fn timed_foreground(&mut self) -> (Vec<Widget>, f64) {
        match self.web_now() {
            Some(web) => web.timed_foreground(),
            None => self.desk.timed_foreground(),
        }
    }
}

pub struct Touching {
    pub desk: Touch,
    pub web: Option<WebHands>,
    pub on_web: OnWeb,
}

impl Touching {
    fn web_now(&mut self) -> Option<&mut WebHands> {
        if self.on_web.get() { self.web.as_mut() } else { None }
    }

    /// What the last action did, on whichever surface it happened.
    pub fn last_done(&self) -> Option<&Did> {
        match (self.on_web.get(), self.web.as_ref()) {
            (true, Some(web)) => web.done.last(),
            _ => self.desk.done.last(),
        }
    }
}

impl Hands for Touching {
    fn live(&self) -> bool {
        match (self.on_web.get(), self.web.as_ref()) {
            (true, Some(web)) => web.live(),
            _ => self.desk.live(),
        }
    }
    fn click(&mut self, widget: &Widget) {
        match self.web_now() {
            Some(web) => web.click(widget),
            None => self.desk.click(widget),
        }
    }
    fn type_text(&mut self, text: &str) {
        match self.web_now() {
            Some(web) => web.type_text(text),
            None => self.desk.type_text(text),
        }
    }
    fn press(&mut self, key: &str) {
        match self.web_now() {
            Some(web) => web.press(key),
            None => self.desk.press(key),
        }
    }
    fn keys(&self) -> Option<Vec<String>> {
        match (self.on_web.get(), self.web.as_ref()) {
            (true, Some(web)) => web.keys(),
            _ => self.desk.keys(),
        }
    }
    fn into_a_secret(&mut self) -> bool {
        match self.web_now() {
            Some(web) => web.into_a_secret(),
            None => self.desk.into_a_secret(),
        }
    }
}

/// Build both halves from what the window handed over, or desktop-only when there is no pane.
pub fn surfaces(desk_eyes: Sight, desk_hands: Touch, pane: Option<&WebPane>, live: bool)
                -> (Looking, Touching, OnWeb) {
    let on_web = OnWeb::default();
    let shared = pane.map(|p| Pane::new(p.wire.clone()));
    let looking = Looking { desk: desk_eyes, web: shared.as_ref().map(|p| p.eyes()),
                            on_web: on_web.clone() };
    let touching = Touching { desk: desk_hands, web: shared.as_ref().map(|p| p.hands(live)),
                              on_web: on_web.clone() };
    (looking, touching, on_web)
}

/// **What the address box means.** Something that looks like an address is one; anything else is a
/// search — on Naver, because Google refused the pane with its automated-traffic page in
/// `--bin browsing`. The window's own navigation check still decides where the pane may go.
pub fn address(typed: &str) -> String {
    let t = typed.trim();
    if t.starts_with("http://") || t.starts_with("https://") {
        return t.to_string();
    }
    if !t.contains(' ') && t.contains('.') && !t.ends_with('.') {
        return format!("https://{t}");
    }
    let mut q = String::new();
    for b in t.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => q.push(b as char),
            b' ' => q.push('+'),
            _ => q.push_str(&format!("%{b:02X}")),
        }
    }
    format!("https://search.naver.com/search.naver?query={q}")
}

#[cfg(test)]
mod tests {
    use super::{address, OnWeb};

    #[test]
    fn the_switch_comes_back_however_the_scope_is_left() {
        let s = OnWeb::default();
        {
            let _g = s.for_now(true);
            assert!(s.get());
        }
        assert!(!s.get());
        s.set(true);
        let early = |s: &OnWeb| -> Option<()> { let _g = s.for_now(false); None?; Some(()) };
        early(&s);
        assert!(s.get(), "an early return still restores");
    }

    #[test]
    fn an_address_is_opened_and_anything_else_is_searched() {
        assert_eq!(address("https://example.com/a"), "https://example.com/a");
        assert_eq!(address("naver.com"), "https://naver.com");
        assert_eq!(address("서울 날씨"),
                   "https://search.naver.com/search.naver?query=%EC%84%9C%EC%9A%B8+%EB%82%A0%EC%94%A8");
        assert_eq!(address("rust lang"), "https://search.naver.com/search.naver?query=rust+lang");
        assert_eq!(address("끝."), "https://search.naver.com/search.naver?query=%EB%81%9D.");
    }
}
