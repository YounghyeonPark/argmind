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
    //: **The three below were not here until 2026-11-19**, and nothing failed: `Hands` gives each a
    //: default, so this wrapper answered *cannot tell* for the real hands on both surfaces — the
    //: check before every `type` and the one after every action, shipped 2026-10-21 and measured on
    //: the hands themselves, never ran in the application, and a point inside a place could not be
    //: pressed at all. Found by the first run of the application's own look
    //: (`examples/looking_live.rs`); `argmind-rules` (`the_surface_hides_no_hand`) now reads this
    //: impl for every method the trait has.
    fn click_inside(&mut self, within: &Widget, at: (i64, i64)) -> bool {
        match self.web_now() {
            Some(web) => web.click_inside(within, at),
            None => self.desk.click_inside(within, at),
        }
    }
    fn typing_lands(&mut self) -> Option<bool> {
        match self.web_now() {
            Some(web) => web.typing_lands(),
            None => self.desk.typing_lands(),
        }
    }
    fn landed(&self) -> Option<bool> {
        match (self.on_web.get(), self.web.as_ref()) {
            (true, Some(web)) => web.landed(),
            _ => self.desk.landed(),
        }
    }
}

/// Build both halves from what the window handed over, or desktop-only when there is no pane.
pub fn surfaces(desk_eyes: Sight, desk_hands: Touch, pane: Option<&WebPane>, live: bool,
                on_web: OnWeb) -> (Looking, Touching, OnWeb) {
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
    argmind_env_browser::search_url(t)
}

/// **Whether a turn moves argmind onto the web or off it**, read by the program from what was said.
///
/// The owner, 2026-10-21: *argmind가 상황에 맞게 웹창을 열고 닫는걸 스스로 하게 해줘* — nobody presses
/// the 웹 switch by hand. **Not a model's decision**: this model has been measured choosing silence 0
/// times in 224 and sorting errands by the position of an option, and *which surface* is a choice a
/// word list can make about as well and always the same way. Each rule is one a person can predict.
#[derive(Debug, PartialEq)]
pub enum WebMove {
    /// Open the pane at this address. `only` when nothing else was asked, so the turn ends there.
    Open { url: String, only: bool },
    /// Open the pane on a search for these words.
    Search(String),
    /// Back to the desktop, with nothing else asked.
    Close,
    /// Back to the desktop, and the turn carries on there.
    ToDesktop,
}

/// Sites by the names people say. **Not 다음**, which is mostly *next*.
const SITES: [(&str, &str); 12] = [
    ("네이버 메일", "https://mail.naver.com/"), ("네이버", "https://www.naver.com/"),
    ("naver", "https://www.naver.com/"), ("유튜브", "https://www.youtube.com/"),
    ("youtube", "https://www.youtube.com/"), ("위키백과", "https://ko.wikipedia.org/"),
    ("나무위키", "https://namu.wiki/"), ("깃허브", "https://github.com/"),
    ("github", "https://github.com/"), ("쿠팡", "https://www.coupang.com/"),
    ("지메일", "https://mail.google.com/"), ("구글", "https://www.google.com/"),
];

/// What makes an opening the whole request: the verbs and particles around a site's name.
const OPEN_WORDS: [&str; 16] = ["열어줘", "열어 줘", "열어", "띄워줘", "띄워 줘", "켜줘", "켜 줘",
                                "가줘", "가 줘", "들어가줘", "들어가 줘", "접속해줘", "좀", "에", "로", "으로"];

/// Things on the desktop people name, which mean the turn is about the desktop.
const DESKTOP: [&str; 16] = ["메모장", "탐색기", "파일 탐색기", "바탕화면", "데스크톱", "vs code",
                             "vscode", "비주얼 스튜디오", "엑셀", "워드", "파워포인트", "카카오톡",
                             "계산기", "그림판", "작업 표시줄", "설정 창"];

const CLOSE: [&str; 8] = ["웹 닫", "웹창 닫", "웹 꺼", "브라우저 닫", "브라우저 꺼", "칸 닫", "웹 숨겨",
                          "브라우저 숨겨"];

const SEARCH: [&str; 7] = ["검색해줘", "검색해 줘", "검색해봐", "검색해 봐", "웹에서 찾아", "인터넷에서 찾아",
                           "찾아봐줘"];

/// **The same moves in English, Chinese and Japanese** — since 2026-11-17, the owner's *대화언어도
/// 다국어 지원이 되어야함*. Each list is read only in a sentence written in its language
/// (`argmind_core::tongue::reads_as`), so every Korean sentence moves the pane exactly as before.
/// English is read by whole words: *word* is not in *password*, *paint* is not in *painting*.
struct Tongue {
    code: &'static str,
    close: &'static [&'static str],
    desktop: &'static [&'static str],
    /// The verb, and whether the words searched for come after it (English, Chinese) or before it.
    search: &'static [(&'static str, bool)],
    /// A box on a page, which is not a request to search.
    search_box: &'static [&'static str],
    sites: &'static [(&'static str, &'static str)],
    open: &'static [&'static str],
}

const TONGUES: [Tongue; 3] = [
    Tongue {
        code: "en",
        close: &["close the web", "close web", "close the browser", "close browser", "hide the web",
                 "hide the browser", "close the pane", "close the web pane", "turn off the web"],
        desktop: &["notepad", "file explorer", "explorer", "desktop", "vs code", "vscode", "visual studio", "excel",
                   "word", "powerpoint", "kakaotalk", "calculator", "paint", "taskbar", "settings window"],
        search: &[("search the web for ", true), ("search for ", true), ("search ", true), ("look up ", true)],
        search_box: &["search box", "search bar", "search field"],
        sites: &[("naver mail", "https://mail.naver.com/"), ("naver", "https://www.naver.com/"),
                 ("youtube", "https://www.youtube.com/"), ("wikipedia", "https://en.wikipedia.org/"),
                 ("github", "https://github.com/"), ("coupang", "https://www.coupang.com/"),
                 ("gmail", "https://mail.google.com/"), ("google", "https://www.google.com/")],
        open: &["open", "go", "to", "launch", "show", "visit", "please", "the", "site", "website", "page", "up", "pull",
                "bring", "me", "can", "you", "could", "on", "at", "in", "a", "into"],
    },
    Tongue {
        code: "zh",
        close: &["关闭网页", "关掉网页", "关网页", "关闭浏览器", "关掉浏览器", "隐藏网页", "关闭网页窗格"],
        desktop: &["记事本", "资源管理器", "文件资源管理器", "桌面", "计算器", "画图", "任务栏", "设置窗口"],
        search: &[("在网上搜索", true), ("网上搜", true), ("在网上找", true), ("搜索一下", true), ("搜一下", true),
                  ("搜搜", true), ("搜索", true)],
        search_box: &["搜索框", "搜索栏"],
        sites: &[("youtube", "https://www.youtube.com/"), ("油管", "https://www.youtube.com/"),
                 ("维基百科", "https://zh.wikipedia.org/"), ("github", "https://github.com/"),
                 ("naver", "https://www.naver.com/"), ("coupang", "https://www.coupang.com/"),
                 ("谷歌", "https://www.google.com/"), ("google", "https://www.google.com/")],
        open: &["帮我", "请", "打开", "进入", "访问", "去", "一下", "网站", "网页", "上", "吧"],
    },
    Tongue {
        code: "ja",
        close: &["ウェブを閉じ", "ウェブ閉じ", "ブラウザを閉じ", "ブラウザ閉じ", "ウェブを隠", "Webを閉じ", "webを閉じ"],
        desktop: &["メモ帳", "エクスプローラー", "デスクトップ", "電卓", "ペイント", "タスクバー", "設定画面", "エクセル",
                   "ワード", "パワーポイント"],
        search: &[("をネットで検索して", false), ("をウェブで検索して", false), ("を検索して", false),
                  ("で検索して", false), ("を検索", false), ("をネットで探して", false), ("をウェブで探して", false)],
        search_box: &["検索ボックス", "検索欄", "検索窓", "検索バー"],
        sites: &[("youtube", "https://www.youtube.com/"), ("ユーチューブ", "https://www.youtube.com/"),
                 ("ウィキペディア", "https://ja.wikipedia.org/"), ("github", "https://github.com/"),
                 ("ギットハブ", "https://github.com/"), ("naver", "https://www.naver.com/"),
                 ("ネイバー", "https://www.naver.com/"), ("coupang", "https://www.coupang.com/"),
                 ("グーグル", "https://www.google.com/"), ("google", "https://www.google.com/")],
        open: &["を開いて", "開いて", "を開けて", "に入って", "に行って", "へ行って", "にアクセスして", "アクセスして",
                "を表示して", "見せて", "サイト", "ください", "を", "に", "へ"],
    },
];

/// Is `word` in the sentence — whole, for English.
fn names(t: &Tongue, low: &str, word: &str) -> bool {
    if t.code == "en" { argmind_core::tongue::has_word(low, word) } else { low.contains(word) }
}

fn web_move_in(t: &Tongue, said: &str, on_web: bool) -> Option<WebMove> {
    let low = argmind_core::tongue::lower(said.trim());
    //: Positions found in the lower-cased sentence cut the sentence itself — unless lower-casing
    //: moved a byte, when the lower-cased one is cut instead.
    let src = if low.len() == said.trim().len() { said.trim() } else { low.as_str() };
    if t.close.iter().any(|w| names(t, &low, w)) {
        return Some(WebMove::Close);
    }
    if on_web && t.desktop.iter().any(|w| names(t, &low, w)) {
        return Some(WebMove::ToDesktop);
    }
    if !t.search_box.iter().any(|w| low.contains(w)) {
        for (verb, after) in t.search {
            let Some(at) = low.find(verb) else { continue };
            if t.code == "en" && at > 0 && low.as_bytes()[at - 1].is_ascii_alphanumeric() {
                continue;
            }
            let words = if *after { &src[at + verb.len()..] } else { &src[..at] };
            let mut words = words.trim().trim_end_matches(['.', '!', '。', '！', ' ']).trim();
            //: The site the search is on, and the softeners around it, are not what is searched.
            for tail in [" on naver", " on the web", " online", " on the internet", " please", "一下"] {
                let lw = words.to_lowercase();
                if lw.len() == words.len() && lw.ends_with(tail) {
                    words = words[..words.len() - tail.len()].trim();
                }
            }
            let words = words.strip_prefix("一下").unwrap_or(words).trim();
            if !words.is_empty() {
                return Some(WebMove::Search(words.to_string()));
            }
        }
    }
    let address = said.split_whitespace().map(|w| w.trim_matches(|c: char| !c.is_ascii_alphanumeric()
                                                                     && c != '.' && c != '/' && c != ':'))
        .find(|w| {
            let w = w.to_lowercase();
            [".com", ".net", ".org", ".kr", ".io", ".dev", ".wiki", ".co"].iter()
                .any(|end| w.contains(end)) && !w.starts_with('.')
        });
    let (url, name) = match address {
        Some(a) => (if a.starts_with("http") { a.to_string() } else { format!("https://{a}") }, a.to_lowercase()),
        None => match t.sites.iter().find(|(name, _)| names(t, &low, name)) {
            Some((name, url)) => (url.to_string(), name.to_string()),
            None => return None,
        },
    };
    let rest = low.replace(&name, " ");
    let only = if t.code == "en" {
        rest.split(|c: char| !(c.is_ascii_alphanumeric() || c == '\''))
            .filter(|w| !w.is_empty() && !t.open.contains(w)).all(|w| w.chars().count() <= 1)
    } else {
        let mut rest = rest;
        let mut words = t.open.to_vec();
        words.sort_by_key(|w| std::cmp::Reverse(w.chars().count()));
        for w in words {
            rest = rest.replace(w, " ");
        }
        rest.split(|c: char| c.is_whitespace() || "，。、！!,.".contains(c)).all(|w| w.chars().count() <= 1)
    };
    Some(WebMove::Open { url, only })
}

pub fn web_move(said: &str, on_web: bool) -> Option<WebMove> {
    if let Some(t) = TONGUES.iter().find(|t| argmind_core::converse::written_in(said) == Some(t.code)) {
        return web_move_in(t, said, on_web);
    }
    let t = said.trim();
    let low = t.to_lowercase();
    if CLOSE.iter().any(|w| t.contains(w)) {
        return Some(WebMove::Close);
    }
    //: **A desktop app named while the pane is up** — the turn is about the desktop.
    if on_web && DESKTOP.iter().any(|w| low.contains(w)) {
        return Some(WebMove::ToDesktop);
    }
    //: **Searching, said in so many words.** *검색창* is a box on a page, not a request to search.
    if let Some(verb) = SEARCH.iter().find(|w| t.contains(*w)) {
        let words = t.split(verb).next().unwrap_or("").trim()
            .trim_end_matches(|c: char| c == '을' || c == '를' || c == ' ').trim();
        if !words.is_empty() && !t.contains("검색창") {
            return Some(WebMove::Search(words.to_string()));
        }
    }
    //: **An address**, typed as one: a token with a dot and a known ending.
    let address = t.split_whitespace().map(|w| w.trim_matches(|c: char| !c.is_ascii_alphanumeric()
                                                                && c != '.' && c != '/' && c != ':'))
        .find(|w| {
            let w = w.to_lowercase();
            [".com", ".net", ".org", ".kr", ".io", ".dev", ".wiki", ".co"].iter()
                .any(|end| w.contains(end)) && !w.starts_with('.')
        });
    let (url, name) = match address {
        Some(a) => (if a.starts_with("http") { a.to_string() } else { format!("https://{a}") },
                    a.to_string()),
        None => match SITES.iter().find(|(name, _)| low.contains(name)) {
            Some((name, url)) => (url.to_string(), name.to_string()),
            None => return None,
        },
    };
    //: **Only the opening was asked** when nothing is left once the site and the verbs are gone.
    let mut rest = low.replace(&name.to_lowercase(), " ");
    //: **Longest first**: *가줘* taken out of *들어가줘* leaves *들어*, which reads as a request.
    let mut words = OPEN_WORDS.to_vec();
    words.sort_by_key(|w| std::cmp::Reverse(w.chars().count()));
    for w in words {
        rest = rest.replace(w, " ");
    }
    let only = rest.split_whitespace().all(|w| w.chars().count() <= 1);
    Some(WebMove::Open { url, only })
}

#[cfg(test)]
mod tests {
    use super::{address, web_move, OnWeb, WebMove};

    #[test]
    fn sites_open_and_a_request_after_them_carries_on() {
        assert_eq!(web_move("네이버 열어줘", false),
                   Some(WebMove::Open { url: "https://www.naver.com/".into(), only: true }));
        assert_eq!(web_move("github.com 들어가줘", false),
                   Some(WebMove::Open { url: "https://github.com".into(), only: true }));
        assert_eq!(web_move("네이버에서 메일 열어줘", false),
                   Some(WebMove::Open { url: "https://www.naver.com/".into(), only: false }));
        assert_eq!(web_move("네이버 메일 열어줘", false),
                   Some(WebMove::Open { url: "https://mail.naver.com/".into(), only: true }));
    }

    #[test]
    fn searching_is_said_and_a_search_box_is_not_a_search() {
        assert_eq!(web_move("러스트 비동기 검색해줘", false),
                   Some(WebMove::Search("러스트 비동기".into())));
        assert_eq!(web_move("서울 맛집을 웹에서 찾아줘", false),
                   Some(WebMove::Search("서울 맛집".into())));
        assert_eq!(web_move("검색창 눌러줘", true), None);
    }

    #[test]
    fn closing_and_the_desktop_take_it_back() {
        assert_eq!(web_move("웹 닫아줘", true), Some(WebMove::Close));
        assert_eq!(web_move("메모장 열어줘", true), Some(WebMove::ToDesktop));
        assert_eq!(web_move("메모장 열어줘", false), None, "already on the desktop");
    }

    #[test]
    fn ordinary_talk_moves_nothing() {
        for t in ["안녕", "고마워", "다음 주 회의 정리해줘", "메일 열어줘", "이거 눌러줘"] {
            assert_eq!(web_move(t, false), None, "{t}");
        }
    }

    #[test]
    fn the_pane_moves_in_english_chinese_and_japanese() {
        let open = |url: &str, only: bool| Some(WebMove::Open { url: url.into(), only });
        assert_eq!(web_move("open YouTube", false), open("https://www.youtube.com/", true));
        assert_eq!(web_move("go to GitHub", false), open("https://github.com/", true));
        assert_eq!(web_move("open Coupang and find wet wipes", false), open("https://www.coupang.com/", false));
        assert_eq!(web_move("打开YouTube", false), open("https://www.youtube.com/", true));
        assert_eq!(web_move("进入GitHub", false), open("https://github.com/", true));
        assert_eq!(web_move("YouTubeを開いて", false), open("https://www.youtube.com/", true));
        assert_eq!(web_move("ウィキペディアを開いて", false), open("https://ja.wikipedia.org/", true));
        assert_eq!(web_move("search for Son Heung-min on Naver", false), Some(WebMove::Search("Son Heung-min".into())));
        assert_eq!(web_move("在Naver上搜索孙兴慜", false), Some(WebMove::Search("孙兴慜".into())));
        assert_eq!(web_move("ラーメンを検索して", false), Some(WebMove::Search("ラーメン".into())));
        for close in ["close the web", "Close the browser please", "关闭网页", "ウェブを閉じて"] {
            assert_eq!(web_move(close, true), Some(WebMove::Close), "{close}");
        }
        assert_eq!(web_move("open Notepad", true), Some(WebMove::ToDesktop));
        assert_eq!(web_move("打开记事本", true), Some(WebMove::ToDesktop));
        assert_eq!(web_move("メモ帳を開いて", true), Some(WebMove::ToDesktop));
        //: A box on a page is not a search; a password is not Word; research is not a search.
        for none in ["click the search box", "type my password", "research three competitors", "thanks",
                     "点一下搜索框", "検索ボックスを押して"] {
            assert_eq!(web_move(none, true), None, "{none}");
        }
    }

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
