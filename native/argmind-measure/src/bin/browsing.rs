//! **What argmind's own browser pane can read, and how much of it there is.**
//!
//! ```text
//!   cargo run -p argmind-measure --bin browsing --release
//! ```
//!
//! From `native/`. **It opens a window**, unfocused, holding only the pane, and visits a handful of
//! public pages. **It presses nothing and types nothing** — every command it sends reads: navigate,
//! wait, read the accessibility tree, read the layout. It needs the network and nothing else: no
//! model, no camera, no argmind running.
//!
//! ## Why
//!
//! The owner chose a browser pane inside argmind's window (2026-10-20) so that argmind's web work
//! never meets the person's pointer, keyboard or windows. Whether that pane is worth driving
//! depends on one number this tree already has a bad reading of: **`crowding` took choosing from
//! 22 of 28 to 3–8 of 28 once the list reached 123 widgets**. A web page as a whole is far past
//! that. The question here is what is left after the deterministic side narrows it — interactive
//! roles, a name, on screen — and that is a count, not a model call.
//!
//! ## Written before the run
//!
//! * **The control first: `example.com` has one link and nothing else to press.** The reading must
//!   say exactly 1 on screen. Anything else and the instrument is wrong, and the run says so before
//!   any other number is read.
//! * **Six real pages**, each with three counts: every node in the accessibility tree; interactive
//!   and named on the whole page; interactive, named and on screen.
//! * **Worth the next step if at least 4 of the 6 are at or under 100 on screen** — under the 123
//!   where choosing fell apart, with room. And on each real page the three counts must fall in
//!   order with the last above zero; a tidy row of zeros is a broken read, not an empty web.
//!
//! ## `--choosing`: does the model pick the right thing from these lists?
//!
//! The counts say the lists are 65–155 long; `crowding` says choosing collapses from 123 and has
//! no reading between its captured screens and that. So: the same pages, the same narrowed list
//! handed to the model through the shipped prompt, and requests whose answer is on the page.
//!
//! **Written before the run.** Four requests a page with one right item, named by role and name, and
//! one request a page that nothing on it can do. **A live page changes**, so a request counts only
//! when its item is in that run's list **exactly once** — a name that is there twice or not at all
//! makes the request unanswerable rather than wrong, and it is printed as skipped.
//!
//! * **Usable as it stands if answerable requests land on the right item in at least 60%** — under
//!   the captured screens' 21–24 of 28, well over `crowding`'s 3–8 of 28 from 123 up;
//! * **and at least 4 of 5 impossible requests are declined** — no effect at all.
//!
//! Below either, the list needs narrowing before a model sees it, which is deterministic and a
//! separate change. Where in the list the item sat is printed beside each answer, because
//! `crowding` could not separate position from length.
//!
//! **What it read, 2026-10-20, one run: 15 of 20 answerable (75%), 5 of 5 declined, none skipped**
//! — the bar met, and read as a flattering result first:
//!
//! * **The requests are easier than a person's.** They were written by looking at the lists, so
//!   most name the item (*뉴스스탠드 탭*, *Code 버튼*). The impossible ones were obvious — alarms,
//!   video — and 5 of 5 on those says little.
//! * **The five misses have three shapes.** A Korean request for an English label, **three times**:
//!   *글 올리는 곳* → `submit`, *구인 글* → `jobs`, *태그 목록* → `165 Tags` (where *브랜치 목록* →
//!   `10 Branches` landed). **Off by one**: *경복궁* is row 30 and it pressed `종묘` at 31, saying
//!   *31*. **Deep in the list**: Naver's search box is row 66 of 68 and it said it could not find one.
//! * **And two things about talking, which the owner's priority puts first.** The index leaks into
//!   the speech in **5 of 25** replies (*10번에 있습니다*, *옵션은 4번입니다*), at least the shipped 5 of
//!   32; and **English pages pull the reply into English** — Korean in, English out on Hacker News
//!   and GitHub, 5 of 25 — the labels in the list doing what an English slot did in `the dials`.
//!
//! ## `--hands`: does pressing in the pane stay in the pane?
//!
//! `argmind-env-browser` presses with page events, never the operating system. Four checks,
//! written before the run, each able to fail:
//!
//! 1. **A click lands and the pointer does not move.** *Learn more* on `example.com` takes the page
//!    to `iana.org`, and the person's pointer reads the same before and after.
//! 2. **Typing lands in the page.** The Wikipedia search box clicked, *경복궁* typed, and the box's
//!    own value read back. Enter is not pressed.
//! 3. **A covered target is refused** — the negative control. A layer is laid over the link in the
//!    pane's own copy of the page; the click must come back `Elsewhere` and the page must not move.
//! 4. **A password field reads as one and an ordinary one does not**, both added to the pane's page
//!    and focused there.
//!
//! ## What the first runs read, 2026-10-20
//!
//! * **The control failed first, and was right to.** The one link on `example.com` read as off
//!   screen: the snapshot's boxes are in device pixels and the viewport compared against them was
//!   in CSS pixels — at a display scale of 1.5 the link sat at (900, 1146) against 1280×900, and the
//!   page itself put it at (600, 764), exactly two thirds. Compared in one unit, the control reads 1.
//! * **Then: Naver 68 on screen of 203 interactive, Wikipedia 100 of 2,678, the weather service 65
//!   of 132, GitHub 90 of 217, Hacker News 154 of 198** — a page of links and almost nothing else.
//! * **Google is not a reading.** 30 nodes and one link, *Why did this happen?*: its automated-
//!   traffic page, which refused the pane. The criterion written above counted it as *under 100*
//!   and the sanity check passed it — **both checks were too easy to pass**, and a flattering row
//!   it should have refused. Without it, **4 of 5 real pages are under 100**, which meets the bar
//!   of 4 with nothing to spare. Wikipedia landing exactly on the line is one sample, not a cap —
//!   nothing in this file clips at 100.
//! * **These counts sit where nothing has been measured**: `crowding` had choosing at 22 of 28 on
//!   the captured screens and 3–8 from 123 up. 65–100 is between, so the next number is choosing on
//!   these lists, not this one. The main frame only — an advert in an iframe is not counted — and
//!   Naver's list already holds *[광고]* lines, which are text a page wrote for a model to read.

#[cfg(not(windows))]
fn main() {
    println!("  the pane is driven through WebView2, which is Windows only");
}

#[cfg(windows)]
fn main() {
    use argmind_window::{run, Pane, Say, Shell};
    use std::sync::mpsc::{channel, Sender};

    let (orders, inbox) = channel::<Say>();
    std::thread::spawn(move || {
        let outcome = measure(&orders);
        if let Err(why) = outcome {
            println!("\n  ** {why}");
        }
        use std::io::Write;
        let _ = std::io::stdout().flush();
        let _ = orders.send(Say::Quit);
    });

    let data = std::env::temp_dir().join("argmind-browsing");
    let shell = Shell {
        url: "http://127.0.0.1/".into(),
        title: "argmind — browsing (measurement)".into(),
        wide: 520,
        tall: 900,
        //: **Unfocused and without an icon.** It appears on the screen; it must not take the
        //: keyboard from whatever the person was doing.
        focused: false,
        tray: false,
        pane: Some(Pane { data, url: "about:blank".into(), wide: 1280, alone: true }),
        ..Shell::default()
    };
    if let Err(why) = run(shell, Some(inbox)) {
        println!("  ** {why}");
    }

    fn ask(orders: &Sender<Say>, method: &str, params: serde_json::Value)
           -> Result<serde_json::Value, String> {
        let (reply, answer) = channel();
        orders.send(Say::Devtools { method: method.into(), params: params.to_string(), reply })
            .map_err(|_| "the window has gone".to_string())?;
        let text = answer.recv_timeout(std::time::Duration::from_secs(30))
            .map_err(|_| format!("{method}: no answer in 30 s"))??;
        serde_json::from_str(&text).map_err(|e| format!("{method}: {e}"))
    }

    /// Interactive roles as Chromium's accessibility tree names them.
    const PRESSABLE: [&str; 16] = ["button", "link", "textbox", "searchbox", "combobox", "checkbox",
                                   "radio", "menuitem", "menuitemcheckbox", "menuitemradio", "tab",
                                   "switch", "slider", "spinbutton", "option", "treeitem"];

    struct Read {
        nodes: usize,
        named: usize,
        shown: Vec<(String, String)>,
        /// The same list as widgets, full names and boxes, for the model.
        widgets: Vec<argmind_core::contract::Widget>,
    }

    fn read(orders: &Sender<Say>, url: &str) -> Result<Read, String> {
        use serde_json::json;
        ask(orders, "Page.navigate", json!({ "url": url }))?;
        //: **Loaded, then two seconds for what loads after loading.**
        let began = std::time::Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_millis(300));
            let r = ask(orders, "Runtime.evaluate",
                        json!({ "expression": "document.readyState", "returnByValue": true }))?;
            if r["result"]["value"] == "complete" {
                break;
            }
            if began.elapsed().as_secs() > 25 {
                return Err(format!("{url}: not loaded in 25 s"));
            }
        }
        std::thread::sleep(std::time::Duration::from_secs(2));

        let tree = ask(orders, "Accessibility.getFullAXTree", json!({}))?;
        let snap = ask(orders, "DOMSnapshot.captureSnapshot", json!({ "computedStyles": [] }))?;
        let metrics = ask(orders, "Page.getLayoutMetrics", json!({}))?;

        //: Layout bounds by backend node, from the main document of the snapshot.
        let doc = &snap["documents"][0];
        let ids = doc["nodes"]["backendNodeId"].as_array().cloned().unwrap_or_default();
        let index = doc["layout"]["nodeIndex"].as_array().cloned().unwrap_or_default();
        let bounds = doc["layout"]["bounds"].as_array().cloned().unwrap_or_default();
        let mut boxes = std::collections::HashMap::new();
        for (i, b) in index.iter().zip(bounds.iter()) {
            let Some(i) = i.as_u64() else { continue };
            let Some(id) = ids.get(i as usize).and_then(|v| v.as_i64()) else { continue };
            let n: Vec<f64> = b.as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
                .unwrap_or_default();
            if n.len() == 4 {
                boxes.insert(id, (n[0], n[1], n[2], n[3]));
            }
        }
        if std::env::var("BROWSING_DEBUG").is_ok() {
            let r = ask(orders, "Runtime.evaluate", json!({ "returnByValue": true,
                "expression": "JSON.stringify([devicePixelRatio, innerWidth, innerHeight, (document.querySelector('a')||document.body).getBoundingClientRect()])" }))?;
            println!("      [debug] page says {}", r["result"]["value"]);
            println!("      [debug] metrics {}", metrics);
        }
        //: **Device pixels on both sides.** The snapshot's boxes are in device pixels and
        //: `cssLayoutViewport` is not: at a display scale of 1.5 the control's one link sat at
        //: (900, 1146) against a 1280×900 viewport and read as off screen — the page itself put it
        //: at (600, 764) CSS, exactly two thirds. `layoutViewport` is the same viewport in the
        //: snapshot's units. The control caught it on the first run.
        let v = &metrics["layoutViewport"];
        let (vx, vy) = (v["pageX"].as_f64().unwrap_or(0.0), v["pageY"].as_f64().unwrap_or(0.0));
        let (vw, vh) = (v["clientWidth"].as_f64().unwrap_or(0.0),
                        v["clientHeight"].as_f64().unwrap_or(0.0));

        let nodes = tree["nodes"].as_array().cloned().unwrap_or_default();
        let mut named = 0;
        let mut shown = Vec::new();
        let mut widgets = Vec::new();
        let title = ask(orders, "Runtime.evaluate",
                        json!({ "expression": "document.title", "returnByValue": true }))?
            ["result"]["value"].as_str().unwrap_or("").to_string();
        for n in &nodes {
            if n["ignored"].as_bool().unwrap_or(false) {
                continue;
            }
            let role = n["role"]["value"].as_str().unwrap_or("");
            let name = n["name"]["value"].as_str().unwrap_or("").trim();
            if !PRESSABLE.contains(&role) || name.is_empty() {
                continue;
            }
            named += 1;
            if std::env::var("BROWSING_DEBUG").is_ok() {
                println!("      [debug] {role} {name:?} id={} box={:?} viewport=({vx},{vy},{vw},{vh}) boxes={}",
                         n["backendDOMNodeId"], n["backendDOMNodeId"].as_i64().and_then(|i| boxes.get(&i)),
                         boxes.len());
            }
            let Some(id) = n["backendDOMNodeId"].as_i64() else { continue };
            let Some(&(x, y, w, h)) = boxes.get(&id) else { continue };
            if w > 0.0 && h > 0.0 && x < vx + vw && x + w > vx && y < vy + vh && y + h > vy {
                shown.push((role.to_string(), name.chars().take(40).collect()));
                widgets.push(argmind_core::contract::Widget::new(
                    name, role, (x as i64, y as i64, w as i64, h as i64), &title));
            }
        }
        Ok(Read { nodes: nodes.len(), named, shown, widgets })
    }

    fn measure(orders: &Sender<Say>) -> Result<(), String> {
        //: The webview takes a moment to exist; the first command waits for it.
        let began = std::time::Instant::now();
        while ask(orders, "Runtime.evaluate", serde_json::json!({ "expression": "1" })).is_err() {
            if began.elapsed().as_secs() > 20 {
                return Err("the pane never answered".into());
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        println!("\n  argmind 의 브라우저 칸이 읽는 것 — 누르지 않음, 1280 폭");
        println!("  harness {}\n", argmind_measure::source_print(include_str!("browsing.rs")));

        let control = read(orders, "https://example.com/")?;
        println!("  대조 example.com: 노드 {} · 누를 수 있는 것 {} · 화면 안 {}",
                 control.nodes, control.named, control.shown.len());
        for (r, n) in &control.shown {
            println!("      {r} \"{n}\"");
        }
        if control.shown.len() != 1 {
            return Err("대조가 1 이 아닙니다 — 읽는 방법이 틀렸으니 아래 숫자를 믿지 마세요".into());
        }

        if std::env::args().any(|a| a == "--choosing") {
            return choosing(orders);
        }
        if std::env::args().any(|a| a == "--hands") {
            return hands(orders);
        }

        const SITES: [&str; 6] = [
            "https://www.naver.com/",
            "https://www.google.com/search?q=%EC%84%9C%EC%9A%B8+%EB%82%A0%EC%94%A8",
            "https://ko.wikipedia.org/wiki/%EC%84%9C%EC%9A%B8%ED%8A%B9%EB%B3%84%EC%8B%9C",
            "https://www.weather.go.kr/w/index.do",
            "https://news.ycombinator.com/",
            "https://github.com/rust-lang/rust",
        ];
        let mut under = 0;
        let mut ordered = 0;
        println!();
        for url in SITES {
            match read(orders, url) {
                Ok(r) => {
                    let ok = r.shown.len() <= 100;
                    under += ok as usize;
                    let sane = r.nodes > r.named && r.named >= r.shown.len() && !r.shown.is_empty();
                    ordered += sane as usize;
                    println!("  {url}\n      노드 {} · 누를 수 있는 것 {} · 화면 안 {} — {}{}",
                             r.nodes, r.named, r.shown.len(),
                             if ok { "100 이하" } else { "**100 넘음**" },
                             if sane { "" } else { " · **순서가 맞지 않음**" });
                    if std::env::var("BROWSING_DUMP").is_ok() {
                        for (i, (role, name)) in r.shown.iter().enumerate() {
                            println!("        {i:>3} {role} \"{name}\"");
                        }
                    }
                    let first: Vec<String> = r.shown.iter().take(12)
                        .map(|(role, name)| format!("{role} \"{name}\"")).collect();
                    println!("      {}", first.join(", "));
                }
                Err(why) => println!("  {url}\n      ** {why}"),
            }
        }
        println!("\n  ── 돌리기 전에 적어둔 대로 ──");
        println!("  · 화면 안 100 이하: {under}/6 — {}",
                 if under >= 4 { "다음 단계로 갈 만함" } else { "**기준 못 미침**" });
        println!("  · 세 숫자가 순서대로이고 0 이 아님: {ordered}/6");
        Ok(())
    }

    /// What a request needs: an item by role, and its name exactly or a part of it.
    enum Want {
        Is(&'static str, &'static str),
        Has(&'static str, &'static str),
        /// Nothing on the page can do this; the right answer is no effect.
        Nothing,
    }

    struct Page {
        url: &'static str,
        asks: [(&'static str, Want); 5],
    }

    const PAGES: [Page; 5] = [
        Page { url: "https://www.naver.com/", asks: [
            ("뉴스스탠드 탭 열어줘", Want::Is("tab", "뉴스스탠드")),
            ("검색창 눌러줘", Want::Is("combobox", "검색어를 입력해 주세요.")),
            ("메일 열어줘", Want::Is("link", "메일")),
            ("로그인 화면으로 가줘", Want::Is("link", "NAVER 로그인")),
            ("장바구니에 담긴 거 결제해줘", Want::Nothing),
        ] },
        Page { url: "https://ko.wikipedia.org/wiki/%EC%84%9C%EC%9A%B8%ED%8A%B9%EB%B3%84%EC%8B%9C", asks: [
            ("위키백과 검색창 눌러줘", Want::Is("searchbox", "위키백과 검색")),
            ("계정 만들어줘", Want::Is("link", "계정 만들기")),
            ("경복궁 문서로 가줘", Want::Is("link", "경복궁")),
            ("이 문서를 다른 언어로 보고 싶어", Want::Has("button", "개 언어")),
            ("동영상 재생해줘", Want::Nothing),
        ] },
        Page { url: "https://www.weather.go.kr/w/index.do", asks: [
            ("지진 정보 열어줘", Want::Is("link", "지진·화산")),
            ("태풍 정보 보여줘", Want::Is("link", "태풍")),
            ("지명 입력하는 칸 눌러줘", Want::Is("textbox", "주요지명으로 입력")),
            ("도움말 열어줘", Want::Is("link", "도움말")),
            ("내일 아침 7시에 알람 맞춰줘", Want::Nothing),
        ] },
        Page { url: "https://news.ycombinator.com/", asks: [
            ("new 페이지 열어줘", Want::Is("link", "new")),
            ("로그인 해줘", Want::Is("link", "login")),
            ("글 올리는 곳으로 가줘", Want::Is("link", "submit")),
            ("구인 글 보여줘", Want::Is("link", "jobs")),
            ("다크 모드 켜줘", Want::Nothing),
        ] },
        Page { url: "https://github.com/rust-lang/rust", asks: [
            ("브랜치 목록 보여줘", Want::Has("link", "Branches")),
            ("태그 목록 보여줘", Want::Has("link", "Tags")),
            ("Code 버튼 눌러줘", Want::Is("button", "Code")),
            ("가입하기", Want::Is("link", "Sign up")),
            ("이 저장소 삭제해줘", Want::Nothing),
        ] },
    ];

    struct Model(argmind_env_ollama::Local);

    impl argmind_core::contract::Speaks for Model {
        fn ask(&mut self, prompt: &str) -> String {
            self.0.ask(prompt).map(|r| r.text).unwrap_or_default()
        }
    }

    fn choosing(orders: &Sender<Say>) -> Result<(), String> {
        use argmind_core::intent::Trigger;
        use argmind_core::session::Session;
        use argmind_measure::drive::{Fixed, Nothing};

        let mut warm = argmind_env_ollama::Local::new("qwen3:8b");
        warm.think = Some(false);
        for _ in 0..2 {
            let _ = warm.ask("Reply with the single word: ready");
        }
        println!("\n  ── 고르기: 실제 페이지의 목록, 출하된 프롬프트, qwen3:8b ──");
        let (mut right, mut answerable, mut declined, mut impossible, mut skipped) = (0, 0, 0, 0, 0);
        for page in &PAGES {
            let r = read(orders, page.url)?;
            println!("\n  {} — 화면 안 {}", page.url, r.widgets.len());
            for (said, want) in &page.asks {
                //: The item this request means, in this run's list — exactly one, or the request is
                //: unanswerable today rather than wrong.
                let target = match want {
                    Want::Nothing => None,
                    Want::Is(role, name) | Want::Has(role, name) => {
                        let hits: Vec<(usize, &argmind_core::contract::Widget)> = r.widgets.iter()
                            .enumerate()
                            .filter(|(_, w)| w.role == *role && match want {
                                Want::Is(..) => w.name == *name,
                                _ => w.name.contains(name),
                            }).collect();
                        if hits.len() != 1 {
                            skipped += 1;
                            println!("     {said}\n       건너뜀 — {role} \"{name}\" 이 {}개", hits.len());
                            continue;
                        }
                        Some(hits[0])
                    }
                };
                let mut model = argmind_env_ollama::Local::new("qwen3:8b");
                model.think = Some(false);
                let mut session = Session::new(Fixed(r.widgets.clone()), Nothing, Model(model));
                let intent = session.decide(&Trigger::said(said));
                let did: Vec<String> = intent.effects.iter()
                    .map(|e| format!("{} {} \"{}\"", e.kind, e.role, e.target)).collect();
                let (ok, at) = match target {
                    None => {
                        impossible += 1;
                        let ok = intent.effects.is_empty();
                        declined += ok as usize;
                        (ok, String::new())
                    }
                    Some((i, w)) => {
                        answerable += 1;
                        let ok = intent.effects.iter()
                            .any(|e| e.kind == "click" && e.target == w.name);
                        right += ok as usize;
                        (ok, format!(" · 정답은 {i}번째"))
                    }
                };
                println!("     {said}\n       {} {}{at}  — {}", if ok { "●" } else { "○" },
                         if did.is_empty() { "(행동 없음)".to_string() } else { did.join(", ") },
                         intent.say.chars().take(50).collect::<String>());
            }
        }
        let rate = right as f64 / answerable.max(1) as f64;
        println!("\n  ── 돌리기 전에 적어둔 대로 ──");
        println!("  · 답이 있는 요청: {right}/{answerable} ({:.0}%) — {}", rate * 100.0,
                 if rate >= 0.6 { "60% 이상" } else { "**60% 못 미침**" });
        println!("  · 할 수 없는 요청을 거절: {declined}/{impossible} — {}",
                 if impossible > 0 && declined * 5 >= impossible * 4 { "기준대로" } else { "**기준 못 미침**" });
        println!("  · 건너뜀 (목록에 0개이거나 2개 이상): {skipped}");
        Ok(())
    }

    fn hands(orders: &Sender<Say>) -> Result<(), String> {
        use argmind_core::contract::{Did, Eyes, Hands};
        use argmind_env_browser::Pane;
        use serde_json::json;
        use std::sync::{Arc, Mutex};

        let wire_orders = Mutex::new(orders.clone());
        let pane = Pane::new(Arc::new(move |method: &str, params: &str| -> Result<String, String> {
            let (reply, answer) = channel();
            wire_orders.lock().map_err(|_| "poisoned".to_string())?
                .send(Say::Devtools { method: method.into(), params: params.into(), reply })
                .map_err(|_| "the window has gone".to_string())?;
            answer.recv_timeout(std::time::Duration::from_secs(30))
                .map_err(|_| format!("{method}: no answer"))?
        }));
        let go = |url: &str| -> Result<(), String> {
            pane.ask("Page.navigate", json!({ "url": url }))?;
            let began = std::time::Instant::now();
            loop {
                std::thread::sleep(std::time::Duration::from_millis(300));
                if pane.eval("document.readyState")? == "complete" { break; }
                if began.elapsed().as_secs() > 25 { return Err(format!("{url}: not loaded")); }
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
            Ok(())
        };
        let host = || pane.eval("location.host").ok().and_then(|v| v.as_str().map(String::from))
            .unwrap_or_default();
        let mut passed = 0;
        println!("\n  ── 칸 안에서 누르기 · 입력 — 페이지 이벤트만, 운영체제 입력 없음 ──");

        //: 1. A click that lands, and a pointer that does not move.
        go("https://example.com/")?;
        let mut eyes = pane.eyes();
        let mut hands = pane.hands(true);
        let link = eyes.foreground().into_iter().find(|w| w.name == "Learn more")
            .ok_or("no Learn more on example.com")?;
        let before = argmind_env_windows::cursor_now();
        hands.click(&link);
        let after = argmind_env_windows::cursor_now();
        std::thread::sleep(std::time::Duration::from_secs(3));
        let landed = host();
        let ok = hands.done.last() == Some(&Did::Sent { live: true }) && landed.contains("iana")
            && before == after;
        passed += ok as usize;
        println!("  1. 누르기: {:?} → {landed} · 포인터 {before:?} → {after:?} — {}",
                 hands.done.last(), if ok { "기준대로" } else { "**기준과 다름**" });

        //: 2. Typing into the page's own box.
        go("https://ko.wikipedia.org/wiki/%EC%84%9C%EC%9A%B8%ED%8A%B9%EB%B3%84%EC%8B%9C")?;
        let boxed = eyes.foreground().into_iter().find(|w| w.role == "searchbox")
            .ok_or("no search box")?;
        hands.click(&boxed);
        std::thread::sleep(std::time::Duration::from_millis(400));
        hands.type_text("경복궁");
        std::thread::sleep(std::time::Duration::from_millis(400));
        let value = pane.eval("document.activeElement && document.activeElement.value")?;
        let ok = value == "경복궁";
        passed += ok as usize;
        println!("  2. 입력: 검색창 값 {value} — {}", if ok { "기준대로" } else { "**기준과 다름**" });

        //: 3. The negative control: the link covered, in the pane's own copy of the page.
        go("https://example.com/")?;
        let link = eyes.foreground().into_iter().find(|w| w.name == "Learn more")
            .ok_or("no Learn more the second time")?;
        pane.eval("(() => { const d = document.createElement('div'); \
                   d.style.cssText = 'position:fixed;inset:0;z-index:99999;background:rgba(0,0,0,0.01)'; \
                   document.body.appendChild(d); return 1; })()")?;
        hands.click(&link);
        std::thread::sleep(std::time::Duration::from_secs(2));
        let stayed = host();
        let ok = hands.done.last() == Some(&Did::Elsewhere { owner: None })
            && stayed.contains("example.com");
        passed += ok as usize;
        println!("  3. 가린 대상: {:?} · 페이지 {stayed} — {}", hands.done.last(),
                 if ok { "거부됨, 기준대로" } else { "**눌렸거나 다른 답**" });

        //: 4. Password, both ways.
        let secret = pane.eval("(() => { const i = document.createElement('input'); i.type = 'password'; \
                                document.body.appendChild(i); i.focus(); return 1; })()")
            .map(|_| hands.into_a_secret())?;
        let plain = pane.eval("(() => { const i = document.createElement('input'); i.type = 'text'; \
                               document.body.appendChild(i); i.focus(); return 1; })()")
            .map(|_| hands.into_a_secret())?;
        let ok = secret && !plain;
        passed += ok as usize;
        println!("  4. 비밀번호 칸 {secret} · 보통 칸 {plain} — {}",
                 if ok { "기준대로" } else { "**기준과 다름**" });

        println!("\n  ── 네 가지 중 {passed} 가지 기준대로 ──");
        Ok(())
    }
}
