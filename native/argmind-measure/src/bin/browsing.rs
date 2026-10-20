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
            }
        }
        Ok(Read { nodes: nodes.len(), named, shown })
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
}
