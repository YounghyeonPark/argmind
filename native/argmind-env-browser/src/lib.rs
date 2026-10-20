//! **Eyes and hands for argmind's own browser pane** — the same contract as UI Automation, over
//! the Chrome DevTools Protocol, inside a page nobody else is using.
//!
//! The owner's choice on 2026-10-20: a pane inside argmind's window rather than a tab in the
//! person's browser or a separate one, **because argmind's work must not collide with the
//! person's**. Everything here acts on the page and nothing reaches the operating system: a click
//! is `Input.dispatchMouseEvent` to that page, typing is `Input.insertText` to that page's focus,
//! a key is `Input.dispatchKeyEvent` to that page. The person's pointer, keyboard focus and windows
//! are never involved, which is the whole of the argument for the pane.
//!
//! ## WHERE is still the deterministic side's
//!
//! **The contract survives a third environment unchanged**: `Widget { name, role, rect, window }`
//! is what an accessibility tree hands over here as on Windows and Linux. Reading is the
//! accessibility tree for names and roles, a layout snapshot for where, and the layout metrics for
//! what is on screen — the narrowing `--bin browsing` measured, 65–155 things a page. **The model is
//! shown names and roles and never a rectangle**, exactly as before.
//!
//! **A rectangle here is in CSS pixels of the visible page**, because that is what the input events
//! take. The snapshot reports device pixels — the harness's own control failed on exactly that
//! difference at a display scale of 1.5 — so the conversion is made once, in [`WebEyes`].
//!
//! ## What a click checks first
//!
//! The same question the Windows hands' reach guard asks: **is the thing at that point the thing
//! that was meant?** A banner or a cookie notice can sit over a link; a click there would press the
//! banner. So the page is asked, by the target node itself, whether `elementFromPoint` at its centre
//! is it or inside it, and a click that would land on something else is refused and recorded as
//! `Did::Elsewhere` rather than sent.
//!
//! ## A password is still the person's
//!
//! [`Hands::into_a_secret`] reads the page's focused element: `type="password"`, or an
//! `autocomplete` that says password. On Windows the same question goes to UI Automation's
//! `IsPassword`; here the page answers it directly.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use argmind_core::contract::{Did, Eyes, Hands, Widget};
use serde_json::{json, Value};

/// **What carries one DevTools command to the page and its answer back.** Handed in, so this crate
/// knows nothing about WebView2 or windows: `method`, `params` as JSON, and the reply's JSON.
pub trait Wire: Send + Sync {
    fn call(&self, method: &str, params: &str) -> Result<String, String>;
}

impl<F: Fn(&str, &str) -> Result<String, String> + Send + Sync> Wire for F {
    fn call(&self, method: &str, params: &str) -> Result<String, String> {
        self(method, params)
    }
}

/// Interactive roles as Chromium's accessibility tree names them — the narrowing `browsing`
/// measured with.
pub const PRESSABLE: [&str; 16] = ["button", "link", "textbox", "searchbox", "combobox", "checkbox",
                                   "radio", "menuitem", "menuitemcheckbox", "menuitemradio", "tab",
                                   "switch", "slider", "spinbutton", "option", "treeitem"];

/// What the eyes saw and the hands need: which page node each listed widget is.
#[derive(Default)]
struct Seen(HashMap<(String, String, (i64, i64, i64, i64)), i64>);

/// The pane, shared by its eyes and its hands.
#[derive(Clone)]
pub struct Pane {
    wire: Arc<dyn Wire>,
    seen: Arc<Mutex<Seen>>,
}

impl Pane {
    pub fn new(wire: Arc<dyn Wire>) -> Pane {
        Pane { wire, seen: Arc::new(Mutex::new(Seen::default())) }
    }

    /// One command, its answer as JSON.
    pub fn ask(&self, method: &str, params: Value) -> Result<Value, String> {
        let text = self.wire.call(method, &params.to_string())?;
        serde_json::from_str(&text).map_err(|e| format!("{method}: {e}"))
    }

    /// One expression's value in the page.
    pub fn eval(&self, expression: &str) -> Result<Value, String> {
        let r = self.ask("Runtime.evaluate",
                         json!({ "expression": expression, "returnByValue": true }))?;
        Ok(r["result"]["value"].clone())
    }

    pub fn eyes(&self) -> WebEyes {
        WebEyes { pane: self.clone() }
    }

    pub fn hands(&self, live: bool) -> WebHands {
        WebHands { pane: self.clone(), live, done: Vec::new() }
    }
}

/// Reading the page.
pub struct WebEyes {
    pane: Pane,
}

/// **The list, from three answers the page gave** — pure, so the arithmetic that went wrong once
/// can be tested without a page. Returns the widgets and, for each, its page node.
pub fn listing(tree: &Value, snapshot: &Value, metrics: &Value, title: &str)
               -> Vec<(Widget, i64)> {
    let doc = &snapshot["documents"][0];
    let ids = doc["nodes"]["backendNodeId"].as_array().cloned().unwrap_or_default();
    let index = doc["layout"]["nodeIndex"].as_array().cloned().unwrap_or_default();
    let bounds = doc["layout"]["bounds"].as_array().cloned().unwrap_or_default();
    let mut boxes = HashMap::new();
    for (i, b) in index.iter().zip(bounds.iter()) {
        let Some(i) = i.as_u64() else { continue };
        let Some(id) = ids.get(i as usize).and_then(Value::as_i64) else { continue };
        let n: Vec<f64> = b.as_array().map(|a| a.iter().filter_map(Value::as_f64).collect())
            .unwrap_or_default();
        if n.len() == 4 {
            boxes.insert(id, (n[0], n[1], n[2], n[3]));
        }
    }
    //: **Device pixels on both sides, then CSS pixels out.** The snapshot's boxes and
    //: `layoutViewport` share a unit; the ratio to `cssLayoutViewport` is the display scale.
    let device = &metrics["layoutViewport"];
    let css = &metrics["cssLayoutViewport"];
    let f = |v: &Value, k: &str| v[k].as_f64().unwrap_or(0.0);
    let (vx, vy, vw, vh) = (f(device, "pageX"), f(device, "pageY"),
                            f(device, "clientWidth"), f(device, "clientHeight"));
    let scale = match f(css, "clientWidth") {
        w if w > 0.0 && vw > 0.0 => vw / w,
        _ => 1.0,
    };
    let mut out = Vec::new();
    for n in tree["nodes"].as_array().into_iter().flatten() {
        if n["ignored"].as_bool().unwrap_or(false) {
            continue;
        }
        let role = n["role"]["value"].as_str().unwrap_or("");
        let name = n["name"]["value"].as_str().unwrap_or("").trim();
        if !PRESSABLE.contains(&role) || name.is_empty() {
            continue;
        }
        let Some(id) = n["backendDOMNodeId"].as_i64() else { continue };
        let Some(&(x, y, w, h)) = boxes.get(&id) else { continue };
        if !(w > 0.0 && h > 0.0 && x < vx + vw && x + w > vx && y < vy + vh && y + h > vy) {
            continue;
        }
        let rect = (((x - vx) / scale).round() as i64, ((y - vy) / scale).round() as i64,
                    (w / scale).round().max(1.0) as i64, (h / scale).round().max(1.0) as i64);
        out.push((Widget::new(name, role, rect, title), id));
    }
    out
}

impl WebEyes {
    /// **The window a rule matches is the page's title and its host** — so a rule can name a site
    /// (`naver.com`) as well as a page.
    pub fn window(&self) -> String {
        let title = self.pane.eval("document.title").ok()
            .and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
        let host = self.pane.eval("location.host").ok()
            .and_then(|v| v.as_str().map(str::to_string)).unwrap_or_default();
        match (title.is_empty(), host.is_empty()) {
            (_, true) => title,
            (true, false) => host,
            (false, false) => format!("{title} — {host}"),
        }
    }

    fn read(&mut self) -> Result<Vec<Widget>, String> {
        let tree = self.pane.ask("Accessibility.getFullAXTree", json!({}))?;
        let snap = self.pane.ask("DOMSnapshot.captureSnapshot", json!({ "computedStyles": [] }))?;
        let metrics = self.pane.ask("Page.getLayoutMetrics", json!({}))?;
        let listed = listing(&tree, &snap, &metrics, &self.window());
        let mut seen = self.pane.seen.lock().map_err(|_| "poisoned".to_string())?;
        seen.0.clear();
        for (w, id) in &listed {
            seen.0.insert((w.name.clone(), w.role.clone(), w.rect), *id);
        }
        Ok(listed.into_iter().map(|(w, _)| w).collect())
    }
}

impl Eyes for WebEyes {
    /// **A page that will not answer is a page with nothing on it**, which is what the Windows eyes
    /// say about a window that closed mid-read — never an invented list.
    fn foreground(&mut self) -> Vec<Widget> {
        self.read().unwrap_or_default()
    }

    fn timed_foreground(&mut self) -> (Vec<Widget>, f64) {
        let began = std::time::Instant::now();
        let w = self.foreground();
        (w, began.elapsed().as_secs_f64())
    }
}

/// The keys the pane can press, by the names the Windows hands use, with what the page is told.
pub const KEYS: [(&str, &str, &str, i64); 13] = [
    ("enter", "Enter", "Enter", 13), ("tab", "Tab", "Tab", 9), ("escape", "Escape", "Escape", 27),
    ("backspace", "Backspace", "Backspace", 8), ("delete", "Delete", "Delete", 46),
    ("up", "ArrowUp", "ArrowUp", 38), ("down", "ArrowDown", "ArrowDown", 40),
    ("left", "ArrowLeft", "ArrowLeft", 37), ("right", "ArrowRight", "ArrowRight", 39),
    ("pageup", "PageUp", "PageUp", 33), ("pagedown", "PageDown", "PageDown", 34),
    ("home", "Home", "Home", 36), ("end", "End", "End", 35),
];

/// Touching the page, and only the page.
pub struct WebHands {
    pane: Pane,
    live: bool,
    /// What each action did, in order — the same record the Windows hands keep, so the application
    /// can tell a refusal from a press that landed.
    pub done: Vec<Did>,
}

impl WebHands {
    /// **Is the thing at this widget's centre the widget, or inside it?** Asked of the node itself,
    /// so the answer is about this node and not about a name that might be on the page twice.
    fn reaches(&self, id: i64, x: f64, y: f64) -> Result<bool, String> {
        let node = self.pane.ask("DOM.resolveNode", json!({ "backendNodeId": id }))?;
        let object = node["object"]["objectId"].as_str()
            .ok_or("the node is gone")?.to_string();
        let r = self.pane.ask("Runtime.callFunctionOn", json!({
            "objectId": object,
            "functionDeclaration": "function(x, y) { const e = document.elementFromPoint(x, y); \
                                    return !!e && (e === this || this.contains(e) || e.contains(this)); }",
            "arguments": [{ "value": x }, { "value": y }],
            "returnByValue": true,
        }))?;
        Ok(r["result"]["value"].as_bool().unwrap_or(false))
    }

    fn click_widget(&mut self, widget: &Widget) -> Did {
        let id = self.pane.seen.lock().ok()
            .and_then(|s| s.0.get(&(widget.name.clone(), widget.role.clone(), widget.rect)).copied());
        let Some(id) = id else {
            return Did::Refused("the page no longer has it".into());
        };
        let (x, y) = {
            let (l, t, w, h) = widget.rect;
            (l as f64 + w as f64 / 2.0, t as f64 + h as f64 / 2.0)
        };
        match self.reaches(id, x, y) {
            Ok(true) => {}
            //: **Covered: refused, not pressed.** The point belongs to something else on the page.
            Ok(false) => return Did::Elsewhere { owner: None },
            Err(why) => return Did::Refused(why),
        }
        if !self.live {
            return Did::Sent { live: false };
        }
        for kind in ["mousePressed", "mouseReleased"] {
            if let Err(why) = self.pane.ask("Input.dispatchMouseEvent", json!({
                "type": kind, "x": x, "y": y, "button": "left", "clickCount": 1 })) {
                return Did::Refused(why);
            }
        }
        Did::Sent { live: true }
    }

    fn type_into(&mut self, text: &str) -> Did {
        if !self.live {
            return Did::Sent { live: false };
        }
        match self.pane.ask("Input.insertText", json!({ "text": text })) {
            Ok(_) => Did::Sent { live: true },
            Err(why) => Did::Refused(why),
        }
    }

    fn press_key(&mut self, name: &str) -> Did {
        let Some(&(_, key, code, vk)) = KEYS.iter().find(|(n, ..)| n.eq_ignore_ascii_case(name)) else {
            return Did::Refused(format!("no key called {name}"));
        };
        if !self.live {
            return Did::Sent { live: false };
        }
        for kind in ["keyDown", "keyUp"] {
            let mut event = json!({ "type": kind, "key": key, "code": code,
                                    "windowsVirtualKeyCode": vk });
            //: Enter has to carry its character or a form does not see a submit.
            if kind == "keyDown" && key == "Enter" {
                event["text"] = json!("\r");
            }
            if let Err(why) = self.pane.ask("Input.dispatchKeyEvent", event) {
                return Did::Refused(why);
            }
        }
        Did::Sent { live: true }
    }
}

impl Hands for WebHands {
    fn live(&self) -> bool {
        self.live
    }
    fn click(&mut self, widget: &Widget) {
        let did = self.click_widget(widget);
        self.done.push(did);
    }
    fn type_text(&mut self, text: &str) {
        let did = self.type_into(text);
        self.done.push(did);
    }
    fn press(&mut self, key: &str) {
        let did = self.press_key(key);
        self.done.push(did);
    }
    fn keys(&self) -> Option<Vec<String>> {
        Some(KEYS.iter().map(|(n, ..)| n.to_string()).collect())
    }
    /// **The page's own focus, read by the page.** Unreadable reads as no, the same direction the
    /// contract chose — and here unreadable means the page did not answer, so typing would fail too.
    fn into_a_secret(&mut self) -> bool {
        self.pane.eval("(() => { const e = document.activeElement; return !!e && \
                        (e.type === 'password' || /password/i.test(e.autocomplete || '')); })()")
            .ok().and_then(|v| v.as_bool()).unwrap_or(false)
    }
}
