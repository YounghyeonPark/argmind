//! The application: a local server, and a TypeScript interface in your browser.
//!
//! ```text
//!   serve              read the screen, propose, and act only on a confirmation
//!   serve --dry        everything except pressing anything
//! ```
//!
//! ## The argument this has to keep
//!
//! **An effect the model returned is not something the agent did.** Every one becomes a row with
//! two buttons, and [`act`] is the only function here that calls `perform` — the same claim
//! `app.py` makes with an AST test, made here by there being one place to look.
//!
//! ## Two things a window gave for free
//!
//! A server is reachable by anything else running as this user, and this agent can click. So it
//! binds to loopback, checked before the socket opens, and every request carries a token minted at
//! startup. **Localhost is not a permission.**
//!
//! ## One thing a browser makes harder
//!
//! The interface is a tab in somebody else's process, and the person is looking at it whenever they
//! ask for anything — so the window in front is never the window they mean. [`Attending`] reads the
//! newest window that is *not* in front, and answers with nothing rather than with the page the
//! person is reading.
//!
//! ## Why one thread owns everything
//!
//! **UI Automation is apartment-bound and its pointers are not `Send`.** The compiler refuses to
//! move the session between threads, which is the same constraint `app.py` meets by putting the
//! agent on its own thread with an inbox. One thread owns the session, the eyes, the hands and the
//! model; connection threads only post [`Job`]s and read a channel back. The ordering of turns is
//! then a queue rather than a lock, which is also what you want.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};

use argmind_core::contract::{Speaks, Widget};
use argmind_core::intent::{Effect, Trigger};
use argmind_core::session::Session;
use argmind_env_ollama::Local;
use argmind_env_windows::hands::Did;
use argmind_env_windows::{Attending, SapiSpeaker, Watcher, WindowsHands};
use argmind_voice::Speaker;
use argmind_serve::proposals::{Answered, Proposal, Proposals};
use argmind_serve::{authorised, event_stream_head, is_loopback, mint_token, parse_request, respond,
                    route, sse, Route};

const INDEX: &str = include_str!("../../../../ui/src/index.html");
const SCRIPT: &str = include_str!("../../../../ui/dist/app.js");
const ADDRESS: &str = "127.0.0.1:7331";

struct Model(Local);

impl Speaks for Model {
    fn ask(&mut self, prompt: &str) -> String {
        self.0.ask(prompt).map(|r| r.text).unwrap_or_else(|e| {
            eprintln!("  the model did not answer: {e}");
            String::new()
        })
    }
}

/// Everything the browser is told, as it happens.
#[derive(Default)]
struct Listeners(Mutex<Vec<Sender<String>>>);

impl Listeners {
    fn send(&self, event: &str, data: &str) {
        let frame = sse(event, data);
        let Ok(mut who) = self.0.lock() else { return };
        //: A page that closed is a channel that errors, and dropping it here is the only cleanup
        //: there is — nothing tells a server that a browser tab went away.
        who.retain(|tx| tx.send(frame.clone()).is_ok());
    }
}

struct Desk {
    session: Session<Attending, WindowsHands, Model>,
    proposals: Proposals,
    dry: bool,
    /// **`None` unless asked for.** Speaking makes a noise in a room, which is not something to
    /// start doing because a program was run — see `--speak`.
    voice: Option<SapiSpeaker>,
}

/// What a connection asks the agent to do.
///
/// **Nothing here is an action.** `Confirm` releases a proposal and the agent decides what that
/// means, which keeps the single call to `perform` in one place rather than one per thread.
enum Job {
    Said(String),
    Confirm(u64),
    Decline(u64),
    /// A page opened: send it the snapshot it needs to draw itself.
    Opened(Sender<String>),
}

fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn as_json(p: &Proposal) -> String {
    format!("{{\"token\":{},\"label\":\"{}\",\"hint\":\"{}\"}}",
            p.token, escape(&p.label), escape(&p.hint))
}

/// What the person reads on the button.
fn label(effect: &Effect, found: Option<&Widget>) -> String {
    match effect.kind.as_str() {
        "click" => match found {
            Some(w) => format!("클릭  {} \u{201c}{}\u{201d}", w.role, w.name),
            None => format!("클릭  \u{201c}{}\u{201d}", effect.target),
        },
        "type" => format!("입력  \u{201c}{}\u{201d}", effect.text),
        _ => format!("키  {}", effect.text),
    }
}

/// One turn: ask, say, and turn every effect into a row that waits.
fn think(desk: &mut Desk, out: &Listeners, text: &str) {
    out.send("busy", "1");
    //: **The last turn's buttons go.** An offer the person never answered and then said something
    //: else about is one they moved on from, and a button whose label describes a screen that has
    //: gone is worse than no button.
    for dropped in desk.proposals.clear() {
        out.send("dropped", &dropped.token.to_string());
    }
    let (handle, title) = desk.session.eyes.window();
    out.send("attending", &title);
    if handle == 0 {
        out.send("note", "보고 있는 창이 없습니다 — 다른 창을 한 번 눌렀다 오세요");
    }

    desk.session.observe();
    let intent = desk.session.decide(&Trigger::said(text));

    desk.session.history.push(("them".into(), text.to_string()));
    if !intent.say.is_empty() {
        desk.session.history.push(("you".into(), intent.say.clone()));
        out.send("say", &intent.say);
        //: **Only because they asked.** A reply to something typed is an answer; nothing here
        //: fires unprompted yet, and when something does this is the line that must not simply
        //: inherit permission from it — a proposal waits behind a button, a sentence waits behind
        //: nothing.
        if let Some(voice) = desk.voice.as_mut() {
            //: Barge-in: a new answer replaces the last one rather than queueing behind ~3.1 s of
            //: speech.
            voice.stop();
            voice.say(&intent.say);
        }
    } else if intent.effects.is_empty() {
        out.send("say", "…");
    }
    for effect in &intent.effects {
        propose(desk, out, effect.clone());
    }
    out.send("busy", "0");
}

/// File one offer. **This does not act.**
fn propose(desk: &mut Desk, out: &Listeners, effect: Effect) {
    let (text, hint) = if effect.kind == "click" {
        match desk.session.find(&effect.target, &effect.role) {
            None => {
                out.send("gone",
                         &format!("{{\"token\":-1,\"what\":\"{}\"}}", escape(&effect.target)));
                return;
            }
            Some(found) => {
                let (cx, cy) = found.centre();
                //: **A second opinion, not a gate.** A widget's own centre resolves back to itself
                //: about seven times in ten, and a caption button answering `application` is the
                //: hit test disagreeing with a pixel that is correct.
                let hint = match desk.session.eyes.inner.at_point(cx, cy) {
                    Some((name, role)) if name != found.name
                        && !found.name.contains(&name) && !name.contains(&found.name) => {
                        format!("그 지점에는 {} {}",
                                if role.is_empty() { "?" } else { role.as_str() },
                                if name.is_empty() { "(이름 없음)" } else { name.as_str() })
                    }
                    _ => String::new(),
                };
                (label(&effect, Some(&found)), hint)
            }
        }
    } else {
        (label(&effect, None), String::new())
    };
    //: **In the conversation, not only on the screen.** An offer the person then argues with —
    //: *"not that one"* — has to be something the next turn can see.
    desk.session.remember_doing(&[], &[], std::slice::from_ref(&effect), &[]);
    desk.proposals.offer(effect, text, hint);
    if let Some(p) = desk.proposals.waiting().last() {
        out.send("propose", &as_json(p));
    }
}

/// **The only place in this program that performs anything.**
fn act(desk: &mut Desk, out: &Listeners, p: Proposal) {
    let (did, _proposed, refused) = desk.session.perform(std::slice::from_ref(&p.effect));
    desk.session.remember_doing(&did, &refused, &[], &[]);
    if let Some(Did::Elsewhere { owner }) = desk.session.hands.done.last() {
        out.send("note",
                 &format!("그 지점은 다른 프로그램({owner:?})의 것이라 누르지 않았습니다"));
        return;
    }
    if !refused.is_empty() {
        //: **Gone between the offer and the button** — the case the whole re-resolve exists for,
        //: and a proper outcome rather than an error.
        out.send("gone",
                 &format!("{{\"token\":{},\"what\":\"{}\"}}", p.token, escape(&p.effect.target)));
        return;
    }
    let confirmed = match did.first().and_then(|d| d.confirmed) {
        Some(b) => b.to_string(),
        None => "null".into(),
    };
    out.send("ran", &format!("{{\"token\":{},\"label\":\"{}\",\"confirmed\":{confirmed}}}",
                             p.token, escape(&p.label)));
}

fn hello(desk: &Desk, model: &str) -> String {
    let waiting: Vec<String> = desk.proposals.waiting().iter().map(as_json).collect();
    let fields = [
        format!("\"model\":\"{}\"", escape(model)),
        format!("\"dry\":{}", desk.dry),
        format!("\"speaks\":{}", desk.voice.is_some()),
        format!("\"attending\":\"{}\"", escape(&desk.session.eyes.window().1)),
        format!("\"waiting\":[{}]", waiting.join(",")),
    ];
    //: Assembled from a list rather than one long format string. The first version wrapped that
    //: string across two lines and the indentation of the second landed **inside the JSON** —
    //: harmless, since whitespace between tokens is legal, and exactly the kind of thing that is
    //: harmless until the day something reads the payload byte for byte.
    format!("{{{}}}", fields.join(","))
}

fn main() {
    let dry = std::env::args().any(|a| a == "--dry");
    //: **Off by default, and that is not a style choice.** Everything measured about this voice was
    //: measured at volume zero into a memory stream so a desktop somebody was using stayed quiet;
    //: making a sound is a thing to ask for, not a thing to inherit from starting a program.
    let speak = std::env::args().any(|a| a == "--speak");
    let model_name = std::env::args().skip(1).find(|a| a.contains(':'))
        .unwrap_or_else(|| "qwen3:8b".into());

    //: **Checked before the socket opens.** An agent that can press buttons on your screen has no
    //: business accepting a connection from the network.
    assert!(is_loopback(ADDRESS), "{ADDRESS} is not loopback");
    let token = mint_token();
    let listener = TcpListener::bind(ADDRESS).expect("could not bind");
    println!("\n  argmind — http://{ADDRESS}/");
    //: Both settings, on the line a person actually reads. *"It did nothing"* and *"it said
    //: nothing"* are the two things that would otherwise look like the agent being broken.
    println!("  {model_name} | {} | {}\n",
             if dry { "dry — nothing will be pressed" } else { "live — it will press buttons" },
             if speak { "speaking aloud" } else { "silent" });

    let out = Arc::new(Listeners::default());
    let (jobs, inbox) = channel::<Job>();
    let agent_out = Arc::clone(&out);
    let agent_model = model_name.clone();
    //: Everything COM touches lives on this thread and never leaves it.
    std::thread::spawn(move || {
        let watcher = Watcher::default();
        watcher.watch();
        let eyes = Attending::new(watcher).expect("UI Automation would not open");
        //: Unrestricted, and **the confirm button is what makes that safe** — the same trade
        //: `app.py` makes by inverting the unprompted flag. Dry unless asked otherwise.
        let hands = if dry {
            WindowsHands::unrestricted()
        } else {
            WindowsHands::unrestricted().arm()
        };
        let mut model = Local::new(&agent_model);
        model.think = Some(false);
        let voice = if speak {
            match SapiSpeaker::new() {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("  no voice: {e}");
                    None
                }
            }
        } else {
            None
        };
        if let Some(v) = &voice {
            //: Said once, at startup, rather than discovered when a Korean reply comes out silent.
            if !v.has_korean() {
                eprintln!("  a Korean voice is not installed — Korean replies will not be spoken");
            }
        }
        let mut desk = Desk { session: Session::new(eyes, hands, Model(model)),
                              proposals: Proposals::default(), dry, voice };
        while let Ok(job) = inbox.recv() {
            match job {
                Job::Said(text) => think(&mut desk, &agent_out, text.trim()),
                Job::Confirm(n) => {
                    if let Answered::Run(p) = desk.proposals.confirm(n) {
                        act(&mut desk, &agent_out, p);
                    }
                }
                Job::Decline(n) => {
                    if let Answered::TurnedDown(p) = desk.proposals.decline(n) {
                        //: Pressing *no* is a thing the person said, and the next turn has to know.
                        desk.session.remember_doing(&[], &[], &[],
                                                    std::slice::from_ref(&p.effect));
                        agent_out.send("dropped", &p.token.to_string());
                    }
                }
                Job::Opened(tx) => {
                    let _ = tx.send(sse("hello", &hello(&desk, &agent_model)));
                }
            }
        }
    });

    for stream in listener.incoming().flatten() {
        let (out, token, jobs) = (Arc::clone(&out), token.clone(), jobs.clone());
        std::thread::spawn(move || serve_one(stream, &out, &token, &jobs));
    }
}

fn serve_one(mut stream: TcpStream, out: &Listeners, token: &str, jobs: &Sender<Job>) {
    let mut reader = BufReader::new(match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    });
    let mut head = String::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).unwrap_or(0) == 0 {
            return;
        }
        let blank = line == "\r\n";
        head.push_str(&line);
        if blank {
            break;
        }
    }
    let length: usize = head.to_lowercase().split("content-length:").nth(1)
        .and_then(|r| r.split("\r\n").next()).and_then(|n| n.trim().parse().ok()).unwrap_or(0);
    let mut body = vec![0u8; length];
    if length > 0 && reader.read_exact(&mut body).is_err() {
        return;
    }
    let raw = format!("{head}{}", String::from_utf8_lossy(&body));
    let Some(request) = parse_request(&raw) else {
        let _ = stream.write_all(&respond("400 Bad Request", "text/plain", "no"));
        return;
    };

    //: **The page and its script are the two things served without a token**, because the token
    //: arrives inside the page. Neither carries anything the agent knows: that all comes down
    //: `/events`, which does need one.
    let is_page = matches!(route(&request), Route::Ui) || request.path == "/app.js";
    if !is_page && !authorised(token, request.token.as_deref()) {
        let _ = stream.write_all(&respond("403 Forbidden", "text/plain", "no token"));
        return;
    }

    if request.path == "/app.js" {
        let _ = stream.write_all(&respond("200 OK", "text/javascript; charset=utf-8", SCRIPT));
        return;
    }
    match route(&request) {
        Route::Ui => {
            let page = INDEX.replace("__TOKEN__", token);
            let _ = stream.write_all(&respond("200 OK", "text/html; charset=utf-8", &page));
        }
        Route::Events => {
            let (tx, rx) = channel::<String>();
            if jobs.send(Job::Opened(tx.clone())).is_err() {
                return;
            }
            if let Ok(mut who) = out.0.lock() {
                who.push(tx);
            }
            let _ = stream.write_all(&event_stream_head());
            //: Blocks for the life of the page. One thread a tab is plenty for an interface with
            //: one reader, and a great deal less machinery than the alternative.
            while let Ok(frame) = rx.recv() {
                if stream.write_all(frame.as_bytes()).is_err() {
                    return;
                }
            }
        }
        Route::Said(text) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Said(text));
        }
        Route::Confirm(n) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Confirm(n));
        }
        Route::Decline(n) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Decline(n));
        }
        Route::Unknown => {
            let _ = stream.write_all(&respond("404 Not Found", "text/plain", "no"));
        }
    }
}
