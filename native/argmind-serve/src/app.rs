//! The application: a local server, and a TypeScript interface.
//!
//! **Two binaries run this, so it lives here.** `serve` is the console one and `argmind-app` is
//! the one with a window and an icon by the clock; a second copy of the server is how the two
//! come to behave differently on the day it matters.
//!
//! **The interface is a page either way.** `argmind-app` puts it in a window this program owns —
//! `argmind-window` — and `serve` leaves it at a URL for whatever you like to open. Nothing below
//! this line knows which, and that is worth keeping: it is HTTP and server-sent events, so the
//! thing on the other end is replaceable.
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
//! ## One thing an interface of any kind makes harder
//!
//! The person is looking at the interface whenever they ask for anything — so the window in front
//! is never the window they mean. [`Attending`] reads the newest window that is *not* in front,
//! and answers with nothing rather than with the page the person is reading. **That was written
//! when the interface was somebody else's browser** and it still holds now that the window is
//! ours; `attending.rs` says what changed and what it did not.
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

use argmind_core::contract::{Eyes, Speaks, Widget};
use argmind_core::intent::{Effect, Trigger};
use argmind_core::session::Session;
use argmind_core::watch::{poll, Looking, Watch};
use argmind_env_ollama::Local;
use argmind_env_whisper::Whisper;
use crate::platform::{self, Did, Ear, Mouth, Sight, Touch, Watching};
use argmind_voice::Speaker;
use crate::decided::{Answer, Standing};
use crate::proposals::{Answered, Proposal, Proposals};
use crate::{authorised, event_stream_head, is_loopback, mint_token, parse_request, respond,
            route, sse, Route};

const INDEX: &str = include_str!("../../../ui/src/index.html");
const SCRIPT: &str = include_str!("../../../ui/dist/app.js");
/// The sphere. **Its own module and its own file** — a hundred lines of projection arithmetic
/// inside the page's logic would bury the part anybody needs to read.
const ORB: &str = include_str!("../../../ui/dist/orb.js");

/// The files served without a token, because the token arrives inside the page.
///
/// **Public so the list can be asserted.** It is the program's whole unauthenticated surface, and
/// a route added to it by an edit somewhere else is the kind of change nobody reviews twice.
pub const SCRIPTS: [(&str, &str); 2] = [("/app.js", SCRIPT), ("/orb.js", ORB)];
/// The port argmind listens on, unless something says otherwise.
pub const PORT: u16 = 7331;

/// Where it listens. **Loopback, checked before the socket opens** — an agent that can press
/// buttons on your screen has no business accepting a connection from the network.
pub fn address(port: u16) -> String {
    format!("127.0.0.1:{port}")
}

struct Model(Local);

impl Speaks for Model {
    fn ask(&mut self, prompt: &str) -> String {
        self.0.ask(prompt).map(|r| r.text).unwrap_or_else(|e| {
            eprintln!("  the model did not answer: {e}");
            String::new()
        })
    }
}

/// Everything the page is told, as it happens.
#[derive(Default)]
struct Listeners(Mutex<Vec<Sender<String>>>);

impl Listeners {
    fn send(&self, event: &str, data: &str) {
        let frame = sse(event, data);
        let Ok(mut who) = self.0.lock() else { return };
        //: A page that closed is a channel that errors, and dropping it here is the only cleanup
        //: there is — nothing tells a server that a page went away.
        who.retain(|tx| tx.send(frame.clone()).is_ok());
    }
}

struct Desk {
    session: Session<Sight, Touch, Model>,
    proposals: Proposals,
    dry: bool,
    /// **`None` unless asked for.** Speaking makes a noise in a room, which is not something to
    /// start doing because a program was run — see `--speak`.
    ///
    /// **Boxed because there are two of them now**, and which one is in here is decided at startup
    /// and can change once: `argmind-env-qwentts` if the neural synthesiser is answering, and
    /// `SapiSpeaker` otherwise. Nothing above this line cares — [`argmind_voice::Speaker`] is four
    /// methods and both have them.
    voice: Option<Box<dyn Speaker>>,
    /// Which one it is, for the page and for the log. **Not derived from the value**, because the
    /// trait deliberately has no name on it: a `Speaker` is a thing that says and stops.
    voice_is: &'static str,
    /// Whether `--speak` was given. **Kept apart from `voice.is_some()`**, which is *can it speak
    /// right now* — the two differ exactly when speaking was asked for and no voice could be had,
    /// which is the case where a later install should still be picked up.
    wants_voice: bool,
    /// **`None` unless asked for**, for the same reason and more so: a microphone is not opened
    /// because a program was run. `--hear` turns it on, and even then it opens only while the talk
    /// key is held.
    ear: Option<(Ear, Whisper)>,
    /// **Off until somebody switches it on in the page.** With it on, a change of window starts a
    /// thought nobody asked for — and something on screen changes 13 seconds out of 14, so this is
    /// a standing invitation to spend a model call every few seconds.
    watches: Vec<Watch>,
    noticing: bool,
    /// What the screen looked like last time this checked. **Empty means there is no before**, and
    /// the first look after switching on reports nothing whatever is on screen.
    before: Vec<argmind_core::contract::Widget>,
    /// How many thoughts started without being asked for, and how many of those said something.
    /// **Counted where a person can see it**, because "how often does this happen on your desktop"
    /// is the question that decides whether it is worth having on.
    noticed: u64,
}

/// What a connection asks the agent to do.
///
/// **Nothing here is an action.** `Confirm` releases a proposal and the agent decides what that
/// means, which keeps the single call to `perform` in one place rather than one per thread.
enum Job {
    Said(String),
    /// Install the thing this names — **the button in the page was the approval**, and the page
    /// had to render what it would do before it could be pressed.
    Fix(String),
    /// Take a fresh reading of what is missing and tell the page.
    Look,
    Confirm(u64),
    Decline(u64),
    /// Watch the screen, or stop.
    Notice(bool),
    /// An answer to a permission question: which thing, and which of the four buttons.
    Decide(String, Answer),
    /// Take back a standing answer, from the settings.
    Forget(String),
    /// The talk key went down or came up.
    Listen(bool),
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

    let (widgets, look_s) = desk.session.observe();
    //: **Exactly what the model was shown, not a description of it.** The same call that builds the
    //: prompt builds this, so the panel cannot drift into showing something the model never saw —
    //: and *"if the thing you want is not in that list, no model will find it"* is only useful if
    //: the list on screen is the list that went out.
    let listed: Vec<_> = widgets.iter().take(desk.session.limit).cloned().collect();
    out.send("seeing", &argmind_core::converse::describe_screen(&listed, desk.session.limit));

    let started = std::time::Instant::now();
    let intent = desk.session.decide(&Trigger::said(text));
    let think_s = started.elapsed().as_secs_f64();
    //: **Both halves, apart.** Reading the screen and asking the model are 0.2 s and 0.6 s of very
    //: different things, and one number covering both hides which of them is slow today.
    out.send("spent", &format!("{{\"widgets\":{},\"listed\":{},\"look_ms\":{:.0},\
                                \"think_ms\":{:.0}}}",
                               widgets.len(), listed.len(), look_s * 1000.0, think_s * 1000.0));

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

/// What is missing, as the page needs it: the whole sentence a person reads before deciding.
fn needs_json(model: &str) -> String {
    let rows: Vec<String> = crate::needs::every(model).iter().map(|n| {
        //: **The description travels with the button.** A page that showed only a name and a
        //: button would be asking the vague question this whole design exists to avoid.
        let (offer, plan) = match &n.fix {
            Some(f) => (true, crate::fix::describe(f)),
            None => (false, String::new()),
        };
        //: **The warning is a property of the fix, not a phrase inside its description.** Emphasis
        //: written into the sentence renders as literal asterisks in a `textContent` node and as
        //: nothing at all in a message box — and worse, a rewrite of the sentence could drop it.
        //: `runs_a_stranger()` cannot be dropped by editing a string.
        let stranger = n.fix.as_ref().is_some_and(crate::fix::Fix::runs_a_stranger);
        format!("{{\"key\":\"{}\",\"what\":\"{}\",\"present\":{},\"required\":{},\
                 \"found\":\"{}\",\"how\":\"{}\",\"offer\":{offer},\"stranger\":{stranger},\
                 \"plan\":\"{}\"}}",
                escape(n.key), escape(n.what), n.present, n.required,
                escape(&n.found), escape(&n.how), escape(&plan))
    }).collect();
    format!("[{}]", rows.join(","))
}

/// The things that are missing, can be fixed, and were answered *always allow*.
///
/// **Read fresh every time**, like everything else about what is missing: a page open for an hour
/// holds an hour-old list, and a decision made five minutes ago should count.
fn standing_yes_and_missing(model: &str) -> Vec<String> {
    crate::needs::every(model).into_iter()
        .filter(|n| !n.present && n.fix.is_some())
        .map(|n| n.key.to_string())
        .filter(|k| crate::decided::standing(k) == Some(Standing::Allowed))
        .collect()
}

/// Exactly what saying yes will do, for the window that asks.
///
/// **The same discipline as [`crate::fix::describe`] and for the same reason.** A question reading
/// *"allow watching?"* is answered yes by everybody and tells them nothing; what they need is the
/// cost, in this case a model call every few seconds for as long as it is on.
fn plan_for(key: &str) -> String {
    match key {
        "notice" => String::from(
            "화면이 바뀌면 argmind가 먼저 말을 겁니다.\n\n\
             · 앞에 있는 창이 바뀔 때마다 알아챕니다\n\
             · 알아챌 때마다 모델을 한 번 부릅니다 — 화면은 14초 중 13초 바뀝니다\n\
             · 스스로 시작한 생각은 말로만 나옵니다. 클릭은 여전히 버튼을 거칩니다\n\
             · 끄기 전까지 계속 돌아갑니다\n\n\
             지켜볼까요?"),
        "microphone" => String::from(
            "말한 것을 받아쓰려면 마이크를 엽니다.\n\n\
             · 스페이스를 **누르고 있는 동안에만** 열립니다 — 켜 두는 것이 아닙니다\n\
             · 녹음한 소리는 이 컴퓨터의 받아쓰기 서버로만 갑니다. 밖으로 나가지 않습니다\n\
             · 받아쓴 글은 화면에 먼저 보여 주고 나서 답합니다\n\
             · 소리로 “응” 해도 행동은 실행되지 않습니다 — 버튼만이 실행합니다\n\n\
             마이크를 쓸까요?"),
        //: Everything else is a need, and the need already carries the sentence it asks with.
        _ => crate::needs::by_key("qwen3:8b", key).and_then(|n| n.fix)
            .map(|f| crate::fix::describe(&f))
            .unwrap_or_else(|| format!("{} 을(를) 허용할까요?", crate::decided::what(key))),
    }
}

/// One permission question, for the page to draw four buttons under.
fn asking_json(key: &str) -> String {
    let stranger = crate::needs::by_key("qwen3:8b", key).and_then(|n| n.fix)
        .is_some_and(|f| f.runs_a_stranger());
    format!("{{\"key\":\"{}\",\"what\":\"{}\",\"stranger\":{stranger},\"plan\":\"{}\"}}",
            escape(key), escape(crate::decided::what(key)), escape(&plan_for(key)))
}

/// Every decidable thing and what is decided about it, for the settings panel.
fn decisions_json() -> String {
    let rows: Vec<String> = crate::decided::all().iter().map(|(key, name, how)| {
        let word = match how {
            Some(Standing::Allowed) => "allowed",
            Some(Standing::Denied) => "denied",
            None => "",
        };
        format!("{{\"key\":\"{}\",\"what\":\"{}\",\"standing\":\"{word}\"}}",
                escape(key), escape(name))
    }).collect();
    format!("[{}]", rows.join(","))
}

/// One of the four buttons was pressed.
///
/// **The remembering and the doing are separate steps, in that order.** A standing *always* is
/// written down before the thing is attempted, so a build that fails does not also lose the
/// decision — and a person who pressed 항상 허용 and then closed the window has still answered.
fn decide(desk: &mut Desk, out: &Listeners, model: &str, key: &str, answer: Answer) {
    if let Some(standing) = answer.standing() {
        if let Err(why) = crate::decided::keep(key, Some(standing)) {
            out.send("note", &why);
        }
        out.send("decisions", &decisions_json());
    }
    let name = crate::decided::what(key);
    if !answer.permits() {
        out.send("note", &match answer {
            //: **Where to take it back is part of the sentence.** A permanent no with no way back
            //: is a program somebody has to reinstall.
            Answer::AlwaysDeny => format!("{name} — 거부했습니다. 설정에서 되돌릴 수 있습니다"),
            _ => format!("{name} — 이번에는 하지 않습니다"),
        });
        return;
    }
    carry_out(desk, out, model, key);
}

/// Do the thing a permission was for. **One place**, so an answer now and a standing answer from
/// last week take exactly the same path.
fn carry_out(desk: &mut Desk, out: &Listeners, model: &str, key: &str) {
    match key {
        //: The two switches. Nothing is installed; something starts running.
        "notice" => switch_noticing(desk, out, true),
        "microphone" => {
            out.send("note", "마이크를 쓸 수 있습니다 — 말하는 동안 스페이스를 누르고 계세요");
            out.send("mic-allowed", "1");
        }
        //: Everything else is a need with a fix, and `fix` re-reads what is missing before acting.
        _ => {
            fix(out, model, key);
            //: **The voice chosen at startup is out of date the moment this one succeeds.** Asked
            //: here rather than assumed from the result, because what matters is whether the
            //: server is answering and not whether a function returned `Ok`.
            if key == "speaker" {
                adopt_voice(desk, out);
            }
        }
    }
}

/// Watching, on or off — **lifted out of the job loop** so a standing permission reaches the same
/// code the button does.
fn switch_noticing(desk: &mut Desk, out: &Listeners, on: bool) {
    desk.noticing = on;
    //: **Cleared on the way in, not on the way out.** Switching on with an old `before` would
    //: compare against a screen from ten minutes ago and report the whole desktop as newly changed.
    desk.before.clear();
    out.send("noticing", if on { "1" } else { "0" });
    out.send("note", if on {
        "화면 변화를 지켜봅니다 — 창을 바꾸면 먼저 말을 겁니다"
    } else {
        "지켜보기를 껐습니다"
    });
}

/// Install one thing, because somebody pressed the button that said what it would do.
fn fix(out: &Listeners, model: &str, key: &str) {
    //: **Re-taken rather than remembered.** A page open for an hour holds an hour-old list, and
    //: acting on it would offer to install something that is already there.
    let Some(need) = crate::needs::by_key(model, key) else {
        out.send("note", &format!("{key} 라는 건 없습니다"));
        return;
    };
    if need.present {
        out.send("note", &format!("{} 은(는) 이미 있습니다", need.what));
        out.send("needs", &needs_json(model));
        return;
    }
    let Some(plan) = need.fix.as_ref() else {
        out.send("note", &format!("{} 은(는) argmind가 할 수 있는 게 없습니다", need.what));
        return;
    };
    out.send("busy", "1");
    out.send("note", &format!("{} — 시작합니다", need.what));
    let done = crate::fix::apply(plan, &mut |said| out.send("note", said));
    out.send("busy", "0");
    match done {
        Ok(said) => out.send("note", &format!("{} — {said}", need.what)),
        Err(why) => out.send("note", &format!("{} — 실패: {why}", need.what)),
    }
    //: Re-read afterwards, so the page shows what is true rather than what was asked for.
    out.send("needs", &needs_json(model));
}

/// Look once, and think if a watch has something to say.
///
/// **Nothing here is asked for**, so whatever the model decides comes back as a sentence rather
/// than as a click — `may_act_unprompted` is false and stays false.
fn notice(desk: &mut Desk, out: &Listeners, now: f64) {
    if !desk.noticing {
        return;
    }
    let after = desk.session.eyes.foreground();
    let fired = poll(&mut desk.watches, &desk.before, &after, now);
    desk.before = after;
    for trigger in fired {
        desk.noticed += 1;
        //: **Said before it is answered.** A thought that arrives with no visible cause reads as
        //: the agent talking to itself; naming what set it off is the difference between noticing
        //: and interrupting.
        out.send("noticed", &trigger.text);
        out.send("busy", "1");
        let intent = desk.session.decide(&trigger);
        out.send("busy", "0");
        desk.session.history.push(("noticed".into(), trigger.text.clone()));
        if intent.say.is_empty() {
            //: **Silence is the common answer and it is the right one.** Saying so once would be
            //: information; saying it every three seconds is the noise this whole floor exists to
            //: prevent, so nothing is sent.
            continue;
        }
        desk.session.history.push(("you".into(), intent.say.clone()));
        out.send("say", &intent.say);
        //: Not spoken aloud. A proposal waits behind a button; a sentence nobody asked for waits
        //: behind nothing, which is why speaking is tied to having been asked.
        for effect in &intent.effects {
            propose(desk, out, effect.clone());
        }
    }
}

/// The talk key, down or up. **Letting go is what starts a turn** — nothing is transcribed while
/// the key is still held, and nothing is recorded while it is not.
fn listen(desk: &mut Desk, out: &Listeners, down: bool) {
    //: **Asked before the device opens, and only on the way down.** A question raised while
    //: somebody is releasing the key would arrive after the moment it was about, and switching
    //: something off never needs permission.
    if down {
        match crate::decided::standing("microphone") {
            Some(Standing::Allowed) => {}
            Some(Standing::Denied) => {
                out.send("note", "마이크는 거부되어 있습니다 — 설정에서 바꿀 수 있습니다");
                return;
            }
            None => {
                out.send("asking", &asking_json("microphone"));
                return;
            }
        }
    }
    let Some((mic, _)) = desk.ear.as_mut() else {
        out.send("note", "마이크가 꺼져 있습니다 — --hear 로 켜세요");
        return;
    };
    mic.hold(down);
    if down {
        out.send("hearing", "1");
        return;
    }
    out.send("hearing", "0");
    let Some((mic, whisper)) = desk.ear.as_mut() else { return };
    let Some(clip) = mic.take() else {
        //: **Two different silences, and they had one message.** A tap on the key is the person
        //: holding it wrongly; a device that never opened is a machine with no microphone, and
        //: telling somebody to hold the key longer when there is nothing to hold is worse than
        //: saying nothing.
        out.send("note", if mic.available {
            "너무 짧습니다 — 말하는 동안 누르고 계세요"
        } else {
            "마이크를 열지 못했습니다 — 입력 장치가 있는지 확인하세요"
        });
        return;
    };
    out.send("busy", "1");
    let heard = whisper.hear(&clip.samples, platform::WANT_HZ);
    out.send("busy", "0");
    match heard {
        Err(why) => out.send("note", &format!("듣지 못했습니다 — {why}")),
        //: **A rejected transcript is said out loud, with the reason and with what it said.**
        //: Silence transcribed as words is 120 of 120 here, so this fires on ordinary room tone —
        //: and a turn that vanished with no explanation is the failure this project keeps finding.
        //: Showing what was thrown away is also the only way anybody notices the guard is wrong.
        Ok(None) => {
            let said = match whisper.last_refusal() {
                Some(("wordless", _)) => "소리만 들렸습니다 — 말로 들리지 않았습니다".to_string(),
                Some(("repeated", text)) => {
                    format!("같은 말이 반복돼 버렸습니다 — {:?}", text.chars().take(40)
                            .collect::<String>())
                }
                Some(("pace", _)) => "너무 빠릅니다 — 받아쓴 것 같지 않습니다".to_string(),
                _ => "아무 말도 들리지 않았습니다".to_string(),
            };
            out.send("note", &said);
        }
        Ok(Some(text)) => {
            //: **Shown before it is answered.** A recogniser's opinion about a waveform is not
            //: what the person said, and they have to be able to see what it thought it heard —
            //: which is also the only way a wrong transcript is ever noticed.
            out.send("them", &text);
            think(desk, out, text.trim());
        }
    }
}

/// Which voice speaks, decided once at startup.
///
/// **The neural one if it is already answering, and it is never started from here.** Starting it
/// means a build, a gigabyte of weights and somebody else's program running on this machine — a
/// decision that has a window in front of it, [`crate::fix::Fix::BuildSpeaker`]. What this does is
/// *use* what is there.
///
/// **The fallback is the point rather than a consolation.** Heami is the old voice and it is also
/// installed, instant and already what this program shipped with; not being able to be neural must
/// not turn into having gone silent.
/// **Public so the choice can be asserted**, for the same reason as [`SCRIPTS`]: the interesting
/// part is one branch, and running the application to reach it means binding a port somebody else
/// may be holding. It opens no audio device — [`argmind_env_windows::loudspeaker::Loudspeaker`]
/// waits until something is said.
pub fn pick_voice() -> (Option<Box<dyn Speaker>>, &'static str) {
    //: **Asked of the platform, not of the socket.** Whether there is a neural voice to prefer
    //: depends on whether this build can play one at all — the synthesiser is a server that runs
    //: anywhere, and the loudspeaker that would play it is WASAPI and runs in one place.
    if let Some(neural) = platform::neural_voice(crate::speaker::QWENTTS_HOST) {
        eprintln!("  voice: {} on {}", crate::speaker::VOICE, crate::speaker::QWENTTS_HOST);
        return (Some(neural), "neural");
    }
    match Mouth::new() {
        Ok(v) => {
            //: Said once, at startup, rather than discovered when a Korean reply comes out silent.
            if !v.has_korean() {
                eprintln!("  a Korean voice is not installed — Korean replies will not be spoken");
            } else if v.korean_is_old() {
                //: **Said once, at startup, and not again.** It is a fact about the machine rather
                //: than about anything argmind did, and a program that mentions it every turn is
                //: nagging about something it cannot fix.
                eprintln!("  the Korean voice is Heami Desktop, the old one — argmind can build a \
neural one instead (the 목소리 row in the page), and Windows 11 can add a newer Windows one under \
설정 → 접근성 → 내레이터 → 자연스러운 음성 추가");
            }
            (Some(Box::new(v)), "windows")
        }
        Err(e) => {
            eprintln!("  no voice: {e}");
            (None, "none")
        }
    }
}

/// Take up the neural voice now that there is one.
///
/// **Called after the fix, not instead of choosing at startup.** The alternative is asking on every
/// utterance whether a better voice has appeared, which is a 400 ms connect in the path of every
/// reply to answer a question whose answer changes once.
fn adopt_voice(desk: &mut Desk, out: &Listeners) {
    if !desk.wants_voice || desk.voice_is == "neural" {
        return;
    }
    let Some(neural) = platform::neural_voice(crate::speaker::QWENTTS_HOST) else { return };
    if let Some(old) = desk.voice.as_mut() {
        //: **Released rather than dropped.** SAPI holds a COM object; letting it go quietly while
        //: it is mid-sentence is how a voice gets talked over by its replacement.
        old.stop();
        old.release();
    }
    desk.voice = Some(neural);
    desk.voice_is = "neural";
    out.send("note", "이제 신경망 목소리로 말합니다");
}

fn hello(desk: &Desk, model: &str) -> String {
    let waiting: Vec<String> = desk.proposals.waiting().iter().map(as_json).collect();
    let fields = [
        format!("\"model\":\"{}\"", escape(model)),
        format!("\"dry\":{}", desk.dry),
        format!("\"speaks\":{}", desk.voice.is_some()),
        format!("\"voice\":\"{}\"", desk.voice_is),
        format!("\"hears\":{}", desk.ear.is_some()),
        format!("\"noticing\":{}", desk.noticing),
        format!("\"noticed\":{}", desk.noticed),
        format!("\"attending\":\"{}\"", escape(&desk.session.eyes.window().1)),
        format!("\"waiting\":[{}]", waiting.join(",")),
    ];
    //: Assembled from a list rather than one long format string. The first version wrapped that
    //: string across two lines and the indentation of the second landed **inside the JSON** —
    //: harmless, since whitespace between tokens is legal, and exactly the kind of thing that is
    //: harmless until the day something reads the payload byte for byte.
    format!("{{{}}}", fields.join(","))
}

/// How the application was asked to run.
///
/// **Every one of these is off unless asked for**, and the defaults are the safe direction: it
/// presses nothing, says nothing and opens no microphone until told to.
pub struct How {
    pub dry: bool,
    pub speak: bool,
    pub hear: bool,
    pub model_name: String,
    /// **Settable because a test has to be able to have its own.** It was a constant, and
    /// `tests/portable.rs` — the file whose whole job is to prove the server runs without an
    /// environment — would have passed by reading a page from a *different* argmind that happened
    /// to be running. It found nothing and reported success, which is the shape of measurement
    /// error this project keeps a list of.
    pub port: u16,
}

impl How {
    /// Read from the command line. `--dry`, `--speak`, `--hear`, and a model name with a colon.
    pub fn from_args() -> Self {
        How {
            dry: std::env::args().any(|a| a == "--dry"),
            //: **Off by default, and that is not a style choice.** Everything measured about this
            //: voice was measured at volume zero into a memory stream so a desktop somebody was
            //: using stayed quiet; making a sound is a thing to ask for.
            speak: std::env::args().any(|a| a == "--speak"),
            //: Same rule, and it matters more: **a microphone is not opened because a program was
            //: run.** Even with this, it opens only while the talk key is held.
            hear: std::env::args().any(|a| a == "--hear"),
            //: **Not from the command line.** There is one argmind and it lives on one port; a
            //: flag for it would be a second copy somebody starts by accident. A test sets it
            //: because a test must not read another argmind's page and call it a pass.
            port: PORT,
            model_name: std::env::args().skip(1).find(|a| a.contains(':'))
                .unwrap_or_else(|| "qwen3:8b".into()),
        }
    }

    /// The one line a person reads to know what it will and will not do.
    ///
    /// **All three, because each is its own way of looking broken.** *It did nothing*, *it said
    /// nothing* and *it never heard me* are three different complaints with three different
    /// settings behind them, and a line that named only two would leave the third a mystery.
    pub fn line(&self) -> String {
        format!("{} | {} | {} | {}", self.model_name,
                if self.dry { "dry — nothing will be pressed" }
                else { "live — it will press buttons" },
                if self.speak { "speaking aloud" } else { "silent" },
                if self.hear { "microphone on the talk key" } else { "not listening" })
    }
}

pub const URL: &str = "http://127.0.0.1:7331/";

/// The page's address for a given port.
pub fn url(port: u16) -> String {
    format!("http://127.0.0.1:{port}/")
}

/// Run the whole thing. **Blocks forever**, so a caller with its own message loop puts it on a
/// thread.
pub fn run(how: How) {

    let How { dry, speak, hear, model_name, port: how_port } = how;

    //: **Checked before the socket opens.** An agent that can press buttons on your screen has no
    //: business accepting a connection from the network.
    let at = address(how_port);
    assert!(is_loopback(&at), "{at} is not loopback");
    let token = mint_token();
    let listener = TcpListener::bind(&at).expect("could not bind");
    //: **Nothing is printed here.** One of the two binaries running this has no console at all, so
    //: a banner from the library goes either nowhere or into somebody else's window — and for the
    //: one that does have a console it was being printed twice. The console binary says it; the
    //: library serves.

    let out = Arc::new(Listeners::default());
    let (jobs, inbox) = channel::<Job>();

    //: **What it is looking at, live.** The window a person is in changes as they alt-tab, and an
    //: interface that only says so once a turn has happened is one you cannot trust before you
    //: speak. This reads a title and a handle — no UI Automation, so it costs nothing and can run
    //: as often as it likes.
    let watcher = Watching::default();
    watcher.watch();
    let watching = watcher.clone();
    let watching_out = Arc::clone(&out);
    std::thread::spawn(move || {
        let mut said = String::new();
        loop {
            std::thread::sleep(std::time::Duration::from_millis(600));
            let front = unsafe { platform::foreground_handle_now() };
            let now = match watching.attended(front) {
                Some(h) => platform::window_title(h),
                None => String::new(),
            };
            if now != said {
                said = now.clone();
                watching_out.send("attending", &said);
            }
        }
    });
    let agent_out = Arc::clone(&out);
    let agent_model = model_name.clone();
    //: Everything COM touches lives on this thread and never leaves it.
    std::thread::spawn(move || {
        let eyes = Sight::new(watcher).expect("the eyes would not open");
        //: Unrestricted, and **the confirm button is what makes that safe** — the same trade
        //: `app.py` makes by inverting the unprompted flag. Dry unless asked otherwise.
        let hands = if dry {
            Touch::unrestricted()
        } else {
            Touch::unrestricted().arm()
        };
        let mut model = Local::new(&agent_model);
        model.think = Some(false);
        let (voice, voice_is) = if speak { pick_voice() } else { (None, "off") };
        let ear = hear.then(|| (Ear::new(), Whisper::default()));
        let mut desk = Desk { session: Session::new(eyes, hands, Model(model)),
                              proposals: Proposals::default(), dry,
                              voice, voice_is, wants_voice: speak, ear,
                              //: One watch for now: the person moved to a different window, which
                              //: is the change that means *look at this*. `Appeared` and `Vanished`
                              //: need a name to look for and there is nowhere yet to type one.
                              watches: vec![Watch::new(Looking::WindowChanged)],
                              noticing: false, before: Vec::new(), noticed: 0 };
        let started = std::time::Instant::now();
        loop {
            //: **A timeout rather than a blocking receive**, so the same thread that answers can
            //: also look. One thread owns the session because COM says so, and a second one
            //: reading the screen would be a second apartment.
            let job = match inbox.recv_timeout(std::time::Duration::from_millis(700)) {
                Ok(job) => job,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    notice(&mut desk, &agent_out, started.elapsed().as_secs_f64());
                    continue;
                }
                Err(_) => break,
            };
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
                Job::Listen(down) => listen(&mut desk, &agent_out, down),
                Job::Notice(on) => match (on, crate::decided::standing("notice")) {
                    //: **Switching off never asks.** A permission question in front of somebody
                    //: trying to stop something is a program arguing with them.
                    (false, _) => switch_noticing(&mut desk, &agent_out, false),
                    (true, Some(Standing::Allowed)) => switch_noticing(&mut desk, &agent_out, true),
                    (true, Some(Standing::Denied)) => {
                        agent_out.send("noticing", "0");
                        agent_out.send("note",
                                       "지켜보기는 거부되어 있습니다 — 설정에서 바꿀 수 있습니다");
                    }
                    //: Nobody has answered, so ask — and the switch stays off until they do.
                    (true, None) => {
                        agent_out.send("noticing", "0");
                        agent_out.send("asking", &asking_json("notice"));
                    }
                },
                Job::Fix(key) => fix(&agent_out, &agent_model, &key),
                Job::Decide(key, answer) => decide(&mut desk, &agent_out, &agent_model, &key,
                                                   answer),
                Job::Forget(key) => {
                    match crate::decided::keep(&key, None) {
                        Ok(()) => agent_out.send(
                            "note", &format!("{} — 다음에 다시 물어봅니다",
                                             crate::decided::what(&key))),
                        Err(why) => agent_out.send("note", &why),
                    }
                    agent_out.send("decisions", &decisions_json());
                }
                Job::Look => {
                    agent_out.send("needs", &needs_json(&agent_model));
                    agent_out.send("decisions", &decisions_json());
                    //: **This is what 항상 허용 buys.** Anything still missing that was answered
                    //: once and for all is done now, without a second question — which is the
                    //: whole difference between the standing answer and the one-off.
                    for key in standing_yes_and_missing(&agent_model) {
                        agent_out.send("note",
                                       &format!("{} — 전에 항상 허용하셨으므로 진행합니다",
                                                crate::decided::what(&key)));
                        fix(&agent_out, &agent_model, &key);
                    }
                    //: **And take up the better voice if one of those just started it.** The
                    //: button path does this and this one did not, so a standing *항상 허용* on
                    //: the voice started the server and then went on talking in the old voice for
                    //: the rest of the session — the feature working and appearing not to.
                    adopt_voice(&mut desk, &agent_out);
                }
                Job::Opened(tx) => {
                    let _ = tx.send(sse("hello", &hello(&desk, &agent_model)));
                    //: **Checked when a page opens, not at startup.** Checking once at boot would
                    //: mean a server left running overnight still telling you about something you
                    //: installed an hour ago.
                    let _ = tx.send(sse("needs", &needs_json(&agent_model)));
                    //: **And what has already been answered**, so the settings panel is drawn
                    //: without anybody having to press 다시 확인 to find out what they agreed to.
                    let _ = tx.send(sse("decisions", &decisions_json()));
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
    let is_page = matches!(route(&request), Route::Ui)
        || SCRIPTS.iter().any(|(at, _)| *at == request.path);
    if !is_page && !authorised(token, request.token.as_deref()) {
        let _ = stream.write_all(&respond("403 Forbidden", "text/plain", "no token"));
        return;
    }

    if let Some((_, body)) = SCRIPTS.iter().find(|(at, _)| *at == request.path) {
        let _ = stream.write_all(&respond("200 OK", "text/javascript; charset=utf-8", body));
        return;
    }
    match route(&request) {
        Route::Ui => {
            let page = INDEX.replace("__TOKEN__", token);
            //: **The cookie rides on the page**, because `EventSource` cannot send a header and
            //: the event stream is the only route that tells the page anything.
            let _ = stream.write_all(&crate::respond_with(
                "200 OK", "text/html; charset=utf-8", &page,
                &crate::cookie_header(token)));
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
        Route::Fix(key) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Fix(key));
        }
        //: **Both halves are checked here**, before anything reaches the agent: a permission the
        //: program does not ask about and a word that is not one of the four are refused rather
        //: than passed on and interpreted later.
        Route::Decide(key, how) => match (crate::decided::is_decidable(&key),
                                          Answer::from_word(&how)) {
            (true, Some(answer)) => {
                let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
                let _ = jobs.send(Job::Decide(key, answer));
            }
            _ => {
                let _ = stream.write_all(&respond("400 Bad Request", "text/plain", "no"));
            }
        },
        Route::Forget(key) => {
            if crate::decided::is_decidable(&key) {
                let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
                let _ = jobs.send(Job::Forget(key));
            } else {
                let _ = stream.write_all(&respond("400 Bad Request", "text/plain", "no"));
            }
        }
        Route::Look => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Look);
        }
        Route::Notice(on) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Notice(on));
        }
        Route::Listen(down) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Listen(down));
        }
        Route::Unknown => {
            let _ = stream.write_all(&respond("404 Not Found", "text/plain", "no"));
        }
    }
}
