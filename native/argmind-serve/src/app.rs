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
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex};

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
    /// **Whether the camera should be on**, which the watching thread reads before every look, so
    /// switching off closes the device and puts the light out within about a frame.
    want_eyes: Arc<AtomicBool>,
    /// **Whether it actually is**, which is a different fact: a device that fails to open leaves
    /// the wish true and this false, and the page is told this one.
    eyeing: Arc<AtomicBool>,
    /// When movement was last seen, **shared with the thread that sees it**.
    ///
    /// The blackboard carries the same fact as a sentence for the prompt; this is the same fact as
    /// a time, for the one place that has to compare it with a number. Reading it back out of
    /// *마지막 움직임 12초 전* would be parsing a string this program wrote, which is the kind of
    /// round trip that works until somebody improves the wording.
    ///
    /// **Set when the camera opens**, not left empty: *nothing has moved since we started looking*
    /// is an answer that should age like any other, and an empty one would make a camera that had
    /// been on for two seconds indistinguishable from an empty room.
    moved_at: Arc<Mutex<Option<Instant>>>,
    /// **Everybody the camera has recognised**, shared with the camera thread. Behind a mutex
    /// rather than owned by that thread because the settings page renames and forgets people while
    /// it is running, and the roster is the one piece of state both of them write.
    people: Arc<Mutex<crate::people::People>>,
    /// Whether to look for faces at all. Read by the camera loop every few seconds, so the switch
    /// does not mean closing the device and opening it again.
    want_faces: Arc<AtomicBool>,
    /// How to speak to every open page, for the one thing here that outlives the call that started
    /// it. **Every other path is handed an `out` by its caller**; the camera thread has no caller
    /// left by the time it has something to say.
    speaks_to: Arc<Listeners>,
    /// When it last revised its own notes, on the same clock `notice` uses.
    thought_at: f64,
    /// Every sense's latest reading, and whether somebody is being answered. See [`crate::senses`].
    senses: crate::senses::Senses,
    turn: crate::senses::Turn,
    /// What the screen looked like last time this checked. **Empty means there is no before**, and
    /// the first look after switching on reports nothing whatever is on screen.
    before: Vec<argmind_core::contract::Widget>,
    /// The last few unprompted remarks, with each one's own subject removed. **The ring the
    /// repetition guard compares against** — see `argmind_core::converse::said_before`.
    lately_noticed: Vec<String>,
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
    /// Act without waiting for the button, or go back to asking.
    Acting(bool),
    /// Keep the camera on and watch for movement, or switch it off.
    Camera(bool),
    /// Remember the faces it sees, or stop.
    Faces(bool),
    /// Name one, forget one, forget everybody.
    NameFace(String, String),
    ForgetFace(String),
    ForgetFaces,
    /// An answer to a permission question: which thing, and which of the four buttons.
    Decide(String, Answer),
    /// Take back a standing answer, from the settings.
    Forget(String),
    /// Use this device from now on.
    Choose(String, String),
    /// Turn one thinking dial.
    Dial(String, u8),
    /// Speak out loud from now on, or stop.
    Aloud(bool),
    /// The thinking thread finished a note. **The only way a tick reaches the session**, so the
    /// session stays on one thread and nothing it holds is locked while a model call runs.
    Thought(argmind_core::thinking::Note),
    /// Write the thinking goal.
    Aim(String),
    /// Open one for a moment and report.
    Try(String),
    /// Remember which voice, and say something in it.
    Voice(String),
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

pub fn as_json(p: &Proposal) -> String {
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
    //: **Held for the whole turn, and it clears itself.** While this is alive the senses stand
    //: down: one declines to start and one already running is abandoned mid-call. A guard rather
    //: than two stores, because an early return that left the flag set would silence every sense
    //: for the rest of the session.
    let _answering = desk.turn.begins();
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
    //: **The words the person actually gets**, which are the model's own unless it mirrored them.
    let reply = spoken(&intent);
    if !reply.is_empty() {
        //: The substitute goes into the history too, and that is the half that matters more than
        //: the screen: leaving `머해?` in there got it said again on each of the next two turns.
        desk.session.history.push(("you".into(), reply.clone()));
        out.send("say", &reply);
        //: **Only because they asked.** A reply to something typed is an answer; nothing here
        //: fires unprompted yet, and when something does this is the line that must not simply
        //: inherit permission from it — a proposal waits behind a button, a sentence waits behind
        //: nothing.
        if let Some(voice) = desk.voice.as_mut() {
            //: Barge-in: a new answer replaces the last one rather than queueing behind ~3.1 s of
            //: speech.
            voice.stop();
            voice.say(&reply);
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
    let doubt = !hint.is_empty();
    //: **Whose idea it was**, read before the effect is handed over. `unprompted()` is `Some(false)`
    //: for anything a watch or a raised judgement produced, and it is the one thing the automatic
    //: path below is not allowed to carry out.
    let its_own_idea = effect.unprompted();
    let token = desk.proposals.offer(effect, text, hint);
    if let Some(p) = desk.proposals.waiting().last() {
        out.send("propose", &as_json(p));
    }
    //: **The offer is made either way, and then taken.**
    //:
    //: Going straight to `act` would be a second path to the only thing in this program that
    //: presses anything, and `Proposals` is what makes a stale token unable to confirm a later
    //: action. So an automatic run is the same two steps a person's button is, pressed here.
    //:
    //: **And the second opinion becomes a gate, only here.** With a person reading the row,
    //: *그 지점에는 link Cargo.lock* is information and the button is the decision; with nobody
    //: reading it, it is the last thing between a wrong resolution and a click. It stays a note
    //: when somebody is going to press the button themselves, because the hit test disagrees with
    //: a correct pixel about three times in ten and a gate would refuse those.
    //: **확인 없이 실행 covers what you asked for and not what argmind thought of.** The button
    //: is a permission a person gave to *this* action; the standing answer is a permission given
    //: once, in advance, to a kind of action — and the kind it was given to is *the thing I just
    //: asked for*. A turn argmind began itself waits, however the switch is set.
    if !doubt && !its_own_idea && crate::decided::standing("acting") == Some(Standing::Allowed) {
        if let Answered::Run(p) = desk.proposals.confirm(token) {
            act(desk, out, p);
        }
    }
}

/// **The only place in this program that performs anything.**
fn act(desk: &mut Desk, out: &Listeners, p: Proposal) {
    let (did, downgraded, refused) = desk.session.perform(std::slice::from_ref(&p.effect));
    desk.session.remember_doing(&did, &refused, &[], &[]);
    //: **The core refused because nobody asked for this**, which is a different outcome from
    //: *done* and from *gone*, and until 2026-09-22 it was reported as `ran`.
    //:
    //: **This cannot fire in this application any more and it is kept anyway.** `perform` downgrades
    //: an unprompted effect only while `may_act_unprompted` is off, and the application now turns it
    //: on — because it has a button, which is what `session.rs` says that flag is for. A caller that
    //: turned it back off would need this branch, and the alternative to keeping it is the silent
    //: `ran` that was here before: a lie nobody could see, because reaching it meant pressing 실행
    //: on something argmind had thought of by itself, which did nothing.
    if !downgraded.is_empty() {
        out.send("gone",
                 &format!("{{\"token\":{},\"what\":\"{}\"}}", p.token, escape(&p.effect.target)));
        out.send("note", "먼저 말을 건 생각이라 누르지 않고 말로만 두었습니다");
        return;
    }
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
    needs_as_json(&crate::needs::every(model), &crate::decided::remembered())
}

/// The same thing with the decisions handed in rather than read.
///
/// **Pure so it can be tested.** The rule it carries — *a thing answered once and for all is never
/// asked again* — is exactly the sort that rots quietly, and testing it against the real
/// `decisions.json` would mean a test that overwrites whatever the person actually decided.
pub fn needs_as_json(needs: &[crate::needs::Need],
                     decided: &std::collections::BTreeMap<String, Standing>) -> String {
    let rows: Vec<String> = needs.iter().map(|n| {
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
        //: **What was already answered, sent with the question.** Without this the page drew the
        //: four buttons for something decided once and for all, every single launch — the standing
        //: answer then resolved it a few seconds later and the question vanished, which is worse
        //: than either asking or not: it is asking and then ignoring the answer.
        let standing = match decided.get(n.key) {
            Some(Standing::Allowed) => "allowed",
            Some(Standing::Denied) => "denied",
            None => "",
        };
        format!("{{\"key\":\"{}\",\"what\":\"{}\",\"present\":{},\"required\":{},\
                 \"found\":\"{}\",\"how\":\"{}\",\"offer\":{offer},\"stranger\":{stranger},\
                 \"standing\":\"{standing}\",\"plan\":\"{}\"}}",
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
pub fn plan_for(key: &str) -> String {
    match key {
        "notice" => String::from(
            "화면이 바뀌면 argmind가 먼저 말을 겁니다.\n\n\
             · 앞에 있는 창이 바뀔 때마다 알아챕니다\n\
             · 알아챌 때마다 모델을 한 번 부릅니다 — 화면은 14초 중 13초 바뀝니다\n\
             · 스스로 시작한 생각은 말로만 나옵니다. 클릭은 여전히 버튼을 거칩니다\n\
             · 끄기 전까지 계속 돌아갑니다\n\n\
             지켜볼까요?"),
        //: **What it can and cannot tell, and that the light stays on.** The temptation here
        //: is to say *argmind can see you*, which is both alarming and false: nothing larger than
        //: 64x36 greys leaves the module and no picture is ever sent anywhere.
        "camera" => String::from(
            "카메라를 켜 두고 앞에 움직임이 있는지 봅니다.\n\n\
             · 켜 두는 동안 **카메라 불이 계속 켜져 있습니다** — 켜진 불이 곧 켜진 카메라입니다\n\
             · 화면은 64x36 회색으로 줄여서만 봅니다. 사진은 모듈 밖으로 나가지 않고 저장도, \
전송도 하지 않습니다\n\
             · 알 수 있는 것은 *마지막 움직임이 몇 초 전인가* 뿐입니다. 누구인지도, 무엇을 \
하는지도 아닙니다\n\
             · 가만히 앉아 책을 읽는 사람은 빈 의자와 구별되지 않습니다 — 잰 것은 움직임입니다\n\
             · 한 번 보는 데 32 ms, 3초마다 한 번 판단합니다\n\n\
             카메라를 켜 둘까요?"),
        //: **The cost is the model's accuracy, so the numbers are in the window.** A question
        //: reading *"act automatically?"* is answered yes by everybody; what a person needs is how
        //: often this model is right, which is measured and is not very often.
        "acting" => String::from(
            "제안을 버튼 없이 바로 실행합니다.\n\n\
             · 지금까지: 고를 수 있는 요청 28/37만 맞게 판단했습니다\n\
             · 대화만 하는 turn의 9%에서 하지 않아도 될 행동을 제안합니다\n\
             · 이름과 역할이 같은 버튼이 여럿이면 엉뚱한 것을 누릅니다 — 실제로 한 번 그랬습니다\n\
             · 먼저 말을 건 생각은 여전히 말로만 나옵니다. 이 설정은 사람이 시킨 것에만 적용됩니다\n\
             · 두 번째 의견이 다른 위젯을 가리키면 그때는 그래도 물어봅니다\n\n\
             버튼 없이 실행할까요?"),
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
pub fn asking_json(key: &str) -> String {
    let stranger = crate::needs::by_key("qwen3:8b", key).and_then(|n| n.fix)
        .is_some_and(|f| f.runs_a_stranger());
    format!("{{\"key\":\"{}\",\"what\":\"{}\",\"stranger\":{stranger},\"plan\":\"{}\"}}",
            escape(key), escape(crate::decided::what(key)), escape(&plan_for(key)))
}

/// Every decidable thing and what is decided about it, for the settings panel.
/// Every device of every kind, which one is chosen, and which the machine would pick anyway.
///
/// **Enumerating opens nothing.** It reads a device list and a property store, so this is as quiet
/// as asking what is missing — and unlike the tests below it, nothing here records, plays or looks.
fn devices_json() -> String {
    let kinds: Vec<String> = crate::chosen::KINDS.iter().map(|kind| {
        let picked = crate::chosen::chosen(kind);
        let (found, trouble) = match crate::platform::devices(kind) {
            Ok(list) => (list, String::new()),
            Err(why) => (Vec::new(), why),
        };
        let rows: Vec<String> = found.iter().map(|d| format!(
            "{{\"id\":\"{}\",\"name\":\"{}\",\"default\":{}}}",
            escape(&d.id), escape(&d.name), d.default)).collect();
        //: **`chosen` is reported even when it is not in the list.** A headset that is unplugged
        //: right now is still the one that was picked, and the page says so rather than silently
        //: showing the default as though nobody had ever chosen.
        format!("{{\"kind\":\"{}\",\"what\":\"{}\",\"chosen\":\"{}\",\"trouble\":\"{}\",\"found\":[{}]}}",
                escape(kind), escape(crate::chosen::what(kind)),
                escape(picked.as_deref().unwrap_or("")), escape(&trouble), rows.join(","))
    }).collect();
    format!("[{}]", kinds.join(","))
}

/// Which voice, and whether the number means anything right now.
///
/// **`matters` is the honest half.** A seed only chooses a speaker for the neural synthesiser; with
/// SAPI answering, or with nothing speaking at all, the row is a number that does nothing, and a
/// settings panel that offered it anyway would be lying about what the button does.
pub fn voice_json(voice_is: &str) -> String {
    let chosen = crate::chosen::voice();
    //: **The measurements go to the page with the names.** A person choosing a voice is entitled to
    //: see the number the ordering rests on — and to catch it being wrong, which is the part a
    //: label alone makes impossible.
    let rows: Vec<String> = crate::speaker::PRESETS.iter().map(|p| format!(
        "{{\"key\":\"{}\",\"what\":\"{}\",\"hz\":{},\"wobble\":{}}}",
        escape(p.key), escape(p.what), p.hz, p.wobble)).collect();
    format!("{{\"chosen\":\"{}\",\"is\":\"{}\",\"matters\":{},\"presets\":[{}]}}",
            escape(chosen.key), escape(voice_is), voice_is == "neural", rows.join(","))
}

/// One short sentence, said whenever the voice changes, so a number becomes something heard.
pub const A_SAMPLE: &str = "안녕하세요. 저는 이 목소리로 말합니다.";

/// The dials, what they are called, where they stand — **and the exact sentence the model will be
/// shown.**
///
/// That last field is the point. A settings panel that renders four sliders and says nothing about
/// what they do is asking a person to trust a number; showing the line means the interface cannot
/// quietly stop matching what is sent, because it is the same string. When nothing is set the line
/// is empty, and that is the truth as well: nothing is added to the prompt at all.
pub fn mind_json(mind: &argmind_core::mind::Mind) -> String {
    let rows: Vec<String> = argmind_core::mind::DIALS.iter().map(|(korean, gloss)| {
        format!("{{\"name\":\"{}\",\"gloss\":\"{}\",\"at\":{}}}",
                escape(korean), escape(gloss), mind.of(korean))
    }).collect();
    format!("{{\"most\":{},\"aim\":\"{}\",\"line\":\"{}\",\"dials\":[{}]}}",
            argmind_core::mind::MOST, escape(&mind.aim),
            escape(&mind.line().unwrap_or_default()), rows.join(","))
}

/// What to say when the model gave back the person's own words with a question mark on them.
///
/// **Here rather than in the core, because it is Korean.** `argmind-core` names no platform and no
/// language; it can tell that a mirror happened and cannot know how to admit it. The owner's
/// screenshot is the case: 머해 → 머해?, 왜 따라해 → 머해?, 왜 반복해 → 머해?.
///
/// **An honest sentence rather than a clever one.** The model mirrors when it cannot parse what was
/// said — `뭐해` is answered and `머해` is mirrored — so *I did not follow that* is not an evasion,
/// it is the true reading of what just happened, and it invites the one thing that fixes it.
pub const DID_NOT_FOLLOW: &str = "무슨 말인지 잘 못 알아들었어요. 다시 말해줄래요?";

/// The words a person gets: the model's own, unless it just mirrored them.
pub fn spoken(intent: &argmind_core::intent::Intent) -> String {
    if intent.parroted {
        return DID_NOT_FOLLOW.to_string();
    }
    intent.say.clone()
}

pub fn decisions_json() -> String {
    let rows: Vec<String> = crate::decided::all_with_defaults().iter()
        .map(|(key, name, how, by_default)| {
            let word = match how {
                Some(Standing::Allowed) => "allowed",
                Some(Standing::Denied) => "denied",
                None => "",
            };
            //: **`standing` is what was written and `default` is what happens when nothing was.**
            //: Collapsing them would put *항상 허용* beside a 잊어버려 button that removes nothing.
            format!("{{\"key\":\"{}\",\"what\":\"{}\",\"standing\":\"{word}\",\
                     \"default\":{by_default}}}",
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
        "camera" => switch_camera(desk, out, true),
        //: **Nothing opens and nothing downloads.** The camera decides whether there is a picture
        //: and `face-models` decides whether the files are here; this only says what may be done
        //: with what is already running.
        "faces" => switch_faces(desk, out, true),
        //: **Nothing starts and nothing installs.** The standing answer *is* the state, so all this
        //: does is tell the page the switch is on — which is why `Job::Acting` writes a 거부 when
        //: it is turned off rather than flipping a field: there is no field.
        "acting" => {
            out.send("acting", "1");
            out.send("note", "이제 확인 없이 실행합니다 — 설정에서 되돌릴 수 있습니다");
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
    //: **Switching on thinks at the next tick rather than thirty seconds later.** Nothing has been
    //: thought about this screen yet, so waiting out a timer that started before the switch is a
    //: panel that stays empty for no reason.
    desk.thought_at = 0.0;
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

/// Turn the camera on or off, and own the thread that keeps it open.
///
/// **The light is the interface.** The owner chose a camera that stays on over one that blinks
/// every half minute — opening costs 2.3–4.4 s of settling to take 3 s of readings, and a blinking
/// light says *mostly off*, which would not be true. So this starts one thread and it holds the
/// device until the wish goes false.
///
/// **What reaches the blackboard is an age, not a verdict.** `camera::MOVING` separates a person
/// moving from a still room by 6.8x, measured twice — but a person reading without shifting is
/// below it and so is an empty chair, so what can honestly be said is *movement was last seen N
/// seconds ago*. How long somebody sits still is not a number anybody here has, and inventing one
/// is how *waiting* became generic in the note.
fn switch_camera(desk: &mut Desk, out: &Listeners, on: bool) {
    if !on {
        desk.want_eyes.store(false, Ordering::SeqCst);
        out.send("camera", "0");
        out.send("note", "카메라를 껐습니다 — 불도 꺼집니다");
        return;
    }
    //: Already running: say so and start nothing. Two threads on one device is two lights' worth
    //: of nothing.
    if desk.eyeing.load(Ordering::SeqCst) {
        out.send("camera", "1");
        return;
    }
    desk.want_eyes.store(true, Ordering::SeqCst);
    let (want_eyes, eyeing) = (Arc::clone(&desk.want_eyes), Arc::clone(&desk.eyeing));
    let moved_at = Arc::clone(&desk.moved_at);
    let (people, want_faces) = (Arc::clone(&desk.people), Arc::clone(&desk.want_faces));
    let senses = desk.senses.clone();
    let out = Arc::clone(&desk.speaks_to);
    let chosen = crate::chosen::chosen("camera");
    std::thread::spawn(move || {
        eyeing.store(true, Ordering::SeqCst);
        out.send("camera", "1");
        out.send("note", "카메라를 켰습니다 — 끌 때까지 불이 켜져 있습니다");
        let carry_on = || want_eyes.load(Ordering::SeqCst);
        //: **Held across the whole run**, so *nobody has moved yet* and *nobody has moved for two
        //: minutes* are different sentences rather than both being silence.
        let began = Instant::now();
        *moved_at.lock().unwrap_or_else(|e| e.into_inner()) = Some(began);
        let mut last_moved: Option<Instant> = None;
        //: **Said once, not every three seconds.** A roster that is full is a standing fact and
        //: repeating it in the transcript would bury the conversation it is meant to support.
        let mut said_full = false;
        let outcome = crate::platform::watch_movement(
            chosen.as_deref(), &carry_on,
            &mut |moved, took| {
                if moved {
                    last_moved = Some(Instant::now());
                    *moved_at.lock().unwrap_or_else(|e| e.into_inner()) = last_moved;
                }
                let what = match &last_moved {
                    Some(at) => format!("마지막 움직임 {:.0}초 전", at.elapsed().as_secs_f64()),
                    None => format!("{:.0}초째 아무 움직임 없음", began.elapsed().as_secs_f64()),
                };
                senses.saw("카메라", what, took);
            },
            //: Three things have to be true: the models on disk, the switch on, and — implicitly —
            //: a camera, since this closure only runs inside its loop.
            &|| want_faces.load(Ordering::SeqCst) && crate::faces::here(),
            &mut |print| {
                let mut roster = people.lock().unwrap_or_else(|e| e.into_inner());
                match roster.saw(print, now_seconds()) {
                    crate::people::Saw::Fresh { .. } => {
                        //: **Written the moment somebody new arrives**, not at shutdown: a crash
                        //: between the two would lose exactly the row that was worth keeping.
                        if let Err(why) = roster.write() {
                            eprintln!("  faces: 명부를 쓰지 못했습니다 — {why}");
                        }
                        out.send("people", &people_json(&roster));
                        out.send("note", "처음 보는 얼굴을 기억했습니다 — \
설정에서 이름을 붙이거나 지울 수 있습니다");
                    }
                    //: **Nothing is sent for a face it already knows.** The counters move in
                    //: memory and are written when the camera closes; an event every three
                    //: seconds saying *still you* is noise on a channel the conversation shares.
                    crate::people::Saw::Known { .. } => {}
                    crate::people::Saw::Full => {
                        if !said_full {
                            said_full = true;
                            out.send("note", &format!(
                                "기억할 수 있는 얼굴 {}명이 다 찼습니다 — 설정에서 지우면 \
자리가 생깁니다", crate::people::MOST_REMEMBERED));
                        }
                    }
                    crate::people::Saw::Wrong => {}
                }
            });
        eyeing.store(false, Ordering::SeqCst);
        out.send("camera", "0");
        if let Err(why) = outcome {
            //: **The wish is cleared too.** A device that cannot be opened would otherwise leave
            //: the switch looking on with nothing behind it, and every page open would say so.
            want_eyes.store(false, Ordering::SeqCst);
            out.send("note", &format!("카메라를 계속 볼 수 없습니다 — {why}"));
        }
        //: Whatever it last saw is stale the moment the device closes, and a reading with an age
        //: that only grows is worse than none. **Both copies of the fact**, or the gate would go on
        //: answering from a camera that is off.
        senses.forget("카메라");
        //: **The counters, on the way out.** Every new person was written when they arrived; what
        //: is left is 마지막 본 때 and how many times, which are worth a write and not worth one
        //: every three seconds.
        {
            let roster = people.lock().unwrap_or_else(|e| e.into_inner());
            if !roster.who.is_empty() {
                let _ = roster.write();
            }
        }
        *moved_at.lock().unwrap_or_else(|e| e.into_inner()) = None;
    });
}

/// Remembering faces, on or off.
///
/// **It does not touch the camera.** Three things have to be true before a face is remembered — the
/// device is open, the models are on disk, and this is on — and they are three switches because
/// they are three decisions. Turning this on while the camera is off is allowed and does nothing
/// yet, which is the honest behaviour: the alternative is a switch that silently opens a device.
fn switch_faces(desk: &mut Desk, out: &Listeners, on: bool) {
    desk.want_faces.store(on, Ordering::SeqCst);
    out.send("faces", if on { "1" } else { "0" });
    if !on {
        //: **Switching off stops the writing, it does not erase what is written.** Forgetting is a
        //: different button, and conflating them would delete a roster somebody meant to keep.
        out.send("note", "얼굴을 더 기억하지 않습니다 — 이미 기억한 것은 그대로입니다");
        return;
    }
    out.send("note", if !crate::faces::here() {
        "얼굴 모델이 없습니다 — 설정의 필요한 것에서 받으세요"
    } else if !desk.eyeing.load(Ordering::SeqCst) {
        "카메라를 켜면 얼굴을 기억합니다"
    } else {
        "이제 본 얼굴을 기억합니다 — 사진은 저장하지 않습니다"
    });
}

/// Everybody argmind has seen, for the page. **No print**: the 128 numbers stay on this side, and
/// a page that never receives them cannot leak them into a log, a screenshot or a bug report.
fn people_json(people: &crate::people::People) -> String {
    let rows: Vec<String> = people.who.iter().map(|p| format!(
        "{{\"key\":\"{}\",\"name\":\"{}\",\"first\":{},\"last\":{},\"times\":{}}}",
        escape(&p.key), escape(&p.name), p.first, p.last, p.times)).collect();
    format!("[{}]", rows.join(","))
}

/// Seconds since the epoch, for a row that a person will read as a date.
fn now_seconds() -> i64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64).unwrap_or(0)
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
/// How long without movement before argmind stops speaking up by itself.
///
/// **Chosen, not measured** — the same footing as `RAISE_AFTER`. What *is* measured is the signal:
/// `camera::MOVING` separates a person moving from a still room by 6.8x, twice. What nobody has is
/// how long a present person can go without moving, and this is not that number; it is a length
/// chosen because being wrong at it is cheap in one direction and not the other.
///
/// **Too short and argmind goes quiet while you are sitting there** — a feature that looks broken,
/// recovered the moment you move. **Too long and it says something to an empty room**, which is
/// exactly what it does today with no camera at all. So it is biased long: five minutes is well
/// past any stillness a person at a keyboard plausibly holds, and the failure it leaves is the one
/// already being lived with.
pub const AWAY_AFTER: f64 = 300.0;

/// **Has anybody moved lately, as far as the camera knows?**
///
/// `None` whenever there is nothing to go on — the camera off, refused, just started, or failing —
/// and `None` never stops anything. A gate that fired on the absence of evidence would turn a
/// camera nobody switched on into a reason for silence.
fn somebody_about(desk: &Desk) -> Option<bool> {
    if !desk.eyeing.load(Ordering::SeqCst) {
        return None;
    }
    //: **The blackboard's own age is checked as well as the movement's.** The sense reports every
    //: three seconds; a thread that stalled would otherwise leave a true answer standing for ever,
    //: and *nobody is here* is the one this must not get stuck on.
    let fresh = desk.senses.of("카메라")
        .is_some_and(|r| r.taken.elapsed().as_secs_f64() < AWAY_AFTER);
    if !fresh {
        return None;
    }
    let at = desk.moved_at.lock().ok()?;
    Some(at.is_some_and(|t| t.elapsed().as_secs_f64() < AWAY_AFTER))
}

fn notice(desk: &mut Desk, out: &Listeners, now: f64) {
    if !desk.noticing {
        return;
    }
    //: **Nobody there, nothing to say.** The whole reason the camera earns its light: watching the
    //: screen is on a timer and will happily talk to an empty chair, and this is the one place that
    //: was true of. It returns before `poll`, so an empty room costs no model call at all rather
    //: than a call whose answer is thrown away.
    if somebody_about(desk) == Some(false) {
        //: The screen is still read on the next tick, so coming back to the machine does not need
        //: the watch to be reset — it needs a person to move.
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
        //: A watch cannot be parroted — nobody said anything to mirror — but it goes through the
        //: same call so there is one place that decides what gets said.
        let reply = spoken(&intent);
        if reply.is_empty() {
            //: **Silence is the common answer and it is the right one.** Saying so once would be
            //: information; saying it every three seconds is the noise this whole floor exists to
            //: prevent, so nothing is sent.
            continue;
        }
        //: **The fourth identical remark is not made.** Measured: over eight window changes it
        //: spoke 8 times out of 8 and 6 of those were one sentence with the name swapped. Telling
        //: it to be quiet in the prompt bought no silence and started it wanting to click things,
        //: so this is caught on the way out — the answer that worked for the parrot.
        if argmind_core::converse::said_before(&reply, &desk.lately_noticed) {
            continue;
        }
        desk.lately_noticed.push(reply.clone());
        //: **A short ring.** A person who comes back to a window ten minutes later should hear it
        //: again; the rule is against a run of them, not against ever saying it twice.
        if desk.lately_noticed.len() > REMARKS_REMEMBERED {
            desk.lately_noticed.remove(0);
        }
        desk.session.history.push(("you".into(), reply.clone()));
        out.send("say", &reply);
        //: Not spoken aloud. A proposal waits behind a button; a sentence nobody asked for waits
        //: behind nothing, which is why speaking is tied to having been asked.
        for effect in &intent.effects {
            propose(desk, out, effect.clone());
        }
    }
}

/// How often argmind revises its own notes, in seconds.
///
/// **Thirty, and it is a cost rather than a taste.** A tick is one model call — 0.61 s median — so
/// this is two calls a minute for as long as argmind is open, on top of whatever watching spends.
/// Three seconds would be four hundred an hour to watch somebody read a document.
///
/// It is also the floor on how stale a thought may be. A person who switches to their editor and
/// types for twenty seconds has not yet been noticed by this; the *watch* sees that in three, and
/// this is the slower loop underneath it.
pub const THINK_EVERY: f64 = 30.0;

/// How many unprompted remarks are remembered for the repetition guard.
///
/// **Six**, which at a floor of three seconds is about twenty seconds of alt-tabbing. Long enough
/// to cover the run of identical remarks that started this, short enough that coming back to a
/// window after a while hears about it again.
pub const REMARKS_REMEMBERED: usize = 6;

/// What the thinking thread is asked to chew on. **A snapshot, not a borrow** — the session stays
/// on its own thread and nothing shared has to be locked while a model call runs.
struct Chew {
    screen: String,
    /// The widgets the screen text was rendered from. **`worth` is a number into this list**, so
    /// the reply has to be resolved against the very same reading rather than a fresher one.
    listed: Vec<argmind_core::contract::Widget>,
    /// What is different since the last look, computed by this program. **The thing a judgement
    /// can be about** — see `argmind_core::thinking::what_changed`.
    changed: String,
    history: Vec<(String, String)>,
    note: argmind_core::thinking::Note,
}

/// A one-deep letterbox to the thinking thread.
///
/// **Dropped rather than queued.** A `Sender` would pile up thoughts about screens that have gone,
/// and the oldest of those is the least worth having. This holds at most one and the thinker takes
/// it when it is free; a tick arriving while one is waiting replaces it.
#[derive(Clone)]
struct Letterbox(Arc<Mutex<Option<Chew>>>, Arc<Condvar>);

impl Letterbox {
    fn new() -> Self {
        Letterbox(Arc::new(Mutex::new(None)), Arc::new(Condvar::new()))
    }

    /// Leave one, replacing whatever was there.
    fn try_send_chew(&self, chew: Chew) {
        if let Ok(mut slot) = self.0.lock() {
            *slot = Some(chew);
            self.1.notify_one();
        }
    }

    /// Wait for one. `None` only if the lock is poisoned, which means the program is going down.
    fn wait(&self) -> Option<Chew> {
        let mut slot = self.0.lock().ok()?;
        while slot.is_none() {
            slot = self.1.wait(slot).ok()?;
        }
        slot.take()
    }
}

/// Start one tick of argmind thinking about what it is looking at.
///
/// **It cannot speak and it cannot act**: the question it asks has no `say` and no `do` in its
/// answer shape. That is what makes a timer safe here — everything else in this file that runs
/// unasked had to be argued about, and this one cannot reach the person except through the panel
/// that shows the note.
///
/// **Tied to watching**, because they are the same permission in substance: both are argmind
/// spending model calls on a screen nobody pointed at. Switching 지켜보기 off stops both.
///
/// **And it no longer happens on this thread.** It used to, and a model call there is a model call
/// during which nothing answers anybody. The snapshot goes to the thinking thread, the note comes
/// back as a `Job`, and the queue below carries on. See [`crate::senses`] for the two rules that
/// keep that from costing the conversation.
fn reflect(desk: &mut Desk, thinking: &Letterbox, now: f64) {
    if !desk.noticing || now - desk.thought_at < THINK_EVERY {
        return;
    }
    //: **Not while somebody is being answered.** The first of the two rules: measured, a sense
    //: running beside the conversation takes the answer from 1.0 s to 3.2 s.
    if desk.turn.in_flight() {
        return;
    }
    desk.thought_at = now;
    let listed: Vec<argmind_core::contract::Widget> =
        desk.session.seen.iter().take(desk.session.limit).cloned().collect();
    //: **The screen it already has**, not a fresh read: this is the blackboard's own rule applied
    //: to itself, and `notice` took a reading moments ago on this same tick.
    let screen = argmind_core::converse::describe_screen(&listed, desk.session.limit);
    desk.senses.saw("화면", format!("{} 개가 보임", listed.len()), 0.0);
    //: **Dropped rather than queued when the thinker is busy.** A backlog of thoughts about screens
    //: that have gone is worse than no thought at all.
    //: `before` is what `notice` last read on this same tick, so the two readings being compared
    //: are a screen apart rather than a thought apart.
    let changed = argmind_core::thinking::what_changed(&desk.before, &listed);
    let _ = thinking.try_send_chew(Chew { screen, changed, listed,
                                          history: desk.session.recent(),
                                          note: desk.session.note.clone() });
}

/// **The one turn that starts because argmind decided the moment had come.**
///
/// The owner, 2026-09-17: 의지가 있어야 하지 않나. A `Watch` fires because a window changed — the
/// world moved and argmind reacted. This fires because a judgement has stood for `RAISE_AFTER`
/// looks and nobody has done anything about it, which is a reason that comes from argmind's own
/// state and from nowhere else.
///
/// **Everything that makes an unprompted turn safe still applies.** `asked = false`, so an effect
/// is downgraded to a proposal behind the confirm button, nothing is spoken aloud, and the
/// repetition guard stands in front of the sentence. What is new is only *why* the turn began.
///
/// **And it happens once.** The judgement is put down the moment it is raised, so a thing brought
/// up and ignored is not brought up again thirty seconds later — the defect this file has found in
/// three other shapes already.
fn raise_it(desk: &mut Desk, out: &Listeners) {
    if desk.turn.in_flight() {
        return;
    }
    let Some(worth) = desk.session.note.worth_raising().cloned() else { return };
    //: **Put down before the turn, not after.** A model call takes a second or two and the tick
    //: that follows must not find it still standing and raise it twice.
    desk.session.note.let_go();
    out.send("note-kept", &note_json(&desk.session.note));

    let trigger = Trigger::perceived(&argmind_core::thinking::raising(&worth));
    out.send("noticed", &trigger.text);
    let _answering = desk.turn.begins();
    out.send("busy", "1");
    let intent = desk.session.decide(&trigger);
    out.send("busy", "0");
    desk.session.history.push(("noticed".into(), trigger.text.clone()));

    let reply = spoken(&intent);
    if reply.is_empty() || argmind_core::converse::said_before(&reply, &desk.lately_noticed) {
        return;
    }
    desk.lately_noticed.push(reply.clone());
    if desk.lately_noticed.len() > REMARKS_REMEMBERED {
        desk.lately_noticed.remove(0);
    }
    desk.session.history.push(("you".into(), reply.clone()));
    out.send("say", &reply);
    //: Not spoken aloud, and nothing here presses anything — **not because of the button**,
    //: which 확인 없이 실행 can take away, but because `perform` downgrades an effect that nobody
    //: asked for. A turn argmind started itself is the last place to relax either of those, and
    //: after 2026-09-22 only one of the two is still a button.
    for effect in &intent.effects {
        propose(desk, out, effect.clone());
    }
}

/// What it is holding in mind, for the panel that shows it.
///
/// **Shown, because a thought nobody can read is indistinguishable from no thought.** It is also
/// the only way anybody notices the note has gone wrong — a belief that is false stays in every
/// prompt until something replaces it.
pub fn note_json(note: &argmind_core::thinking::Note) -> String {
    //: **The sentence around the judgement is the program's.** The model chose a widget; the
    //: wording is not its to write, which is what leaves no room for *도움을 줄 수 있다*.
    let worth = note.worth.as_ref()
        .map(|w| format!("{} \"{}\"", w.role, w.name))
        .unwrap_or_default();
    //: **A real newline landed inside a key name here** — `\"` where a line-continuation `\` was
    //: meant — and the page's `JSON.parse` threw on every note for two commits, so the panel was
    //: simply dead. Nothing caught it because nothing parsed what this function produces; a test
    //: now does, which is the only reason to write a JSON string by hand at all.
    format!("{{\"believes\":\"{}\",\"worth\":\"{}\",\"tried\":\"{}\",\
             \"held\":{},\"stuck\":{},\"revisions\":{}}}",
            escape(&note.believes), escape(&worth), escape(&note.tried),
            note.held, note.stuck, note.revisions)
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
        let chosen = crate::chosen::voice();
        eprintln!("  voice: {} / {} ({} Hz) on {}", crate::speaker::MODEL, chosen.key, chosen.hz,
                  crate::speaker::QWENTTS_HOST);
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
    //: **And the header, which is otherwise a session-long lie.** `hello` is sent when a page
    //: opens and never again, so a voice swapped after that left the status line saying
    //: `소리 켜짐 (Windows)` directly above a transcript line saying it had changed.
    out.send("voice", desk.voice_is);
    //: **And the seed row, whose `matters` is false for SAPI and true for this.** The engine and
    //: *whether the voice number does anything* are one fact, so they are sent together.
    out.send("voice-seed", &voice_json(desk.voice_is));
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
        //: Read rather than held: the standing answer is the state.
        format!("\"acts\":{}",
                crate::decided::standing("acting") == Some(Standing::Allowed)),
        //: **Held rather than read**, unlike `acts`: a standing 허용 says the camera *may* be on
        //: and this says it *is*, and they come apart when the device fails to open.
        format!("\"watching_you\":{}", desk.eyeing.load(Ordering::SeqCst)),
        format!("\"noticed\":{}", desk.noticed),
        format!("\"attending\":\"{}\"", escape(&desk.session.eyes.window().1)),
        format!("\"waiting\":[{}]", waiting.join(",")),
        format!("\"remembering\":{}", desk.want_faces.load(Ordering::SeqCst)),
        format!("\"face_models\":{}", crate::faces::here()),
        format!("\"people\":{}",
                people_json(&desk.people.lock().unwrap_or_else(|e| e.into_inner()))),
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

/// Does argmind make a sound?
///
/// **The settings decide and the flag fills the gap**, the same arrangement as [`hearing`] and for
/// the same reason: an answer written down is an answer, and a program that reads it and then asks
/// for a command-line flag anyway has made the setting decorative.
///
/// **Off unless something says otherwise**, which is unchanged. Everything measured about this
/// voice was measured at volume zero into a memory stream so that a desktop somebody was using
/// stayed quiet; making a noise in a room is a thing to be asked for, not assumed.
pub fn aloud(flag: bool) -> bool {
    aloud_from(flag, crate::decided::standing("aloud"))
}

/// The same with the answer handed in. **Pure so it can be tested** without overwriting what the
/// person actually decided.
pub fn aloud_from(flag: bool, standing: Option<Standing>) -> bool {
    match standing {
        Some(Standing::Allowed) => true,
        Some(Standing::Denied) => false,
        None => flag,
    }
}

/// Does argmind listen on the talk key?
///
/// **The standing answer wins in both directions**, and a refusal wins over the flag on purpose: a
/// person who answered 항상 거부 in the settings has said something more deliberate than whatever
/// is on the shortcut they clicked, and a launcher that could override it would make 거부 mean
/// *until the next time somebody edits a shortcut*.
pub fn hearing(flag: bool) -> bool {
    hearing_from(flag, crate::decided::standing("microphone"))
}

/// The same with the answer handed in rather than read. **Pure so it can be tested** — testing it
/// against the real `decisions.json` would mean a test that overwrites what the person decided in
/// order to check that deciding works, which is the reason `needs_as_json` is split the same way.
pub fn hearing_from(flag: bool, standing: Option<Standing>) -> bool {
    match standing {
        Some(Standing::Allowed) => true,
        Some(Standing::Denied) => false,
        None => flag,
    }
}

impl How {
    /// Read from the command line. `--dry`, `--speak`, `--hear`, and a model name with a colon.
    pub fn from_args() -> Self {
        How {
            dry: std::env::args().any(|a| a == "--dry"),
            //: **Off by default, and that is not a style choice.** Everything measured about this
            //: voice was measured at volume zero into a memory stream so a desktop somebody was
            //: using stayed quiet; making a sound is a thing to ask for. **Asked for in the
            //: settings since 2026-09-17**, with the flag left as the way to say yes when there is
            //: no standing answer to read.
            speak: aloud(std::env::args().any(|a| a == "--speak")),
            //: Same rule, and it matters more: **a microphone is not opened because a program was
            //: run.** Even with this, it opens only while the talk key is held.
            //:
            //: **But a standing 항상 허용 is an answer, not a note.** The owner had answered
            //: `microphone: allowed` in the settings and argmind still would not listen without the
            //: flag, which makes the setting decorative — the same complaint they raised about
            //: being asked again every launch, in a second place. The settings decide and the flag
            //: is how you say yes when there is no standing answer to read.
            hear: hearing(std::env::args().any(|a| a == "--hear")),
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
    //: **The blackboard and the one bit that keeps it honest**, shared by every thread that senses
    //: and the one that answers. See `crate::senses`.
    let senses = crate::senses::Senses::default();
    let turn = crate::senses::Turn::default();

    //: **Thinking happens here, not on the thread that answers.** It used to be inline, and a model
    //: call there is a model call during which nobody gets a reply. What comes back is a `Job`.
    let letters = Letterbox::new();
    {
        let (letters, turn, senses, jobs, model_name) =
            (letters.clone(), turn.clone(), senses.clone(), jobs.clone(), model_name.clone());
        std::thread::spawn(move || {
            let mut model = Local::new(&model_name);
            model.think = Some(false);
            model.keep_alive = "300s".into();
            while let Some(chew) = letters.wait() {
                //: Checked again here: a turn can have begun between the tick and this thread
                //: waking, and starting anyway would be the rule holding only when it is easy.
                if turn.in_flight() {
                    continue;
                }
                let others = senses.render("화면").unwrap_or_default();
                let screen = if others.is_empty() {
                    chew.screen
                } else {
                    format!("{}\n\n{others}", chew.screen)
                };
                let prompt = argmind_core::thinking::ask_for_a_note(
                    &chew.note, &screen, &chew.changed, &chew.history,
                    argmind_core::thinking::MOST_CHARS);
                //: **Abandoned the moment somebody is answered.** Measured: dropping the socket
                //: frees the card — an answer after abandoning costs 1.09 s against 1.12 s with a
                //: free card and 3.22 s with a busy one.
                let carry_on = turn.carry_on();
                let Ok(reply) = model.ask_while(&prompt, &[], &carry_on, &mut |_| {}) else {
                    continue;
                };
                if reply.reason == "abandoned" {
                    continue;
                }
                let mut note = chew.note;
                //: The same list the model was shown — `worth` is a number into it.
                if argmind_core::thinking::read_note(&reply.text, &mut note, &chew.listed, &chew.changed) {
                    let _ = jobs.send(Job::Thought(note));
                }
            }
        });
    }

    let agent_out = Arc::clone(&out);
    let agent_model = model_name.clone();
    //: Everything COM touches lives on this thread and never leaves it.
    std::thread::spawn(move || {
        let eyes = Sight::new(watcher).expect("the eyes would not open");
        //: Unrestricted, and **the confirm button is what makes that safe** — the same trade
        //: `app.py` makes by inverting the unprompted flag. Dry unless asked otherwise, and how
        //: they are built is `platform`'s to decide: on Windows that now means control patterns
        //: rather than synthetic clicks, which is a fact about Windows and not about the app.
        let hands = platform::hands(dry);
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
                              noticing: false, thought_at: 0.0,
                              want_eyes: Arc::new(AtomicBool::new(false)),
                              eyeing: Arc::new(AtomicBool::new(false)),
                              moved_at: Arc::new(Mutex::new(None)),
                              //: **Read off disk at startup**, so a roster survives a restart —
                              //: which is the whole of what 기억 means here.
                              people: Arc::new(Mutex::new(crate::people::People::read())),
                              want_faces: Arc::new(AtomicBool::new(
                                  crate::decided::standing("faces") == Some(Standing::Allowed))),
                              speaks_to: Arc::clone(&agent_out),
                              senses: senses.clone(), turn: turn.clone(),
                              before: Vec::new(), lately_noticed: Vec::new(), noticed: 0 };
        //: **The button is the permission, which is what this flag is for** — `session.rs`:
        //: *an interface with a confirm button turns this on and gates with the button instead*.
        //: This application has the button and did not turn it on, so `perform` refused every
        //: effect a watch or a raised judgement produced and the 실행 beside it did nothing. It was
        //: drawn, it was pressed, and until 2026-09-22 it did not even say so.
        //:
        //: **It is not a relaxation, because the automatic path refuses what this allows.** 확인
        //: 없이 실행 carries out what somebody asked for; a turn argmind began itself still waits
        //: for a person, however that switch is set. See `propose`.
        desk.session.may_act_unprompted = true;
        //: **On from the start when it is allowed**, which it is by default since 2026-09-17. A
        //: permission that says yes and a program that waits for a button anyway is the defect
        //: this session has already found twice in other places.
        if crate::decided::standing("notice") == Some(Standing::Allowed) {
            switch_noticing(&mut desk, &agent_out, true);
        }
        //: **Read once, here.** Every turn renders from `desk.session.mind`, so the file is not
        //: touched on the hot path — and a mind that was never set is a resting one, which renders
        //: the prompt this program shipped with byte for byte.
        desk.session.mind = crate::mind::remembered();
        //: Whether the standing answers have been acted on yet this run. See `Job::Opened`.
        let mut caught_up = false;
        let started = std::time::Instant::now();
        loop {
            //: **A timeout rather than a blocking receive**, so the same thread that answers can
            //: also look. One thread owns the session because COM says so, and a second one
            //: reading the screen would be a second apartment.
            let job = match inbox.recv_timeout(std::time::Duration::from_millis(700)) {
                Ok(job) => job,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    let now = started.elapsed().as_secs_f64();
                    notice(&mut desk, &agent_out, now);
                    //: **After noticing, not instead of it.** A window change is what a person is
                    //: owed an answer about; revising a note is what happens in the gaps.
                    reflect(&mut desk, &letters, now);
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
                Job::Camera(on) => match (on, crate::decided::standing("camera")) {
                    //: **Switching off never asks**, the same as 지켜보기: a permission question
                    //: in front of somebody turning a camera off is a program arguing with them.
                    (false, _) => switch_camera(&mut desk, &agent_out, false),
                    (true, Some(Standing::Allowed)) => switch_camera(&mut desk, &agent_out, true),
                    (true, Some(Standing::Denied)) => {
                        agent_out.send("camera", "0");
                        agent_out.send("note", "카메라는 거부되어 있습니다 — 설정에서 바꿀 수 있습니다");
                    }
                    (true, None) => {
                        agent_out.send("camera", "0");
                        agent_out.send("asking", &asking_json("camera"));
                    }
                },
                Job::Acting(on) => match (on, crate::decided::standing("acting")) {
                    //: **Off is written down, not just obeyed.** Unlike 지켜보기 there is no
                    //: runtime flag behind this — the standing answer is the whole of it — so a
                    //: switch turned off that wrote nothing would be on again at the next launch.
                    (false, _) => {
                        if let Err(why) = crate::decided::keep("acting", Some(Standing::Denied)) {
                            agent_out.send("note", &why);
                        }
                        agent_out.send("acting", "0");
                        agent_out.send("note", "다시 확인을 받고 실행합니다");
                        agent_out.send("decisions", &decisions_json());
                    }
                    (true, Some(Standing::Allowed)) => agent_out.send("acting", "1"),
                    //: **A previous 거부 is re-asked here rather than refused**, which is the one
                    //: place this differs from 지켜보기. That one is a standing cost; this is a
                    //: thing somebody turns on for an hour of clicking and off again afterwards,
                    //: and sending them to the settings panel to undo their own last answer would
                    //: be a switch that only works once.
                    (true, _) => {
                        agent_out.send("acting", "0");
                        agent_out.send("asking", &asking_json("acting"));
                    }
                },
                Job::Fix(key) => fix(&agent_out, &agent_model, &key),
                Job::Decide(key, answer) => decide(&mut desk, &agent_out, &agent_model, &key,
                                                   answer),
                Job::Choose(kind, id) => {
                    let pick = (!id.is_empty()).then_some(id.as_str());
                    match crate::chosen::keep(&kind, pick) {
                        Ok(()) => {
                            agent_out.send("note", &format!(
                                "{} — {}", crate::chosen::what(&kind),
                                match pick {
                                    Some(_) => "골랐습니다. 다음 사용부터 적용됩니다",
                                    None => "기본 장치로 되돌렸습니다",
                                }));
                            //: **Next use, not now.** The microphone and the speaker are opened
                            //: when they are wanted, and a device swapped underneath a sentence
                            //: that is being spoken would cut it off — which is a worse surprise
                            //: than waiting for the next one.
                        }
                        Err(why) => agent_out.send("note", &why),
                    }
                    agent_out.send("devices", &devices_json());
                }
                Job::Try(kind) => {
                    agent_out.send("busy", "1");
                    let tried = crate::platform::try_device(
                        &kind, crate::chosen::chosen(&kind).as_deref());
                    agent_out.send("busy", "0");
                    agent_out.send("tried", &format!(
                        "{{\"kind\":\"{}\",\"worked\":{},\"said\":\"{}\"}}",
                        escape(&kind), tried.worked, escape(&tried.said)));
                }
                //: **Rebuilt rather than re-seeded.** `QwenVoice` is handed its seed at
                //: construction and the one place that builds one reads the setting, so changing
                //: the voice is the same act as adopting one — stop, release, build. A setter
                //: would be a second place that decides which voice, which is the thing the
                //: previous version of this got wrong by having none at all.
                Job::Voice(key) => {
                    if let Err(why) = crate::chosen::keep_voice(&key) {
                        agent_out.send("note", &why);
                    }
                    match desk.voice_is {
                        "neural" => {
                            if let Some(fresh) =
                                crate::platform::neural_voice(crate::speaker::QWENTTS_HOST) {
                                if let Some(old) = desk.voice.as_mut() {
                                    old.stop();
                                    old.release();
                                }
                                desk.voice = Some(fresh);
                            }
                            //: The sample is the point of the press, so it is said even though
                            //: nobody asked a question — this is the one utterance in the program
                            //: that is a demonstration rather than an answer.
                            if let Some(voice) = desk.voice.as_mut() {
                                voice.say(A_SAMPLE);
                            }
                        }
                        "windows" => agent_out.send(
                            "note", "지금은 Windows 목소리로 말하고 있어서 이 번호는 쓰이지 않습니다"),
                        _ => agent_out.send(
                            "note", "소리 내어 말하기가 꺼져 있습니다 — 켜면 이 목소리로 말합니다"),
                    }
                    agent_out.send("voice-seed", &voice_json(desk.voice_is));
                }
                Job::Dial(name, to) => {
                    match crate::mind::turn(&name, to) {
                        Ok(mind) => {
                            agent_out.send("note", &format!("{name} {to}"));
                            desk.session.mind = mind;
                        }
                        Err(why) => agent_out.send("note", &why),
                    }
                    agent_out.send("mind", &mind_json(&desk.session.mind));
                }
                Job::Aim(text) => {
                    match crate::mind::aim(&text) {
                        Ok(mind) => {
                            agent_out.send("note", if mind.aim.is_empty() {
                                "생각 목표를 지웠습니다"
                            } else {
                                "생각 목표를 저장했습니다"
                            });
                            desk.session.mind = mind;
                        }
                        Err(why) => agent_out.send("note", &why),
                    }
                    agent_out.send("mind", &mind_json(&desk.session.mind));
                }
                Job::Aloud(on) => {
                    let how = if on { Standing::Allowed } else { Standing::Denied };
                    if let Err(why) = crate::decided::keep("aloud", Some(how)) {
                        agent_out.send("note", &why);
                    }
                    desk.wants_voice = on;
                    if on {
                        //: **Picked now, not at the next launch.** The whole complaint behind this
                        //: was a setting that was recorded and then not honoured until a restart.
                        let (voice, is) = pick_voice();
                        desk.voice = voice;
                        desk.voice_is = is;
                    } else if let Some(old) = desk.voice.as_mut() {
                        //: **Released rather than dropped**, and stopped first: SAPI holds a COM
                        //: object and letting it go mid-sentence is how a voice talks over itself.
                        old.stop();
                        old.release();
                        desk.voice = None;
                        desk.voice_is = "off";
                    }
                    agent_out.send("note", if desk.voice.is_some() {
                        "이제 소리 내어 말합니다"
                    } else if on {
                        "소리를 낼 수 없습니다 — 목소리가 없습니다"
                    } else {
                        "이제 소리를 내지 않습니다"
                    });
                    agent_out.send("voice", desk.voice_is);
                    agent_out.send("decisions", &decisions_json());
                }
                Job::Thought(note) => {
                    //: **A tick reaches the person through the panel, and one other way.** See
                    //: `raise_it` — the only turn in this program that begins because argmind has
                    //: been holding something rather than because the world moved.
                    desk.session.note = note;
                    agent_out.send("note-kept", &note_json(&desk.session.note));
                    raise_it(&mut desk, &agent_out);
                }
                Job::Forget(key) => {
                    match crate::decided::keep(&key, None) {
                        //: **What it goes back to, which is not always a question.** A key with a
                        //: default returns to the default; saying *다음에 다시 물어봅니다* there
                        //: would promise a step that no longer happens.
                        Ok(()) => agent_out.send("note", &format!(
                            "{} — {}", crate::decided::what(&key),
                            if crate::decided::ALLOWED_UNLESS_REFUSED.contains(&key.as_str()) {
                                "기본값(허용)으로 돌아갑니다"
                            } else {
                                "다음에 다시 물어봅니다"
                            })),
                        Err(why) => agent_out.send("note", &why),
                    }
                    //: And a default that is allowed means it starts again now, rather than at the
                    //: next launch — the same rule the rest of this file learned the hard way.
                    if crate::decided::standing(&key) == Some(Standing::Allowed) && key == "notice" {
                        switch_noticing(&mut desk, &agent_out, true);
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
                //: **Gated exactly like the camera**, and by its own key. A standing 허용 turns
                //: it on at the first page open; anything else asks.
                Job::Faces(on) => match (on, crate::decided::standing("faces")) {
                    (false, _) => switch_faces(&mut desk, &agent_out, false),
                    (true, Some(Standing::Allowed)) => switch_faces(&mut desk, &agent_out, true),
                    (true, Some(Standing::Denied)) => agent_out.send("note",
                        "거부해 두셨습니다 — 설정에서 되돌릴 수 있습니다"),
                    (true, None) => {
                        agent_out.send("faces", "0");
                        agent_out.send("asking", &asking_json("faces"));
                    }
                },
                Job::NameFace(key, name) => {
                    let mut roster = desk.people.lock().unwrap_or_else(|e| e.into_inner());
                    if roster.call(&key, &name) {
                        let _ = roster.write();
                        agent_out.send("people", &people_json(&roster));
                    }
                }
                Job::ForgetFace(key) => {
                    let mut roster = desk.people.lock().unwrap_or_else(|e| e.into_inner());
                    if roster.forget(&key) {
                        let _ = roster.write();
                        agent_out.send("people", &people_json(&roster));
                        agent_out.send("note", "한 사람을 잊었습니다");
                    }
                }
                Job::ForgetFaces => {
                    let mut roster = desk.people.lock().unwrap_or_else(|e| e.into_inner());
                    match roster.forget_everybody() {
                        //: **The file goes too.** Somebody pressing this means *there is nothing
                        //: of mine here*, and an empty file is a worse answer than no file.
                        Ok(()) => agent_out.send("note", "기억한 얼굴을 모두 지웠습니다 — \
파일도 지웠습니다"),
                        Err(why) => agent_out.send("note",
                            &format!("파일을 지우지 못했습니다 — {why}")),
                    }
                    agent_out.send("people", &people_json(&roster));
                }
                Job::Opened(tx) => {
                    let _ = tx.send(sse("hello", &hello(&desk, &agent_model)));
                    //: **What 항상 허용 buys, without anybody pressing 다시 확인.**
                    //:
                    //: It only happened on `Job::Look`, which is the button — so after every
                    //: restart the page drew *전에 항상 허용하셨습니다 — 물어보지 않고 진행합니다*
                    //: beside a recogniser that was not running and never would be. The standing
                    //: answer was being reported and not honoured, which is worse than either.
                    //:
                    //: **On the first page rather than at startup**, because a fix can be a
                    //: three-minute build and its progress has to go somewhere a person can see;
                    //: at startup there is no page connected and the notes would go nowhere. Once
                    //: per process, so a second window cannot start a second build.
                    if !caught_up {
                        caught_up = true;
                        for key in standing_yes_and_missing(&agent_model) {
                            agent_out.send("note",
                                           &format!("{} — 전에 항상 허용하셨으므로 진행합니다",
                                                    crate::decided::what(&key)));
                            fix(&agent_out, &agent_model, &key);
                        }
                        adopt_voice(&mut desk, &agent_out);
                        //: **A standing 허용 turns the camera on without being asked again**, the
                        //: same arrangement the microphone and the installs already have. Two
                        //: places used to report a standing answer and not honour it, and the page
                        //: then said *물어보지 않고 진행합니다* beside something that was not
                        //: running; this is the third and it is wired the same way rather than
                        //: differently.
                        if crate::decided::standing("camera") == Some(Standing::Allowed) {
                            switch_camera(&mut desk, &agent_out, true);
                        }
                    }
                    //: **Checked when a page opens, not at startup.** Checking once at boot would
                    //: mean a server left running overnight still telling you about something you
                    //: installed an hour ago.
                    let _ = tx.send(sse("needs", &needs_json(&agent_model)));
                    //: **And what has already been answered**, so the settings panel is drawn
                    //: without anybody having to press 다시 확인 to find out what they agreed to.
                    let _ = tx.send(sse("decisions", &decisions_json()));
                    //: **Listed when a page opens rather than at startup.** Enumerating reads a
                    //: property store and opens nothing, but a headset plugged in since boot
                    //: should appear without restarting the program.
                    let _ = tx.send(sse("devices", &devices_json()));
                    //: **What it is trying to be**, so the sliders and the goal are drawn where
                    //: they were left rather than at rest.
                    let _ = tx.send(sse("mind", &mind_json(&desk.session.mind)));
                    //: **After `adopt_voice` above**, so `matters` is drawn against the voice that
                    //: is actually speaking rather than the one this process started with.
                    let _ = tx.send(sse("voice-seed", &voice_json(desk.voice_is)));
                    //: **And what it currently thinks**, so a page opened halfway through a session
                    //: shows the thought rather than an empty panel that fills in thirty seconds.
                    let _ = tx.send(sse("note-kept", &note_json(&desk.session.note)));
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
        //: **The kind is checked here and not only in the job.** A route that reaches the agent
        //: thread with a name nobody recognises is a bad request, and answering 400 says so at the
        //: moment it can still mean something.
        Route::Choose(kind, id) => {
            if crate::chosen::is_a_kind(&kind) {
                let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
                let _ = jobs.send(Job::Choose(kind, id));
            } else {
                let _ = stream.write_all(&respond("400 Bad Request", "text/plain", "no"));
            }
        }
        Route::Try(kind) => {
            if crate::chosen::is_a_kind(&kind) {
                let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
                let _ = jobs.send(Job::Try(kind));
            } else {
                let _ = stream.write_all(&respond("400 Bad Request", "text/plain", "no"));
            }
        }
        //: Checked here as well as in the job, like `/choose` and for the same reason.
        Route::Dial(name, to) => {
            if crate::mind::is_a_dial(&name) && to <= argmind_core::mind::MOST {
                let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
                let _ = jobs.send(Job::Dial(name, to));
            } else {
                let _ = stream.write_all(&respond("400 Bad Request", "text/plain", "no"));
            }
        }
        //: Checked against the fixed list here, like `/dial`, so a page that asked for a voice
        //: nobody has is told at the moment it can still mean something.
        Route::Voice(key) => {
            if crate::speaker::preset(&key).is_some() {
                let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
                let _ = jobs.send(Job::Voice(key));
            } else {
                let _ = stream.write_all(&respond("400 Bad Request", "text/plain", "no"));
            }
        }
        Route::Aim(text) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Aim(text));
        }
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
        Route::Aloud(on) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Aloud(on));
        }
        Route::Notice(on) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Notice(on));
        }
        Route::Acting(on) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Acting(on));
        }
        Route::Camera(on) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Camera(on));
        }
        Route::Faces(on) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Faces(on));
        }
        //: **Length-checked here rather than at the far end**, because a page is allowed to send a
        //: long name by accident and the roster should not be where that is discovered. `call`
        //: trims to 40 characters too; this refuses a request that is obviously not a name.
        Route::NameFace(key, name) => {
            if name.chars().count() <= 200 {
                let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
                let _ = jobs.send(Job::NameFace(key, name));
            } else {
                let _ = stream.write_all(&respond("400 Bad Request", "text/plain", "no"));
            }
        }
        Route::ForgetFace(key) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::ForgetFace(key));
        }
        Route::ForgetFaces => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::ForgetFaces);
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
