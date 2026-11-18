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
/// The page's words in four languages, and what puts them on screen — `crate::interface`.
const WORDS: &str = include_str!("../../../ui/dist/words.js");
const SPEAK: &str = include_str!("../../../ui/dist/speak.js");

/// The files served without a token, because the token arrives inside the page.
///
/// **Public so the list can be asserted.** It is the program's whole unauthenticated surface, and
/// a route added to it by an edit somewhere else is the kind of change nobody reviews twice.
pub const SCRIPTS: [(&str, &str); 4] =
    [("/app.js", SCRIPT), ("/orb.js", ORB), ("/speak.js", SPEAK), ("/words.js", WORDS)];
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
    fn ask_scored(&mut self, prompt: &str) -> (String, Vec<(String, f64)>) {
        self.0.ask_scored(prompt).map(|(r, t)| (r.text, t)).unwrap_or_else(|e| {
            eprintln!("  the model did not answer: {e}");
            (String::new(), Vec::new())
        })
    }
}

/// Everything the page is told, as it happens — **and the timeline of it**, kept in the one place
/// every event passes through. See [`crate::activity`].
#[derive(Default)]
struct Listeners(Mutex<Vec<Sender<String>>>, Mutex<crate::activity::Activity>);

impl Listeners {
    fn send(&self, event: &str, data: &str) {
        if let Some((kind, text)) = crate::activity::reading(event, data) {
            self.record(kind, &text);
        }
        self.frame(event, data);
    }

    fn frame(&self, event: &str, data: &str) {
        let frame = sse(event, data);
        let Ok(mut who) = self.0.lock() else { return };
        //: A page that closed is a channel that errors, and dropping it here is the only cleanup
        //: there is — nothing tells a server that a page went away.
        who.retain(|tx| tx.send(frame.clone()).is_ok());
    }

    /// **One line on the timeline, and the page told of it.** Sent through `frame`, never `send`,
    /// so the line about a line is not itself a line.
    fn record(&self, kind: &'static str, text: &str) {
        let line = {
            let Ok(mut log) = self.1.lock() else { return };
            log.push(crate::activity::now(), kind, text).map(crate::activity::entry_json)
        };
        if let Some(line) = line {
            self.frame("activity-line", &line);
        }
    }

    fn timeline(&self) -> String {
        self.1.lock().map(|log| log.json()).unwrap_or_else(|_| "[]".into())
    }

    fn mark(&self) -> u64 {
        self.1.lock().map(|log| log.total()).unwrap_or(0)
    }

    /// **What came of a press, in the timeline's own words** — the lines `act` and `propose` already
    /// write (*실행 — 화면이 바뀜*, *못 함 — 그새 사라짐*, a rule's note), so a task is told what a
    /// person would read and nothing is written twice.
    fn pressed_since(&self, mark: u64) -> String {
        let Ok(log) = self.1.lock() else { return String::new() };
        let lines: Vec<String> = log.since(mark).into_iter()
            .filter(|e| ["제안", "실행", "못 함", "알림", "물음"].contains(&e.kind))
            .map(|e| format!("{}: {}", e.kind, e.text)).collect();
        if lines.is_empty() { "아무 일도 일어나지 않았다".into() } else { lines.join("\n") }
    }

    fn forget_timeline(&self) {
        if let Ok(mut log) = self.1.lock() {
            log.clear();
        }
    }
}

struct Desk {
    session: Session<crate::surface::Looking, crate::surface::Touching, Model>,
    /// **Whether argmind is looking at its browser pane**, and the pane itself when there is one.
    on_web: crate::surface::OnWeb,
    web: Option<crate::surface::WebPane>,
    proposals: Proposals,
    /// **The last look's picture with its point drawn on it**, until `propose` files it beside the
    /// row it belongs to — `crate::looking::keep_shown`. `None` between looks.
    looked: Option<Vec<u8>>,
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
    /// **The one thing argmind does not know and has asked about.** Shared with the camera
    /// thread, which is what notices a gap; the job thread is what hears the answer.
    wonders: Arc<Mutex<crate::wonders::Wonders>>,
    /// Whether to work out where the head is pointed. Read by the camera loop every second, so
    /// the switch does not mean closing the device and opening it again.
    want_attention: Arc<AtomicBool>,
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
    /// **눈치**, 0–10. **Never shown to the model** — see `tact.rs` for the measurement that says
    /// why it must not be. 0 is today's behaviour exactly.
    nunchi: u8,
    /// One remark argmind decided not to say yet, and the window it was about. **One, not a
    /// queue**: the newest replaces it, because a tactful person does not arrive with a backlog.
    held: Option<(argmind_core::intent::Intent, String, Instant)>,
    /// When argmind last spoke unprompted, for the gap 눈치 keeps between remarks.
    last_remark: Option<Instant>,
    /// **맡긴 일**, read off disk at startup and written on every change. See [`crate::errands`].
    errands: Vec<argmind_core::errand::Errand>,
    /// When an errand was last stepped, on the clock `notice` uses.
    stepped_at: f64,
    /// The last transcript sent as `them`, so the turn it starts does not put it on the timeline
    /// twice. Taken by the turn.
    heard_last: Option<String>,
    /// How many lines of the history are already in the conversation file.
    kept_lines: usize,
    /// **A task under way** — one at a time — and the bit that stops it between steps.
    working: Arc<AtomicBool>,
    stop_work: Arc<AtomicBool>,
    /// **Whether the task under way is argmind's own** — work toward the goal, `crate::pursuit`.
    own_work: Arc<AtomicBool>,
    /// A person's request that stopped argmind's own work, started when that work has stopped.
    after_own: Option<String>,
    /// 멈춰 said to argmind's own work: the next piece waits [`OWN_BACKOFF`].
    own_halted: bool,
    /// Pieces of argmind's own work nobody has been told about yet.
    own_untold: Vec<argmind_core::pursuit::Pursued>,
    /// **The language the person last wrote in** — what the program's own sentences are said in
    /// when no language is set. See [`speaking`].
    heard_in: String,
    /// When the next piece may be picked, and how many in a row did not finish.
    own_not_before: Instant,
    own_misses: u32,
    /// A pick is being asked of the model.
    own_choosing: bool,
    /// Why the last pick was not taken, for the next one to hear.
    own_refused: Option<String>,
    /// The person was seen away since the last report — `somebody_about` read `Some(false)`.
    own_away: bool,
    /// Where a finished task reports back, so its result enters the history on this thread.
    jobs: Sender<Job>,
    /// **A task's question waiting for the person**: the next thing they say is its answer, unless it
    /// is *멈춰*. Empty when nothing is asked.
    task_asks: Arc<Mutex<Option<Sender<String>>>>,
    /// **A task's press waiting on a row**: its token, where the answer goes, and the timeline mark
    /// from before the press.
    task_presses: Vec<(u64, Sender<String>, u64)>,
    /// Work-screen presses waiting on the person, by `About::Press`'s `ask`.
    screen_asks: Vec<(u64, Sender<Result<(), String>>)>,
    /// Judgements already raised this session, by name and role — see `raise_it`.
    raised: Vec<(String, String)>,
    /// **규칙**, read at startup and replaced whole from the page. See [`argmind_core::rule`].
    rules: Vec<argmind_core::rule::Rule>,
    /// **What the camera last said about where the head points, and when.** Written by the camera
    /// thread each window, read by `notice`. The age matters: a reading from a camera that stalled
    /// a minute ago is no evidence, and `tact` treats no evidence as *say it*.
    facing_now: Arc<Mutex<Option<(crate::platform::Facing, Instant)>>>,
    /// **When a face was last in view.** Presence evidence apart from movement — see `present`.
    face_seen_at: Arc<Mutex<Option<Instant>>>,
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
    /// Work out where the head is pointed, or stop.
    Attention(bool),
    /// Turn 눈치.
    Nunchi(u8),
    /// Remember the faces it sees, or stop.
    Faces(bool),
    /// Answer the one open question, or skip it with an empty string.
    Answer(u64, String),
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
    /// Answer in this language from now on — a code, or `""` for the language they used.
    Language(String),
    /// A new errand, an answer to one, or one taken off the list.
    Errand(String),
    ErrandAnswer(u64, String),
    ErrandDrop(u64),
    /// The rules, as the page sent them.
    Rules(String),
    Initiative(bool),
    Looking(bool),
    Web(bool),
    WebGo(String),
    Keep(String),
    /// A task ended, or asks something, with what to tell the person — said and kept in the history.
    Worked(Worked),
    /// **What a task made, to show in the pane** — `(path, text)` a file, found by
    /// `crate::deliver` and read through the workbench. Sent just before its `Worked`.
    Made(Vec<(String, String)>),
    Unkeep(usize),
    ForgetConversation,
    Activity,
    ClearActivity,
    /// **A task looks at the pane** — `(role, name)` of what can be pressed.
    PaneLook(Sender<Result<Vec<(String, String)>, String>>),
    /// **A task presses in the pane**, through `propose` and `act` like a turn. The answer is what
    /// came of it, in the timeline's words — at once, or when the person answers the row.
    PanePress(Effect, Sender<String>),
    /// A task's press or write on the work screen, for the person's 규칙 and 확인 없이 실행:
    /// window, kind, what, where the answer goes, and whether the work is argmind's own — which
    /// is asked of 스스로 실행 instead.
    ScreenGate(String, String, String, Sender<Result<(), String>>, bool),
    /// **argmind picked its next piece of work toward the goal** — or could not, and why.
    OwnPicked(Result<String, String>),
    /// A piece of argmind's own work ended, as the program recorded it.
    OwnEnded(argmind_core::pursuit::Pursued),
    /// The thinking thread took one step: **the copy it stepped from and the result**, so a step
    /// computed before an answer arrived is recognised and thrown away rather than written over it.
    Stepped(argmind_core::errand::Errand, argmind_core::errand::Errand),
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
    format!("{{\"token\":{},\"label\":\"{}\",\"hint\":\"{}\",\"picture\":{}}}",
            p.token, escape(&p.label), escape(&p.hint), crate::looking::has_shown(p.token))
}

/// What the person reads on the button.
fn label(effect: &Effect, found: Option<&Widget>) -> String {
    match effect.kind.as_str() {
        //: **Where, and what the picture showed** — never the point, which the row does not need.
        "click" if effect.inside.is_some() => format!("클릭  \u{201c}{}\u{201d} 안 — 그림에서 본 \u{201c}{}\u{201d}",
                                                     found.map_or(effect.target.as_str(), |w| w.name.as_str()),
                                                     effect.text),
        "click" => match found {
            Some(w) => format!("클릭  {} \u{201c}{}\u{201d}", w.role, w.name),
            None => format!("클릭  \u{201c}{}\u{201d}", effect.target),
        },
        "type" => format!("입력  \u{201c}{}\u{201d}", effect.text),
        _ => format!("키  {}", effect.text),
    }
}

/// **What a task tells the person**, sent to the worker as what happened rather than as a Korean
/// sentence, so it is said in the language the person is spoken to in: the program's words from
/// [`crate::spoken`], and the work model's own words as it wrote them — [`REWRITES_WORK`] says why
/// they are not rewritten.
pub enum Worked {
    /// The task asks the person this.
    Asks(String),
    /// Done, in the work model's closing words.
    Done(String),
    /// Stopped before the end, and why.
    NotFinished(String),
    /// Ended with no reason given.
    Stopped,
}

/// **Whether what the work model wrote is rewritten into the person's language** before it is
/// said — a task's closing words and question, an errand's result, the request argmind set itself.
/// **Off, measured** (2026-11-17, `--bin tongues --reports`, the bar written first): twelve Korean
/// texts said into English, Chinese and Japanese came back in the language **7, 8 and 10 of 12**
/// against 11, and of what was taken **six of 36 changed what was found** against a bar of one —
/// *이스타항공* became *Etihad Airways* and *济州航空*, and *369,700원* became *369,700元* and
/// *369,700円*, a price in the wrong currency with every digit kept, which the facts check cannot
/// see. So the program's words are said in the language and the work model's stay as it wrote
/// them, Korean inside an English sentence, which is at least not wrong.
const REWRITES_WORK: bool = false;

/// What the work model wrote, as it is said — see [`REWRITES_WORK`].
fn work_words(session: &mut Session<crate::surface::Looking, crate::surface::Touching, Model>,
              text: &str, code: &str) -> String {
    if REWRITES_WORK { session.say_in(text, code) } else { text.to_string() }
}

/// **The language the program's own sentences are said in**: the one set in 설정 → 생각, or else
/// the one the person last wrote in — Korean until they write in another. The reply's own rule,
/// *the language they used*, for what the model does not write.
fn speaking(desk: &Desk) -> String {
    let set = &desk.session.mind.language;
    if argmind_core::mind::is_a_language(set) { set.clone() } else { desk.heard_in.clone() }
}

/// One turn: ask, say, and turn every effect into a row that waits.
/// **A request to remember, answered by the program.** No model call: what to keep is the
/// person's sentence, and asking a model to decide what argmind should remember is asking it to
/// write its own instructions. The reply is fixed words, and both halves go into the history so the
/// next turn can see what was kept.
fn kept_instead(desk: &mut Desk, out: &Listeners, text: &str) -> bool {
    use crate::spoken::{say, KEEP_FULL, KEEP_WHAT, KEPT};
    let Some(asked) = argmind_core::mind::asked_to_keep(text) else { return false };
    let code = speaking(desk);
    let reply = match asked {
        argmind_core::mind::Keep::This(said) => match crate::mind::keep(&said) {
            Ok(mind) => {
                desk.session.mind = mind;
                out.send("mind", &mind_json(&desk.session.mind));
                say(&KEPT, &code, &[&said])
            }
            Err(why) if why.starts_with("기억한 것이") =>
                say(&KEEP_FULL, &code, &[&argmind_core::mind::MOST_KEPT.to_string()]),
            Err(why) => why,
        },
        argmind_core::mind::Keep::Vague => say(&KEEP_WHAT, &code, &[]),
    };
    desk.session.history.push(("them".into(), text.to_string()));
    desk.session.history.push(("you".into(), reply.clone()));
    out.send("say", &reply);
    true
}

fn think(desk: &mut Desk, out: &Listeners, text: &str) {
    //: **What the person said, on the timeline.** Not sent — the page drew it when it was typed —
    //: and a spoken turn arrives here already sent as `them`, so this records only typed ones.
    if !desk.heard_last.take().is_some_and(|h| h == text) {
        out.record("들음", text);
    }
    //: **A sentence nobody can read is asked again, with no model call and nothing done** — not
    //: routed, not answered by a turn, not handed to a task waiting on an answer, and not kept in the
    //: history, where it would read as something said. `converse::unreadable` has the case.
    if argmind_core::converse::unreadable(text) {
        out.record("길", "읽을 수 없음 — 글자가 깨져 들어옴");
        out.send("say", &crate::spoken::say(&crate::spoken::UNREADABLE, &speaking(desk), &[]));
        return;
    }
    //: **The language they wrote in**, for what the program says back — after the check above, so
    //: a sentence that arrived as `?? 40mm ??` is not read as English.
    if let Some(code) = argmind_core::tongue::of(text) {
        desk.heard_in = code.to_string();
    }
    //: **Held for the whole turn, and it clears itself.** While this is alive the senses stand
    //: down: one declines to start and one already running is abandoned mid-call. A guard rather
    //: than two stores, because an early return that left the flag set would silence every sense
    //: for the rest of the session. Taken before routing, since routing may ask the model and a
    //: task's call should yield the card to it.
    let _answering = desk.turn.begins();
    //: **A task asked the person something, and this is the answer** — before any routing, since an
    //: answer like *박서연, 21일* reads as nothing the rules know. *멈춰* is not an answer: dropping
    //: the question lets the task go on to the stop below.
    let waiting = desk.task_asks.lock().map(|mut w| w.take()).unwrap_or(None);
    if let Some(answer_to) = waiting {
        if !argmind_core::tongue::stops(text) && answer_to.send(text.to_string()).is_ok() {
            out.record("길", "작업의 물음에 대한 답");
            let reply = crate::spoken::say(&crate::spoken::CARRYING_ON, &speaking(desk), &[]);
            desk.session.history.push(("them".into(), text.to_string()));
            desk.session.history.push(("you".into(), reply.clone()));
            out.send("say", &reply);
            return;
        }
    }
    //: **One decision, in `crate::route`**, which `--bin routing` measures — the same function, so
    //: a miss there is a miss here. The rules first, and where they say *a turn* the model is asked
    //: whether it is work, shown the window in front: 0.15 s median, 0.23 s worst, measured. Each
    //: path still checks what it needs before it acts.
    let (_, front) = desk.session.eyes.window();
    let (route, asked) = crate::route::routed(text, desk.web.is_some(), desk.on_web.get(),
                                              desk.working.load(Ordering::SeqCst), &front,
                                              &mut desk.session.model);
    match asked {
        Some((said, s)) => out.record("길", &format!("{} — 모델 {said}, {s:.2}초", route.name())),
        None => out.record("길", route.name()),
    }
    //: **A bare go-ahead is a yes to what was offered, not a request of its own.** With a row
    //: waiting it is the turn's to accept, as it always was; sent to work, the task is given the
    //: request the last reply asked about — the owner's *실행해줘* once started a task whose whole
    //: request was *실행해줘* — and with nothing asked about, no task starts.
    let mut route = route;
    let mut request = text.to_string();
    if argmind_core::converse::goes_ahead(text) {
        if !desk.proposals.waiting().is_empty() {
            route = crate::route::Route::Turn;
        } else if route == crate::route::Route::Work {
            match argmind_core::converse::offer_answered(&desk.session.history) {
                Some(asked) => {
                    out.record("길", &format!("work — 「{asked}」에 대한 승낙"));
                    request = asked;
                }
                None => route = crate::route::Route::Turn,
            }
        }
    }
    let handled = match route {
        crate::route::Route::Keep => kept_instead(desk, out, text),
        crate::route::Route::Work => worked_instead(desk, out, &request, text),
        crate::route::Route::LookUp => looked_up_instead(desk, out, text),
        //: On a turn too: a desktop app named with the pane up takes the pane down and returns
        //: false, so the turn carries on at the desktop.
        crate::route::Route::Web | crate::route::Route::Turn => moved_web_instead(desk, out, text),
    };
    if handled {
        return;
    }
    out.send("busy", "1");
    //: **The last turn's buttons go.** An offer the person never answered and then said something
    //: else about is one they moved on from, and a button whose label describes a screen that has
    //: gone is worse than no button.
    for dropped in desk.proposals.clear() {
        out.send("dropped", &dropped.token.to_string());
        settle_press(desk, out, dropped.token, "실행 줄이 대화에 밀려 사라졌다 — 누르지 않았다");
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

    //: **Where the list was blind, a picture — only there, and only with a press.** The owner's
    //: choice of 2026-11-18; `looking.rs` has the measurements and the gate. Asked before anything
    //: is said, so the person hears one answer rather than a refusal followed by a find.
    let mut intent = intent;
    let mut from_the_picture: Option<String> = None;
    if !desk.on_web.get() && crate::decided::standing("looking") == Some(Standing::Allowed)
        && crate::looking::wants_a_look(text, &intent.effects, &widgets) {
        let chose_a_place = intent.effects.iter().any(|e| e.kind == "click");
        match look_inside(desk, out, text, &intent.effects, &widgets) {
            Some((effect, said)) => {
                intent.effects = vec![effect];
                from_the_picture = Some(said);
            }
            //: **The turn chose the middle of a place the list cannot see into**, and the picture
            //: found nothing there: the middle of a remote desktop is not what was asked for.
            None if chose_a_place => {
                intent.effects.clear();
                from_the_picture = Some(crate::spoken::say(&crate::spoken::LOOKED_NOTHING, &speaking(desk), &[]));
            }
            None => {}
        }
    }
    desk.session.history.push(("them".into(), text.to_string()));
    //: **The words the person actually gets**, which are the model's own unless it mirrored them —
    //: or the program's, when the picture decided what is offered.
    let reply = from_the_picture.unwrap_or_else(|| spoken_in(&intent, &speaking(desk)));
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
    //: **One step at a time, and the rest stopped by a step that did not go ahead** — the owner,
    //: 2026-10-21: 매번 화면상태를 확인하고 해야함. A click, *서울* and Enter had run one after
    //: another into a login page. Each step is checked against the screen in `perform`; this is
    //: the other half — what follows a failed step is not offered, because it was planned against a
    //: screen that step was supposed to produce.
    //: **Either reading of it**: a rewrite into the language set (`Intent::rewritten_from`) must
    //: not turn *열어드릴까요?* into a sentence this rule cannot read as an offer.
    let asks = argmind_core::converse::asks_first(&reply)
        || intent.rewritten_from.as_deref().is_some_and(argmind_core::converse::asks_first);
    let mut rest = intent.effects.iter();
    while let Some(effect) = rest.next() {
        if !propose(desk, out, effect.clone(), intent.sure, asks) {
            let left: Vec<String> = rest.map(|e| label(e, None)).collect();
            if !left.is_empty() {
                desk.session.history.push(("could not".into(), format!(
                    "{} — an earlier step did not go ahead, so these were not tried; nothing was done",
                    left.join(", "))));
                out.send("note", &format!("앞 단계가 되지 않아 나머지는 하지 않았습니다: {}",
                                          left.join(" · ")));
            }
            break;
        }
    }
    out.send("busy", "0");
}

/// **The one place in this program that shows a model a picture of the screen** — `looking.rs`
/// says when and holds what comes back. The window argmind is attending to, or the one place in it
/// the list cannot see into, made [`crate::looking::MOST_WIDE`] wide; the vision model's box; and
/// the point kept only where `landed` puts it inside a region. Nothing is saved; the picture with
/// the point drawn on it waits in `desk.looked` for the row.
fn look_inside(desk: &mut Desk, out: &Listeners, text: &str, effects: &[Effect], seen: &[Widget])
               -> Option<(Effect, String)> {
    use crate::looking::{asking, effect, landed, on_screen, place_to_show, read, Landed, Saw, MOST_WIDE,
                         SHOWN_WIDE, VISION_MODEL};
    desk.looked = None;
    let (handle, _) = desk.session.eyes.window();
    let part = place_to_show(effects, seen);
    let shown = part.as_ref().map_or_else(|| "창 전체".to_string(), |w| format!("「{}」", w.name));
    let Some(picture) = crate::platform::window_picture(handle, part.as_ref().map(|w| w.rect), MOST_WIDE)
    else {
        out.record("그림", "창을 그리지 못함");
        return None;
    };
    let mut model = Local::new(VISION_MODEL);
    model.think = Some(false);
    model.max_tokens = 256;
    model.keep_alive = "300s".into();
    let began = Instant::now();
    let (png, rect) = (&picture.png, picture.rect);
    let reply = match model.look_at(&asking(text), png) {
        Ok(r) => r,
        Err(why) => {
            out.record("그림", &format!("시각 모델이 답하지 않음 — {why}"));
            out.send("note", &format!("그림 보기 모델이 답하지 않았습니다 — ollama pull {VISION_MODEL}"));
            return None;
        }
    };
    let secs = began.elapsed().as_secs_f64();
    match read(&reply.text) {
        Saw::At { x, y, label } => {
            let (px, py) = on_screen(x, y, rect);
            match landed(seen, px, py) {
                Landed::Inside(place) => {
                    let called = if label.is_empty() { text.to_string() } else { label.clone() };
                    out.record("그림", &format!("{shown}에서 「{called}」 — 「{}」 안, {secs:.2}초", place.name));
                    let said = crate::spoken::say(&crate::spoken::LOOKED_FOUND, &speaking(desk),
                                                  &[&called, &place.name]);
                    let found = effect(place, px, py, &called);
                    //: **Marked where the press will land**, which is the point inside the place
                    //: as the effect holds it — not the model's own number, which `effect` rounded.
                    let at = found.inside.map(|i| i.within(place.rect)).and_then(|(x, y)| picture.at(x, y));
                    desk.looked = at.and_then(|at| crate::platform::marked_picture(&picture, at, SHOWN_WIDE));
                    Some((found, said))
                }
                //: **The list's to answer**, and it was the more exact of the two wherever it had
                //: the thing.
                Landed::OnAControl(w) => {
                    out.record("그림", &format!("{shown}에서 「{label}」 — 목록에 있는 「{}」 위라 버림, {secs:.2}초",
                                               w.name));
                    None
                }
                Landed::OnNothing => {
                    out.record("그림", &format!("{shown}에서 「{label}」 — 목록에 아무것도 없는 곳이라 버림, {secs:.2}초"));
                    None
                }
            }
        }
        Saw::Nothing => {
            out.record("그림", &format!("{shown}에서 찾지 못함, {secs:.2}초"));
            None
        }
        Saw::Unreadable => {
            out.record("그림", &format!("{shown} — 답을 읽지 못함, {secs:.2}초"));
            None
        }
    }
}

/// **Show the pane and look there, or hide it and look at the desktop** — the one way it happens,
/// whether a person pressed 웹 or a turn moved it.
fn show_web(desk: &mut Desk, out: &Listeners, on: bool) -> bool {
    let Some(pane) = desk.web.as_ref() else { return false };
    if desk.on_web.get() == on {
        return true;
    }
    (pane.show)(on);
    desk.on_web.set(on);
    //: The list on screen changes surface, so the next reading is a fresh one.
    desk.session.seen.clear();
    out.send("web", if on { "1" } else { "0" });
    true
}

/// **The turn moves the pane by itself**, by what was said — see `surface::web_move`. Returns true
/// when that was the whole of the request and the turn ends here.
fn moved_web_instead(desk: &mut Desk, out: &Listeners, text: &str) -> bool {
    if desk.web.is_none() {
        return false;
    }
    use crate::spoken::{say, WEB_CLOSED, WEB_OPENED, WEB_SEARCHED};
    let code = speaking(desk);
    let reply = match crate::surface::web_move(text, desk.on_web.get()) {
        None => return false,
        Some(crate::surface::WebMove::ToDesktop) => {
            show_web(desk, out, false);
            return false;
        }
        Some(crate::surface::WebMove::Close) => {
            show_web(desk, out, false);
            say(&WEB_CLOSED, &code, &[])
        }
        Some(crate::surface::WebMove::Open { url, only }) => {
            show_web(desk, out, true);
            if let Some(pane) = desk.web.as_ref() {
                let reader = argmind_env_browser::Pane::new(pane.wire.clone());
                if let Err(why) = reader.open(&url) {
                    out.send("note", &format!("열지 못했습니다 ({why})"));
                }
            }
            desk.session.seen.clear();
            if !only {
                //: Something was asked of the page; the turn carries on, looking at it.
                return false;
            }
            say(&WEB_OPENED, &code, &[&url])
        }
        Some(crate::surface::WebMove::Search(words)) => {
            show_web(desk, out, true);
            if let Some(pane) = desk.web.as_ref() {
                let reader = argmind_env_browser::Pane::new(pane.wire.clone());
                if let Err(why) = reader.open(&argmind_env_browser::search_url(&words)) {
                    out.send("note", &format!("검색하지 못했습니다 ({why})"));
                }
            }
            desk.session.seen.clear();
            say(&WEB_SEARCHED, &code, &[&words])
        }
    };
    desk.session.history.push(("them".into(), text.to_string()));
    desk.session.history.push(("you".into(), reply.clone()));
    out.send("say", &reply);
    true
}

/// The model a task runs on. **Chosen by `--bin tasks`, written down before it ran**: qwen3:14b
/// passed 6 of 6 against qwen3:8b's 0, where 8b ended its tasks by saying they were done. The
/// conversation stays on the session's model, which every other harness measured.
pub const TASK_MODEL: &str = "qwen3:14b";

/// **Reasoning on, for tasks only.** `--bin complex`, the same twelve tasks and the same loop:
/// **11 of 12 with no false finish, against 8 of 12 and two** — at **148 s a task against 9 s**.
/// A task runs on its own thread with the person free to talk, so the time is spent where nobody
/// waits on it; the conversation keeps thinking off, where `repair` measured it halving Korean.
pub const TASK_THINKS: bool = true;

/// Whether a task writes down first what must be true at its end — see `Task::planned`. **Off,
/// measured**: `--bin complex` with thinking on read **8 of 12 with a plan against 11 of 12
/// without**, under the 10 written first — and six of the eight passes were stopped anyway, the
/// work right and an item nobody could quote still open (*the code is written correctly*). A plan
/// written before looking guesses, and quoting the guess holds back the work that proves it.
pub const TASK_PLANS: bool = false;

/// Whether a task may ask the person — `task::Asking`, answered through the conversation. **Off,
/// measured**: on the three errands only the person could finish it read **3 of 3 and 2 of 3 with
/// the tool against 0 of 3 without**, so asking works — but on `--bin complex`, thinking on, the
/// same twelve tasks read **8 of 12 and 10 of 12 with it, each with a false finish, against 12 of 12
/// and none without it in the same hour**, and the one question it asked twice was for a file it
/// could have read. The bar written first was 10 of 12 *and* no false finish.
pub const TASK_ASKS: bool = false;

/// **Reasoning only where it counts** — `argmind_core::task::reasons_here`, and a quick *done*
/// asked again with reasoning. **Off, by the bar written first** (2026-11-06): complex 9 of 12
/// against 12, two false finishes against one, and 68 s and 69 s a task where 66 and 43 were
/// asked — the reasoned calls were fewer and still nine tenths of the time. `*_SPARING=1` in
/// `--bin complex` and `--bin documents`.
pub const TASK_SPARES: bool = false;

/// The task's model, **giving the card up whenever somebody is being answered**: a step with
/// reasoning on runs for minutes, and a turn waiting behind it is the conversation freezing. The
/// call is abandoned the moment a turn begins — dropping the socket frees the card, measured — and
/// asked again once the turn is over.
/// **And let go of when the work is stopped** (2026-11-09): with argmind's own work giving way to a
/// person's request, the request would otherwise wait out a reasoning call of minutes.
struct Yielding(Local, crate::senses::Turn, Arc<AtomicBool>);

impl Speaks for Yielding {
    fn ask_reasoning(&mut self, prompt: &str, reason: bool) -> String {
        let was = self.0.think;
        self.0.think = Some(reason);
        let reply = self.ask(prompt);
        self.0.think = was;
        reply
    }
    fn ask(&mut self, prompt: &str) -> String {
        loop {
            while self.1.in_flight() && !self.2.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            if self.2.load(Ordering::SeqCst) {
                return String::new();
            }
            let (turn, stop) = (self.1.carry_on(), Arc::clone(&self.2));
            let carry_on = move || turn() && !stop.load(Ordering::SeqCst);
            match self.0.ask_while(prompt, &[], &carry_on, &mut |_| {}) {
                Ok(_) if self.2.load(Ordering::SeqCst) => return String::new(),
                Ok(r) if r.reason == "abandoned" => continue,
                Ok(r) => return r.text,
                Err(e) => {
                    eprintln!("  the task model did not answer: {e}");
                    return String::new();
                }
            }
        }
    }
}

/// **Work, rather than a turn: carried to the end by the task loop** — the owner, 2026-11-03:
/// *문서수정이나 작성 그리고 코드도 필요하면 생성해야하고 실행도 스스로.* One task at a time, on its
/// own thread so the conversation goes on; each step goes on the timeline as it happens, and what
/// it ended with is said and kept. Between steps it waits out a turn, the rule every sense keeps.
/// **Not gated by 확인 없이 실행**: what it touches is the work folder, a sandbox that can reach only
/// that folder, and PowerShell in it — the last one weaker, and said so in `workspace.rs`.
fn worked_instead(desk: &mut Desk, out: &Listeners, text: &str, said: &str) -> bool {
    use crate::spoken::{say, NO_WORKBENCH, OWN_GIVES_WAY, OWN_HALTED, STILL_WORKING, WORK_HALTED, WORK_STARTED};
    let busy = desk.working.load(Ordering::SeqCst);
    let own = desk.own_work.load(Ordering::SeqCst);
    let code = speaking(desk);
    let reply = if busy && argmind_core::tongue::stops(text) {
        desk.stop_work.store(true, Ordering::SeqCst);
        if own {
            desk.own_halted = true;
            say(&OWN_HALTED, &code, &[])
        } else {
            say(&WORK_HALTED, &code, &[])
        }
    //: **argmind's own work gives way to the person's**, the owner's word: *다른작업이나 대화를
    //: 해야할떄는 다른일 우선*. It stops between steps and the request starts when it has.
    } else if busy && own {
        desk.stop_work.store(true, Ordering::SeqCst);
        desk.after_own = Some(text.to_string());
        out.record("작업", "스스로 하던 일을 멈추고 맡기신 일부터 합니다");
        say(&OWN_GIVES_WAY, &code, &[])
    //: Not asked again whether this is work: `crate::route` decided, and the model's *work* is a
    //: sentence the word rule did not recognise — asking the rule here would undo it.
    } else if busy {
        say(&STILL_WORKING, &code, &[])
    } else if let Some(sets) = work_sets(desk, false) {
        start_work(desk, text, sets, false);
        say(&WORK_STARTED, &code, &[&crate::platform::work_folder().display().to_string()])
    } else {
        say(&NO_WORKBENCH, &code, &[])
    };
    desk.session.history.push(("them".into(), said.to_string()));
    desk.session.history.push(("you".into(), reply.clone()));
    out.send("say", &reply);
    true
}

/// **Everything a task can work with**, ranked — `None` where there is no workbench. `own` is work
/// argmind set itself: it asks nobody, and what it presses is its own idea.
fn work_sets(desk: &Desk, own: bool)
    -> Option<Vec<(argmind_core::task::Rank, Box<dyn argmind_core::task::Tools + Send>)>> {
    let files = crate::platform::workbench()?;
    {
        //: **And the web, when there is a pane** — a trip plan is searching before it is writing.
        use argmind_core::task::Rank;
        //: **Ranked**, the owner's order of 2026-11-08: MCP or an API, then code, then a screen —
        //: the MCP servers join on the task's thread, in `start_work`.
        let mut sets: Vec<(Rank, Box<dyn argmind_core::task::Tools + Send>)> = vec![(Rank::Code, files)];
        //: **And the person**, who is asked through the conversation: the question is said like a
        //: result, and the next thing they say comes back here — see `task_asks` in `think`.
        let (slot, jobs) = (Arc::clone(&desk.task_asks), desk.jobs.clone());
        let stop = Arc::clone(&desk.stop_work);
        if TASK_ASKS && !own { sets.push((Rank::Other, Box::new(argmind_core::task::Asking::new(move |q: &str| {
            let (tx, rx) = std::sync::mpsc::channel::<String>();
            if let Ok(mut w) = slot.lock() {
                *w = Some(tx);
            }
            let _ = jobs.send(Job::Worked(Worked::Asks(q.to_string())));
            //: Ten minutes, and *멈춰* drops the question at once.
            let asked_at = std::time::Instant::now();
            let got = loop {
                match rx.recv_timeout(std::time::Duration::from_millis(500)) {
                    Ok(a) => break Some(a),
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break None,
                    Err(_) if stop.load(Ordering::SeqCst) || asked_at.elapsed().as_secs() >= 600 => break None,
                    Err(_) => {}
                }
            };
            if let Ok(mut w) = slot.lock() {
                w.take();
            }
            got
        })))); }
        if let Some(pane) = desk.web.as_ref() {
            sets.push((Rank::Other, Box::new(argmind_env_browser::WebTools(
                argmind_env_browser::Pane::new(pane.wire.clone())))));
            //: **And pressing there — the pane only**, the owner's choice of 2026-11-06: a task runs
            //: for minutes beside somebody using their own desktop.
            sets.push((Rank::Other, pane_hands(desk.jobs.clone(), Arc::clone(&desk.stop_work), own)));
        }
        //: **And a screen of argmind's own, last** — a desktop the person never sees.
        //: **Installed programs by their own scripting interfaces, with the APIs** — Blender's
        //: Python and kicad-cli, when this machine has them.
        if let Some(programs) = crate::platform::installed_programs() {
            sets.push((Rank::Api, programs));
        }
        //: **Every press and write there asks the person's 규칙 first**, on the worker that holds
        //: them; a question waits as long as a row would, and 멈춰 lets go of it.
        let (gate_jobs, gate_stop) = (desk.jobs.clone(), Arc::clone(&desk.stop_work));
        let gate = Box::new(move |window: &str, kind: &str, what: &str| -> Result<(), String> {
            let (tx, rx) = std::sync::mpsc::channel();
            gate_jobs.send(Job::ScreenGate(window.into(), kind.into(), what.into(), tx, own))
                .map_err(|_| "argmind 이 닫혔다 — 하지 않았다".to_string())?;
            let asked_at = std::time::Instant::now();
            loop {
                match rx.recv_timeout(std::time::Duration::from_millis(500)) {
                    Ok(answer) => return answer,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return Err("답이 오지 않았다".into()),
                    Err(_) if gate_stop.load(Ordering::SeqCst) => return Err("사람이 작업을 멈췄다".into()),
                    Err(_) if asked_at.elapsed().as_secs() >= 600 => return Err("10분 동안 답이 없었다 — 하지 않았다".into()),
                    Err(_) => {}
                }
            }
        });
        if let Some(screen) = crate::platform::work_screen(gate) {
            sets.push((Rank::Screen, screen));
        }
        Some(sets)
    }
}

/// **What can be pressed in the pane, read there**, whatever the switch says — the switch is put
/// back on return.
fn pane_look(desk: &mut Desk) -> Result<Vec<(String, String)>, String> {
    if desk.web.is_none() {
        return Err("웹 칸이 없다".into());
    }
    let _pane = desk.on_web.for_now(true);
    let (widgets, _) = desk.session.observe();
    Ok(widgets.into_iter().take(desk.session.limit).map(|w| (w.role, w.name)).collect())
}

/// **A task's press, as a turn's** — `propose` with the effect a person asked for, on the pane. If
/// it ran or was refused, the task hears at once; if it waits on a row — 확인 없이 실행 off, or a
/// rule that always asks — the task waits for the person, and hears when they answer.
fn pane_press(desk: &mut Desk, out: &Listeners, effect: Effect, tx: Sender<String>) {
    if desk.web.is_none() {
        let _ = tx.send("웹 칸이 없다 — 누르지 않았다".into());
        return;
    }
    let mark = out.mark();
    let _pane = desk.on_web.for_now(true);
    let before: Vec<u64> = desk.proposals.waiting().iter().map(|p| p.token).collect();
    let went = propose(desk, out, effect, None, false);
    let row = desk.proposals.waiting().iter().map(|p| p.token).find(|t| !before.contains(t));
    match row {
        Some(token) if went => {
            out.send("note", "작업이 웹 칸에서 누르려고 합니다 — 실행 줄을 눌러 주세요");
            desk.task_presses.push((token, tx, mark));
        }
        _ => {
            let _ = tx.send(out.pressed_since(mark));
        }
    }
}

/// **A task's press or write on the work screen, put to the person's 규칙** — refused, let through,
/// or asked as a question whose one button is yes. Never a `Proposal`: the work screen is not the
/// person's, nothing there reaches `perform`, and the answer goes back to the waiting task.
fn screen_gate(desk: &mut Desk, out: &Listeners, window: &str, kind: &str, what: &str,
               tx: Sender<Result<(), String>>, own: bool) {
    use crate::rules::OnTheWorkScreen;
    //: **Whose idea the work was decides which standing answer counts**, as in `propose`.
    let acting = crate::decided::standing(if own { "initiative" } else { "acting" }) == Some(Standing::Allowed);
    match crate::rules::on_the_work_screen(&desk.rules, window, kind, acting) {
        OnTheWorkScreen::Go => {
            let _ = tx.send(Ok(()));
        }
        OnTheWorkScreen::Refuse(said) => {
            out.record("작업", &format!("작업 화면 「{window}」 {what} → 규칙으로 하지 않음"));
            out.send("note", &said);
            let _ = tx.send(Err(format!("{said} — 하지 않았다. 다른 방법을 쓰거나 사람에게 말하라")));
        }
        OnTheWorkScreen::Ask => {
            static ASKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
            let ask = ASKS.fetch_add(1, Ordering::SeqCst);
            let yes = if kind == "type" { "넣기" } else { "누르기" };
            let about = crate::wonders::About::Press { ask, what: what.to_string(), window: window.to_string(),
                                                       among: vec![yes.to_string()] };
            let raised = desk.wonders.lock().unwrap_or_else(|e| e.into_inner()).ask(about).map(wonder_json);
            match raised {
                Some(json) => {
                    out.record("작업", &format!("작업 화면 「{window}」 {what} → 여쭤봄"));
                    out.send("wonder", &json);
                    desk.screen_asks.push((ask, tx));
                }
                //: One question at a time — a second one is not stacked behind it.
                None => {
                    let _ = tx.send(Err("다른 질문에 아직 답이 없어 여쭤보지 못했다 — 하지 않았다".into()));
                }
            }
        }
    }
}

/// The person answered a row a task was waiting on — or it went.
fn settle_press(desk: &mut Desk, out: &Listeners, token: u64, how: &str) {
    if let Some(i) = desk.task_presses.iter().position(|(t, ..)| *t == token) {
        let (_, tx, mark) = desk.task_presses.remove(i);
        let said = out.pressed_since(mark);
        let _ = tx.send(if how.is_empty() { said } else { format!("{said}\n{how}") });
    }
}

/// **The pane's hands, for a task** — the task thread asks the worker, which looks and presses as a
/// turn does. A press waits for its answer as long as a person might take over a row, and *멈춰*
/// lets go of it.
fn pane_hands(jobs: Sender<Job>, stop: Arc<AtomicBool>, own: bool) -> Box<dyn argmind_core::task::Tools + Send> {
    let looks = jobs.clone();
    let make = if own { argmind_core::task::Pressing::own } else { argmind_core::task::Pressing::new };
    Box::new(make(
        move || {
            let (tx, rx) = std::sync::mpsc::channel();
            looks.send(Job::PaneLook(tx)).map_err(|_| "argmind 이 닫혔다".to_string())?;
            rx.recv_timeout(std::time::Duration::from_secs(30)).map_err(|_| "웹 칸을 보지 못했다".to_string())?
        },
        move |effect| {
            let (tx, rx) = std::sync::mpsc::channel();
            if jobs.send(Job::PanePress(effect, tx)).is_err() {
                return "argmind 이 닫혔다 — 누르지 않았다".into();
            }
            let asked_at = std::time::Instant::now();
            loop {
                match rx.recv_timeout(std::time::Duration::from_millis(500)) {
                    Ok(said) => return said,
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return "답이 오지 않았다".into(),
                    Err(_) if stop.load(Ordering::SeqCst) => return "사람이 작업을 멈췄다".into(),
                    Err(_) if asked_at.elapsed().as_secs() >= 600 => return "10분 동안 실행 줄에 답이 없었다".into(),
                    Err(_) => {}
                }
            }
        }))
}

fn start_work(desk: &Desk, what: &str,
              mut sets: Vec<(argmind_core::task::Rank, Box<dyn argmind_core::task::Tools + Send>)>, own: bool) {
    use argmind_core::task::{open, step, Ended, Task};
    let (working, stop, turn) = (Arc::clone(&desk.working), Arc::clone(&desk.stop_work), desk.turn.clone());
    let (jobs, out, what) = (desk.jobs.clone(), Arc::clone(&desk.speaks_to), what.to_string());
    let own_work = Arc::clone(&desk.own_work);
    working.store(true, Ordering::SeqCst);
    own_work.store(own, Ordering::SeqCst);
    stop.store(false, Ordering::SeqCst);
    std::thread::spawn(move || {
        let mut local = Local::new(TASK_MODEL);
        local.think = Some(TASK_THINKS);
        local.keep_alive = "300s".into();
        if own {
            local.num_gpu = Some(OWN_GPU_LAYERS);
        }
        let mut model = Yielding(local, turn.clone(), Arc::clone(&stop));
        let mut task = if TASK_PLANS { Task::planned(&what) } else { Task::new(&what) };
        task.sparing = TASK_SPARES;
        let today = crate::platform::today();
        out.record("작업", &format!("{}시작: {what}", if own { "스스로 " } else { "" }));
        //: **And the MCP servers the person configured**, started here rather than on the turn's
        //: thread — `npx` can take a minute the first time — and stopped when this thread ends.
        let (servers, notes) = crate::mcp::started();
        for n in notes {
            out.record("작업", &n);
        }
        if let Some(s) = servers {
            sets.push((argmind_core::task::Rank::Api, Box::new(s)));
        }
        let mut bench: Box<dyn argmind_core::task::Tools + Send> = Box::new(argmind_core::task::Tiered::new(sets).screen_asked_for(&what));
        //: **The folder before**, so what the task made is found by the program at the end.
        let folder = crate::platform::work_folder();
        let before = crate::deliver::snapshot(&folder);
        //: **What the request names is read before the model's first step** — `--bin documents`
        //: measured it in `run`, and this loop steps by itself, so it is called here too. Until
        //: 2026-11-06 it was not, and the application ran without the fix the harness credited.
        open(&mut task, bench.as_mut());
        for s in &task.steps {
            out.record("작업", &format!("{} {} → 먼저 읽음", s.tool, s.args.to_string().chars().take(80).collect::<String>()));
        }
        let card = begun(&out, &if own { format!("스스로: {what}") } else { what.clone() });
        loop {
            while turn.in_flight() && !stop.load(Ordering::SeqCst) {
                std::thread::sleep(std::time::Duration::from_millis(200));
            }
            if stop.load(Ordering::SeqCst) {
                task.ended = Some(Ended::Stopped("사람이 멈췄습니다".into()));
                break;
            }
            let (before, had_plan) = (task.steps.len(), task.planned);
            let going = step(&mut task, &mut model, bench.as_mut(), &today);
            if !had_plan && task.planned {
                let items: Vec<String> = task.plan.iter().enumerate()
                    .map(|(n, i)| format!("{}. {}", n + 1, i.what)).collect();
                out.record("작업", &format!("계획: {}", if items.is_empty() { "(읽지 못함 — 계획 없이)".into() }
                                                        else { items.join(" · ") }));
            }
            for s in &task.steps[before..] {
                let first = gist(&s.observed, s.failed);
                let args: String = s.args.to_string().chars().take(80).collect();
                out.record("작업", &format!("{}{} {args} → {}", s.tool, if s.failed { " ✗" } else { "" },
                                            first.chars().take(100).collect::<String>()));
            }
            if let Some(s) = task.steps[before..].last() {
                let first = gist(&s.observed, s.failed);
                stepped(&out, card, task.steps.len(), &format!("{} — {}", s.tool, first.chars().take(80).collect::<String>()), s.failed);
            }
            if !going {
                break;
            }
        }
        //: **Korean here**, for the card and the record; what the person hears is made from `told`
        //: on the worker, in the language they are spoken to in.
        let (said, finished, told) = match task.ended {
            Some(Ended::Done(s)) => (s.clone(), Some(s.clone()), Worked::Done(s)),
            Some(Ended::Stopped(why)) => (format!("작업을 다 끝내지 못했습니다 — {why}"), None, Worked::NotFinished(why)),
            None => ("작업이 멈췄습니다".into(), None, Worked::Stopped),
        };
        //: **What it made, shown** — read the way the task read it, so a Word file is its text.
        //: **Every changed file is named; only the readable ones are shown.** The record of
        //: argmind's own work had kept only what reads as text, so an STL or a .blend it made was
        //: missing from it — and from what the next pick was shown (2026-11-09).
        let after = crate::deliver::snapshot(&folder);
        let names: Vec<String> = crate::deliver::changed(&before, &after);
        let made: Vec<(String, String)> = crate::deliver::made(&before, &after).into_iter()
            .filter_map(|rel| argmind_core::task::read_through(bench.as_mut(), &rel).ok()
                .map(|text| (rel, crate::deliver::trimmed(&text))))
            .collect();
        if !names.is_empty() {
            out.record("작업", &format!("만든 것: {}", names.join(", ")));
            //: **Not shown for argmind's own work**: opening the pane on it would take the view
            //: from somebody who did not ask for anything. It is told when they come back.
            if !own {
                let _ = jobs.send(Job::Made(made));
            }
        }
        working.store(false, Ordering::SeqCst);
        own_work.store(false, Ordering::SeqCst);
        ended(&out, card, task.steps.len(), finished.is_some(), &said);
        if own {
            let _ = jobs.send(Job::OwnEnded(argmind_core::pursuit::Pursued {
                asked: what.clone(), finished: finished.is_some(), said: finished.clone().unwrap_or(said), made: names }));
        } else {
            let _ = jobs.send(Job::Worked(told));
        }
    });
}

/// **How much of the task model argmind's own work keeps on the graphics card** — the rest on the
/// CPU, so the conversation's model stays beside it. The owner chose this over waiting for a quiet
/// spell (2026-11-16), after a turn during argmind's own work waited **6.1 s to route and 10 s to
/// answer** while qwen3:8b came back onto a 16 GB card qwen3:14b had filled. Measured with the
/// criteria written first — both stay loaded, at most 13.4 GB with 2.5 GB left for the voice and
/// the recogniser, an 8b turn under 1.5 s, and asking again past 5x slower: **24 layers** read 12.6
/// GB, an 8b turn **0.41 s**, both loaded, at **8.3 tokens a second against 36.7 — 4.4x slower**; 28
/// layers was 13.5 GB, over, and hardly faster (9.0); 20 was 5.6x. One run each, 400 tokens with
/// reasoning on. A request a person hands over still has the whole card.
const OWN_GPU_LAYERS: u32 = 24;

/// Between one piece of argmind's own work and the next. Chosen: long enough to be found between
/// pieces now and then, short of idling — the owner asked for it to go on without limit.
const OWN_PAUSE: std::time::Duration = std::time::Duration::from_secs(60);
/// After 멈춰, after three pieces in a row that did not finish, or picks that could not be read.
/// Chosen: a goal that keeps failing is a loop spending the card, not work.
const OWN_BACKOFF: std::time::Duration = std::time::Duration::from_secs(1800);

/// **The goal, worked on when nobody needs argmind** — the owner, 2026-11-09: *사용자 작업이
/// 없을떄 스스로 목표를 설정하고 이루도록 수행*, continuously, the conversation and handed work first.
/// When there is a goal, no task, no turn, no row or question waiting and the pause has passed, the
/// task model is asked for the next piece (`argmind_core::pursuit`) on a thread of its own, and
/// what it answers comes back as [`Job::OwnPicked`].
fn pursue(desk: &mut Desk) {
    let aim = desk.session.mind.aim.trim().to_string();
    let asking = desk.task_asks.lock().map(|w| w.is_some()).unwrap_or(false);
    if aim.is_empty() || desk.own_choosing || desk.working.load(Ordering::SeqCst) || desk.turn.in_flight()
        || Instant::now() < desk.own_not_before || desk.after_own.is_some() || asking
        || !desk.proposals.waiting().is_empty() || !desk.screen_asks.is_empty() {
        return;
    }
    desk.own_choosing = true;
    let done = crate::pursuit::remembered();
    //: **Only what its own work made and is still there** — not the whole folder. The first piece
    //: in the application (2026-11-09) was *a Blender script for a 3D-printable model*, toward a goal
    //: about money: the folder held the person's own tasks' `box.py` and `part.stl`, and the pick
    //: followed them. `--bin pursuit` had passed with an empty folder.
    let folder = crate::platform::work_folder();
    let mut files: Vec<String> = Vec::new();
    for f in done.iter().flat_map(|d| d.made.iter()) {
        if folder.join(f).exists() && !files.contains(f) {
            files.push(f.clone());
        }
    }
    let (jobs, turn, today) = (desk.jobs.clone(), desk.turn.clone(), crate::platform::today());
    let refused = desk.own_refused.clone();
    //: A pick is let go of only for a turn; nothing stops it but argmind closing.
    let never = Arc::new(AtomicBool::new(false));
    std::thread::spawn(move || {
        let mut local = Local::new(TASK_MODEL);
        local.think = Some(TASK_THINKS);
        local.keep_alive = "300s".into();
        local.num_gpu = Some(OWN_GPU_LAYERS);
        let mut model = Yielding(local, turn, never);
        let reply = model.ask(&argmind_core::pursuit::ask_for_next(&aim, &done, &files, &today, refused.as_deref()));
        let _ = jobs.send(Job::OwnPicked(argmind_core::pursuit::read_next(&reply, &done)));
    });
}

/// The next piece was picked: run it, if nothing else started meanwhile.
fn own_picked(desk: &mut Desk, out: &Listeners, picked: Result<String, String>) {
    desk.own_choosing = false;
    match picked {
        Ok(next) if !desk.working.load(Ordering::SeqCst) && desk.after_own.is_none()
            && !desk.session.mind.aim.trim().is_empty() => {
            desk.own_refused = None;
            match work_sets(desk, true) {
                Some(sets) => start_work(desk, &next, sets, true),
                None => desk.own_not_before = Instant::now() + OWN_BACKOFF,
            }
        }
        Ok(_) => {}
        Err(why) => {
            out.record("작업", &format!("스스로 할 일을 정하지 못함 — {why}"));
            desk.own_refused = Some(why.clone());
            desk.own_misses += 1;
            desk.own_not_before = Instant::now() + if desk.own_misses >= 3 { OWN_BACKOFF } else { OWN_PAUSE };
        }
    }
}

/// A piece of argmind's own work ended: kept, owed to the person, and the next one timed — and a
/// person's request that stopped it started now.
fn own_ended(desk: &mut Desk, out: &Listeners, done: argmind_core::pursuit::Pursued) {
    if let Err(why) = crate::pursuit::append(&done) {
        out.send("note", &why);
    }
    let gave_way = desk.after_own.is_some() || desk.own_halted;
    if done.finished {
        desk.own_misses = 0;
    } else if !gave_way {
        desk.own_misses += 1;
    }
    let wait = if desk.own_halted || desk.own_misses >= 3 { OWN_BACKOFF } else { OWN_PAUSE };
    desk.own_halted = false;
    desk.own_not_before = Instant::now() + wait;
    desk.own_untold.push(done);
    if let Some(text) = desk.after_own.take() {
        if let Some(sets) = work_sets(desk, false) {
            start_work(desk, &text, sets, false);
            out.send("note", "맡기신 일을 시작합니다");
        }
    }
}

/// **What argmind did by itself, said when somebody comes back** — the owner's choice, short. Back
/// is a person talking to argmind (`spoke`), or the camera and input reading somebody about after
/// they read nobody. Into the history, so *그 파일 보여줘* has something to refer to.
fn tell_own(desk: &mut Desk, out: &Listeners, spoke: bool) {
    match somebody_about(desk) {
        Some(false) => {
            desk.own_away = true;
            return;
        }
        Some(true) if desk.own_away => {}
        _ if spoke => {}
        _ => return,
    }
    if desk.own_untold.is_empty() || desk.turn.in_flight() {
        return;
    }
    let code = speaking(desk);
    let untold = std::mem::take(&mut desk.own_untold);
    let session = &mut desk.session;
    let said = crate::pursuit::report_in(&untold, &code, &mut |t: &str| work_words(session, t, &code));
    desk.own_away = false;
    desk.session.history.push(("you".into(), said.clone()));
    out.send("say", &said);
}

/// **Each task as a card, where a person is looking.** The owner, 2026-11-09: a box asked for in
/// Blender was answered *해 볼게요* and nothing more could be seen — its steps went to the 한 일
/// fold, closed. Then: *수행하는 작업별로 진행상황이랑 완료여부를 간단하게 표시해줘.* So every task is
/// a card — its request, how many steps, what the last one did, and how it ended — the last
/// [`CARDS_KEPT`] of them, sent whole as a `tasks` event on every change and to a page that opens
/// mid-task. Memory only, like 한 일. Not a timeline line: those are written beside each step.
struct Card {
    id: u64,
    what: String,
    steps: usize,
    last: String,
    failed: bool,
    /// `running`, `done` or `stopped`.
    state: &'static str,
    at: i64,
}

/// **The line of a step's result a person should read.** Its first line, unless the step failed:
/// then the line that names the error. The owner's box in Blender, 2026-11-09, showed *오류:
/// [stderr]* on the card twice — the first line of every failed command — while the reason,
/// *keyword "use_selection" unrecognized*, sat at the bottom of a traceback.
pub fn gist(observed: &str, failed: bool) -> String {
    let lines: Vec<&str> = observed.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let first = lines.first().copied().unwrap_or("");
    if !failed {
        return first.to_string();
    }
    //: Not the exit code and not Blender's closing *script failed*, which say only that it failed.
    let saying: Vec<&str> = lines.iter().copied().filter(|l| !l.starts_with("(종료 코드")
        && !l.starts_with("Error: script failed") && !l.ends_with("[stderr]")).collect();
    saying.iter().rev()
        .find(|l| l.contains("Error") || l.contains("error") || l.contains("오류") || l.contains("실패"))
        .or_else(|| saying.last())
        .copied().unwrap_or(first).to_string()
}

const CARDS_KEPT: usize = 5;
static CARDS: Mutex<Vec<Card>> = Mutex::new(Vec::new());

fn cards_json() -> String {
    let cards = CARDS.lock().unwrap_or_else(|e| e.into_inner());
    format!("[{}]", cards.iter().map(|c| format!(
        "{{\"id\":{},\"what\":\"{}\",\"steps\":{},\"last\":\"{}\",\"failed\":{},\"state\":\"{}\",\"at\":{}}}",
        c.id, escape(&c.what), c.steps, escape(&c.last), c.failed, c.state, c.at)).collect::<Vec<_>>().join(","))
}

fn begun(out: &Listeners, what: &str) -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let id = NEXT.fetch_add(1, Ordering::SeqCst);
    {
        let mut cards = CARDS.lock().unwrap_or_else(|e| e.into_inner());
        cards.push(Card { id, what: what.into(), steps: 0, last: "시작했습니다".into(), failed: false,
                          state: "running", at: now_seconds() });
        let over = cards.len().saturating_sub(CARDS_KEPT);
        cards.drain(..over);
    }
    out.frame("tasks", &cards_json());
    id
}

fn stepped(out: &Listeners, id: u64, steps: usize, last: &str, failed: bool) {
    if let Some(c) = CARDS.lock().unwrap_or_else(|e| e.into_inner()).iter_mut().find(|c| c.id == id) {
        (c.steps, c.last, c.failed, c.at) = (steps, last.into(), failed, now_seconds());
    }
    out.frame("tasks", &cards_json());
}

fn ended(out: &Listeners, id: u64, steps: usize, done: bool, said: &str) {
    if let Some(c) = CARDS.lock().unwrap_or_else(|e| e.into_inner()).iter_mut().find(|c| c.id == id) {
        (c.steps, c.last, c.failed, c.at) = (steps, said.chars().take(120).collect(), !done, now_seconds());
        c.state = if done { "done" } else { "stopped" };
    }
    out.frame("tasks", &cards_json());
}

/// **What a task made, in the pane** — opened for it, the files written into `about:blank` as
/// escaped text. Without a pane the summary is all there is, which is what there was before.
fn show_made(desk: &mut Desk, out: &Listeners, files: &[(String, String)]) {
    if desk.web.is_none() {
        return;
    }
    show_web(desk, out, true);
    let Some(pane) = desk.web.as_ref() else { return };
    let page = argmind_env_browser::deliverable_page(files);
    match argmind_env_browser::Pane::new(pane.wire.clone()).show_page(&page) {
        Ok(()) => out.send("note", &format!("만든 것을 웹 칸에 띄웠습니다 — {}",
                                            files.iter().map(|(p, _)| p.as_str()).collect::<Vec<_>>().join(", "))),
        Err(why) => out.send("note", &format!("만든 것을 웹 칸에 띄우지 못했습니다 ({why})")),
    }
    desk.session.seen.clear();
}

/// **A question about now, asked while argmind looks at its pane, is looked up by the program.**
/// The owner's session, 2026-10-21: *오늘 서울날씨 알려줘* on a login page, and the model's plan was
/// to type *서울* into the ID box. A lookup is not something to plan by clicking; the program puts
/// the search in the pane where the person can see it, reads the page, and hands that to an errand
/// to conclude — the errand path whose answer shapes have no `say` and no `do`.
fn looked_up_instead(desk: &mut Desk, out: &Listeners, text: &str) -> bool {
    if desk.web.is_none() || !argmind_core::errand::asks_to_look_up(text) {
        return false;
    }
    //: **Opened by the question**, since 2026-10-21 — it used to need the 웹 switch on first.
    show_web(desk, out, true);
    let Some(pane) = desk.web.as_ref() else { return false };
    out.send("busy", "1");
    let reader = argmind_env_browser::Pane::new(pane.wire.clone());
    let read = reader.look_up(text);
    out.send("busy", "0");
    let reply = match read {
        Ok((page, at)) => match crate::errands::add(&mut desk.errands, text) {
            Ok(id) => {
                if let Some(e) = desk.errands.iter_mut().find(|e| e.id == id) {
                    e.looked_up(&page, &at);
                    e.needs_web = true;
                }
                keep_errands(desk, out);
                desk.session.seen.clear();
                crate::spoken::say(&crate::spoken::LOOKED_UP, &speaking(desk), &[])
            }
            Err(why) => crate::spoken::known(why, &speaking(desk)),
        },
        Err(why) => crate::spoken::say(&crate::spoken::NOT_LOOKED_UP, &speaking(desk), &[&why.to_string()]),
    };
    desk.session.history.push(("them".into(), text.to_string()));
    desk.session.history.push(("you".into(), reply.clone()));
    out.send("say", &reply);
    true
}

/// **The model's context, in tokens** — `Local`'s `num_ctx`, which the prompt must fit inside.
/// The history fills what the screen and the reply leave of it.
pub const CONTEXT_TOKENS: usize = 8192;

/// **Write what the conversation gained since last time**, and keep the history in memory bounded
/// — the file holds it all, the prompt takes what fits.
fn keep_conversation(desk: &mut Desk, out: &Listeners) {
    let h = &desk.session.history;
    if desk.kept_lines > h.len() {
        desk.kept_lines = h.len();
    }
    if let Err(why) = crate::conversation::append(&h[desk.kept_lines..]) {
        out.send("note", &why);
    }
    desk.kept_lines = h.len();
    if desk.session.history.len() > 2 * crate::conversation::READ_BACK {
        let drop = desk.session.history.len() - crate::conversation::READ_BACK;
        desk.session.history.drain(..drop);
        desk.kept_lines -= drop;
    }
}

/// **Below this, an action waits for a person even when it would otherwise run by itself.**
///
/// Measured 2026-10-21 by `--bin sure`, the probability of the tokens of the reply's `do`: on the
/// 37 goals of `choosing`, 21 of 27 actions sat at 1.000 and the threshold that kept 90% right was
/// **0.94** — 21 of 23 right above it, 22 of 27 without it. **Then checked where it was not
/// chosen**, the browsing pane's 25 goals on live pages: **15 of 16 right at or above 0.94, and
/// both actions below it wrong** — *구인 글* at 0.223 and *브랜치 목록* at 0.926, each clicking links
/// nobody asked for beside the right one, which multiplying over every token of `do` pulls down.
/// Small, and it does not catch a wrong action the model is certain of: two sat at 1.000.
pub const SURE_ENOUGH: f64 = 0.94;

/// File one offer, and run it when 확인 없이 실행 allows. **Returns whether the step after this
/// one may go ahead**: not when this one is gone, waits on a question, is refused by a rule, or ran
/// and did not land.
fn propose(desk: &mut Desk, out: &Listeners, effect: Effect, sure: Option<f64>, asked: bool) -> bool {
    //: **Which app this would happen in.** A click knows its widget's window; typing and keys go
    //: to the window in front, which is the one argmind is attending to.
    let mut window = desk.session.eyes.window().1;
    let (text, hint) = if effect.kind == "click" {
        match desk.session.find(&effect.target, &effect.role) {
            argmind_core::session::Found::Gone => {
                out.send("gone",
                         &format!("{{\"token\":-1,\"what\":\"{}\"}}", escape(&effect.target)));
                return false;
            }
            //: **Asked rather than guessed.** Before 2026-09-28 the lookup picked whichever copy
            //: an arbitrary anchor favoured and this drew a button for it; asked for the font list
            //: in Character Map's advanced view, that opened the character set list.
            //:
            //: **The question replaces the offer rather than joining it.** A proposal is still
            //: filed, so the action keeps its token and `Proposals` keeps making a stale one unable
            //: to confirm a later act — but no `propose` event goes out, because a row with a 실행
            //: button beside a question asking which one is two decisions for one action.
            //: **Answering is the confirmation**: somebody who picked the second `Open` has been at
            //: least as deliberate as somebody who pressed a button, so a second press would be
            //: ceremony.
            argmind_core::session::Found::Several(same) => {
                ask_which(desk, out, effect, &same);
                return false;
            }
            argmind_core::session::Found::One(found) => {
                if !found.window.is_empty() {
                    window = found.window.clone();
                }
                //: The point a press would land on: the widget's middle, or the point inside it.
                let (cx, cy) = effect.inside.map_or_else(|| found.centre(), |i| i.within(found.rect));
                //: **A second opinion, not a gate.** A widget's own centre resolves back to itself
                //: about seven times in ten, and a caption button answering `application` is the
                //: hit test disagreeing with a pixel that is correct.
                let hint = match desk.session.eyes.at_point(cx, cy) {
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
    //: **A rule the person wrote, checked by the program and never shown to the model.** *Never*
    //: is not offered at all — a row with a 실행 button for something forbidden would be the
    //: button arguing with the rule — and the turn is told in words so the next one can see why.
    let rule = argmind_core::rule::applies(&desk.rules, &window, &effect.kind);
    if rule == Some(argmind_core::rule::Then::Never) {
        let said = argmind_core::rule::told(&desk.rules, &window, &effect.kind);
        desk.session.history.push(("could not".into(), format!(
            "{} — the person has a rule against it in this app; nothing was done",
            label(&effect, None))));
        out.send("note", &said);
        return false;
    }
    let doubt = !hint.is_empty();
    //: *Always ask* here means the button, whatever 확인 없이 실행 says — and so does a reply that
    //: itself asked (*찾아볼까요?*): the words wait for the person, so the action does too.
    //: **And a point the picture found, always** — the owner's condition of 2026-11-18 for letting a
    //: model say where at all: 확인 없이 실행 never covers it. `argmind-rules` reads this line.
    let must_confirm = rule == Some(argmind_core::rule::Then::Confirm) || asked || effect.inside.is_some();
    //: **How sure the model was of this action** — below [`SURE_ENOUGH`] argmind's own idea is let
    //: go. `None` (a model that cannot say) gates nothing.
    let unsure = sure.is_some_and(|s| s < SURE_ENOUGH);
    //: **Whose idea it was**, read before the effect is handed over. `unprompted()` is `Some(false)`
    //: for anything a watch or a raised judgement produced.
    let its_own_idea = effect.unprompted();
    //: **argmind's own idea, doubtful, with 스스로 실행 on: neither asked nor done.** The owner,
    //: 2026-11-03, with five 실행 rows waiting on screen: 자꾸 실행을 물어봐서 일이 진행이 안됨. A
    //: row for something nobody asked for and the model was not sure of is the pile of buttons that
    //: stalls the work; dropping it costs a suggestion, and nobody was waiting for that suggestion.
    //: With 스스로 실행 off, argmind's ideas are still offered as rows — that is a person who wants
    //: to be asked.
    if its_own_idea && (doubt || unsure)
        && crate::decided::standing("initiative") == Some(Standing::Allowed) {
        desk.session.history.push(("could not".into(), format!(
            "{} — not sure enough of an idea nobody asked for; nothing was done",
            label(&effect, None))));
        return false;
    }
    //: **In the conversation, not only on the screen.** An offer the person then argues with —
    //: *"not that one"* — has to be something the next turn can see.
    desk.session.remember_doing(&[], &[], std::slice::from_ref(&effect), &[]);
    //: **The picture goes with the point it marks**, and only with it.
    let picture = if effect.inside.is_some() { desk.looked.take() } else { None };
    let token = desk.proposals.offer(effect, text, hint, desk.on_web.get());
    if let Some(png) = picture {
        crate::looking::keep_shown(token, png);
    }
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
    //: **Whose idea it was decides which standing answer counts.** What a person asked for runs on
    //: `acting`; what argmind thought of runs only on `initiative` — two questions, two answers,
    //: the owner's of 2026-09-22 and 2026-10-21.
    //: **What a person asked for runs, since 2026-11-03** — no sureness gate and no second
    //: opinion in front of it, at the owner's word: 그냥 자동으로 수행해야돼. The sureness gate was
    //: measured on single clicks and multiplies over every token of `do`, so a click, a typed word
    //: and Enter fell below it almost by construction — the request that stalled was exactly that.
    //: What still stands is the person's own 항상 물어봄 rule, the *which one?* question a program
    //: cannot answer for them, and every check `perform` makes against the screen at each step.
    if !must_confirm && (!its_own_idea || (!doubt && !unsure)) && crate::decided::standing(if its_own_idea { "initiative" } else { "acting" }) == Some(Standing::Allowed) {
        if let Answered::Run(p) = desk.proposals.confirm(token) {
            return act(desk, out, p);
        }
        return false;
    }
    //: Waiting for a person's press, which goes through the same checks when it comes.
    true
}

/// **The only place in this program that performs anything.**
fn act(desk: &mut Desk, out: &Listeners, p: Proposal) -> bool {
    //: **On the surface it was offered on**, and the switch put back however this returns — a click
    //: offered on a web page and confirmed after flipping to the desktop must not look for a desktop
    //: widget. A guard rather than a wrapper, so `perform` stays inside `act` where the rule wants it.
    let _surface = desk.on_web.for_now(p.web && desk.web.is_some());
    //: **The rule again at the press**, because the window in front can change between an offer
    //: and a button — a row offered in the editor and pressed after switching to the messenger.
    let window = desk.session.eyes.window().1;
    if argmind_core::rule::applies(&desk.rules, &window, &p.effect.kind)
        == Some(argmind_core::rule::Then::Never) {
        out.send("dropped", &p.token.to_string());
        out.send("note", &argmind_core::rule::told(&desk.rules, &window, &p.effect.kind));
        return false;
    }
    let before = (desk.session.ambiguous, desk.session.chose_too_late);
    let (no_field, not_landed) = (desk.session.nowhere_to_type, desk.session.not_landed);
    let secret_before = desk.session.secret;
    let pointing_before = desk.session.cannot_point;
    let (did, downgraded, refused) = desk.session.perform(std::slice::from_ref(&p.effect));
    //: **Hands that cannot press a point inside anything**, said as that.
    if desk.session.cannot_point > pointing_before {
        desk.session.history.push(("could not".into(), format!(
            "{} — these hands cannot press a point inside anything; nothing was done", label(&p.effect, None))));
        out.send("dropped", &p.token.to_string());
        out.send("note", "여기서는 칸 안의 한 점을 누를 수 없습니다");
        return false;
    }
    let after = (desk.session.ambiguous, desk.session.chose_too_late);
    //: **A password field is the hand-off**, not a failure: told as what it is, and written into
    //: the conversation as such — `remember_doing` would record it as *it had gone by then*.
    if desk.session.secret > secret_before {
        desk.session.remember_doing(&did, &[], &[], &[]);
        desk.session.history.push(("could not".into(),
            "type into a password field — the person types passwords themselves; nothing was done"
                .into()));
        out.send("dropped", &p.token.to_string());
        out.send("note", "비밀번호 칸이라 입력하지 않았습니다 — 비밀번호는 직접 입력해 주세요");
        return false;
    }
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
        return false;
    }
    if let Some(Did::Elsewhere { owner }) = desk.session.hands.last_done() {
        out.send("note", &match owner {
            Some(_) => format!("그 지점은 다른 프로그램({owner:?})의 것이라 누르지 않았습니다"),
            //: The pane's hands ask the page, and a page has no programs — something on it covered
            //: the target, a banner or a notice.
            None => "그 자리를 다른 것이 가리고 있어서 누르지 않았습니다".to_string(),
        });
        return false;
    }
    //: **Refused for a reason the screen gave**, said as that reason.
    if desk.session.nowhere_to_type > no_field {
        out.send("dropped", &p.token.to_string());
        out.send("note", "입력할 칸이 선택되어 있지 않아서 입력하지 않았습니다");
        return false;
    }
    if desk.session.not_landed > not_landed {
        out.send("dropped", &p.token.to_string());
        out.send("note", "그 동작이 화면에 닿지 않았습니다");
        return false;
    }
    if !refused.is_empty() {
        match why_refused((before.0, after.0), (before.1, after.1)) {
            //: **Gone between the offer and the button** — the case the whole re-resolve exists
            //: for, and a proper outcome rather than an error.
            Refusal::Gone => out.send("gone",
                &format!("{{\"token\":{},\"what\":\"{}\"}}", p.token, escape(&p.effect.target))),
            //: **There, and more than once.** `dropped` takes the row off without the page's
            //: *그새 사라짐*, which would be false, and the question goes up from a fresh look —
            //: the list it offers is the screen as it is now, not as it was when the offer was made.
            why => {
                out.send("dropped", &p.token.to_string());
                out.send("note", if why == Refusal::Outrun {
                    "고르시는 동안 화면이 바뀌어 누르지 않았습니다 — 지금 화면으로 다시 여쭙니다"
                } else {
                    "제안한 뒤에 같은 이름이 더 생겨서 누르지 않았습니다 — 어느 것인지 여쭙니다"
                });
                match desk.session.find(&p.effect.target, &p.effect.role) {
                    argmind_core::session::Found::Several(same) =>
                        ask_which(desk, out, p.effect.clone(), &same),
                    //: It settled back to one, or went, in the moment between the refusal and
                    //: this look. Either way there is nothing to ask and the person is told.
                    argmind_core::session::Found::One(_) => out.send("note",
                        "다시 보니 하나뿐입니다 — 필요하면 다시 말씀해 주세요"),
                    argmind_core::session::Found::Gone => out.send("gone",
                        &format!("{{\"token\":{},\"what\":\"{}\"}}", p.token,
                                 escape(&p.effect.target))),
                }
            }
        }
        return false;
    }
    let confirmed = match did.first().and_then(|d| d.confirmed) {
        Some(b) => b.to_string(),
        None => "null".into(),
    };
    out.send("ran", &format!("{{\"token\":{},\"label\":\"{}\",\"confirmed\":{confirmed}}}",
                             p.token, escape(&p.label)));
    true
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
        //: **What it actually did the day it was asked for**, in the window that asks.
        "initiative" => String::from(
            "argmind 가 스스로 떠올린 클릭도 묻지 않고 실행합니다.\n\n\
             · 창이 바뀔 때 무언가 누르고 싶어 한 것이 28번 중 5~9번이었습니다 (측정)\n\
             · 하루 동안 같은 버튼(Move all to Desktop — 파일 이동)을 여러 번 꺼냈습니다\n\
             · 누르기 직전에 화면을 다시 읽고, 대상이 없거나 단계가 안 되면 멈춥니다\n\
             · 규칙의 '하지 않음'과, 그 지점이 다른 것을 가리킬 때는 여전히 멈추거나 묻습니다\n\n\
             스스로 판단한 것도 실행할까요?"),
        //: **What it sees, what it costs, and that it never presses by itself** — measured, in the
        //: window that asks, the way `acting` prints its numbers.
        "looking" => String::from(
            "목록에 없는 곳은 그림으로 찾습니다.

\n             · 원격 화면, 문자표의 글자 격자처럼 목록이 속을 못 보는 칸 안을 눌러 달라고 할 때만 봅니다
\n             · 그 창의 그림을 이 PC의 시각 모델에 보여 줍니다 — 밖으로 나가지 않고, 저장하지 않습니다
\n             · 찾은 곳은 항상 실행을 눌러야 누릅니다 — 확인 없이 실행이 켜져 있어도 마찬가지입니다
\n             · 실제 문자표에서 글자 10개 중 8개를 맞혔습니다 — 옆 칸을 누를 때가 있습니다 (측정)
\n             · 그래픽 메모리를 약 6 GB 더 씁니다 — argmind 가 스스로 일하는 중이면 다시 올리느라 10초쯤 걸립니다

\n             그림으로 찾아볼까요?"),
        //: **Found by the test below, not by reading.** It had been asking with the bare default
        //: since it became a setting on 2026-09-17 — the permission to make a noise in the room.
        //: What it says is what `argmind-rules` holds: a remark argmind makes on its own is text.
        "aloud" => String::from(
            "argmind 가 대답을 소리 내어 말합니다.\n\n\
             · 말을 건 것에 대한 대답만 소리로 나옵니다 — 먼저 거는 말은 글로만 나옵니다\n\
             · 신경망 목소리가 떠 있으면 그 목소리로, 아니면 윈도우 기본 음성으로 말합니다\n\
             · 목소리는 *장치와 목소리* 에서 고르고, 고르면 한 번 들려줍니다\n\
             · 끄면 대답은 화면에 글로만 나옵니다\n\n\
             소리 내어 말할까요?"),
        //: **Written at the camera's standard, because it asks for more than the camera does.**
        //: Until 2026-09-29 these two fell through to the default and asked a bare *…을(를)
        //: 허용할까요?* — and one of them writes something about a person to disk.
        "attention" => String::from(
            "카메라로 고개가 화면 쪽을 향하는지 봅니다.\n\n\
             · 카메라가 켜져 있을 때만 동작합니다. 켜기만 해서는 아무것도 열지 않습니다\n\
             · 1초마다 얼굴의 눈·코 위치 다섯 점을 찾습니다 — 모델을 부르지 않고 약 0.1초\n\
             · **고개** 방향입니다. 고개는 두고 눈만 돌리면 보고 있는 것으로 읽습니다\n\
             · 아무것도 저장하지 않습니다. 결과는 위 줄에 *화면 쪽을 봄* 처럼 보이고 곧 사라집니다\n\
             · 모델에게는 보내지 않습니다 — 보내면 *고개를 돌리셨네요* 라고 말하는 것을 쟀습니다\n\n\
             켤까요?"),
        "faces" => String::from(
            "카메라에 보인 얼굴이 전에 본 사람인지 기억합니다.\n\n\
             · **사진은 저장하지 않습니다.** 사람마다 숫자 128개와 처음·마지막 본 날짜만 남깁니다\n\
             · 그 숫자로 얼굴을 되살릴 수는 없습니다\n\
             · 카메라 앞을 지나간 누구든 기록될 수 있습니다 — 그래서 최대 20명에서 멈춥니다\n\
             · 처음 보는 얼굴이면 *누구신가요?* 라고 한 번만 묻습니다. 넘어가도 됩니다\n\
             · 설정에서 이름을 고치거나 한 명씩, 또는 전부 지울 수 있습니다 (%LOCALAPPDATA%\\\
argmind\\faces.json)\n\n\
             기억할까요?"),
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
    let kept: Vec<String> = mind.kept.iter().map(|k| format!("\"{}\"", escape(k))).collect();
    //: **The list the page offers is the list the program accepts**, sent rather than copied into
    //: the page, so a language cannot be offered that `/language` would refuse.
    let languages: Vec<String> = argmind_core::mind::LANGUAGES.iter()
        .map(|l| format!("{{\"code\":\"{}\",\"name\":\"{}\"}}", l.code, escape(l.name))).collect();
    format!("{{\"most\":{},\"aim\":\"{}\",\"line\":\"{}\",\"dials\":[{}],\"kept\":[{}],\
             \"language\":\"{}\",\"languages\":[{}]}}",
            argmind_core::mind::MOST, escape(&mind.aim),
            escape(&mind.line().unwrap_or_default()), rows.join(","), kept.join(","),
            escape(&mind.language), languages.join(","))
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
pub const DID_NOT_FOLLOW: &str = crate::spoken::DID_NOT_FOLLOW[0];

/// The words a person gets: the model's own, unless it just mirrored them.
pub fn spoken(intent: &argmind_core::intent::Intent) -> String {
    spoken_in(intent, "")
}

/// **[`spoken`], in the language the person set** — the one sentence in a reply the program writes
/// itself, so it follows the setting the model was given. Korean when nothing is set, as it was.
pub fn spoken_in(intent: &argmind_core::intent::Intent, language: &str) -> String {
    if intent.parroted {
        return crate::spoken::say(&crate::spoken::DID_NOT_FOLLOW, language, &[]);
    }
    intent.say.clone()
}

/// **What the recogniser is told to hear**: the language set, or Korean when none is — which is
/// what it was always told, since a recogniser left to guess a two-second clip guesses wrong.
pub fn hearing_language(language: &str) -> String {
    if argmind_core::mind::is_a_language(language) { language.to_string() } else { "ko".to_string() }
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
        //: Same shape as `faces`: nothing opens, nothing downloads. The camera decides whether
        //: there is a picture and `face-models` whether the files are here.
        "attention" => switch_attention(desk, out, true),
        //: **Nothing starts and nothing installs.** The standing answer *is* the state, so all this
        //: does is tell the page the switch is on — which is why `Job::Acting` writes a 거부 when
        //: it is turned off rather than flipping a field: there is no field.
        "acting" => {
            out.send("acting", "1");
            out.send("note", "이제 확인 없이 실행합니다 — 설정에서 되돌릴 수 있습니다");
        }
        "initiative" => {
            out.send("initiative", "1");
            out.send("note", "이제 스스로 판단한 것도 실행합니다 — 설정에서 되돌릴 수 있습니다");
        }
        //: **Nothing downloads here**: the model is `vision-model`'s, a need of its own.
        "looking" => {
            out.send("looking", "1");
            out.send("note", "이제 목록에 없는 곳은 그림으로 찾아봅니다 — 찾은 곳은 실행을 눌러야 누릅니다");
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
    let want_attention = Arc::clone(&desk.want_attention);
    let facing_now = Arc::clone(&desk.facing_now);
    let face_seen_at = Arc::clone(&desk.face_seen_at);
    let wonders = Arc::clone(&desk.wonders);
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
            &mut |seen| {
                if let Some(why) = &seen.broke {
                    out.send("note", why);
                }
                if seen.moved {
                    last_moved = Some(Instant::now());
                    *moved_at.lock().unwrap_or_else(|e| e.into_inner()) = last_moved;
                }
                if seen.face {
                    *face_seen_at.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
                }
                *facing_now.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some((seen.facing, Instant::now()));
                let what = camera_line(last_moved.map(|at| at.elapsed().as_secs_f64()),
                                       began.elapsed().as_secs_f64(), seen.facing);
                //: **To the person as well as to the model.** Until 2026-09-29 this line went to
                //: the blackboard only, which is read by the thinking tick and shown to nobody —
                //: so every camera feature was invisible by construction, and turning one on and
                //: seeing nothing was indistinguishable from turning on one that was broken.
                let here = present(
                    moved_at.lock().ok().and_then(|m| m.map(|t| t.elapsed().as_secs_f64())),
                    face_seen_at.lock().ok().and_then(|f| f.map(|t| t.elapsed().as_secs_f64())),
                    crate::platform::seconds_since_input());
                out.send("sense", &format!("{{\"camera\":\"{}\",\"present\":{here}}}",
                                           escape(&what)));
                senses.saw("카메라", what, seen.took);
            },
            &|| want_attention.load(Ordering::SeqCst) && crate::faces::here(),
            //: Three things have to be true: the models on disk, the switch on, and — implicitly —
            //: a camera, since this closure only runs inside its loop.
            &|| want_faces.load(Ordering::SeqCst) && crate::faces::here(),
            &mut |print| {
                let mut roster = people.lock().unwrap_or_else(|e| e.into_inner());
                match roster.saw(print, now_seconds()) {
                    crate::people::Saw::Fresh { key } => {
                        //: **Written the moment somebody new arrives**, not at shutdown: a crash
                        //: between the two would lose exactly the row that was worth keeping.
                        if let Err(why) = roster.write() {
                            eprintln!("  faces: 명부를 쓰지 못했습니다 — {why}");
                        }
                        out.send("people", &people_json(&roster));
                        //: **Asked rather than filed.** The owner, the same day: 모르는건
                        //: 물어보는 기능도 있어야함. Writing an unnamed row into a settings panel
                        //: and waiting for somebody to go looking for it is filing, not asking.
                        //: **About the row just made**, not the first unnamed one on the list.
                        //: Until 2026-09-30 this asked `worth_asking()`, which returns the oldest
                        //: unnamed row — and with stray rows in the roster that was a stranger
                        //: from last week rather than the face in front of the camera now.
                        let raised = {
                            let mut open = wonders.lock().unwrap_or_else(|e| e.into_inner());
                            open.ask(crate::wonders::About::Who(key)).map(wonder_json)
                        };
                        match raised {
                            Some(json) => out.send("wonder", &json),
                            //: One question at a time, so a second stranger while the first is
                            //: unanswered is a note. They stay in the roster and the settings list
                            //: is still where a name can be typed.
                            None => out.send("note", "처음 보는 얼굴을 기억했습니다 — \
설정에서 이름을 붙일 수 있습니다"),
                        }
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
                    //: One frame's word, held for a second to agree. Nothing to say yet.
                    crate::people::Saw::Unsure => {}
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
        //: **The reading goes with the device.** A 눈치 that went on holding remarks because the last
        //: thing a closed camera said was *turned away* would be a mute nobody pressed.
        *facing_now.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *face_seen_at.lock().unwrap_or_else(|e| e.into_inner()) = None;
        out.send("sense", "null");
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

/// **What the camera puts on the blackboard**, as one sentence.
///
/// **One slot, because it is one device and one window.** Two slots would age separately and let a
/// reader pair a movement reading with a head reading from a different tick — and `senses.rs` puts
/// an age on a slot precisely so that cannot happen quietly.
///
/// **Nothing is added at rest**, which is the whole ground this was allowed on: prompt edits here
/// are 1 for 7, and `Facing::NotLookedFor` renders nothing, so with the switch off the line is byte
/// for byte what it was before any of this existed. A test below holds that.
///
/// Lifted out of the camera thread's closure for one reason: a claim about what a prompt does at
/// rest is worth nothing if it cannot be run.
fn camera_line(moved_secs_ago: Option<f64>, since_open: f64,
               facing: crate::platform::Facing) -> String {
    let mut what = match moved_secs_ago {
        Some(ago) => format!("마지막 움직임 {ago:.0}초 전"),
        None => format!("{since_open:.0}초째 아무 움직임 없음"),
    };
    match facing {
        crate::platform::Facing::NotLookedFor => {}
        //: **Nobody to read**, which is not the same as a head that could not be read.
        crate::platform::Facing::NoFace => what.push_str(" · 얼굴 안 보임"),
        crate::platform::Facing::AtIt => what.push_str(" · 화면 쪽을 봄"),
        crate::platform::Facing::Aside => what.push_str(" · 고개를 돌림"),
        //: **Said as itself.** It correlates with a turned head and is not it — 0 and 1 of 95
        //: against 22 and 48 of 96 — and a word that claimed otherwise would be a finding nobody
        //: made. `docs/findings/attention.md`.
        crate::platform::Facing::Unreadable => what.push_str(" · 고개 방향을 읽지 못함"),
    }
    what
}

/// Knowing where the head is pointed, on or off.
///
/// **It does not touch the camera and it writes nothing down.** What reaches the blackboard is a
/// word with an age and it evaporates when the device closes — which is why this sits above `faces`
/// and below `camera` in how much it gives away, and why it is still a key of its own: somebody may
/// want the light on for movement without argmind working out where they are looking.
///
/// **And it does not gate anything.** `docs/findings/attention.md` measured that a turned head
/// reads thirty times a facing one; nothing has measured that the fact should change what argmind
/// does on a tick. Reporting it and acting on it are two steps and only the first is earned.
fn switch_attention(desk: &mut Desk, out: &Listeners, on: bool) {
    desk.want_attention.store(on, Ordering::SeqCst);
    out.send("attention", if on { "1" } else { "0" });
    out.send("note", if !on {
        "고개 방향은 더 보지 않습니다"
    } else if !crate::faces::here() {
        "얼굴 모델이 없습니다 — 설정의 필요한 것에서 받으세요"
    } else if !desk.eyeing.load(Ordering::SeqCst) {
        "카메라를 켜면 화면을 보고 있는지 알 수 있습니다"
    } else {
        "화면을 보고 있는지 봅니다 — 고개 방향이고, 시선은 아닙니다"
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

/// **Ask which of several identically named things was meant.** The question replaces the offer.
///
/// Called from two places and that is why it is a function: from `propose`, when the copies were
/// already there at offer time, and from `act`, when they appeared between the offer and the press.
/// **Neither caller can reach `act` again through here** — the only way from this question to a
/// click is a person answering it — which is what keeps the one path that presses anything free of
/// a loop on a screen that keeps changing.
fn ask_which(desk: &mut Desk, out: &Listeners, mut effect: Effect,
             same: &[argmind_core::contract::Widget]) {
    //: **A stale answer is not carried into a new question.** The count it was chosen against is
    //: exactly what just failed to match.
    effect.which = None;
    let among: Vec<String> = same.iter().enumerate()
        //: **Coordinates, to a person.** The rule is that the *model* is never shown a rectangle;
        //: somebody looking at their own screen is exactly who a position helps, and it is the
        //: only thing that tells two identically named buttons apart. First cut — an ordinal and a
        //: point; whether that is enough to choose by is unmeasured.
        .map(|(i, w)| {
            let (x, y) = w.centre();
            format!("{}번째 · 화면 {x}, {y}", i + 1)
        }).collect();
    let what = label(&effect, None);
    desk.session.remember_doing(&[], &[], std::slice::from_ref(&effect), &[]);
    let token = desk.proposals.offer(effect, what.clone(), String::new(), desk.on_web.get());
    let raised = {
        let mut open = desk.wonders.lock().unwrap_or_else(|e| e.into_inner());
        open.ask(crate::wonders::About::WhichOne { token, what, among }).map(wonder_json)
    };
    match raised {
        Some(json) => out.send("wonder", &json),
        //: One question at a time. A second ambiguous click while the first is unanswered is told
        //: and **declined**, so its token does not sit in `Proposals` waiting for a press that has
        //: no row to be pressed from.
        None => {
            desk.proposals.decline(token);
            out.send("note", "어느 것인지 물어볼 것이 이미 하나 있어서 이번 건은 넘어갑니다");
        }
    }
}

/// **Why `perform` refused**, read off the counters it keeps rather than guessed from the effect.
///
/// Three reasons and they want three different sentences. Until 2026-09-28 `act` reported every
/// refusal as `gone` — which the page prints as *그새 사라짐* — so somebody who had just answered
/// *which `Open`?* and been outrun by the screen was told the button had vanished, when there were
/// in fact two of it. The core counted them apart from the day `Several` existed; the one place that
/// speaks to a person folded them back together, which is pitfall 5 arriving in the newest code.
#[derive(Debug, PartialEq)]
pub enum Refusal {
    /// Not on screen any more — the case re-resolving exists for.
    Gone,
    /// It was one thing at offer time and is several now.
    Copies,
    /// A person chose one, and by the press the count had changed under them.
    Outrun,
}

/// Compare the counters from before and after one `perform`.
pub fn why_refused(ambiguous: (u32, u32), late: (u32, u32)) -> Refusal {
    if late.1 > late.0 {
        Refusal::Outrun
    } else if ambiguous.1 > ambiguous.0 {
        Refusal::Copies
    } else {
        Refusal::Gone
    }
}

/// Carry out whatever one answered question meant.
///
/// **Two kinds and one slot.** A name is typed and a copy is chosen; what they share is that the
/// deterministic side found a gap it could name, which is why they are one mechanism.
fn which_one(desk: &mut Desk, out: &Listeners, about: crate::wonders::About, said: String) {
    out.send("wonder", "null");
    match about {
        crate::wonders::About::Who(key) => {
            let mut roster = desk.people.lock().unwrap_or_else(|e| e.into_inner());
            //: **Marked whether or not a name came back.** What is recorded is that argmind asked,
            //: so somebody who skipped is not asked again tomorrow.
            roster.mark_asked(&key);
            let named = !said.is_empty() && roster.call(&key, &said);
            let _ = roster.write();
            out.send("people", &people_json(&roster));
            if named {
                out.send("note", &format!("{said} 님으로 기억하겠습니다"));
            }
        }
        crate::wonders::About::WhichOne { token, among, .. } => {
            //: **Skipping declines the action.** The proposal is taken off the board the same way
            //: 아니 takes it off, so nothing is left waiting for a press that will not come.
            let Ok(index) = said.parse::<usize>() else {
                desk.proposals.decline(token);
                out.send("note", "어느 것인지 고르지 않아 넘어갑니다");
                return;
            };
            if index >= among.len() {
                desk.proposals.decline(token);
                out.send("note", "그런 선택지가 없습니다");
                return;
            }
            //: **Answering is the confirmation, and it still goes through `Proposals`** — so a
            //: token that has gone stale cannot confirm a later action, which is the guarantee that
            //: exists whether a person pressed a button or picked a copy.
            if let Answered::Run(mut p) = desk.proposals.confirm(token) {
                p.effect.which = Some(argmind_core::intent::Among {
                    index, of: among.len(),
                });
                let _ = act(desk, out, p);
            }
        }
        //: **Only the one button is yes**; skipping, or anything else, is no.
        crate::wonders::About::Press { ask, what, .. } => {
            if let Some(i) = desk.screen_asks.iter().position(|(a, _)| *a == ask) {
                let (_, tx) = desk.screen_asks.remove(i);
                let _ = tx.send(if said == "0" { Ok(()) } else {
                    out.send("note", &format!("{what} — 하지 않습니다"));
                    Err("사람이 하지 않기로 했다 — 하지 않았다".into())
                });
            }
        }
    }
}

/// One open question, for the page.
fn wonder_json(w: &crate::wonders::Wonder) -> String {
    format!("{{\"id\":{},\"asked\":\"{}\",\"hint\":\"{}\",\"among\":[{}]}}",
            w.id, escape(&w.about.asked()), escape(w.about.hint()),
            w.about.among().iter().map(|a| format!("\"{}\"", escape(a)))
                .collect::<Vec<_>>().join(","))
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
    let moved = desk.moved_at.lock().ok()?.map(|t| t.elapsed().as_secs_f64());
    let face = desk.face_seen_at.lock().ok()?.map(|t| t.elapsed().as_secs_f64());
    Some(present(moved, face, crate::platform::seconds_since_input()))
}

/// **Is somebody there**, from how long ago something moved and how long ago a face was seen.
///
/// **Movement alone went quiet on a person sitting still**, and `camera.rs` said it would: the
/// threshold was measured with somebody moving on purpose, and *a person reading without shifting
/// is below it*. Five minutes of reading and `notice` stopped for good — the owner, 2026-09-29,
/// tried every camera feature and saw nothing, and this gate returning *nobody here* before any
/// trigger was polled is the one thing that silences all of them at once.
///
/// **A face is the stronger evidence and it is measured**: over the `attending` run, with a person
/// at the desk, YuNet found a face on **382 of 382** passes — not one miss, looking or turned. It
/// only counts when a face pass is running, which is when *화면을 보고 있는지 알기* or *본 얼굴
/// 기억하기* is on; with both off this is movement alone, exactly as before.
///
/// **And somebody using the keyboard or the mouse is there**, whatever the camera saw — the owner,
/// 2026-10-21: the camera read *601초째 아무 움직임 없음 · 얼굴 안 보임* while they typed to argmind,
/// and a finished errand sat unreported because the room was *empty*. The camera can be pointed
/// elsewhere, and a person can face another monitor; input cannot be produced by nobody.
pub fn present(moved_ago: Option<f64>, face_ago: Option<f64>, input_ago: Option<f64>) -> bool {
    let recent = |ago: Option<f64>| ago.is_some_and(|s| s < AWAY_AFTER);
    recent(moved_ago) || recent(face_ago) || recent(input_ago)
}

fn notice(desk: &mut Desk, out: &Listeners, now: f64) {
    if !desk.noticing {
        return;
    }
    //: **Not while argmind looks at its pane.** A desktop window change would be planned against
    //: the pane's list — the owner's session of 2026-11-03 had a switch to Snipping Tool turn into
    //: *click the Naver search box, type 일본* — the same mistake the thinking tick had, paused for
    //: the same reason.
    if desk.on_web.get() {
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
    //: **A held remark first**, so somebody who looks back hears what argmind was waiting to say
    //: before anything new is noticed on top of it.
    release_held(desk, out);
    //: **The desktop, always** — see `surface.rs`: watching the pane would be argmind noticing
    //: itself.
    let after = desk.session.eyes.desk.foreground();
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
        //: A watch cannot be parroted — nobody said anything to mirror — but it goes through the
        //: same call so there is one place that decides what gets said.
        //: **Nothing to do, nothing said** — the owner's choice on 2026-10-21, after eleven hours of
        //: *아, VS Code에서 작업 중이시군요!* on every window change, 37 of them. The model speaks on
        //: every one it is given (224 of 224 in `turned`), so the silence is the program's: a window
        //: change is remarked on only when the remark comes with something to do.
        if intent.effects.is_empty() {
            continue;
        }
        let reply = spoken_in(&intent, &speaking(desk));
        //: **The fourth identical remark is not made.** Measured: over eight window changes it
        //: spoke 8 times out of 8 and 6 of those were one sentence with the name swapped. Telling
        //: it to be quiet in the prompt bought no silence and started it wanting to click things,
        //: so this is caught on the way out — the answer that worked for the parrot.
        if !reply.is_empty() && argmind_core::converse::said_before(&reply, &desk.lately_noticed) {
            continue;
        }
        //: **Into the conversation only now that something comes of it.** `noticed` opens a turn and
        //: the prompt keeps twelve, so every silent window change used to push one of the person's
        //: own turns out — 97 notices against a handful of words from the owner in eleven hours, on
        //: 2026-10-21, when they said argmind did not remember what they had talked about. A window
        //: change argmind did nothing with is not part of the conversation.
        desk.session.history.push(("noticed".into(), trigger.text.clone()));
        //: **눈치 decides when, not whether.** At 0 this is always *now* and nothing below changed.
        //: Above 0 a remark to somebody turned away, or too soon after the last one, is held
        //: rather than said — and the program does that, because the model measurably cannot.
        match crate::tact::now(desk.nunchi, evidence(desk), since_last(desk)) {
            crate::tact::Now::Say => say_unprompted(desk, out, &intent, &reply),
            crate::tact::Now::Hold => {
                let window = desk.session.eyes.desk.window().1;
                desk.held = Some((intent, window, Instant::now()));
                //: **Holding is shown.** 눈치's only effect is a remark not made, and a dial whose
                //: effect is invisible is one nobody can tell from a broken one — which is what
                //: the owner reported on 2026-09-29.
                out.send("held", "1");
            }
        }
    }
}

/// What the camera says, reduced to what `tact` needs. **Only a fresh, positive *turned away*
/// counts**; ten seconds is three camera windows, past which the reading is not evidence of now.
fn evidence(desk: &Desk) -> crate::tact::Evidence {
    match *desk.facing_now.lock().unwrap_or_else(|e| e.into_inner()) {
        Some((crate::platform::Facing::Aside, at)) if at.elapsed().as_secs_f64() < 10.0 =>
            crate::tact::Evidence::TurnedAway,
        _ => crate::tact::Evidence::NotTurnedAway,
    }
}

fn since_last(desk: &Desk) -> Option<f64> {
    desk.last_remark.map(|at| at.elapsed().as_secs_f64())
}

/// **Say a held remark if 눈치 now allows it**, drop it if it is about a screen that is gone.
fn release_held(desk: &mut Desk, out: &Listeners) {
    let Some((intent, window, at)) = desk.held.take() else { return };
    let same = desk.session.eyes.desk.window().1 == window;
    match crate::tact::held(desk.nunchi, evidence(desk), since_last(desk), same,
                            at.elapsed().as_secs_f64()) {
        crate::tact::Held::Keep => desk.held = Some((intent, window, at)),
        crate::tact::Held::Drop => out.send("held", "0"),
        crate::tact::Held::Release => {
            out.send("held", "0");
            let reply = spoken_in(&intent, &speaking(desk));
            //: Checked again: something else may have been said while this one waited.
            if !reply.is_empty()
                && !argmind_core::converse::said_before(&reply, &desk.lately_noticed) {
                say_unprompted(desk, out, &intent, &reply);
            }
        }
    }
}

/// **The one place an unprompted remark is actually made**, whether now or after waiting.
fn say_unprompted(desk: &mut Desk, out: &Listeners, intent: &argmind_core::intent::Intent,
                  reply: &str) {
    if reply.is_empty() {
        for effect in &intent.effects {
            if !propose(desk, out, effect.clone(), intent.sure, false) {
                break;
            }
        }
        return;
    }
    desk.lately_noticed.push(reply.to_string());
    //: **A short ring.** A person who comes back to a window ten minutes later should hear it
    //: again; the rule is against a run of them, not against ever saying it twice.
    if desk.lately_noticed.len() > REMARKS_REMEMBERED {
        desk.lately_noticed.remove(0);
    }
    desk.last_remark = Some(Instant::now());
    desk.session.history.push(("you".into(), reply.to_string()));
    out.send("say", reply);
    //: Not spoken aloud. A proposal waits behind a button; a sentence nobody asked for waits
    //: behind nothing, which is why speaking is tied to having been asked.
    for effect in &intent.effects {
        if !propose(desk, out, effect.clone(), intent.sure, false) {
            break;
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

/// What the thinking thread is handed: a note to revise, or an errand to step.
enum Letter {
    Note(Chew),
    /// The errand as it stands, and today's date — read by the caller because the core names no
    /// clock.
    Errand(argmind_core::errand::Errand, String),
}

/// A one-deep letterbox to the thinking thread.
///
/// **Dropped rather than queued.** A `Sender` would pile up thoughts about screens that have gone,
/// and the oldest of those is the least worth having. This holds at most one and the thinker takes
/// it when it is free; a tick arriving while one is waiting replaces it.
#[derive(Clone)]
struct Letterbox(Arc<Mutex<Option<Letter>>>, Arc<Condvar>);

impl Letterbox {
    fn new() -> Self {
        Letterbox(Arc::new(Mutex::new(None)), Arc::new(Condvar::new()))
    }

    /// Leave one, replacing whatever was there.
    fn try_send_chew(&self, chew: Chew) {
        self.post(Letter::Note(chew));
    }

    fn post(&self, letter: Letter) {
        if let Ok(mut slot) = self.0.lock() {
            *slot = Some(letter);
            self.1.notify_one();
        }
    }

    /// Wait for one. `None` only if the lock is poisoned, which means the program is going down.
    fn wait(&self) -> Option<Letter> {
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
    //: **Not while argmind is looking at its own pane.** The note is about the person's desktop,
    //: and `seen` is then the pane's list: the owner's first session had *아까부터 link "메일" 가
    //: 눈에 걸린다* raised and offered as a click — argmind thinking about its own page, the defect
    //: the watch had once when it read argmind's transcript.
    if desk.on_web.get() {
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
    if desk.turn.in_flight() || desk.on_web.get() {
        return;
    }
    let Some(worth) = desk.session.note.worth_raising().cloned() else { return };
    //: **Raised once a session, not once per three looks.** The owner's eleven hours on
    //: 2026-10-21 had *Move all to Desktop* raised every minute and a half, in Chrome and in a
    //: search page as much as in the dialog it belonged to: putting a judgement down did not stop
    //: the next ticks choosing the same widget and counting to three again.
    let key = (worth.name.clone(), worth.role.clone());
    if desk.raised.contains(&key) {
        desk.session.note.let_go();
        return;
    }
    desk.raised.push(key);
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

    let reply = spoken_in(&intent, &speaking(desk));
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
        if !propose(desk, out, effect.clone(), intent.sure, false) {
            break;
        }
    }
}

/// **Take one step on the oldest open errand**, if it is time and nobody is being answered.
///
/// The same two rules every sense keeps: not while a turn is in flight, and abandoned on the
/// thinking thread the moment one begins. Not gated on the camera — working while nobody is there
/// is half the point; only *telling* waits for somebody.
fn step_errand(desk: &mut Desk, thinking: &Letterbox, now: f64) {
    if now - desk.stepped_at < crate::errands::STEP_EVERY || desk.turn.in_flight() {
        return;
    }
    let Some(e) = crate::errands::next_open(&desk.errands) else { return };
    desk.stepped_at = now;
    thinking.post(Letter::Errand(e.clone(), crate::platform::today()));
}

/// **Say one finished errand, or one that needs the person, when somebody is there to hear it.**
///
/// *There* is the camera's word when it has one — `somebody_about` — and 눈치's when it is turned
/// up: a report is an unprompted remark like any other, so it waits for a head turned back and
/// keeps 눈치's gap. With no camera and 눈치 at 0 it is said on the next tick, which is today's
/// behaviour for every other remark. **Waiting loses nothing**: `told` is on the list, on disk.
fn tell_errand(desk: &mut Desk, out: &Listeners) {
    let Some(i) = crate::errands::due(&desk.errands) else { return };
    if desk.turn.in_flight() || somebody_about(desk) == Some(false) {
        return;
    }
    if crate::tact::now(desk.nunchi, evidence(desk), since_last(desk)) == crate::tact::Now::Hold {
        return;
    }
    let code = speaking(desk);
    let errand = desk.errands[i].clone();
    let session = &mut desk.session;
    let said = crate::errands::report_in(&errand, &code, &mut |t: &str| work_words(session, t, &code));
    desk.errands[i].told = true;
    desk.last_remark = Some(Instant::now());
    //: **Into the history**, so *두 번째 거 좀 더 자세히* is a reply the next turn can read.
    desk.session.history.push(("you".into(), said.clone()));
    out.send("say", &said);
    keep_errands(desk, out);
}

/// Write the list and tell every page. **Every change goes to disk**, because a list that lives in
/// memory is a list that loses what somebody asked for when the window closes.
fn keep_errands(desk: &Desk, out: &Listeners) {
    if let Err(why) = crate::errands::keep(&desk.errands) {
        out.send("note", &why);
    }
    out.send("errands", &crate::errands::list_json(&desk.errands));
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
            desk.heard_last = Some(text.trim().to_string());
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
        format!("\"initiative\":{}",
                crate::decided::standing("initiative") == Some(Standing::Allowed)),
        format!("\"looking\":{}",
                crate::decided::standing("looking") == Some(Standing::Allowed)),
        //: **Held rather than read**, unlike `acts`: a standing 허용 says the camera *may* be on
        //: and this says it *is*, and they come apart when the device fails to open.
        format!("\"watching_you\":{}", desk.eyeing.load(Ordering::SeqCst)),
        format!("\"noticed\":{}", desk.noticed),
        format!("\"attending\":\"{}\"", escape(&desk.session.eyes.window().1)),
        format!("\"waiting\":[{}]", waiting.join(",")),
        //: **Sent with the page rather than only as an event**, so a reload does not lose a
        //: question nobody has answered. The event is for the question that arrives while somebody
        //: is looking; this is for the one that was already there.
        format!("\"wonder\":{}",
                desk.wonders.lock().unwrap_or_else(|e| e.into_inner()).open()
                    .map(wonder_json).unwrap_or_else(|| "null".into())),
        format!("\"attending_to_you\":{}", desk.want_attention.load(Ordering::SeqCst)),
        format!("\"nunchi\":{}", desk.nunchi),
        format!("\"remembering\":{}", desk.want_faces.load(Ordering::SeqCst)),
        format!("\"face_models\":{}", crate::faces::here()),
        format!("\"people\":{}",
                people_json(&desk.people.lock().unwrap_or_else(|e| e.into_inner()))),
        format!("\"errands\":{}", crate::errands::list_json(&desk.errands)),
        format!("\"rules\":{}", argmind_core::rule::for_the_page(&desk.rules)),
        format!("\"has_web\":{}", desk.web.is_some()),
        format!("\"web\":{}", desk.on_web.get()),
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
    /// **argmind's own browser pane**, when there is a window to hold one — handed over by
    /// `argmind-app`. `None` for the console server and every test: no pane, desktop only.
    pub web: Option<crate::surface::WebPane>,
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
                .unwrap_or_else(|| "qwen3:8b".into()), web: None, }
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

    let How { dry, speak, hear, model_name, port: how_port, web } = how;

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
    //: **One switch for the whole program**, made before either thread that reads it.
    let on_web = crate::surface::OnWeb::default();
    {
        let (letters, turn, senses, jobs, model_name) =
            (letters.clone(), turn.clone(), senses.clone(), jobs.clone(), model_name.clone());
        let (web, on_web) = (web.clone(), on_web.clone());
        std::thread::spawn(move || {
            let mut model = Local::new(&model_name);
            model.think = Some(false);
            model.keep_alive = "300s".into();
            while let Some(letter) = letters.wait() {
                //: Checked again here: a turn can have begun between the tick and this thread
                //: waking, and starting anyway would be the rule holding only when it is easy.
                if turn.in_flight() {
                    continue;
                }
                let chew = match letter {
                    Letter::Note(chew) => chew,
                    Letter::Errand(before, today) => {
                        //: **An empty screen, on purpose** — see `crate::errands`.
                        let prompt = argmind_core::errand::ask_for_a_step(&before, "", &today);
                        let carry_on = turn.carry_on();
                        let Ok(reply) = model.ask_while(&prompt, &[], &carry_on, &mut |_| {}) else {
                            continue;
                        };
                        //: Abandoned is not a step: the errand is untouched and the next tick
                        //: asks the same question again.
                        if reply.reason == "abandoned" {
                            continue;
                        }
                        let mut after = before.clone();
                        argmind_core::errand::read_step(&reply.text, &mut after);
                        //: **Looked up by the program, in the pane, when the errand is about now
                        //: or the check said it needs the web** — and only while nobody is using
                        //: the pane, because an errand must not navigate a page somebody is
                        //: reading. The model never chooses where to go: the search is the
                        //: errand's own words. Without a pane, or with it in use, nothing changes.
                        if after.wants_looking_up() && !on_web.get() {
                            if let Some(pane) = web.as_ref() {
                                let reader = argmind_env_browser::Pane::new(pane.wire.clone());
                                if let Ok((text, at)) = reader.look_up(&after.what) {
                                    after.looked_up(&text, &at);
                                }
                            }
                        }
                        let _ = jobs.send(Job::Stepped(before, after));
                        continue;
                    }
                };
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
    let agent_jobs = jobs.clone();
    let agent_model = model_name.clone();
    //: Everything COM touches lives on this thread and never leaves it.
    std::thread::spawn(move || {
        let eyes = Sight::new(watcher.clone()).expect("the eyes would not open");
        //: Unrestricted, and **the confirm button is what makes that safe** — the same trade
        //: `app.py` makes by inverting the unprompted flag. Dry unless asked otherwise, and how
        //: they are built is `platform`'s to decide: on Windows that now means control patterns
        //: rather than synthetic clicks, which is a fact about Windows and not about the app.
        let hands = platform::hands(dry, &watcher);
        let mut model = Local::new(&agent_model);
        model.think = Some(false);
        //: **Each token's probability, for [`SURE_ENOUGH`].** Asked on this one model only: the
        //: thinking thread and the errands have no actions to be sure of.
        model.logprobs = true;
        let (voice, voice_is) = if speak { pick_voice() } else { (None, "off") };
        let ear = hear.then(|| (Ear::new(), Whisper::default()));
        let (eyes, hands, on_web) =
            crate::surface::surfaces(eyes, hands, web.as_ref(), !dry, on_web);
        let mut desk = Desk { session: Session::new(eyes, hands, Model(model)), on_web,
                              web: web.clone(),
                              proposals: Proposals::default(), looked: None, dry,
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
                              want_attention: Arc::new(AtomicBool::new(
                                  crate::decided::standing("attention") == Some(Standing::Allowed))),
                              wonders: Arc::new(Mutex::new(Default::default())),
                              want_faces: Arc::new(AtomicBool::new(
                                  crate::decided::standing("faces") == Some(Standing::Allowed))),
                              speaks_to: Arc::clone(&agent_out),
                              senses: senses.clone(), turn: turn.clone(),
                              before: Vec::new(), lately_noticed: Vec::new(), noticed: 0,
                              nunchi: crate::tact::remembered(), held: None, last_remark: None,
                              errands: crate::errands::remembered(), stepped_at: 0.0,
                              rules: crate::rules::remembered(), heard_last: None,
                              raised: Vec::new(), kept_lines: 0,
                              working: Arc::new(AtomicBool::new(false)),
                              stop_work: Arc::new(AtomicBool::new(false)), jobs: agent_jobs,
                              own_work: Arc::new(AtomicBool::new(false)), after_own: None, own_halted: false,
                              own_untold: Vec::new(), heard_in: "ko".into(),
                              own_not_before: Instant::now() + OWN_PAUSE, own_misses: 0,
                              own_choosing: false, own_away: false, own_refused: None,
                              task_asks: Arc::new(Mutex::new(None)),
                              task_presses: Vec::new(), screen_asks: Vec::new(),
                              facing_now: Arc::new(Mutex::new(None)),
                              face_seen_at: Arc::new(Mutex::new(None)) };
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
        if let Some((_, whisper)) = desk.ear.as_mut() {
            whisper.language = Some(hearing_language(&desk.session.mind.language));
        }
        //: **The conversation where it left off**, and the prompt filling the model's context with
        //: it rather than stopping at twelve turns — see `crate::conversation`.
        desk.session.history = crate::conversation::remembered();
        desk.kept_lines = desk.session.history.len();
        //: And the language they were last talking in, so the first thing the program says after a
        //: restart is in it.
        if let Some(code) = desk.session.history.iter().rev().filter(|(who, _)| who == "them")
            .find_map(|(_, t)| argmind_core::tongue::of(t)) {
            desk.heard_in = code.to_string();
        }
        desk.session.context_tokens = Some(CONTEXT_TOKENS);
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
                    keep_conversation(&mut desk, &agent_out);
                    let now = started.elapsed().as_secs_f64();
                    notice(&mut desk, &agent_out, now);
                    //: **After noticing, not instead of it.** A window change is what a person is
                    //: owed an answer about; revising a note is what happens in the gaps.
                    reflect(&mut desk, &letters, now);
                    //: **Errands are not tied to watching**: somebody handed this over by name,
                    //: which is a stronger permission than a switch about the screen.
                    step_errand(&mut desk, &letters, now);
                    tell_errand(&mut desk, &agent_out);
                    //: **And the goal, when nobody needs argmind** — `crate::pursuit`.
                    pursue(&mut desk);
                    tell_own(&mut desk, &agent_out, false);
                    continue;
                }
                Err(_) => break,
            };
            //: Whatever the job, what it added to the conversation goes to the file after it.
            let keep_after = !matches!(job, Job::ForgetConversation);
            match job {
                Job::Said(text) => {
                    let text = text.trim();
                    think(&mut desk, &agent_out, text);
                    //: **Somebody talking to argmind is somebody back** — what it did by itself is
                    //: said after the answer, once.
                    tell_own(&mut desk, &agent_out, true);
                    //: **Offered, never kept.** A correction or a preference in the person's words
                    //: is a row asking 기억해 둘까요? — the word list is rough, and a rough list
                    //: that only offers costs one dismissed row when it is wrong.
                    if argmind_core::mind::worth_offering(text)
                        && !desk.session.mind.kept.iter().any(|k| k.trim() == text) {
                        agent_out.send("keep-offer", text);
                    }
                    //: **And a question about now gets the way to answer it**, when there is a pane:
                    //: an errand, which looks it up and reports back. Offered; the press decides.
                    //: **Not with the pane on**: there the program has already searched it,
                    //: and the row made a second errand that could not look anything up.
                    //: **Only without a pane**: with one, the question opened it and was looked
                    //: up, and a row would make a second errand.
                    if desk.web.is_none() && argmind_core::errand::asks_to_look_up(text) {
                        agent_out.send("lookup-offer", text);
                    }
                }
                Job::Confirm(n) => {
                    if let Answered::Run(p) = desk.proposals.confirm(n) {
                        let _ = act(&mut desk, &agent_out, p);
                    }
                    settle_press(&mut desk, &agent_out, n, "");
                }
                Job::Decline(n) => {
                    if let Answered::TurnedDown(p) = desk.proposals.decline(n) {
                        //: Pressing *no* is a thing the person said, and the next turn has to know.
                        desk.session.remember_doing(&[], &[], &[],
                                                    std::slice::from_ref(&p.effect));
                        agent_out.send("dropped", &p.token.to_string());
                    }
                    settle_press(&mut desk, &agent_out, n, "사람이 실행 줄을 거절했다 — 누르지 않았다");
                }
                Job::PaneLook(tx) => {
                    let _ = tx.send(pane_look(&mut desk));
                }
                Job::PanePress(effect, tx) => pane_press(&mut desk, &agent_out, effect, tx),
                Job::ScreenGate(window, kind, what, tx, own) =>
                    screen_gate(&mut desk, &agent_out, &window, &kind, &what, tx, own),
                Job::OwnPicked(picked) => own_picked(&mut desk, &agent_out, picked),
                Job::OwnEnded(done) => own_ended(&mut desk, &agent_out, done),
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
                //: **The same shape as 확인 없이 실행**: off is written down, on asks unless it
                //: was already answered.
                Job::Initiative(on) => match (on, crate::decided::standing("initiative")) {
                    (false, _) => {
                        if let Err(why) = crate::decided::keep("initiative", Some(Standing::Denied)) {
                            agent_out.send("note", &why);
                        }
                        agent_out.send("initiative", "0");
                        agent_out.send("note", "스스로 판단한 것은 다시 물어보고 실행합니다");
                        agent_out.send("decisions", &decisions_json());
                    }
                    (true, Some(Standing::Allowed)) => agent_out.send("initiative", "1"),
                    (true, _) => {
                        agent_out.send("initiative", "0");
                        agent_out.send("asking", &asking_json("initiative"));
                    }
                },
                //: **The same shape again**: off is written down, on asks unless it was answered.
                Job::Looking(on) => match (on, crate::decided::standing("looking")) {
                    (false, _) => {
                        if let Err(why) = crate::decided::keep("looking", Some(Standing::Denied)) {
                            agent_out.send("note", &why);
                        }
                        agent_out.send("looking", "0");
                        agent_out.send("note", "이제 그림은 보지 않습니다 — 목록에 있는 것만 누릅니다");
                        agent_out.send("decisions", &decisions_json());
                    }
                    (true, Some(Standing::Allowed)) => agent_out.send("looking", "1"),
                    (true, _) => {
                        agent_out.send("looking", "0");
                        agent_out.send("asking", &asking_json("looking"));
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
                Job::Language(code) => {
                    match crate::mind::speak_in(&code) {
                        Ok(mind) => {
                            let name = argmind_core::mind::language(&mind.language).map(|l| l.name);
                            agent_out.send("note", &match name {
                                Some(name) => format!("이제 {name}로 대답합니다"),
                                None => "이제 상대가 쓴 언어로 대답합니다".to_string(),
                            });
                            //: **And hears it.** The recogniser is told the same language, so a
                            //: person who chose English is not transcribed as Korean.
                            if let Some((_, whisper)) = desk.ear.as_mut() {
                                whisper.language = Some(hearing_language(&mind.language));
                            }
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
                Job::Errand(what) => {
                    match crate::errands::add(&mut desk.errands, &what) {
                        Ok(_) => {
                            agent_out.send("note", "맡았습니다 — 생각해서 알려 드릴게요");
                            keep_errands(&desk, &agent_out);
                        }
                        Err(why) => agent_out.send("note", why),
                    }
                }
                Job::ErrandAnswer(id, text) => {
                    if crate::errands::answer(&mut desk.errands, id, &text) {
                        agent_out.send("note", "알겠습니다 — 이어서 생각하겠습니다");
                        keep_errands(&desk, &agent_out);
                    }
                }
                Job::Web(on) => match desk.web.is_some() {
                    true => {
                        show_web(&mut desk, &agent_out, on);
                        agent_out.send("note", if on {
                            "이제 argmind 의 브라우저 칸을 보고 그 안에서만 누릅니다"
                        } else {
                            "다시 데스크톱을 봅니다"
                        });
                    }
                    false => {
                        agent_out.send("web", "0");
                        agent_out.send("note", "이 창에는 브라우저 칸이 없습니다 — argmind-app 으로 여세요");
                    }
                },
                Job::WebGo(typed) => match desk.web.as_ref() {
                    Some(pane) => {
                        let url = crate::surface::address(&typed);
                        let params = format!("{{\"url\":\"{}\"}}", escape(&url));
                        match pane.wire.call("Page.navigate", &params) {
                            //: Where it went is on the timeline; whether it was allowed to go is the
                            //: window's navigation check, not this.
                            Ok(_) => agent_out.send("note", &format!("브라우저 칸: {url}")),
                            Err(why) => agent_out.send("note", &format!("열지 못했습니다 ({why})")),
                        }
                        desk.session.seen.clear();
                    }
                    None => agent_out.send("note", "이 창에는 브라우저 칸이 없습니다"),
                },
                Job::ForgetConversation => {
                    desk.session.history.clear();
                    desk.kept_lines = 0;
                    match crate::conversation::forget() {
                        Ok(()) => agent_out.send("note", "대화 기억을 지웠습니다 — 파일도 비웠습니다"),
                        Err(why) => agent_out.send("note", &why),
                    }
                }
                Job::Activity => agent_out.frame("activity", &agent_out.timeline()),
                Job::ClearActivity => {
                    agent_out.forget_timeline();
                    agent_out.frame("activity", "[]");
                }
                Job::Worked(told) => {
                    use crate::spoken::{say, TASK_ASKS, TASK_NOT_FINISHED, TASK_STOPPED};
                    let code = speaking(&desk);
                    let said = match told {
                        Worked::Asks(q) => say(&TASK_ASKS, &code, &[&work_words(&mut desk.session, &q, &code)]),
                        Worked::Done(s) => work_words(&mut desk.session, &s, &code),
                        Worked::NotFinished(why) =>
                            say(&TASK_NOT_FINISHED, &code, &[&work_words(&mut desk.session, &why, &code)]),
                        Worked::Stopped => say(&TASK_STOPPED, &code, &[]),
                    };
                    desk.session.history.push(("you".into(), said.clone()));
                    agent_out.send("say", &said);
                }
                Job::Made(files) => show_made(&mut desk, &agent_out, &files),
                Job::Keep(text) => {
                    match crate::mind::keep(&text) {
                        Ok(mind) => {
                            desk.session.mind = mind;
                            agent_out.send("note", "기억했습니다");
                        }
                        Err(why) => agent_out.send("note", &why),
                    }
                    agent_out.send("mind", &mind_json(&desk.session.mind));
                }
                Job::Unkeep(i) => {
                    match crate::mind::unkeep(i) {
                        Ok(mind) => {
                            desk.session.mind = mind;
                            agent_out.send("note", "잊었습니다");
                        }
                        Err(why) => agent_out.send("note", &why),
                    }
                    agent_out.send("mind", &mind_json(&desk.session.mind));
                }
                Job::Rules(lines) => {
                    match argmind_core::rule::from_lines(&lines) {
                        Ok(rules) => match crate::rules::keep(&rules) {
                            Ok(()) => {
                                desk.rules = rules;
                                agent_out.send("note", "규칙을 저장했습니다");
                            }
                            Err(why) => agent_out.send("note", &why),
                        },
                        //: Refused whole and said why, and the page is sent what still stands.
                        Err(why) => agent_out.send("note", &why),
                    }
                    agent_out.send("rules", &argmind_core::rule::for_the_page(&desk.rules));
                }
                Job::ErrandDrop(id) => {
                    if crate::errands::drop(&mut desk.errands, id) {
                        keep_errands(&desk, &agent_out);
                    }
                }
                Job::Stepped(before, after) => {
                    if crate::errands::apply(&mut desk.errands, &before, after) {
                        keep_errands(&desk, &agent_out);
                        //: Said now if it can be, rather than up to a tick later.
                        tell_errand(&mut desk, &agent_out);
                    }
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
                Job::Nunchi(to) => match crate::tact::turn(to) {
                    Ok(n) => {
                        desk.nunchi = n;
                        //: A held remark under a lower setting may now be due; the next tick says
                        //: it. Turning to 0 releases on that tick too, which is *today* again.
                        agent_out.send("nunchi", &n.to_string());
                        agent_out.send("note", &format!("눈치 {n}"));
                    }
                    Err(why) => agent_out.send("note", &why),
                },
                Job::Attention(on) => match (on, crate::decided::standing("attention")) {
                    (false, _) => switch_attention(&mut desk, &agent_out, false),
                    (true, Some(Standing::Allowed)) =>
                        switch_attention(&mut desk, &agent_out, true),
                    (true, Some(Standing::Denied)) => agent_out.send("note",
                        "거부해 두셨습니다 — 설정에서 되돌릴 수 있습니다"),
                    (true, None) => {
                        agent_out.send("attention", "0");
                        agent_out.send("asking", &asking_json("attention"));
                    }
                },
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
                Job::Answer(id, text) => {
                    let answer = desk.wonders.lock().unwrap_or_else(|e| e.into_inner())
                        .answered(id, &text);
                    if let crate::wonders::Answered::Say(about, said) = answer {
                        which_one(&mut desk, &agent_out, about, said);
                    }
                }
                Job::Opened(tx) => {

                    let _ = tx.send(sse("hello", &hello(&desk, &agent_model)));
                    let _ = tx.send(sse("tasks", &cards_json()));
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
            if keep_after {
                keep_conversation(&mut desk, &agent_out);
            }
            //: **A marked picture lives as long as its row**, whichever job answered or cleared it.
            let waiting: Vec<u64> = desk.proposals.waiting().iter().map(|p| p.token).collect();
            crate::looking::keep_only(&waiting);
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
    //: **Before anything is read or served, the page included** — the page is what carries the
    //: token, so a request addressed to some other name must not get even that. `addressed_here`
    //: says why loopback alone was not enough.
    if !crate::addressed_here(&head) {
        let _ = stream.write_all(&respond("403 Forbidden", "text/plain", "not addressed to this machine"));
        return;
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
            let page = crate::interface::stamped(&INDEX.replace("__TOKEN__", token),
                                                 &crate::interface::remembered());
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
        //: **The page's own language**, written here and nowhere else: nothing the server says
        //: changes with it, so there is no job and no event — the page switches itself.
        Route::Interface(code) => {
            match crate::interface::choose(&code) {
                Ok(()) => { let _ = stream.write_all(&respond("204 No Content", "text/plain", "")); }
                Err(why) => { let _ = stream.write_all(&respond("400 Bad Request", "text/plain", &why)); }
            }
        }
        //: Checked against the fixed list here as well as in the job, like `/dial`.
        Route::Language(code) => {
            if code.is_empty() || argmind_core::mind::is_a_language(&code) {
                let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
                let _ = jobs.send(Job::Language(code));
            } else {
                let _ = stream.write_all(&respond("400 Bad Request", "text/plain", "no"));
            }
        }
        Route::Forget(key) => {
            if crate::decided::is_decidable(&key) {
                let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
                let _ = jobs.send(Job::Forget(key));
            } else {
                let _ = stream.write_all(&respond("400 Bad Request", "text/plain", "no"));
            }
        }
        //: **The one place the work screen's picture goes, and it goes to the page.** 204 while
        //: nothing is open there; `same` when the page already has this one — not a 304, which a
        //: browser may answer from a cache this response never asked it to keep.
        Route::WorkScreen(seen) => {
            let answer = match crate::platform::work_picture(seen) {
                None => respond("204 No Content", "text/plain", ""),
                Some((_, None)) => respond("200 OK", "text/plain", "same"),
                Some((seq, Some((titles, bmp)))) => crate::respond_bytes("200 OK", "image/bmp", &bmp,
                    &format!("X-Frame: {seq}\r\nX-Titles: {}\r\n", crate::encoded(&titles))),
            };
            let _ = stream.write_all(&answer);
        }
        //: **The one place the marked picture goes, and it goes to the page** — beside the row whose
        //: number it carries, while that row waits. A model never asks for this.
        Route::Looked(row) => {
            let answer = match crate::looking::shown(row) {
                Some(png) => crate::respond_bytes("200 OK", "image/png", &png, ""),
                None => respond("404 Not Found", "text/plain", "no"),
            };
            let _ = stream.write_all(&answer);
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
        Route::Initiative(on) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Initiative(on));
        }
        Route::Acting(on) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Acting(on));
        }
        Route::Looking(on) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Looking(on));
        }
        Route::Camera(on) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Camera(on));
        }
        Route::Nunchi(to) => {
            if to <= crate::tact::MOST {
                let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
                let _ = jobs.send(Job::Nunchi(to));
            } else {
                let _ = stream.write_all(&respond("400 Bad Request", "text/plain", "no"));
            }
        }
        Route::Attention(on) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Attention(on));
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
        //: Length-checked here like `/face/name`, and for the same reason: a page can send a long
        //: body by accident and the roster is not where that should be discovered.
        Route::Answer(id, text) => {
            if text.chars().count() <= 200 {
                let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
                let _ = jobs.send(Job::Answer(id, text));
            } else {
                let _ = stream.write_all(&respond("400 Bad Request", "text/plain", "no"));
            }
        }
        Route::Listen(down) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Listen(down));
        }
        //: Length-checked here like `/wonder`; `errands::add` checks again and says why.
        Route::Errand(what) => {
            if what.chars().count() <= crate::errands::MOST_CHARS {
                let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
                let _ = jobs.send(Job::Errand(what));
            } else {
                let _ = stream.write_all(&respond("400 Bad Request", "text/plain", "no"));
            }
        }
        Route::ErrandAnswer(id, text) => {
            if text.chars().count() <= crate::errands::MOST_CHARS {
                let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
                let _ = jobs.send(Job::ErrandAnswer(id, text));
            } else {
                let _ = stream.write_all(&respond("400 Bad Request", "text/plain", "no"));
            }
        }
        Route::Web(on) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Web(on));
        }
        Route::WebGo(typed) => {
            if !typed.trim().is_empty() && typed.chars().count() <= 2000 {
                let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
                let _ = jobs.send(Job::WebGo(typed));
            } else {
                let _ = stream.write_all(&respond("400 Bad Request", "text/plain", "no"));
            }
        }
        Route::ForgetConversation => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::ForgetConversation);
        }
        Route::Activity => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Activity);
        }
        Route::ClearActivity => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::ClearActivity);
        }
        //: **The person's button, not argmind's action** — nothing here goes through `act`, since
        //: what opens is a window the person asked for on their own desktop.
        Route::OpenWorkFolder => {
            let _ = stream.write_all(&match crate::platform::open_work_folder() {
                Ok(()) => respond("204 No Content", "text/plain", ""),
                Err(why) => respond("500 Internal Server Error", "text/plain; charset=utf-8", &why),
            });
        }
        Route::Keep(text) => {
            if text.chars().count() <= argmind_core::mind::KEPT_CHARS {
                let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
                let _ = jobs.send(Job::Keep(text));
            } else {
                let _ = stream.write_all(&respond("400 Bad Request", "text/plain", "no"));
            }
        }
        Route::Unkeep(i) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::Unkeep(i));
        }
        //: Thirty rules of eighty characters fit easily; anything far beyond is not a rule list.
        Route::Rules(lines) => {
            if lines.len() <= 16_384 {
                let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
                let _ = jobs.send(Job::Rules(lines));
            } else {
                let _ = stream.write_all(&respond("400 Bad Request", "text/plain", "no"));
            }
        }
        Route::ErrandDrop(id) => {
            let _ = stream.write_all(&respond("204 No Content", "text/plain", ""));
            let _ = jobs.send(Job::ErrandDrop(id));
        }
        Route::Unknown => {
            let _ = stream.write_all(&respond("404 Not Found", "text/plain", "no"));
        }
    }
}

#[cfg(test)]
mod question_tests {
    use super::plan_for;

    #[test]
    fn every_switch_asks_with_more_than_its_own_name() {
        //: **The default is one line — *…을(를) 허용할까요?* — and two switches were shipping on it.**
        //: One of them writes something about a person to disk. Every key a person can be asked
        //: about must say what it does before the buttons, the way the camera always did.
        //: **Only what can actually be asked.** A need with no fix raises no question — there is
        //: nothing to carry out — and in the headless build no need has one, because nothing here
        //: has run a fix off Windows. The first version of this test demanded a sentence for all
        //: fourteen keys and the headless build refused it: seven needs asking with the default,
        //: none of which a person could ever be shown. The rule is about questions a person sees.
        //: **The switches, named**, because in the headless build a switch and a need that is not
        //: listed there look identical — `korean-voice` and `vulkan` are Windows facts and vanish
        //: off Windows — and the second attempt at this test treated both as switches. A switch can
        //: always be asked; a need only when it is listed here with a fix.
        const SWITCHES: [&str; 9] =
            ["notice", "microphone", "aloud", "acting", "initiative", "looking", "camera", "attention", "faces"];
        for key in SWITCHES {
            assert!(crate::decided::DECIDABLE.contains(&key), "{key} is no longer decidable");
        }
        //: **And the list cannot quietly go stale** on the build the hook runs by default: every
        //: decidable key that is not a switch must be a need there. A new switch left off the list
        //: above would otherwise be skipped as an unlisted need and never checked.
        #[cfg(all(windows, not(feature = "headless")))]
        for key in crate::decided::DECIDABLE {
            if !SWITCHES.contains(&key) {
                assert!(crate::needs::by_key("qwen3:8b", key).is_some(),
                        "{key} is neither a switch nor a need — add it to SWITCHES");
            }
        }
        let askable = |key: &&&str| SWITCHES.contains(key)
            || crate::needs::by_key("qwen3:8b", key).is_some_and(|need| need.fix.is_some());
        //: Collected rather than stopping at the first, so one run names every gap.
        let thin: Vec<String> = crate::decided::DECIDABLE.iter()
            .filter(askable)
            .map(|key| (key, plan_for(key)))
            .filter(|(_, plan)| plan.lines().filter(|l| !l.trim().is_empty()).count() < 3
                                || !plan.trim_end().ends_with("까요?"))
            .map(|(key, plan)| format!("{key}: {}", plan.replace('\n', " / ")))
            .collect();
        assert!(thin.is_empty(), "asked with almost nothing:\n{}", thin.join("\n"));
    }

    #[test]
    fn remembering_faces_says_what_is_kept_and_where() {
        let plan = plan_for("faces");
        assert!(plan.contains("사진은 저장하지 않습니다"));
        assert!(plan.contains("128"));
        assert!(plan.contains("20명"));
        assert!(plan.contains(r"%LOCALAPPDATA%\argmind\faces.json"), "{plan}");
    }

    #[test]
    fn attention_says_it_is_a_head_and_that_the_model_never_sees_it() {
        let plan = plan_for("attention");
        assert!(plan.contains("고개"));
        assert!(plan.contains("모델에게는 보내지 않습니다"));
    }
}

#[cfg(test)]
mod presence_tests {
    use super::{present, AWAY_AFTER};

    #[test]
    fn a_face_keeps_a_still_person_present() {
        //: **The case that silenced everything.** Nothing has crossed the movement threshold for
        //: ten minutes, and there is a face in view.
        assert!(present(Some(600.0), Some(2.0), None));
        assert!(present(None, Some(2.0), None));
    }

    #[test]
    fn somebody_typing_is_there_whatever_the_camera_saw() {
        //: The owner's case: ten minutes of nothing on camera, input a few seconds ago.
        assert!(present(Some(601.0), None, Some(3.0)));
        //: And input long ago is no evidence of now.
        assert!(!present(Some(601.0), None, Some(AWAY_AFTER + 1.0)));
    }

    #[test]
    fn with_no_face_pass_it_is_movement_alone_as_before() {
        assert!(present(Some(10.0), None, None));
        assert!(!present(Some(AWAY_AFTER + 1.0), None, None));
        assert!(!present(None, None, None));
    }

    #[test]
    fn an_old_face_is_not_a_person_now() {
        assert!(!present(None, Some(AWAY_AFTER + 1.0), None));
    }
}

#[cfg(test)]
mod refusal_tests {
    use super::{why_refused, Refusal};

    #[test]
    fn nothing_counted_is_the_ordinary_gone() {
        assert_eq!(why_refused((0, 0), (0, 0)), Refusal::Gone);
        assert_eq!(why_refused((4, 4), (2, 2)), Refusal::Gone);
    }

    #[test]
    fn copies_that_appeared_are_not_reported_as_vanishing() {
        //: **The defect this exists for.** Every refusal used to reach the page as `gone`, which
        //: prints 그새 사라짐 — about a button that was there twice.
        assert_eq!(why_refused((0, 1), (0, 0)), Refusal::Copies);
    }

    #[test]
    fn an_answer_the_screen_outran_is_its_own_outcome() {
        //: Somebody answered the question a moment ago. Telling them the thing vanished would be
        //: the worst of the three sentences to get wrong, because they were looking at it.
        assert_eq!(why_refused((0, 0), (0, 1)), Refusal::Outrun);
    }

    #[test]
    fn late_wins_when_both_moved() {
        //: `perform` handles one effect here and counts one of them, but a caller handing it more
        //: than one could see both move. *Outrun* is the more specific fact about a person's answer.
        assert_eq!(why_refused((0, 1), (0, 1)), Refusal::Outrun);
    }
}

#[cfg(test)]
mod attention_tests {
    use super::camera_line;
    use crate::platform::Facing;

    #[test]
    fn the_switch_off_changes_the_line_by_nothing() {
        //: **The claim this feature was allowed on.** Prompt edits here are 1 for 7, and the only
        //: sentence-level one that paid was a slot fed by state and absent when there is no state.
        //: If this ever fails, a prompt grew a line nobody measured.
        let before = "마지막 움직임 3초 전";
        assert_eq!(camera_line(Some(3.0), 99.0, Facing::NotLookedFor), before);
        assert_eq!(camera_line(None, 12.0, Facing::NotLookedFor), "12초째 아무 움직임 없음");
    }

    #[test]
    fn each_of_the_three_readings_says_a_different_thing() {
        let of = |f| camera_line(Some(1.0), 9.0, f);
        let three = [of(Facing::AtIt), of(Facing::Aside), of(Facing::Unreadable),
                     of(Facing::NoFace)];
        for (i, a) in three.iter().enumerate() {
            for b in three.iter().skip(i + 1) {
                assert_ne!(a, b, "two readings render the same sentence");
            }
        }
    }

    #[test]
    fn unreadable_does_not_claim_the_head_was_turned() {
        //: The measurement says it *correlates* with turning. A sentence that said 고개를 돌림 here
        //: would be asserting what nobody measured, which is the small step that later reads as a
        //: finding.
        let unreadable = camera_line(Some(1.0), 9.0, Facing::Unreadable);
        let aside = camera_line(Some(1.0), 9.0, Facing::Aside);
        assert!(!unreadable.contains("돌림"), "{unreadable}");
        assert!(aside.contains("돌림"));
    }

    #[test]
    fn the_movement_half_is_untouched_by_the_head_half() {
        //: One slot carries two facts and the second must not rewrite the first — the failure would
        //: be silent, because both sentences read fine.
        for f in [Facing::NotLookedFor, Facing::AtIt, Facing::Aside, Facing::Unreadable,
                  Facing::NoFace] {
            assert!(camera_line(Some(7.0), 9.0, f).starts_with("마지막 움직임 7초 전"));
        }
    }
}
