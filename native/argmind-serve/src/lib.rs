//! The seam between the Rust core and a front end that is not Rust.
//!
//! The window is the one part of the tree whose successor is **not** a port: the plan is a Rust core
//! and a TypeScript interface, so what has to exist here is a way for the two to talk. This is that,
//! and it is deliberately the smallest thing that works.
//!
//! **Raw HTTP/1.1 and server-sent events, with no new dependency.** `argmind-env-ollama` already
//! speaks HTTP to the model with nothing but `std`, so the same is possible in the other direction;
//! a WebSocket would need a SHA-1 handshake and frame parsing, and SSE needs neither — a response
//! that never ends and lines beginning `data: `. The licence rule pays for itself here: nothing new
//! was added, so there is nothing new to check.
//!
//! ## What a window gave for free and a server does not
//!
//! `app.py` was in one process with the agent. A server is reachable by **anything else running as
//! this user**, and this agent can click. So two things are structural rather than advisory:
//!
//! * it binds to loopback and [`is_loopback`] is checked before the socket opens — an agent that
//!   can press buttons on your screen must not be reachable from the network;
//! * every request carries a token minted at startup, and a request without it is refused before it
//!   is routed. Localhost is not a permission: any program you run is on localhost.
//!
//! **The token travels two ways, and the second one is not a convenience.** `fetch` sends it as a
//! header; `EventSource` — which is what opens the event stream — **cannot set headers at all**, so
//! the page would be locked out of the only route that tells it anything. It also arrives as a
//! cookie, set on the page itself: automatic, invisible to script, and absent from the URL and from
//! shell history, which is why the query string is still read by nothing. `SameSite=Strict` is what
//! makes the cookie safe to have — it is not attached to a request another site caused, so a page
//! at somebody else's address cannot press this agent's buttons by pointing a form at it.
//!
//! ## What must survive the move
//!
//! **An unprompted effect without standing permission is downgraded to speech**, and the interface
//! is what holds that. The Python app inverts the flag deliberately and one AST test keeps `perform`
//! reachable from exactly one place; the same argument has to hold here, which is why the protocol
//! has a `confirm` route at all rather than acting on what the model returned.

pub mod activity;
pub mod app;
pub mod building;
pub mod chosen;
pub mod conversation;
pub mod decided;
pub mod deliver;
pub mod mcp;
pub mod errands;
pub mod faces;
pub mod fetch;
pub mod fix;
pub mod interface;
pub mod looking;
pub mod mind;
pub mod needs;
pub mod ollama;
pub mod people;
pub mod platform;
pub mod pursuit;
pub mod proposals;
pub mod recogniser;
pub mod rules;
pub mod senses;
pub mod speaker;
pub mod spoken;
pub mod route;
pub mod surface;
pub mod tact;
pub mod wonders;

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::net::IpAddr;

/// One parsed request: what a front end asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub method: String,
    pub path: String,
    pub token: Option<String>,
    pub body: String,
}

/// Parse the head and body of an HTTP/1.1 request.
///
/// Deliberately small: this speaks to one front end on one machine, and a general parser is more
/// surface than the job needs. It reads the method, the path with any query dropped, the token
/// header, and the body — and **refuses anything it does not fully understand** rather than
/// guessing, because a request this misreads is a request the agent acts on.
pub fn parse_request(raw: &str) -> Option<Request> {
    let (head, body) = raw.split_once("\r\n\r\n").unwrap_or((raw, ""));
    let mut lines = head.split("\r\n");
    let mut start = lines.next()?.split(' ');
    let method = start.next()?.to_string();
    let target = start.next()?;
    let version = start.next()?;
    if !version.starts_with("HTTP/1.") || method.is_empty() {
        return None;
    }
    //: The query is dropped rather than parsed: **a token in a URL ends up in logs and history**,
    //: so it travels in a header and nothing here reads a query string.
    let path = target.split(['?', '#']).next()?.to_string();
    let mut token = None;
    let mut cookie = None;
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("x-argmind-token") {
                token = Some(value.trim().to_string());
            }
            if name.eq_ignore_ascii_case("cookie") {
                cookie = cookie_value(value, COOKIE);
            }
        }
    }
    //: **The header wins.** A programmatic client that sent one meant it; a stale cookie from an
    //: older run of the server must not quietly override it.
    let token = token.or(cookie);
    Some(Request { method, path, token, body: body.to_string() })
}

/// The cookie the page carries its token in.
pub const COOKIE: &str = "argmind";

/// One cookie out of a `Cookie:` header, which holds several separated by `; `.
pub fn cookie_value(header: &str, want: &str) -> Option<String> {
    header.split(';').find_map(|pair| {
        let (name, value) = pair.split_once('=')?;
        //: **Trimmed on the name and not on the value.** A cookie value may legitimately contain
        //: spaces, and eating them would silently change the token.
        (name.trim() == want).then(|| value.to_string())
    })
}

/// The header that puts the token in the page, for `EventSource` to send back.
///
/// `HttpOnly` because nothing in the page needs to read it — it uses the copy stamped into the
/// document for its `fetch` headers. `SameSite=Strict` because that is the whole safety of having a
/// cookie at all: it is not attached to a request another site caused.
pub fn cookie_header(token: &str) -> String {
    format!("Set-Cookie: {COOKIE}={token}; Path=/; SameSite=Strict; HttpOnly\r\n")
}

/// A token minted at startup, from the operating system's entropy.
///
/// **No new crate for this.** Rust seeds `RandomState` from the OS, and two of them in one process
/// disagree, so hashing a counter under a fresh state gives real entropy without a dependency —
/// which the licence rule makes worth the two lines it costs.
pub fn mint_token() -> String {
    let mut out = String::new();
    while out.len() < 32 {
        let mut h = RandomState::new().build_hasher();
        h.write_usize(out.len());
        out.push_str(&format!("{:016x}", h.finish()));
    }
    out.truncate(32);
    out
}

/// Is this request allowed to reach the agent at all?
///
/// **Compared in constant time.** The threat is not a remote attacker — it is another program on
/// this machine, which can try as often as it likes and can time the answer precisely.
pub fn authorised(want: &str, given: Option<&str>) -> bool {
    let Some(given) = given else { return false };
    if given.len() != want.len() {
        return false;
    }
    let mut differs = 0u8;
    for (a, b) in want.bytes().zip(given.bytes()) {
        differs |= a ^ b;
    }
    differs == 0
}

/// **Loopback only, checked before the socket opens.**
///
/// An agent that can press buttons on somebody's screen has no business accepting a connection from
/// the network, and the difference between `127.0.0.1` and `0.0.0.0` is one character in a config
/// nobody reads twice.
/// **IPv6 is why this is not one line.** The first version split on the last colon and called the
/// rest the host, which turns `::1` into host `::` and port `1` — the unspecified address, not
/// loopback. It failed in the safe direction and it was still wrong, and a guard that is wrong in
/// the safe direction today is a guard nobody trusts tomorrow.
pub fn is_loopback(addr: &str) -> bool {
    let host = if let Some(rest) = addr.strip_prefix('[') {
        //: `[v6]` or `[v6]:port` — the brackets exist precisely because a colon is ambiguous.
        match rest.split_once(']') {
            Some((host, _)) => host,
            None => return false,
        }
    } else if let Ok(ip) = addr.parse::<IpAddr>() {
        //: A bare address of either family, with no port.
        return ip.is_loopback();
    } else {
        addr.rsplit_once(':').map(|(host, _)| host).unwrap_or(addr)
    };
    //: A hostname is not an address, and resolving one would mean trusting whatever resolved it.
    host.parse::<IpAddr>().map(|ip| ip.is_loopback()).unwrap_or(false)
}

/// **Is this request addressed to this machine by name?** — read off its `Host`, since 2026-11-16,
/// found while asking whether the repository could go public.
///
/// Loopback stops another machine connecting. It does not stop a page in a browser on *this* one:
/// a site whose name is made to resolve to `127.0.0.1` after it has loaded — DNS rebinding — is the
/// same origin as itself as far as the browser knows, so it may read `/`, which is served without a
/// token because the token is inside it, and then send `/say` with that token. **Reasoned from the
/// code, not attempted.** What such a request cannot choose is its `Host`: a browser writes the name
/// the page was loaded from, and a script may not set that header. So a request is served only when
/// the name is a loopback address or `localhost` — the window, the headless test and every `curl`
/// here say `127.0.0.1` — and argmind's own browser pane, which opens the web, is covered by the
/// same line. **No `Host` at all is refused too**: HTTP/1.1 requires one and every client here
/// sends it.
pub fn addressed_here(head: &str) -> bool {
    let host = head.split("\r\n").skip(1).find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.trim().eq_ignore_ascii_case("host").then(|| value.trim())
    });
    let Some(host) = host else { return false };
    let name = if host.starts_with('[') { host } else { host.rsplit_once(':').map(|(n, _)| n).unwrap_or(host) };
    name.eq_ignore_ascii_case("localhost") || is_loopback(host)
}

/// What a request is asking for.
#[derive(Debug, Clone, PartialEq)]
pub enum Route {
    /// The interface itself.
    Ui,
    /// Somebody said something.
    Said(String),
    /// The stream of what happened.
    Events,
    /// **Do the thing that was proposed.** The only route that can make the agent act, and it
    /// exists so that acting is a separate decision from deciding — the same argument the Python
    /// app's confirm button carries, and the reason one AST test there counts the callers of
    /// `perform`.
    Confirm(u64),
    /// Do not.
    Decline(u64),
    /// Install the missing thing this names. **The button in the page is the approval**, and the
    /// page had to show what it would do before it could be pressed.
    Fix(String),
    /// An answer to a permission question: the thing, and which of the four buttons.
    ///
    /// **Four rather than two**, because *yes* and *no* leave a program asking the same question
    /// every morning, and that is how people learn to press the first button without reading it.
    /// What may be answered *always* is a fixed list in [`crate::decided`] — and until 2026-09-22
    /// *a click is not on it* was written here. It is now, as `acting`, at the owner's asking; what
    /// that key permits is the absence of the button, not the absence of every gate. See
    /// [`Route::Acting`].
    Decide(String, String),
    /// Turn one thinking dial: its name and where to put it.
    Dial(String, u8),
    /// Write the thinking goal — the sentence the dials cannot express. An empty one clears it.
    Aim(String),
    /// Answer in this language — a code from `argmind_core::mind::LANGUAGES`, or `""` for the
    /// language the person used, which is what `/language/follow` sends.
    Language(String),
    /// Write the page in this language — a code from `interface::LANGUAGES`, or `""` for the
    /// system's, which is what `/interface/follow` sends.
    Interface(String),
    /// Take back something that was answered once and for all.
    Forget(String),
    /// Use this microphone, speaker or camera from now on. An empty id means *go back to whatever
    /// the operating system prefers*.
    Choose(String, String),
    /// Open one for a moment and say what happened. **The only thing in this program that turns a
    /// device on because somebody asked rather than because a turn needed it.**
    Try(String),
    /// Which of [`crate::speaker::PRESETS`] the synthesiser speaks in — **and say a sentence in
    /// it**, because a name and a number in a list are not a voice until you have heard one.
    ///
    /// One route rather than a *set* and a *preview*: choosing and hearing are the same act here.
    /// The key is checked against the fixed list before anything happens, like a dial's name and
    /// unlike a device id.
    Voice(String),
    /// Say answers out loud, or stop. **Written down as a standing answer** rather than held in
    /// memory, because a switch a person flips and then loses on the next launch is the defect this
    /// replaced.
    Aloud(bool),
    /// Keep the camera on and notice whether anybody is moving, or switch it off.
    ///
    /// **The one route that leaves a device running.** `/listen` opens the microphone while a key
    /// is held and closes it on the way up; this one is on until it is turned off, which is why the
    /// window that asks says so and why the light is the honest part of the design.
    Camera(bool),
    /// Set 눈치, 0–10. **Behaviour, never prompt** — `tact.rs`.
    Nunchi(u8),
    /// Report whether a head is pointed at the screen, or stop. **Not the same as `Camera`** —
    /// that one is a light that is on; this is a fact about a person that the same light can give.
    Attention(bool),
    /// Remember the faces the camera sees, or stop. **A different question from `Camera`** — one
    /// is a light that is on, the other is a row in a file that outlives the session.
    Faces(bool),
    /// Give one remembered person a name. The key, and the name from the body.
    NameFace(String, String),
    /// Forget one person.
    ForgetFace(String),
    /// Forget everybody, and delete the file.
    ForgetFaces,
    /// Answer the one open question. **An empty body is a skip** — the same act as the button
    /// beside the box, because clearing the field and pressing enter says the same thing.
    Answer(u64, String),
    /// Replace the rules with these, one per line, `app<TAB>action<TAB>then`.
    Rules(String),
    /// Let argmind carry out what it thought of by itself, or go back to waiting for a person.
    Initiative(bool),
    Looking(bool),
    /// Show argmind's browser pane and look there, or hide it and look at the desktop.
    Web(bool),
    /// Open an address in the pane, or search for the words.
    WebGo(String),
    /// Empty the conversation, in memory and in its file.
    ForgetConversation,
    /// Send the timeline, or empty it.
    Activity,
    ClearActivity,
    /// Open the folder tasks work in, in the person's file manager. A POST like every route that
    /// does something, so a link or a prefetch cannot open a window.
    OpenWorkFolder,
    /// Remember this sentence, in the person's words.
    Keep(String),
    /// Forget the kept sentence at this place in the list.
    Unkeep(usize),
    /// Hand argmind an errand, in the person's own words.
    Errand(String),
    /// Answer an errand's question, or correct its result.
    ErrandAnswer(u64, String),
    /// Take one off the list.
    ErrandDrop(u64),
    /// Carry out what was proposed without waiting for 실행, or go back to asking.
    ///
    /// **The one route that changes what argmind may do rather than what it may know.** Everything
    /// else in this enum turns a sense on, writes a preference, or answers a question; this decides
    /// whether [`Route::Confirm`] is still the only way an action happens. See
    /// `app::propose` for the one case it still does not cover.
    Acting(bool),
    /// Watch the screen and think when it changes, or stop.
    ///
    /// **Off until asked for**, and the reason is a measured one: something on screen changes 13
    /// seconds out of 14, so this is a standing invitation to spend a model call every few seconds
    /// for as long as it is on.
    Notice(bool),
    /// Take a fresh reading of what is missing. **Read-only** — it installs nothing, so the page
    /// can offer it freely after somebody has installed something themselves.
    Look,
    /// Hold the talk key, or let it go.
    ///
    /// **Two routes rather than one recording call**, because push-to-talk is the design: the
    /// microphone is open exactly while somebody is holding it down, which is the floor an open
    /// microphone has no equivalent of and the reason the agent cannot hear itself.
    Listen(bool),
    /// What argmind's work screen looks like, for the page — `/work-screen/<the number it has>`,
    /// so a picture is sent only when it changed. A GET, since it changes nothing.
    WorkScreen(u64),
    /// Anything else, named so a front end that asked wrongly is told rather than ignored.
    Unknown,
}

/// Per cent escapes back to bytes.
///
/// **A WASAPI endpoint id is `{0.0.1.00000000}.{b264…}`** — braces and dots, which survive a URL
/// untouched — but a camera's name is whatever its driver called it and can hold a space, a comma
/// or a bracket. The page encodes it; this undoes that, and anything malformed is left as it was
/// rather than guessed at.
pub fn decoded(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let pair = std::str::from_utf8(&bytes[i + 1..i + 3]).ok()
                .and_then(|h| u8::from_str_radix(h, 16).ok());
            if let Some(byte) = pair {
                out.push(byte);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

/// Bytes to per cent escapes, everything but letters, digits and `-_.~` — `decoded`'s inverse, for
/// a window title carried in a header, where only ASCII is safe.
pub fn encoded(text: &str) -> String {
    text.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) {
        (b as char).to_string()
    } else {
        format!("%{b:02X}")
    }).collect()
}

pub fn route(request: &Request) -> Route {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/") | ("GET", "/index.html") => Route::Ui,
        ("GET", "/events") => Route::Events,
        ("POST", "/say") => Route::Said(request.body.clone()),
        ("POST", p) if p.starts_with("/confirm/") => match p[9..].parse() {
            Ok(n) => Route::Confirm(n),
            Err(_) => Route::Unknown,
        },
        ("POST", p) if p.starts_with("/decline/") => match p[9..].parse() {
            Ok(n) => Route::Decline(n),
            Err(_) => Route::Unknown,
        },
        ("POST", p) if p.starts_with("/fix/") && p.len() > 5 =>
            Route::Fix(p[5..].to_string()),
        //: `/decide/<thing>/<how>`. Two segments, and both are checked against fixed lists before
        //: anything happens — a permission is not a string the page gets to invent.
        ("POST", p) if p.starts_with("/decide/") => match p[8..].split_once('/') {
            Some((key, how)) if !key.is_empty() && !how.is_empty() =>
                Route::Decide(key.to_string(), how.to_string()),
            _ => Route::Unknown,
        },
        ("POST", p) if p.starts_with("/forget/") && p.len() > 8 =>
            Route::Forget(p[8..].to_string()),
        //: `/choose/<kind>/<id>` and `/choose/<kind>` for none. The kind is checked against a fixed
        //: list before anything happens; the id is opaque and is only ever handed back to Windows.
        ("POST", p) if p.starts_with("/choose/") && p.len() > 8 => match p[8..].split_once('/') {
            Some((kind, id)) if !kind.is_empty() =>
                Route::Choose(kind.to_string(), decoded(id)),
            None => Route::Choose(p[8..].to_string(), String::new()),
            _ => Route::Unknown,
        },
        ("POST", p) if p.starts_with("/try/") && p.len() > 5 =>
            Route::Try(p[5..].to_string()),
        //: `/voice/<key>`. Checked against the fixed list at the far end, like `/dial`.
        ("POST", p) if p.starts_with("/voice/") && p.len() > 7 && !p[7..].contains('/') =>
            Route::Voice(p[7..].to_string()),
        //: `/dial/<name>/<0-10>`. The name arrives percent-encoded because it is Korean, and it is
        //: checked against the fixed list before anything is written — a dial is not a string the
        //: page gets to invent, for the same reason a permission is not.
        ("POST", p) if p.starts_with("/dial/") && p.len() > 6 => match p[6..].rsplit_once('/') {
            Some((name, n)) if !name.is_empty() => match n.parse::<u8>() {
                Ok(n) => Route::Dial(decoded(name), n),
                Err(_) => Route::Unknown,
            },
            _ => Route::Unknown,
        },
        //: The sentence comes in the body rather than the path: it is prose, it can be long, and a
        //: path is the wrong place for either.
        ("POST", "/aim") => Route::Aim(request.body.clone()),
        ("POST", "/interface/follow") => Route::Interface(String::new()),
        ("POST", p) if p.starts_with("/interface/") && p.len() > 11 => Route::Interface(p[11..].to_string()),
        ("POST", "/language/follow") => Route::Language(String::new()),
        ("POST", p) if p.starts_with("/language/") && p.len() > 10 => Route::Language(p[10..].to_string()),
        ("POST", "/aloud/on") => Route::Aloud(true),
        ("POST", "/aloud/off") => Route::Aloud(false),
        ("POST", "/camera/on") => Route::Camera(true),
        ("POST", "/camera/off") => Route::Camera(false),
        //: **Before the prefixed ones**, so `forget-all` is never read as a person called
        //: `forget-all`. Exact matches first is the whole rule and it is easy to get wrong once.
        ("POST", p) if p.starts_with("/nunchi/") => match p[8..].parse::<u8>() {
            Ok(n) => Route::Nunchi(n),
            Err(_) => Route::Unknown,
        },
        ("POST", "/attention/on") => Route::Attention(true),
        ("POST", "/attention/off") => Route::Attention(false),
        ("POST", "/faces/on") => Route::Faces(true),
        ("POST", "/faces/off") => Route::Faces(false),
        ("POST", "/faces/forget-all") => Route::ForgetFaces,
        ("POST", p) if p.starts_with("/wonder/") && p.len() > 8 => match p[8..].parse() {
            Ok(id) => Route::Answer(id, request.body.clone()),
            Err(_) => Route::Unknown,
        },
        //: `/face/forget/<key>` and `/face/name/<key>`, the name in the body because it is a
        //: person's name — it can be Korean, it can be long, and a path is the wrong place for
        //: either. The same reasoning `/aim` already uses.
        ("POST", p) if p.starts_with("/face/forget/") && p.len() > 13 =>
            Route::ForgetFace(p[13..].to_string()),
        ("POST", p) if p.starts_with("/face/name/") && p.len() > 11 =>
            Route::NameFace(p[11..].to_string(), request.body.clone()),
        //: `/errand` with the words in the body, `/errand/<id>/answer` likewise, and
        //: `/errand/<id>/drop`. The words are a person's sentence, so never the path.
        ("POST", "/errand") => Route::Errand(request.body.clone()),
        //: The whole list, because the page edits it as a list and a route per row would be three
        //: routes to keep in step with one table.
        ("POST", "/rules") => Route::Rules(request.body.clone()),
        //: The sentence in the body, like `/aim`; forgetting by place in the list the page drew.
        ("POST", "/kept") => Route::Keep(request.body.clone()),
        ("POST", "/activity") => Route::Activity,
        ("POST", "/conversation/forget") => Route::ForgetConversation,
        ("POST", "/web/on") => Route::Web(true),
        ("POST", "/web/off") => Route::Web(false),
        ("POST", "/web/go") => Route::WebGo(request.body.clone()),
        ("POST", "/activity/clear") => Route::ClearActivity,
        ("POST", "/work-folder/open") => Route::OpenWorkFolder,
        ("POST", p) if p.starts_with("/kept/") => match p[6..].strip_suffix("/drop")
            .and_then(|i| i.parse().ok()) {
            Some(i) => Route::Unkeep(i),
            None => Route::Unknown,
        },
        ("POST", p) if p.starts_with("/errand/") => match p[8..].split_once('/') {
            Some((id, "answer")) => match id.parse() {
                Ok(id) => Route::ErrandAnswer(id, request.body.clone()),
                Err(_) => Route::Unknown,
            },
            Some((id, "drop")) => match id.parse() {
                Ok(id) => Route::ErrandDrop(id),
                Err(_) => Route::Unknown,
            },
            _ => Route::Unknown,
        },
        ("POST", "/initiative/on") => Route::Initiative(true),
        ("POST", "/initiative/off") => Route::Initiative(false),
        ("POST", "/looking/on") => Route::Looking(true),
        ("POST", "/looking/off") => Route::Looking(false),
        ("POST", "/acting/on") => Route::Acting(true),
        ("POST", "/acting/off") => Route::Acting(false),
        ("POST", "/notice/on") => Route::Notice(true),
        ("POST", "/notice/off") => Route::Notice(false),
        ("POST", "/look") => Route::Look,
        ("GET", p) if p.starts_with("/work-screen/") => match p[13..].parse() {
            Ok(n) => Route::WorkScreen(n),
            Err(_) => Route::Unknown,
        },
        ("POST", "/listen/start") => Route::Listen(true),
        ("POST", "/listen/stop") => Route::Listen(false),
        _ => Route::Unknown,
    }
}

/// One server-sent event.
///
/// **Every line of the payload gets its own `data:` prefix**, because a bare newline inside a
/// payload ends the event — and a reply with a line break in it is the ordinary case here, not an
/// edge case.
pub fn sse(event: &str, data: &str) -> String {
    let mut out = format!("event: {event}\n");
    for line in data.split('\n') {
        out.push_str("data: ");
        out.push_str(line.trim_end_matches('\r'));
        out.push('\n');
    }
    out.push('\n');
    out
}

/// A complete HTTP response.
pub fn respond(status: &str, content_type: &str, body: &str) -> Vec<u8> {
    respond_with(status, content_type, body, "")
}

/// The same, with an extra header — used to hand the page its cookie.
pub fn respond_with(status: &str, content_type: &str, body: &str, extra: &str) -> Vec<u8> {
    let mut out = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\n{extra}\
         Connection: close\r\n\r\n", body.len());
    out.push_str(body);
    out.into_bytes()
}

/// A response whose body is bytes — the work screen's picture.
pub fn respond_bytes(status: &str, content_type: &str, body: &[u8], extra: &str) -> Vec<u8> {
    let mut out = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Cache-Control: no-store\r\n{extra}Connection: close\r\n\r\n", body.len()).into_bytes();
    out.extend_from_slice(body);
    out
}

/// The head of a response that never ends.
pub fn event_stream_head() -> Vec<u8> {
    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\n\
      Connection: keep-alive\r\n\r\n".to_vec()
}
