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

pub mod app;
pub mod fix;
pub mod needs;
pub mod proposals;

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
    /// Anything else, named so a front end that asked wrongly is told rather than ignored.
    Unknown,
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
        ("POST", "/notice/on") => Route::Notice(true),
        ("POST", "/notice/off") => Route::Notice(false),
        ("POST", "/look") => Route::Look,
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

/// The head of a response that never ends.
pub fn event_stream_head() -> Vec<u8> {
    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\n\
      Connection: keep-alive\r\n\r\n".to_vec()
}
