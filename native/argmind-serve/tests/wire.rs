//! The seam's protocol, checked without a socket, a display or a model.
//!
//! Two of these are safety properties rather than correctness ones. A server puts an agent that can
//! press buttons within reach of **anything else running as this user**, which the in-process window
//! it replaces did not — so the loopback check and the token are not conveniences, and they are
//! tested the way a guard is tested: by trying to get past them.

use argmind_serve::{authorised, cookie_header, cookie_value, event_stream_head, is_loopback,
                    mint_token, parse_request, respond,
                    route, sse, Request, Route};

fn get(path: &str) -> Request {
    parse_request(&format!("GET {path} HTTP/1.1\r\nHost: x\r\n\r\n")).unwrap()
}

/// **Loopback only**, and the difference is one character in a config nobody reads twice.
#[test]
fn the_agent_is_not_reachable_from_the_network() {
    assert!(is_loopback("127.0.0.1:7331"));
    assert!(is_loopback("127.0.0.1"));
    assert!(is_loopback("[::1]:7331"));
    assert!(is_loopback("::1"));
    //: The three ways somebody ends up listening to the world.
    assert!(!is_loopback("0.0.0.0:7331"));
    assert!(!is_loopback("[::]:7331"));
    assert!(!is_loopback("192.168.1.20:7331"));
    //: **And the same three with no port at all.** Every non-loopback case above carries one, so
    //: they all take the split-off-the-port branch — and a mutation that made the *bare address*
    //: branch answer `true` for everything survived the whole suite. That is the dangerous
    //: direction, found by trying to get past the guard rather than by reading it.
    assert!(!is_loopback("0.0.0.0"));
    assert!(!is_loopback("::"));
    assert!(!is_loopback("192.168.1.20"));
    //: A bracket that never closes is malformed, and malformed is refused — not read as far as it
    //: parses and then allowed.
    assert!(!is_loopback("[::1"));
    assert!(!is_loopback("["));
    //: A hostname is not an address, and resolving one here would mean trusting whatever resolved
    //: it — so anything unparseable is refused rather than looked up.
    assert!(!is_loopback("localhost:7331"));
    assert!(!is_loopback(""));
    assert!(!is_loopback("nonsense"));
}

/// **Localhost is not a permission**: any program you run is on localhost.
#[test]
fn a_request_without_the_token_gets_nowhere() {
    let want = mint_token();
    assert!(authorised(&want, Some(&want)));
    assert!(!authorised(&want, None), "no token at all must not pass");
    assert!(!authorised(&want, Some("")));
    assert!(!authorised(&want, Some(&want[..want.len() - 1])), "a prefix is not the token");
    assert!(!authorised(&want, Some(&format!("{want}x"))), "an extension is not the token");
    let mut wrong: Vec<u8> = want.clone().into_bytes();
    wrong[7] ^= 1;
    assert!(!authorised(&want, Some(&String::from_utf8(wrong).unwrap())));
}

/// **`EventSource` cannot set headers**, and that is why the token also travels as a cookie.
///
/// This is the bug the suite missed. Every test above sent the token the way `curl -H` sends it,
/// and the only client that matters — a browser opening the event stream — has no way to do that
/// at all. The page loaded, the stream was refused, and the interface said *연결 끊김* while every
/// test passed. **A server checked only by the thing that was written to check it agrees with
/// itself.**
#[test]
fn the_page_can_authenticate_without_being_able_to_send_a_header() {
    let want = mint_token();
    let header = cookie_header(&want);
    assert!(header.starts_with("Set-Cookie: argmind="), "{header}");
    //: **`SameSite=Strict` is the whole safety of having a cookie.** Without it a page at somebody
    //: else's address could point a form at this agent and the browser would attach the token.
    assert!(header.contains("SameSite=Strict"), "{header}");
    //: Nothing in the page reads it — it uses the copy stamped into the document — so script has
    //: no business being able to.
    assert!(header.contains("HttpOnly"), "{header}");
    //: Every route, not just the page it was set on.
    assert!(header.contains("Path=/"), "{header}");

    //: A request carrying only a cookie is a request that authenticates, which is precisely what
    //: was broken.
    let asked = parse_request(&format!(
        "GET /events HTTP/1.1\r\nCookie: argmind={want}\r\n\r\n")).unwrap();
    assert_eq!(asked.token.as_deref(), Some(want.as_str()));
    assert!(authorised(&want, asked.token.as_deref()));

    //: One cookie among several, which is what a browser actually sends.
    let many = parse_request(&format!(
        "GET /events HTTP/1.1\r\nCookie: other=1; argmind={want}; last=2\r\n\r\n")).unwrap();
    assert_eq!(many.token.as_deref(), Some(want.as_str()));
    //: A different cookie is not this one.
    let wrong = parse_request("GET /events HTTP/1.1\r\nCookie: something=else\r\n\r\n").unwrap();
    assert_eq!(wrong.token, None);
    assert!(!authorised(&want, wrong.token.as_deref()));

    //: **The header wins over the cookie.** A programmatic client that sent one meant it, and a
    //: stale cookie from an older run of the server must not override it.
    let both = parse_request(
        "GET /events HTTP/1.1\r\nCookie: argmind=old\r\nX-Argmind-Token: new\r\n\r\n").unwrap();
    assert_eq!(both.token.as_deref(), Some("new"));
}

#[test]
fn one_cookie_is_read_out_of_the_several_a_browser_sends() {
    assert_eq!(cookie_value("argmind=abc", "argmind").as_deref(), Some("abc"));
    assert_eq!(cookie_value(" argmind=abc ", "argmind").as_deref(), Some("abc "));
    assert_eq!(cookie_value("a=1; argmind=abc; b=2", "argmind").as_deref(), Some("abc"));
    assert_eq!(cookie_value("a=1; b=2", "argmind"), None);
    assert_eq!(cookie_value("", "argmind"), None);
    //: **A prefix is not the name.** `notargmind=x` must not answer for `argmind`.
    assert_eq!(cookie_value("notargmind=x", "argmind"), None);
    assert_eq!(cookie_value("argmindextra=x", "argmind"), None);
}

/// A token has to be long, unguessable and **different every time**.
#[test]
fn the_token_is_freshly_minted_and_not_a_constant() {
    let a = mint_token();
    let b = mint_token();
    assert_eq!(a.len(), 32);
    assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    assert_ne!(a, b, "two tokens in one process were identical — that is not entropy");
    //: Sixteen of them, because two agreeing once could be luck and a constant would agree always.
    let many: std::collections::BTreeSet<String> = (0..16).map(|_| mint_token()).collect();
    assert_eq!(many.len(), 16);
}

/// The parser refuses what it does not fully understand, rather than guessing.
#[test]
fn a_request_it_cannot_read_is_refused_rather_than_guessed_at() {
    let r = parse_request("POST /say HTTP/1.1\r\nX-Argmind-Token: abc\r\n\r\n안녕").unwrap();
    assert_eq!(r.method, "POST");
    assert_eq!(r.path, "/say");
    assert_eq!(r.token.as_deref(), Some("abc"));
    assert_eq!(r.body, "안녕");
    //: The header name is case-insensitive, as HTTP says.
    let r = parse_request("GET / HTTP/1.1\r\nx-ARGMIND-token:  spaced  \r\n\r\n").unwrap();
    assert_eq!(r.token.as_deref(), Some("spaced"));

    assert!(parse_request("").is_none());
    assert!(parse_request("GET\r\n\r\n").is_none());
    assert!(parse_request("GET /\r\n\r\n").is_none(), "no version");
    assert!(parse_request("GET / HTTP/2\r\n\r\n").is_none(), "not a version this speaks");
}

/// **A token in a URL ends up in logs and shell history**, so the query is dropped unread.
#[test]
fn nothing_reads_a_query_string() {
    assert_eq!(get("/events?token=secret").path, "/events");
    assert_eq!(get("/events?token=secret").token, None);
    assert_eq!(get("/#fragment").path, "/");
    assert_eq!(route(&get("/events?anything=at&all=1")), Route::Events);
}

/// Acting is a **separate decision** from deciding, and the protocol says so.
#[test]
fn the_only_way_to_make_it_act_is_a_confirmation() {
    assert_eq!(route(&get("/")), Route::Ui);
    assert_eq!(route(&get("/index.html")), Route::Ui);
    assert_eq!(route(&get("/events")), Route::Events);

    let said = parse_request("POST /say HTTP/1.1\r\n\r\n지우개 골라줘").unwrap();
    assert_eq!(route(&said), Route::Said("지우개 골라줘".into()));
    //: **Saying something does not act.** That is the whole permission model: the model's answer
    //: becomes a proposal, and a proposal waits.
    assert!(!matches!(route(&said), Route::Confirm(_)));

    let ok = parse_request("POST /confirm/3 HTTP/1.1\r\n\r\n").unwrap();
    assert_eq!(route(&ok), Route::Confirm(3));
    let no = parse_request("POST /decline/3 HTTP/1.1\r\n\r\n").unwrap();
    assert_eq!(route(&no), Route::Decline(3));

    //: A confirmation names **which** proposal. Without the number a stale button would confirm
    //: whatever happens to be pending, which is a different action from the one the person read.
    assert_eq!(route(&parse_request("POST /confirm/ HTTP/1.1\r\n\r\n").unwrap()), Route::Unknown);
    assert_eq!(route(&parse_request("POST /confirm/x HTTP/1.1\r\n\r\n").unwrap()), Route::Unknown);
    //: And a confirmation is never a GET: a link, an image tag or a prefetch must not be able to
    //: press the button.
    assert_eq!(route(&get("/confirm/3")), Route::Unknown);
    assert_eq!(route(&get("/say")), Route::Unknown);
}

/// **A bare newline ends an event**, and a reply with a line break in it is the ordinary case.
#[test]
fn every_line_of_a_payload_is_framed() {
    let one = sse("said", "hello");
    assert_eq!(one, "event: said\ndata: hello\n\n");

    let two = sse("said", "first\nsecond");
    assert_eq!(two, "event: said\ndata: first\ndata: second\n\n");
    //: Two events, not one with a stray line — which is what an unframed newline would produce.
    assert_eq!(two.matches("\n\n").count(), 1);

    //: A carriage return from a Windows-authored reply must not survive into the frame, or the
    //: browser reads a trailing `\r` on every line.
    assert_eq!(sse("x", "a\r\nb"), "event: x\ndata: a\ndata: b\n\n");
    assert_eq!(sse("x", ""), "event: x\ndata: \n\n");
    //: Korean is bytes on the wire and characters to the reader; nothing here counts either.
    assert!(sse("said", "지우개는 22번입니다").contains("지우개는 22번입니다"));
}

#[test]
fn a_response_says_how_long_it_is_in_bytes() {
    let body = "안녕";
    let out = String::from_utf8(respond("200 OK", "text/plain; charset=utf-8", body)).unwrap();
    //: **Bytes, not characters.** Six here, not two — a length in characters truncates every reply
    //: in this project's own language.
    assert!(out.contains("Content-Length: 6"), "{out}");
    assert!(out.ends_with("\r\n\r\n안녕"));
    assert!(out.contains("Cache-Control: no-store"));

    let head = String::from_utf8(event_stream_head()).unwrap();
    assert!(head.contains("text/event-stream"));
    //: A stream that never ends must not advertise a length.
    assert!(!head.contains("Content-Length"));
}

/// **A fix names what it fixes**, and the name is stable rather than a position.
///
/// A button in a page that had been open a while would otherwise install whatever had moved into
/// its slot — the same defect the proposal numbers are built to avoid, arriving in a different part
/// of the program.
#[test]
fn installing_something_names_it_and_is_never_a_get() {
    let post = |p: &str| parse_request(&format!("POST {p} HTTP/1.1\r\n\r\n")).unwrap();
    assert_eq!(route(&post("/fix/model")), Route::Fix("model".into()));
    assert_eq!(route(&post("/fix/korean-voice")), Route::Fix("korean-voice".into()));
    //: **Never by GET.** A link, a prefetch or an image tag must not be able to start an install.
    assert_eq!(route(&get("/fix/model")), Route::Unknown);
    //: And a fix with no name is not a fix — it would act on whatever the server felt like.
    assert_eq!(route(&post("/fix/")), Route::Unknown);
    assert_eq!(route(&post("/fix")), Route::Unknown);

    //: Re-reading what is missing installs nothing, so the page may offer it freely — but it still
    //: is not a GET, because it makes the server go and look.
    assert_eq!(route(&post("/look")), Route::Look);
    assert_eq!(route(&get("/look")), Route::Unknown);
}

/// **Holding and letting go are two routes**, and neither is a GET.
///
/// Push-to-talk is the design: the microphone is open exactly while somebody holds the key. A
/// single "record" call would have to decide for itself when to stop, which is the open-microphone
/// problem wearing a different name.
#[test]
fn the_microphone_opens_only_while_the_key_is_held() {
    let post = |p: &str| parse_request(&format!("POST {p} HTTP/1.1\r\n\r\n")).unwrap();
    assert_eq!(route(&post("/listen/start")), Route::Listen(true));
    assert_eq!(route(&post("/listen/stop")), Route::Listen(false));
    //: Not by GET: a prefetch, a link or an image tag must not be able to open a microphone.
    assert_eq!(route(&get("/listen/start")), Route::Unknown);
    assert_eq!(route(&get("/listen/stop")), Route::Unknown);
    //: And there is no third spelling that means either of them.
    assert_eq!(route(&post("/listen")), Route::Unknown);
    assert_eq!(route(&post("/listen/")), Route::Unknown);
    assert_eq!(route(&post("/listen/start/x")), Route::Unknown);
}

/// **The whole unauthenticated surface, written down.**
///
/// The page and its scripts are served without a token, because the token arrives inside the page.
/// That is two files today and was one yesterday — the sphere is its own module — and a third added
/// by an edit somewhere else is exactly the kind of change nobody reviews twice. So the list is
/// asserted rather than described.
#[test]
fn only_the_page_and_its_scripts_are_served_without_a_token() {
    let free: Vec<&str> = argmind_serve::app::SCRIPTS.iter().map(|(at, _)| *at).collect();
    assert_eq!(free, ["/app.js", "/orb.js"]);

    for (at, body) in argmind_serve::app::SCRIPTS {
        //: **Nothing the agent knows is in them.** The page carries the token; a script must not,
        //: because a script is the thing served to anybody who asks.
        assert!(!body.contains("__TOKEN__"), "{at} has the token placeholder in it");
        //: And they are not empty. `include_str!` catches a missing file at compile time and
        //: nothing catches a build that produced nothing.
        assert!(body.len() > 500, "{at} is {} bytes — was `npx tsc` run?", body.len());
    }
}
