//! Talking to Ollama — **an environment that happens to look like part of the core.**
//!
//! `docs/findings/porting-the-core.md` argues this belongs outside `argmind-core`: it is an HTTP
//! client for one server on one loopback port, dead on a phone that has no Ollama. So the core owns
//! the *reading* of a reply — `argmind_core::llm::Reading`, replayed against 128 recorded streams —
//! and this owns the socket and the bytes that go out on it.
//!
//! **No dependency.** A localhost HTTP/1.1 POST and a chunked-transfer reader are about a hundred
//! lines, and the alternative pulls a client crate and its whole transitive tree — which
//! `tests/test_licences.py` cannot audit, by its own admission. For one loopback call with no TLS
//! the trade is not close.
//!
//! What is verified offline and what is not, stated rather than implied:
//!
//! * **The request body, byte for byte**, against 28 recorded from the shipped Python. It has two
//!   traps that only a byte comparison finds: `json.dumps` defaults to `ensure_ascii=True`, so
//!   every Korean character leaves as `\uXXXX`, and a Python dict serialises in **insertion** order
//!   where a sorted map would not.
//! * **The de-chunking**, against canned bytes.
//! * **Not the socket.** One live call does that, and nothing here can.

use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;

use argmind_core::llm::{Reading, Reply};

pub mod manage;

pub const OLLAMA: &str = "http://127.0.0.1:11434";

/// One call to a local model. No streaming callbacks, no session state — the loop owns those.
pub struct Local {
    pub model: String,
    pub host: String,
    pub ctx: i64,
    pub max_tokens: i64,
    pub keep_alive: String,
    /// `None` leaves the field out entirely, which is **not** the same as `false`: a reasoning model
    /// with the field absent reasons anyway. One recorded arm was mislabelled `off` for exactly
    /// that reason and only the token count gave it away.
    pub think: Option<bool>,
    pub stop_when_answered: bool,
    pub stop_when_looping: bool,
}

impl Local {
    pub fn new(model: &str) -> Self {
        Local { model: model.into(), host: OLLAMA.into(), ctx: 8192, max_tokens: 4096,
                keep_alive: "60s".into(), think: None,
                stop_when_answered: true, stop_when_looping: true }
    }

    /// The exact bytes of the request. **A pure function of the settings and the prompt**, which is
    /// what lets it be compared without a server.
    ///
    /// No `format: "json"`: it silences this model completely — 366 ms, empty output. The shape is
    /// asked for in the prompt and pulled out of the text by the scanner.
    pub fn body(&self, prompt: &str) -> String {
        self.body_with(prompt, &[])
    }

    /// The same, with pictures. **Base64 in `images`**, which is where `/api/chat` takes them.
    ///
    /// `body` is this with none, so **a call that sends no picture is byte-identical to what
    /// it was before this existed**. The requests in this tree are compared against recorded
    /// bytes, and a stray empty `images` would break every one of them while changing nothing
    /// anybody meant.
    pub fn body_with(&self, prompt: &str, pictures: &[String]) -> String {
        let mut out = String::from("{");
        out.push_str(&format!("\"model\": {}, ", py_string(&self.model)));
        let images = if pictures.is_empty() {
            String::new()
        } else {
            let each: Vec<String> = pictures.iter().map(|p| format!("\"{p}\"")).collect();
            format!(", \"images\": [{}]", each.join(", "))
        };
        out.push_str(&format!("\"messages\": [{{\"role\": \"user\", \"content\": {}{images}}}], ",
                              py_string(prompt)));
        out.push_str("\"stream\": true, ");
        out.push_str(&format!("\"keep_alive\": {}, ", py_string(&self.keep_alive)));
        out.push_str(&format!(
            "\"options\": {{\"num_predict\": {}, \"num_ctx\": {}, \"temperature\": 0}}",
            self.max_tokens, self.ctx));
        if let Some(think) = self.think {
            out.push_str(&format!(", \"think\": {}", if think { "true" } else { "false" }));
        }
        out.push('}');
        out
    }

    /// Ask, and read until there is nothing left worth reading.
    ///
    /// The request is byte-identical to what it was before the stops existed — same messages, same
    /// `num_predict`, same `num_ctx`, same temperature. **Only the reading stops early**, so the
    /// tokens the model produces up to the stop are the tokens it would have produced anyway.
    pub fn ask(&self, prompt: &str) -> std::io::Result<Reply> {
        self.ask_into(prompt, &mut |_| {})
    }

    /// Ask about a picture.
    ///
    /// **A different call on purpose.** A model that can be handed a photograph should not be
    /// reachable by accident from the one that answers about the screen — `camera.rs` makes
    /// the same separation from the other end, where a [`Frame`] is 64×36 greys and there is
    /// no accessor for anything larger.
    ///
    /// [`Frame`]: https://docs.rs/
    pub fn look_at(&self, prompt: &str, png: &[u8]) -> std::io::Result<Reply> {
        self.ask_with(prompt, &[base64(png)], &mut |_| {})
    }

    /// The same call, with **every stream line handed to `keep` as the reader takes it.**
    ///
    /// One read loop and not two: a recorder written as a second copy of this loop would agree with
    /// it until the day it did not, and the disagreement would be invisible — the corpus would
    /// simply describe a reader nobody ships. `keep` sees exactly the lines the reader saw, in
    /// order, and stops where the reader stopped, so a stream recorded with the stops **off** is
    /// the whole stream and one recorded with them on is the prefix the stops left.
    ///
    /// It changes nothing about the request: same body, same `num_predict`, same `num_ctx`.
    pub fn ask_into(&self, prompt: &str, keep: &mut dyn FnMut(&str)) -> std::io::Result<Reply> {
        self.ask_with(prompt, &[], keep)
    }

    /// The one read loop, pictures optional. Everything above funnels here, for the reason
    /// given on `ask_into`: a second copy of this loop would agree with it until the day it
    /// did not, and the disagreement would be invisible.
    pub fn ask_with(&self, prompt: &str, pictures: &[String],
                    keep: &mut dyn FnMut(&str)) -> std::io::Result<Reply> {
        self.ask_while(prompt, pictures, &|| true, keep)
    }

    /// The same, **abandonable**: `carry_on` is asked before every line, and the moment it says no
    /// the socket is dropped and whatever arrived so far is returned.
    ///
    /// **This is not a nicety, it is the third rule of running senses in parallel.** Measured
    /// 2026-09-17: a look running beside the conversation takes the answer from a median of 1.0 s
    /// to 3.2 s, and one answer in ten waits out most of a whole look — 10.81 s. Declining to
    /// *start* a look while somebody is being answered fixes the median; only abandoning one
    /// already in flight can fix that tail.
    ///
    /// **Whether dropping the socket actually frees the card is a question about ollama, not about
    /// this function**, and it is measured by `argmind-measure --bin aborting`. If the server keeps
    /// generating for a reader that has gone, this returns promptly and buys nothing.
    ///
    /// The reply carries `reason = "abandoned"`, which is neither a stop nor an answer: a caller
    /// that treats it as a reply would be quoting a sentence nobody finished.
    pub fn ask_while(&self, prompt: &str, pictures: &[String], carry_on: &dyn Fn() -> bool,
                     keep: &mut dyn FnMut(&str)) -> std::io::Result<Reply> {
        let body = self.body_with(prompt, pictures);
        let (host, port) = split_host(&self.host);
        let stream = TcpStream::connect((host.as_str(), port))?;
        let mut w = &stream;
        write!(w, "POST /api/chat HTTP/1.1\r\nHost: {host}:{port}\r\n\
                   Content-Type: application/json\r\nContent-Length: {}\r\n\
                   Connection: close\r\n\r\n", body.len())?;
        w.write_all(body.as_bytes())?;
        w.flush()?;

        let mut r = BufReader::new(&stream);
        let chunked = read_headers(&mut r)?;
        let mut reading = Reading::new(self.stop_when_answered, self.stop_when_looping);
        let mut abandoned = false;
        for line in Lines::new(&mut r, chunked) {
            //: **Asked before the line is used, not after.** Checking at the bottom of the loop
            //: would still pay for one more line of generation, which on a thinking model is the
            //: thing being avoided.
            if !carry_on() {
                abandoned = true;
                break;
            }
            let line = line?;
            keep(&line);
            if line.trim().is_empty() {
                reading.blank();
                continue;
            }
            let Ok(d) = serde_json::from_str::<serde_json::Value>(&line) else { continue };
            let message = &d["message"];
            let thinking = message.get("thinking").and_then(|v| v.as_str()).unwrap_or("");
            let chunk = message.get("content").and_then(|v| v.as_str());
            let done = if d.get("done").and_then(|v| v.as_bool()).unwrap_or(false) {
                Some((d.get("prompt_eval_count").and_then(|v| v.as_i64()),
                      d.get("eval_count").and_then(|v| v.as_i64()),
                      d.get("done_reason").and_then(|v| v.as_str()).unwrap_or("").to_string()))
            } else {
                None
            };
            //: The thinking text is discarded and **its length is kept** — throwing the text away
            //: also threw away the only sign of what it cost.
            if !reading.line(thinking.chars().count(), chunk, done) {
                break;
            }
        }
        //: Dropping `stream` here is the whole mechanism — there is nothing to send, only a reader
        //: that stops being there.
        let mut out = reading.finish();
        if abandoned {
            out.reason = "abandoned".into();
        }
        Ok(out)
    }
}

/// `json.dumps`'s string escaping with `ensure_ascii=True`, which is Python's default and therefore
/// what every recorded body carries.
///
/// Python escapes everything outside `0x20..=0x7e` plus `"` and `\`, so `0x7f` goes too — and an
/// astral character leaves as a **surrogate pair**, which is the part a naive `\u{...}` would get
/// wrong.
/// Base64, hand-written.
///
/// **No new dependency**, the same reason HTTP and the WAV header in this tree are hand-written:
/// the licence rule is cheapest to keep when there is nothing new to check. Twenty lines, the
/// standard alphabet, padding included, because that is what `/api/chat` reads.
pub fn base64(bytes: &[u8]) -> String {
    const ABC: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = ((b[0] as u32) << 16) | ((b[1] as u32) << 8) | b[2] as u32;
        out.push(ABC[(n >> 18) as usize & 63] as char);
        out.push(ABC[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { ABC[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { ABC[n as usize & 63] as char } else { '=' });
    }
    out
}

pub fn py_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (' '..='~').contains(&c) => out.push(c),
            c => {
                let n = c as u32;
                if n > 0xFFFF {
                    let n = n - 0x10000;
                    out.push_str(&format!("\\u{:04x}\\u{:04x}",
                                          0xD800 + (n >> 10), 0xDC00 + (n & 0x3FF)));
                } else {
                    out.push_str(&format!("\\u{n:04x}"));
                }
            }
        }
    }
    out.push('"');
    out
}

fn split_host(host: &str) -> (String, u16) {
    let rest = host.trim_end_matches('/').trim_start_matches("http://");
    match rest.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().unwrap_or(11434)),
        None => (rest.to_string(), 11434),
    }
}

/// Read the response headers; returns whether the body is chunked.
fn read_headers(r: &mut impl BufRead) -> std::io::Result<bool> {
    let mut chunked = false;
    loop {
        let mut line = String::new();
        if r.read_line(&mut line)? == 0 || line.trim().is_empty() {
            return Ok(chunked);
        }
        if line.to_ascii_lowercase().starts_with("transfer-encoding:")
            && line.to_ascii_lowercase().contains("chunked") {
            chunked = true;
        }
    }
}

/// The body as lines, de-chunked if it has to be.
///
/// Ollama streams NDJSON with `Transfer-Encoding: chunked`, and a chunk boundary is **not** a line
/// boundary — a JSON line can be split across two chunks and two lines can share one. Treating them
/// as the same thing is the obvious bug here and it would only show on long replies.
pub struct Lines<R: BufRead> {
    inner: R,
    chunked: bool,
    buffer: String,
    done: bool,
}

impl<R: BufRead> Lines<R> {
    pub fn new(inner: R, chunked: bool) -> Self {
        Lines { inner, chunked, buffer: String::new(), done: false }
    }

    fn pull(&mut self) -> std::io::Result<bool> {
        if !self.chunked {
            let mut line = String::new();
            return Ok(self.inner.read_line(&mut line).map(|n| {
                self.buffer.push_str(&line);
                n > 0
            })?);
        }
        let mut header = String::new();
        if self.inner.read_line(&mut header)? == 0 {
            return Ok(false);
        }
        let size = usize::from_str_radix(header.trim().split(';').next().unwrap_or("0").trim(), 16)
            .unwrap_or(0);
        if size == 0 {
            return Ok(false);
        }
        let mut buf = vec![0u8; size];
        self.inner.read_exact(&mut buf)?;
        let mut crlf = [0u8; 2];
        let _ = self.inner.read_exact(&mut crlf);
        self.buffer.push_str(&String::from_utf8_lossy(&buf));
        Ok(true)
    }
}

impl<R: BufRead> Iterator for Lines<R> {
    type Item = std::io::Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(at) = self.buffer.find('\n') {
                let line: String = self.buffer.drain(..=at).collect();
                return Some(Ok(line.trim_end_matches(['\r', '\n']).to_string()));
            }
            if self.done {
                if self.buffer.is_empty() {
                    return None;
                }
                let rest = std::mem::take(&mut self.buffer);
                return Some(Ok(rest));
            }
            match self.pull() {
                Ok(true) => continue,
                Ok(false) => self.done = true,
                Err(e) => return Some(Err(e)),
            }
        }
    }
}
