//! Asking Ollama what it has, and asking it to fetch what it does not.
//!
//! **This is the one install that is safe to do without asking twice**, and the reason is worth
//! stating: it runs no installer, downloads no executable and touches nothing outside a program the
//! person already chose to run. `POST /api/pull` is Ollama fetching a model into its own store,
//! over the same socket this project already talks to it on — the same shape as every other call
//! here, with no new dependency and nothing to audit.
//!
//! Everything else argmind might want — Ollama itself, a recogniser, a language pack — means
//! running an installer or elevating, and none of that happens from here. Those are reported and
//! the exact command is printed; a program that can click your screen should not also be able to
//! install software while you are not looking.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

use crate::{py_string, Lines};

fn split(host: &str) -> (String, u16) {
    let bare = host.trim_start_matches("http://").trim_start_matches("https://");
    match bare.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().unwrap_or(11434)),
        None => (bare.to_string(), 11434),
    }
}

fn request(host: &str, method: &str, path: &str, body: Option<&str>)
           -> std::io::Result<(TcpStream, BufReader<TcpStream>)> {
    let (h, port) = split(host);
    let stream = TcpStream::connect((h.as_str(), port))?;
    let mut w = &stream;
    write!(w, "{method} {path} HTTP/1.1\r\nHost: {h}:{port}\r\nConnection: close\r\n")?;
    match body {
        Some(b) => {
            write!(w, "Content-Type: application/json\r\nContent-Length: {}\r\n\r\n", b.len())?;
            w.write_all(b.as_bytes())?;
        }
        None => write!(w, "\r\n")?,
    }
    w.flush()?;
    let reading = BufReader::new(stream.try_clone()?);
    Ok((stream, reading))
}

fn headers(reader: &mut BufReader<TcpStream>) -> std::io::Result<bool> {
    let mut chunked = false;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        if line.to_lowercase().starts_with("transfer-encoding:") && line.contains("chunked") {
            chunked = true;
        }
        if line == "\r\n" {
            break;
        }
    }
    Ok(chunked)
}

/// Is Ollama answering at all?
pub fn awake(host: &str) -> bool {
    installed(host).is_ok()
}

/// Every model Ollama already has, by name.
pub fn installed(host: &str) -> Result<Vec<String>, String> {
    let (_s, mut reader) = request(host, "GET", "/api/tags", None)
        .map_err(|e| format!("{e}"))?;
    headers(&mut reader).map_err(|e| format!("{e}"))?;
    let mut body = String::new();
    reader.read_to_string(&mut body).map_err(|e| format!("{e}"))?;
    //: Read out of the body rather than parsed into a document — one field, and a JSON parser
    //: would be a dependency for it. Names come back as `"name":"qwen3:8b"`.
    Ok(names_in(&body))
}

/// Every `"name"` field in a body, in order.
pub fn names_in(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(at) = rest.find("\"name\":") {
        rest = &rest[at + 7..];
        let Some(open) = rest.find('"') else { break };
        let Some(close) = rest[open + 1..].find('"') else { break };
        out.push(rest[open + 1..open + 1 + close].to_string());
        rest = &rest[open + 1 + close..];
    }
    out
}

/// **Is this model here, under any of the names Ollama knows it by?**
///
/// `qwen3:8b` and `qwen3:8b` are the same thing, but a person types `qwen3` and Ollama stores
/// `qwen3:latest` — so a plain equality check reports a model as missing that is sitting right
/// there and starts a five gigabyte download for it.
pub fn has(host: &str, model: &str) -> bool {
    let Ok(names) = installed(host) else { return false };
    let want = if model.contains(':') { model.to_string() } else { format!("{model}:latest") };
    names.iter().any(|n| *n == want || *n == model)
}

/// How far a pull has got.
#[derive(Debug, Clone, PartialEq)]
pub struct Progress {
    pub status: String,
    pub completed: u64,
    pub total: u64,
}

impl Progress {
    /// A whole-number percentage, or `None` when the server has not said how big it is yet.
    pub fn percent(&self) -> Option<u64> {
        (self.total > 0).then(|| self.completed.saturating_mul(100) / self.total)
    }
}

/// One line of pull progress.
pub fn progress_of(line: &str) -> Option<Progress> {
    let field = |name: &str| -> Option<u64> {
        let at = line.find(&format!("\"{name}\":"))? + name.len() + 3;
        let rest = &line[at..];
        let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
        rest[..end].parse().ok()
    };
    let at = line.find("\"status\":")? + 9;
    let rest = &line[at..];
    let open = rest.find('"')? + 1;
    let close = rest[open..].find('"')? + open;
    Some(Progress { status: rest[open..close].to_string(),
                    completed: field("completed").unwrap_or(0),
                    total: field("total").unwrap_or(0) })
}

/// Ask Ollama to fetch a model, reporting progress as it goes.
///
/// **It runs no installer and downloads no executable.** This is Ollama putting a model in its own
/// store, over the socket this project already uses — which is why it is the one thing `setup` does
/// without a second question.
pub fn pull(host: &str, model: &str, mut on: impl FnMut(&Progress)) -> Result<(), String> {
    let body = format!("{{\"model\": {}, \"stream\": true}}", py_string(model));
    let (_s, mut reader) = request(host, "POST", "/api/pull", Some(&body))
        .map_err(|e| format!("could not reach ollama at {host} ({e})"))?;
    let chunked = headers(&mut reader).map_err(|e| format!("{e}"))?;
    let mut last = String::new();
    for line in Lines::new(reader, chunked) {
        let line = line.map_err(|e| format!("{e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        //: **An error comes back on the same stream as the progress**, so a pull that failed looks
        //: exactly like one that is still going unless this is read.
        if let Some(at) = line.find("\"error\":") {
            let rest = &line[at + 8..];
            let open = rest.find('"').map(|i| i + 1).unwrap_or(0);
            let close = rest[open..].find('"').map(|i| i + open).unwrap_or(rest.len());
            return Err(rest[open..close].to_string());
        }
        if let Some(p) = progress_of(&line) {
            last = p.status.clone();
            on(&p);
        }
    }
    //: Ollama's last word on a good pull is `success`. Anything else means the stream ended
    //: without one, which is not the same as finishing.
    if last == "success" {
        Ok(())
    } else {
        Err(format!("the pull ended on {last:?} rather than \"success\""))
    }
}
