//! **MCP servers as a task's tools** — since 2026-11-08.
//!
//! The owner, on a desktop of argmind's own: *문서작성이나 도면 설계는 코딩이나 mcp로 접근하는 방법도
//! 있음.* A program driven by its buttons is the last way to reach it, not the first: a document is
//! a file a script can write (the workbench already does `.docx` and `.xlsx`), and a program that
//! publishes a Model Context Protocol server says what it can do in names and arguments, which is
//! exactly the shape a task step already has — one tool, its arguments, a result in words.
//!
//! So a server the person configured is started for a task, asked `tools/list`, and each tool it
//! names becomes a [`ToolSpec`] called `server__tool`; a call is `tools/call`, and what comes back
//! is the text of its content, verbatim, the way every other tool's result is recorded. **The
//! model still picks one tool and its arguments a step; the program still decides when to stop.**
//!
//! **What this is not.** A server runs as the person, outside `argmind-sandbox`, with whatever it
//! can reach — the same standing as `run_windows`, and weaker than `run`. Which servers exist is the
//! person's file, `mcp.json`, in the format other MCP clients use; none at rest, so at rest a task
//! is exactly what it was. `__` and not `.` between the server and the tool, because a task's `done`
//! is checked for file names and `blender.render` reads as one.
//!
//! The wire: one JSON-RPC message a line on the server's stdin and stdout (MCP 2025-06-18,
//! *stdio transport*). `initialize`, `notifications/initialized`, `tools/list` with its cursor,
//! `tools/call`. A request the server makes of argmind — roots, sampling — is answered *method not
//! found*, since argmind offers it nothing; a notification is read and dropped. Closing stdin is
//! how a server is told to stop, and it is killed only if it does not.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use argmind_core::task::{ToolSpec, Tools};
use serde_json::{json, Map, Value};

/// The protocol version argmind speaks. A server that answers with another is used anyway: the
/// four methods here have not changed shape across the published versions.
pub const PROTOCOL: &str = "2025-06-18";

/// Waiting for a server to answer `initialize`. Chosen long: `npx` fetches a package the first
/// time it runs one.
pub const START_SECONDS: u64 = 90;

/// Waiting for one tool call. Chosen: a render or a build is minutes; a task's whole budget is
/// 25 steps.
pub const CALL_SECONDS: u64 = 180;

/// Tools taken from one server, most. A server with sixty tools would be most of the prompt and
/// `crowding` says what a long list does to choosing; the rest are dropped and said so.
pub const MOST_FROM_ONE: usize = 16;

/// Characters of a tool's description kept.
const DESCRIBED: usize = 240;

/// **One configured server**, as `mcp.json` writes it — `command`, `args`, `env`.
#[derive(Clone, Debug, PartialEq)]
pub struct Configured {
    pub name: String,
    pub command: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
}

/// **The servers a file names**, in the shape Claude Desktop and most clients use:
/// `{"mcpServers": {"blender": {"command": "uvx", "args": ["blender-mcp"]}}}`. A server with
/// `"disabled": true` is left out; one with no command is skipped and named in the second list.
pub fn configured(file: &Value) -> (Vec<Configured>, Vec<String>) {
    let (mut out, mut bad) = (Vec::new(), Vec::new());
    let Some(servers) = file.get("mcpServers").and_then(Value::as_object) else { return (out, bad) };
    for (name, s) in servers {
        if s.get("disabled").and_then(Value::as_bool) == Some(true) {
            continue;
        }
        let Some(command) = s.get("command").and_then(Value::as_str).filter(|c| !c.trim().is_empty()) else {
            bad.push(format!("{name}: command 가 없다"));
            continue;
        };
        let strings = |k: &str| s.get(k).and_then(Value::as_array).map(|a| {
            a.iter().filter_map(Value::as_str).map(String::from).collect::<Vec<_>>()
        }).unwrap_or_default();
        let env = s.get("env").and_then(Value::as_object).map(|m| {
            m.iter().filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_string()))).collect()
        }).unwrap_or_default();
        out.push(Configured { name: short(name), command: command.to_string(), args: strings("args"), env });
    }
    (out, bad)
}

/// **The same, from the file's text** — for a caller that parses no JSON of its own.
pub fn configured_text(text: &str) -> Result<(Vec<Configured>, Vec<String>), String> {
    serde_json::from_str::<Value>(text).map(|v| configured(&v)).map_err(|e| e.to_string())
}

/// A server's name as part of a tool name: letters, digits and `_`, nothing else.
fn short(name: &str) -> String {
    let mut s = String::new();
    for c in name.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }) {
        //: One `_` at most in a row, so the `__` that joins server and tool is found in one place.
        if !(c == '_' && s.ends_with('_')) {
            s.push(c);
        }
    }
    let s = s.trim_matches('_').to_string();
    if s.is_empty() { "mcp".into() } else { s }
}

/// **A running server**: its stdin, a thread reading its stdout into a channel, and the last lines
/// it wrote to stderr, for when it fails.
pub struct Server {
    pub name: String,
    child: Child,
    stdin: Option<ChildStdin>,
    lines: Receiver<String>,
    stderr: Arc<Mutex<Vec<String>>>,
    next_id: u64,
    tools: Vec<(String, String, Value)>,
    pub dropped: usize,
}

impl Server {
    /// Started, initialised and listed, or a sentence saying where that stopped.
    pub fn start(c: &Configured) -> Result<Server, String> {
        let mut child = spawn(c).map_err(|e| format!("{}: 시작하지 못했다 — {e}", c.name))?;
        let stdin = child.stdin.take();
        let out = child.stdout.take().ok_or("stdout 없음")?;
        let err = child.stderr.take();
        let (tx, lines) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(out).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        let stderr = Arc::new(Mutex::new(Vec::new()));
        if let Some(err) = err {
            let keep = Arc::clone(&stderr);
            std::thread::spawn(move || {
                for line in BufReader::new(err).lines().map_while(Result::ok) {
                    if let Ok(mut k) = keep.lock() {
                        k.push(line);
                        if k.len() > 20 {
                            k.remove(0);
                        }
                    }
                }
            });
        }
        let mut s = Server { name: c.name.clone(), child, stdin, lines, stderr, next_id: 1, tools: Vec::new(), dropped: 0 };
        s.request("initialize", json!({
            "protocolVersion": PROTOCOL,
            "capabilities": {},
            "clientInfo": {"name": "argmind", "version": env!("CARGO_PKG_VERSION")},
        }), START_SECONDS)?;
        s.send(&json!({"jsonrpc": "2.0", "method": "notifications/initialized"}))?;
        let mut cursor: Option<Value> = None;
        loop {
            let params = match &cursor { Some(c) => json!({"cursor": c}), None => json!({}) };
            let r = s.request("tools/list", params, START_SECONDS)?;
            for t in r.get("tools").and_then(Value::as_array).into_iter().flatten() {
                let Some(name) = t.get("name").and_then(Value::as_str) else { continue };
                let does = t.get("description").and_then(Value::as_str).unwrap_or("").to_string();
                let schema = t.get("inputSchema").cloned().unwrap_or_else(|| json!({}));
                if s.tools.len() < MOST_FROM_ONE {
                    s.tools.push((name.to_string(), does, schema));
                } else {
                    s.dropped += 1;
                }
            }
            cursor = r.get("nextCursor").filter(|c| !c.is_null()).cloned();
            if cursor.is_none() {
                break;
            }
        }
        Ok(s)
    }

    fn send(&mut self, message: &Value) -> Result<(), String> {
        let stdin = self.stdin.as_mut().ok_or("이미 닫혔다")?;
        writeln!(stdin, "{message}").and_then(|_| stdin.flush()).map_err(|e| format!("{}: 보내지 못했다 — {e}{}", self.name, self.last_words()))
    }

    fn last_words(&self) -> String {
        let k = self.stderr.lock().map(|k| k.join("\n")).unwrap_or_default();
        if k.trim().is_empty() { String::new() } else { format!("\n[stderr]\n{}", k.trim()) }
    }

    /// One request, and its `result` — reading past notifications and answering the server's own
    /// requests, until the answer with this id comes or `seconds` pass.
    fn request(&mut self, method: &str, params: Value, seconds: u64) -> Result<Value, String> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))?;
        let until = Instant::now() + Duration::from_secs(seconds);
        loop {
            let left = until.saturating_duration_since(Instant::now());
            let line = match self.lines.recv_timeout(left) {
                Ok(l) => l,
                Err(RecvTimeoutError::Timeout) => return Err(format!("{}: {method} 에 {seconds}초 안에 답이 없다", self.name)),
                Err(RecvTimeoutError::Disconnected) => return Err(format!("{}: 서버가 끝났다{}", self.name, self.last_words())),
            };
            //: A line that is not JSON is a server printing to the wrong stream; skipped.
            let Ok(m) = serde_json::from_str::<Value>(line.trim()) else { continue };
            let has_method = m.get("method").is_some();
            match (m.get("id"), has_method) {
                //: The server asking argmind something: nothing is offered, so nothing is found.
                (Some(their), true) => {
                    let their = their.clone();
                    self.send(&json!({"jsonrpc": "2.0", "id": their,
                                      "error": {"code": -32601, "message": "argmind offers no such method"}}))?;
                }
                (None, true) => {}
                (Some(got), false) if got.as_u64() == Some(id) => {
                    if let Some(e) = m.get("error") {
                        let text = e.get("message").and_then(Value::as_str).unwrap_or("오류");
                        return Err(format!("{}: {text}", self.name));
                    }
                    return Ok(m.get("result").cloned().unwrap_or(Value::Null));
                }
                _ => {}
            }
        }
    }

    /// The tools this server offers, as a task sees them.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.iter().map(|(name, does, schema)| ToolSpec {
            name: format!("{}__{name}", self.name),
            does: described(does, schema),
            args: example(schema),
        }).collect()
    }

    /// `tools/call`, and the result's text. `isError` is a failure the task counts.
    pub fn call(&mut self, tool: &str, args: &Value) -> Result<String, String> {
        let args = if args.is_object() { args.clone() } else { json!({}) };
        let r = self.request("tools/call", json!({"name": tool, "arguments": args}), CALL_SECONDS)?;
        let text = content_text(&r);
        if r.get("isError").and_then(Value::as_bool) == Some(true) { Err(text) } else { Ok(text) }
    }
}

impl Drop for Server {
    //: **Closing stdin is the protocol's *stop***; a server that has not gone in two seconds is
    //: killed. `npx` puts a `node` under a `cmd`, and killing the `cmd` would leave the `node`.
    fn drop(&mut self) {
        self.stdin.take();
        let until = Instant::now() + Duration::from_secs(2);
        while Instant::now() < until {
            if let Ok(Some(_)) = self.child.try_wait() {
                return;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn(c: &Configured) -> std::io::Result<Child> {
    let go = |program: &str| {
        let mut cmd = Command::new(program);
        cmd.args(&c.args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        for (k, v) in &c.env {
            cmd.env(k, v);
        }
        cmd.spawn()
    };
    match go(&c.command) {
        //: **`npx` and `uvx` are scripts on one system**: a `.cmd` beside no `.exe`, which a bare
        //: name does not find there. Tried once more with the suffix, and only when not found.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && cfg!(windows) && !c.command.contains('.') => {
            go(&format!("{}.cmd", c.command)).map_err(|_| e)
        }
        r => r,
    }
}

/// What a result says in words: its text items, and what each other item is.
pub fn content_text(result: &Value) -> String {
    let mut parts: Vec<String> = Vec::new();
    for c in result.get("content").and_then(Value::as_array).into_iter().flatten() {
        match c.get("type").and_then(Value::as_str) {
            Some("text") => parts.push(c.get("text").and_then(Value::as_str).unwrap_or("").to_string()),
            Some("image") => parts.push("(그림 하나 — 여기서는 볼 수 없다)".into()),
            Some("audio") => parts.push("(소리 하나 — 여기서는 들을 수 없다)".into()),
            Some("resource") => {
                let r = c.get("resource").cloned().unwrap_or_default();
                parts.push(r.get("text").and_then(Value::as_str).map(String::from)
                    .unwrap_or_else(|| format!("(자료: {})", r.get("uri").and_then(Value::as_str).unwrap_or("?"))));
            }
            Some("resource_link") => parts.push(format!("(자료: {})",
                c.get("uri").and_then(Value::as_str).unwrap_or("?"))),
            _ => {}
        }
    }
    //: **Structured content with no text beside it** is still an answer; the spec asks a server to
    //: send both, and not every one does.
    if parts.iter().all(|p| p.trim().is_empty()) {
        if let Some(s) = result.get("structuredContent") {
            return s.to_string();
        }
    }
    parts.join("\n")
}

/// The description, cut, and each argument with its type and meaning — what the schema says that
/// the example cannot.
fn described(does: &str, schema: &Value) -> String {
    let mut d: String = does.split_whitespace().collect::<Vec<_>>().join(" ");
    if d.chars().count() > DESCRIBED {
        d = d.chars().take(DESCRIBED).collect::<String>() + "…";
    }
    let required: Vec<&str> = schema.get("required").and_then(Value::as_array).into_iter().flatten()
        .filter_map(Value::as_str).collect();
    let mut args: Vec<String> = Vec::new();
    for (k, p) in schema.get("properties").and_then(Value::as_object).into_iter().flatten() {
        let kind = p.get("type").and_then(Value::as_str).unwrap_or("값");
        let mut a = format!("{k}: {kind}{}", if required.contains(&k.as_str()) { "" } else { ", 생략 가능" });
        if let Some(e) = p.get("enum").and_then(Value::as_array) {
            let e: Vec<String> = e.iter().take(8).map(|v| v.to_string()).collect();
            a.push_str(&format!(" ({})", e.join("|")));
        }
        args.push(a);
    }
    if args.is_empty() { d } else { format!("{d} [인자 — {}]", args.join("; ")) }
}

/// **An example of the arguments**, built from the schema, since a task step is shown one: the
/// required ones, or every one when none is required, each from its `default`, its first
/// `examples` or `enum` entry, or a placeholder of its type.
pub fn example(schema: &Value) -> Value {
    let Some(props) = schema.get("properties").and_then(Value::as_object) else { return json!({}) };
    let required: Vec<&str> = schema.get("required").and_then(Value::as_array).into_iter().flatten()
        .filter_map(Value::as_str).collect();
    let mut out = Map::new();
    for (k, p) in props {
        if !required.is_empty() && !required.contains(&k.as_str()) {
            continue;
        }
        let v = p.get("default").cloned()
            .or_else(|| p.get("examples").and_then(Value::as_array).and_then(|e| e.first().cloned()))
            .or_else(|| p.get("enum").and_then(Value::as_array).and_then(|e| e.first().cloned()))
            .unwrap_or_else(|| match p.get("type").and_then(Value::as_str) {
                Some("number") | Some("integer") => json!(1),
                Some("boolean") => json!(true),
                Some("array") => json!([]),
                Some("object") => json!({}),
                _ => json!("..."),
            });
        out.insert(k.clone(), v);
    }
    Value::Object(out)
}

/// **Every configured server, started, as one tool set** — and what could not be started, in
/// words, for the timeline. Dropped at the end of the task, which stops them.
pub struct Servers {
    pub servers: Vec<Server>,
}

impl Servers {
    pub fn start(configured: &[Configured]) -> (Servers, Vec<String>) {
        let (mut servers, mut failed) = (Vec::new(), Vec::new());
        for c in configured {
            match Server::start(c) {
                Ok(s) => servers.push(s),
                Err(e) => failed.push(e),
            }
        }
        (Servers { servers }, failed)
    }
}

impl Tools for Servers {
    fn list(&self) -> Vec<ToolSpec> {
        self.servers.iter().flat_map(Server::specs).collect()
    }
    fn call(&mut self, tool: &str, args: &Value) -> Result<String, String> {
        let Some((server, name)) = tool.split_once("__") else { return Err(format!("그런 도구는 없다: {tool}")) };
        let s = self.servers.iter_mut().find(|s| s.name == server)
            .ok_or_else(|| format!("그런 서버는 없다: {server}"))?;
        if !s.tools.iter().any(|(n, _, _)| n == name) {
            return Err(format!("{server} 에 그런 도구는 없다: {name}"));
        }
        s.call(name, args)
    }
}
