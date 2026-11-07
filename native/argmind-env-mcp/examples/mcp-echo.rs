//! **A tiny MCP server for the tests**, over stdio: two tools across two pages of `tools/list`, a
//! notification before every answer and a request of its own before the first call, so the
//! client's reading past both is what gets tested. Ends when stdin closes, as the protocol says.
//!
//!     cargo run -p argmind-env-mcp --example mcp-echo

use std::io::{BufRead, Write};

use serde_json::{json, Value};

fn main() {
    let stdin = std::io::stdin();
    let mut out = std::io::stdout();
    let mut asked = false;
    let say = |v: Value, out: &mut std::io::Stdout| {
        writeln!(out, "{v}").unwrap();
        out.flush().unwrap();
    };
    eprintln!("mcp-echo: started");
    loop {
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 { break }
        let Ok(m) = serde_json::from_str::<Value>(&line) else { continue };
        let (Some(id), Some(method)) = (m.get("id").cloned(), m.get("method").and_then(Value::as_str)) else { continue };
        say(json!({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "x"}}), &mut out);
        let result = match method {
            "initialize" => json!({"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                                   "serverInfo": {"name": "mcp-echo", "version": "0"}}),
            "tools/list" if m["params"].get("cursor").is_none() => json!({"nextCursor": "2", "tools": [
                {"name": "add", "description": "두 수를 더한다",
                 "inputSchema": {"type": "object", "required": ["a", "b"],
                                 "properties": {"a": {"type": "number"}, "b": {"type": "number"},
                                                "note": {"type": "string"}}}}]}),
            "tools/list" => json!({"tools": [
                {"name": "fail", "description": "늘 실패한다", "inputSchema": {"type": "object"}}]}),
            "tools/call" => {
                //: **And waits for the answer**, as a real server asking for roots does — so a
                //: client that leaves a server's request unanswered hangs here and fails.
                if !asked {
                    asked = true;
                    say(json!({"jsonrpc": "2.0", "id": "srv-1", "method": "roots/list"}), &mut out);
                    let mut answer = String::new();
                    if stdin.lock().read_line(&mut answer).is_err() || !answer.contains("srv-1") {
                        say(json!({"jsonrpc": "2.0", "id": id, "error": {"code": -1, "message": "roots unanswered"}}), &mut out);
                        continue;
                    }
                }
                let a = &m["params"]["arguments"];
                match m["params"]["name"].as_str() {
                    Some("add") => json!({"content": [{"type": "text", "text":
                        format!("{}", a["a"].as_f64().unwrap_or(0.0) + a["b"].as_f64().unwrap_or(0.0))}]}),
                    _ => json!({"isError": true, "content": [{"type": "text", "text": "일부러 실패"}]}),
                }
            }
            _ => {
                say(json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "no"}}), &mut out);
                continue;
            }
        };
        say(json!({"jsonrpc": "2.0", "id": id, "result": result}), &mut out);
    }
}
