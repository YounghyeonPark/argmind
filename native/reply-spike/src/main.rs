//! Stage 2 of the port: does a Rust `_Balanced` find the same object as the Python one?
//!
//! `native/prompt-spike` showed the prompt goes out byte-identically. This is the other end of the
//! call — the scanner that decides where the model's JSON object starts and stops, and therefore
//! whether a turn produces an action at all. It is the piece with the worst history in this
//! repository: it used to be `re.search(r"\{.*?\}")`, which matched up to the *inner* brace the
//! moment a reply grew a list of things to do, so `Reply.json()` returned `None` and **every
//! action silently disappeared while the model was answering perfectly**.
//!
//! Verified against `measure/replies-*.json` — 128 real recorded replies, not invented ones. The
//! expectations are produced by the shipped Python and carried in the input file, so this compares
//! two implementations over the same bytes rather than over two people's idea of a hard case.
//!
//! **The trap this corpus is uniquely able to catch**: `_Balanced` iterates a Python `str`, so
//! `start` and `end` are *character* indices and `Reply.json()` slices with them. 94 of the 128
//! replies are non-ASCII — mostly Korean, where a code point is three bytes — so a scanner written
//! over `as_bytes()` agrees on every ASCII fixture and is wrong on three quarters of the real ones.

use std::env;
use std::fs;

/// Where the first *balanced* JSON object starts and ends. A port of `argmind/core/llm.py`'s
/// `_Balanced`, kept line for line: quotes and escapes are tracked because a brace inside a string
/// is not a brace — `{"say": "press } to close"}` would otherwise close one level early.
struct Balanced {
    start: i64,
    end: i64,
    depth: i64,
    in_string: bool,
    escaped: bool,
    at: i64,
}

impl Balanced {
    fn new() -> Self {
        Balanced { start: -1, end: -1, depth: 0, in_string: false, escaped: false, at: 0 }
    }

    /// Consume more text; returns whether a whole object has now been seen. Indices are into
    /// everything fed so far, so streaming and handing over the finished text agree.
    fn feed(&mut self, chunk: &str) -> bool {
        //: `.chars()`, never `.as_bytes()`. See the module docstring — this is the whole point.
        for ch in chunk.chars() {
            let at = self.at;
            self.at += 1;
            if self.end >= 0 {
                continue;
            }
            if self.start < 0 {
                if ch != '{' {
                    continue;
                }
                self.start = at;
                //: Deliberate fall-through, as in the Python: this same '{' takes depth to 1.
            }
            if self.in_string {
                if self.escaped {
                    self.escaped = false;
                } else if ch == '\\' {
                    self.escaped = true;
                } else if ch == '"' {
                    self.in_string = false;
                }
            } else if ch == '"' {
                self.in_string = true;
            } else if ch == '{' {
                self.depth += 1;
            } else if ch == '}' {
                self.depth -= 1;
                if self.depth == 0 {
                    self.end = at;
                }
            }
        }
        self.end >= 0
    }
}

/// `Reply.json()`: the first balanced object, parsed, or nothing.
fn reply_json(text: &str) -> (i64, i64, bool) {
    let mut scan = Balanced::new();
    if !scan.feed(text) {
        return (scan.start, scan.end, false);
    }
    let slice: String = text
        .chars()
        .skip(scan.start as usize)
        .take((scan.end - scan.start + 1) as usize)
        .collect();
    let ok = serde_json::from_str::<serde_json::Value>(&slice).is_ok();
    (scan.start, scan.end, ok)
}

fn main() {
    let path = env::args().nth(1).expect("usage: reply-spike <replies.json>");
    let raw = fs::read_to_string(&path).expect("read");
    let cases: serde_json::Value = serde_json::from_str(&raw).expect("parse");
    let cases = cases.as_array().expect("an array of cases");

    let (mut n, mut bad_index, mut bad_parse, mut wide, mut chunked) = (0, 0, 0, 0, 0);
    for case in cases {
        n += 1;
        let text = case["text"].as_str().unwrap_or("");
        let want_start = case["start"].as_i64().unwrap_or(-1);
        let want_end = case["end"].as_i64().unwrap_or(-1);
        let want_parse = case["reply_json_ok"].as_bool().unwrap_or(false);
        if text.len() != text.chars().count() {
            wide += 1;
        }

        let (start, end, ok) = reply_json(text);
        if start != want_start || end != want_end {
            bad_index += 1;
            if bad_index <= 5 {
                println!("  INDEX  {}  python ({}, {})  rust ({}, {})",
                         case["key"].as_str().unwrap_or("?"), want_start, want_end, start, end);
            }
        }
        if ok != want_parse {
            bad_parse += 1;
            if bad_parse <= 5 {
                println!("  PARSE  {}  python {}  rust {}",
                         case["key"].as_str().unwrap_or("?"), want_parse, ok);
            }
        }

        //: **Streaming must land on the same two numbers as one shot**, which is the property
        //: `Local.ask` relies on and the one a port is most likely to break. Fed one character at
        //: a time, which is the harshest split there is.
        let mut piece = Balanced::new();
        for ch in text.chars() {
            piece.feed(&ch.to_string());
        }
        if piece.start != start || piece.end != end {
            chunked += 1;
        }
    }

    println!("\n  {n} recorded replies, {wide} of them non-ASCII");
    println!("  disagreements on (start, end) : {bad_index}");
    println!("  disagreements on parsing      : {bad_parse}");
    println!("  one-char-at-a-time != one shot: {chunked}");
    if bad_index == 0 && bad_parse == 0 && chunked == 0 {
        println!("\n  IDENTICAL — the Rust scanner finds the same object in every recorded reply.");
    } else {
        println!("\n  DIFFERENT — the port is not equivalent.");
        std::process::exit(1);
    }
}
