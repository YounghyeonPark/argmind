//! Reading a reply. **Not the client** — `argmind/core/llm.py`'s HTTP half is an environment that
//! happens to live in `core/`, dead on a phone with no Ollama, and is deliberately not ported.
//! What is here is the part any core needs: finding the JSON object in what came back.

/// Where the first *balanced* JSON object starts and ends, fed either a whole reply or a stream.
///
/// **One scanner with two callers.** The extractor needs it to cut the object out; a streaming stop
/// needs it to know an answer is complete while tokens are still arriving. Written twice, the two
/// would eventually disagree about what complete means, and the stop would cut a reply at a point
/// the extractor then refuses.
///
/// This was once a non-greedy regex, correct for exactly as long as no reply contained a nested
/// object — and it stopped being correct the moment answers grew a list of things to do, so
/// **every action silently disappeared while the model was answering perfectly**.
///
/// **Indices are character indices, never bytes.** 94 of the 128 recorded replies are non-ASCII;
/// a scanner over `as_bytes()` agrees on every ASCII fixture and is wrong on three quarters of the
/// real ones.
#[derive(Default)]
pub struct Balanced {
    pub start: i64,
    pub end: i64,
    depth: i64,
    in_string: bool,
    escaped: bool,
    at: i64,
}

impl Balanced {
    pub fn new() -> Self {
        Balanced { start: -1, end: -1, depth: 0, in_string: false, escaped: false, at: 0 }
    }

    pub fn closed(&self) -> bool {
        self.end >= 0
    }

    /// Consume more text; returns whether a whole object has now been seen. Indices are into
    /// everything fed so far, so streaming and handing over the finished text agree.
    pub fn feed(&mut self, chunk: &str) -> bool {
        for ch in chunk.chars() {
            let here = self.at;
            self.at += 1;
            if self.end >= 0 {
                continue;
            }
            if self.start < 0 {
                if ch != '{' {
                    continue;
                }
                self.start = here;
                //: Deliberate fall-through: this same '{' takes the depth to 1.
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
                //: Quotes and escapes are tracked because a brace inside a string is not a brace —
                //: `{"say": "press } to close"}` would otherwise close one level early.
                self.in_string = true;
            } else if ch == '{' {
                self.depth += 1;
            } else if ch == '}' {
                self.depth -= 1;
                if self.depth == 0 {
                    self.end = here;
                }
            }
        }
        self.end >= 0
    }
}

/// The first balanced JSON object in a reply, parsed — or nothing.
///
/// Extracted rather than parsed whole, because a model asked for JSON usually returns JSON and
/// occasionally returns JSON with a sentence in front of it.
pub fn first_object(text: &str) -> Option<serde_json::Value> {
    let mut scan = Balanced::new();
    if !scan.feed(text) {
        return None;
    }
    let slice: String = text
        .chars()
        .skip(scan.start as usize)
        .take((scan.end - scan.start + 1) as usize)
        .collect();
    serde_json::from_str(&slice).ok()
}
