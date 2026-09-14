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

// ---- reading a stream, and when to stop reading it ------------------------------------------------

/// The server's own reasons, and this reader's two.
pub const FINISHED: &str = "stop";
pub const CEILING: &str = "length";
pub const CLOSED: &str = "closed";
pub const DEGENERATE: &str = "degenerate";

/// How many content chunks to keep reading after the object has closed.
///
/// **A gap no output occupies, and that is the honest description.** Measured over 128 recorded
/// replies: 124 close an envelope, and of those **109 emit nothing at all afterwards, 13 emit
/// exactly one chunk, and 2 emit 505**. So 2 is a floor held by real output — `GRACE = 1` turns
/// those 13 into `closed` with no server token counts — and every value from 2 to 600 behaves
/// identically on the shipped arm. 24 sits in the gap; moving it buys nothing measured.
pub const GRACE: i64 = 24;

/// Characters, not tokens, because characters are what the extractor reads and a chunk boundary is
/// the server's business.
pub const LOOP_FLOOR: usize = 256;
pub const LOOP_WINDOW: usize = 512;
pub const LOOP_PROBE: usize = 32;
pub const LOOP_TIMES: usize = 4;

/// Is the tail of this reply a short cycle repeating itself?
///
/// Non-overlapping occurrences of the tail rather than an exact period over the window, so a cycle
/// that drifts slightly — a changing index, a stray space — is still caught.
///
/// **It is a length backstop, not a classifier**, and the file it came from says so: the pathology
/// it was built for and its one measured false positive are the same shape, and what separated a
/// stopped reply from an identical one that was not was whether the repeating unit happened to
/// align to `LOOP_PROBE` characters.
pub fn looping(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < LOOP_FLOOR {
        return false;
    }
    let window: String = chars[chars.len().saturating_sub(LOOP_WINDOW)..].iter().collect();
    let probe: String = chars[chars.len() - LOOP_PROBE..].iter().collect();
    //: `str::matches` is non-overlapping, which is the same counting `str.count` does in Python.
    window.matches(&probe).count() >= LOOP_TIMES
}

/// What came back, and what it cost — both, always. A latency figure detached from the call it
/// describes is how a stale number gets published.
#[derive(Debug, Clone)]
pub struct Reply {
    pub text: String,
    /// `None` whenever the reason is one of this reader's own stops: the server puts its prompt
    /// count only in the final message, and a read that ended early never saw one. **Absent rather
    /// than zero** — a prompt of unknown size and a prompt of no size are not the same fact, and
    /// the second is impossible.
    pub prompt_tokens: Option<i64>,
    pub out_tokens: Option<i64>,
    pub reason: String,
    pub thought: usize,
}

/// The reading of one reply, line by line — **and the two decisions about when to stop.**
///
/// A state machine rather than a loop so it can be replayed offline: 128 recorded streams go back
/// through it with no server, which is the only way the stops have ever been tested against real
/// output. The alternative was streams somebody imagined.
pub struct Reading {
    pub stop_when_answered: bool,
    pub stop_when_looping: bool,
    text: String,
    envelope: Balanced,
    spare: i64,
    arrived: i64,
    thinking: usize,
    ended: Option<&'static str>,
    last: Option<(Option<i64>, Option<i64>, String)>,
    pub lines_read: usize,
}

impl Reading {
    pub fn new(stop_when_answered: bool, stop_when_looping: bool) -> Self {
        Reading { stop_when_answered, stop_when_looping, text: String::new(),
                  envelope: Balanced::new(), spare: GRACE, arrived: 0, thinking: 0,
                  ended: None, last: None, lines_read: 0 }
    }

    /// One line of the stream. Returns whether to keep reading.
    ///
    /// `thinking` is a **length** rather than the text, because `ask` only ever takes `len()` of it
    /// and truthiness from it — recording the words would be recording something nothing reads.
    pub fn line(&mut self, thinking: usize, chunk: Option<&str>,
                done: Option<(Option<i64>, Option<i64>, String)>) -> bool {
        self.lines_read += 1;
        self.thinking += thinking;
        if thinking > 0 || chunk.map(|c| !c.is_empty()).unwrap_or(false) {
            self.arrived += 1;
        }
        if let Some(chunk) = chunk.filter(|c| !c.is_empty()) {
            self.text.push_str(chunk);
            if self.envelope.closed() {
                self.spare -= 1;
                if self.stop_when_answered && self.spare <= 0 {
                    self.ended = Some(CLOSED);
                    return false;
                }
            } else if !self.envelope.feed(chunk) && self.stop_when_looping && looping(&self.text) {
                self.ended = Some(DEGENERATE);
                return false;
            }
        }
        if let Some(d) = done {
            self.last = Some(d);
        }
        true
    }

    /// A blank line — the server sends them and `ask` skips them before anything else. Counted,
    /// because `lines_read` is a count of what was pulled from the stream.
    pub fn blank(&mut self) {
        self.lines_read += 1;
    }

    pub fn finish(self) -> Reply {
        let text = self.text.trim().to_string();
        match self.ended {
            None => {
                let (prompt, out, reason) = self.last
                    .unwrap_or((None, None, String::new()));
                Reply { text, prompt_tokens: prompt, out_tokens: out, reason, thought: self.thinking }
            }
            //: A stop of this reader's own: no server counts, and `arrived` stands in for the
            //: output count, which is the only quantity it can honestly report.
            Some(reason) => Reply { text, prompt_tokens: None, out_tokens: Some(self.arrived),
                                    reason: reason.to_string(), thought: self.thinking },
        }
    }
}
