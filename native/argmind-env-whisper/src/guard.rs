//! Whether a transcript is a thing a person said.
//!
//! **Near-silence does not make Whisper silent — it makes it talk.** Measured: silence transcribed
//! as words **120 times out of 120**. What comes back is confident, well-formed, and sometimes
//! action-shaped: a real command buried at 0 dB SNR came back as `곱창 눌러줘 한글자막 by …`, forty
//! copies long. So something has to look at the shape of the text before it becomes a turn.
//!
//! **Two guards that reject, and one that is present and rejects nothing.**
//!
//! * [`wordless`] — no letter and no digit anywhere. `... ... ... ...` is what 25 seconds of room
//!   tone produced, twice, and `.` is what five seconds produced. **No threshold is involved**: a
//!   transcript with nothing but punctuation in it is not something a person said, at any pace, in
//!   any language.
//! * [`repeating`] — a short cycle repeating inside the transcript.
//! * pace — [`argmind_voice::Heard::pace`] with a limit the caller supplies. **`None` is what
//!   ships**, because the only pace measured on this machine is a synthesiser's and a guessed
//!   cut-off that silently drops real turns is worse than none.
//!
//! **A reason, not a bool.** A build where `repeated` climbs has a recogniser problem; one where
//! `pace` climbs has a threshold problem; a single count could not tell them apart.
//!
//! Ported from `argmind/env/whisper.py`, and `expectations/hearing.json` holds that Python's own
//! verdict on every string here so the two can be compared while both exist.

/// Below this many characters, nothing is called a repetition.
///
/// **This is what protects a person repeating themselves.** `아니 그거 말고` three times is 29
/// characters and somebody could say it; it is under the floor, and that — not a margin in the
/// counting — is what saves it. Said plainly because it is the guard's weakest edge: the same
/// phrase four times over *would* be rejected.
pub const REPEAT_FLOOR: usize = 32;

/// The length of the repeating unit looked for.
///
/// **Ten and not thirty-two**, because the cycles actually observed are short — `한글자막 by ` is 8
/// characters, `ご視聴ありがとうございました` is 14, `Thank you. ` is 11 — and a probe much longer than
/// the cycle cannot count it. (**The Python said 9 for the first one and it is 8.** Counted while
/// porting; the argument it supports gets stronger rather than weaker, which is why it survived
/// being wrong for so long — a shorter cycle recurs *more* often inside a ten-character window.)
///
/// The grid was measured rather than picked: at 8 the same 7 are caught **and two of the 193
/// recorded replies go with them**; at 12 the false-positive record is still clean but
/// `Mm-hmm Thank you. Thank you. Thank you.` survives. Ten catches the most with nothing false.
pub const REPEAT_PROBE: usize = 10;

/// How many times a cycle has to occur before it is a loop. Two copies is a person saying 네, 네.
pub const REPEAT_TIMES: usize = 3;

/// Is there no letter and no digit in this at all?
pub fn wordless(text: &str) -> bool {
    //: Python asks for Unicode category L or N. `is_alphabetic()` is the Alphabetic property, which
    //: is category L plus a few marks, and `is_numeric()` is N — so this is the same question with
    //: a slightly wider yes, and `expectations/hearing.json` is what says the two agree on every
    //: string either has been shown.
    !text.chars().any(|c| c.is_alphabetic() || c.is_numeric())
}

/// Is a short cycle repeating itself inside this transcript?
///
/// **Counted at every position rather than off the tail.** The loop that mattered — a real command
/// coming back as `곱창 눌러줘 한글자막 by` forty times — drifts in its final copy, which defeats a
/// tail probe (1 occurrence) and not a count of every window (60).
pub fn repeating(text: &str) -> bool {
    let squeezed: Vec<char> = text.split_whitespace().collect::<Vec<_>>().join(" ")
        .chars().collect();
    //: **Characters, not bytes.** Python slices by character and `한글자막 by ` is 8 of them and 16
    //: bytes; a byte window would measure a different thing entirely in the language this is for.
    if squeezed.len() < REPEAT_FLOOR {
        return false;
    }
    let mut counts: std::collections::HashMap<&[char], usize> = std::collections::HashMap::new();
    let mut most = 0;
    for start in 0..=squeezed.len() - REPEAT_PROBE {
        let window = &squeezed[start..start + REPEAT_PROBE];
        let seen = counts.entry(window).or_insert(0);
        *seen += 1;
        most = most.max(*seen);
    }
    most >= REPEAT_TIMES
}

/// Why this transcript is not a turn, or `None` if it is one. **One place, and every caller uses it.**
///
/// `pace_limit` of `None` — the default, and what ships — rejects nothing on pace.
pub fn why_rejected(text: &str, seconds: f64, pace_limit: Option<f64>) -> Option<&'static str> {
    if text.trim().is_empty() {
        return Some("empty");
    }
    if wordless(text) {
        return Some("wordless");
    }
    if repeating(text) {
        return Some("repeated");
    }
    if let Some(limit) = pace_limit {
        let heard = argmind_voice::Heard { text: text.to_string(), seconds, at: 0.0 };
        if heard.pace() > limit {
            return Some("pace");
        }
    }
    None
}
