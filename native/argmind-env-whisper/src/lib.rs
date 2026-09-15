//! Transcription, over HTTP, to a whisper.cpp server on this machine.
//!
//! **Why a server and not a binding.** This project already speaks raw HTTP/1.1 to a local model
//! with no dependency at all — `argmind-env-ollama` — and a recogniser is the same shape of thing:
//! something you install, that runs on your machine, that answers on a socket. Doing it this way
//! costs **no new crate, no FFI, no C++ in this tree and no build script**, and it puts the choice
//! of CPU or Vulkan or anything else where it belongs: in how you built the server, not in how we
//! built the client.
//!
//! It also settles a question the toolchain forced. There is no MSVC and no Vulkan SDK here, so a
//! vendored whisper.cpp would build CPU-only against MinGW — and the Python it replaces answers in
//! 234 ms on the GPU. Owning that build would have meant owning a large regression; borrowing the
//! server means the person who wants the GPU installs the SDK once and this file never learns
//! about it.
//!
//! **Why not the recogniser already in Windows.** SAPI is here and free and speaks on this machine
//! already. It has one recogniser installed — `MS-1033-80-DESK`, English — and the primary language
//! of this project is Korean. A recogniser that cannot hear Korean is not a recogniser for this.
//!
//! ## What is not settled
//!
//! The measured pathology this whole guard structure exists for — **silence transcribed as words,
//! 120 times out of 120** — was measured against `whisper-large-v3-turbo` through transformers. It
//! is a Whisper family trait and it very probably transfers, but *probably* is not measured, and
//! the thresholds that would reject a fabrication still need a real recording session to set. Until
//! then this reports what it got and rejects nothing on confidence it does not have.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;

pub const WHISPER: &str = "127.0.0.1:8178";

/// 16-bit PCM WAV around 16 kHz mono samples.
///
/// Written out rather than reached for: a WAV header is forty-four bytes of little-endian integers,
/// and a crate for it would be a dependency to audit for something a test can check byte for byte.
pub fn wav(samples: &[f32], hz: u32) -> Vec<u8> {
    let data_len = samples.len() * 2;
    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&hz.to_le_bytes());
    out.extend_from_slice(&(hz * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for s in samples {
        //: **Clamped before scaling.** A sample slightly outside the range — which resampling can
        //: produce — wraps rather than saturating when it is cast, and a wrap is a loud click in
        //: the middle of a word.
        let clamped = s.clamp(-1.0, 1.0);
        out.extend_from_slice(&((clamped * 32767.0) as i16).to_le_bytes());
    }
    out
}

/// One `multipart/form-data` body: the audio, and the fields beside it.
pub fn multipart(boundary: &str, audio: &[u8], fields: &[(&str, &str)]) -> Vec<u8> {
    let mut out = Vec::new();
    for (name, value) in fields {
        out.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n\
                     {value}\r\n").as_bytes());
    }
    out.extend_from_slice(
        format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; \
                 filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n").as_bytes());
    out.extend_from_slice(audio);
    out.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    out
}

/// The transcript out of whisper.cpp's JSON.
///
/// **Read out of the body rather than parsed into a document.** The server answers
/// `{"text": "..."}`, sometimes with segments beside it; what is wanted is one string, and a JSON
/// parser here would be a dependency for one field.
pub fn transcript_of(body: &str) -> Option<String> {
    let at = body.find("\"text\"")?;
    let rest = &body[at + 6..];
    let open = rest.find('"')? + 1;
    let mut out = String::new();
    let mut chars = rest[open..].chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out.trim().to_string()),
            '\\' => match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('u') => {
                    //: A `\uXXXX` escape, which is how a server with `ensure_ascii` sends Korean.
                    let hex: String = chars.by_ref().take(4).collect();
                    match u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) {
                        Some(c) => out.push(c),
                        None => return None,
                    }
                }
                Some(other) => out.push(other),
                None => return None,
            },
            c => out.push(c),
        }
    }
    //: **An unterminated string is not a transcript.** Returning what was read so far would hand
    //: back half a sentence and call it what the person said.
    None
}

/// A whisper.cpp server on this machine.
pub struct Whisper {
    pub host: String,
    /// `None` lets the server guess. **Naming it is better**: this project's turns are Korean, and
    /// a recogniser guessing the language of a two-second clip guesses wrong on short ones.
    pub language: Option<String>,
    pub asked: u64,
    pub failed: u64,
}

impl Default for Whisper {
    fn default() -> Self {
        Whisper { host: WHISPER.to_string(), language: Some("ko".into()), asked: 0, failed: 0 }
    }
}

impl Whisper {
    /// Send one clip and get back what it heard.
    pub fn hear(&mut self, samples: &[f32], hz: u32) -> Result<String, String> {
        self.asked += 1;
        let boundary = "argmind0aA0argmind";
        let language = self.language.clone().unwrap_or_default();
        let mut fields: Vec<(&str, &str)> = vec![("response_format", "json"),
                                                 ("temperature", "0")];
        if !language.is_empty() {
            fields.push(("language", &language));
        }
        let body = multipart(boundary, &wav(samples, hz), &fields);
        let answer = self.post(&body, boundary).map_err(|e| {
            self.failed += 1;
            e
        })?;
        transcript_of(&answer).ok_or_else(|| {
            self.failed += 1;
            format!("no transcript in the answer: {}", answer.chars().take(200)
                    .collect::<String>())
        })
    }

    fn post(&self, body: &[u8], boundary: &str) -> Result<String, String> {
        let (host, port) = match self.host.rsplit_once(':') {
            Some((h, p)) => (h.to_string(), p.parse::<u16>().map_err(|e| e.to_string())?),
            None => (self.host.clone(), 80),
        };
        let stream = TcpStream::connect((host.as_str(), port))
            .map_err(|e| format!("no whisper server at {} ({e})", self.host))?;
        let mut w = &stream;
        write!(w, "POST /inference HTTP/1.1\r\nHost: {host}:{port}\r\n\
                   Content-Type: multipart/form-data; boundary={boundary}\r\n\
                   Content-Length: {}\r\nConnection: close\r\n\r\n", body.len())
            .map_err(|e| e.to_string())?;
        w.write_all(body).map_err(|e| e.to_string())?;
        w.flush().map_err(|e| e.to_string())?;

        let mut reader = BufReader::new(&stream);
        loop {
            let mut line = String::new();
            if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
                return Err("the server closed before answering".into());
            }
            if line == "\r\n" {
                break;
            }
        }
        let mut out = String::new();
        reader.read_to_string(&mut out).map_err(|e| e.to_string())?;
        Ok(out)
    }
}
