//! A voice that is a neural synthesiser on this machine, spoken to over HTTP.
//!
//! The same shape as [`argmind_env_ollama`] and [`argmind_env_whisper`], for the same reason: a
//! thing you install, that runs on your machine, that answers on a socket. **No new crate, no FFI,
//! no C++ in this tree** — `qwentts.cpp` is built from source at a pinned commit the way whisper.cpp
//! is, and this speaks raw HTTP/1.1 to it.
//!
//! ## Why bother, when SAPI already speaks
//!
//! `Microsoft Heami Desktop` is the only Korean voice Windows ships here and it is a Windows 7-era
//! concatenative one. The owner's verdict was 어색함. Measured against it,
//! `docs/findings/a-better-voice.md` has the whole accounting, and the short version is: the
//! licences are clean, it runs at **RTF 0.15** on the GPU, and the server answers a reply-length
//! sentence in about **0.7 s** with the model already loaded.
//!
//! ## What is different from SAPI, and what it costs
//!
//! **SAPI synthesises and plays in one call. This does not play anything.** It returns samples, and
//! [`argmind_env_windows::loudspeaker`] puts them on the device — which is why that file exists and
//! why it is careful about stopping.
//!
//! **So `say` is two waits, not one**: the model call, and then the audio. The first is off the
//! calling thread — `say` must return at once, because a blocking one cannot be interrupted and
//! barge-in is the whole point — and the second is the speaker's.
//!
//! ## Why it streams
//!
//! Asking for the whole utterance and then playing it means **1.8 s of silence** after a question —
//! measured, against SAPI's 4–43 ms — and the model call before it already costs 0.61 s. The server
//! decodes frame by frame and sends `Transfer-Encoding: chunked` with `X-Accel-Buffering: no`, so
//! the pieces are there long before the last one. Reading them as they land and handing each to the
//! speaker is the difference between an agent that answers and one that pauses first.
//!
//! ## Why every request carries a seed
//!
//! **The talker is a *base* model, so it has no speakers** — `GET /v1/audio/voices` answers with an
//! empty list and every name offered to `voice` comes back `unknown voice`. What it has instead is
//! a sampler, and the timbre is whatever that sampler drew: measured 2026-09-21, four calls with
//! the same text gave four different voices and a 21% spread in length, while `seed = 7` gave the
//! **same bytes four times out of four**. So a seed is not a tuning knob here, it is the only
//! handle on *which voice this is*, and [`speak`] and [`speak_streaming`] take one rather than
//! defaulting — a call that did not decide would be a different person answering.
//!
//! **Raw PCM rather than WAV for the streamed path.** A WAV header carries the rate, which is worth
//! forty-four bytes when the whole file arrives at once; a stream has no length to put in one, and
//! the rate is then [`HZ`] by agreement. Both paths exist and [`speak`] is still the one that
//! checks the header — a server that changed its rate would be caught there.
//!
//! ## Barge-in across a request
//!
//! A cut-off can land while the server is still thinking, and then the reply arrives for something
//! nobody is waiting for any more. **A generation counter, checked after the response**, is what
//! stops that reply from being spoken over whatever came next. The same device the loudspeaker uses
//! for the same reason.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Where the synthesiser listens. **One port away from the recogniser**, which is 8178.
pub const QWENTTS: &str = "127.0.0.1:8179";

/// What `qwentts.cpp` emits, and what the WAV header is checked against.
pub const HZ: u32 = 24_000;

/// Ask a `tts-server` for one utterance, as mono `f32` and the rate it came at.
///
/// **WAV rather than raw PCM**, which costs forty-four bytes and buys the rate being read rather
/// than assumed. A server that changed what it emits would otherwise be a voice that got faster.
pub fn speak(host: &str, model: &str, text: &str, seed: i64)
             -> Result<(Vec<f32>, u32), String> {
    let body = asking(model, text, seed, "\"response_format\":\"wav\"");
    let answer = post(host, "/v1/audio/speech", &body)?;
    from_wav(&answer)
}

/// The body both paths send, differing only in the last field.
///
/// **One function because the seed has to be in both.** The streamed path is the one that ships and
/// the whole-file one is what every measurement of the voice goes through, so a seed carried by
/// only one of them would make every finding about the voice a finding about a voice nobody hears.
/// Written as one string rather than two because that is a mistake you cannot make twice.
pub fn asking(model: &str, text: &str, seed: i64, and: &str) -> String {
    format!("{{\"model\":\"{}\",\"input\":\"{}\",\"seed\":{},{}}}",
            escape(model), escape(text), seed, and)
}

/// JSON string escaping — the same set `argmind-serve` writes, for the same reason.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Mono `f32` and the sample rate, out of a 16-bit PCM WAV.
///
/// **The header is read, not skipped.** Forty-four bytes is the usual size and it is not the only
/// one: a `LIST` chunk before `data` is legal and common, so the chunks are walked. Assuming 44
/// would play the chunk header as audio, which is a click and then everything shifted.
pub fn from_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32), String> {
    if bytes.len() < 44 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(format!("WAV이 아닙니다 ({}바이트)", bytes.len()));
    }
    let word = |at: usize| -> u32 {
        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    };
    let half = |at: usize| -> u16 { u16::from_le_bytes([bytes[at], bytes[at + 1]]) };

    let (mut at, mut hz, mut channels, mut bits) = (12usize, 0u32, 0u16, 0u16);
    while at + 8 <= bytes.len() {
        let kind = &bytes[at..at + 4];
        let size = word(at + 4) as usize;
        let from = at + 8;
        if kind == b"fmt " && from + 16 <= bytes.len() {
            channels = half(from + 2);
            hz = word(from + 4);
            bits = half(from + 14);
        } else if kind == b"data" {
            let to = (from + size).min(bytes.len());
            if bits != 16 {
                return Err(format!("16비트가 아닙니다 ({bits}비트)"));
            }
            if channels == 0 {
                return Err("채널이 없습니다".into());
            }
            //: **Mixed down rather than assumed mono.** The server sends one channel today; a
            //: caller that read a stereo stream as mono would play it at half speed.
            let frames = (to - from) / 2 / channels as usize;
            let mut out = Vec::with_capacity(frames);
            for f in 0..frames {
                let mut sum = 0f32;
                for c in 0..channels as usize {
                    let i = from + (f * channels as usize + c) * 2;
                    sum += i16::from_le_bytes([bytes[i], bytes[i + 1]]) as f32 / 32768.0;
                }
                out.push(sum / channels as f32);
            }
            return Ok((out, hz));
        }
        //: Chunks are word aligned, and an odd size carries a pad byte that is not in the size.
        at = from + size + (size & 1);
    }
    Err("data 청크가 없습니다".into())
}

/// Ask for one utterance and hand over each piece as it arrives.
///
/// `on_piece` gets mono `f32` at [`HZ`]; returning `false` abandons the request, which is what a
/// barge-in during synthesis does. The `bool` in the answer is whether it ran to the end.
pub fn speak_streaming(host: &str, model: &str, text: &str, seed: i64,
                       on_piece: &mut dyn FnMut(&[f32]) -> bool) -> Result<bool, String> {
    let body = asking(model, text, seed, "\"stream\":true");
    //: **A sample can straddle a chunk boundary.** The server sends bytes and does not care where
    //: a sixteen-bit sample begins, so an odd byte at the end of one piece is the first half of a
    //: sample completed by the next. Dropping it would put a click in every chunk.
    let mut spare: Option<u8> = None;
    stream(host, "/v1/audio/speech", &body, &mut |bytes| {
        let mut samples = Vec::with_capacity(bytes.len() / 2 + 1);
        let mut from = 0;
        if let Some(low) = spare.take() {
            if let Some(high) = bytes.first() {
                samples.push(i16::from_le_bytes([low, *high]) as f32 / 32768.0);
                from = 1;
            } else {
                spare = Some(low);
            }
        }
        let body = &bytes[from..];
        let whole = body.len() / 2 * 2;
        for pair in body[..whole].chunks_exact(2) {
            samples.push(i16::from_le_bytes([pair[0], pair[1]]) as f32 / 32768.0);
        }
        if whole < body.len() {
            spare = Some(body[whole]);
        }
        on_piece(&samples)
    })
}

/// One POST, and every piece of the body as it arrives. **Chunked transfer encoding**, which is
/// what the server uses when it cannot know the length in advance.
fn stream(host: &str, path: &str, body: &str,
          on_bytes: &mut dyn FnMut(&[u8]) -> bool) -> Result<bool, String> {
    let mut socket = TcpStream::connect(host)
        .map_err(|e| format!("{host} 에 닿지 못했습니다 ({e})"))?;
    let _ = socket.set_read_timeout(Some(std::time::Duration::from_secs(120)));
    let (name, port) = host.rsplit_once(':').unwrap_or((host, "80"));
    write!(socket, "POST {path} HTTP/1.1\r\nHost: {name}:{port}\r\n\
                    Content-Type: application/json\r\nContent-Length: {}\r\n\
                    Connection: close\r\n\r\n{body}", body.len())
        .map_err(|e| format!("{e}"))?;

    let mut reading = BufReader::new(socket);
    let mut line = String::new();
    reading.read_line(&mut line).map_err(|e| format!("{e}"))?;
    let status = line.split_whitespace().nth(1).unwrap_or("").to_string();
    let mut chunked = false;
    loop {
        let mut header = String::new();
        if reading.read_line(&mut header).map_err(|e| format!("{e}"))? == 0 {
            break;
        }
        if header.trim().is_empty() {
            break;
        }
        if header.to_lowercase().starts_with("transfer-encoding:")
            && header.to_lowercase().contains("chunked") {
            chunked = true;
        }
    }
    if status != "200" {
        let mut out = Vec::new();
        let _ = reading.read_to_end(&mut out);
        return Err(format!("{status}: {}", String::from_utf8_lossy(&out).trim()));
    }
    if !chunked {
        //: Not streaming after all — a server that answered in one piece. Handing the whole body
        //: over once is the same thing with one call.
        let mut out = Vec::new();
        reading.read_to_end(&mut out).map_err(|e| format!("{e}"))?;
        return Ok(on_bytes(&out));
    }

    loop {
        let mut size_line = String::new();
        if reading.read_line(&mut size_line).map_err(|e| format!("{e}"))? == 0 {
            return Ok(false);
        }
        //: A chunk size is hex, and may carry `;extensions` nobody here uses.
        let size = usize::from_str_radix(
            size_line.trim().split(';').next().unwrap_or("").trim(), 16)
            .map_err(|_| format!("길이가 이상합니다: {size_line:?}"))?;
        if size == 0 {
            //: The trailer, then the end.
            return Ok(true);
        }
        let mut piece = vec![0u8; size];
        reading.read_exact(&mut piece).map_err(|e| format!("{e}"))?;
        let mut crlf = [0u8; 2];
        reading.read_exact(&mut crlf).map_err(|e| format!("{e}"))?;
        if !on_bytes(&piece) {
            //: **Abandoned.** The socket is dropped with the server still writing, which it will
            //: notice; nothing here waits for the rest of an answer nobody wants.
            return Ok(false);
        }
    }
}

/// One POST, and the body back. Raw HTTP/1.1, no dependency.
fn post(host: &str, path: &str, body: &str) -> Result<Vec<u8>, String> {
    let mut socket = TcpStream::connect(host)
        .map_err(|e| format!("{host} 에 닿지 못했습니다 ({e})"))?;
    //: **Generous, because synthesis is not instant** — measured at 0.7 s for a reply, and a cold
    //: server has a model to load first.
    let _ = socket.set_read_timeout(Some(std::time::Duration::from_secs(120)));
    let (name, port) = host.rsplit_once(':').unwrap_or((host, "80"));
    write!(socket, "POST {path} HTTP/1.1\r\nHost: {name}:{port}\r\n\
                    Content-Type: application/json\r\nContent-Length: {}\r\n\
                    Connection: close\r\n\r\n{body}", body.len())
        .map_err(|e| format!("{e}"))?;

    let mut reading = BufReader::new(socket);
    let mut line = String::new();
    reading.read_line(&mut line).map_err(|e| format!("{e}"))?;
    let status = line.split_whitespace().nth(1).unwrap_or("").to_string();
    loop {
        let mut header = String::new();
        if reading.read_line(&mut header).map_err(|e| format!("{e}"))? == 0 {
            break;
        }
        if header.trim().is_empty() {
            break;
        }
    }
    let mut out = Vec::new();
    reading.read_to_end(&mut out).map_err(|e| format!("{e}"))?;
    if status != "200" {
        //: The server's own words, which name the thing — `unknown voice 'default'` was one.
        return Err(format!("{status}: {}", String::from_utf8_lossy(&out).trim()));
    }
    Ok(out)
}

/// A neural voice: [`argmind_voice::Speaker`], with the server doing the work and the loudspeaker
/// doing the talking.
pub struct QwenVoice {
    pub host: String,
    pub model: String,
    /// **Which voice.** See the module note: without it the speaker changes between one sentence
    /// and the next.
    pub seed: i64,
    mouth: argmind_env_windows::loudspeaker::Loudspeaker,
    /// Bumped by every `say` and every `stop`. **A reply that comes back for an old generation is
    /// dropped**, because a cut-off that lands while the server is still thinking must not be
    /// undone by the answer arriving.
    generation: Arc<AtomicU64>,
    /// A request is out and has not been answered or abandoned.
    thinking: Arc<AtomicU64>,
    pub asked: u64,
    pub failed: u64,
}

impl QwenVoice {
    /// **Does not open the device and does not ask the server anything.** Whether either works is
    /// found out when something is said, and reported then.
    pub fn new(model: &str, seed: i64) -> Self {
        QwenVoice {
            host: QWENTTS.to_string(),
            model: model.to_string(),
            seed,
            mouth: argmind_env_windows::loudspeaker::Loudspeaker::new(),
            generation: Arc::new(AtomicU64::new(0)),
            thinking: Arc::new(AtomicU64::new(0)),
            asked: 0,
            failed: 0,
        }
    }

    /// Is sound actually coming out?
    ///
    /// **Narrower than [`argmind_voice::Speaker::speaking`]**, which is also true while the server
    /// is still thinking. That is the right answer for a caller deciding whether to interrupt and
    /// the wrong one for measuring how long the silence was.
    pub fn making_sound(&self) -> bool {
        self.mouth.speaking()
    }

    /// Is the server answering? A connect and nothing else — **it is not asked to synthesise
    /// anything**, because a check that made the thing it is checking do work is a check with a
    /// side effect.
    pub fn answering(host: &str) -> bool {
        host.parse().is_ok_and(|at| {
            std::net::TcpStream::connect_timeout(&at, std::time::Duration::from_millis(400)).is_ok()
        })
    }
}

impl argmind_voice::Speaker for QwenVoice {
    /// **Returns at once.** The request goes on its own thread and the samples reach the speaker
    /// when they arrive — which is between half a second and a second later, and is why this cannot
    /// block the thread that owns the session.
    fn say(&mut self, text: &str) {
        if text.trim().is_empty() {
            return;
        }
        self.asked += 1;
        let mine = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        self.thinking.store(mine, Ordering::SeqCst);
        //: The speaker is stopped now rather than when the audio arrives, so a new turn silences
        //: the old one immediately instead of after the new one has been synthesised.
        self.mouth.stop();

        let (host, model, text) = (self.host.clone(), self.model.clone(), text.to_string());
        let seed = self.seed;
        let generation = Arc::clone(&self.generation);
        let thinking = Arc::clone(&self.thinking);
        let mut mouth = self.mouth.clone_handle();
        std::thread::spawn(move || {
            let mut started = false;
            let answer = speak_streaming(&host, &model, &text, seed, &mut |samples| {
                //: **Checked on every piece.** A stop that lands mid-synthesis leaves the rest of
                //: an utterance nobody is waiting for, and speaking it would talk over whatever
                //: came next. Returning false drops the connection.
                if generation.load(Ordering::SeqCst) != mine {
                    return false;
                }
                if samples.is_empty() {
                    return true;
                }
                if started {
                    mouth.append(samples, HZ);
                } else {
                    //: The first piece is what stops the thinking and starts the sound.
                    thinking.store(0, Ordering::SeqCst);
                    mouth.play(samples, HZ);
                    started = true;
                }
                true
            });
            thinking.store(0, Ordering::SeqCst);
            match answer {
                //: **Only ours.** A run that was abandoned has already had its audio replaced.
                Ok(_) if generation.load(Ordering::SeqCst) == mine => mouth.finished(),
                Ok(_) => {}
                Err(why) => eprintln!("  the voice did not answer: {why}"),
            }
        });
    }

    /// Cut it off — **both halves**: the audio that is playing, and the reply that has not come
    /// back yet.
    fn stop(&mut self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.thinking.store(0, Ordering::SeqCst);
        self.mouth.stop();
    }

    /// Speaking, or about to be. **Both count**: a caller deciding whether to interrupt should not
    /// be told the agent is silent while a sentence is being synthesised for it.
    fn speaking(&self) -> bool {
        self.mouth.speaking() || self.thinking.load(Ordering::SeqCst) != 0
    }

    fn release(&mut self) {
        self.stop();
        self.mouth.release();
    }
}
