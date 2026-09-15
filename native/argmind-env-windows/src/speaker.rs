//! SAPI, natively — the voice, with no new dependency at all.
//!
//! The Python side reaches this through comtypes; here it is the same system COM objects through
//! `windows-rs`. Measured there and to be reproduced here: synthesis is 4–43 ms, 100–590× real
//! time, so **speaking does not enter the latency budget** — what costs is the *listening*, at
//! ~3.1 s for a median reply.
//!
//! **The routing rule is not a preference, it is a defect being avoided.** An English voice handed
//! pure Korean produces **0 bytes, 79 of 79 — silent, and silently so**. Handed Korean containing a
//! digit or a Latin word it produces audio of the digits only: 46 of 125 recorded Korean replies
//! (37%) would be spoken as a bare widget index, which is the index leak with the sentence stripped
//! off. So: any Hangul routes to the Korean voice, everything else to the **system default**, never
//! to an English one, and a machine with no Korean voice counts a silent utterance rather than
//! emitting one.

use argmind_voice::Speaker;
use windows::core::{Interface, BSTR, PCWSTR};
use windows::Win32::Media::Speech::{
    ISpObjectToken, ISpVoice, ISpeechVoice, SpVoice, SPF_ASYNC, SPF_PURGEBEFORESPEAK,
    SPRS_IS_SPEAKING,
};
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CLSCTX_ALL, COINIT_MULTITHREADED,
};

/// The Korean language id, as SAPI publishes it on a voice token's `Language` attribute.
const KOREAN: &str = "412";

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// Does this text contain Hangul? **The routing decision, and it is made on the text rather than on
/// the person's language** — the agent answers some Korean in English, and those replies should be
/// spoken by the voice that can say them.
pub fn has_hangul(text: &str) -> bool {
    text.chars().any(|c| {
        matches!(c as u32,
            0xAC00..=0xD7A3 | 0x1100..=0x11FF | 0x3130..=0x318F | 0xA960..=0xA97F | 0xD7B0..=0xD7FF)
    })
}

pub struct SapiSpeaker {
    voice: ISpVoice,
    korean: Option<ISpObjectToken>,
    default: Option<ISpObjectToken>,
    pub counts: Counts,
}

#[derive(Default, Debug)]
pub struct Counts {
    pub said: u64,
    pub stopped: u64,
    pub empty: u64,
    pub by_korean: u64,
    pub by_default: u64,
    /// Korean text with no Korean voice installed. **Counted rather than emitted**, because the
    /// alternative is an English voice producing nothing while reporting success.
    pub silent_no_korean_voice: u64,
}

impl SapiSpeaker {
    pub fn new() -> windows::core::Result<Self> {
        unsafe {
            let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
            let voice: ISpVoice = CoCreateInstance(&SpVoice, None, CLSCTX_ALL)?;
            let default = voice.GetVoice().ok();
            //: **Enumeration lives on the automation interface, everything else on the low-level
            //: one.** `SpVoice` implements both; one cast is cheaper than reimplementing a token
            //: category walk, and it is the same object the Python reaches through comtypes.
            let mut korean = None;
            if let Ok(auto) = voice.cast::<ISpeechVoice>() {
                #[allow(clippy::needless_borrow)]
                if let Ok(tokens) = auto.GetVoices(&BSTR::new(), &BSTR::new()) {
                    for i in 0..tokens.Count().unwrap_or(0) {
                        let Ok(token) = tokens.Item(i) else { continue };
                        let attr = token
                            .GetAttribute(&BSTR::from("Language"))
                            .map(|s| s.to_string())
                            .unwrap_or_default();
                        //: The attribute can list several, semicolon separated.
                        if attr.split(';').any(|part| part.eq_ignore_ascii_case(KOREAN)) {
                            korean = token.cast::<ISpObjectToken>().ok();
                            break;
                        }
                    }
                }
            }
            Ok(SapiSpeaker { voice, korean, default, counts: Counts::default() })
        }
    }

    pub fn has_korean(&self) -> bool {
        self.korean.is_some()
    }

    /// The installed voices, by description. Off the trait, for the reason `contract.py` gives for
    /// `attending_title()`: a caller that wants it resolves it and has an honest fallback.
    pub fn voices(&self) -> Vec<String> {
        let mut out = Vec::new();
        unsafe {
        if let Ok(auto) = self.voice.cast::<ISpeechVoice>() {
            if let Ok(tokens) = auto.GetVoices(&BSTR::new(), &BSTR::new()) {
                for i in 0..tokens.Count().unwrap_or(0) {
                    if let Ok(t) = tokens.Item(i) {
                        if let Ok(d) = t.GetDescription(0) {
                            out.push(d.to_string());
                        }
                    }
                }
            }
        }
        }
        out
    }

    /// Volume 0–100. **Every measurement in this file was taken at 0**, so a desktop somebody is
    /// using stays quiet while the numbers are produced.
    pub fn set_volume(&self, volume: i32) {
        unsafe {
            let _ = self.voice.SetVolume(volume as u16);
        }
    }

    fn route(&mut self, text: &str) -> bool {
        unsafe {
            if has_hangul(text) {
                match &self.korean {
                    Some(t) => {
                        let _ = self.voice.SetVoice(t);
                        self.counts.by_korean += 1;
                        true
                    }
                    None => {
                        self.counts.silent_no_korean_voice += 1;
                        false
                    }
                }
            } else {
                if let Some(t) = &self.default {
                    let _ = self.voice.SetVoice(t);
                }
                self.counts.by_default += 1;
                true
            }
        }
    }

    /// Synthesise **forcing the default voice**, whatever the script. Exists only to reproduce the
    /// defect the routing rule avoids: an English voice handed Korean produces 0 bytes, and a
    /// measurement that cannot produce that number cannot show the rule is doing anything.
    pub fn synthesise_forcing_default(&mut self, text: &str) -> windows::core::Result<Vec<u8>> {
        unsafe {
            if let Some(t) = self.default.clone() {
                let _ = self.voice.SetVoice(&t);
            }
        }
        self.synthesise_routed(text, false)
    }

    /// Speak into memory instead of at the room, and return the PCM. **How every number here is
    /// produced without making a sound**, and how the 0-byte finding is reproducible.
    pub fn synthesise(&mut self, text: &str) -> windows::core::Result<Vec<u8>> {
        self.synthesise_routed(text, true)
    }

    fn synthesise_routed(&mut self, text: &str, route: bool) -> windows::core::Result<Vec<u8>> {
        use windows::Win32::Media::Speech::ISpStream;
        use windows::Win32::System::Com::StructuredStorage::CreateStreamOnHGlobal;
        unsafe {
            if route && !self.route(text) {
                return Ok(Vec::new());
            }
            let base: windows::Win32::System::Com::IStream =
                CreateStreamOnHGlobal(windows::Win32::Foundation::HGLOBAL(std::ptr::null_mut()), true)?;
            let stream: ISpStream = CoCreateInstance(
                &windows::Win32::Media::Speech::SpStream, None, CLSCTX_ALL)?;
            let guid = format_guid();
            let mut wave = wave_format();
            stream.SetBaseStream(&base, &guid, &mut wave)?;
            self.voice.SetOutput(&stream, false)?;
            self.voice.Speak(PCWSTR(wide(text).as_ptr()), 0, None)?;
            //: Read the whole thing back out of the HGLOBAL stream.
            let mut out = Vec::new();
            let mut buf = [0u8; 8192];
            base.Seek(0, windows::Win32::System::Com::STREAM_SEEK_SET, None)?;
            loop {
                let mut read = 0u32;
                base.Read(buf.as_mut_ptr() as *mut _, buf.len() as u32, Some(&mut read)).ok()?;
                if read == 0 {
                    break;
                }
                out.extend_from_slice(&buf[..read as usize]);
            }
            self.counts.said += 1;
            //: **Put the output back.** `SetOutput` is sticky: a voice that last spoke into a
            //: stream keeps speaking into it, so a `say()` after any synthesis would be silent
            //: while reporting success — the same shape as the 0-byte defect the routing rule
            //: avoids, arriving from the other direction. `None` is the default audio device, and
            //: it is not the same as never having set an output.
            let _ = self.voice.SetOutput(None, false);
            Ok(out)
        }
    }
}

/// `SPDFID_WaveFormatEx` — the format id SAPI wants when the stream carries a `WAVEFORMATEX`.
///
/// My first attempt passed `KSDATAFORMAT_SUBTYPE_PCM` here, which is the *subtype* of a wave format
/// and not the id of one, and SAPI answered `E_INVALIDARG` — the honest failure, and it failed at
/// the first synthesis rather than producing something subtly wrong.
fn format_guid() -> windows::core::GUID {
    windows::core::GUID::from_u128(0xC31ADBAE_527F_4ff5_A230_F62BB61FF70C)
}

fn wave_format() -> windows::Win32::Media::Audio::WAVEFORMATEX {
    windows::Win32::Media::Audio::WAVEFORMATEX {
        wFormatTag: 1, nChannels: 1, nSamplesPerSec: 16000, nAvgBytesPerSec: 32000,
        nBlockAlign: 2, wBitsPerSample: 16, cbSize: 0,
    }
}

impl Speaker for SapiSpeaker {
    fn say(&mut self, text: &str) {
        if text.trim().is_empty() {
            self.counts.empty += 1;
            return;
        }
        if !self.route(text) {
            return;
        }
        unsafe {
            //: Asynchronous, and that is forced by barge-in: a blocking speak cannot be interrupted.
            let _ = self.voice.Speak(PCWSTR(wide(text).as_ptr()), SPF_ASYNC.0 as u32, None);
        }
        self.counts.said += 1;
    }

    fn stop(&mut self) {
        unsafe {
            //: Purge what is queued and speak nothing — the documented way to cut SAPI off
            //: mid-word.
            let _ = self.voice.Speak(PCWSTR::null(),
                                     (SPF_ASYNC.0 | SPF_PURGEBEFORESPEAK.0) as u32, None);
        }
        self.counts.stopped += 1;
    }

    fn speaking(&self) -> bool {
        unsafe {
            let mut status = Default::default();
            if self.voice.GetStatus(&mut status, std::ptr::null_mut()).is_err() {
                return false;
            }
            status.dwRunningState == SPRS_IS_SPEAKING.0 as u32
        }
    }

    fn release(&mut self) {
        self.stop();
    }
}
