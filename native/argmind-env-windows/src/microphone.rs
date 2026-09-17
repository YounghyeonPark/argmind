//! The microphone: WASAPI capture, held down.
//!
//! **Push-to-talk is the design and not a placeholder.** The key is the floor an open microphone
//! has no equivalent of; it makes self-hearing impossible because the microphone is shut while the
//! agent talks; and it is the only thing that tells a turn addressed to the agent from a phone call
//! in the same room. Nothing here opens the device until somebody holds the key, and it closes
//! again when they let go.
//!
//! **No new dependency.** WASAPI is in the `windows` crate this file already uses for UI Automation
//! and for the voice, so the audio path costs nothing the licence rule has to check.
//!
//! What comes out is what every speech recogniser in sight wants: **16 kHz mono `f32`**. The device
//! decides its own format and it is usually 44.1 or 48 kHz stereo, so the conversion happens here
//! and the arithmetic is [`mono`] and [`resample`] — pure, and tested without opening anything.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// What a recogniser is handed.
pub const WANT_HZ: u32 = 16_000;

/// **Shorter than this is a key press, not a sentence.**
///
/// A tap on the talk key produces a few milliseconds of room noise, and a recogniser handed that
/// does not answer *"nothing"* — Whisper answers `Thank you.` 120 times out of 120. Rejecting it
/// here costs nothing and removes the whole class.
pub const FLOOR_SECONDS: f64 = 0.25;

/// One finished recording.
#[derive(Debug, Clone, PartialEq)]
pub struct Clip {
    /// Mono, [`WANT_HZ`], in −1.0..=1.0.
    pub samples: Vec<f32>,
    pub seconds: f64,
}

impl Clip {
    pub fn new(samples: Vec<f32>) -> Self {
        let seconds = samples.len() as f64 / WANT_HZ as f64;
        Clip { samples, seconds }
    }

    /// Is there enough here to be worth asking about?
    pub fn worth_hearing(&self) -> bool {
        self.seconds >= FLOOR_SECONDS
    }

    /// The loudest sample, 0.0..=1.0 — **what an empty room sounds like, measured rather than
    /// assumed.**
    ///
    /// Not a gate. The threshold that would make it one needs real recorded speech to set, and that
    /// recording does not exist yet; until it does this is reported and nothing is rejected by it.
    pub fn peak(&self) -> f32 {
        self.samples.iter().fold(0.0f32, |a, s| a.max(s.abs()))
    }
}

/// Interleaved frames to mono, by averaging.
///
/// **Averaging, not "take the left channel".** A headset that puts the microphone on one channel
/// and silence on the other is common, and dropping a channel throws away half the recordings on
/// that hardware — while sounding correct on the machine it was written on.
pub fn mono(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    interleaved.chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect()
}

/// Linear resampling to [`WANT_HZ`].
///
/// Linear and not something better on purpose: speech at 16 kHz from 44.1 or 48 is a downsample,
/// the recogniser's own front end filters far more aggressively than any interpolation here would,
/// and a resampler is a dependency and a source of bugs in exchange for nothing measurable.
pub fn resample(samples: &[f32], from_hz: u32) -> Vec<f32> {
    if from_hz == WANT_HZ || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = from_hz as f64 / WANT_HZ as f64;
    let out_len = ((samples.len() as f64) / ratio).floor() as usize;
    (0..out_len)
        .map(|i| {
            let at = i as f64 * ratio;
            let left = at.floor() as usize;
            let right = (left + 1).min(samples.len() - 1);
            let frac = (at - left as f64) as f32;
            samples[left] * (1.0 - frac) + samples[right] * frac
        })
        .collect()
}

/// 16-bit integer samples as floats.
pub fn from_i16(raw: &[i16]) -> Vec<f32> {
    //: Divided by 32768 rather than 32767, so that full-scale negative is exactly −1.0 and nothing
    //: can come out beyond the range a recogniser expects.
    raw.iter().map(|s| *s as f32 / 32768.0).collect()
}

/// What the capture thread and the caller share.
#[derive(Default)]
struct Shared {
    /// Samples so far, already mono and already at [`WANT_HZ`].
    taking: Mutex<Vec<f32>>,
    /// The last finished recording, waiting to be collected.
    done: Mutex<Option<Clip>>,
}

/// A microphone that records only while held.
pub struct Microphone {
    shared: Arc<Shared>,
    holding: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    /// Whether a device could be opened at all. **Reported rather than retried**: a machine with no
    /// microphone should say so once, not fail quietly on every key press.
    pub available: bool,
    pub clips: u64,
    pub too_short: u64,
    /// Which endpoint was asked for. `None` means whichever Windows prefers.
    want: Option<String>,
    /// Set by the capture thread when the one that was asked for was not there.
    fell_back: Arc<AtomicBool>,
}

impl Default for Microphone {
    fn default() -> Self {
        Self::new()
    }
}

impl Microphone {
    /// Whichever one Windows prefers.
    pub fn new() -> Self {
        Microphone::on(None)
    }

    /// A particular one, by endpoint id — **or the default when that one has gone.**
    ///
    /// An id is remembered rather than a name: a friendly name changes when a driver is updated
    /// and an endpoint id does not. Unplugging the headset that was chosen should not make argmind
    /// deaf, so it falls back, and [`Self::fell_back`] is how anybody finds out that it did.
    pub fn on(want: Option<&str>) -> Self {
        Microphone { shared: Arc::new(Shared::default()), holding: Arc::new(AtomicBool::new(false)),
                     running: Arc::new(AtomicBool::new(false)), available: false, clips: 0,
                     too_short: 0,
                     want: want.map(str::to_string), fell_back: Arc::new(AtomicBool::new(false)) }
    }

    /// Was a device asked for by id and not there?
    pub fn fell_back(&self) -> bool {
        self.fell_back.load(Ordering::SeqCst)
    }

    /// Hold it down, or let it go. Letting go finishes the clip.
    pub fn hold(&mut self, down: bool) {
        let was = self.holding.swap(down, Ordering::SeqCst);
        if down && !was {
            self.shared.taking.lock().map(|mut t| t.clear()).ok();
            self.start();
            return;
        }
        if !down && was {
            let samples = self.shared.taking.lock().map(|mut t| std::mem::take(&mut *t))
                .unwrap_or_default();
            let clip = Clip::new(samples);
            if clip.worth_hearing() {
                self.clips += 1;
                if let Ok(mut done) = self.shared.done.lock() {
                    *done = Some(clip);
                }
            } else {
                //: **Counted, not silently dropped.** A talk key that is being tapped rather than
                //: held is a thing a person is doing wrong and would want to be told about.
                self.too_short += 1;
            }
        }
    }

    pub fn held(&self) -> bool {
        self.holding.load(Ordering::SeqCst)
    }

    /// The last finished recording, once.
    pub fn take(&mut self) -> Option<Clip> {
        self.shared.done.lock().ok().and_then(|mut d| d.take())
    }

    pub fn release(&mut self) {
        self.holding.store(false, Ordering::SeqCst);
        self.running.store(false, Ordering::SeqCst);
    }

    #[cfg(not(windows))]
    fn start(&mut self) {}

    #[cfg(windows)]
    fn start(&mut self) {
        if self.running.swap(true, Ordering::SeqCst) {
            return;
        }
        let (shared, holding, running) =
            (Arc::clone(&self.shared), Arc::clone(&self.holding), Arc::clone(&self.running));
        let (want, fell_back) = (self.want.clone(), Arc::clone(&self.fell_back));
        std::thread::spawn(move || {
            let done = capture(&shared, &holding, want.as_deref(), &fell_back);
            running.store(false, Ordering::SeqCst);
            if let Err(e) = done {
                eprintln!("  the microphone stopped: {e}");
            }
        });
        self.available = true;
    }
}

#[cfg(windows)]
fn capture(shared: &Shared, holding: &AtomicBool, want: Option<&str>,
           fell_back: &AtomicBool) -> windows::core::Result<()> {
    use windows::Win32::Media::Audio::{
        eCapture, IAudioCaptureClient, IAudioClient,
        AUDCLNT_SHAREMODE_SHARED, WAVEFORMATEX, WAVE_FORMAT_PCM,
    };
    use windows::Win32::System::Com::{CoInitializeEx, CLSCTX_ALL, COINIT_MULTITHREADED};

    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        let (device, missed) = crate::devices::open_audio(eCapture, want)
            .map_err(|e| windows::core::Error::new(windows::Win32::Foundation::E_FAIL,
                                                   e.as_str()))?;
        fell_back.store(missed, Ordering::SeqCst);
        let client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;
        let format = client.GetMixFormat()?;
        let (channels, rate, bits, tag) = ((*format).nChannels as usize,
                                           (*format).nSamplesPerSec,
                                           (*format).wBitsPerSample,
                                           (*format).wFormatTag);
        //: Ten million hundred-nanosecond units is one second of buffer — generous for an
        //: utterance and small enough that letting go of the key is felt as immediate.
        client.Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 10_000_000, 0, format, None)?;
        let capture: IAudioCaptureClient = client.GetService()?;
        client.Start()?;

        while holding.load(Ordering::SeqCst) {
            let frames = capture.GetNextPacketSize()?;
            if frames == 0 {
                std::thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }
            let mut data = std::ptr::null_mut();
            let mut got = 0u32;
            let mut flags = 0u32;
            capture.GetBuffer(&mut data, &mut got, &mut flags, None, None)?;
            if got > 0 && !data.is_null() {
                let count = got as usize * channels;
                //: The mix format is float on every machine seen, and integer is handled rather
                //: than assumed away — a wrong guess here is silence that looks like a dead
                //: microphone.
                let interleaved: Vec<f32> = if tag == WAVE_FORMAT_PCM as u16 && bits == 16 {
                    from_i16(std::slice::from_raw_parts(data as *const i16, count))
                } else {
                    std::slice::from_raw_parts(data as *const f32, count).to_vec()
                };
                let at_rate = mono(&interleaved, channels);
                let ours = resample(&at_rate, rate);
                if let Ok(mut taking) = shared.taking.lock() {
                    taking.extend_from_slice(&ours);
                }
            }
            capture.ReleaseBuffer(got)?;
        }
        client.Stop()?;
        //: `WAVEFORMATEX` came from `GetMixFormat`, which allocates; releasing it is the caller's.
        windows::Win32::System::Com::CoTaskMemFree(Some(format as *const WAVEFORMATEX as *const _));
    }
    Ok(())
}
