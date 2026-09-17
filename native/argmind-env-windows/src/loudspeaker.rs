//! Putting audio on the speakers, and **stopping it mid-word.**
//!
//! `SapiSpeaker` synthesises and plays in one call, so nothing here was needed while the voice was
//! SAPI's. A neural synthesiser hands back samples instead — `qwentts.cpp` answers with 24 kHz mono
//! — and something has to take them to the device. This is that something, and it is the mirror of
//! [`crate::microphone`]: same WASAPI, same shared mode, same mix-format handling, opposite
//! direction.
//!
//! ## Stopping is the requirement, not playing
//!
//! Playing audio is easy and is not why this file is careful. **Barge-in is the reason**: a median
//! reply is about three seconds of speech against a 0.61 s model call, and a voice that cannot be
//! interrupted turns a conversation into a lecture. `Speaker::stop` means *cut it off now and do
//! not resume*, so:
//!
//! * The device buffer is kept **short on purpose**. Whatever has already been handed to WASAPI is
//!   going to be heard no matter what this does, so the buffer length is the floor on how fast a
//!   stop can take effect. Sixty milliseconds is below what anybody notices and still leaves room
//!   for the feeder thread to miss a beat.
//! * `stop` clears the queue **and resets the device**, which drops what it is already holding.
//!   The first version of this file claimed that in a comment and did not do it — it cleared the
//!   queue and left up to a buffer of speech still to come out, which is a stop that is late by
//!   exactly the amount nobody would measure.
//!
//! ## What it does not do
//!
//! **No mixing and no queueing of utterances.** One thing speaks at a time; a new utterance
//! replaces the old one, which is what barge-in means and also what `SapiSpeaker` already does.
//!
//! ## An utterance arrives in pieces
//!
//! The synthesiser streams: it decodes frame by frame and the first audio is ready long before the
//! last. So [`Self::play`] starts one and [`Self::append`] adds to it, and the feeder keeps going
//! across a gap between pieces rather than treating an empty queue as the end. **That distinction
//! is the whole reason [`Self::finished`] exists** — without it the device would be reset between
//! chunks, which is a click in the middle of every sentence.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// How much audio the device is allowed to be holding, in hundred-nanosecond units.
///
/// **This is the floor on how quickly speech can stop**, so it is short rather than comfortable:
/// once samples are in WASAPI's buffer they will be heard. Sixty milliseconds.
pub const BUFFER: i64 = 600_000;

/// What the synthesiser speaks at. Everything is resampled from this to whatever the device wants.
pub const FROM_HZ: u32 = 24_000;

struct Queue {
    /// Mono samples, oldest first.
    waiting: Vec<f32>,
    /// What rate `waiting` is at. **Carried with the samples**, because a server quietly
    /// changing what it emits is the one way this could be wrong, and audio at the wrong
    /// rate is not a failure — it is a chipmunk.
    hz: u32,
    /// Bumped by `stop` and by each new utterance. **The feeder checks it rather than a flag**, so
    /// a stop that lands while the thread is mid-buffer cannot be missed and then un-missed by the
    /// next utterance setting the flag back.
    generation: u64,
    /// Whether any more is coming. **An empty queue is not the end of an utterance** while a
    /// synthesiser is still streaming into it, and resetting the device on one would put a click
    /// between every pair of chunks.
    done: bool,
}

impl Default for Queue {
    fn default() -> Self {
        Queue { waiting: Vec::new(), hz: FROM_HZ, generation: 0, done: true }
    }
}

/// A voice's mouth: hand it samples, or cut it off.
pub struct Loudspeaker {
    queue: Arc<Mutex<Queue>>,
    /// Whether the feeder thread is alive. Not whether sound is coming out — see [`Self::speaking`].
    running: Arc<AtomicBool>,
    /// The generation the feeder is working on, so `speaking` can tell "still going" from "the
    /// queue is empty because it finished".
    playing: Arc<AtomicBool>,
    generation: Arc<AtomicU64>,
    /// **`false` when the device could not be opened.** A machine with no output is a fact to
    /// report, not an error to fail on: the agent still works, it is just quiet.
    pub available: bool,
    /// How many utterances were cut off. Counted because barge-in working is a thing to be able to
    /// show, and because barge-in firing constantly would mean something else is wrong.
    pub cut_off: u64,
    /// Which endpoint was asked for. `None` means whichever Windows prefers.
    want: Option<String>,
    /// Set by the feeding thread when the one that was asked for was not there.
    fell_back: Arc<AtomicBool>,
}

impl Default for Loudspeaker {
    fn default() -> Self {
        Self::new()
    }
}

impl Loudspeaker {
    /// Whichever one Windows prefers.
    pub fn new() -> Self {
        Loudspeaker::on(None)
    }

    /// A particular one, by endpoint id — **or the default when that one has gone.**
    ///
    /// The same bargain the microphone makes: an id survives a driver update where a name does
    /// not, and a headset that was chosen and then unplugged should leave argmind audible rather
    /// than silent. [`Self::fell_back`] says when that happened.
    pub fn on(want: Option<&str>) -> Self {
        Loudspeaker {
            queue: Arc::new(Mutex::new(Queue::default())),
            running: Arc::new(AtomicBool::new(false)),
            playing: Arc::new(AtomicBool::new(false)),
            generation: Arc::new(AtomicU64::new(0)),
            available: false,
            cut_off: 0,
            want: want.map(str::to_string),
            fell_back: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Was a device asked for by id and not there?
    pub fn fell_back(&self) -> bool {
        self.fell_back.load(Ordering::SeqCst)
    }

    /// Speak these samples, which are mono at `hz`. **Returns at once**, and replaces whatever
    /// was playing.
    pub fn play(&mut self, samples: &[f32], hz: u32) {
        if self.speaking() {
            self.cut_off += 1;
        }
        let generation = {
            let Ok(mut queue) = self.queue.lock() else { return };
            queue.generation += 1;
            queue.waiting.clear();
            queue.waiting.extend_from_slice(samples);
            queue.hz = hz;
            //: **Started, not finished.** `finished()` is what says an utterance is complete; until
            //: then the feeder waits through an empty queue instead of resetting the device.
            queue.done = false;
            queue.generation
        };
        self.generation.store(generation, Ordering::SeqCst);
        self.start();
    }

    /// Add to the utterance that is playing. **For a synthesiser that streams**: the first piece
    /// goes to [`Self::play`] and every piece after it comes here.
    ///
    /// Does nothing if nothing is playing — a piece that arrives after a `stop` belongs to an
    /// utterance nobody is waiting for.
    pub fn append(&mut self, samples: &[f32], hz: u32) {
        let Ok(mut queue) = self.queue.lock() else { return };
        if queue.done {
            return;
        }
        queue.hz = hz;
        queue.waiting.extend_from_slice(samples);
    }

    /// No more is coming. **The feeder may reset the device once it runs out.**
    pub fn finished(&mut self) {
        if let Ok(mut queue) = self.queue.lock() {
            queue.done = true;
        }
    }

    /// Cut it off now and do not resume. **This is what barge-in is.**
    ///
    /// **Returns before the sound has stopped.** It tells the feeder to drop everything, and the
    /// feeder resets the device on its next pass — a millisecond or two later. `speaking()` is what
    /// says whether it has actually happened.
    pub fn stop(&mut self) {
        if let Ok(mut queue) = self.queue.lock() {
            queue.generation += 1;
            queue.waiting.clear();
            queue.done = true;
            self.generation.store(queue.generation, Ordering::SeqCst);
        }
    }

    /// Is sound coming out?
    ///
    /// **Set by the feeder thread and by nothing else.** An earlier version had `stop` clear this
    /// directly, which made *how long does stopping take* a measurement of one line setting a
    /// boolean: it read 0 ms every time, whatever the device was doing.
    pub fn speaking(&self) -> bool {
        self.playing.load(Ordering::SeqCst)
    }

    /// Another handle on the same mouth.
    ///
    /// **Shares the queue rather than copying it**, so a thread that was given one can `play` into
    /// the device the owner is holding — which is what a synthesiser answering on its own thread
    /// needs. `cut_off` is per handle, because it counts what *this* caller interrupted.
    pub fn clone_handle(&self) -> Loudspeaker {
        Loudspeaker {
            queue: Arc::clone(&self.queue),
            running: Arc::clone(&self.running),
            playing: Arc::clone(&self.playing),
            generation: Arc::clone(&self.generation),
            available: self.available,
            cut_off: 0,
            want: self.want.clone(),
            fell_back: Arc::clone(&self.fell_back),
        }
    }

    /// Let the thread go. Safe to call twice.
    pub fn release(&mut self) {
        self.stop();
        self.running.store(false, Ordering::SeqCst);
    }

    #[cfg(not(windows))]
    fn start(&mut self) {}

    #[cfg(windows)]
    fn start(&mut self) {
        if self.running.swap(true, Ordering::SeqCst) {
            //: Already feeding. It will pick up the new generation on its next pass, which is the
            //: whole reason the queue carries one.
            return;
        }
        let queue = Arc::clone(&self.queue);
        let running = Arc::clone(&self.running);
        let playing = Arc::clone(&self.playing);
        let (want, fell_back) = (self.want.clone(), Arc::clone(&self.fell_back));
        playing.store(true, Ordering::SeqCst);
        std::thread::spawn(move || {
            if let Err(e) = feed(&queue, &running, &playing, want.as_deref(), &fell_back) {
                eprintln!("  the speaker would not open: {e}");
            }
            playing.store(false, Ordering::SeqCst);
            running.store(false, Ordering::SeqCst);
        });
        self.available = true;
    }
}

/// Stretch mono samples from `from_hz` to the device's rate, and spread them across its channels.
///
/// **Linear, like the capture side.** A synthesiser's output is already band-limited and the
/// destination rate is usually a clean 2:1; the argument for anything better is an argument about
/// audio that has not been measured to matter here.
pub fn to_device(samples: &[f32], from_hz: u32, to_hz: u32, channels: usize) -> Vec<f32> {
    if samples.is_empty() || from_hz == 0 || to_hz == 0 || channels == 0 {
        return Vec::new();
    }
    let ratio = from_hz as f64 / to_hz as f64;
    //: **Counted in integers.** `(len / ratio).floor()` loses a frame at 44.1 kHz — 24000 / (24000
    //: / 44100) comes out as 44099.999… and floors to one short. One frame is nothing to hear and
    //: the arithmetic being wrong is something to fix.
    let out_frames = samples.len() * to_hz as usize / from_hz as usize;
    let mut out = Vec::with_capacity(out_frames * channels);
    for i in 0..out_frames {
        let at = i as f64 * ratio;
        let low = at.floor() as usize;
        let part = (at - low as f64) as f32;
        let a = samples.get(low).copied().unwrap_or(0.0);
        let b = samples.get(low + 1).copied().unwrap_or(a);
        let value = a + (b - a) * part;
        for _ in 0..channels {
            out.push(value);
        }
    }
    out
}

#[cfg(windows)]
fn feed(queue: &Mutex<Queue>, running: &AtomicBool, playing: &AtomicBool, want: Option<&str>,
        fell_back: &AtomicBool) -> windows::core::Result<()> {
    use windows::Win32::Media::Audio::{
        eRender, IAudioClient, IAudioRenderClient,
        AUDCLNT_SHAREMODE_SHARED, WAVEFORMATEX, WAVE_FORMAT_PCM,
    };
    use windows::Win32::System::Com::{CoInitializeEx, CLSCTX_ALL, COINIT_MULTITHREADED};

    unsafe {
        let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
        let (device, missed) = crate::devices::open_audio(eRender, want)
            .map_err(|e| windows::core::Error::new(windows::Win32::Foundation::E_FAIL,
                                                   e.as_str()))?;
        fell_back.store(missed, Ordering::SeqCst);
        let client: IAudioClient = device.Activate(CLSCTX_ALL, None)?;
        let format = client.GetMixFormat()?;
        let (channels, rate, bits, tag) = ((*format).nChannels as usize,
                                           (*format).nSamplesPerSec,
                                           (*format).wBitsPerSample,
                                           (*format).wFormatTag);
        client.Initialize(AUDCLNT_SHAREMODE_SHARED, 0, BUFFER, 0, format, None)?;
        let render: IAudioRenderClient = client.GetService()?;
        let total = client.GetBufferSize()?;
        //: **Not started until there is something to say.** A started client with an empty buffer
        //: is a glitch waiting for the first utterance, and this one stops and resets between
        //: utterances anyway.
        let mut started = false;

        //: What this thread is currently speaking. A change means somebody called `play` or `stop`
        //: while it was mid-buffer, and everything converted from the old utterance is dropped.
        let mut mine = 0u64;
        let mut ours: Vec<f32> = Vec::new();
        let mut at = 0usize;
        let mut quiet_passes = 0u32;
        let mut ended;

        while running.load(Ordering::SeqCst) {
            {
                let Ok(mut held) = queue.lock() else { break };
                if held.generation != mine {
                    //: A new utterance, or a stop. Everything converted from the old one goes.
                    mine = held.generation;
                    ours.clear();
                    at = 0;
                }
                ended = held.done;
                if !held.waiting.is_empty() {
                    //: **Converted as it arrives**, rather than per output buffer: resampling is
                    //: cheap and doing it in the buffer path would put it on every deadline.
                    //:
                    //: Each piece is resampled on its own, so the interpolation restarts at a chunk
                    //: boundary — one sample of discontinuity per piece, at a rate ratio that is
                    //: usually exactly two. Inaudible, and cheaper than carrying state across.
                    let more = to_device(&held.waiting, held.hz, rate, channels);
                    ours.extend_from_slice(&more);
                    held.waiting.clear();
                }
            }

            if at >= ours.len() && !ended {
                //: **Out of audio but not out of utterance.** The synthesiser is still streaming;
                //: resetting the device here would be a click between every pair of chunks.
                std::thread::sleep(std::time::Duration::from_millis(5));
                continue;
            }

            if at >= ours.len() {
                //: **The device is reset the moment there is nothing left**, which is what makes a
                //: stop immediate rather than a buffer late: `Reset` drops the frames WASAPI is
                //: still holding. Without it, `stop` returns and up to `BUFFER` of speech carries
                //: on, and nobody hears the difference between that and a slow stop.
                if playing.load(Ordering::SeqCst) {
                    let _ = client.Stop();
                    let _ = client.Reset();
                    started = false;
                    //: **After the reset, not before.** The flag is what a caller times against,
                    //: so dropping it first would report a stop that had not happened yet — the
                    //: same self-measurement this file already made once.
                    playing.store(false, Ordering::SeqCst);
                }
                //: Nothing to say. **Not an exit**, because the next utterance usually arrives
                //: within a turn and reopening the device costs more than waiting — but not
                //: forever either, or an idle agent holds the speakers all day.
                quiet_passes += 1;
                if quiet_passes > 300 {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
                continue;
            }
            quiet_passes = 0;
            if !started {
                client.Start()?;
                started = true;
            }
            playing.store(true, Ordering::SeqCst);

            let used = client.GetCurrentPadding()?;
            let room = total.saturating_sub(used) as usize;
            if room == 0 {
                std::thread::sleep(std::time::Duration::from_millis(5));
                continue;
            }
            let left = (ours.len() - at) / channels;
            let frames = room.min(left);
            if frames == 0 {
                at = ours.len();
                continue;
            }
            let data = render.GetBuffer(frames as u32)?;
            let count = frames * channels;
            //: The mix format is float on every machine seen here, and integer is handled rather
            //: than assumed away — a wrong guess is noise, which is worse than silence.
            if tag == WAVE_FORMAT_PCM as u16 && bits == 16 {
                let out = std::slice::from_raw_parts_mut(data as *mut i16, count);
                for (i, slot) in out.iter_mut().enumerate() {
                    *slot = (ours[at + i].clamp(-1.0, 1.0) * 32767.0) as i16;
                }
            } else {
                let out = std::slice::from_raw_parts_mut(data as *mut f32, count);
                out.copy_from_slice(&ours[at..at + count]);
            }
            render.ReleaseBuffer(frames as u32, 0)?;
            at += count;
        }

        if started {
            client.Stop()?;
        }
        //: `WAVEFORMATEX` came from `GetMixFormat`, which allocates; releasing it is the caller's.
        windows::Win32::System::Com::CoTaskMemFree(Some(format as *const WAVEFORMATEX as *const _));
    }
    Ok(())
}
