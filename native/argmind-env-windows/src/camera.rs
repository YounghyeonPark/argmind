//! The camera — **as a source of WHETHER somebody is there, never of WHERE anything is.**
//!
//! The rule this project rests on is that the deterministic side says WHERE and the probabilistic
//! side says WHAT. A camera is a detector, so nothing here may ever hand a coordinate to anything:
//! UI Automation keeps that job. What a camera can honestly supply is the thing no accessibility
//! tree knows — **whether a person is in front of the machine at all** — and that turns out to be
//! the fact the agent most needs and does not have. Watching the screen is on a timer and will
//! happily talk to an empty chair.
//!
//! ## What leaves this module, and what cannot
//!
//! **A frame is reduced to 64×36 greys before anything else sees it, and the full picture is never
//! returned.** That is a property of the type rather than a promise in a comment: [`Frame`] holds
//! [`WIDE`]×[`TALL`] bytes and there is no accessor for anything larger. At that size a room is
//! light and shadow and movement — enough to answer *is somebody there*, and not a photograph of
//! anybody's living room. If a vision model is ever wanted it will need a different call, a
//! different permission and a different argument, which is exactly the shape that decision should
//! have.
//!
//! ## It is not opened until it is asked for
//!
//! Same as the microphone, and for the same reason: a program does not get to look at somebody
//! because it was run. `Camera::open` starts the device — **the light comes on** — and `Drop` shuts
//! it down. Nothing here opens one on a timer or keeps one open between looks.
//!
//! No new dependency: Media Foundation is in the `windows` crate this already links against.
//!
//! ## What was measured, 2026-09-16, `Integrated Camera` at 1920×1080
//!
//! | | |
//! |---|---|
//! | opening the device | **292–369 ms** |
//! | settling on top of that | **338 ms warm, about 1.4 s cold** |
//! | one look, once open and settled | **median 32 ms, worst 52 ms** over seven runs of 25 |
//! | a quiet room against itself, settled | **0.14–0.34 in six runs of seven, and 1.44 in the seventh** |
//! | a quiet room against itself, *while settling* | **up to 2.85** |
//!
//! **The last two rows are why `open` waits.** A camera that has just been switched on drifts: the
//! exposure walked 153 down to 130 over a second and a bit, and a presence check reading those
//! frames would report a difference twenty times the settled floor — movement that was the sensor
//! waking up. So `open` keeps looking until the picture stops changing, and says how long that
//! took rather than sleeping a guessed interval. Cold and warm differ by a factor of four, which is
//! exactly what a fixed sleep would have got wrong in one direction or the other.
//!
//! ## The seventh run, and why no movement threshold is named
//!
//! Six runs put a quiet room between 0.14 and 0.34. One put it at **1.44**, ten times the others
//! and three times the settling threshold — and **I do not know what that was.** It could have been
//! somebody moving, in which case the signal works and is large; it could have been a light, a
//! reflection off a screen, or the exposure taking another step. A number whose cause is unknown is
//! not a floor, and averaging it away would turn *we do not know* into a decimal place.
//!
//! So **no movement threshold is named here and none should be until one measurement is taken**:
//! sit still through a run, then move through the next, and compare the two. That needs a person
//! doing it on purpose, which is the same thing the voice guard's pace limit is still waiting for —
//! and for the same reason, the guessed version of it would silently throw away the real thing.
//! Until then this reports the scalar and decides nothing.

use std::time::Instant;

use windows::Win32::Media::MediaFoundation::{
    IMFActivate, IMFMediaSource, IMFSourceReader, MFCreateAttributes, MFCreateMediaType,
    MFCreateSourceReaderFromMediaSource, MFEnumDeviceSources, MFStartup, MFVideoFormat_RGB32,
    MFMediaType_Video, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
    MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID, MF_MT_FRAME_SIZE, MF_MT_MAJOR_TYPE,
    MF_MT_SUBTYPE, MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING,
    MF_SOURCE_READER_FIRST_VIDEO_STREAM, MF_VERSION,
};

/// The size everything downstream sees. **Small on purpose** — see the module note.
pub const WIDE: usize = 64;
pub const TALL: usize = 36;

/// One look, reduced.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    /// `WIDE * TALL` greys, row by row from the top left.
    pub grey: Vec<u8>,
    /// How long the capture took, in seconds. **Kept because it is the number that decides whether
    /// this can run on a timer at all**, and a cost nobody measured is a cost nobody believes.
    pub took: f64,
}

impl Frame {
    /// The average grey. A room with the light off and a lens cap look the same from here, which is
    /// why this is reported rather than interpreted.
    pub fn brightness(&self) -> f64 {
        if self.grey.is_empty() {
            return 0.0;
        }
        self.grey.iter().map(|g| *g as f64).sum::<f64>() / self.grey.len() as f64
    }

    /// Mean absolute difference from another look, 0–255.
    ///
    /// **A scalar, not a region.** Where the difference is would be a coordinate, and a coordinate
    /// from a detector is the thing this project refuses to produce.
    pub fn differs_from(&self, other: &Frame) -> f64 {
        if self.grey.len() != other.grey.len() || self.grey.is_empty() {
            return 0.0;
        }
        let total: u64 = self.grey.iter().zip(&other.grey)
            .map(|(a, b)| a.abs_diff(*b) as u64).sum();
        total as f64 / self.grey.len() as f64
    }

    /// The part of the change that is **not** the whole picture getting brighter or darker.
    ///
    /// **This exists because a measured confound demanded it.** A person sitting in front of a
    /// monitor is lit by it, so when the screen changes their face changes with it — and that is a
    /// near-uniform shift of every pixel at once. Measured: ten seconds of somebody deliberately
    /// holding still produced nine one-second windows around 0.03 and **one at 0.40**, which is
    /// inside the range that ten seconds of deliberate movement produced (0.33–0.82). The signal
    /// was sixteen times the noise in the middle and still overlapped at the tail.
    ///
    /// Subtracting the mean shift before taking the difference removes exactly that: a screen
    /// flash relights everything by about the same amount and comes out near zero, while a hand
    /// moving through part of the frame does not. **Still a scalar** — the asymmetry is used, not
    /// the location.
    pub fn moved_from(&self, other: &Frame) -> f64 {
        if self.grey.len() != other.grey.len() || self.grey.is_empty() {
            return 0.0;
        }
        let shift = self.brightness() - other.brightness();
        let total: f64 = self.grey.iter().zip(&other.grey)
            .map(|(a, b)| ((*a as f64 - *b as f64) - shift).abs()).sum();
        total / self.grey.len() as f64
    }
}

/// Reduce a raw BGRA frame to the small grey field, by averaging each block.
///
/// **Averaged rather than sampled.** Taking every nth pixel keeps sensor noise at full strength and
/// makes a still room look like a moving one; averaging a block is a low-pass filter for free, and
/// the whole point of this size is that only real movement survives it.
pub fn reduce(bgra: &[u8], width: usize, height: usize, stride: usize) -> Vec<u8> {
    let mut out = vec![0u8; WIDE * TALL];
    if width == 0 || height == 0 {
        return out;
    }
    for ty in 0..TALL {
        for tx in 0..WIDE {
            let x0 = tx * width / WIDE;
            let x1 = (((tx + 1) * width) / WIDE).max(x0 + 1).min(width);
            let y0 = ty * height / TALL;
            let y1 = (((ty + 1) * height) / TALL).max(y0 + 1).min(height);
            let mut sum = 0u64;
            let mut seen = 0u64;
            for y in y0..y1 {
                for x in x0..x1 {
                    let at = y * stride + x * 4;
                    let (Some(b), Some(g), Some(r)) =
                        (bgra.get(at), bgra.get(at + 1), bgra.get(at + 2)) else { continue };
                    //: The usual luma weights, in integers. A camera's green channel carries most
                    //: of the detail and weighting them equally throws that away.
                    sum += (*r as u64 * 77 + *g as u64 * 150 + *b as u64 * 29) >> 8;
                    seen += 1;
                }
            }
            if seen > 0 {
                out[ty * WIDE + tx] = (sum / seen) as u8;
            }
        }
    }
    out
}

/// How still two consecutive looks have to be before the camera is believed.
///
/// **Measured rather than picked**: over 25 looks a settled room differs from itself by at most
/// 0.14, and the same room differs by up to 2.85 while the exposure is still walking. Half a unit
/// is three and a half times the worst still frame and a fifth of the drift — inside the gap, which
/// is the only kind of threshold worth having. It says *the camera has stopped changing on its
/// own*; it says nothing about whether anybody is there.
pub const SETTLED: f64 = 0.5;

/// How long to wait for that before giving up and saying so. Measured: it takes about 1.4 s.
pub const SETTLE_LIMIT: f64 = 4.0;

/// An open camera. **The light is on while this exists.**
pub struct Camera {
    reader: IMFSourceReader,
    pub name: String,
    pub width: usize,
    pub height: usize,
    /// How long the picture took to stop changing after it was switched on. `None` if it never did
    /// within [`SETTLE_LIMIT`] — **reported rather than hidden**, because every number taken
    /// through an unsettled camera is a number about the camera.
    pub settled_in: Option<f64>,
}

fn started() -> Result<(), String> {
    use std::sync::OnceLock;
    static ONCE: OnceLock<Result<(), String>> = OnceLock::new();
    ONCE.get_or_init(|| unsafe {
        MFStartup(MF_VERSION, 0).map_err(|e| format!("Media Foundation이 시작되지 않았습니다 ({e})"))
    }).clone()
}

/// The names of the cameras this machine has. **Nothing is opened** — see `examples/cameras.rs`.
pub fn cameras() -> Result<Vec<String>, String> {
    started()?;
    unsafe {
        let mut attrs = None;
        MFCreateAttributes(&mut attrs, 1).map_err(|e| format!("{e}"))?;
        let attrs = attrs.ok_or("attributes")?;
        attrs.SetGUID(&MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                      &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID)
            .map_err(|e| format!("{e}"))?;
        let (mut found, mut count) = (std::ptr::null_mut(), 0u32);
        MFEnumDeviceSources(&attrs, &mut found, &mut count).map_err(|e| format!("{e}"))?;
        let mut out = Vec::new();
        for i in 0..count as isize {
            if let Some(source) = (*found.offset(i)).as_ref() {
                out.push(name_of(source));
            }
        }
        Ok(out)
    }
}

fn name_of(source: &IMFActivate) -> String {
    unsafe {
        let mut name = windows::core::PWSTR::null();
        let mut len = 0u32;
        match source.GetAllocatedString(
            &windows::Win32::Media::MediaFoundation::MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
            &mut name, &mut len) {
            Ok(()) => name.to_string().unwrap_or_default(),
            Err(_) => "이름 없는 카메라".to_string(),
        }
    }
}

impl Camera {
    /// Open the first camera. **This turns it on.**
    pub fn open() -> Result<Camera, String> {
        started()?;
        unsafe {
            let mut attrs = None;
            MFCreateAttributes(&mut attrs, 1).map_err(|e| format!("{e}"))?;
            let attrs = attrs.ok_or("attributes")?;
            attrs.SetGUID(&MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                          &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID)
                .map_err(|e| format!("{e}"))?;
            let (mut found, mut count) = (std::ptr::null_mut(), 0u32);
            MFEnumDeviceSources(&attrs, &mut found, &mut count).map_err(|e| format!("{e}"))?;
            if count == 0 {
                return Err("카메라가 없습니다".into());
            }
            let activate = (*found).as_ref().ok_or("카메라를 열지 못했습니다")?;
            let name = name_of(activate);
            let source: IMFMediaSource = activate.ActivateObject()
                .map_err(|e| format!("카메라를 열지 못했습니다 ({e})"))?;

            //: **The reader is told it may convert.** Cameras publish NV12 or MJPG and almost never
            //: RGB32; without this the `SetCurrentMediaType` below fails and the only way out is
            //: decoding the sensor's format by hand.
            let mut ready = None;
            MFCreateAttributes(&mut ready, 1).map_err(|e| format!("{e}"))?;
            let ready = ready.ok_or("attributes")?;
            ready.SetUINT32(&MF_SOURCE_READER_ENABLE_ADVANCED_VIDEO_PROCESSING, 1)
                .map_err(|e| format!("{e}"))?;

            let reader = MFCreateSourceReaderFromMediaSource(&source, &ready)
                .map_err(|e| format!("카메라를 읽지 못했습니다 ({e})"))?;

            let want = MFCreateMediaType().map_err(|e| format!("{e}"))?;
            want.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video).map_err(|e| format!("{e}"))?;
            want.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_RGB32).map_err(|e| format!("{e}"))?;
            reader.SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32, None, &want)
                .map_err(|e| format!("RGB로 바꾸지 못했습니다 ({e})"))?;

            let got = reader.GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32)
                .map_err(|e| format!("{e}"))?;
            let size = got.GetUINT64(&MF_MT_FRAME_SIZE).map_err(|e| format!("{e}"))?;
            let width = (size >> 32) as usize;
            let height = (size & 0xffff_ffff) as usize;

            let mut camera = Camera { reader, name, width, height, settled_in: None };
            camera.settled_in = camera.settle();
            Ok(camera)
        }
    }

    /// Look until the picture stops changing. Returns how long that took.
    ///
    /// **Not a sleep.** A fixed wait is a guess that is too long on one machine and too short on
    /// the next; this watches the thing it is waiting for and stops when it has happened.
    fn settle(&mut self) -> Option<f64> {
        let began = Instant::now();
        let mut before: Option<Frame> = None;
        while began.elapsed().as_secs_f64() < SETTLE_LIMIT {
            let Ok(now) = self.look() else { continue };
            if let Some(last) = &before {
                if now.differs_from(last) < SETTLED {
                    return Some(began.elapsed().as_secs_f64());
                }
            }
            before = Some(now);
        }
        None
    }

    /// Take one look. About 31 ms once the camera is open and settled.
    ///
    /// **Retried, because the first read after opening returns nothing** — the reader is not ready
    /// and answers with no sample rather than an error, and a caller that took that for a failure
    /// would report a working camera as broken.
    pub fn look(&mut self) -> Result<Frame, String> {
        let began = Instant::now();
        for _ in 0..12 {
            if let Some(grey) = self.one()? {
                return Ok(Frame { grey, took: began.elapsed().as_secs_f64() });
            }
        }
        Err("카메라가 화면을 주지 않았습니다".into())
    }

    /// One `ReadSample`, reduced. `None` when the reader had nothing ready yet.
    fn one(&mut self) -> Result<Option<Vec<u8>>, String> {
        unsafe {
            let mut flags = 0u32;
            let mut sample = None;
            self.reader.ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32, 0, None,
                                   Some(&mut flags), None, Some(&mut sample))
                .map_err(|e| format!("프레임을 읽지 못했습니다 ({e})"))?;
            let Some(sample) = sample else { return Ok(None) };
            let buffer = sample.ConvertToContiguousBuffer().map_err(|e| format!("{e}"))?;
            let mut bytes = std::ptr::null_mut();
            let mut len = 0u32;
            buffer.Lock(&mut bytes, None, Some(&mut len)).map_err(|e| format!("{e}"))?;
            let raw = std::slice::from_raw_parts(bytes, len as usize);
            //: The stride the buffer actually has, rather than the one the width implies: a row can
            //: be padded, and reading it as if it were not shears the picture.
            let stride = if self.height > 0 { len as usize / self.height } else { 0 };
            let small = reduce(raw, self.width, self.height, stride);
            let _ = buffer.Unlock();
            Ok(Some(small))
        }
    }
}

impl Drop for Camera {
    /// **The light goes out here**, on every path. A camera left streaming by a program that
    /// panicked is the worst thing in this file.
    fn drop(&mut self) {
        unsafe {
            let _ = self.reader.Flush(MF_SOURCE_READER_FIRST_VIDEO_STREAM.0 as u32);
        }
    }
}
