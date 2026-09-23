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
//! **A frame is reduced to 64×36 greys, and until 2026-09-23 that was all this module could
//! produce.** [`Frame`] holds [`WIDE`]×[`TALL`] bytes and has no accessor for anything larger; at
//! that size a room is light and shadow and movement — enough to answer *is somebody there*, and
//! not a photograph of anybody's living room. That is still the only thing [`Camera::look`] gives
//! and the only thing the movement sense uses.
//!
//! ## The second call, and what it costs to have one
//!
//! This file used to say: *if a vision model is ever wanted it will need a different call, a
//! different permission and a different argument.* The owner asked for one on 2026-09-23, so
//! [`Camera::photograph`] exists. **It takes the size as an argument**, which is deliberate: there
//! is no call that hands back *whatever the sensor had*, so the caller always states, in the code,
//! how much of a room it is asking for.
//!
//! **Be plain about what was given up.** It was impossible to get a picture out of here; now it is
//! possible and gated by something weaker than a type. What stands in that place is a rule read off
//! the source — **the application cannot take a photograph at all**, only a harness a person runs
//! on purpose, and `argmind-rules` checks it. When the application does want one it will need the
//! permission this note has always said it would, and moving that rule will be a deliberate act
//! rather than a drift.
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

/// A picture, at a size the caller asked for. **RGB, three bytes a pixel, row by row.**
pub struct Photo {
    pub rgb: Vec<u8>,
    pub wide: usize,
    pub tall: usize,
    /// How long the capture took, in seconds.
    pub took: f64,
}

/// Box-average a BGRA buffer down to `wide`×`tall` RGB.
///
/// **The same shape as [`reduce`] and deliberately not the same function.** That one produces one
/// byte a block and this produces three, and it sits on the three-second movement path that
/// `MOVING = 0.15` was measured against — rewriting it in terms of this would put a measured
/// constant behind a refactor nobody asked for.
pub fn shrink(bgra: &[u8], width: usize, height: usize, stride: usize,
              wide: usize, tall: usize) -> Vec<u8> {
    let mut out = vec![0u8; wide * tall * 3];
    if width == 0 || height == 0 || wide == 0 || tall == 0 {
        return out;
    }
    for ty in 0..tall {
        for tx in 0..wide {
            let x0 = tx * width / wide;
            let x1 = (((tx + 1) * width) / wide).max(x0 + 1).min(width);
            let y0 = ty * height / tall;
            let y1 = (((ty + 1) * height) / tall).max(y0 + 1).min(height);
            let (mut r, mut g, mut b, mut seen) = (0u64, 0u64, 0u64, 0u64);
            for y in y0..y1 {
                for x in x0..x1 {
                    let at = y * stride + x * 4;
                    let (Some(bb), Some(gg), Some(rr)) =
                        (bgra.get(at), bgra.get(at + 1), bgra.get(at + 2)) else { continue };
                    b += *bb as u64;
                    g += *gg as u64;
                    r += *rr as u64;
                    seen += 1;
                }
            }
            if seen > 0 {
                let at = (ty * wide + tx) * 3;
                out[at] = (r / seen) as u8;
                out[at + 1] = (g / seen) as u8;
                out[at + 2] = (b / seen) as u8;
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

/// Above this, somebody moved. **Measured on a person doing it on purpose, twice** — 2026-09-22.
///
/// Ten seconds holding still and ten seconds moving, read as the median of a three-second window
/// with the mean shift removed ([`Frame::moved_from`]):
///
/// | | run 1 | run 2 |
/// |---|---|---|
/// | highest still window | 0.05 | 0.06 |
/// | lowest moving window | 0.46 | 0.41 |
///
/// **The still side is the tight one** — four values across two runs, 0.05 to 0.06 — and the moving
/// side is the loose one, which is what varies: how much a person moves. So the room to leave is on
/// the moving side, and this sits at the geometric middle of the gap, **2.5x above the highest
/// still window and 2.7x below the lowest moving one.**
///
/// **It says movement, and movement is not presence.** A person reading a screen without shifting
/// is below this, and so is an empty chair. What the caller may say from it is *movement was last
/// seen N seconds ago* — the number, with its age — and not *somebody is there*. This file reports
/// scalars and decides nothing; that has been its rule since it was written and it is not relaxed
/// by finally having a floor.
///
/// **Three-second windows, not one.** One-second ones separate too, at 3.1x and 3.3x against 6.2x
/// and 6.8x — a monitor changing what it shows lasts a fraction of a second and a person moving
/// lasts seconds, which is what `measure/presence` predicted before it was run.
pub const MOVING: f64 = 0.15;

/// The window [`MOVING`] is a threshold on. A median over anything shorter is a different number.
pub const MOVING_WINDOW: f64 = 3.0;

/// How long to wait for that before giving up and saying so.
///
/// **Four seconds until 2026-09-22, and it was four seconds for a settle check that could not
/// spend them.** The old one returned on the first consecutive pair under [`SETTLED`], and a
/// warming camera produces those constantly — frame-level median 1.50 with a ninetieth of 4.72
/// still has a quiet pair every few frames. It reported 329–352 ms every time and the picture went
/// on moving for most of ten seconds afterwards. With a window instead of a pair the wait is real,
/// so the limit has to be long enough to contain it.
///
/// **Fifteen, from two runs and not from a measurement of the settle time itself.** Cold, one-second
/// windows read 0.57–2.84 between three and seven seconds after opening and 0.39–0.72 between ten
/// and fourteen; warm, 0.10–0.28 from the start. So it is done somewhere between seven and ten
/// seconds cold, n=1, and fifteen is that with room rather than a number anybody derived.
pub const SETTLE_LIMIT: f64 = 15.0;

/// How long a stretch has to be quiet before the camera counts as settled.
///
/// **A window, because a pair is not a statistic** — `measure/presence` reached that conclusion
/// about this same signal and for the same reason: the distribution is *almost always nothing with
/// rare enormous spikes*, so one sample says nothing in either direction. A second of it cannot be
/// crossed by a single flash and cannot be faked by a camera that is still walking its exposure.
pub const SETTLE_WINDOW: f64 = 1.0;

/// Fewer differences than this in a window and it is not a median of anything. This camera has
/// been seen at 13–16 frames a second and at 30, so a second holds between fourteen and thirty.
pub const ENOUGH_TO_JUDGE: usize = 6;

/// How many quiet windows in a row before the camera counts as settled.
///
/// **Two, because one was still not enough and the run said so.** With a single quiet window the
/// check reported 1030 ms on a cold open and `presence`'s first phase then read a median of 0.22
/// with a **2.99** in it, against 0.07 in the phase ten seconds later with nobody moving in either.
/// Cold, quiet seconds alternate with drifting ones, so one of them is a coin landing the right way
/// up. Two in a row is the same criterion asked twice rather than a longer sleep — this file
/// refuses fixed waits, and a guessed ten seconds is what that would be.
pub const SETTLE_RUNS: usize = 2;

/// Is a window of consecutive differences quiet enough to call the camera settled?
///
/// `None` when there are too few to judge — **which is not the same as *not settled***, and saying
/// so is the point of the type. The caller waits for another window rather than concluding
/// anything from four frames.
///
/// **Public and pure so the defect it replaced can be written down as a test.** The old check was
/// `one pair < SETTLED`, and the reason that fails is a property of the numbers rather than of the
/// code: a warming camera's differences are *almost always nothing with rare enormous spikes*, so a
/// quiet pair arrives within a few frames of opening while the picture is still walking. A median
/// cannot be moved by one sample in either direction, which is the whole reason to take one.
pub fn quiet_enough(window: &mut Vec<f64>) -> Option<bool> {
    if window.len() < ENOUGH_TO_JUDGE {
        return None;
    }
    window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(window[window.len() / 2] < SETTLED)
}

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
        Camera::on(None)
    }

    /// Open the one with this name, or the first when there is no name or it has gone.
    ///
    /// **By name, unlike the audio endpoints, and that is a weaker key.** Media Foundation's stable
    /// handle is a symbolic link that no person could pick out of a list, and nothing here has
    /// measured whether a friendly name survives a driver update. So this is the choice with the
    /// known weakness rather than the one with the unknown one, and falling back to the first
    /// camera is what happens when the name stops matching.
    pub fn on(want: Option<&str>) -> Result<Camera, String> {
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
            //: The one that was asked for, else the first. `count` is at least one here.
            let mut pick = 0isize;
            if let Some(wanted) = want.filter(|w| !w.is_empty()) {
                for i in 0..count as isize {
                    if let Some(one) = (*found.offset(i)).as_ref() {
                        if name_of(one) == wanted {
                            pick = i;
                            break;
                        }
                    }
                }
            }
            let activate = (*found.offset(pick)).as_ref().ok_or("카메라를 열지 못했습니다")?;
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
    ///
    /// **And not one pair, since 2026-09-22.** The first version returned as soon as any two
    /// consecutive looks differed by less than [`SETTLED`], which a warming camera offers within
    /// about five frames — it reported 329–352 ms on every open while the picture went on moving
    /// ten times the settled floor for most of ten seconds. Everything measured in that window was
    /// measured on a camera this function had called ready.
    ///
    /// **Found by a control nobody had run**: `measure/presence` asks a person to hold still and
    /// then move, and its still phase goes first. Run with nobody moving in either phase it
    /// separated them 2.17 against 0.65 — three times, in the wrong direction — and running it
    /// again with the camera warm collapsed the first phase to 0.22. The prediction was written
    /// before that second run.
    fn settle(&mut self) -> Option<f64> {
        let began = Instant::now();
        let mut before: Option<Frame> = None;
        let mut window: Vec<f64> = Vec::new();
        let mut ends = SETTLE_WINDOW;
        let mut in_a_row = 0usize;
        while began.elapsed().as_secs_f64() < SETTLE_LIMIT {
            let Ok(now) = self.look() else { continue };
            if let Some(last) = &before {
                window.push(now.differs_from(last));
            }
            before = Some(now);
            //: **Cut by the clock rather than by a count**, the same as `presence`: this camera has
            //: been measured at 32 ms a look and at 111 ms, so a window of *n* frames is a window
            //: of between half a second and a second and a half depending on the day.
            let so_far = began.elapsed().as_secs_f64();
            if so_far < ends {
                continue;
            }
            ends += SETTLE_WINDOW;
            match quiet_enough(&mut window) {
                //: **A loud window resets the count**, so the two have to be consecutive — two
                //: quiet seconds either side of a drifting one is a drifting camera.
                Some(true) => in_a_row += 1,
                Some(false) => in_a_row = 0,
                None => {}
            }
            window.clear();
            if in_a_row >= SETTLE_RUNS {
                return Some(so_far);
            }
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

    /// Take one picture, at the size asked for. **The light is already on** — this does not open
    /// the device, it reads from one that [`Camera::on`] opened and settled.
    ///
    /// **The size is an argument and there is no call without one.** See the module note: the point
    /// is that the caller says in its own source how much of a room it wants, so the question can be
    /// answered by reading the code rather than by running it.
    pub fn photograph(&mut self, wide: usize, tall: usize) -> Result<Photo, String> {
        let began = Instant::now();
        for _ in 0..12 {
            if let Some(rgb) = self.one_with(&mut |bgra, w, h, stride| {
                shrink(bgra, w, h, stride, wide, tall)
            })? {
                return Ok(Photo { rgb, wide, tall, took: began.elapsed().as_secs_f64() });
            }
        }
        Err("카메라가 화면을 주지 않았습니다".into())
    }

    /// One `ReadSample`, reduced. `None` when the reader had nothing ready yet.
    fn one(&mut self) -> Result<Option<Vec<u8>>, String> {
        self.one_with(&mut |bgra, w, h, stride| reduce(bgra, w, h, stride))
    }

    /// One `ReadSample`, handed to whoever is reducing it.
    ///
    /// **One place talks to Media Foundation and two things reduce what it gives.** A second copy
    /// of this would be a second place to get the buffer lifetime wrong, and the `Unlock` at the
    /// end of it is the whole reason that matters.
    fn one_with<T>(&mut self, reduce_it: &mut dyn FnMut(&[u8], usize, usize, usize) -> T)
                   -> Result<Option<T>, String> {
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
            let small = reduce_it(raw, self.width, self.height, stride);
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
