//! **The one file in this crate that knows which operating system this is.**
//!
//! Everything else in `argmind-serve` says `platform::Sight` and `platform::Touch` and does not
//! know or care what they are. That is the same discipline `argmind-core` keeps — and until
//! 2026-09-16 the application did not keep it: `app.rs`, `needs.rs`, `recogniser.rs`, `speaker.rs`
//! and both binaries named Windows types directly, in twenty-one places. **Adding a second
//! environment would have meant rewriting the application rather than plugging something in**, and
//! that, rather than the 1,844 lines of `argmind-env-windows`, was what actually blocked the port
//! the owner asked for.
//!
//! `argmind-rules` holds this by reading the source, the way it holds the core's blindness.
//!
//! ## What the choice is
//!
//! Windows gets `argmind-env-windows`. Everything else gets [`argmind_env_none`], which sees
//! nothing, presses nothing and says so — because there is no macOS or Linux environment yet and
//! **pretending is worse than refusing.**
//!
//! `--features headless` forces the empty one even on Windows. That is not a convenience: it is
//! how a claim about portability gets checked on a machine that only has one operating system.
//! `tests/portable.rs` builds and runs the whole application that way, so if this seam is a
//! fiction it does not compile.
//!
//! ## The names are neutral and the shapes are not a contract
//!
//! `Sight`, `Touch`, `Mouth`, `Ear`, `Watching` — `Eyes` and `Hands` are traits in the core, and
//! these are the concrete types behind them plus the capabilities that have no trait yet: knowing
//! which window a person is attending to, a microphone, starting a supervised child, and asking a
//! yes-or-no question.
//!
//! **Those four are interface rather than contract**, in the same sense `argmind-voice` says its
//! three traits are. Nothing yet justifies promoting them — a contract argued from one
//! implementation is a contract shaped like that implementation, and there will not be a second
//! opinion until somebody writes `argmind-env-linux`. What is written down instead is the list:
//! **whatever is in `argmind-env-none/src/lib.rs` is what a third environment must provide**, and
//! it was not guessed — the compiler enumerated it.

//: **Re-exported from the core rather than from an environment.** Two environments now report what
//: an action did, and a second copy of that enum is how they would come to mean different things.
pub use argmind_core::contract::Did;

#[cfg(all(windows, not(feature = "headless")))]
mod chosen {
    pub use argmind_env_windows::ask::confirm;
    pub use argmind_env_windows::microphone::WANT_HZ;
    pub use argmind_env_windows::supervise::Supervised;
    pub use argmind_env_windows::{foreground_handle_now, window_title};
    pub use argmind_env_windows::{Attending as Sight, Microphone as Ear, SapiSpeaker as Mouth,
                                  Watcher as Watching, WindowsEyes as Plain,
                                  WindowsHands as Touch};

    /// Is there a neural voice to prefer over the built-in one?
    ///
    /// **Its own function because the answer is not the same everywhere.** The neural voice needs
    /// a loudspeaker, and the only one written is WASAPI — so on an environment with no audio
    /// there is nothing to prefer and the question is answered at compile time rather than by a
    /// socket timeout.


    /// Everything of one kind this machine has.
    pub fn devices(kind: &str) -> Result<Vec<super::Device>, String> {
        let found = match kind {
            "microphone" => argmind_env_windows::devices::microphones()?,
            "speaker" => argmind_env_windows::devices::speakers()?,
            "camera" => {
                //: Cameras come back as names with no id, so the name is both — see
                //: `argmind-env-windows/src/devices.rs` for why that asymmetry is deliberate.
                return Ok(argmind_env_windows::camera::cameras()?.into_iter().enumerate()
                    .map(|(i, name)| super::Device { id: name.clone(), name, default: i == 0 })
                    .collect());
            }
            _ => return Err(format!("{kind} 라는 장치 종류는 없습니다")),
        };
        Ok(found.into_iter()
            .map(|d| super::Device { id: d.id, name: d.name, default: d.default })
            .collect())
    }

    /// **Use one, briefly, and say what happened.**
    ///
    /// The one place in this program that opens a device because somebody asked it to, rather than
    /// because a turn needed it. Every one of these makes a sound, records, or looks — which is
    /// the point of pressing the button, and is why nothing here happens without one being pressed.
    pub fn try_device(kind: &str, id: Option<&str>) -> super::Tried {
        match kind {
            "speaker" => try_speaker(id),
            "microphone" => try_microphone(id),
            "camera" => try_camera(id),
            _ => super::Tried { worked: false, said: format!("{kind} 는 시험할 수 없습니다") },
        }
    }

    /// A second of a quiet tone, faded at both ends so the start and the end are not clicks.
    fn try_speaker(id: Option<&str>) -> super::Tried {
        use argmind_env_windows::loudspeaker::{Loudspeaker, FROM_HZ};
        let n = FROM_HZ as usize;
        let fade = n / 20;
        let tone: Vec<f32> = (0..n).map(|i| {
            let t = i as f64 / FROM_HZ as f64;
            let level = 0.12 * (i.min(fade) as f64 / fade as f64)
                .min((n - i).min(fade) as f64 / fade as f64);
            (level * (t * 440.0 * std::f64::consts::TAU).sin()) as f32
        }).collect();

        let mut mouth = Loudspeaker::on(id);
        mouth.play(&tone, FROM_HZ);
        let began = std::time::Instant::now();
        while !mouth.speaking() && began.elapsed().as_secs_f64() < 2.0 {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let started = mouth.speaking();
        std::thread::sleep(std::time::Duration::from_millis(1200));
        let missed = mouth.fell_back();
        mouth.release();
        super::Tried {
            worked: started,
            said: if !started {
                "소리를 내지 못했습니다".into()
            } else if missed {
                "삐 소리를 냈습니다 — 고른 장치가 없어서 기본 장치로 냈습니다".into()
            } else {
                "삐 소리를 1초 냈습니다 — 들리셨나요?".into()
            },
        }
    }

    /// **Listen for a moment and report the loudest thing heard**, as a number.
    ///
    /// Not a transcription. What a person is checking is whether this device hears them at all,
    /// and a peak answers that without a model, a language or a wait.
    fn try_microphone(id: Option<&str>) -> super::Tried {
        use argmind_env_windows::Microphone;
        let mut ear = Microphone::on(id);
        ear.hold(true);
        std::thread::sleep(std::time::Duration::from_millis(1800));
        ear.hold(false);
        std::thread::sleep(std::time::Duration::from_millis(250));
        let clip = ear.take();
        let missed = ear.fell_back();
        ear.release();

        let Some(clip) = clip else {
            return super::Tried { worked: false, said: "아무것도 녹음되지 않았습니다".into() };
        };
        let peak = clip.peak();
        let seconds = clip.samples.len() as f64
            / argmind_env_windows::microphone::WANT_HZ as f64;
        //: **A floor, not a judgement.** Below this is indistinguishable from a quiet room, and
        //: saying so is more useful than a number nobody can interpret.
        let heard = peak > 0.02;
        super::Tried {
            worked: heard,
            said: format!("{seconds:.1}초 녹음, 가장 큰 소리 {:.0}%{}{}", peak * 100.0,
                          if heard { " — 들립니다" }
                          else { " — 너무 조용합니다. 말하면서 눌러보세요" },
                          if missed { " (고른 장치가 없어 기본 장치로 녹음했습니다)" }
                          else { "" }),
        }
    }

    /// **One frame, and how bright it was.** Enough to tell a covered lens from a working one.
    fn try_camera(name: Option<&str>) -> super::Tried {
        use argmind_env_windows::camera::Camera;
        let mut eye = match Camera::on(name) {
            Ok(eye) => eye,
            Err(why) => return super::Tried { worked: false, said: why },
        };
        //: **The first frame is often black** while the sensor is still setting its exposure, so a
        //: few are taken and the last one is the answer. Measured when the presence harness was
        //: written and true of this camera.
        let mut last = None;
        for _ in 0..5 {
            last = eye.look().ok();
            std::thread::sleep(std::time::Duration::from_millis(120));
        }
        match last {
            Some(frame) => {
                let light = frame.brightness();
                super::Tried {
                    worked: light > 2.0,
                    said: format!("한 장 찍었습니다 — 밝기 {light:.0}/255{}",
                                  if light > 2.0 { "" }
                                  else { " — 거의 검습니다. 렌즈 덮개를 확인해보세요" }),
                }
            }
            None => super::Tried { worked: false, said: "화면을 받지 못했습니다".into() },
        }
    }

    /// What speaks when the neural voice is not there, in the words a person would recognise.
    pub fn fallback_voice() -> Option<&'static str> {
        Some("Windows 음성")
    }


    /// Start this child **without giving it a console window.**
    ///
    /// A windowed program has no console, so Windows makes one for any console child it starts —
    /// and argmind starts a great many: `git`, `cmake`, `objdump`, `curl.exe`, `certutil`, `reg`,
    /// `where`, and five `--version` calls every time the page asks what is missing. Each one is a
    /// black rectangle that appears and vanishes. The owner's word for it on 2026-09-17 was
    /// 거슬림.
    ///
    /// `CREATE_NO_WINDOW`, which is `0x0800_0000`. **It does nothing to a program that has a window
    /// of its own** — an installer still shows its installer — so it is safe on everything here,
    /// and every child in this program goes through it.
    pub fn without_a_console(command: &mut std::process::Command) {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }

    /// Hand a URL to whatever the person chose as their browser.
    pub fn open_url(url: &str) -> Result<(), String> {
        //: `explorer` is a shell command, not a downloader — nothing is fetched or run here.
        super::child("explorer").arg(url).spawn()
            .map(|_| ()).map_err(|e| format!("{e}"))
    }

    /// Run a documented command with administrator rights, so Windows asks as well.
    pub fn run_elevated(command: &str) -> Result<bool, String> {
        let done = super::child("powershell")
            .args(["-NoProfile", "-Command",
                   &format!("Start-Process -Verb RunAs -Wait -FilePath cmd.exe                              -ArgumentList '/c {command}'")])
            .status().map_err(|e| format!("{e}"))?;
        Ok(done.success())
    }

    /// **The voice is read and *taught* here**, and that is deliberate: this is the one place a
    /// neural voice is built, so *which voice* is decided in one place too. A caller that could
    /// pass a different one is a caller that could pass none.
    ///
    /// **Teaching costs 0.69 s and happens here rather than on the way to a reply** — 0.53 s to
    /// draw the reference and 0.16 s to extract the speaker out of it, measured. It has to happen
    /// at all because the server keeps registered voices in memory and hands back no latents to
    /// cache, so a restart forgets them.
    ///
    /// **A failure falls back rather than going quiet.** A seed does not hold a person across
    /// sentences — that is the whole reason for the clone — but it holds each sentence, which beats
    /// a fresh draw every turn, and it beats silence by more.
    /// Keep the camera open and say, every few seconds, whether anybody moved.
    ///
    /// **The light stays on for as long as this runs, and that is the owner's choice** — 2026-09-22,
    /// against opening and closing on a timer. Opening costs 2.3–4.4 s of settling to take 3 s of
    /// readings, and a light that blinks every half minute suggests a camera that is mostly off,
    /// which would not be true. A light that is on means a camera that is on.
    ///
    /// **The threshold lives here rather than in the caller.** `camera.rs` publishes
    /// [`argmind_env_windows::camera::MOVING`] and the scalar to compare against it and decides
    /// nothing itself; the application may not name a platform at all. This file is the one that
    /// may, so the comparison is here and what crosses the seam is a yes or a no.
    ///
    /// `carry_on` is asked before every look, so switching the setting off closes the device within
    /// about one frame and `Drop` puts the light out.
    ///
    /// **It does not stand down for a turn**, which every other sense does. That rule exists
    /// because a look and an answer contend for one graphics card; this never speaks to the model.
    /// The hands the application uses. **Patterns on, since 2026-09-22.**
    ///
    /// `patterns.rs` measured them in 2026-09-16 and left them off, for a reason it wrote down:
    /// *the shipped click is confirmed by a person pressing a button, so the cursor was already
    /// about to move to something they chose.* **확인 없이 실행 removed that reason the day before
    /// this** — nobody is pressing, so nobody chose to have the pointer taken.
    ///
    /// What the measurement says, three runs each: task success **3 of 3 either way**, reaching
    /// past something covering the target **missed either way** (the pattern resolves by
    /// `ElementFromPoint` and walks up, so a cover hands it the cover), and the pointer **unmoved
    /// against dragged to the widget, 3 of 3**. One axis, and it is the one that matters on a
    /// machine somebody is working at. n=3 on all of it.
    ///
    /// **It is not a second path past the guard**: `click_button` checks `may_touch` before it
    /// chooses a path, and falls back to a click when no pattern fits.
    /// **And typing through one too, since 2026-09-23.** Measured on Character Map, a fresh window
    /// per case: both `hello` and `안녕하세요 반갑습니다` went in through `ValuePattern`, and a box
    /// already holding `먼저` was appended to rather than replaced. It writes at the point of the
    /// widget argmind just clicked, so in the 250 ms between the click and the typing a person can
    /// click into a box of their own and argmind's sentence still goes where argmind aimed.
    ///
    /// **Where it does not apply it changes nothing** — no value pattern, read-only, or anything
    /// already in the box, and it is the keyboard exactly as before. The residual risk is a control
    /// that supports `ValuePattern`, reports itself blank, and holds content anyway; that is
    /// unmeasured and would need other applications. One application, one run.
    pub fn hands(dry: bool) -> Touch {
        let hands = Touch::unrestricted().with_patterns().with_typing_patterns();
        if dry { hands } else { hands.arm() }
    }

    /// **Yes** — there is a camera here, so a face detector has something to look at.
    pub fn watches_people() -> bool {
        true
    }

    pub fn watch_movement(want: Option<&str>, carry_on: &dyn Fn() -> bool,
                          saw: &mut dyn FnMut(super::Seen),
                          attending: &dyn Fn() -> bool,
                          faces: &dyn Fn() -> bool,
                          met: &mut dyn FnMut(Vec<f32>)) -> Result<(), String> {
        use argmind_env_windows::camera::{Camera, ENOUGH_TO_JUDGE, MOVING, MOVING_WINDOW};
        let mut camera = Camera::on(want)?;
        //: **Asked every tick rather than read once**, so turning the switch on does not mean
        //: closing the camera and opening it again — which is a light going off and on for no
        //: reason a person watching it could explain. Switching it off drops the 38 MB model.
        let mut looker: Option<argmind_env_face::Faces> = None;
        let mut gave_up = false;
        let mut last_face = std::time::Instant::now();
        //: **Accumulated across the window and classified at its edge**, the way movement already
        //: is. One frame compared against `HEAD_ASIDE` would be using a constant derived from
        //: medians against a statistic it was not derived from.
        let mut heads: Vec<f32> = Vec::new();
        let mut unreadable = 0u32;
        let mut faces_found = 0u32;
        let mut embedded = std::time::Instant::now();
        let mut broke: Option<String> = None;
        let mut before = None;
        let mut window: Vec<f64> = Vec::new();
        let mut began = std::time::Instant::now();
        //: A camera unplugged mid-run answers nothing for ever, and a loop that shrugged at that
        //: would spin a core reporting a person who left.
        let mut empty_handed = 0u32;
        while carry_on() {
            match camera.look() {
                Ok(now) => {
                    empty_handed = 0;
                    if let Some(last) = &before {
                        //: **The mean shift removed**, the statistic that measured 6.8x and 8.6x
                        //: against the plain one's 6.2x and 7.4x — a monitor relighting a face
                        //: moves every pixel by about the same amount and cancels here.
                        window.push(now.moved_from(last));
                    }
                    before = Some(now);
                }
                Err(_) => {
                    empty_handed += 1;
                    if empty_handed > 50 {
                        return Err("카메라가 더 이상 화면을 주지 않습니다".into());
                    }
                    continue;
                }
            }
            //: **Before the window closes**, so the readings belong to the window they are
            //: reported with. Below the boundary they would describe the previous one.
            let wanted = attending() || faces();
            let cadence = if attending() { super::FACE_PASS } else { super::FACE_EVERY };
            if wanted && last_face.elapsed().as_secs_f64() >= cadence {
                last_face = std::time::Instant::now();
                if looker.is_none() && !gave_up {
                    //: **A failure is not fatal.** Movement is what the light was turned on for,
                    //: and a missing model file must not close the camera.
                    //:
                    //: **And not silent either.** Until 2026-09-29 this went to stderr only, which
                    //: nobody running the window reads: the owner turned both face switches on,
                    //: saw nothing happen, and had no way to know whether a model had failed or a
                    //: face had simply not been found. It rides out on the next `Seen` instead.
                    match argmind_env_face::Faces::load(&crate::faces::finder_at())
                        .and_then(|f| f.also_remember(&crate::faces::namer_at())) {
                        Ok(f) => looker = Some(f),
                        Err(why) => {
                            eprintln!("  faces: 모델을 읽지 못했습니다 — {why}");
                            broke = Some(format!("얼굴 모델을 읽지 못했습니다 — {why}"));
                            gave_up = true;
                        }
                    }
                }
                if let Some(looker) = &looker {
                    if let Ok(shot) = camera.photograph(super::FACE_WIDE, super::FACE_TALL) {
                        match looker.find(&shot.rgb, shot.wide, shot.tall, super::A_FACE) {
                            Ok(found) if !found.is_empty() => {
                                faces_found += 1;
                                //: The largest: one person at a desk is the case this is for, and a
                                //: face on a poster behind them is smaller. **Two people is not
                                //: handled** and would show in the roster as one never being seen.
                                let biggest = found.iter().max_by(|a, b| (a.rect.2 * a.rect.3)
                                    .partial_cmp(&(b.rect.2 * b.rect.3))
                                    .unwrap_or(std::cmp::Ordering::Equal));
                                if let Some(face) = biggest {
                                    match face.head_on() {
                                        Some(h) => heads.push(h),
                                        None => unreadable += 1,
                                    }
                                    //: **Embedding stays on the slow cadence.** A stranger does not
                                    //: arrive three times a second, and 45 ms a pass spent for
                                    //: nothing is 45 ms movement did not get.
                                    if faces()
                                        && embedded.elapsed().as_secs_f64() >= super::FACE_EVERY {
                                        embedded = std::time::Instant::now();
                                        if let Ok(Some(print)) =
                                            looker.embed(&shot.rgb, shot.wide, shot.tall, face) {
                                            met(print);
                                        }
                                    }
                                }
                            }
                            Ok(_) => {}
                            Err(_) => {}
                        }
                    }
                }
            }
            if !wanted {
                //: Off means off: the model goes, and so does the memory of having failed to read
                //: it — somebody who fixes a missing file and switches back on should not be told
                //: about the old failure for the rest of the session.
                looker = None;
                gave_up = false;
                heads.clear();
                unreadable = 0;
            }

            if began.elapsed().as_secs_f64() < MOVING_WINDOW {
                continue;
            }
            let so_far = began.elapsed().as_secs_f64();
            began = std::time::Instant::now();
            if window.len() < ENOUGH_TO_JUDGE {
                window.clear();
                continue;
            }
            window.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let middle = window[window.len() / 2];
            window.clear();
            //: **Reported as one value**, and `NotLookedFor` whenever the switch is off — which is
            //: a different answer from *looked and could not tell*.
            let facing = if !attending() {
                super::Facing::NotLookedFor
            } else if faces_found == 0 {
                super::Facing::NoFace
            } else {
                match argmind_env_face::pointed(&heads) {
                    argmind_env_face::Pointed::AtIt { .. } => super::Facing::AtIt,
                    argmind_env_face::Pointed::Aside { .. } => super::Facing::Aside,
                    argmind_env_face::Pointed::Unreadable => super::Facing::Unreadable,
                }
            };
            let face = faces_found > 0;
            heads.clear();
            unreadable = 0;
            faces_found = 0;
            saw(super::Seen { moved: middle >= MOVING, facing, face, took: so_far,
                              broke: broke.take() });

            //: **On the movement thread rather than beside it**, because two threads on one device
            //: is two lights' worth of nothing — the same reason `switch_camera` refuses to start
            //: a second watcher. The cost is that about 150 ms of every `FACE_EVERY` seconds is
            //: spent not sampling movement; at a 3 s window needing 6 samples that is one or two
            //: out of a window, written down rather than assumed harmless.
        }
        Ok(())
    }

    pub fn neural_voice(host: &str) -> Option<Box<dyn argmind_voice::Speaker>> {
        if !argmind_env_qwentts::QwenVoice::answering(host) {
            return None;
        }
        let chosen = crate::chosen::voice();
        //: **The reference comes from the model itself**, drawn at the preset's seed — which makes
        //: it the same recording on every machine and every run, since a seed fixes one recording
        //: exactly. That is the thing a seed turned out to be good for once it was clear it does
        //: not fix a person.
        //:
        //: **Written here rather than in `speaker.rs`, and a rule said so.** That file holds the
        //: presets and the addresses and deliberately names no environment crate — it is compiled
        //: into the headless build, where there is no synthesiser at all. A first draft put this
        //: beside the presets and both `the_app_names_no_platform` and the `--features headless`
        //: build refused it within a minute of each other.
        let teach = || -> Result<(), String> {
            let reference = argmind_env_qwentts::speak_wav(
                host, crate::speaker::MODEL, crate::speaker::REFERENCE,
                &argmind_env_qwentts::Voice::Drawn(chosen.seed))?;
            argmind_env_qwentts::register_voice(host, chosen.key, &reference,
                                                crate::speaker::REFERENCE)
        };
        let who = match teach() {
            Ok(()) => argmind_env_qwentts::Voice::Cloned(chosen.key.to_string()),
            Err(why) => {
                eprintln!("  voice: {} 를 등록하지 못해 씨앗으로 돌아갑니다 ({why})", chosen.key);
                argmind_env_qwentts::Voice::Drawn(chosen.seed)
            }
        };
        Some(Box::new(argmind_env_qwentts::QwenVoice::new(crate::speaker::MODEL, who)))
    }
}

#[cfg(all(target_os = "linux", feature = "eyes", not(feature = "headless")))]
mod chosen {
    //: **Eyes, and nothing else yet.** `argmind-env-linux` re-exports the rest from
    //: `argmind-env-none`, so the hands are still dry and the voice still silent — see that crate
    //: for why hands on Linux are a permission question before they are a coding one.
    pub use argmind_env_linux::ask::confirm;
    pub use argmind_env_linux::supervise::Supervised;
    pub use argmind_env_linux::WANT_HZ;
    pub use argmind_env_linux::{foreground_handle_now, window_title};
    pub use argmind_env_linux::{Attending as Sight, Deaf as Ear, Hands as Touch, Mute as Mouth,
                                Plain, Watcher as Watching};

    /// **Not yet.** The synthesiser is a server and runs here; nothing plays what it sends back.
    pub fn neural_voice(_host: &str) -> Option<Box<dyn argmind_voice::Speaker>> {
        None
    }

    /// `xdg-open` is the right answer and has not been run from here yet.
    pub fn open_url(_url: &str) -> Result<(), String> {
        Err("이 운영체제에서 페이지를 여는 방법이 아직 확인되지 않았습니다".into())
    }

    pub fn run_elevated(_command: &str) -> Result<bool, String> {
        Err("이 운영체제에서는 관리자 권한 실행이 아직 준비되지 않았습니다".into())
    }


    /// Nothing to do: no other operating system invents a terminal for a child that did not ask.
    pub fn without_a_console(_command: &mut std::process::Command) {}


    /// **Nothing to list.** An environment with no eyes has no cameras either, and refusing is
    /// better than an empty list, which reads as a machine with no hardware.
    pub fn devices(_kind: &str) -> Result<Vec<super::Device>, String> {
        Err("이 운영체제에서는 장치 목록을 읽지 못합니다".into())
    }

    pub fn try_device(kind: &str, _id: Option<&str>) -> super::Tried {
        super::Tried { worked: false,
                       said: format!("이 운영체제에서는 {kind} 를 시험할 수 없습니다") }
    }

    /// Nothing speaks here yet — `speech-dispatcher` exists and is not written.
    /// **No camera, so nothing to watch.** Refusing is not failing — see `argmind-env-none`.
    /// The hands the application uses. **No control patterns here** — that is a Windows notion,
    /// and `argmind-env-none` is the list a third environment must satisfy rather than a place to
    /// put one platform's optimisation.
    pub fn hands(dry: bool) -> Touch {
        let hands = Touch::unrestricted();
        if dry { hands } else { hands.arm() }
    }

    /// **No** — there is no camera here, so nothing would feed a face detector. The models
    /// themselves are portable; what is missing is the thing that would look through.
    pub fn watches_people() -> bool {
        false
    }

    pub fn watch_movement(_want: Option<&str>, _carry_on: &dyn Fn() -> bool,
                          _saw: &mut dyn FnMut(super::Seen),
                          _attending: &dyn Fn() -> bool,
                          _faces: &dyn Fn() -> bool,
                          _met: &mut dyn FnMut(Vec<f32>)) -> Result<(), String> {
        Err("이 운영체제에서는 카메라를 쓸 수 없습니다".into())
    }

    pub fn fallback_voice() -> Option<&'static str> {
        None
    }
}

/// **Where a head is pointed, as the application is allowed to know it.**
///
/// Four states and not an `Option<bool>`. *Nobody asked* and *it could not be read* are different
/// facts, and collapsing them is pitfall 5 in `docs/findings/harness-pitfalls.md` — *a boolean where
/// the delta was needed* — which cost this project a round of work once already. The measurement is
/// in `docs/findings/attention.md`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Facing {
    /// The switch is off, or there are no models. **Nothing was looked for.**
    NotLookedFor,
    /// Looked for, and **no face was in view all window**. Separate from `Unreadable` since
    /// 2026-09-29 — see `Seen::face`.
    NoFace,
    /// Looked for and the head could not be read all window — near profile, where the two eyes
    /// converge. Correlates with turning (0 and 1 of 95 against 22 and 48 of 96) and is not it.
    Unreadable,
    AtIt,
    Aside,
}

/// What one turn of the camera loop found. **Both facts from one tick, in one value**, because they
/// are one tick: reporting them separately would let a caller pair a movement reading with a head
/// reading from three seconds earlier and never notice.
#[derive(Clone, Debug)]
pub struct Seen {
    pub moved: bool,
    pub facing: Facing,
    /// **Whether any face was found this window, apart from where it pointed.** Before 2026-09-29
    /// the two were one value: a window with no face in it and a window with a face whose eyes
    /// had converged both came out `Unreadable`, which is pitfall 5 in this project's own list and
    /// it was in code written the day after that list was quoted at it. `false` whenever no face
    /// pass ran, so a camera with both face switches off reports exactly what it did before.
    pub face: bool,
    /// How long the window actually took, in seconds.
    pub took: f64,
    /// **Something that went wrong and a person should be told**, once. `None` almost always.
    pub broke: Option<String>,
}

/// How often a face pass runs while attention is switched on.
///
/// **One second, which buys about three readings per movement window.** The measurement took its
/// medians over roughly ninety samples a phase; three is not that statistic and a caller must not
/// quote the two as comparable. It is enough to land on the right side of a thirty-fold gap, and
/// what it costs is about a tenth of the loop — at 99 ms a pass against a 32 ms look, three passes
/// take 300 ms of every 3,000, where a window needs six looks to judge movement and has room for
/// ninety.
pub const FACE_PASS: f64 = 1.0;

/// How often the camera loop stops sampling movement and looks for a face.
///
/// **Three seconds, matching the movement window**, so the two land on the same beat rather than
/// drifting past each other. It is a person at a desk: they do not become somebody else between
/// ticks, and a faster rate would buy nothing and cost a sample.
pub const FACE_EVERY: f64 = 3.0;

/// What the detector is fed. YuNet letterboxes to 640 square, so this leaves the picture at scale
/// 1.0 and invents no pixel.
pub const FACE_WIDE: usize = 640;
pub const FACE_TALL: usize = 360;

/// **Below this a detection is not a face.** OpenCV's own default for YuNet. Nothing here has
/// measured a better one, and a number chosen to make something pass would be the failure this
/// project's standards exist against.
pub const A_FACE: f32 = 0.6;

//: Everything else: no environment has been written for it, or this build was asked not to
//: use the one there is.
#[cfg(any(feature = "headless",
          all(not(windows),
              any(not(target_os = "linux"), not(feature = "eyes")))))]
mod chosen {
    pub use argmind_env_none::ask::confirm;
    pub use argmind_env_none::supervise::Supervised;
    pub use argmind_env_none::WANT_HZ;
    pub use argmind_env_none::{foreground_handle_now, window_title};
    pub use argmind_env_none::{Attending as Sight, Deaf as Ear, Mute as Mouth, Plain,
                               Still as Touch, Watcher as Watching};

    /// **Never.** The synthesiser may well be answering on its port — it is a server and it builds
    /// on all three — but nothing here can play what it sends back. Saying no is the honest
    /// answer; asking the socket and then being unable to make a sound would not be.


    /// Nothing to do: no other operating system invents a terminal for a child that did not ask.
    pub fn without_a_console(_command: &mut std::process::Command) {}


    /// **Nothing to list.** An environment with no eyes has no cameras either, and refusing is
    /// better than an empty list, which reads as a machine with no hardware.
    pub fn devices(_kind: &str) -> Result<Vec<super::Device>, String> {
        Err("이 운영체제에서는 장치 목록을 읽지 못합니다".into())
    }

    pub fn try_device(kind: &str, _id: Option<&str>) -> super::Tried {
        super::Tried { worked: false,
                       said: format!("이 운영체제에서는 {kind} 를 시험할 수 없습니다") }
    }

    /// **Nothing speaks here.** `speech-dispatcher` on Linux and `AVSpeechSynthesizer` on macOS
    /// both exist; neither is written. Saying *Windows 음성으로 말합니다* on Ubuntu, which is what
    /// this replaced, is worse than saying nothing does.
    /// **No camera, so nothing to watch.** Refusing is not failing — see `argmind-env-none`.
    /// The hands the application uses. **No control patterns here** — that is a Windows notion,
    /// and `argmind-env-none` is the list a third environment must satisfy rather than a place to
    /// put one platform's optimisation.
    pub fn hands(dry: bool) -> Touch {
        let hands = Touch::unrestricted();
        if dry { hands } else { hands.arm() }
    }

    /// **No** — there is no camera here, so nothing would feed a face detector. The models
    /// themselves are portable; what is missing is the thing that would look through.
    pub fn watches_people() -> bool {
        false
    }

    pub fn watch_movement(_want: Option<&str>, _carry_on: &dyn Fn() -> bool,
                          _saw: &mut dyn FnMut(super::Seen),
                          _attending: &dyn Fn() -> bool,
                          _faces: &dyn Fn() -> bool,
                          _met: &mut dyn FnMut(Vec<f32>)) -> Result<(), String> {
        Err("이 운영체제에서는 카메라를 쓸 수 없습니다".into())
    }

    pub fn fallback_voice() -> Option<&'static str> {
        None
    }

    /// **Refused rather than guessed.** `xdg-open` and `open` exist and would probably work; that
    /// is not the same as having been run, and nothing here offers a fix that would reach this.
    pub fn open_url(_url: &str) -> Result<(), String> {
        Err("이 운영체제에서 페이지를 여는 방법이 아직 확인되지 않았습니다".into())
    }

    /// There is no tested way to ask this operating system for administrator rights.
    pub fn run_elevated(_command: &str) -> Result<bool, String> {
        Err("이 운영체제에서는 관리자 권한 실행이 아직 준비되지 않았습니다".into())
    }

    pub fn neural_voice(_host: &str) -> Option<Box<dyn argmind_voice::Speaker>> {
        None
    }
}

pub use chosen::*;


/// One thing that can record, play or see, as the page needs to show it.
///
/// **Not the environment's own type**, because there are three environments and only one of them
/// has a `devices` module. A name, whatever opens it, and whether the operating system would have
/// picked it anyway.
#[derive(Debug, Clone, PartialEq)]
pub struct Device {
    pub id: String,
    pub name: String,
    pub default: bool,
}

/// What a test of one device found.
///
/// **A number and a sentence**, because *it worked* is not a measurement and a number on its own
/// is not an answer.
pub struct Tried {
    pub worked: bool,
    pub said: String,
}

/// **The only way this crate starts a child process.**
///
/// Everything argmind shells out to comes through here — `git`, `cmake`, `objdump`, `curl.exe`,
/// `certutil`, `reg`, `where`, an installer, five `--version` calls every time the page asks what
/// is missing — and every one of them used to open a window. A rule in `argmind-rules` keeps
/// `Command::new` out of the rest of the crate, because *remembering* to be quiet is the same kind
/// of promise as remembering to say `platform::Sight`.
///
/// `stdin` is closed here too. A child that waits on input nobody can give is a build that has
/// hung, and it looks exactly like a build that is slow.
pub fn child(program: impl AsRef<std::ffi::OsStr>) -> std::process::Command {
    let mut command = std::process::Command::new(program);
    command.stdin(std::process::Stdio::null());
    without_a_console(&mut command);
    command
}

/// What this build can actually do, in one line, for a person reading a log or a page.
///
/// **Said out loud rather than discovered.** An agent that sees nothing and presses nothing looks
/// broken; an agent that says *I have no eyes on this operating system* is a program somebody can
/// reason about.
pub fn describe() -> &'static str {
    if cfg!(all(windows, not(feature = "headless"))) {
        "windows"
    } else if cfg!(windows) {
        //: Windows underneath, deliberately blinded — this is the measurement build.
        "headless (on windows)"
    } else {
        "headless — no environment for this operating system yet"
    }
}

/// **Minutes to add to UTC for the date on this machine's wall**, for an errand that is told
/// today rather than left to guess it.
///
/// Off Windows this is 0 — the UTC date, a day out for a few hours either side of midnight — and
/// that is said rather than fixed, because the only other source written so far would be parsing
/// `date +%z`, and nothing here has run on a Linux desktop long enough to know it is right.
pub fn utc_offset_minutes() -> i64 {
    #[cfg(windows)]
    {
        argmind_env_windows::clock::utc_offset_minutes()
    }
    #[cfg(not(windows))]
    {
        0
    }
}

/// **Seconds since the last keyboard or mouse input**, or `None` where nothing can say. The time
/// only — see `argmind_env_windows::seconds_since_input`.
pub fn seconds_since_input() -> Option<f64> {
    #[cfg(all(windows, not(feature = "headless")))]
    {
        argmind_env_windows::seconds_since_input()
    }
    #[cfg(not(all(windows, not(feature = "headless"))))]
    {
        None
    }
}

/// **The folder a task works in** — `%USERPROFILE%\argmind-work`, made if it is not there.
pub fn work_folder() -> std::path::PathBuf {
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))
        .map(std::path::PathBuf::from).unwrap_or_else(std::env::temp_dir);
    home.join("argmind-work")
}

/// **The workbench a task works at**, or `None` where nothing has been written: files in
/// [`work_folder`], commands in the `argmind-sandbox` WSL distro, PowerShell for what only runs on
/// Windows. See `argmind_env_windows::workspace`.
pub fn workbench() -> Option<Box<dyn argmind_core::task::Tools + Send>> {
    #[cfg(all(windows, not(feature = "headless")))]
    {
        Some(Box::new(argmind_env_windows::workspace::Workspace::new(work_folder())))
    }
    #[cfg(not(all(windows, not(feature = "headless"))))]
    {
        None
    }
}

/// **Programs this machine has, by their own scripting interfaces** — Blender's Python and
/// `kicad-cli` — ranked with the APIs. `None` when neither is installed. See
/// `argmind_env_windows::programs`.
pub fn installed_programs() -> Option<Box<dyn argmind_core::task::Tools + Send>> {
    #[cfg(all(windows, not(feature = "headless")))]
    {
        argmind_env_windows::programs::Programs::found(work_folder())
            .map(|p| Box::new(p) as Box<dyn argmind_core::task::Tools + Send>)
    }
    #[cfg(not(all(windows, not(feature = "headless"))))]
    {
        None
    }
}

/// **A screen of argmind's own for work**, ranked last — a Windows desktop the person never sees,
/// operated by control patterns only. See `argmind_env_windows::screen`.
/// Every press and write there is put to `gate` first — the person's 규칙, from `app.rs`.
pub fn work_screen(gate: Box<dyn FnMut(&str, &str, &str) -> Result<(), String> + Send>)
    -> Option<Box<dyn argmind_core::task::Tools + Send>> {
    #[cfg(all(windows, not(feature = "headless")))]
    {
        Some(Box::new(argmind_env_windows::screen::Screen::shown(work_view().clone()).gated(gate)))
    }
    #[cfg(not(all(windows, not(feature = "headless"))))]
    {
        let _ = gate;
        None
    }
}

/// The one picture every work screen draws into — one at a time, since one task runs at a time.
#[cfg(all(windows, not(feature = "headless")))]
fn work_view() -> &'static argmind_env_windows::screen::View {
    static VIEW: std::sync::OnceLock<argmind_env_windows::screen::View> = std::sync::OnceLock::new();
    VIEW.get_or_init(Default::default)
}

/// What the work screen looks like, **for the person and never for a model**: `None` when nothing
/// is open there, `Some((seq, None))` when the page already has picture `seen`, and otherwise the
/// picture's number, the windows' titles and a `.bmp`. `argmind-rules` holds that one route reads it.
pub fn work_picture(seen: u64) -> Option<(u64, Option<(String, Vec<u8>)>)> {
    #[cfg(all(windows, not(feature = "headless")))]
    {
        let v = work_view().lock().ok()?;
        let p = v.as_ref()?;
        Some((p.seq, (p.seq != seen).then(|| (p.titles.join(" · "), p.bmp.clone()))))
    }
    #[cfg(not(all(windows, not(feature = "headless"))))]
    {
        let _ = seen;
        None
    }
}

/// Today, as an errand is told it.
pub fn today() -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64).unwrap_or(0);
    argmind_core::errand::today(now, utc_offset_minutes())
}

/// Can this build see the screen and press things at all?
pub fn embodied() -> bool {
    cfg!(all(windows, not(feature = "headless")))
}

/// Can this build actually carry out the things it offers to do?
///
/// **Every fix argmind offers has been run on Windows and none of them anywhere else.** The
/// installers are `.exe`, the elevation is `powershell -Verb RunAs`, the hash check is `certutil`,
/// the builds want `mingw32-make` and `objdump`, and the Vulkan SDK arrives through `winget`.
///
/// So off Windows the needs are **reported and not offered** — which is a state `needs.rs` already
/// has a word for, *argmind가 할 수 있는 게 없습니다*, and which is the honest one. Running on
/// Linux is what made this urgent: the page was offering
/// `DISM /Online /Add-Capability …` and saying *Windows가 한 번 더 물어봅니다*, on Ubuntu.
///
/// **This is a statement about argmind, not about the operating system.** Ollama installs on Linux
/// and macOS, whisper.cpp builds there, and a Vulkan SDK is a package away. None of it is written
/// here yet, and a button that has never been run is worse than a sentence saying what to do.
pub fn can_carry_out_fixes() -> bool {
    cfg!(all(windows, not(feature = "headless")))
}

#[cfg(all(test, windows, not(feature = "headless")))]
mod tests {
    /// **The work screen's picture reaches the route's function** — a program opened on the work
    /// screen the way a task opens one, `work_picture` handing out a `.bmp`, the same number
    /// answered with nothing new, and nothing at all once it is closed. Ignored because it starts
    /// a program (`winver`, on argmind's own desktop): `cargo test -p argmind-serve --lib
    /// work_screen -- --ignored`. **A refusal fails rather than skips** — the first version
    /// skipped when Character Map was already open, and passed in 0.02 s having checked nothing.
    #[test]
    #[ignore]
    fn the_work_screen_is_drawn_for_the_page_and_gone_when_closed() {
        let mut s = super::work_screen(Box::new(|_: &str, _: &str, _: &str| Ok(()))).unwrap();
        s.call("screen_open", &serde_json::json!({"program": "winver.exe"})).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(1500));
        let (seq, picture) = super::work_picture(0).expect("a picture while a program is open");
        let (titles, bmp) = picture.expect("a picture the page does not have");
        assert!(bmp.starts_with(b"BM") && bmp.len() > 54 + 100 * 100 * 3, "{} bytes", bmp.len());
        assert!(!titles.is_empty());
        assert!(super::work_picture(seq).is_some_and(|(_, p)| p.is_none()), "the same number got the picture again");
        s.call("screen_close", &serde_json::json!({})).unwrap();
        assert!(super::work_picture(0).is_none(), "a closed work screen still has a picture");
    }
}
