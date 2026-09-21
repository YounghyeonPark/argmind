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

    /// **The seed is read here rather than passed in**, and that is deliberate: this is the one
    /// place a neural voice is built, so *which voice* is decided in one place too. A caller that
    /// could pass a different one is a caller that could pass none.
    pub fn neural_voice(host: &str) -> Option<Box<dyn argmind_voice::Speaker>> {
        argmind_env_qwentts::QwenVoice::answering(host)
            .then(|| Box::new(argmind_env_qwentts::QwenVoice::new(crate::speaker::MODEL,
                                                                 crate::chosen::voice()))
                as Box<dyn argmind_voice::Speaker>)
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
    pub fn fallback_voice() -> Option<&'static str> {
        None
    }
}

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
