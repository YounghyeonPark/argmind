//! Getting the neural voice onto this machine, from source, and starting it.
//!
//! The same shape as [`crate::recogniser`] and for the same reason: `qwentts.cpp` publishes source,
//! not an installer, so the supported way to have it is to build it. The machinery is shared —
//! [`crate::building`] — and what is here is what is true of the voice and of nothing else.
//!
//! ## Why this exists at all, when Windows already speaks Korean
//!
//! It speaks it with `Microsoft Heami Desktop`, a Windows 7-era concatenative voice, and the
//! owner's verdict on 2026-09-16 was 어색함. `docs/findings/a-better-voice.md` is the accounting:
//! the licence-clean alternatives, the two that were refused for a non-commercial model card, and
//! this one — MIT runtime, Apache-2.0 weights, C++ over the same GGML whisper.cpp already vendors.
//! **No Python and no new crate**, which is the owner's rule of that day.
//!
//! ## The GPU is not optional, and that is measured
//!
//! **RTF 0.15 on Vulkan against 4.8 on the CPU** — thirty-two times, same binary, same weights,
//! isolated with `GGML_VK_VISIBLE_DEVICES=""`. A three-second reply costs 0.45 s of synthesis on
//! the GPU and about fourteen seconds on the processor, and fourteen seconds is not a conversation.
//! So the Vulkan SDK is a requirement rather than a preference, and it is asked for separately —
//! see [`sdk`] and the `vulkan` need.
//!
//! ## Q8_0 for the talker, and it is not a quality decision
//!
//! **`ggml-vulkan`'s K-quant path gives a silently wrong answer here.** With `Q4_K_M` the talker
//! never emits its end token: a forty-character sentence decoded to 2,048 frames, which is 164
//! seconds of audio, at a flattering RTF of 0.123. The same binary with the same weights on the CPU
//! stops at step 33. Q8_0 on the GPU is correct and is what is pinned.
//!
//! The codec stays at `Q4_K_M` because that is the configuration that was measured end to end, and
//! a working configuration is not swapped for an untested one on a theory — **a Q8_0 codec is
//! published and has not been tried.** The defect was in the talker's sampling loop, which is a
//! different graph, and every run of this pairing produced correct speech. That is an observation,
//! not a proof.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::platform::Supervised;

use crate::building::{self, Recipe};
use crate::fix::Fix;

/// Where the source comes from, and exactly which commit.
pub const REPO: &str = "https://github.com/ServeurpersoCom/qwentts.cpp.git";
/// **A SHA rather than a tag**, for the reason [`crate::recogniser`] gives: a git object name is
/// the content, so asking for this and getting anything else is not something git can be talked
/// into.
pub const COMMIT: &str = "90efdc60d62f7d6c70113994d0be17e820fd8673";
/// The cmake target. One binary out of the five this tree can produce.
pub const TARGET: &str = "tts-server";
/// What a build needs on the machine before it can start. **The Vulkan SDK is not in this list**
/// because it cannot be found this way — see [`sdk`].
pub const TOOLS: [&str; 5] = ["git", "cmake", "g++", "mingw32-make", "objdump"];

/// Where the synthesiser listens.
///
/// **Named here rather than reached through the client crate**, which is Windows-only in this
/// tree — it owns a WASAPI loudspeaker. The address is not: it is a port number, and the server
/// that answers on it builds on all three operating systems.
pub const QWENTTS_HOST: &str = "127.0.0.1:8179";

/// The label the server reports at `GET /v1/models`, and what the client asks for.
///
/// **The server does not check it** — it is a name for the one pair of weights that are loaded, not
/// a selector. Passed anyway, so the two ends say the same word and `/v1/models` is readable.
///
/// **Called `MODEL` rather than `VOICE`, since 2026-09-21.** It was named for the thing a person
/// hears, and then a real voice setting arrived — [`crate::chosen::VOICE`], the sampler seed that
/// actually decides who is speaking. Two constants called `VOICE`, one of which has nothing to do
/// with the voice, is the sort of thing that gets the wrong one passed.
pub const MODEL: &str = "qwen3-tts";

// ---- which voice ---------------------------------------------------------------------------------

/// The sentence a reference recording says.
///
/// **Long enough to carry a person and short enough to cost half a second.** The extractor is given
/// one utterance and has to find a speaker in it; measured at 0.53 s to synthesise and 0.16 s to
/// extract, with clones that track their reference's pitch at +0.99.
pub const REFERENCE: &str = "안녕하세요. 만나서 반갑습니다. 오늘은 무엇을 도와드릴까요?";

/// One voice argmind can speak in.
///
/// **A preset is a seed plus what was measured about it**, not a name somebody made up. Every
/// number here is a reading, and they are in the struct so the page can show them: a person
/// choosing a voice is entitled to see what the ordering rests on, and to catch it being wrong.
pub struct Preset {
    /// What goes in `devices.json` and on the wire as the registered voice's name.
    pub key: &'static str,
    /// What the page calls it.
    ///
    /// **Numbered rather than described, and that is the finding rather than laziness.** Three of
    /// the five sit within 22 Hz of each other and are told apart by timbre, not by pitch — see
    /// `apart` — so calling one 낮은 and the next 조금 낮은 would be a distinction the measurement
    /// does not support. The Hz goes beside the name in the page and carries what is real.
    pub what: &'static str,
    /// The seed the reference is drawn at. Reproducible: a seed and a text give the same bytes
    /// every time, which is the one thing a seed turned out to be good for.
    pub seed: i64,
    /// Median f0 of the clone over six sentences, in Hz. Measured 2026-09-21.
    pub hz: u32,
    /// How far that moved between the six.
    ///
    /// **Bigger than it first looked, and the first look is why this is n=6.** Over three sentences
    /// these read 4–19 Hz and the same seeds read 16–33 over six: a max-minus-min of three samples
    /// underestimates a range, and a preset list chosen on the small numbers had two pairs in it
    /// that are the same person.
    pub wobble: u32,
    /// **How far this voice is from its nearest neighbour, as a speaker**, in thousandths.
    ///
    /// Long-term average spectra, which have no notion of pitch in them: the mean correlation
    /// between this voice's renditions and another's, subtracted from the mean within each. A clone
    /// is 0.955–0.982 like itself across sentences, so anything near zero here is two names for one
    /// person. This is the number that chose the list — of seven candidates that passed a
    /// pitch-gap rule, two pairs measured **+0.016** and **−0.004** and were dropped.
    pub apart: u32,
}

/// Five, and the list is what survived being measured rather than what was drawn.
///
/// Seventy-two seeds were cloned and measured. A pitch-gap rule — neighbours further apart than
/// they wobble — cut that to seven, and then the wobble turned out to be an n=3 artefact and the
/// rule turned out to be the wrong question: **two voices can share a pitch and still be two
/// people, and two voices can differ in pitch and be one.** Asking the right question, whether the
/// spectra separate, dropped two of the seven and kept these.
///
/// The weakest pair here is +0.042 (the two lowest), against 0.979 within either. Whether +0.042 is
/// *audibly* two people is not something a correlation can say — which is why choosing one speaks.
pub const PRESETS: [Preset; 5] = [
    Preset { key: "one",   what: "목소리 1", seed: 20, hz: 105, wobble: 21, apart: 42 },
    Preset { key: "two",   what: "목소리 2", seed: 35, hz: 119, wobble: 16, apart: 42 },
    Preset { key: "three", what: "목소리 3", seed: 59, hz: 127, wobble: 21, apart: 68 },
    Preset { key: "four",  what: "목소리 4", seed: 56, hz: 177, wobble: 27, apart: 46 },
    Preset { key: "five",  what: "목소리 5", seed: 64, hz: 209, wobble: 33, apart: 46 },
];

/// Below this, two presets are one voice under two names.
///
/// **Not a threshold anybody derived** — it is the weakest pair that was kept, written down so the
/// list cannot quietly grow a pair worse than the ones already rejected at +0.016 and −0.004.
pub const LEAST_APART: u32 = 40;

/// The one used when nobody has chosen.
///
/// **Arbitrary among five, and that is the whole of its justification** — what matters is that it
/// is the same one on every run. The middle of the range rather than an end, so a first impression
/// is not of an extreme.
pub const A_VOICE: &str = "four";

pub fn preset(key: &str) -> Option<&'static Preset> {
    PRESETS.iter().find(|p| p.key == key)
}

/// The preset for a remembered key, falling back rather than failing.
///
/// **A key nobody knows is the default, not an error.** `devices.json` is hand-editable, and until
/// 2026-09-21 this setting held a *seed* — a bare number — so every file written by the version
/// before this one lands here. That is the migration: an old seed is not a preset, and the honest
/// thing is the default voice rather than a guess at which preset the number was nearest.
pub fn preset_or_default(key: &str) -> &'static Preset {
    preset(key).or_else(|| preset(A_VOICE)).expect("A_VOICE names a preset")
}

/// The talker: the model that turns text into codec frames.
pub const TALKER_URL: &str = "https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/\
                              resolve/main/qwen-talker-0.6b-base-Q8_0.gguf";
pub const TALKER_SHA256: &str =
    "d54dbaf10591421fa764ed630d764efa717ae40cd959bd48c66d4eb1af226426";
/// **Measured on the file that produced every number in the findings.**
pub const TALKER_BYTES: u64 = 992_615_488;
/// **Derived rather than written down**, because the first version had the size as its own number
/// and a test that added two of them together — which is true whatever the numbers are. A size a
/// person is shown before a download should come from the file.
pub const TALKER_MB: u64 = TALKER_BYTES / 1_048_576;

/// The codec: the model that turns those frames into audio. Shared by every talker.
pub const CODEC_URL: &str = "https://huggingface.co/Serveurperso/Qwen3-TTS-GGUF/\
                             resolve/main/qwen-tokenizer-12hz-Q4_K_M.gguf";
pub const CODEC_SHA256: &str =
    "cf3788b4d50aaa665fb6e57c170396aae03a3555fea52d2b5d0cda902d658039";
/// **Measured.**
pub const CODEC_BYTES: u64 = 254_974_752;
pub const CODEC_MB: u64 = CODEC_BYTES / 1_048_576;

/// How long to say it might take, as an **upper bound rather than an estimate**.
///
/// **Measured once**, on the machine this was written on, at the path it will actually use: a
/// fresh clone with submodules and a full build at `-j 8` took **7 minutes**, and that figure is
/// for every target rather than the one this asks for. Fifteen is the number a person is told
/// before they commit an evening, and it is deliberately not the measured one — the machines this
/// will run on are not this one.
pub const MINUTES: u64 = 15;

/// Where the source tree and the build live.
///
/// **`b` rather than `build`, and that is not a style choice.** `ggml-vulkan` generates its shaders
/// by configuring and compiling a helper inside its own build tree, and cmake puts **155
/// characters** below the build folder doing it — measured, counting the try-compile scratch path
/// that is deleted afterwards. Against Windows' 260-character limit this build failed from a
/// 126-character directory with *"The C compiler is not able to compile a simple test program"*,
/// which reads as a broken toolchain and is not one. Every character saved here is a character
/// that buys.
pub fn shed() -> PathBuf {
    building::shed("qwentts.cpp")
}

/// **Not `build/bin`.** This tree puts its binaries at the top of the build folder.
pub fn server_exe() -> PathBuf {
    shed().join("b").join("tts-server.exe")
}

pub fn built_at() -> Option<String> {
    building::built_at(&server_exe())
}

pub fn ready() -> bool {
    building::ready(&server_exe(), COMMIT)
}

pub fn remember_build(commit: &str) {
    building::remember_build(&server_exe(), commit);
}

pub fn talker_at() -> PathBuf {
    crate::needs::model_shed().join("qwen-talker-0.6b-base-Q8_0.gguf")
}

/// Both weights on disk.
///
/// **Part of *already built*, and not the same question as [`ready`].** The binary and the commit
/// stamp say the tree was compiled; these say there is anything for it to load. A gigabyte deleted
/// to make room is the ordinary way the two come apart, and the honest answer to that is to fetch
/// it again rather than to start a server that will fail.
pub fn weights_here() -> bool {
    talker_at().is_file() && codec_at().is_file()
}

pub fn codec_at() -> PathBuf {
    crate::needs::model_shed().join("qwen-tokenizer-12hz-Q4_K_M.gguf")
}

/// The two fixes that would fetch the weights on their own.
///
/// [`crate::fix::apply`] runs *these* for the weights rather than a second copy of the same
/// download, so what the person agreed to and what happens cannot come apart.
pub fn weight_fixes() -> [Fix; 2] {
    [
        Fix::Fetch {
            url: TALKER_URL.into(),
            into: talker_at().to_string_lossy().to_string(),
            why: "목소리 모델입니다.".into(),
            about_mb: TALKER_MB,
            sha256: Some(TALKER_SHA256.into()),
        },
        Fix::Fetch {
            url: CODEC_URL.into(),
            into: codec_at().to_string_lossy().to_string(),
            why: "소리로 바꾸는 모델입니다.".into(),
            about_mb: CODEC_MB,
            sha256: Some(CODEC_SHA256.into()),
        },
    ]
}

/// Both weights together, for the sentence that has to name a size.
pub const WEIGHTS_MB: u64 = TALKER_MB + CODEC_MB;

/// What a build needs and this machine has not got.
pub fn missing_tools() -> Vec<&'static str> {
    building::missing(&TOOLS)
}

/// Where the Vulkan SDK is, if it is anywhere.
///
/// **Not `glslc --version`, and the difference matters.** The SDK puts its tools on the machine's
/// PATH, and a process that started before the SDK was installed has the PATH it was born with — so
/// the check that looks like the obvious one reports *not installed* on a machine where it is,
/// in exactly the case this program creates: argmind asks for it, the person installs it, and
/// argmind is still the same process. [`building::machine_env`] reads the variable the installer
/// wrote rather than the one this process inherited.
///
/// The directory is then checked on disk, because a variable left behind by an uninstall is a
/// variable that points at nothing.
pub fn sdk() -> Option<PathBuf> {
    let at = PathBuf::from(building::machine_env("VULKAN_SDK")?);
    at.join("Bin").join("glslc.exe").is_file().then_some(at)
}

/// What argmind offers to do about a missing SDK.
///
/// **`--interactive` on purpose.** The one attempt at a headless install on this machine ended in
/// `Installation aborted!`; letting the installer put its own window up is the path with evidence
/// behind it, and `Start-Process -Wait` waits for the person either way.
///
/// **Elevated, because it is a machine-wide install** that sets a machine-wide variable — so
/// Windows asks a second time after argmind's window has, which is the whole point of the variant.
pub fn sdk_fix() -> Fix {
    Fix::RunElevated {
        command: "winget install --id KhronosGroup.VulkanSDK -e --interactive \
                  --accept-package-agreements --accept-source-agreements".into(),
        why: "그래픽카드로 말하려면 Vulkan SDK가 필요합니다 — 없으면 32배 느려서 대화가 안 됩니다."
            .into(),
    }
}

/// Put one commit of qwentts.cpp in the shed.
///
/// **With submodules**, unlike the recogniser: this tree links ggml rather than vendoring it as
/// files, and without the second fetch cmake stops on a directory that is there and empty.
pub fn clone_at(repo: &str, commit: &str, into: &Path, say: &mut dyn FnMut(&str))
                -> Result<(), String> {
    building::clone_at(repo, commit, into, true, say)
}

/// Configure and compile.
///
/// Three things here are not the defaults, and each of them was a failure first:
///
/// * **`GGML_VULKAN=ON`** — the measured reason this is worth building at all.
/// * **`_WIN32_WINNT=0x0A00`** — `cpp-httplib` gates `CreateFile2` behind Windows 8 and MinGW's
///   default target is older than that, so the server fails to compile with
///   `'::CreateFile2' has not been declared`. Both languages, because the tree has C in it too.
/// * **The SDK's `Bin` in front of the PATH** — shader generation runs a program this build just
///   produced, and it has to find the SDK's tools and the toolchain's libraries rather than
///   whatever else on the machine has those names. See [`crate::building`].
pub fn build(at: &Path, target: &str, say: &mut dyn FnMut(&str)) -> Result<(), String> {
    let Some(vulkan) = sdk() else {
        return Err("Vulkan SDK를 찾지 못했습니다".into());
    };
    Recipe {
        at: at.to_path_buf(),
        into: "b".into(),
        target: target.into(),
        defines: vec![
            "-DGGML_VULKAN=ON".into(),
            "-DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00".into(),
            "-DCMAKE_C_FLAGS=-D_WIN32_WINNT=0x0A00".into(),
        ],
        //: Set for the child rather than assumed of the parent — this process may predate the
        //: install. cmake's Vulkan finder reads this variable.
        env: vec![("VULKAN_SDK".into(), vulkan.to_string_lossy().to_string())],
        path_first: vec![vulkan.join("Bin")],
    }.run(say)
}

/// The one voice argmind started, held for as long as argmind runs.
///
/// **Its cleanup is the job object's rather than this value's** — a static is never dropped, which
/// is precisely why the guarantee was put in the operating system instead.
static RUNNING: Mutex<Option<Supervised>> = Mutex::new(None);

/// Is a synthesiser answering on the port?
///
/// **A connect and nothing else** — it is not asked to synthesise anything, because a check that
/// made the thing it is checking do work is a check with a side effect. Written here rather than
/// borrowed from the client crate, which is Windows-only in this tree because it owns a WASAPI
/// loudspeaker; a TCP connect is not.
pub fn answering() -> bool {
    QWENTTS_HOST.parse().is_ok_and(|at| {
        std::net::TcpStream::connect_timeout(&at, Duration::from_millis(400)).is_ok()
    })
}

/// Did *argmind* start the one that is running?
pub fn ours() -> bool {
    RUNNING.lock().map(|mut it| it.as_mut().is_some_and(Supervised::alive)).unwrap_or(false)
}

/// Start the built server and wait until it answers.
///
/// **Waiting is the point.** A gigabyte of weights goes to the GPU before the socket opens, and
/// reporting success on a process that has started but is not listening produces a first attempt at
/// speaking that fails for no visible reason.
pub fn serve(say: &mut dyn FnMut(&str)) -> Result<(), String> {
    let exe = server_exe();
    if !exe.is_file() {
        return Err(format!("{} 이(가) 없습니다", exe.display()));
    }
    for weight in [talker_at(), codec_at()] {
        if !weight.is_file() {
            return Err(format!("{} 이(가) 없습니다", weight.display()));
        }
    }
    let (host, port) = QWENTTS_HOST.rsplit_once(':')
        .ok_or("주소가 이상합니다")?;

    let log = shed().join("server.log");
    let talker = talker_at().to_string_lossy().to_string();
    let codec = codec_at().to_string_lossy().to_string();
    say(&format!("목소리 서버를 {} 에 띄웁니다", QWENTTS_HOST));
    let mut started = Supervised::start(
        &exe,
        //: **`--lang auto`, which is the default, and it is chosen rather than inherited.** The
        //: model infers the language from the text when no id is given, and this agent answers in
        //: Korean 30 times in 36 and in English the rest — a pinned `korean` would read the English
        //: ones in Korean. An unknown name is fatal at startup, which is the other reason not to
        //: pass one.
        &["--model", &talker, "--codec", &codec,
          "--host", host, "--port", port, "--alias", MODEL],
        Some(&shed()),
        Some(&log),
    ).map_err(|e| format!("서버를 시작하지 못했습니다 ({e})"))?;

    let by = Instant::now() + Duration::from_secs(120);
    while Instant::now() < by {
        if answering() {
            let promise = if started.supervised() {
                "argmind를 닫으면 함께 꺼집니다"
            } else {
                "argmind가 정상적으로 끝나면 함께 꺼집니다"
            };
            say(&format!("서버가 응답합니다 — {promise}"));
            *RUNNING.lock().map_err(|_| "잠금 실패")? = Some(started);
            return Ok(());
        }
        if !started.alive() {
            let why = std::fs::read_to_string(&log).unwrap_or_default();
            let all: Vec<&str> = why.lines().collect();
            return Err(format!("서버가 바로 꺼졌습니다 —\n{}",
                               building::last_few(&all, 8).iter().map(|l| format!("    {l}"))
                                   .collect::<Vec<_>>().join("\n")));
        }
        std::thread::sleep(Duration::from_millis(300));
    }
    Err(format!("120초 안에 응답하지 않았습니다 — {} 를 보세요", log.display()))
}
