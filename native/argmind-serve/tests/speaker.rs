//! The voice's build, **without building anything.**
//!
//! What a suite can check here is the half that is decided before a person is asked: which commit,
//! which weights, what the window says it will do, and whether the offer is withheld when it cannot
//! be honoured. The other half is a compiler and a gigabyte of downloads, and it is exercised by
//! hand — the run behind `MINUTES` is in `docs/findings/a-better-voice.md`.
//!
//! **Two of these are about something that has already gone wrong once.** The build path has to be
//! short, and the Vulkan SDK has to be found somewhere other than this process's environment.
//!
//! Needs no network, no model, no GPU and no speaker.

use argmind_serve::fix::{describe, Fix};
use argmind_serve::{building, speaker};

/// The commit is a commit, and it is asked for by name rather than by tag.
#[test]
fn the_source_is_pinned_to_one_object_name() {
    assert_eq!(speaker::COMMIT.len(), 40, "a SHA-1 object name is forty hex characters");
    assert!(speaker::COMMIT.chars().all(|c| c.is_ascii_hexdigit()));
    assert!(speaker::REPO.starts_with("https://"), "{}", speaker::REPO);
    assert!(speaker::REPO.ends_with(".git"));
}

/// **Both weights are hash-checked, and neither is executed.**
///
/// The commit protects the program; nothing protects a `.gguf` except the hash, because a model
/// file has no name that is its content.
#[test]
fn the_weights_are_fetched_with_a_hash_and_are_not_a_stranger() {
    let fixes = speaker::weight_fixes();
    assert_eq!(fixes.len(), 2, "a talker and a codec");
    let mut total = 0;
    for weights in &fixes {
        let Fix::Fetch { url, sha256, about_mb, into, .. } = weights else {
            panic!("weights are data and must be a Fetch: {weights:?}");
        };
        assert!(url.starts_with("https://huggingface.co/"), "{url}");
        assert_eq!(sha256.as_deref().map(str::len), Some(64), "{url}");
        assert!(sha256.as_deref().is_some_and(|h| h.chars().all(|c| c.is_ascii_hexdigit())));
        //: **The name it is saved as is the name it was fetched as.** They are two literals and
        //: nothing but this ties them together; crossed over, argmind downloads the talker into
        //: the file the codec is loaded from and the only thing that notices is a hash check,
        //: after a gigabyte has been fetched.
        let published = url.rsplit('/').next().expect("a file name");
        assert!(into.ends_with(published), "{url} would be saved as {into}");
        assert!(!weights.runs_a_stranger(), "a weights file is data");
        total += about_mb;
    }
    assert_eq!(total, speaker::WEIGHTS_MB,
               "the sentence names a total the two downloads do not add up to");

    //: **And the size is the file's, when the file is here.** The sum above is arithmetic and
    //: would hold for any two numbers; this is the only check that can tell a URL pointing at a
    //: different quantisation from one pointing at the right one. Skipped rather than failed on a
    //: machine that has not fetched them — the suite needs no network.
    for (at, bytes) in [(speaker::talker_at(), speaker::TALKER_BYTES),
                        (speaker::codec_at(), speaker::CODEC_BYTES)] {
        let Ok(found) = std::fs::metadata(&at) else { continue };
        assert_eq!(found.len(), bytes,
                   "{} is {} bytes and the record says {bytes}", at.display(), found.len());
    }
}

/// **The talker is Q8_0 and the codec is Q4_K_M**, and that pairing is not an accident.
///
/// `ggml-vulkan`'s K-quant path made the talker run past its end token — 2,048 frames for a
/// forty-character sentence. A future edit that "tidies up" the quantisations to match would
/// reintroduce a failure whose symptom is a voice that never stops talking.
#[test]
fn the_quantisations_are_the_ones_that_were_measured() {
    assert!(speaker::TALKER_URL.ends_with("-Q8_0.gguf"), "{}", speaker::TALKER_URL);
    assert!(speaker::CODEC_URL.ends_with("-Q4_K_M.gguf"), "{}", speaker::CODEC_URL);
    assert!(speaker::talker_at().to_string_lossy().contains("Q8_0"));
    assert!(speaker::codec_at().to_string_lossy().contains("Q4_K_M"));
}

/// **The build path leaves room for cmake's own.**
///
/// The failure this guards is not a compile error; it is *"The C compiler is not able to compile a
/// simple test program"* from a working toolchain, and it took an afternoon to recognise.
///
/// **155 is measured rather than estimated**, from the tree a successful build left behind plus the
/// try-compile scratch path from the failing one:
/// `ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-build/CMakeFiles/` is 87
/// characters below the build folder, and `CMakeScratch/TryCompile-xxxxxx/CMakeFiles/cmTC_xxxxx.dir/
/// build.make` adds 68 more. The run that failed was 126 + 155 = 281; the one that works is
/// 52 + 155 = 207.
///
/// **What this checks is the installed path, not the folder's name.** A short `into` buys margin
/// and does not create it — the thing that would break this is a deep `LOCALAPPDATA`, which is
/// somebody else's machine rather than an edit to this repository.
#[test]
fn there_is_room_under_the_path_limit_for_what_cmake_puts_below_it() {
    let at = speaker::shed().join("b");
    let used = at.to_string_lossy().chars().count();
    assert!(used + 155 < 260,
            "{} is {used} characters and cmake puts 155 below it", at.display());
    //: And the binary really is at the top of that folder rather than in `bin` — getting this
    //: wrong means a successful build reported as a missing file.
    assert!(speaker::server_exe().starts_with(&at), "{}", speaker::server_exe().display());
}

/// **What the window says, checked against what `apply` would do.**
///
/// A description that drifts from the doing is an approval that means nothing, which is the whole
/// argument of `fix.rs`.
#[test]
fn the_window_names_the_commit_the_size_the_port_and_the_gpu() {
    let offer = Fix::BuildSpeaker {
        repo: speaker::REPO.into(),
        commit: speaker::COMMIT.into(),
        why: "이유".into(),
        minutes: speaker::MINUTES,
        built: false,
    };
    let said = describe(&offer);
    assert!(said.contains(&speaker::COMMIT[..12]), "{said}");
    assert!(said.contains(speaker::REPO), "{said}");
    assert!(said.contains(&format!("{}분", speaker::MINUTES)), "{said}");
    assert!(said.contains(&format!("{} MB", speaker::WEIGHTS_MB)), "{said}");
    assert!(said.contains(argmind_env_qwentts::QWENTTS), "{said}");
    //: The two facts a person cannot get anywhere else: it will use the graphics card, and once it
    //: is up the machine will make a noise.
    assert!(said.contains("그래픽카드"), "{said}");
    assert!(said.contains("소리"), "{said}");
    assert!(offer.runs_a_stranger(), "it compiles and runs somebody else's program");
    assert!(offer.changes_the_machine());

    //: **The second time is a different question**, and it must not describe a download.
    let again = Fix::BuildSpeaker {
        repo: speaker::REPO.into(), commit: speaker::COMMIT.into(), why: "이유".into(),
        minutes: speaker::MINUTES, built: true,
    };
    let shorter = describe(&again);
    assert!(!shorter.contains("MB"), "the shorter question still describes a download: {shorter}");
    assert!(!shorter.contains(speaker::REPO), "{shorter}");
    assert!(shorter.contains("이미 빌드되어"), "{shorter}");
    assert!(again.title() != offer.title(), "two questions, two titles");
}

/// **The recogniser's approval is not the voice's.**
///
/// They are the same shape and a shared variant with a flag would have made one yes cover both.
#[test]
fn building_the_voice_is_a_different_decision_from_building_the_recogniser() {
    let voice = Fix::BuildSpeaker { repo: speaker::REPO.into(), commit: speaker::COMMIT.into(),
                                    why: "이유".into(), minutes: speaker::MINUTES, built: false };
    let ears = Fix::BuildRecogniser { repo: speaker::REPO.into(), commit: speaker::COMMIT.into(),
                                      why: "이유".into(), minutes: speaker::MINUTES, built: false };
    assert_ne!(voice, ears);
    assert_ne!(voice.title(), ears.title());
}

/// **The SDK is asked for with elevation, and not silently.**
///
/// The one headless attempt on this machine ended in `Installation aborted!`, so the command says
/// `--interactive` on purpose. `RunElevated` is the variant that makes Windows ask a second time.
#[test]
fn the_sdk_is_offered_as_an_elevated_interactive_install() {
    let Fix::RunElevated { command, why } = speaker::sdk_fix() else {
        panic!("a machine-wide SDK install is an elevated one");
    };
    assert!(command.contains("KhronosGroup.VulkanSDK"), "{command}");
    assert!(command.contains("--interactive"), "silent installs of this SDK have failed: {command}");
    //: Without these it waits on a prompt nobody can see and looks like a hang.
    assert!(command.contains("--accept-package-agreements"), "{command}");
    assert!(command.contains("--accept-source-agreements"), "{command}");
    //: The reason a person is shown has to be the measured one rather than "it is faster".
    assert!(why.contains("32배"), "{why}");
    assert!(describe(&speaker::sdk_fix()).contains("관리자 권한"));
}

/// **The SDK is not looked for on this process's PATH.**
///
/// A process keeps the environment it was born with, and the sequence that matters is argmind
/// asking for the SDK, the person installing it, and argmind still running. `glslc --version`
/// would answer *not installed* on a machine where it is.
#[test]
fn the_sdk_is_found_through_the_machines_setting_rather_than_this_processs_path() {
    //: Whether it is installed here is not the assertion — both answers are legitimate and the
    //: test has to work on a machine without it.
    match speaker::sdk() {
        Some(at) => {
            assert!(at.join("Bin").join("glslc.exe").is_file(),
                    "{} was returned without the tool that makes it usable", at.display());
            //: And it is not in `TOOLS`, which is the list checked by running `--version`.
            assert!(!speaker::TOOLS.contains(&"glslc"),
                    "glslc in TOOLS means the stale-PATH check is back");
        }
        None => assert!(speaker::missing_tools().len() <= speaker::TOOLS.len()),
    }
    //: The reader itself, against what `reg.exe` actually prints. **A path with a space in it is
    //: the silent failure** — split on whitespace it becomes `C:\Program`, and the build then
    //: fails somewhere that says nothing about why.
    let printed = concat!(
        "\r\n",
        r"HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment", "\r\n",
        r"    VULKAN_SDK    REG_SZ    C:\Program Files\Vulkan SDK", "\r\n\r\n");
    assert_eq!(building::value_in(printed, "VULKAN_SDK").as_deref(),
               Some(r"C:\Program Files\Vulkan SDK"));
    //: The expandable kind is the other thing an installer writes.
    assert_eq!(building::value_in(r"    NAME    REG_EXPAND_SZ    %SystemRoot%\x", "NAME")
                   .as_deref(), Some(r"%SystemRoot%\x"));
    //: Absent, and empty, are both *no*. An empty value read as a path would be a build configured
    //: with `VULKAN_SDK=""`, which cmake reports as the SDK being broken rather than missing.
    assert!(building::value_in(printed, "OTHER").is_none());
    assert!(building::value_in("    NAME    REG_SZ    ", "NAME").is_none());

    //: And end to end on a variable every Windows machine sets.
    let windir = building::machine_env("windir").expect("every Windows machine sets windir");
    assert!(std::path::Path::new(&windir).is_dir(), "{windir:?} is not a directory");
    assert!(building::machine_env("ARGMIND_NOT_A_REAL_VARIABLE").is_none());
}

/// **Nothing is offered that cannot be done**, and the SDK comes before the build.
#[test]
fn the_build_is_only_offered_when_the_tools_and_the_sdk_are_both_there() {
    let needs = argmind_serve::needs::every("qwen3:8b");
    let speaking = needs.iter().find(|n| n.key == "speaker").expect("a 목소리 row");

    //: **The Vulkan row is a Windows fact rather than a Windows fix**, so off Windows there is no
    //: row at all — the SDK does not arrive through `winget` anywhere else, and saying it is
    //: missing would be describing somebody's Ubuntu in terms of a package manager it has not got.
    //: `tests/portable.rs` is where its absence is the assertion.
    let Some(vulkan) = needs.iter().find(|n| n.key == "vulkan") else {
        assert!(!argmind_serve::platform::can_carry_out_fixes(),
                "the Vulkan row is gone on a build that could have used it");
        assert!(speaking.fix.is_none(), "offered a voice build with no way to run it");
        return;
    };

    if speaker::sdk().is_none() || !speaker::missing_tools().is_empty() {
        assert!(speaking.fix.is_none(),
                "offered a build that would fail: sdk={:?} missing={:?}",
                speaker::sdk(), speaker::missing_tools());
        assert!(!speaking.how.is_empty(), "and said nothing about what to do instead");
    } else {
        assert!(matches!(speaking.fix, Some(Fix::BuildSpeaker { .. })));
    }
    //: The SDK's own row always offers something, because installing it needs nothing else.
    assert!(matches!(vulkan.fix, Some(Fix::RunElevated { .. })));
    //: **Neither is required.** Losing the neural voice is losing a better voice, not losing sound.
    assert!(!speaking.required);
    assert!(!vulkan.required);
}

/// Both are remembered decisions, and each is about one thing.
#[test]
fn the_voice_and_the_sdk_are_each_their_own_permission() {
    for key in ["vulkan", "speaker"] {
        assert!(argmind_serve::decided::is_decidable(key), "{key} cannot be remembered");
        assert_ne!(argmind_serve::decided::what(key), "알 수 없는 권한", "{key} has no name");
    }
    //: *Every* need being decidable is asserted in `tests/decided.rs`, where the list lives —
    //: one rule, one place.
}

/// **The neural voice is used when it is there and not when it is not.**
///
/// The branch that decides which voice the application speaks with. Both answers are legitimate
/// — what is asserted is that it agrees with the server, so inverting the condition fails here
/// rather than in a room.
///
/// Opens no audio device: the loudspeaker waits until something is said.
#[test]
fn the_neural_voice_is_chosen_exactly_when_the_server_is_answering() {
    //: **Two conditions, not one**, and the second was learned the hard way — on a headless build
    //: the synthesiser may well be answering, because it is a server and somebody's real argmind
    //: is running it, and this build still cannot play a note. `platform::neural_voice` says so at
    //: compile time. The rule got narrower rather than weaker.
    let possible = argmind_serve::platform::neural_voice(speaker::QWENTTS_HOST).is_some();
    let answering = speaker::answering();
    let (voice, which) = argmind_serve::app::pick_voice();
    if possible {
        assert!(answering, "a neural voice was available with nothing answering");
        assert_eq!(which, "neural", "the synthesiser is up and the old voice was chosen");
    } else {
        assert_ne!(which, "neural",
                   "chose the neural voice on a build that cannot play one");
        //: Windows still speaks, or it says it cannot. Both are honest; silently claiming to speak
        //: is not.
        assert!(matches!(which, "windows" | "none"), "{which}");
    }
    assert_eq!(voice.is_some(), which != "none");
    if let Some(mut it) = voice {
        //: Released rather than dropped mid-test: it may hold a COM object.
        it.release();
    }
}

/// The two built things do not share a shed, a port or a stamp.
#[test]
fn the_voice_and_the_recogniser_do_not_stand_on_each_other() {
    assert_ne!(speaker::shed(), argmind_serve::recogniser::shed());
    assert_ne!(speaker::server_exe(), argmind_serve::recogniser::server_exe());
    assert_ne!(argmind_env_qwentts::QWENTTS, argmind_env_whisper::WHISPER);
    //: `built-at.txt` sits beside the binary, so two binaries in one folder would share one stamp
    //: and each would claim the other's commit.
    assert_ne!(speaker::server_exe().parent(),
               argmind_serve::recogniser::server_exe().parent());
}

/// **A stamp from another commit does not count as built.**
///
/// Writes and restores the file beside the binary, so it says nothing about whether a build exists
/// on the machine running the suite.
#[test]
fn a_binary_from_another_commit_is_not_ready() {
    let stamp = speaker::server_exe().with_file_name("built-at.txt");
    let before = std::fs::read_to_string(&stamp).ok();
    if !speaker::server_exe().is_file() {
        //: Nothing built here: `ready` is false whatever the stamp says, which is the other half of
        //: the rule and is worth asserting on a clean machine.
        assert!(!speaker::ready());
        return;
    }
    speaker::remember_build("0000000000000000000000000000000000000000");
    assert!(!speaker::ready(), "a binary built from another commit was called ready");
    speaker::remember_build(speaker::COMMIT);
    assert!(speaker::ready());
    assert_eq!(speaker::built_at().as_deref(), Some(speaker::COMMIT));
    match before {
        Some(it) => speaker::remember_build(it.trim()),
        None => { let _ = std::fs::remove_file(&stamp); }
    }
}
