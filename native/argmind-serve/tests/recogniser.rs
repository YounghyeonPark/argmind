//! Building the recogniser, and the pieces of that which can be checked without building it.
//!
//! **The build itself is `#[ignore]`d**: it clones from GitHub, downloads 141 MB and compiles for
//! minutes. It is here rather than absent because the only thing that answers *does this work on a
//! machine that has never had it* is doing it on such a machine — and every part of it that can be
//! checked cheaply is checked below without the network.
//!
//! Run the slow one with
//! `cargo test -p argmind-serve --test recogniser -- --ignored --nocapture`.

use argmind_serve::fix::{apply, describe, Fix};
use argmind_serve::recogniser::{self, imports_of, last_few, missing_tools, percent_of,
                                server_exe, toolchain_dir, COMMIT, MINUTES, REPO};

/// **The commit is a commit**, not a tag and not a branch.
///
/// The whole verification story rests on this: git names an object by its content, so asking for
/// this string can only ever produce this tree. A tag in the same slot would look identical in the
/// window and mean nothing, because a tag can be moved.
#[test]
fn what_is_pinned_is_a_full_object_name() {
    assert_eq!(COMMIT.len(), 40, "a short name is not a name git verifies the same way");
    assert!(COMMIT.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
    assert!(REPO.starts_with("https://"), "{REPO}");
    assert!(REPO.ends_with(".git"));
}

/// The percentage cmake prints, and everything that is not one.
#[test]
fn a_progress_line_is_recognised_and_nothing_else_is() {
    assert_eq!(percent_of("[ 42%] Building CXX object x.obj"), Some(42));
    assert_eq!(percent_of("[100%] Linking CXX executable whisper-server.exe"), Some(100));
    assert_eq!(percent_of("  [  4%] indented by make"), Some(4));
    //: A line that merely starts with a bracket is not progress — and a build prints plenty.
    assert_eq!(percent_of("[warning] something"), None);
    assert_eq!(percent_of("-- Configuring done"), None);
    assert_eq!(percent_of(""), None);
    assert_eq!(percent_of("["), None);
    assert_eq!(percent_of("[%]"), None);
}

/// A tail is a tail, and a short list is all of it rather than a panic.
#[test]
fn the_last_few_of_a_short_list_is_the_whole_list() {
    let all = [1, 2, 3];
    assert_eq!(last_few(&all, 12), &[1, 2, 3]);
    assert_eq!(last_few(&all, 2), &[2, 3]);
    assert_eq!(last_few(&all, 0), &[] as &[i32]);
    assert_eq!(last_few(&[] as &[i32], 5), &[] as &[i32]);
}

/// **What a binary imports is read from the binary.**
///
/// Uses the test binary itself, which is the one executable guaranteed to be present. The point is
/// that the list is not empty and is made of names — the production use of it is on a freshly
/// compiled server, and a remembered list of DLLs is the thing this avoids.
#[test]
fn the_libraries_a_binary_needs_come_from_the_binary() {
    let Some(_) = toolchain_dir() else {
        eprintln!("  건너뜀: g++이 없습니다");
        return;
    };
    let me = std::env::current_exe().expect("this test binary");
    let names = imports_of(&me).expect("objdump read it");
    assert!(!names.is_empty(), "an executable that imports nothing at all");
    assert!(names.iter().any(|n| n.to_lowercase().contains("kernel32")), "{names:?}");
    assert!(names.iter().all(|n| n.to_lowercase().ends_with(".dll")), "{names:?}");

    //: A file that is not a binary is not answered with an empty list.
    let not_a_binary = std::env::temp_dir().join("argmind-not-a-binary.txt");
    std::fs::write(&not_a_binary, b"hello").unwrap();
    assert!(imports_of(&not_a_binary).is_err());
    let _ = std::fs::remove_file(&not_a_binary);
}

/// Where things are put: under the person's own folder, never beside the executable.
#[test]
fn the_source_and_the_binary_live_under_the_users_own_folder() {
    let shed = recogniser::shed();
    assert!(shed.to_string_lossy().contains("argmind"), "{shed:?}");
    assert!(server_exe().starts_with(&shed));
    assert!(server_exe().ends_with("whisper-server.exe"));
    //: Not inside the source tree — models are shared with anything else that wants one.
    assert!(!recogniser::model_at().starts_with(&shed));
    assert!(recogniser::model_at().ends_with("ggml-base.bin"));
}

/// The model half is **the same fix** that would be offered for the model on its own.
#[test]
fn the_model_half_is_a_fetch_with_the_published_hash() {
    let Fix::Fetch { url, sha256, about_mb, .. } = recogniser::model_fix() else {
        panic!("the model is data and must be fetched as data, never executed");
    };
    assert_eq!(url, recogniser::MODEL_URL);
    assert_eq!(sha256.as_deref(), Some(recogniser::MODEL_SHA256));
    assert_eq!(about_mb, recogniser::MODEL_MB);
    assert_eq!(recogniser::MODEL_SHA256.len(), 64);
    assert!(!recogniser::model_fix().runs_a_stranger());
}

/// **The window says the sequence**, including the part somebody would object to.
#[test]
fn the_window_says_it_will_run_what_it_compiles() {
    let fix = Fix::BuildRecogniser { repo: REPO.into(), commit: COMMIT.into(),
                                     why: "음성 입력에는 받아쓰기 서버가 필요합니다.".into(),
                                     minutes: MINUTES, built: false };
    let said = describe(&fix);
    //: Where from, and which commit — the first twelve characters, which is what a person could
    //: actually compare against a page on GitHub.
    assert!(said.contains("github.com/ggml-org/whisper.cpp"), "{said}");
    assert!(said.contains(&COMMIT[..12]), "{said}");
    //: **That a commit name is not a tag.** It is the whole reason this is verified at all, and it
    //: is the sentence somebody would need in order to check the claim.
    assert!(said.contains("태그가 아니라"), "{said}");
    //: That it compiles here, roughly how long, and **that it then runs**.
    assert!(said.contains("빌드"), "{said}");
    assert!(said.contains(&format!("{MINUTES}분")), "{said}");
    assert!(said.contains("실행합니다"), "{said}");
    assert!(said.contains("남이 쓴 코드"), "{said}");
    //: The model comes with it, so nobody is surprised by 141 MB.
    assert!(said.contains(&format!("{} MB", recogniser::MODEL_MB)), "{said}");
    //: And where it will listen, and that it goes when argmind goes.
    assert!(said.contains(argmind_env_whisper::WHISPER), "{said}");
    assert!(said.contains("함께 꺼집니다"), "{said}");
    assert!(said.trim_end().ends_with("까요?"), "{said}");

    //: Compiling it yourself does not make it yours: this is running a stranger, and the property
    //: says so rather than the wording.
    assert!(fix.runs_a_stranger());
    assert!(fix.changes_the_machine());
    assert!(fix.title().contains("argmind"));
}

/// **The second time, it does not promise a download and a build.**
///
/// This is the ordinary case — the tree stays between runs — and describing ten minutes of
/// compiling to somebody whose machine will take five seconds is the description drifting from the
/// doing. What must not drift is the other half: it still runs code somebody else wrote.
#[test]
fn an_already_built_server_is_offered_as_a_start_not_a_build() {
    let already = Fix::BuildRecogniser { repo: REPO.into(), commit: COMMIT.into(),
                                         why: "왜".into(), minutes: MINUTES, built: true };
    let said = describe(&already);
    assert!(said.contains("이미 빌드되어 있습니다"), "{said}");
    assert!(said.contains("내려받지도, 컴파일하지도 않습니다"), "{said}");
    //: None of the promises that would now be false.
    assert!(!said.contains(&format!("{MINUTES}분")), "{said}");
    assert!(!said.contains("MB"), "{said}");
    assert!(!said.contains("github.com"), "{said}");
    //: But the commit is still named, because what runs is what was built from it.
    assert!(said.contains(&COMMIT[..12]), "{said}");
    //: **And it is still a stranger's code.** The grade does not soften because the compiling
    //: happened last week.
    assert!(already.runs_a_stranger());
    assert!(said.contains("남이 쓴 코드"), "{said}");
    //: The title asks the question the body asks.
    assert!(already.title().contains("띄울까요"), "{}", already.title());
    assert!(!already.title().contains("빌드"), "{}", already.title());
}

/// `ready()` means **built from this commit**, not merely a file with the right name.
#[test]
fn a_binary_from_some_other_commit_does_not_count_as_ready() {
    let stamp = recogniser::server_exe().with_file_name("built-at.txt");
    let was = std::fs::read_to_string(&stamp).ok();
    if !recogniser::server_exe().is_file() {
        //: Nothing built here, so the only claim to check is the one that costs nothing.
        assert!(!recogniser::ready());
        return;
    }
    recogniser::remember_build("0000000000000000000000000000000000000000");
    assert!(!recogniser::ready(), "a binary built from another commit was called ready");
    recogniser::remember_build(COMMIT);
    assert!(recogniser::ready());
    assert_eq!(recogniser::built_at().as_deref(), Some(COMMIT));
    //: Left as it was found.
    match was {
        Some(it) => recogniser::remember_build(it.trim()),
        None => {
            let _ = std::fs::remove_file(&stamp);
        }
    }
}

/// **A short commit does not panic the window.**
///
/// `describe` is called on whatever it is given, and a slice by byte index is how a description
/// crashes the thing that was about to ask a question.
#[test]
fn a_description_survives_a_commit_that_is_not_forty_characters() {
    for commit in ["", "abc", "a".repeat(40).as_str()] {
        for built in [false, true] {
            let said = describe(&Fix::BuildRecogniser { repo: REPO.into(), commit: commit.into(),
                                                       why: "왜".into(), minutes: 1, built });
            assert!(said.ends_with("까요?"), "{said}");
        }
    }
}

/// **The offer is not made when it cannot be kept.**
///
/// On this machine the tools are there, so what is checked is the shape of the rule rather than
/// its unhappy branch: a need that reports a missing tool has no button, and one that does not
/// report one has a button that is a build.
#[test]
fn the_button_exists_only_when_the_tools_do() {
    let whisper = argmind_serve::needs::by_key("qwen3:8b", "whisper").expect("the whisper need");
    //: **Two conditions now, not one.** The tools have to be here *and* this build has to have a
    //: tested way to use them — a headless build has the compiler and none of the rest, and an
    //: offer there would be a button that has never worked. The rule got narrower rather than
    //: weaker, so it is spelled out rather than relaxed.
    let could = argmind_serve::platform::can_carry_out_fixes();
    match (could && missing_tools().is_empty(), &whisper.fix) {
        (true, Some(Fix::BuildRecogniser { .. })) => {
            assert!(whisper.how.contains("빌드"), "{}", whisper.how);
        }
        (false, None) => {
            //: And it names the program to install rather than saying it cannot — where the tools
            //: are the thing in the way. Where the *build* is the thing in the way it says that
            //: instead, and either way it says something.
            assert!(!whisper.how.is_empty());
            if could {
                for tool in missing_tools() {
                    assert!(whisper.how.contains(tool), "{} does not name {tool}", whisper.how);
                }
            }
        }
        (offerable, fix) => panic!("offerable: {offerable}, and the offer is {fix:?}"),
    }
    //: Either way it is optional — argmind runs without voice input, and a need that cannot be met
    //: must not be the thing that makes the program look broken.
    assert!(!whisper.required);
}

/// **The whole sequence, on a machine that has none of it.**
///
/// Clone at the pinned commit, fetch and check the model, compile, bring the compiler's runtime
/// along, start the server, and ask it to transcribe. Ignored by default — it needs the network
/// and several minutes.
///
/// It deletes the source tree and the build first, because a test that passes on what a previous
/// run left behind answers a different question. **It does not delete the model**: the fetch is
/// covered against the real server in `fetch.rs`, and 141 MB per run buys nothing that is not
/// already checked. Delete it by hand to exercise that half too.
#[ignore = "clones, downloads 141 MB and compiles for minutes"]
#[test]
fn it_builds_from_nothing_and_the_server_answers() {
    let _ = std::fs::remove_dir_all(recogniser::shed());
    assert!(!server_exe().exists());

    let fix = Fix::BuildRecogniser { repo: REPO.into(), commit: COMMIT.into(),
                                     why: "시험".into(), minutes: MINUTES, built: false };
    let began = std::time::Instant::now();
    let said = apply(&fix, &mut |s| println!("  {s}")).expect("built and started");
    println!("  {said} — {:.0}초", began.elapsed().as_secs_f64());

    assert!(server_exe().is_file());
    //: Fetched by this run or already there — either way, the sequence does not report success
    //: with the model missing.
    assert!(recogniser::model_at().is_file());
    //: **The runtime came with it.** Without this the binary runs from a shell that happens to
    //: have the compiler on its path and nowhere else.
    let beside = server_exe().parent().unwrap().to_path_buf();
    assert!(std::fs::read_dir(&beside).unwrap()
                .filter_map(Result::ok)
                .any(|e| e.file_name().to_string_lossy().to_lowercase().ends_with(".dll")),
            "no runtime libraries were put beside the server");
    assert!(recogniser::answering());
    assert!(recogniser::ours(), "argmind should know it started this one");

    //: And it transcribes — the thing all of the above was for. A file, not a microphone.
    let sample = recogniser::shed().join("samples").join("jfk.wav");
    let out = std::process::Command::new("curl.exe")
        .args(["-s", "--max-time", "120", "-F"])
        .arg(format!("file=@{}", sample.display()))
        .args(["-F", "response_format=json", "-F", "language=en",
               &format!("http://{}/inference", argmind_env_whisper::WHISPER)])
        .output().expect("curl");
    let heard = String::from_utf8_lossy(&out.stdout).to_lowercase();
    assert!(heard.contains("ask not what your country"), "{heard}");
}
