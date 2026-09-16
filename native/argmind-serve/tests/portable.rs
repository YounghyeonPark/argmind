//! **The application, running on an operating system nobody has ported to.**
//!
//! The owner asked on 2026-09-16 for Windows, Linux and macOS. The answer given was that
//! `argmind-core` names no platform and the contract is small enough that AT-SPI2 and the macOS
//! Accessibility API both fit it — and that the real blocker was the *application*, which named
//! Windows types directly in six files.
//!
//! **That is a claim, and a claim about portability made on one operating system is worth
//! nothing.** This file is how it gets checked without a second machine. `--features headless`
//! makes `platform.rs` pick [`argmind_env_none`] even here, and what follows starts the whole
//! server, talks to it over the socket, and reads what it says about itself.
//!
//!     cargo test --manifest-path native/Cargo.toml -p argmind-serve --features headless
//!
//! **Two failures it can produce, and both are the point.** If the seam is incomplete it does not
//! compile — which is how the surface was enumerated in the first place, rather than guessed. If
//! the seam is a fiction it compiles and then does not run, and these tests say so.
//!
//! Without the feature every test here reports that it did nothing, rather than passing quietly. A
//! test that is silently inert on the ordinary build is a test that reads green for years.

/// **Nothing here runs on the ordinary build**, and it says so rather than passing.
#[test]
#[cfg(not(feature = "headless"))]
fn this_file_needs_the_headless_feature_and_did_not_run() {
    eprintln!("  portable.rs did nothing: re-run with --features headless");
}

#[cfg(feature = "headless")]
mod without_an_environment {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpStream;

    use argmind_core::contract::{Eyes, Hands};
    use argmind_voice::Speaker;
    use argmind_serve::platform;

    /// The build knows what it is, and says so in one word.
    #[test]
    fn it_says_it_has_no_body() {
        assert!(!platform::embodied(), "the headless build claims it can see and press");
        assert!(platform::describe().contains("headless"), "{}", platform::describe());
    }

    /// **Eyes that answer, and answer nothing.**
    ///
    /// An empty reading is a legitimate reading: the agent then has nothing to offer. What would
    /// not be legitimate is a panic, or a pretend widget.
    #[test]
    fn the_eyes_open_and_see_nothing() {
        let mut eyes = platform::Sight::new(platform::Watching::default()).expect("eyes open");
        assert!(eyes.foreground().is_empty());
        let (widgets, took) = eyes.timed_foreground();
        assert!(widgets.is_empty());
        assert_eq!(took, 0.0);
        assert_eq!(eyes.window().1, "");
        assert!(eyes.inner.at_point(10, 10).is_none());
    }

    /// **Hands that are dry, and say so through the flag the rest of the program already reads.**
    #[test]
    fn the_hands_are_never_live() {
        let mut hands = platform::Touch::unrestricted().arm();
        assert!(!hands.live(), "arming made these hands claim to be live");
        hands.press("enter");
        //: Still recorded. An empty log would make every refusal look like a success.
        assert_eq!(hands.done.len(), 1);
        assert_eq!(hands.done[0], platform::Did::Sent { live: false });
    }

    /// **Nothing is approved when there is nowhere to ask.**
    ///
    /// The sharpest line in the whole file. Every install, every build of somebody else's source,
    /// every opening of the microphone is guarded by a question — and an environment that answered
    /// yes on the person's behalf would turn *cannot ask* into *said yes for you*.
    #[test]
    fn a_question_with_nowhere_to_ask_it_is_a_no() {
        assert!(!platform::confirm("argmind", "설치할까요?"));
    }

    /// **The voice is silent and is honest about Korean.**
    ///
    /// `has_korean` being false is what stops the router handing it Korean — the measured Windows
    /// failure was an English voice accepting Korean, emitting zero bytes and reporting success.
    #[test]
    fn the_voice_makes_no_sound_and_does_not_claim_korean() {
        let mut mouth = platform::Mouth::new().expect("a voice");
        assert!(!mouth.has_korean());
        mouth.say("들리나요");
        assert!(!mouth.speaking());
        //: And no neural one either: the synthesiser is a server that runs anywhere, and the
        //: loudspeaker that would play it does not.
        assert!(platform::neural_voice("127.0.0.1:8179").is_none());
    }

    /// **A child process really starts**, and the promise about its death is the weaker one.
    ///
    /// `std::process` is the same everywhere, so this is not stubbed. What is missing is the job
    /// object, and [`platform::Supervised::supervised`] says so rather than letting the caller
    /// make a promise the operating system is not keeping.
    #[test]
    fn a_child_starts_and_the_promise_is_the_weaker_one() {
        //: `cmd /c exit 0` on Windows; the headless build still runs here, so this is the one test
        //: in the file that knows what machine it is on.
        let exe = std::path::PathBuf::from(if cfg!(windows) { "cmd" } else { "sh" });
        let args: &[&str] = if cfg!(windows) { &["/c", "exit", "0"] } else { &["-c", "exit 0"] };
        let mut child = platform::Supervised::start(&exe, args, None, None)
            .expect("a child process starts");
        assert!(!child.supervised(), "the empty environment claimed a job object");
        for _ in 0..100 {
            if child.finished() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        panic!("the child never finished");
    }

    /// **The whole server runs**, and the page it serves is the same page.
    ///
    /// This is the test the rest of the file exists for. It starts `app::run` on a thread exactly
    /// as `argmind-app` does, waits for the socket, and reads the page — with no eyes, no hands,
    /// no voice and no microphone anywhere in the process.
    #[test]
    fn the_server_starts_and_serves_its_page_with_no_environment_at_all() {
        //: **A port of its own, and this is not fussiness.** On the ordinary one a real argmind may
        //: be listening, and then this test reads *that* page and reports that the headless server
        //: works — which it did, the first time it was written, while the thread it started was
        //: panicking on `could not bind`. A port nothing else uses is what makes the pass mean
        //: something.
        const MINE: u16 = 7731;
        let how = argmind_serve::app::How {
            dry: true, speak: false, hear: false, model_name: "qwen3:8b".into(), port: MINE,
        };
        std::thread::spawn(move || argmind_serve::app::run(how));

        let mut page = None;
        for _ in 0..100 {
            if let Ok(got) = fetch(&argmind_serve::app::address(MINE), "/") {
                page = Some(got);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        let page = page.expect("the server answered within five seconds");
        assert!(page.contains("<canvas"), "the orb is not in the page");
        assert!(page.contains("data-token"), "the page carries no token");
    }

    /// One GET, and the body back. The server's own client is not used, because this is testing it.
    fn fetch(host: &str, path: &str) -> Result<String, String> {
        let mut socket = TcpStream::connect(host).map_err(|e| format!("{e}"))?;
        let _ = socket.set_read_timeout(Some(std::time::Duration::from_secs(5)));
        write!(socket, "GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n")
            .map_err(|e| format!("{e}"))?;
        let mut reading = BufReader::new(socket);
        let mut line = String::new();
        reading.read_line(&mut line).map_err(|e| format!("{e}"))?;
        if !line.contains("200") {
            return Err(line);
        }
        loop {
            let mut header = String::new();
            if reading.read_line(&mut header).map_err(|e| format!("{e}"))? == 0
                || header.trim().is_empty() {
                break;
            }
        }
        let mut body = String::new();
        let _ = reading.read_to_string(&mut body);
        Ok(body)
    }
}
