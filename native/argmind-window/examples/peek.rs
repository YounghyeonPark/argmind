//! Open the window, look at it, close it. **Draws on screen.**
//!
//!     cargo run -p argmind-window --example peek -- http://127.0.0.1:7331
//!     cargo run -p argmind-window --example peek -- http://127.0.0.1:7331 --quiet --seconds 6
//!
//! `--quiet` builds the window without focus, which is what a measurement wants: a window that
//! takes the keyboard away from whatever the person was doing is an interruption, and that has
//! happened here more than once. `--seconds N` closes it on a timer, so nothing is left behind if
//! nobody is watching.
//!
//! The URL has to be loopback. There is no address bar and this is not a browser.

use std::sync::mpsc::channel;
use std::time::Duration;

use argmind_window::{run, Say, Shell};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let url = args.iter().find(|a| a.starts_with("http")).cloned()
        .unwrap_or_else(|| "http://127.0.0.1:7331".into());
    let quiet = args.iter().any(|a| a == "--quiet");
    let seconds: Option<u64> = args.iter().position(|a| a == "--seconds")
        .and_then(|at| args.get(at + 1)).and_then(|n| n.parse().ok());

    println!("  여는 중: {url}{}", if quiet { "  (창을 앞으로 내세우지 않습니다)" } else { "" });
    let (tell, orders) = channel();
    if let Some(after) = seconds {
        println!("  {after}초 뒤에 닫습니다");
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(after));
            let _ = tell.send(Say::Quit);
        });
    }

    let shell = Shell {
        url,
        tooltip: "argmind — peek".into(),
        focused: !quiet,
        //: The same folder the real application uses, so this does not leave a second cache
        //: somewhere nobody would look for it.
        data: std::env::var("LOCALAPPDATA").ok()
            .map(|at| std::path::PathBuf::from(at).join("argmind").join("webview")),
        ..Shell::default()
    };
    if let Err(why) = run(shell, Some(orders)) {
        eprintln!("  열지 못했습니다: {why}");
        std::process::exit(1);
    }
}
