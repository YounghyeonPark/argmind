//! `argmind-app` — double-click it. **No terminal, and something to close.**
//!
//! The interface is a browser tab, and a tab is not the program: closing it leaves a server running
//! with nothing on screen to say so. A console window would say so and is not an interface. So this
//! is the same server as `serve`, with an icon by the clock and a menu with two items.
//!
//! `#![windows_subsystem = "windows"]` is what makes double-clicking it not flash a black window.
//! It also means there is nowhere for `println!` to go, which is why anything worth saying goes to
//! the page rather than to a stream nobody will read.
//!
//! **It refuses to be the second one.** Two of these would fight over the port, and the second
//! would die with a message nobody can see — so it looks first, and if argmind is already running
//! it opens the page and gets out of the way.

#![windows_subsystem = "windows"]

use std::net::TcpStream;
use std::time::Duration;

use argmind_serve::app::{How, URL};

const ADDRESS: &str = "127.0.0.1:7331";

/// Is argmind already up?
fn already_running() -> bool {
    TcpStream::connect_timeout(&ADDRESS.parse().expect("a literal address"),
                               Duration::from_millis(300)).is_ok()
}

fn main() {
    //: **Opening the page for an argmind that is already running is the right answer**, not an
    //: error: somebody double-clicking a second time wants to see it, and telling them off for a
    //: reasonable thing is how a program earns a reputation.
    if already_running() {
        let _ = std::process::Command::new("explorer").arg(URL).spawn();
        return;
    }

    let how = How::from_args();
    let tooltip = format!("argmind — {}", how.line());
    //: The server owns a thread and the tray owns this one, because a message loop that stops
    //: pumping stops the icon responding — and an icon that does not respond is the failure this
    //: whole binary exists to avoid.
    std::thread::spawn(move || argmind_serve::app::run(how));

    //: Wait for it to answer before drawing anything. An icon that appears and then opens a page
    //: to a refused connection is worse than one that appears a second later.
    for _ in 0..40 {
        if already_running() {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let _ = std::process::Command::new("explorer").arg(URL).spawn();

    //: Blocks until 끝내기. When it returns the process ends, and the server thread with it —
    //: which is the whole point of there being a menu.
    let _ = argmind_env_windows::tray::run(&tooltip, URL);
}
