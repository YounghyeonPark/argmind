//! `argmind-app` — double-click it. **A window of its own, and something to close.**
//!
//! It used to open a browser tab, and a tab is not the program: closing it left a server running
//! with nothing on screen to say so. The owner's instruction on 2026-09-16 was
//! **브라우저말고 단독앱으로 만들어줘**, so the page now lives in a window this program owns —
//! [`argmind_window`], which is WebView2 here and WKWebView or WebKitGTK elsewhere.
//!
//! `#![windows_subsystem = "windows"]` is what makes double-clicking it not flash a black window.
//! It also means there is nowhere for `println!` to go, which is why anything worth saying goes to
//! the page rather than to a stream nobody will read.
//!
//! **A second launch is a second window, not a second argmind.** It finds the server already
//! answering, starts none of its own, draws no icon, and just shows the page — so there is one
//! server, one tray icon and one thing to quit. Closing that window closes only it.
//!
//! **What quits what.** 끝내기 in the tray menu ends this process, and the job object ends the
//! recogniser and the voice with it. That is the operating system's promise rather than a
//! destructor's, which matters here more than usual: the window library ends the process itself
//! when its loop stops, so destructors do not run at all.

#![windows_subsystem = "windows"]

use std::net::TcpStream;
use std::path::PathBuf;
use std::time::Duration;

use argmind_serve::app::{How, URL};
use argmind_window::Shell;

const ADDRESS: &str = "127.0.0.1:7331";

/// Is argmind already up?
fn already_running() -> bool {
    TcpStream::connect_timeout(&ADDRESS.parse().expect("a literal address"),
                               Duration::from_millis(300)).is_ok()
}

/// Where the webview keeps its cache and its storage.
///
/// **Beside the person's own files rather than beside the executable**, which is where the library
/// would put it — the same reason `needs::model_shed()` exists, and the same failure if it is
/// forgotten: a folder under the program needs write access to wherever argmind was unpacked.
///
/// This is the one place a platform is named for the window, and it is named here rather than in
/// [`argmind_window`] so that crate can stay ignorant of all three.
fn webview_shed() -> Option<PathBuf> {
    std::env::var("LOCALAPPDATA").ok()
        .map(|at| PathBuf::from(at).join("argmind").join("webview"))
}

/// Say it in a window, because there is no console to say it in.
fn complain(title: &str, body: &str) {
    argmind_serve::platform::confirm(title, body);
}

fn main() {
    let second = already_running();
    let how = How::from_args();
    let tooltip = format!("argmind — {}", how.line());

    if !second {
        //: The server owns a thread and the window owns this one. The window library's event loop
        //: must be on the thread the process started on — that is a rule on macOS rather than a
        //: preference — so the server is the one that moves.
        std::thread::spawn(move || argmind_serve::app::run(how));

        //: Wait for it to answer before drawing anything. A window that appears and then shows a
        //: refused connection is worse than one that appears a second later.
        let mut up = false;
        for _ in 0..40 {
            if already_running() {
                up = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        if !up {
            complain("argmind — 서버가 뜨지 않았습니다",
                     &format!("{ADDRESS} 가 4초 안에 응답하지 않았습니다.\n\n\
                               다른 프로그램이 그 포트를 쓰고 있을 수 있습니다.\n\
                               콘솔에서 쓰시려면 serve.exe 를 쓰세요."));
            return;
        }
    }

    let shell = Shell {
        url: URL.into(),
        tooltip,
        //: **The second one draws no icon**, so there is one thing in the tray and quitting it
        //: quits the program rather than a viewer.
        tray: !second,
        data: webview_shed(),
        ..Shell::default()
    };

    //: **A failure here is not something to swallow.** There is no console, so an error was
    //: becoming an application that started and vanished — leaving a server nobody could see.
    if let Err(why) = argmind_window::run(shell, None) {
        complain("argmind — 창을 만들지 못했습니다",
                 &format!("{why}\n\n\
                           WebView2 런타임이 필요합니다 — Windows 11에는 기본으로 들어 있습니다.\n\
                           콘솔에서 쓰시려면 serve.exe 를 쓰세요."));
    }
}
