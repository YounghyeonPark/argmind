//! A window of its own, rather than a tab in somebody else's program.
//!
//! The interface was a browser tab and the owner's instruction on 2026-09-16 was
//! **브라우저말고 단독앱으로 만들어줘**. So: a window, an icon in the tray, and closing the window
//! hides it rather than leaving a server running with nothing on screen to say so.
//!
//! **This file names no operating system except in one function.** It is handed a URL and some
//! strings and puts them in a window; it knows nothing about sessions, models or permissions, and
//! depends on nothing of argmind's. [`attach`] is the exception and says why.
//!
//! ## Why three dependencies, in a tree that has three in total
//!
//! Every other capability here is hand-written against the platform — WASAPI, Media Foundation,
//! SAPI, UI Automation, raw HTTP/1.1. **Doing that here is worse rather than harder**, and that is
//! the whole argument:
//!
//! * The documented entry point is `CreateCoreWebView2EnvironmentWithOptions`, exported by
//!   `WebView2Loader.dll`. That DLL ships with the SDK, not with Windows, so using it means
//!   keeping somebody's binary beside the executable. whisper.cpp and qwentts.cpp are fetched and
//!   compiled here so that nothing has to be.
//! * The runtime's own `EmbeddedBrowserWebView.dll` exports
//!   `CreateWebViewEnvironmentWithOptionsInternal` and nothing else. **Measured on this machine**,
//!   with `objdump -p`, against runtime 153.0.4234.32. It is undocumented, it is what the loader
//!   calls, and a program built on it breaks when the runtime updates underneath it.
//! * And neither of those is a window on macOS or Linux, which is what was asked for.
//!
//! So the dependency is taken, and it is taken from the Tauri project, whose bindings these are:
//! `wry` (Apache-2.0 OR MIT), `tao` (Apache-2.0), `tray-icon` (MIT OR Apache-2.0). 81 crates enter
//! the Windows build graph behind them, **none of them copyleft or non-commercial** — checked
//! across the whole graph, not just the three. `docs/licences.toml` records it.
//!
//! **And the first version of that argument was too clean.** `webview2-com-sys` links
//! `WebView2LoaderStatic.lib` under `target_env = "msvc"` — a prebuilt Microsoft static library,
//! compiled into the executable. Microsoft's loader code ships either way. What is actually
//! preserved is narrower and is worth saying in its own words: **nothing sits beside the
//! executable as a loose file, and the runtime itself — the hundreds of megabytes of Chromium —
//! is never redistributed.** It is found on the machine, the way Ollama is.
//!
//! Under `-gnu` it is the DLL instead, and `argmind-app.exe` then exits on a double-click with
//! `0xC0000135` and no message while running perfectly under `cargo run`. `rust-toolchain.toml`
//! and `argmind-serve/tests/launchable.rs` are what keep that from shipping.
//!
//! ## What is verified and what is not
//!
//! **Windows, on this machine: verified.** WebView2 runtime 153.0.4234.32, which Windows 11 ships.
//! The page was fetched and its JavaScript ran — reported by a throwaway server the window was
//! pointed at, with `Edg/153` as the user agent — and the window was a real top-level one owned by
//! this process, rendering in `msedgewebview2.exe` children that were gone afterwards.
//!
//! **macOS and Linux: written for, not verified.** There is no Mac here and no WebKitGTK, and
//! `cargo check` for a target whose system libraries are absent is not evidence. Saying so is the
//! point: the code names no platform, which makes the port plausible and does not make it done.
//!
//! **And the agent inside the window is Windows-only regardless.** `argmind-core` names no
//! platform — its only dependency is `serde_json` — but the eyes are UI Automation, the hands are
//! SendInput, the microphone is WASAPI, the camera is Media Foundation and both voices are
//! Windows. On another operating system this window would open onto a server that cannot start.
//! A window is the first piece of that port and it is not the port.

use std::sync::mpsc::Receiver;

use tao::event::{Event, StartCause, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoopBuilder};
use tao::window::{Window, WindowBuilder};
use tray_icon::menu::{Menu, MenuEvent, MenuItem};
use tray_icon::{Icon, TrayIconBuilder, TrayIconEvent};
use wry::WebViewBuilder;

/// What to put in the window.
pub struct Shell {
    /// The page. **Loopback, and checked** — see [`is_loopback`].
    pub url: String,
    /// The title bar, and the name in the task switcher.
    pub title: String,
    /// The line under the pointer when it rests on the tray icon.
    pub tooltip: String,
    pub wide: u32,
    pub tall: u32,
    /// **Off for a measurement, on for a person.** A window that appears and takes the keyboard is
    /// right when somebody double-clicked the program and wrong when something is testing it —
    /// stealing the foreground from whatever the person was doing is a real interruption, and it
    /// has happened here.
    pub focused: bool,
    /// Start in the tray with no window, for a program that was launched at login.
    pub hidden: bool,
    /// Draw an icon in the tray.
    ///
    /// **Off for a second window onto an argmind that is already running.** One program, one icon:
    /// two would be two things to quit, and only one of them would be the program.
    pub tray: bool,
    /// Where the webview keeps its own files — its cache, its storage, its crash logs.
    ///
    /// **It has to be said, and it is the caller who can say it.** Left to itself WebView2 writes
    /// an `EBWebView` folder *next to the executable*, which works from a build directory and
    /// fails from `Program Files` — the same mistake `model_shed()` exists to avoid, arriving
    /// through a library. Naming the right folder needs to know the operating system, which this
    /// crate deliberately does not, so the caller passes one.
    pub data: Option<std::path::PathBuf>,
}

impl Default for Shell {
    fn default() -> Self {
        Shell {
            url: String::new(),
            title: "argmind".into(),
            tooltip: "argmind".into(),
            //: Tall rather than wide: it is a conversation with a sphere above it, not a
            //: spreadsheet.
            wide: 520,
            tall: 820,
            focused: true,
            hidden: false,
            tray: true,
            data: None,
        }
    }
}

/// **Loopback only, and refused otherwise.**
///
/// The same check `argmind-serve` makes before it binds, made again on the other side of the
/// socket. A webview pointed somewhere else is a browser, and this program is not offering one —
/// there is no address bar to type into and there should be no way to arrive anywhere else.
pub fn is_loopback(url: &str) -> bool {
    let rest = url.strip_prefix("http://").or_else(|| url.strip_prefix("https://"));
    let Some(rest) = rest else { return false };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    //: **Anything with a user in it is refused rather than parsed.** `http://127.0.0.1@evil/` is a
    //: URL whose host is `evil`, and it is the oldest way of making an address read as one place
    //: and resolve as another. This program builds its own URL and never needs credentials, so
    //: there is nothing to lose by not understanding them.
    if authority.contains('@') {
        return false;
    }
    //: **The bracketed form keeps its colons.** Splitting the authority on `:` turns `[::1]:7331`
    //: into `[`, which is not the loopback address and is not anything — the first version of this
    //: did exactly that and the test caught it.
    let host = match authority.strip_prefix('[') {
        Some(inside) => match inside.split_once(']') {
            Some((address, _)) => address,
            None => return false,
        },
        None => authority.split(':').next().unwrap_or(""),
    };
    host == "127.0.0.1" || host == "localhost" || host == "::1"
}

/// The icon, drawn rather than shipped.
///
/// **A file would be a fourth thing to keep in step** — an `.ico` for Windows, a `.png` at two
/// sizes for the others — and this is a circle. It is the same shape the page draws in the middle
/// of itself, which is the point: the thing by the clock and the thing in the window are the same
/// object.
pub fn orb(side: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((side * side * 4) as usize);
    let middle = (side as f32 - 1.0) / 2.0;
    let edge = middle - 0.5;
    for y in 0..side {
        for x in 0..side {
            let (dx, dy) = (x as f32 - middle, y as f32 - middle);
            let away = (dx * dx + dy * dy).sqrt();
            //: One pixel of softness at the rim. A hard circle at 32 px is a cog.
            let solid = ((edge - away) / 1.5).clamp(0.0, 1.0);
            //: Lit from the upper left, so it reads as a sphere rather than a dot. The same
            //: direction the page lights it from.
            let lit = (1.0 - (dx + dy) / (side as f32)).clamp(0.55, 1.25);
            let level = |base: f32| (base * lit).clamp(0.0, 255.0) as u8;
            pixels.extend_from_slice(&[level(126.0), level(176.0), level(232.0),
                                       (solid * 255.0) as u8]);
        }
    }
    pixels
}

/// Put the page in the window.
///
/// **The one function that names an operating system**, and it has to: every backend but GTK
/// attaches the webview to the window, and GTK attaches it to a box inside the window. There is no
/// arrangement of this that is platform-free, so it is one function rather than a `cfg` in the
/// middle of the loop.
fn attach(window: &Window, url: &str, held: &mut wry::WebContext) -> Result<wry::WebView, String> {
    let building = WebViewBuilder::new_with_web_context(held)
        .with_url(url)
        //: **Nothing opens a second window.** A link that asked for one would be a browser, and
        //: the page has no links out; refusing is the honest answer rather than a surprise.
        .with_new_window_req_handler(|_, _| wry::NewWindowResponse::Deny);

    #[cfg(any(target_os = "windows", target_os = "macos"))]
    let view = building.build(window);
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let view = {
        use tao::platform::unix::WindowExtUnix;
        use wry::WebViewBuilderExtUnix;
        let Some(box_inside) = window.default_vbox() else {
            return Err("창 안에 붙일 자리가 없습니다".into());
        };
        building.build_gtk(box_inside)
    };
    view.map_err(|e| format!("화면을 만들지 못했습니다 ({e})"))
}

/// What the loop is being told to do, from outside it.
pub enum Say {
    /// Bring the window up — a second copy of the program was started and handed the job over.
    Show,
    /// The server has something to put in the tooltip.
    Tooltip(String),
    /// Close everything.
    Quit,
}

/// Open the window and stay in it until somebody quits.
///
/// **This does not return.** `tao` ends the process itself when the loop stops, which skips every
/// destructor — the same reason the recogniser and the voice are held by a job object rather than
/// by a `Drop`, and why that was the right call. Anything that must happen on the way out has to
/// be the operating system's promise, not this program's.
///
/// `orders` is optional and is how another thread asks for the window: a second launch that found
/// argmind already running sends [`Say::Show`] rather than starting a second one.
pub fn run(shell: Shell, orders: Option<Receiver<Say>>) -> Result<(), String> {
    if !is_loopback(&shell.url) {
        return Err(format!("{} 은(는) 이 컴퓨터가 아닙니다 — 열지 않습니다", shell.url));
    }

    let events = EventLoopBuilder::new().build();
    let window = WindowBuilder::new()
        .with_title(&shell.title)
        .with_inner_size(tao::dpi::LogicalSize::new(shell.wide, shell.tall))
        .with_visible(!shell.hidden)
        .with_focused(shell.focused && !shell.hidden)
        .build(&events)
        .map_err(|e| format!("창을 만들지 못했습니다 ({e})"))?;
    //: Held for as long as the webview is, which is why it is a binding and not an argument.
    let mut context = wry::WebContext::new(shell.data.clone());
    let _view = attach(&window, &shell.url, &mut context)?;

    const SIDE: u32 = 32;
    let mut show_id = None;
    let mut quit_id = None;
    //: **Held for as long as the loop runs, and dropped before it ends.** An icon that vanishes
    //: while the program is still running is the failure a tray exists to prevent; one that stays
    //: after the program has gone is the other failure, and it is the one this file has to work
    //: for — see `LoopDestroyed` below.
    let mut tray = None;
    if shell.tray {
        let icon = Icon::from_rgba(orb(SIDE), SIDE, SIDE)
            .map_err(|e| format!("아이콘을 만들지 못했습니다 ({e})"))?;
        let menu = Menu::new();
        let show = MenuItem::new("보이기", true, None);
        let quit = MenuItem::new("끝내기", true, None);
        menu.append(&show).map_err(|e| format!("{e}"))?;
        menu.append(&quit).map_err(|e| format!("{e}"))?;
        show_id = Some(show.id().clone());
        quit_id = Some(quit.id().clone());
        tray = Some(TrayIconBuilder::new()
            .with_menu(Box::new(menu))
            .with_tooltip(&shell.tooltip)
            .with_icon(icon)
            .build()
            .map_err(|e| format!("트레이 아이콘을 만들지 못했습니다 ({e})"))?);
    }
    let clicks = TrayIconEvent::receiver();
    let menus = MenuEvent::receiver();

    events.run(move |event, _, flow| {
        //: **Woken on a timer rather than only on input.** The tray's events and the orders from
        //: other threads arrive on channels this loop has to look at, and a loop that sleeps until
        //: the mouse moves would answer a second launch a minute late.
        *flow = ControlFlow::WaitUntil(
            std::time::Instant::now() + std::time::Duration::from_millis(100));

        let mut raise = false;
        let mut leave = false;

        match event {
            //: Stated so the reason is in the code: closing is hiding. The program is still there,
            //: it is still listening, and the icon by the clock is what says so.
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } if shell.tray =>
                window.set_visible(false),
            //: **With no tray there is nothing to come back from**, so the close button is the
            //: close button. A window that hid itself with no icon anywhere would be a process
            //: nobody could see and nobody could stop.
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => leave = true,
            //: **The icon is taken down here, by hand.** `tao` ends the process when the loop
            //: stops and a destructor does not run — Windows then leaves the icon by the clock
            //: until somebody hovers over it, which is a worse last impression than never having
            //: drawn one.
            Event::LoopDestroyed => { tray.take(); }
            Event::NewEvents(StartCause::Init) => {}
            _ => {}
        }

        while let Ok(click) = clicks.try_recv() {
            //: Left button, on the way up. A press that is still held is somebody deciding.
            if let TrayIconEvent::Click { button: tray_icon::MouseButton::Left,
                                          button_state: tray_icon::MouseButtonState::Up, .. } = click {
                raise = true;
            }
        }
        while let Ok(chosen) = menus.try_recv() {
            if Some(&chosen.id) == show_id.as_ref() {
                raise = true;
            } else if Some(&chosen.id) == quit_id.as_ref() {
                leave = true;
            }
        }
        if let Some(orders) = orders.as_ref() {
            while let Ok(order) = orders.try_recv() {
                match order {
                    Say::Show => raise = true,
                    Say::Tooltip(line) => {
                        if let Some(icon) = tray.as_ref() {
                            let _ = icon.set_tooltip(Some(line));
                        }
                    }
                    Say::Quit => leave = true,
                }
            }
        }

        if raise {
            window.set_visible(true);
            //: **Both, and in this order.** A minimised window that is only `set_visible` comes
            //: back where it was, still minimised, and looks like a click that did nothing.
            window.set_minimized(false);
            window.set_focus();
        }
        if leave {
            *flow = ControlFlow::Exit;
        }
    });
}
