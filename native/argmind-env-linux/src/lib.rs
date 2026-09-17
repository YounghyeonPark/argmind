//! Eyes for Linux. **AT-SPI2 over D-Bus, which is the same bargain UI Automation is.**
//!
//! The rule everything hangs on is that *the deterministic side says WHERE*. On Windows that is UI
//! Automation; here it is the accessibility bus, and it answers the same question in the same
//! shape: a name, a role, and a rectangle in screen pixels. Measured on Ubuntu 26.04 before a line
//! of this was written —
//!
//! ```text
//! toggle button   Menu      (1352, 23, 36, 46)
//! radio button    Page 1    (551, 23, 121, 46)
//! combo box       Left      (36, 300, 121, 34)
//!
//! nodes: 121   named: 63   with a rectangle: 120   buttons: 24
//! ```
//!
//! — which is `Widget { name, role, rect, window }` with nothing missing and nothing invented.
//!
//! ## What is here and what is not
//!
//! **Eyes.** Hands, a voice, a microphone and a yes-or-no question are re-exported from
//! [`argmind_env_none`], where they answer *no* honestly. That is deliberate: a half-written set of
//! stubs in this file would read like work in progress, and re-exporting says plainly which parts
//! of this environment exist.
//!
//! Hands on Linux are a larger question than they look, and it is not a coding one: **X11 lets any
//! client inject input with XTEST, and Wayland does not.** Under Wayland the sanctioned route is
//! `xdg-desktop-portal`'s RemoteDesktop interface, with the person granting a session — so the
//! permission model changes rather than the implementation, and argmind's four-way permission has
//! somewhere to go. Nothing is written until that is decided against a real desktop.
//!
//! ## Turning accessibility on is something argmind does to the whole machine
//!
//! Toolkits publish nothing until an assistive technology says it is listening: `IsEnabled` on
//! `org.a11y.Status`. **Measured** — with it false, `gtk3-widget-factory` offers one node and no
//! tree; with it true, 121. So [`Sight`] sets it, and that is a real side effect on every other
//! application on the desktop, not just the one argmind is looking at.
//!
//! It is the same bargain a screen reader makes and argmind is the same kind of program, but it is
//! worth being explicit: **this is switched on when eyes are opened, not when the program starts**,
//! so a build that never looks at anything never touches it.
//!
//! ## Two things the probe found that an implementation would otherwise get wrong
//!
//! **Off-screen widgets report `(-2147483648, -2147483648, 1, 1)`** — `i32::MIN` — rather than an
//! error or an empty rectangle. Menu items inside a closed combo box do it. Handing those to the
//! core would put widgets at coordinates no screen has, so [`sane`] drops them.
//!
//! **A frame often has no name**, and the window title lives on it anyway through the `Name`
//! property of the application above it or the frame itself. Where there is nothing, the window is
//! reported as empty rather than as a guess.

#[cfg(target_os = "linux")]
mod atspi;

#[cfg(target_os = "linux")]
pub use atspi::{Attending, Plain, Sight, Watcher};

//: **Not written yet, and borrowed rather than faked.** See the note above.
pub use argmind_env_none::{ask, supervise, Clip, Deaf, Mute, Still, WANT_HZ};
pub use argmind_env_none::{foreground_handle_now, window_title};

//: On anything that is not Linux this crate is the empty environment entire, so a `cargo check`
//: from a Mac does not fall over on a D-Bus binding it has no libraries for.
#[cfg(not(target_os = "linux"))]
pub use argmind_env_none::{Attending, Plain, Watcher};
#[cfg(not(target_os = "linux"))]
pub use argmind_env_none::Attending as Sight;
