//! The window, **without opening one.**
//!
//! Almost everything here happens inside an event loop that owns the thread and ends the process
//! when it stops, so a test cannot run it and come back. What a suite can check is the part that
//! is decided before the loop starts — where it is allowed to point, what it draws for an icon,
//! and whether the defaults are the ones written down.
//!
//! The rest is exercised by hand, `cargo run -p argmind-window --example peek`, and the evidence
//! that it works is in `docs/findings/the-window.md`: the page was fetched and its JavaScript ran.
//!
//! Needs no display, no network and no webview runtime.

use argmind_window::{is_loopback, orb, Shell};

/// **It points at this machine or it does not open.**
///
/// There is no address bar in this window and there should be no way to arrive anywhere else. A
/// webview that can be handed any URL is a browser, which is the thing this replaced.
#[test]
fn only_this_machine_is_a_place_this_window_will_go() {
    for mine in ["http://127.0.0.1:7331/", "http://127.0.0.1:7331", "http://localhost:7331/x",
                 "https://127.0.0.1/", "http://[::1]:7331/"] {
        assert!(is_loopback(mine), "{mine} is this machine");
    }
    for elsewhere in ["http://example.com/", "https://127.0.0.1.example.com/",
                      "http://127.0.0.2:7331/", "file:///C:/x.html", "javascript:alert(1)",
                      "http://evil/?x=127.0.0.1", "", "127.0.0.1:7331"] {
        assert!(!is_loopback(elsewhere), "{elsewhere} is not this machine and was allowed");
    }
    //: **The one that would slip past a `starts_with`.** A host whose name merely begins with the
    //: loopback address belongs to whoever registered it.
    assert!(!is_loopback("http://127.0.0.1.attacker.test/page"));
    //: And a credential-shaped prefix, which is how a URL is made to read as one host and resolve
    //: as another.
    assert!(!is_loopback("http://127.0.0.1@example.com/"));
    //: **This is the one the `@` check is actually for**, and the first version of this test did
    //: not have it: with a bracketed address the reader stops at the `]` and never sees what
    //: follows, so `[::1]@evil` would read as loopback and resolve as `evil`. Mutation testing
    //: found it — removing the check broke nothing, which meant nothing was checking it.
    assert!(!is_loopback("http://[::1]@example.com/"));
    assert!(!is_loopback("http://[::1]@evil:7331/"));
}

/// The icon is a filled circle with a soft rim, and it is the right size.
///
/// **Drawn rather than shipped**, so there is no `.ico` to keep in step with a `.png`. What could
/// go wrong silently is the buffer length: the library takes width, height and bytes separately
/// and a mismatch is an icon that is skewed or absent rather than an error.
#[test]
fn the_icon_is_four_bytes_a_pixel_and_a_circle() {
    for side in [16u32, 32, 64] {
        let pixels = orb(side);
        assert_eq!(pixels.len(), (side * side * 4) as usize, "side {side}");

        let alpha = |x: u32, y: u32| pixels[((y * side + x) * 4 + 3) as usize];
        //: Solid in the middle, empty in the corners. A square icon is what a wrong radius makes.
        assert_eq!(alpha(side / 2, side / 2), 255, "the middle is not filled at side {side}");
        assert_eq!(alpha(0, 0), 0, "the corner is not clear at side {side}");
        assert_eq!(alpha(side - 1, side - 1), 0, "the far corner is not clear at side {side}");
        //: And it is lit from one side rather than flat, which is what makes it read as a sphere.
        let bright = |x: u32, y: u32| pixels[((y * side + x) * 4) as usize] as u32
            + pixels[((y * side + x) * 4 + 1) as usize] as u32;
        let (near, far) = (side / 3, side * 2 / 3);
        assert!(bright(near, near) > bright(far, far),
                "the upper left is not the lit side at side {side}");
    }
}

/// **The defaults are the ones the window is described by**, and two of them are safety rather
/// than taste.
#[test]
fn what_it_does_when_nobody_says_otherwise() {
    let plain = Shell::default();
    //: Visible and focused: somebody double-clicked the program.
    assert!(plain.focused);
    assert!(!plain.hidden);
    //: With an icon, because a window that hides itself and leaves nothing behind is a process
    //: nobody can see and nobody can stop.
    assert!(plain.tray);
    //: **No URL by default.** An empty one fails the loopback check, so a caller that forgot to
    //: say where it is going gets a refusal rather than a blank window.
    assert!(!is_loopback(&plain.url));
    //: Taller than wide: a conversation with a sphere over it.
    assert!(plain.tall > plain.wide);
    //: **And nowhere to put its files until told.** Left alone the library writes beside the
    //: executable, which fails from `Program Files`; the caller names the folder because naming it
    //: needs to know the operating system and this crate does not.
    assert!(plain.data.is_none());
}
