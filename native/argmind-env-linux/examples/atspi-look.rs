//! What the Linux eyes see, and how long they took.
//!
//! **Named `atspi-look` rather than `look`** because `argmind-env-windows` already has a `look` and
//! cargo puts every example in one directory: two of them collide, and cargo says that may stop
//! being a warning.
//!
//!     cargo run -p argmind-env-linux --example atspi-look
//!
//! The same question `argmind look` asks on Windows, against the accessibility bus. **Reads and
//! changes nothing** — except that opening the eyes switches accessibility on for the desktop,
//! which is said out loud below because it is a real side effect on every other application.

#[cfg(target_os = "linux")]
fn main() {
    use argmind_core::contract::Eyes;
    use argmind_env_linux::{Sight, Watcher};

    println!();
    println!("  접근성을 켭니다 — 이 데스크톱의 다른 프로그램에도 영향이 있습니다.");
    let mut eyes = match Sight::new(Watcher::default()) {
        Ok(eyes) => eyes,
        Err(why) => {
            println!("  눈을 뜨지 못했습니다: {why}");
            std::process::exit(1);
        }
    };

    let (handle, title) = eyes.window();
    println!("  보고 있는 창: {title:?}  (handle {handle})");

    let (widgets, took) = eyes.timed_foreground();
    println!("  {}개, {:.0} ms\n", widgets.len(), took * 1000.0);

    //: Named ones first: they are the ones an effect could ever refer to, because an effect carries
    //: a name and a role and never a rectangle.
    let (named, rest): (Vec<_>, Vec<_>) =
        widgets.iter().partition(|w| !w.name.trim().is_empty());
    println!("  {:<22} {:<30} {}", "role", "name", "rect");
    for w in named.iter().take(30) {
        println!("  {:<22} {:<30} {:?}", w.role, w.name.chars().take(28).collect::<String>(),
                 w.rect);
    }
    println!("\n  이름 있는 것 {}개, 이름 없는 것 {}개", named.len(), rest.len());

    //: The hit test, against the middle of the first named widget — the one question the `Eyes`
    //: trait does not carry and the page uses to say what a click would land on.
    if let Some(first) = named.first() {
        let (cx, cy) = first.centre();
        println!("  ({cx}, {cy}) 지점: {:?}", eyes.at_point(cx, cy));
    }
    println!();
}

#[cfg(not(target_os = "linux"))]
fn main() {
    println!("  이 예제는 Linux에서만 의미가 있습니다.");
}
