//! Operate one widget through the accessibility layer, and check it actually moved.
//!
//!     cargo run -p argmind-env-linux --example touch            # dry: find it, do not press it
//!     cargo run -p argmind-env-linux --example touch -- --arm   # press it
//!
//! **Presses a real button in whatever window is active**, so `--arm` is a thing to type on
//! purpose. It picks a toggle button, because a toggle is the only control that can be asked
//! afterwards whether it changed — a push button that worked and a push button that did nothing
//! look identical from outside.

#[cfg(target_os = "linux")]
fn main() {
    use argmind_core::contract::{Eyes, Hands as _};
    use argmind_env_linux::{Hands, Sight, Watcher};

    let armed = std::env::args().any(|a| a == "--arm");
    println!();
    let mut eyes = match Sight::new(Watcher::default()) {
        Ok(eyes) => eyes,
        Err(why) => { println!("  눈을 뜨지 못했습니다: {why}"); std::process::exit(1) }
    };
    println!("  보고 있는 창: {:?}", eyes.window().1);

    let widgets = eyes.foreground();
    //: A toggle, because it is the one control whose state can be read back.
    let Some(target) = widgets.iter()
        .find(|w| w.role == "toggle button" && !w.name.trim().is_empty())
        .cloned()
    else {
        println!("  누를 만한 토글 버튼이 없습니다 ({}개 중)", widgets.len());
        return;
    };
    println!("  고른 것: {:?} ({}) {:?}", target.name, target.role, target.rect);

    let mut hands = if armed { Hands::unrestricted().arm() } else { Hands::unrestricted() };
    println!("  hands live: {}", hands.live());

    //: **What it looked like before**, so afterwards is a comparison rather than a hope.
    let before = argmind_env_linux::state_of(&target);
    println!("  누르기 전 checked: {before:?}");

    hands.click(&target);
    println!("  결과: {:?}", hands.done.last());

    std::thread::sleep(std::time::Duration::from_millis(400));
    let after = argmind_env_linux::state_of(&target);
    println!("  누른 뒤 checked: {after:?}");

    match (armed, before, after) {
        (true, Some(a), Some(b)) if a != b => println!("\n  바뀌었습니다 — 진짜로 눌렸습니다."),
        (true, Some(_), Some(_)) => println!("\n  안 바뀌었습니다 — 눌렸다고 했지만 움직이지 않았습니다."),
        (false, _, _) => println!("\n  마른 상태였습니다. --arm 을 붙이면 진짜로 누릅니다."),
        _ => println!("\n  상태를 읽지 못했습니다."),
    }
    println!();
}

#[cfg(not(target_os = "linux"))]
fn main() {
    println!("  이 예제는 Linux에서만 의미가 있습니다.");
}
