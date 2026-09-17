//! Click a text box and type Korean into it, then read it back out of the widget.
//!
//!     cargo run -p argmind-env-linux --example say            # dry
//!     cargo run -p argmind-env-linux --example say -- --arm   # really type
//!
//! **Puts text into a real window**, so `--arm` is typed on purpose. It reads the result back from
//! the widget rather than trusting the return value: `InsertText` answers `true` whether five
//! characters went in or one, which is how the byte-versus-character mistake stayed invisible.
//!
//! Korean on purpose. ASCII works either way and would have proved nothing.

#[cfg(target_os = "linux")]
fn main() {
    use argmind_core::contract::{Eyes, Hands as _};
    use argmind_env_linux::{text_of, Hands, Sight, Watcher};

    const SAY: &str = "안녕하세요";
    let armed = std::env::args().any(|a| a == "--arm");
    println!();

    let mut eyes = match Sight::new(Watcher::default()) {
        Ok(eyes) => eyes,
        Err(why) => { println!("  눈을 뜨지 못했습니다: {why}"); std::process::exit(1) }
    };
    println!("  보고 있는 창: {:?}", eyes.window().1);

    let widgets = eyes.foreground();
    let Some(box_) = widgets.iter().find(|w| w.role == "text").cloned() else {
        println!("  입력란이 없습니다 ({}개 중)", widgets.len());
        return;
    };
    println!("  입력란: {:?} {:?}", box_.role, box_.rect);
    println!("  넣기 전: {:?}", text_of(&box_));

    let mut hands = if armed { Hands::unrestricted().arm() } else { Hands::unrestricted() };

    hands.click(&box_);
    println!("  누르기: {:?}", hands.done.last());

    hands.type_text(SAY);
    println!("  넣기:   {:?}", hands.done.last());

    std::thread::sleep(std::time::Duration::from_millis(300));
    let after = text_of(&box_);
    println!("  넣은 뒤: {after:?}");

    //: **The whole string, character for character.** A count would pass on `안` five times.
    match (armed, after) {
        (true, Some(got)) if got.contains(SAY) =>
            println!("\n  {SAY:?} 가 통째로 들어갔습니다."),
        (true, Some(got)) =>
            println!("\n  들어갔지만 {SAY:?} 가 아닙니다 — {got:?}"),
        (false, _) => println!("\n  마른 상태였습니다. --arm 을 붙이면 진짜로 넣습니다."),
        (true, None) => println!("\n  읽어내지 못했습니다."),
    }

    //: A named key is the one thing this environment still cannot do, and it says which.
    hands.press("enter");
    println!("  enter:  {:?}\n", hands.done.last());
}

#[cfg(not(target_os = "linux"))]
fn main() {
    println!("  이 예제는 Linux에서만 의미가 있습니다.");
}
