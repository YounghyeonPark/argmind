//! Read one window and print what is in it, as JSON — the Rust half of the equivalence check.
//!
//! Takes a window handle so both implementations read **the same window**, and reads it without
//! bringing it to the front. With no argument it reads the foreground window.
//!
//! It only ever reads. Nothing here clicks, types or presses.
use argmind_env_windows::WindowsEyes;

fn main() {
    let mut eyes = WindowsEyes::new().expect("UI Automation would not open");
    let handle: isize = std::env::args().nth(1)
        .map(|a| a.parse().expect("the handle must be a number"))
        .unwrap_or_else(|| eyes.foreground_handle());
    let t0 = std::time::Instant::now();
    let widgets = eyes.read(handle);
    let took = t0.elapsed().as_secs_f64();
    let rows: Vec<_> = widgets.iter().map(|w| serde_json::json!({
        "name": w.name, "role": w.role,
        "rect": [w.rect.0, w.rect.1, w.rect.2, w.rect.3], "window": w.window,
    })).collect();
    println!("{}", serde_json::json!({
        "handle": handle, "seconds": took, "vanished": eyes.vanished, "widgets": rows,
    }));
}
