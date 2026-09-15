//! `serve` — the application with a console, for running it from a terminal.
//!
//! The whole server is [`argmind_serve::app`]; this is the four lines that start it and the window
//! that shows its output. `argmind-app` is the same server with an icon by the clock instead.

fn main() {
    let how = argmind_serve::app::How::from_args();
    println!("\n  argmind — {}", argmind_serve::app::URL);
    println!("  {}\n", how.line());
    argmind_serve::app::run(how);
}
