//! Does argmind start an Ollama that is installed and stopped?
//!
//!     cargo run -p argmind-serve --example starting-ollama
//!
//! **Reports, and only starts if there is nothing answering.** Stopping Ollama is left to whoever
//! runs this: it is somebody's own program and a harness that took it down would be doing the one
//! thing this module exists to say argmind does not do.
fn main() {
    let here = argmind_serve::ollama::here();
    let awake = argmind_serve::ollama::awake();
    println!("  설치되어 있는가: {here}");
    println!("  응답하는가:      {awake}");
    if awake {
        println!("\n  이미 응답 중이라 아무것도 하지 않습니다. 멈춘 상태에서 다시 실행해 보세요.");
        return;
    }
    let began = std::time::Instant::now();
    match argmind_serve::ollama::serve(&mut |line| println!("    {line}")) {
        Ok(()) => println!("\n  {:.1} 초 만에 응답합니다", began.elapsed().as_secs_f64()),
        Err(why) => println!("\n  실패: {why}"),
    }
}
