//! Say one Korean sentence with the neural voice, and interrupt a second one. **Makes a sound.**
//!
//!     cargo run -p argmind-env-qwentts --example hello
use argmind_voice::Speaker;
use std::time::Instant;

fn main() {
    let model = std::env::args().nth(1)
        .unwrap_or_else(|| "qwen-talker-0.6b-base-Q8_0.gguf".into());
    if !argmind_env_qwentts::QwenVoice::answering(argmind_env_qwentts::QWENTTS) {
        println!("  {} 에 서버가 없습니다", argmind_env_qwentts::QWENTTS);
        return;
    }
    let mut voice = argmind_env_qwentts::QwenVoice::new(&model);

    let asked = Instant::now();
    voice.say("안녕하세요. 저는 argmind입니다. 지금 새 목소리로 말하고 있습니다.");
    println!("  say() 가 돌아오기까지: {:.0} ms", asked.elapsed().as_secs_f64() * 1000.0);
    while !argmind_voice::Speaker::speaking(&voice) {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let mut first = None;
    while argmind_voice::Speaker::speaking(&voice) {
        if first.is_none() && asked.elapsed().as_secs_f64() > 0.0 {
            first = Some(asked.elapsed().as_secs_f64());
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    println!("  말이 끝나기까지: {:.1} s", asked.elapsed().as_secs_f64());

    println!("\n  긴 문장을 말하다가 1.5초 뒤에 끊습니다.");
    let began = Instant::now();
    voice.say("이번에는 훨씬 긴 문장입니다. 중간에 끊기는지 확인하려고 여러 절을 이어서 말하고 \
               있으며, 여기까지 들리면 안 됩니다.");
    std::thread::sleep(std::time::Duration::from_millis(1500));
    //: **Asked before the stop, not after.** The first version of this printed `speaking()` on the
    //: line below the `stop()` call and reported `false` every time — a test of the order of two
    //: statements rather than of the thing.
    println!("  1.5 s 시점: 말하는 중? {}", argmind_voice::Speaker::speaking(&voice));
    voice.stop();
    println!("  stop() 호출: {:.1} s", began.elapsed().as_secs_f64());
    std::thread::sleep(std::time::Duration::from_millis(600));
    println!("  0.6초 뒤: 말하는 중? {}", argmind_voice::Speaker::speaking(&voice));
    voice.release();
    println!("  끝\n");
}
