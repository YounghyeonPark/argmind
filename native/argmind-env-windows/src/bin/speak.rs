//! What speaking costs, measured **without making a sound** — volume 0 and a memory stream.
//!
//! Reproduces the figures the Python side produced through comtypes, so the port can be compared
//! rather than assumed: synthesis time, how long the person would spend listening, and the routing
//! defect that makes the rule necessary.
use argmind_env_windows::SapiSpeaker;

const LINES: &[(&str, &str)] = &[
    ("ko", "복사 버튼을 클릭해 드릴게요."),
    ("ko", "글꼴을 변경하려면 'Font :' 콤보박스를 클릭해 주세요."),
    ("ko", "지금은 화면에 관련된 정보가 없어서 날씨에 대해 말씀드리지 못해요. 도와드릴 수 있는 일이 있으신가요?"),
    ("en", "I can't see that here."),
    ("en", "I'll click the Copy button for you."),
];

fn seconds(bytes: usize) -> f64 {
    bytes as f64 / (16000.0 * 2.0)      //: 16 kHz, 16-bit, mono
}

fn main() {
    let mut sp = SapiSpeaker::new().expect("SAPI would not open");
    sp.set_volume(0);
    println!("  voices: {:?}", sp.voices());
    println!("  a Korean voice is installed: {}\n", sp.has_korean());

    println!("  {:4} {:>5} {:>9} {:>9} {:>8}   text", "", "chars", "synth ms", "heard s", "x real");
    for (lang, text) in LINES {
        let t0 = std::time::Instant::now();
        let pcm = sp.synthesise(text).expect("synthesis");
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let heard = seconds(pcm.len());
        println!("  {:4} {:>5} {:>9.1} {:>9.2} {:>8.0}   {}", lang, text.chars().count(), ms,
                 heard, heard * 1000.0 / ms, &text.chars().take(34).collect::<String>());
    }

    println!("\n  THE WRONG VOICE — Korean handed to the default (English) voice");
    let ko = "복사 버튼을 클릭해 드릴게요.";
    let right = sp.synthesise(ko).expect("ko");
    let wrong = sp.synthesise_forcing_default(ko).expect("forced");
    println!("    Korean voice : {:>7} bytes, {:.2} s", right.len(), seconds(right.len()));
    println!("    default voice: {:>7} bytes, {:.2} s", wrong.len(), seconds(wrong.len()));
    println!("    -> the wrong voice produces {:.0}% of the audio",
             if right.is_empty() { 0.0 } else { wrong.len() as f64 * 100.0 / right.len() as f64 });
    //: **The first use of each voice is cold** and the table above shows it. What a running agent
    //: pays is the steady state, so it is measured separately rather than averaged in.
    println!("\n  WARM — the same sentence again, five times");
    let mut ms = Vec::new();
    for _ in 0..5 {
        let t0 = std::time::Instant::now();
        let _ = sp.synthesise(ko).expect("warm");
        ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("    median {:.1} ms, {:.1}-{:.1}, for 2.87 s of audio -> {:.0}x realtime",
             ms[2], ms[0], ms[4], 2870.0 / ms[2]);

    println!("\n  counts: {:?}", sp.counts);
}
