//! The same Korean, in both voices, written to two files. **Makes no sound.**
//!
//!     cargo run -p argmind-serve --example voices
//!
//! The owner's verdict on `Microsoft Heami Desktop` was 어색함, and the neural voice was built to
//! answer that. Judging it is theirs to do and nobody else's — so this renders the same sentences
//! through both and writes them side by side, rather than playing anything at a machine the person
//! may not be sitting at.
//!
//! **Not a measurement of quality.** Nothing here scores anything; it produces two files. What it
//! does measure, because the numbers are free, is how long each took and how much audio came out.

use std::path::PathBuf;
use std::time::Instant;

/// What the agent actually sounds like saying — a reply, a number, and a question.
///
/// Not a tongue-twister and not a poem. The comparison that matters is the one the owner will
/// hear every day, so these are the shapes argmind really produces: an acknowledgement, a sentence
/// with a model name and a digit in it, and an offer.
const LINES: [&str; 3] = [
    "네, 지우개를 눌렀습니다.",
    "qwen3:8b 모델로 답하고 있습니다. 화면에서 버튼 세 개를 찾았습니다.",
    "글꼴 목록을 열까요?",
];

fn shed() -> PathBuf {
    let at = std::env::var("LOCALAPPDATA").map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    at.join("argmind").join("voices")
}

fn main() {
    let said = LINES.join(" ");
    let where_to = shed();
    let _ = std::fs::create_dir_all(&where_to);
    println!("\n  말할 것: {said}\n");

    // ---- the one Windows ships ------------------------------------------------
    match argmind_env_windows::SapiSpeaker::new() {
        Ok(mut voice) => {
            let korean = voice.has_korean();
            println!("  Windows 음성: {}", if voice.korean_is_old() {
                "Heami Desktop (오래된 것)"
            } else if korean {
                "한국어 있음"
            } else {
                "한국어 없음"
            });
            let began = Instant::now();
            match voice.synthesise(&said) {
                Ok(pcm) => {
                    //: **Raw PCM, not a WAV**, and `synthesise` says so — *return the PCM*. It
                    //: exists so numbers can be taken without making a sound, and a byte count
                    //: needs no header. The first version of this example wrote those bytes
                    //: straight to a `.wav` and produced a file no player would open: the audio
                    //: was fine and the container was missing.
                    //:
                    //: 16 kHz, mono, sixteen bits — `speaker.rs`'s `wave_format`, which is what
                    //: the stream was told to produce.
                    let at = where_to.join("1-windows.wav");
                    let _ = std::fs::write(&at, wav_around(&pcm, 16_000));
                    let seconds = pcm.len() as f64 / 32_000.0;
                    println!("    {:.0} ms, {:.1}초치 -> {}",
                             began.elapsed().as_secs_f64() * 1000.0, seconds, at.display());
                }
                Err(e) => println!("    소리를 만들지 못했습니다: {e}"),
            }
        }
        Err(e) => println!("  Windows 음성을 열지 못했습니다: {e}"),
    }

    // ---- the one argmind builds -----------------------------------------------
    if !argmind_serve::speaker::answering() {
        println!("\n  목소리 서버가 꺼져 있습니다 — 띄웁니다 (모델을 그래픽카드에 올리는 데 몇십 초).");
        if let Err(why) = argmind_serve::speaker::serve(&mut |line| println!("    {line}")) {
            println!("    띄우지 못했습니다: {why}");
            println!("\n  Windows 쪽만 만들어졌습니다.\n");
            return;
        }
    }
    println!("\n  신경망 음성: {} @ {}",
             argmind_serve::speaker::VOICE, argmind_serve::speaker::QWENTTS_HOST);
    let began = Instant::now();
    //: The WAV the server sends, written out whole rather than decoded and re-encoded — fewer
    //: chances to change what it sounds like on the way to the file.
    match argmind_env_qwentts::speak(argmind_serve::speaker::QWENTTS_HOST,
                                     argmind_serve::speaker::VOICE, &said) {
        Ok((samples, hz)) => {
            let at = where_to.join("2-neural.wav");
            let pcm: Vec<u8> = samples.iter()
                .flat_map(|s| ((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes())
                .collect();
            let _ = std::fs::write(&at, wav_around(&pcm, hz));
            let took = began.elapsed().as_secs_f64();
            let seconds = samples.len() as f64 / hz as f64;
            println!("    {:.0} ms 만에 {:.1}초치 — RTF {:.2}  -> {}",
                     took * 1000.0, seconds, took / seconds, at.display());
        }
        Err(why) => println!("    소리를 만들지 못했습니다: {why}"),
    }
    println!("\n  두 파일을 들어보시고 어느 쪽이 나은지 알려주세요.\n");
}

/// Forty-four bytes of header around mono 16-bit PCM, because that is what every player agrees
/// about — and because neither voice hands over one.
///
/// The synthesiser sends a WAV over HTTP and the client has already thrown the header away to give
/// back samples; SAPI never had one. Both end up here.
fn wav_around(pcm: &[u8], hz: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(pcm.len() + 44);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + pcm.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&hz.to_le_bytes());
    out.extend_from_slice(&(hz * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
    out.extend_from_slice(&pcm);
    out
}
