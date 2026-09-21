//! Is it the same person speaking twice?
//!
//! ```text
//!   voicing             4 calls per arm
//!   voicing 8           8
//! ```
//!
//! **The owner's report, 2026-09-21: 음성이 매번 바뀜.** The talker is
//! `qwen-talker-0.6b-base-Q8_0` — a *base* model, which in this server means it ships with no
//! speakers at all. `GET /v1/audio/voices` starts empty and every built-in name offered to `voice`
//! comes back `400 unknown voice`, including the `default` recorded in the client's error-path
//! comment. What it has instead of speakers is a sampler, and the timbre is whatever that sampler
//! drew. `tts-server --help` offers no `--voice` and no `--seed`; the *request* takes `seed`, whose
//! default is `-1`, the server's word for *choose for me*.
//!
//! ## Part one: a seed fixes a recording
//!
//! | arm | seed | what it shows |
//! |---|---|---|
//! | **고르지 않음** | `-1` every call | what shipped until today |
//! | **고정** | one seed every call | whether pinning works |
//! | **씨앗마다** | a different seed each call | whether the seed is what chooses |
//!
//! **Without the third arm, `고정` is unreadable.** *Four identical answers* is equally consistent
//! with *the seed pinned the voice* and with *this server is deterministic and the owner was
//! hearing something else*. Then the same pair again over the **streamed** path, because that is
//! the one a reply goes through and `speak` is the one every measurement goes through.
//!
//! ## Part two: and a recording is not a person
//!
//! **This is the part that matters and it was nearly missed.** Everything above says one sentence
//! twice. Every reply is a *different* sentence, and across four of them a seed held to 119 Hz of
//! itself while different seeds on one sentence spanned 137 Hz — 1.2x, which is no ratio. Long-term
//! average spectra agreed with no notion of pitch in them: 0.821 alike within a seed against 0.813
//! across seeds, control at 1.000. So part one is true and does not fix the complaint.
//!
//! What does is a **cloned voice**: one reference drawn at a fixed seed, registered, and then used
//! for everything. Part two runs the two against each other over the same sentences.
//!
//! **Makes no sound.** Both paths hand back samples; nothing here opens a loudspeaker.

use std::collections::BTreeSet;

use argmind_env_qwentts::{Voice, QWENTTS};

/// The alias the server was started under. Named here rather than reaching into `argmind-serve`,
/// which is the application and is not something a harness should depend on.
const MODEL: &str = "qwen3-tts";

/// The seed the `고정` arms hold.
///
/// **Deliberately not `chosen::A_VOICE`.** The question is whether holding a seed holds anything,
/// and *which* seed answers that is immaterial — copying a shipped default here would be a second
/// place to keep in step, to measure something that does not depend on it.
const HELD: i64 = 7;

/// One sentence with something for a voice to do in it — a greeting, a clause and a question.
const SAID: &str = "안녕하세요. 지금 보고 있는 창에서 도와드릴 일이 있으면 말씀해 주세요.";

/// Part two says each of these once per arm. **Different lengths and different endings**, because
/// the thing being tested is whether the speaker survives the sentence changing.
const SENTENCES: [&str; 5] = [
    "안녕하세요.",
    "오늘 하루는 어떠셨나요?",
    "지금 보고 있는 창에서 도와드릴 일이 있으면 말씀해 주세요.",
    "저장하지 않은 변경 내용이 있는 대화상자가 떠 있습니다.",
    "지우개 버튼을 눌러 드릴까요? 확인을 누르시면 진행합니다.",
];

/// What a reference recording says. The same sentence `argmind-serve` uses, by value rather than by
/// import — see `HELD`.
const REFERENCE: &str = "안녕하세요. 만나서 반갑습니다. 오늘은 무엇을 도와드릴까요?";

/// A fingerprint of the audio rather than of the file. **Bytes would also compare the WAV header**,
/// forty-four bytes that are the same whatever was said, and it is the only fingerprint the two
/// request paths can share since the streamed one has no header at all.
fn fingerprint(samples: &[f32]) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    for s in samples {
        hash.update(s.to_le_bytes());
    }
    format!("{:x}", hash.finalize())[..10].to_string()
}

/// Median f0 over the frames loud enough to be voiced, in Hz, or `None` if none are.
///
/// **Autocorrelation with octave correction.** A true period of T also correlates at 2T and 3T, so
/// taking the biggest peak in a wide window lands on whichever happens to be largest — and the
/// first version of this measurement, in Python, produced pairs like 117/245 and 267/122 that are
/// almost exactly two apart. That looked like the model changing speaker and was the estimator
/// halving. Walking down to the shortest sub-multiple that is still nearly as strong fixes it. It
/// changed the reading not at all, which is the only reason the reading is believed.
fn pitch(samples: &[f32], hz: u32) -> Option<f64> {
    let win = (hz as f64 * 0.04) as usize;
    if samples.len() < win * 2 {
        return None;
    }
    let frames: Vec<&[f32]> = samples.chunks_exact(win).collect();
    let loudest = frames.iter()
        .map(|f| (f.iter().map(|s| s * s).sum::<f32>() / win as f32).sqrt())
        .fold(0.0f32, f32::max);
    let mut found: Vec<f64> = Vec::new();
    for frame in frames {
        let power = (frame.iter().map(|s| s * s).sum::<f32>() / win as f32).sqrt();
        //: Silence and breath are not pitch.
        if power < loudest * 0.15 {
            continue;
        }
        let mean = frame.iter().sum::<f32>() / win as f32;
        let centred: Vec<f32> = frame.iter().map(|s| s - mean).collect();
        let base: f32 = centred.iter().map(|s| s * s).sum();
        if base <= 0.0 {
            continue;
        }
        let (lo, hi) = ((hz / 400) as usize, ((hz / 60) as usize).min(win - 1));
        if hi <= lo {
            continue;
        }
        let at = |lag: usize| -> f32 {
            centred[..win - lag].iter().zip(&centred[lag..]).map(|(a, b)| a * b).sum::<f32>() / base
        };
        let mut best = lo;
        let mut peak = at(lo);
        for lag in lo + 1..hi {
            let r = at(lag);
            if r > peak {
                peak = r;
                best = lag;
            }
        }
        if peak < 0.3 {
            continue;
        }
        for div in [4usize, 3, 2] {
            let candidate = best / div;
            if candidate >= lo && at(candidate) > peak * 0.9 {
                best = candidate;
                break;
            }
        }
        found.push(hz as f64 / best as f64);
    }
    if found.is_empty() {
        return None;
    }
    found.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(found[found.len() / 2])
}

fn once(text: &str, voice: &Voice, streamed: bool) -> Result<(Vec<f32>, u32), String> {
    if !streamed {
        return argmind_env_qwentts::speak(QWENTTS, MODEL, text, voice);
    }
    let mut all = Vec::new();
    argmind_env_qwentts::speak_streaming(QWENTTS, MODEL, text, voice, &mut |piece| {
        all.extend_from_slice(piece);
        true
    })?;
    //: A stream carries no header, so the rate is `HZ` by agreement — the whole-file path is the
    //: one that checks it.
    Ok((all, argmind_env_qwentts::HZ))
}

fn arm(what: &str, seeds: &[i64], streamed: bool) -> usize {
    println!();
    println!("  {what}");
    let mut seen: BTreeSet<String> = BTreeSet::new();
    for (i, seed) in seeds.iter().enumerate() {
        match once(SAID, &Voice::Drawn(*seed), streamed) {
            Ok((samples, hz)) => {
                let mark = fingerprint(&samples);
                seen.insert(mark.clone());
                println!("    {:>2}  seed {:>5}  {}  {:.2} 초", i + 1, seed, mark,
                         samples.len() as f64 / hz as f64);
            }
            Err(why) => println!("    {:>2}  seed {:>5}  실패: {why}", i + 1, seed),
        }
    }
    println!("    → 서로 다른 목소리 {} 개 / {} 번", seen.len(), seeds.len());
    seen.len()
}

/// Say all five sentences as one voice and report how far its pitch moved.
fn across_sentences(what: &str, voice: &Voice) -> Option<f64> {
    println!();
    println!("  {what}");
    let mut got: Vec<f64> = Vec::new();
    for text in SENTENCES {
        match once(text, voice, false) {
            Ok((samples, hz)) => match pitch(&samples, hz) {
                Some(f0) => {
                    println!("    {f0:>5.0} Hz   {text}");
                    got.push(f0);
                }
                None => println!("        무성   {text}"),
            },
            Err(why) => println!("        실패: {why}"),
        }
    }
    if got.len() < 2 {
        return None;
    }
    let (lo, hi) = (got.iter().cloned().fold(f64::MAX, f64::min),
                    got.iter().cloned().fold(f64::MIN, f64::max));
    println!("    → 폭 {:.0} Hz", hi - lo);
    Some(hi - lo)
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(4);
    if !argmind_env_qwentts::QwenVoice::answering(QWENTTS) {
        println!("  {QWENTTS} 에 목소리 서버가 없습니다 — argmind-app 을 띄우거나 설정에서 \
목소리 서버를 빌드하세요");
        return;
    }
    println!("  소리는 내지 않습니다.");
    println!();
    println!("  ══ 1부: 씨앗은 한 문장을 고정하는가 — 같은 문장 {n} 번씩 ══");
    println!("  \"{SAID}\"");

    let loose = arm("고르지 않음 (seed = -1, 처음에 배송되던 것)", &vec![-1; n], false);
    let pinned = arm(&format!("고정 (seed = {HELD})"), &vec![HELD; n], false);
    //: Seeds that are not the pinned one, so this arm cannot accidentally repeat it.
    let spread: Vec<i64> = (0..n as i64).map(|i| HELD + 1 + i).collect();
    let varied = arm("씨앗마다 다르게 (대조군)", &spread, false);
    //: **And the path a person actually hears.** Everything above goes through `speak`, which is
    //: not what a reply uses; a fix that held only there would be a fix for the instrument.
    let flow_loose = arm("고르지 않음, 스트리밍 (실제로 들리는 쪽)", &vec![-1; n], true);
    let flow = arm(&format!("고정, 스트리밍 (seed = {HELD})"), &vec![HELD; n], true);

    println!();
    println!("  ── 1부 읽기 ──");
    if varied < n {
        //: **The control failed, so nothing else is reported as a finding.** Two different seeds
        //: producing the same audio mean the fingerprint is not watching the voice.
        println!("  대조군이 {varied}/{n} 입니다 — 지문이 목소리를 보고 있지 않습니다. \
다른 줄은 읽지 마세요.");
        return;
    }
    println!("  씨앗마다 다른 소리 ({varied}/{n}). 파일 경로 {loose}/{n} → {pinned}/{n}, \
스트리밍 {flow_loose}/{n} → {flow}/{n}.");
    if pinned == 1 && flow == 1 && loose > 1 && flow_loose > 1 {
        println!("  → 두 경로 모두 씨앗 하나로 고정됩니다. **한 문장에 대해서는.**");
    } else {
        println!("  → 씨앗을 고정해도 흔들립니다 — 2부를 읽을 필요가 없습니다.");
        return;
    }

    println!();
    println!("  ══ 2부: 그런데 녹음 하나는 사람 한 명이 아니다 — 문장 {} 개 ══",
             SENTENCES.len());
    let by_seed = across_sentences(&format!("씨앗만 고정 (seed = {HELD})"), &Voice::Drawn(HELD));

    //: The reference is drawn at the same seed, so the two arms start from the same voice and
    //: differ only in whether it was registered.
    print!("\n  참조를 만들어 등록하는 중");
    let cloned = match argmind_env_qwentts::speak_wav(QWENTTS, MODEL, REFERENCE,
                                                     &Voice::Drawn(HELD))
        .and_then(|wav| argmind_env_qwentts::register_voice(QWENTTS, "voicing", &wav, REFERENCE))
    {
        Ok(()) => {
            println!(" — 됨");
            across_sentences("복제된 목소리", &Voice::Cloned("voicing".into()))
        }
        Err(why) => {
            println!(" — 실패: {why}");
            None
        }
    };

    println!();
    println!("  ── 2부 읽기 ──");
    match (by_seed, cloned) {
        (Some(seeded), Some(clone)) => {
            println!("  문장이 바뀔 때 음높이가 움직인 폭:  씨앗 {seeded:.0} Hz,  복제 {clone:.0} Hz");
            if clone * 2.0 < seeded {
                println!("  → 복제는 사람을 붙잡고 씨앗은 붙잡지 못합니다. 답마다 문장이 다르므로, \
말하는 사람을 고정하는 것은 복제뿐입니다.");
            } else {
                println!("  → 복제가 씨앗보다 낫지 않습니다 ({clone:.0} 대 {seeded:.0} Hz). \
이 합성기로는 사람을 붙잡을 수 없습니다.");
            }
        }
        _ => println!("  둘 중 하나를 재지 못했습니다."),
    }
}
