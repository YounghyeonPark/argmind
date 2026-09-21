//! Is it the same person speaking twice?
//!
//! ```text
//!   voicing             4 calls per arm
//!   voicing 8           8
//! ```
//!
//! **The owner's report, 2026-09-21: 음성이 매번 바뀜.** The neural talker is
//! `qwen-talker-0.6b-base-Q8_0` — a *base* model, which in this server means it has no speakers at
//! all. `GET /v1/audio/voices` answers `{"voices":[]}` and every name offered to `voice` comes back
//! `400 unknown voice`, including the `default` that is recorded in the client's error-path comment.
//! What the model has instead of speakers is a sampler, and the timbre is whatever that sampler
//! drew. `tts-server --help` offers no `--voice` and no `--seed`; the *request* takes `seed`, and
//! its default is `-1`, which is the server's word for *choose for me*.
//!
//! ## The arms, and the third is why the second means anything
//!
//! | arm | seed | what it shows |
//! |---|---|---|
//! | **고르지 않음** | `-1` every call | what shipped until today |
//! | **고정** | one seed every call | whether pinning works |
//! | **씨앗마다** | a different seed each call | whether the seed is what chooses |
//!
//! **Without the third arm, `고정` is unreadable.** *Four identical answers* is equally consistent
//! with *the seed pinned the voice* and with *this server is deterministic and the owner was
//! hearing something else* — and a uniformly tidy result is this project's standing reason to
//! suspect the instrument. `씨앗마다` has to come back **all different** or the other two arms say
//! nothing.
//!
//! Then the same pinned-and-loose pair again over the **streamed** path, because that is the one a
//! reply goes through. Everything else here calls `speak`, which nobody hears.
//!
//! ## What this was: a confirmation, not a prediction
//!
//! Said plainly because the standards here require it. The arms were run at a shell first and this
//! file was written afterwards to keep them; the numbers in the doc comment above `chosen::VOICE`
//! are that run. So it is a harness that reproduces a finding rather than one that made it, and the
//! only pre-registered thing about it is the `씨앗마다` control, which was written down as the
//! condition that would withdraw the reading before it was run.
//!
//! **Makes no sound.** Both paths hand back samples; nothing here opens a loudspeaker.

use std::collections::BTreeSet;

use argmind_env_qwentts::QWENTTS;

/// The alias the server was started under. Named here rather than reaching into `argmind-serve`,
/// which is the application and is not something a harness should depend on.
const MODEL: &str = "qwen3-tts";

/// The seed the `고정` arms hold.
///
/// **Deliberately not `chosen::A_VOICE`.** The question is whether holding a seed holds the voice,
/// and *which* seed answers that is immaterial — copying the shipped default here would be a second
/// place that has to be kept in step with the first, to measure something that does not depend on
/// it. What ships is the owner's to change from the page; this is not.
const HELD: i64 = 7;

/// One sentence with something for a voice to do in it — a greeting, a clause and a question.
const SAID: &str = "안녕하세요. 지금 보고 있는 창에서 도와드릴 일이 있으면 말씀해 주세요.";

/// A fingerprint of the audio rather than of the file. **Bytes would also compare the WAV header**,
/// which is forty-four bytes that are the same whatever was said, so a hash over the samples is the
/// one that cannot pass by accident — and it is the only fingerprint the two paths can share, since
/// the streamed one has no header at all.
fn fingerprint(samples: &[f32]) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    for s in samples {
        hash.update(s.to_le_bytes());
    }
    format!("{:x}", hash.finalize())[..10].to_string()
}

/// One utterance, by whichever path.
///
/// **Both paths, because only one of them ships.** `speak_streaming` is what a reply goes through
/// and `speak` is what every measurement goes through. They build their request in one place
/// precisely so that a seed cannot reach one and miss the other — `argmind-env-qwentts`'s
/// `nothing_else_writes_a_request_body` is the test that keeps it so. This is the other half of
/// that: the test says the two bodies are the same, and this says the server does the same thing
/// with them.
fn once(seed: i64, streamed: bool) -> Result<(Vec<f32>, u32), String> {
    if !streamed {
        return argmind_env_qwentts::speak(QWENTTS, MODEL, SAID, seed);
    }
    let mut all = Vec::new();
    argmind_env_qwentts::speak_streaming(QWENTTS, MODEL, SAID, seed, &mut |piece| {
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
        match once(*seed, streamed) {
            Ok((samples, hz)) => {
                let mark = fingerprint(&samples);
                seen.insert(mark.clone());
                //: **The length is printed beside the hash** because it is the part a person can
                //: check against their own ears: two different draws of this sentence differed by
                //: 21% in how long they took to say it.
                println!("    {:>2}  seed {:>5}  {}  {:.2} 초", i + 1, seed, mark,
                         samples.len() as f64 / hz as f64);
            }
            Err(why) => println!("    {:>2}  seed {:>5}  실패: {why}", i + 1, seed),
        }
    }
    println!("    → 서로 다른 목소리 {} 개 / {} 번", seen.len(), seeds.len());
    seen.len()
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(4);
    if !argmind_env_qwentts::QwenVoice::answering(QWENTTS) {
        println!("  {QWENTTS} 에 목소리 서버가 없습니다 — argmind-app 을 띄우거나 설정에서 \
목소리 서버를 빌드하세요");
        return;
    }
    println!("  같은 문장 {n} 번씩. 소리는 내지 않습니다.");
    println!("  \"{SAID}\"");

    let loose = arm("고르지 않음 (seed = -1, 오늘까지 배송되던 것)", &vec![-1; n], false);
    let pinned = arm(&format!("고정 (seed = {HELD})"), &vec![HELD; n], false);
    //: Seeds that are not the pinned one, so this arm cannot accidentally repeat it.
    let spread: Vec<i64> = (0..n as i64).map(|i| HELD + 1 + i).collect();
    let varied = arm("씨앗마다 다르게 (대조군)", &spread, false);

    //: **And the path a person actually hears.** Everything above goes through `speak`, which is
    //: not what a reply uses; a fix that held only there would be a fix for the instrument.
    let flow_loose = arm("고르지 않음, 스트리밍 (실제로 들리는 쪽)", &vec![-1; n], true);
    let flow = arm(&format!("고정, 스트리밍 (seed = {HELD})"), &vec![HELD; n], true);

    println!();
    println!("  ── 읽기 ──");
    if varied < n {
        //: **The control failed, so nothing else is reported as a finding.** Two different seeds
        //: that produce the same audio mean the hash is not measuring the voice, and then `고정`
        //: reading 1 is an artefact rather than a result.
        println!("  대조군이 {varied}/{n} 입니다 — 씨앗이 목소리를 고르지 않거나 지문이 목소리를 \
보고 있지 않습니다. 다른 줄은 읽지 마세요.");
        return;
    }
    println!("  씨앗마다 다른 목소리가 나옵니다 ({varied}/{n}) — 씨앗이 화자를 고릅니다.");
    println!("  파일 경로:      고르지 않으면 {loose}/{n} 가지, 고정하면 {pinned}/{n} 가지.");
    println!("  스트리밍 경로:  고르지 않으면 {flow_loose}/{n} 가지, 고정하면 {flow}/{n} 가지.");
    if pinned == 1 && flow == 1 && loose > 1 && flow_loose > 1 {
        println!("  → 두 경로 모두 씨앗 하나로 고정됩니다. argmind 는 이제 항상 하나를 보냅니다.");
    } else if pinned == 1 && flow > 1 {
        //: The asymmetry the client's one-body rule exists to prevent, named so that a run which
        //: found it would be read correctly rather than as a flaky stream.
        println!("  → 파일 경로만 고정되고 배송 경로는 흔들립니다 ({flow}/{n}) — 스트리밍 요청이 \
씨앗을 싣지 않고 있습니다. 들리는 목소리는 고정되지 않았습니다.");
    } else if pinned > 1 || flow > 1 {
        println!("  → 씨앗을 고정해도 목소리가 흔들립니다 — 설정만으로는 해결되지 않습니다.");
    } else {
        println!("  → 고르지 않아도 흔들리지 않았습니다 — 이 문장에서는 재현되지 않았으니 더 긴 \
문장이나 더 많은 반복이 필요합니다.");
    }
}
