//! **Is the same face the same print?** The one number face memory rests on, and nobody had it.
//!
//! ```text
//!   cargo run -p argmind-measure --bin recognising --release
//! ```
//!
//! From `native/`. **Stop argmind first** — it holds the camera. The light comes on for about fifteen
//! seconds; nothing is saved, only numbers reach the terminal.
//!
//! ## Why this exists
//!
//! The owner, 2026-09-30: *누구인지 물어봄 … 얼굴을 치웠다 다시 보이면 이름을 또 물어봄.* The roster
//! held **six rows for one person**, and their prints sat at cosine **0.01 to 0.36** against each
//! other — beside a threshold of 0.363. The cause was found by reading OpenCV rather than by guessing:
//! `FaceRecognizerSF::feature` builds its blob with `swapRB = true` from a BGR image, so SFace was
//! trained on **RGB**, and this tree handed it **BGR**, sampled nearest-pixel where `warpAffine`
//! defaults to bilinear.
//!
//! **Those six rows are not a fair baseline** and are not used as one: they are exactly the prints
//! that *failed* to match, since every one that matched was folded into the first row and never
//! stored. So both preparations are measured here on the same photographs.
//!
//! ## What it does
//!
//! One photograph about every 0.7 s for twelve seconds while the person sits as they normally would,
//! the largest face in each, and **each face embedded both ways** — [`SHIPPED`] and [`BEFORE`] — so the
//! only thing that differs between the two columns is the preparation. Then every pair within a
//! column.
//!
//! **One person, so every pair is the same person.** This measures the half of the threshold question
//! that can be measured alone: whether one face stays one print. Whether two people stay two is the
//! other half and it needs a second person; until then 0.363 is OpenCV's number, and with the
//! preparation now matching OpenCV's that is a better reason to trust it than it was.
//!
//! ## Written before the run
//!
//! * **`BEFORE` reproduces the failure**: most pairs under 0.363. If it does not, the six rows had a
//!   different cause and this fix explains nothing.
//! * **`SHIPPED` clears it**: most pairs over 0.363, median well above — predicted 0.6 or more. If
//!   it does not, the channel order was not the whole problem and alignment is the next suspect.
//! * **The two columns differ.** If they are identical the switch is not reaching the network.

use std::time::{Duration, Instant};

use argmind_env_face::{alike, Faces, BEFORE, SHIPPED};
use argmind_env_windows::camera::Camera;

/// **OpenCV's cosine threshold for SFace**, the same number as `people::ALIKE_ENOUGH` — which this
/// crate cannot name, since `argmind-measure` does not depend on the application.
const SAME: f32 = 0.363;
const SECONDS: f64 = 12.0;
const EVERY: Duration = Duration::from_millis(700);
const LOOKED_AT: (usize, usize) = (640, 360);
const A_FACE: f32 = 0.6;

fn main() {
    std::process::exit(run());
}

fn spread(v: &mut [f32]) -> Option<(f32, f32, f32)> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some((v[v.len() / 10], v[v.len() / 2], v[v.len() * 9 / 10]))
}

fn pairs(prints: &[Vec<f32>]) -> Vec<f32> {
    let mut out = Vec::new();
    for i in 0..prints.len() {
        for j in i + 1..prints.len() {
            if let Some(c) = alike(&prints[i], &prints[j]) {
                out.push(c);
            }
        }
    }
    out
}

fn run() -> i32 {
    println!("\n  같은 얼굴이 같은 숫자로 나오는가");
    println!("  **카메라 불이 약 15초 켜집니다.** 저장하는 것은 없습니다 — 숫자만 화면에 남습니다.");
    println!("  harness {}", argmind_measure::source_print(include_str!("recognising.rs")));

    let base = std::env::var("LOCALAPPDATA").map(std::path::PathBuf::from)
        .unwrap_or_else(|_| ".".into()).join("argmind").join("models");
    let faces = match Faces::load(&base.join("face_detection_yunet_2023mar.onnx"))
        .and_then(|f| f.also_remember(&base.join("face_recognition_sface_2021dec.onnx"))) {
        Ok(f) => f,
        Err(why) => { println!("  모델을 읽지 못했습니다 — {why}"); return 2; }
    };

    let mut camera = match Camera::on(None) {
        Ok(c) => c,
        Err(why) => { println!("  카메라를 열지 못했습니다 — {why}"); return 2; }
    };
    println!("  {:?} {}x{}\n", camera.name, camera.width, camera.height);
    println!("  평소처럼 앉아 계세요. 고개를 조금씩 움직여도 됩니다.");
    for n in (1..=3).rev() {
        println!("     {n} ...");
        std::thread::sleep(Duration::from_secs(1));
    }

    let (mut shipped, mut before) = (Vec::new(), Vec::new());
    let (mut shots, mut faceless) = (0, 0);
    let began = Instant::now();
    while began.elapsed().as_secs_f64() < SECONDS {
        let tick = Instant::now();
        if let Ok(shot) = camera.photograph(LOOKED_AT.0, LOOKED_AT.1) {
            shots += 1;
            match faces.find(&shot.rgb, shot.wide, shot.tall, A_FACE) {
                Ok(found) if !found.is_empty() => {
                    let face = found.iter().max_by(|a, b| (a.rect.2 * a.rect.3)
                        .partial_cmp(&(b.rect.2 * b.rect.3))
                        .unwrap_or(std::cmp::Ordering::Equal)).unwrap();
                    //: **The same face, both ways.** Nothing else differs between the columns.
                    if let (Ok(Some(a)), Ok(Some(b))) =
                        (faces.embed_as(&shot.rgb, shot.wide, shot.tall, face, SHIPPED),
                         faces.embed_as(&shot.rgb, shot.wide, shot.tall, face, BEFORE)) {
                        shipped.push(a);
                        before.push(b);
                    }
                }
                _ => faceless += 1,
            }
        }
        std::thread::sleep(EVERY.saturating_sub(tick.elapsed()));
    }
    drop(camera);
    println!("     끝 — 불을 껐습니다. 사진 {shots}장, 얼굴 {}개, 얼굴 없음 {faceless}\n",
             shipped.len());

    if shipped.len() < 4 {
        println!("  얼굴이 {}번밖에 안 잡혀서 비교할 수 없습니다.", shipped.len());
        return 1;
    }

    println!("  같은 사람끼리의 코사인 — 짝 {}개, 기준 {SAME}", pairs(&shipped).len());
    println!("  {:<22} {:>7} {:>7} {:>7} {:>12}", "", "10%", "중앙", "90%", "기준 넘음");
    let mut report = |what: &str, prints: &[Vec<f32>]| -> f32 {
        let mut v = pairs(prints);
        let over = v.iter().filter(|c| **c >= SAME).count();
        let n = v.len();
        let (lo, mid, hi) = spread(&mut v).unwrap();
        println!("  {what:<22} {lo:>7.3} {mid:>7.3} {hi:>7.3} {over:>5} / {n:<4}");
        mid
    };
    let after_mid = report("고친 것 (RGB, bilinear)", &shipped);
    let before_mid = report("예전 것 (BGR, nearest)", &before);

    //: **The switch reaching the network, checked** rather than assumed.
    let differs = shipped.iter().zip(&before)
        .any(|(a, b)| alike(a, b).is_some_and(|c| c < 0.9999));

    println!("\n  ── 읽기, 돌리기 전에 적어둔 대로 ──");
    if !differs {
        println!("  · **두 열이 같습니다.** 준비 방식이 네트워크까지 가지 않았습니다 — 측정이 무효입니다.");
        return 1;
    }
    if before_mid >= SAME {
        println!("  · 예전 방식도 기준을 넘습니다 ({before_mid:.3}). 여섯 줄은 다른 원인이었을 수 있습니다.");
    } else {
        println!("  · 예전 방식은 실패를 재현합니다: 중앙 {before_mid:.3} < {SAME}.");
    }
    if after_mid >= SAME {
        println!("  · **고친 방식은 넘습니다: 중앙 {after_mid:.3}.** 같은 얼굴이 같은 사람으로 남습니다.");
    } else {
        println!("  · **고친 방식도 못 넘습니다: 중앙 {after_mid:.3}.** 색 순서만의 문제가 아니었습니다 —");
        println!("    다음 용의자는 정렬입니다.");
    }
    println!("  · 다른 사람 둘이 다른 줄로 남는지는 이걸로 알 수 없습니다 — 두 번째 사람이 필요합니다.");
    0
}
