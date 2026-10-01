//! **Is the same face the same print — and when is it not?**
//!
//! ```text
//!   cargo run -p argmind-measure --bin recognising --release
//! ```
//!
//! From `native/`. **Stop argmind first** — it holds the camera. The light comes on for about forty
//! seconds in three phases; nothing is saved, only numbers reach the terminal.
//!
//! ## What the first version found, and the claim it withdrew
//!
//! The owner, 2026-09-30: *얼굴을 치웠다 다시 보이면 이름을 또 물어봄.* Six rows for one person,
//! their prints at cosine 0.01 to 0.36. The cause was claimed to be the channel order — SFace takes
//! RGB and this tree fed it BGR — and the first run of this harness, sitting still in front of the
//! camera, **refuted that**: 18 faces, 153 pairs, **0.715 median prepared the new way and 0.714 the
//! old**, 152 of 153 over the threshold either way. The fix matches OpenCV and stays; **it was not
//! why there were six rows.** Written down because the claim went into a commit message.
//!
//! What that run could not see is the condition the owner described: a face leaving the frame and
//! coming back. The application embeds whichever face is largest every three seconds, and that
//! includes a face half out of frame, turned to profile, or moving — exactly the frames where five
//! landmarks land in the wrong places and alignment makes a print of something that is not quite a
//! face. The six rows are consistent with that and nothing here has shown it.
//!
//! ## Three phases, and every print carries how good it was
//!
//! 1. **정면** — sit facing the camera. These are the reference.
//! 2. **고개** — turn the head slowly side to side.
//! 3. **들락날락** — lean out of the frame and back, two or three times.
//!
//! Each print records `|head_on|`, the detector's score, and how close the face box came to the
//! edge of the picture. Each is compared with the centre of the reference prints. Then a **candidate
//! gate** — facing (`|head_on| < HEAD_ASIDE`), fully inside the frame, score at least [`SURE`] — is
//! applied, and the prints that pass and the prints that fail are counted apart.
//!
//! ## The second run, and what replaced the gate
//!
//! Sitting still 20 of 20, turning the head **20 of 20**, leaning out and back **8 of 13** — the bad
//! frames are the arriving and leaving ones. The candidate gate **did not separate them** (81% of
//! passing frames over threshold against 88% of failing) and would have thrown away sixteen good
//! turned-head frames to do it. And an old-way print against the new-way centre read **0.880, 20 of
//! 20**, so the application's dropping of old rows was reverted before it ever ran.
//!
//! What the bad frames share is that they do not agree with each other — the nine stray rows sat
//! under 0.363 in 36 of 36 pairs — so the rule now in `argmind_core::recognise` enrols a new face
//! only when **two consecutive unknown frames agree**. This harness replays every recorded frame, in
//! order, through **that same rule** and through the old one, into an empty roster, and counts the
//! rows each makes for one person. At the application's cadence — a frame every three seconds —
//! and at every frame, which is the harder case: more chances for two bad frames to agree.
//!
//! ## Written before the run
//!
//! * **Replay: the new rule makes one row, the old rule more than one**, at the application's
//!   cadence. If the new rule also makes several, agreement between bad frames is more common than
//!   the stray rows suggested and the rule needs a longer run of agreement.
//!
//! * **The reference holds**: phase 1 like the first run, nearly every print over 0.363.
//! * **The bad prints are phases 2 and 3**, and they are the ones the gate rejects: of prints that
//!   **pass**, 95% or more over 0.363; of prints that **fail**, clearly fewer. If passing prints
//!   fall under the threshold as often as failing ones, the gate is the wrong gate.
//! * **Old prints are comparable with new**: the first run did not check whether a print made the
//!   old way matches one made the new way. It is checked here, because the application dropped every
//!   old row on the assumption that they would not.

use std::time::{Duration, Instant};

use argmind_core::recognise::{best, Newcomer};
use argmind_env_face::{alike, Face, Faces, BEFORE, HEAD_ASIDE, SHIPPED};
use argmind_env_windows::camera::Camera;

/// **OpenCV's cosine threshold for SFace**, from the rule the application runs.
const SAME: f32 = argmind_core::recognise::ALIKE_ENOUGH;
/// **How often the application embeds a face**, `platform::FACE_EVERY`, which this crate cannot
/// name. The replay at this cadence is the one that describes what a person sees.
const APP_EVERY: f64 = 3.0;
/// A detector score the candidate gate asks for. **Chosen** — higher than the 0.6 that finds a face,
/// because finding one and trusting its landmarks are different questions.
const SURE: f32 = 0.8;
/// How far inside the picture the face box must stay, as a share of the picture's width. Chosen.
const MARGIN: f32 = 0.02;
const EVERY: Duration = Duration::from_millis(500);
const LOOKED_AT: (usize, usize) = (640, 360);
const A_FACE: f32 = 0.6;

const PHASES: [(&str, &str, f64); 3] = [
    ("정면", "카메라를 정면으로 보고 가만히 계세요", 10.0),
    ("고개", "고개를 천천히 좌우로 돌려 주세요", 10.0),
    ("들락날락", "몸을 카메라 밖으로 뺐다가 돌아오기를 두세 번 해 주세요", 12.0),
];

struct Print {
    phase: usize,
    /// Seconds since the first phase began.
    at: f64,
    shipped: Vec<f32>,
    before: Vec<f32>,
    head: Option<f32>,
    score: f32,
    inside: bool,
}

impl Print {
    /// **The candidate gate.** Each part is something the application already has in hand when it
    /// decides to embed, which is the only kind of gate it could ship.
    fn passes(&self) -> bool {
        self.head.is_some_and(|h| h.abs() < HEAD_ASIDE) && self.inside && self.score >= SURE
    }
}

fn main() {
    std::process::exit(run());
}

fn inside(face: &Face, w: usize, h: usize) -> bool {
    let m = MARGIN * w as f32;
    let (x, y, fw, fh) = face.rect;
    x >= m && y >= m && x + fw <= w as f32 - m && y + fh <= h as f32 - m
}

/// The normalised mean of a set of prints — the reference a new face is compared with.
fn centre(prints: &[&Vec<f32>]) -> Option<Vec<f32>> {
    let n = prints.first()?.len();
    let mut sum = vec![0.0f32; n];
    for p in prints {
        let norm = p.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        for (s, x) in sum.iter_mut().zip(p.iter()) {
            *s += x / norm;
        }
    }
    Some(sum)
}

fn share(v: &[f32]) -> String {
    if v.is_empty() {
        return "—".into();
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let over = s.iter().filter(|c| **c >= SAME).count();
    format!("중앙 {:.3} · 최저 {:.3} · 넘음 {over}/{}", s[s.len() / 2], s[0], s.len())
}

fn run() -> i32 {
    println!("\n  같은 얼굴이 같은 숫자로 나오는가 — 언제 아닌가");
    println!("  **카메라 불이 약 40초 켜집니다.** 저장하는 것은 없습니다 — 숫자만 화면에 남습니다.");
    println!("  조건: 기준 {SAME} · 후보 관문 = 정면(|head_on| < {HEAD_ASIDE}) · 화면 안 ({:.0}% 여백) \
· 확신 ≥ {SURE}", MARGIN * 100.0);
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

    let mut prints: Vec<Print> = Vec::new();
    let mut faceless = [0usize; 3];
    let start = Instant::now();
    for (i, (what, tell, seconds)) in PHASES.iter().enumerate() {
        println!("  ── {what} ──  {tell}");
        for n in (1..=3).rev() {
            println!("     {n} ...");
            std::thread::sleep(Duration::from_secs(1));
        }
        println!("     시작");
        let began = Instant::now();
        while began.elapsed().as_secs_f64() < *seconds {
            let tick = Instant::now();
            if let Ok(shot) = camera.photograph(LOOKED_AT.0, LOOKED_AT.1) {
                match faces.find(&shot.rgb, shot.wide, shot.tall, A_FACE) {
                    Ok(found) if !found.is_empty() => {
                        let face = found.iter().max_by(|a, b| (a.rect.2 * a.rect.3)
                            .partial_cmp(&(b.rect.2 * b.rect.3))
                            .unwrap_or(std::cmp::Ordering::Equal)).unwrap();
                        if let (Ok(Some(a)), Ok(Some(b))) =
                            (faces.embed_as(&shot.rgb, shot.wide, shot.tall, face, SHIPPED),
                             faces.embed_as(&shot.rgb, shot.wide, shot.tall, face, BEFORE)) {
                            prints.push(Print {
                                phase: i, at: start.elapsed().as_secs_f64(),
                                shipped: a, before: b, head: face.head_on(),
                                score: face.score, inside: inside(face, shot.wide, shot.tall),
                            });
                        }
                    }
                    _ => faceless[i] += 1,
                }
            }
            std::thread::sleep(EVERY.saturating_sub(tick.elapsed()));
        }
        println!("     얼굴 {}개, 얼굴 없음 {}\n",
                 prints.iter().filter(|p| p.phase == i).count(), faceless[i]);
    }
    drop(camera);
    println!("  끝 — 불을 껐습니다.\n");

    let reference: Vec<&Vec<f32>> = prints.iter().filter(|p| p.phase == 0).map(|p| &p.shipped)
        .collect();
    let reference_old: Vec<&Vec<f32>> = prints.iter().filter(|p| p.phase == 0)
        .map(|p| &p.before).collect();
    let (Some(ref_new), Some(ref_old)) = (centre(&reference), centre(&reference_old)) else {
        println!("  정면 구간에서 얼굴을 못 잡아서 기준을 만들 수 없습니다.");
        return 1;
    };
    let to_ref = |p: &Print| alike(&p.shipped, &ref_new).unwrap_or(0.0);

    println!("  ── 정면 기준과의 코사인, 구간별 ──");
    for (i, (what, _, _)) in PHASES.iter().enumerate() {
        let v: Vec<f32> = prints.iter().filter(|p| p.phase == i).map(to_ref).collect();
        println!("  {what:<8} {}", share(&v));
    }

    //: **The comparison that decides whether the gate ships.** The reference phase is left out so
    //: the gate is judged on the frames it would actually be asked about.
    let rest: Vec<&Print> = prints.iter().filter(|p| p.phase > 0).collect();
    let pass: Vec<f32> = rest.iter().filter(|p| p.passes()).map(|p| to_ref(p)).collect();
    let fail: Vec<f32> = rest.iter().filter(|p| !p.passes()).map(|p| to_ref(p)).collect();
    println!("\n  ── 후보 관문, 고개·들락날락 구간에서 ──");
    println!("  통과    {}", share(&pass));
    println!("  탈락    {}", share(&fail));
    //: Which part of the gate did the rejecting, so a gate that works for the wrong reason is seen.
    let turned = rest.iter().filter(|p| !p.head.is_some_and(|h| h.abs() < HEAD_ASIDE)).count();
    let edge = rest.iter().filter(|p| !p.inside).count();
    let unsure = rest.iter().filter(|p| p.score < SURE).count();
    println!("  탈락 이유 (겹칠 수 있음): 정면 아님 {turned} · 화면 가장자리 {edge} · 확신 낮음 {unsure}");

    //: **Old against new**, which the application's decision to drop old rows assumed without
    //: checking.
    let cross: Vec<f32> = prints.iter().filter(|p| p.phase == 0)
        .map(|p| alike(&p.before, &ref_new).unwrap_or(0.0)).collect();
    let old_self: Vec<f32> = prints.iter().filter(|p| p.phase == 0)
        .map(|p| alike(&p.before, &ref_old).unwrap_or(0.0)).collect();
    println!("\n  ── 예전 방식 특징값이 새 방식과 비교되는가 (정면) ──");
    println!("  예전 ↔ 새 기준   {}", share(&cross));
    println!("  예전 ↔ 예전 기준 {}", share(&old_self));

    //: **The replay: what the roster would have done with this session.** An empty roster, every
    //: frame in the order it was taken, matched by `recognise::best` — the application's own
    //: function — and a frame nobody matches either enrolled at once (the rule until 2026-09-30) or
    //: handed to `Newcomer` (the rule now).
    let replay = |frames: &[&Print], agree: bool| -> usize {
        let mut roster: Vec<Vec<f32>> = Vec::new();
        let mut newcomer = Newcomer::default();
        for p in frames {
            if best(roster.iter().map(|r| r.as_slice()), &p.shipped).is_some() {
                newcomer.known();
                continue;
            }
            if !agree {
                roster.push(p.shipped.clone());
            } else if let Some((print, _)) = newcomer.unknown(p.shipped.clone(), p.at) {
                roster.push(print);
            }
        }
        roster.len()
    };
    let every: Vec<&Print> = prints.iter().collect();
    let mut cadence: Vec<&Print> = Vec::new();
    for p in &prints {
        if cadence.last().is_none_or(|last| p.at - last.at >= APP_EVERY) {
            cadence.push(p);
        }
    }
    let (old_app, new_app) = (replay(&cadence, false), replay(&cadence, true));
    let (old_all, new_all) = (replay(&every, false), replay(&every, true));
    println!("\n  ── 재생: 한 사람에게 명부가 몇 줄을 만드는가 (정답 1) ──");
    println!("  {:<24} {:>10} {:>14}", "", "한 번 보면 등록", "두 번 맞아야 등록");
    println!("  {:<24} {old_app:>10} {new_app:>14}", format!("앱 속도 ({APP_EVERY}초, {}장)", cadence.len()));
    println!("  {:<24} {old_all:>10} {new_all:>14}", format!("모든 장 ({}장)", every.len()));

    println!("\n  ── 읽기, 돌리기 전에 적어둔 대로 ──");
    if new_app == 1 && old_app > 1 {
        println!("  · **새 규칙이 한 줄로 막습니다**: 앱 속도에서 예전 {old_app}줄 → {new_app}줄.");
    } else if new_app == 1 {
        println!("  · 새 규칙은 한 줄입니다. 이번 세션은 예전 규칙도 {old_app}줄이라 차이를 보여주지 못했습니다.");
    } else {
        println!("  · **새 규칙도 {new_app}줄입니다.** 나쁜 장끼리 연달아 맞는 경우가 생각보다 흔합니다 —");
        println!("    두 장이 아니라 더 긴 일치가 필요합니다.");
    }
    if new_all > 1 {
        println!("  · 모든 장에서는 {new_all}줄 — 장이 촘촘할수록 나쁜 장 둘이 맞을 기회가 늘어납니다.");
    }
    let rate = |v: &[f32]| if v.is_empty() { f32::NAN }
                           else { v.iter().filter(|c| **c >= SAME).count() as f32 / v.len() as f32 };
    let (rp, rf) = (rate(&pass), rate(&fail));
    //: **The verdict was too easy to earn, and it said *ship* on two frames.** The first version
    //: shipped the gate whenever passing frames cleared 95% and beat failing ones by anything at
    //: all; the third run read 14 of 14 against 17 of 19 and printed *이 관문을 싣습니다* — on the
    //: run after one where the same gate failed, 81% against 88%. It also costs every turned-head
    //: frame, which matched 20 of 20 in both runs. So the gap now has to be large before this says
    //: the gate separates, and it never says *ship*: two runs disagreeing is not a decision.
    if pass.is_empty() {
        println!("  · 고개·들락날락 구간에서 관문을 통과한 얼굴이 없습니다 — 관문이 너무 좁거나 구간이 짧았습니다.");
    } else if rp >= 0.95 && rf <= rp - 0.25 {
        println!("  · **관문이 크게 가릅니다**: 통과 {:.0}% 대 탈락 {:.0}%. 다른 실행에서도 그런지 보세요.",
                 rp * 100.0, rf * 100.0);
    } else {
        println!("  · **관문이 가르지 못합니다**: 통과 {:.0}% 대 탈락 {:.0}% — 차이가 작거나 반대입니다.",
                 rp * 100.0, rf * 100.0);
    }
    if rate(&cross) >= 0.95 {
        println!("  · 예전 특징값도 새 기준과 맞습니다 — 예전 줄을 지운 건 필요 없는 조치였습니다.");
    } else {
        println!("  · 예전 특징값은 새 기준과 잘 안 맞습니다 — 지운 게 맞았습니다.");
    }
    0
}
