//! **Two ways of answering *are they looking at the screen*, on the same person at the same desk.**
//!
//! ```text
//!   cargo run -p argmind-measure --bin attending --release
//!   cargo run -p argmind-measure --bin attending --release -- --quick
//!   cargo run -p argmind-measure --bin attending --release -- --check
//! ```
//!
//! From `native/`, and `--release` — a debug build of `tract` is not the number anybody wants.
//! `--check` **opens no camera**: it reads the models, runs the detector on a drawn picture and
//! says whether the plumbing works, so a broken harness is found without spending somebody's time.
//!
//! ## The question
//!
//! `seeing` measured that `qwen3-vl:8b` gets the gaze right at 320×180, 4 of 4 across two room
//! states, for **4–5 seconds and a model call**. `argmind-env-face` gets a box and five points for
//! **103 ms and no model**. Thirty times cheaper is only interesting if it is also right, and
//! nothing here has been pointed at a person yet.
//!
//! **The two do not answer the same question and that is the finding this harness has to protect.**
//! `Face::head_on` reads where a *head* is pointed; somebody who holds their head still and looks
//! to the side is invisible to it and plain to the model. So the arms are *looking* and *head
//! turned away*, which is what the cheap path could possibly do, and the case it cannot do is named
//! in the reading rather than measured around.
//!
//! ## Audited before it was run, against this repository's own list
//!
//! `docs/findings/harness-pitfalls.md` has ten ways the model has lied here and seven ways the
//! measuring code has. Six applied, and the first had already destroyed a measurement in this tree:
//!
//! 1. **The phase order was fixed.** `presence` separated a still room from a still room **2.17
//!    against 0.65, three to one in the wrong direction, purely by which phase ran first.** This
//!    now runs both orders and says so when they disagree; a harness that only ever asks
//!    looking-then-turned cannot tell a real gap from a warming camera.
//! 2. **The detector was never warmed.** `tract`'s first pass carries setup the rest do not, and it
//!    would have landed entirely inside whichever arm ran first — the same shape as the 17.2 s in
//!    `seeing` that turned out to be a model loading.
//! 3. **The model's picture was taken at t=0** while the detector sampled t=0 to t=10, so the two
//!    were not looking at the same thing. It is taken at the midpoint now.
//! 4. **Two failures were collapsed into one.** A sample was dropped both when no face was found
//!    and when the eyes landed on top of each other, and only the first was counted — pitfall 5,
//!    *a boolean where the delta was needed*. It would have read as a clean arm with a smaller `n`.
//! 5. **The conditions were not in the result.** The threshold, the sizes, the model and the
//!    fingerprint are printed now, because a number whose conditions live only in the source is a
//!    number nobody can compare against next month. Three baselines here died of that.
//! 6. **A `partial_cmp().unwrap()`** on face area, which panics on a NaN rather than dropping it.
//!
//! ## How it asks
//!
//! Phases with a person doing each on purpose, the shape `presence` used to derive
//! `camera::MOVING` — a threshold taken from one state has never seen the thing it is meant to
//! separate. Each phase samples the detector at whatever rate it runs and takes **one** model call,
//! on a photograph from the middle of that phase.
//!
//! **The two numbers are different in kind.** The detector's is a distribution over tens of
//! samples; the model's is one sentence per phase. That is not a flaw, it is the difference being
//! measured: one of them is cheap enough to ask continuously.
//!
//! **It says whether they separate. It does not pick a threshold to make them.** If the two ranges
//! touch, they touch, and the cheap path does not answer this question on this face at this desk.
//!
//! **Nothing is written to disk.** The camera light comes on, the phases run, the light goes out.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use argmind_env_face::Faces;
use argmind_env_ollama::Local;
use argmind_env_windows::camera::Camera;
use argmind_measure::png;

const SEES: &str = "qwen3-vl:8b";
/// The size `seeing` measured as the smallest that reads a gaze correctly.
const ASKED_AT: (usize, usize) = (320, 180);
/// What the detector is fed. YuNet letterboxes to 640 square, so this leaves the picture at scale
/// 1.0 and invents no pixel.
const LOOKED_AT: (usize, usize) = (640, 360);
/// **Below this a detection is not a face.** OpenCV's own default for YuNet. Nothing here has
/// measured a better one, and a number chosen to make an arm pass would be the failure this
/// project's standards exist against.
const A_FACE: f32 = 0.6;

/// **The question `seeing` asked, and not a tidier one.**
///
/// The first run of this harness forced a binary — *answer with 보고 있다 or 보고 있지 않다* — and
/// got 보고 있다 on all four phases, **including both with the head turned**, in two words each, at
/// 2.3 s against the 4-5 s `seeing` took. `seeing` asked the open question below and read the gaze
/// correctly 4 of 4 across two room states.
///
/// **A forced choice with the affirmative listed first is a leading question**, and four identical
/// two-word answers is what this project calls a uniformly tidy result: evidence the measurement is
/// broken rather than evidence about the model. The first reading of that run — *the expensive path
/// failed* — is withdrawn; nothing was established about it either way.
///
/// The cost of asking the open question is that the two arms no longer answer in the same shape,
/// and somebody has to read a sentence rather than compare a word. That is the smaller cost: a
/// harness that changes the question and keeps the baseline is comparing nothing.
const ASK: &str = "이 사진에 무엇이 있는지 한국어 한두 문장으로 말하라. \
사람이 있는지, 있다면 화면 쪽을 보고 있는지, 그리고 눈에 띄는 물건 하나를 포함하라.";

#[derive(Clone, Copy, PartialEq)]
enum Doing {
    Looking,
    Turned,
}

impl Doing {
    fn what(self) -> &'static str {
        match self {
            Doing::Looking => "화면 보기",
            Doing::Turned => "고개 돌리기",
        }
    }
    fn tell(self) -> &'static str {
        match self {
            Doing::Looking => "화면을 똑바로 보고 계세요",
            Doing::Turned => "고개를 옆으로 돌려 주세요 (눈만 말고 고개를)",
        }
    }
}

/// What one phase produced.
struct Went {
    doing: Doing,
    round: usize,
    /// `|head_on|` per sample.
    heads: Vec<f32>,
    /// Face area per sample, so a subject swapped mid-run is visible rather than silent.
    areas: Vec<f32>,
    /// Milliseconds per `find`.
    costs: Vec<f64>,
    /// **Counted apart, because they are different failures.** Pitfall 5.
    faceless: u32,
    unreadable: u32,
    said: String,
    spoke: f64,
}

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--check") {
        return check();
    }
    let quick = args.iter().any(|a| a == "--quick");
    let seconds = if quick { 4.0 } else { 10.0 };
    //: **Both orders, unless somebody is only checking the harness.** See the module note: a fixed
    //: order is what made `presence` separate a still room from itself.
    let rounds: Vec<[Doing; 2]> = if quick {
        vec![[Doing::Looking, Doing::Turned]]
    } else {
        vec![[Doing::Looking, Doing::Turned], [Doing::Turned, Doing::Looking]]
    };

    println!("\n  값싼 쪽과 비싼 쪽이 같은 사실에 동의하는가");
    println!("  **카메라 불이 켜집니다.** {}단계, 각 {seconds:.0}초. 시작 전에 알려드립니다.",
             rounds.len() * 2);
    println!("  **아무것도 저장하지 않습니다** — 숫자와 문장만 화면에 남습니다.");
    //: **Every condition the numbers depend on, printed.** Pitfall 2, and the reason three
    //: baselines in this tree had to be thrown away.
    println!("  조건: 검출 {}x{} 문턱 {A_FACE} · 모델 {SEES} {}x{} think:false(무시됨) · 두 순서",
             LOOKED_AT.0, LOOKED_AT.1, ASKED_AT.0, ASKED_AT.1);
    println!("  harness {}", argmind_measure::source_print(include_str!("attending.rs")));

    let (finder, namer) = where_the_models_are();
    if !finder.is_file() {
        return missing(&finder, &namer);
    }
    let loading = Instant::now();
    let faces = match Faces::load(&finder) {
        Ok(f) => f,
        Err(why) => { println!("\n  YuNet 을 읽지 못했습니다 — {why}"); return 2; }
    };
    //: **Warmed, and the cost printed.** Point 2 of this file's own audit: the first pass carries
    //: setup and would otherwise land entirely inside whichever arm ran first.
    let warming = Instant::now();
    let blank = vec![110u8; LOOKED_AT.0 * LOOKED_AT.1 * 3];
    let _ = faces.find(&blank, LOOKED_AT.0, LOOKED_AT.1, A_FACE);
    println!("  YuNet {:.2}s 만에 올림 · 첫 통과 {:.0} ms (버림)",
             loading.elapsed().as_secs_f64(), warming.elapsed().as_secs_f64() * 1000.0);

    let mut model = Local::new(SEES);
    model.think = Some(false);
    model.keep_alive = "600s".into();
    print!("  모델을 올리는 중 ...");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let warming = Instant::now();
    let _ = model.look_at("한 단어로 답하라: 색.", &png(&[128u8; 48], 4, 4));
    println!(" {:.1}s", warming.elapsed().as_secs_f64());

    let mut camera = match Camera::on(None) {
        Ok(c) => c,
        Err(why) => { println!("  카메라를 열지 못했습니다 — {why}"); return 2; }
    };
    println!("  {:?} {}x{} · 안정화 {}\n", camera.name, camera.width, camera.height,
             match camera.settled_in {
                 Some(s) => format!("{:.0} ms", s * 1000.0),
                 None => "안 됨".into(),
             });

    let mut all: Vec<Went> = Vec::new();
    for (round, order) in rounds.iter().enumerate() {
        for doing in order {
            all.push(one_phase(&mut camera, &faces, &model, *doing, round, seconds));
        }
    }
    drop(camera);
    report(&all);
    0
}

fn one_phase(camera: &mut Camera, faces: &Faces, model: &Local, doing: Doing, round: usize,
             seconds: f64) -> Went {
    println!("  ── {} ({}번째) ──  {}", doing.what(), round + 1, doing.tell());
    for n in (1..=3).rev() {
        println!("     {n} ...");
        std::thread::sleep(Duration::from_secs(1));
    }
    println!("     시작");

    let mut went = Went {
        doing, round, heads: Vec::new(), areas: Vec::new(), costs: Vec::new(),
        faceless: 0, unreadable: 0, said: "(사진을 못 찍음)".into(), spoke: 0.0,
    };
    //: **Taken at the midpoint, not at the start.** The detector samples the whole window; a
    //: picture from t=0 catches somebody still settling into position, and the two arms would then
    //: be looking at different moments.
    let mut for_model = None;
    let began = Instant::now();
    while began.elapsed().as_secs_f64() < seconds {
        if for_model.is_none() && began.elapsed().as_secs_f64() >= seconds / 2.0 {
            for_model = camera.photograph(ASKED_AT.0, ASKED_AT.1).ok();
        }
        let Ok(shot) = camera.photograph(LOOKED_AT.0, LOOKED_AT.1) else { continue };
        let ran = Instant::now();
        let found = faces.find(&shot.rgb, shot.wide, shot.tall, A_FACE);
        went.costs.push(ran.elapsed().as_secs_f64() * 1000.0);
        match found {
            Ok(list) if !list.is_empty() => {
                //: The largest face: one person at a desk is the case this is for, and a face on a
                //: poster behind them is smaller. `unwrap_or(Equal)` rather than `unwrap` — point 6
                //: of this file's own audit.
                let biggest = list.iter().max_by(|a, b| (a.rect.2 * a.rect.3)
                    .partial_cmp(&(b.rect.2 * b.rect.3)).unwrap_or(std::cmp::Ordering::Equal));
                match biggest.and_then(|f| f.head_on().map(|h| (h, f.rect.2 * f.rect.3))) {
                    Some((h, area)) => { went.heads.push(h.abs()); went.areas.push(area); }
                    //: **A face whose eyes landed on top of each other is not a missing face.**
                    //: Collapsing the two is pitfall 5, and it would read as a clean arm with a
                    //: smaller n.
                    None => went.unreadable += 1,
                }
            }
            Ok(_) => went.faceless += 1,
            Err(why) => println!("     찾기 실패 — {why}"),
        }
    }
    println!("     표본 {} · 얼굴 없음 {} · 읽을 수 없음 {}",
             went.heads.len(), went.faceless, went.unreadable);

    if let Some(shot) = for_model {
        let encoded = png(&shot.rgb, shot.wide, shot.tall);
        let ran = Instant::now();
        match model.look_at(ASK, &encoded) {
            Ok(r) => {
                went.said = r.text.replace('\n', " ");
                went.spoke = ran.elapsed().as_secs_f64();
            }
            Err(e) => went.said = format!("(실패: {e})"),
        }
    }
    println!();
    went
}

/// p10, median, p90.
fn spread(v: &[f32]) -> Option<(f32, f32, f32)> {
    if v.is_empty() {
        return None;
    }
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some((v[v.len() / 10], v[v.len() / 2], v[v.len() * 9 / 10]))
}

fn pooled(all: &[Went], doing: Doing) -> Vec<f32> {
    all.iter().filter(|w| w.doing == doing).flat_map(|w| w.heads.clone()).collect()
}

/// Whether looking's 90th percentile clears turned's 10th, and both numbers.
fn apart(look: &[f32], away: &[f32]) -> Option<(bool, f32, f32)> {
    let (_, _, look_hi) = spread(look)?;
    let (away_lo, _, _) = spread(away)?;
    Some((look_hi < away_lo, look_hi, away_lo))
}

fn report(all: &[Went]) {
    println!("  ── 값싼 쪽: 고개 방향 |head_on|, 0 이 정면 ──");
    println!("  {:>14} {:>7} {:>7} {:>7} {:>6} {:>9}   {}",
             "", "10%", "중앙", "90%", "n", "얼굴크기", "못읽음/없음");
    for w in all {
        match spread(&w.heads) {
            Some((lo, mid, hi)) => {
                let side = spread(&w.areas).map(|(_, m, _)| m.sqrt()).unwrap_or(0.0);
                println!("  {:>10}({}) {lo:>7.2} {mid:>7.2} {hi:>7.2} {:>6} {side:>8.0}px   {}/{}",
                         w.doing.what(), w.round + 1, w.heads.len(), w.unreadable, w.faceless);
            }
            None => println!("  {:>10}({})  표본 없음 — 얼굴을 한 번도 못 찾았습니다",
                             w.doing.what(), w.round + 1),
        }
    }

    let (look, away) = (pooled(all, Doing::Looking), pooled(all, Doing::Turned));
    println!();
    match apart(&look, &away) {
        None => println!("  한쪽 표본이 비어서 비교할 수 없습니다."),
        Some((clear, look_hi, away_lo)) => {
            if clear {
                println!("  **갈립니다.** 보는 쪽 90% {look_hi:.2} < 돌린 쪽 10% {away_lo:.2}");
                println!("  둘 사이의 기하 중간은 {:.2} — 문턱을 쓴다면 여기입니다.",
                         (look_hi * away_lo).sqrt());
            } else {
                println!("  **겹칩니다.** 보는 쪽 90% {look_hi:.2} >= 돌린 쪽 10% {away_lo:.2}");
                println!("  이 얼굴 이 책상에서는 고개 방향만으로 가를 수 없습니다. 실패가 아니라");
                println!("  결과이고, 문턱을 낮춰 통과시키는 것이 바로 하면 안 되는 일입니다.");
            }
        }
    }

    //: **The order check, and it is the reason this runs twice.** `presence` read three to one in
    //: the wrong direction from order alone; two rounds that disagree mean the gap is the harness.
    let rounds = all.iter().map(|w| w.round).max().unwrap_or(0) + 1;
    if rounds > 1 {
        let verdicts: Vec<Option<bool>> = (0..rounds).map(|r| {
            let of = |d| -> Vec<f32> {
                all.iter().filter(|w| w.round == r && w.doing == d)
                    .flat_map(|w| w.heads.clone()).collect()
            };
            apart(&of(Doing::Looking), &of(Doing::Turned)).map(|(c, _, _)| c)
        }).collect();
        let first = verdicts.first().copied().flatten();
        let agree = first.is_some() && verdicts.iter().all(|v| *v == first);
        println!();
        println!("  순서 대조: {}",
                 verdicts.iter().enumerate()
                     .map(|(i, v)| format!("{}번째 {}", i + 1, match v {
                         Some(true) => "갈림",
                         Some(false) => "겹침",
                         None => "표본 없음",
                     }))
                     .collect::<Vec<_>>().join(" · "));
        println!("  {}", if agree {
            "두 순서가 같은 답을 냅니다 — 순서 효과는 아닙니다."
        } else {
            "**두 순서가 다른 답을 냅니다.** 위의 결론은 순서의 것이지 고개의 것이 아닙니다 — \
presence 가 정확히 이것으로 세 배를 거꾸로 읽었습니다."
        });
    }

    println!("\n  ── 비싼 쪽: 같은 단계 한가운데의 사진에 모델이 한 말 ──");
    for w in all {
        println!("  {:>10}({}) {:>5.1}s   {}", w.doing.what(), w.round + 1, w.spoke,
                 w.said.chars().take(64).collect::<String>());
    }

    println!("\n  ── 값 ──");
    let every: Vec<f32> = all.iter().flat_map(|w| w.costs.iter().map(|c| *c as f32)).collect();
    if let Some((lo, mid, hi)) = spread(&every) {
        println!("  찾기  10% {lo:.0} · 중앙 {mid:.0} · 90% {hi:.0} ms   (n={})", every.len());
    }
    let spoke: Vec<f32> = all.iter().filter(|w| w.spoke > 0.0).map(|w| w.spoke as f32).collect();
    if let Some((_, mid, _)) = spread(&spoke) {
        println!("  모델  중앙 {:.0} ms   (n={})", mid * 1000.0, spoke.len());
    }

    println!("\n  ── 읽기 ──");
    println!("  · 값싼 쪽이 갈리면, 그건 **고개**를 가른 것입니다. 고개를 고정하고 눈만 돌리는");
    println!("    경우는 이 하네스가 묻지 않았고 값싼 쪽은 원리상 못 봅니다.");
    println!("  · 얼굴크기가 단계마다 크게 다르면 다른 것을 보고 있었다는 뜻입니다 — 그러면");
    println!("    위의 숫자는 한 사람에 대한 것이 아닙니다.");
    println!("  · 모델 문장이 단계마다 다르면 비싼 쪽도 맞은 것이고, 같으면 이 조명 이 각도에서는");
    println!("    비싼 쪽도 못 본 것입니다. **네 답이 똑같이 짧으면 질문을 의심하세요** — 첫 판이");
    println!("    그랬고, 유도 질문이었지 모델이 틀린 게 아니었습니다.");
    println!("  · 둘 다 맞으면 남는 질문은 정확도가 아니라 값입니다.");
}

/// **Everything except the person.** Reads both models, runs the detector on a drawn picture and
/// the embedder on a drawn face, and reports. **It opens no camera.**
///
/// It expects to find **no** face in what it draws, and says so: a detector that answered here
/// would be finding faces in a gradient, which is the failure that would otherwise show up as a
/// beautifully separated measurement.
fn check() -> i32 {
    println!();
    println!("  배선 확인 — 카메라를 열지 않습니다");
    println!("  harness {}", argmind_measure::source_print(include_str!("attending.rs")));
    println!();

    let (finder, namer) = where_the_models_are();
    for (what, path, bytes) in [("YuNet", &finder, 232_589u64), ("SFace", &namer, 38_696_353)] {
        match std::fs::metadata(path) {
            Ok(m) if m.len() == bytes => println!("  {what:<6} {} bytes  있음", m.len()),
            Ok(m) => println!("  {what:<6} {} bytes  **크기가 다름** (기대 {bytes})", m.len()),
            Err(_) => return missing(&finder, &namer),
        }
    }

    let began = Instant::now();
    let faces = match Faces::load(&finder).and_then(|f| f.also_remember(&namer)) {
        Ok(f) => f,
        Err(why) => { println!("\n  읽지 못했습니다 — {why}"); return 2; }
    };
    println!("  둘 다 올림 {:.2}s", began.elapsed().as_secs_f64());

    let (w, h) = LOOKED_AT;
    let mut drawn = vec![110u8; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            let at = (y * w + x) * 3;
            drawn[at] = (60 + (x * 120 / w)) as u8;
            drawn[at + 1] = 110;
            drawn[at + 2] = (200 - (y * 120 / h)) as u8;
        }
    }
    //: Warmed here too, so the printed cost is the one the phases will see.
    let _ = faces.find(&drawn, w, h, A_FACE);
    let ran = Instant::now();
    match faces.find(&drawn, w, h, A_FACE) {
        Err(why) => { println!("  찾기가 실패했습니다 — {why}"); return 2; }
        Ok(found) => {
            let took = ran.elapsed().as_secs_f64() * 1000.0;
            if !found.is_empty() {
                println!("  그린 그림에서 얼굴 {}개 · {took:.0} ms  — **틀린 답입니다.** \
그라디언트에 얼굴은 없습니다", found.len());
                return 1;
            }
            println!("  그린 그림에서 얼굴 0개 · {took:.0} ms  — 맞는 답입니다");
        }
    }

    let made_up = argmind_env_face::Face {
        rect: (200.0, 100.0, 120.0, 120.0),
        marks: [(240.0, 140.0), (290.0, 140.0), (265.0, 165.0), (245.0, 190.0), (285.0, 190.0)],
        score: 0.9,
    };
    let ran = Instant::now();
    match faces.embed(&drawn, w, h, &made_up) {
        Err(why) => { println!("  특징값이 실패했습니다 — {why}"); return 2; }
        Ok(None) => { println!("  특징값이 꺼져 있습니다 — also_remember 가 안 붙었습니다"); return 2; }
        Ok(Some(v)) => {
            let took = ran.elapsed().as_secs_f64() * 1000.0;
            let same = argmind_env_face::alike(&v, &v).unwrap_or(0.0);
            println!("  특징값 {}개 · {took:.0} ms · 자기 자신과 {same:.3}", v.len());
            if v.len() != 128 || (same - 1.0).abs() > 1e-4 {
                println!("  **모양이 틀렸습니다.**");
                return 1;
            }
        }
    }
    //: **The made-up face is a shape check, not a quality one**, and saying so is the difference
    //: between a check and a claim.
    println!("  (지어낸 랜드마크입니다 — 그래프가 돈다는 뜻이지 숫자가 좋다는 뜻은 아닙니다)");
    println!();
    println!("  배선은 됩니다. 사람이 필요한 부분은 --quick 또는 인자 없이.");
    0
}

fn missing(finder: &PathBuf, namer: &PathBuf) -> i32 {
    println!("\n  모델이 없습니다. 두 개를 받아야 합니다 (MIT / Apache-2.0):");
    println!("    mkdir -p \"{}\"", finder.parent().unwrap().display());
    println!("    curl -L -o \"{}\" \\\n      {}", finder.display(), YUNET);
    println!("    curl -L -o \"{}\" \\\n      {}", namer.display(), SFACE);
    println!("  또는 argmind 를 띄우고 설정 → 필요한 것 → 얼굴 모델.");
    2
}

const YUNET: &str = "https://huggingface.co/opencv/opencv_zoo/resolve/main/\
models/face_detection_yunet/face_detection_yunet_2023mar.onnx";
const SFACE: &str = "https://huggingface.co/opencv/opencv_zoo/resolve/main/\
models/face_recognition_sface/face_recognition_sface_2021dec.onnx";

/// **Beside the rest of argmind's own state**, not in the repository: a 37 MB weight file is not
/// source, and a harness that writes one into the tree is a harness that gets it committed.
fn where_the_models_are() -> (PathBuf, PathBuf) {
    let base = std::env::var("LOCALAPPDATA").map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("argmind").join("models");
    (base.join("face_detection_yunet_2023mar.onnx"),
     base.join("face_recognition_sface_2021dec.onnx"))
}
