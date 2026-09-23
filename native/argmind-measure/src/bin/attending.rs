//! **Two ways of answering *are they looking at the screen*, on the same person at the same desk.**
//!
//! ```text
//!   cargo run -p argmind-measure --bin attending --release
//!   cargo run -p argmind-measure --bin attending --release -- --quick
//!   cargo run -p argmind-measure --bin attending --release -- --check
//! ```
//!
//! `--check` **opens no camera**: it reads the models, runs the detector on a drawn picture, and
//! says whether the plumbing works. A harness that crashes after asking somebody to sit still for
//! twenty seconds has spent their twenty seconds, and this one asks them to move on purpose.
//!
//! From `native/`, and `--release` — a debug build of `tract` is not the number anybody wants.
//!
//! ## The question
//!
//! `seeing` measured that `qwen3-vl:8b` gets the gaze right at 320×180, 4 of 4 across two room
//! states, for **4–5 seconds and a model call**. `argmind-env-face` gets a box and five points for
//! **103 ms and no model**. Thirty times cheaper is only interesting if it is also right, and
//! nothing here has been pointed at a person yet.
//!
//! **The two do not answer the same question and that is the finding this harness has to protect.**
//! [`Face::head_on`] reads where a *head* is pointed; a person who holds their head still and looks
//! to the side is invisible to it and plain to the model. So the arms below are *looking* and *head
//! turned away*, which is what the cheap path could possibly do, and the case it cannot do is named
//! in the reading rather than measured around.
//!
//! ## How it asks
//!
//! Two phases with a person doing each on purpose, the shape `presence` used to derive
//! `camera::MOVING` — because a threshold taken from one state has never seen the thing it is meant
//! to separate. Each phase samples the detector at whatever rate it runs and takes **one** model
//! call for the comparison, on a photograph from the same phase.
//!
//! **The two numbers this produces are different in kind.** The detector's is a distribution over
//! tens of samples; the model's is one sentence per phase. That is not a flaw in the harness, it is
//! the difference being measured: one of them is cheap enough to ask continuously.
//!
//! **It says whether they separate. It does not pick a threshold to make them.** If the two ranges
//! touch, they touch, and the cheap path does not answer this question on this face at this desk.
//!
//! ## What it does with the pictures
//!
//! **Nothing is written to disk.** The camera light comes on, the phases run, the light goes out.
//! The detector never leaves the process and the one caption per phase reaches the terminal.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use argmind_env_face::Faces;
use argmind_env_ollama::Local;
use argmind_env_windows::camera::Camera;
use argmind_measure::png;

const SEES: &str = "qwen3-vl:8b";
/// The size `seeing` measured as the smallest that reads a gaze correctly.
const ASKED_AT: (usize, usize) = (320, 180);
/// What the detector is fed. Letterboxing to 640 leaves this at scale 1.0, so no pixel is invented.
const LOOKED_AT: (usize, usize) = (640, 360);
/// **Below this a detection is not a face.** OpenCV's own default for YuNet; nothing here improves
/// on it and a number chosen to make an arm pass would be the failure this project exists against.
const A_FACE: f32 = 0.6;

const ASK: &str = "이 사진 속 사람이 화면 쪽을 보고 있는가? \
'보고 있다' 또는 '보고 있지 않다' 로 시작해서 한 문장으로만 답하라.";

struct Phase {
    what: &'static str,
    tell: &'static str,
}

const PHASES: [Phase; 2] = [
    Phase { what: "화면 보기", tell: "화면을 똑바로 보고 계세요" },
    Phase { what: "고개 돌리기", tell: "고개를 옆으로 돌려 주세요 (눈만 말고 고개를)" },
];

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    if std::env::args().any(|a| a == "--check") {
        return check();
    }
    let quick = std::env::args().any(|a| a == "--quick");
    let seconds = if quick { 4.0 } else { 10.0 };

    println!("\n  값싼 쪽과 비싼 쪽이 같은 사실에 동의하는가");
    println!("  **카메라 불이 켜집니다.** 두 단계, 각 {seconds:.0}초. 시작 전에 알려드립니다.");
    println!("  **아무것도 저장하지 않습니다** — 숫자와 문장 두 개만 화면에 남습니다.");
    println!("  harness {}", argmind_measure::source_print(include_str!("attending.rs")));

    let (finder, namer) = where_the_models_are();
    if !finder.is_file() {
        println!("\n  모델이 없습니다. 두 개를 받아야 합니다 (MIT / Apache-2.0):");
        println!("    mkdir -p \"{}\"", finder.parent().unwrap().display());
        println!("    curl -L -o \"{}\" \\\n      {}", finder.display(), YUNET);
        println!("    curl -L -o \"{}\" \\\n      {}", namer.display(), SFACE);
        return 2;
    }
    let loading = Instant::now();
    let faces = match Faces::load(&finder) {
        Ok(f) => f,
        Err(why) => { println!("\n  YuNet 을 읽지 못했습니다 — {why}"); return 2; }
    };
    println!("  YuNet {:.2}s 만에 올림\n", loading.elapsed().as_secs_f64());

    let mut model = Local::new(SEES);
    model.think = Some(false);
    model.keep_alive = "300s".into();
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

    let mut readings: Vec<Vec<f32>> = Vec::new();
    let mut costs: Vec<Vec<f64>> = Vec::new();
    let mut said: Vec<String> = Vec::new();
    let mut spoke: Vec<f64> = Vec::new();

    for phase in &PHASES {
        println!("  ── {} ──  {}", phase.what, phase.tell);
        for n in (1..=3).rev() {
            println!("     {n} ...");
            std::thread::sleep(Duration::from_secs(1));
        }
        println!("     시작");

        //: **One photograph for the model, taken at the start of the phase**, so its sentence is
        //: about a moment inside the window the detector sampled and not about whatever the room
        //: looked like once the person relaxed.
        let for_model = camera.photograph(ASKED_AT.0, ASKED_AT.1).ok();

        let mut heads = Vec::new();
        let mut each = Vec::new();
        let began = Instant::now();
        let mut faceless = 0;
        while began.elapsed().as_secs_f64() < seconds {
            let Ok(shot) = camera.photograph(LOOKED_AT.0, LOOKED_AT.1) else { continue };
            let ran = Instant::now();
            let found = faces.find(&shot.rgb, shot.wide, shot.tall, A_FACE);
            each.push(ran.elapsed().as_secs_f64() * 1000.0);
            match found {
                Ok(list) if !list.is_empty() => {
                    //: The largest face, because a reflection or a photograph on the wall is
                    //: smaller than the person at the desk and this harness has one subject.
                    let biggest = list.iter().max_by(|a, b| (a.rect.2 * a.rect.3)
                        .partial_cmp(&(b.rect.2 * b.rect.3)).unwrap()).unwrap();
                    if let Some(h) = biggest.head_on() {
                        heads.push(h.abs());
                    }
                }
                Ok(_) => faceless += 1,
                Err(why) => println!("     찾기 실패 — {why}"),
            }
        }
        println!("     {} 표본, 얼굴 없음 {}", heads.len(), faceless);

        let (mut caption, mut took) = (String::from("(사진을 못 찍음)"), 0.0);
        if let Some(shot) = for_model {
            let encoded = png(&shot.rgb, shot.wide, shot.tall);
            let ran = Instant::now();
            match model.look_at(ASK, &encoded) {
                Ok(r) => { caption = r.text.replace('\n', " "); took = ran.elapsed().as_secs_f64(); }
                Err(e) => caption = format!("(실패: {e})"),
            }
        }
        readings.push(heads);
        costs.push(each);
        said.push(caption);
        spoke.push(took);
        println!();
    }
    drop(camera);

    report(&readings, &costs, &said, &spoke);
    0
}

fn report(readings: &[Vec<f32>], costs: &[Vec<f64>], said: &[String], spoke: &[f64]) {
    println!("  ── 값싼 쪽: 고개 방향 |head_on|, 0 이 정면 ──");
    let mut middles = Vec::new();
    for (i, phase) in PHASES.iter().enumerate() {
        let mut v = readings[i].clone();
        if v.is_empty() {
            println!("  {:>12}  표본 없음 — 얼굴을 한 번도 못 찾았습니다", phase.what);
            middles.push(None);
            continue;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = v[v.len() / 2];
        println!("  {:>12}  중앙 {:.2} · 10퍼센타일 {:.2} · 90퍼센타일 {:.2} · n={}",
                 phase.what, mid, v[v.len() / 10], v[v.len() * 9 / 10], v.len());
        middles.push(Some((v[v.len() / 10], mid, v[v.len() * 9 / 10])));
    }

    //: **The honest comparison is the tails, not the middles.** Two medians far apart with
    //: overlapping ranges is a threshold that fires on the wrong phase several times a minute, and
    //: `presence` learned that from a still maximum four hundred times its own median.
    match (middles.first().and_then(|m| *m), middles.get(1).and_then(|m| *m)) {
        (Some((_, look_mid, look_hi)), Some((away_lo, away_mid, _))) => {
            println!();
            if look_hi < away_lo {
                let between = (look_hi * away_lo).sqrt();
                println!("  **갈립니다.** 보는 쪽 90퍼센타일 {look_hi:.2} < 돌린 쪽 10퍼센타일 \
{away_lo:.2}");
                println!("  둘 사이의 기하 중간은 {between:.2} — 문턱을 쓴다면 여기입니다.");
                println!("  중앙값 비 {:.1}x", away_mid / look_mid.max(1e-6));
            } else {
                println!("  **겹칩니다.** 보는 쪽 90퍼센타일 {look_hi:.2} >= 돌린 쪽 10퍼센타일 \
{away_lo:.2}");
                println!("  이 얼굴 이 책상에서는 고개 방향만으로 가를 수 없습니다. 이건 실패가");
                println!("  아니라 결과이고, 문턱을 낮춰서 통과시키는 것이 바로 하면 안 되는 일입니다.");
            }
        }
        _ => println!("\n  한쪽 표본이 비어서 비교할 수 없습니다."),
    }

    println!("\n  ── 비싼 쪽: 같은 순간에 모델이 한 말 ──");
    for (i, phase) in PHASES.iter().enumerate() {
        println!("  {:>12}  {:.1}s   {}", phase.what, spoke[i],
                 said[i].chars().take(70).collect::<String>());
    }

    println!("\n  ── 값 ──");
    for (i, phase) in PHASES.iter().enumerate() {
        let mut c = costs[i].clone();
        if c.is_empty() { continue; }
        c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("  {:>12}  찾기 중앙 {:.0} ms, 최악 {:.0} ms   ·   모델 {:.0} ms",
                 phase.what, c[c.len() / 2], c[c.len() - 1], spoke[i] * 1000.0);
    }

    println!("\n  ── 읽기 ──");
    println!("  · 값싼 쪽이 갈리면, 그건 **고개**를 가른 것입니다. 고개를 고정하고 눈만 돌리는");
    println!("    경우는 이 하네스가 묻지 않았고 값싼 쪽은 원리상 못 봅니다.");
    println!("  · 모델 문장이 두 단계에서 다르면 비싼 쪽도 맞은 것이고, 같으면 이 조명 이 각도");
    println!("    에서는 비싼 쪽도 못 본 것입니다.");
    println!("  · 둘 다 맞으면 남는 질문은 정확도가 아니라 값입니다.");
}

/// **Everything except the person.** Reads both models, runs the detector on a drawn picture and
/// the embedder on a drawn face, and reports. **It opens no camera.**
///
/// It expects to find **no** face in what it draws, and says so: a detector that answered here
/// would be finding faces in flat grey, which is the failure that would otherwise show up as a
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
            Err(_) => {
                println!("  {what:<6} 없음 — {}", path.display());
                println!("\n  받는 법:");
                println!("    mkdir -p \"{}\"", finder.parent().unwrap().display());
                println!("    curl -L -o \"{}\" \\\n      {}", finder.display(), YUNET);
                println!("    curl -L -o \"{}\" \\\n      {}", namer.display(), SFACE);
                return 2;
            }
        }
    }

    let began = Instant::now();
    let faces = match Faces::load(&finder).and_then(|f| f.also_remember(&namer)) {
        Ok(f) => f,
        Err(why) => { println!("\n  읽지 못했습니다 — {why}"); return 2; }
    };
    println!("  둘 다 올림 {:.2}s", began.elapsed().as_secs_f64());

    //: A drawn picture with no face in it. **Finding one here would be the bug**, and a check that
    //: only proves the call returns is a check that passes on a detector answering at random.
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
    let ran = Instant::now();
    match faces.find(&drawn, w, h, A_FACE) {
        Err(why) => { println!("  찾기가 실패했습니다 — {why}"); return 2; }
        Ok(found) => {
            let took = ran.elapsed().as_secs_f64() * 1000.0;
            if found.is_empty() {
                println!("  그린 그림에서 얼굴 0개 · {took:.0} ms  — 맞는 답입니다");
            } else {
                println!("  그린 그림에서 얼굴 {}개 · {took:.0} ms  — **틀린 답입니다.** \
그라디언트에 얼굴은 없습니다", found.len());
                return 1;
            }
        }
    }

    //: The embedder is exercised on a made-up face, which says the graph runs and the alignment
    //: does not panic. It says nothing about whether the numbers are any good.
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

    println!();
    println!("  배선은 됩니다. 사람이 필요한 부분은 --quick 또는 인자 없이.");
    0
}

const YUNET: &str = "https://huggingface.co/opencv/opencv_zoo/resolve/main/\
models/face_detection_yunet/face_detection_yunet_2023mar.onnx";
const SFACE: &str = "https://huggingface.co/opencv/opencv_zoo/resolve/main/\
models/face_recognition_sface/face_recognition_sface_2021dec.onnx";

/// **Beside the rest of argmind's own state**, not in the repository: a 38 MB weight file is not
/// source, and a harness that writes one into the tree is a harness that gets it committed.
fn where_the_models_are() -> (PathBuf, PathBuf) {
    let base = std::env::var("LOCALAPPDATA").map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("argmind").join("models");
    (base.join("face_detection_yunet_2023mar.onnx"),
     base.join("face_recognition_sface_2021dec.onnx"))
}
