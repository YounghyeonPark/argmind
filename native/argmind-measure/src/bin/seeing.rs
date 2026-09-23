//! Can a vision model say anything true about **this room**, and at what size?
//!
//! ```text
//!   seeing                  one photograph, captioned at four sizes
//!   seeing 320x180          one size only
//! ```
//!
//! **The 8 s figure everybody quotes was measured on a drawn test card.** Four coloured squares, and
//! the caption was accurate because there was nothing to be wrong about. Nobody has ever pointed
//! this at a room. So the two things that decide whether the vision path is worth building are both
//! unmeasured:
//!
//! * **does it say something true** about a real scene, with a person in it and ordinary clutter;
//! * **at what size**, because `prompt_tokens` was 1110 from 640×360 all the way down to 80×46 —
//!   the tokens do not fall with the pixels, so if a small picture works it is free.
//!
//! ## Why this needs a door that did not exist
//!
//! `camera.rs` reduced every frame to 64×36 greys and had **no accessor for anything larger**. That
//! was the guarantee, and asking this question means giving it up: [`Camera::photograph`] now
//! exists and takes the size as an argument, so the caller states in its own source how much of a
//! room it is asking for. What stands where the type stood is a rule —
//! `argmind-rules/tests/the_app_takes_no_photograph.rs` — **the application cannot take one at
//! all**, only this, run on purpose by the person in the picture.
//!
//! ## What it does with the picture
//!
//! **Nothing is written to disk and nothing is kept.** Each size is encoded, sent to the model, and
//! dropped; the only thing that survives the process is the caption, on the terminal, read by the
//! person who was in front of the lens. They are the ground truth — a saved photograph would let
//! somebody check the caption later, and that is exactly the artefact this is trying not to make.
//!
//! **The camera light comes on** and goes out when it finishes. One photograph is taken, at the
//! sensor's full size, and shrunk for each arm — so four captions cost one picture rather than
//! four, and every arm is looking at the same instant.
//!
//! **Stop argmind first if you want the times to mean anything.** It holds `qwen3:8b` on the card,
//! and `crowding` lost thirty-five minutes to exactly that: a second model does not get the
//! graphics card to itself, and *a look costs the answer about 3x* was measured from the other
//! side of the same contention.

use argmind_env_ollama::Local;
use argmind_env_windows::camera::Camera;
use argmind_measure::png;

/// The vision model this tree already has. Apache-2.0, and already on the machine.
const SEES: &str = "qwen3-vl:8b";

/// **Four sizes over one photograph.** The largest is what the sensor gives; the smallest is below
/// where `prompt_tokens` stopped falling, so if the answer holds there the picture is nearly free.
const SIZES: [(usize, usize); 4] = [(1920, 1080), (640, 360), (320, 180), (160, 90)];

/// **Asked so a wrong answer is visible.** *Describe this image* invites a sentence that is true of
/// any room; naming the three things makes a caption that is wrong say so.
const ASK: &str = "이 사진에 무엇이 있는지 한국어 한두 문장으로 말하라. \
사람이 있는지, 있다면 화면 쪽을 보고 있는지, 그리고 눈에 띄는 물건 하나를 포함하라.";

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let only = std::env::args().skip(1).find_map(|a| {
        let (w, h) = a.split_once('x')?;
        Some((w.parse().ok()?, h.parse().ok()?))
    });
    let sizes: Vec<(usize, usize)> = match only {
        Some(one) => vec![one],
        None => SIZES.to_vec(),
    };

    println!("\n  카메라가 본 방을 모델이 설명할 수 있는가");
    println!("  **카메라 불이 켜집니다.** 사진 한 장을 찍어 크기별로 줄여서 물어봅니다.");
    println!("  **아무것도 저장하지 않습니다** — 캡션만 화면에 남고, 사진은 이 프로세스와 함께 \
사라집니다.");
    println!("  argmind 를 멈추고 돌리면 시간이 의미가 있습니다 — 안 그러면 카드를 나눠 씁니다.");
    println!("  harness {}\n", argmind_measure::source_print(include_str!("seeing.rs")));

    let opening = std::time::Instant::now();
    let mut camera = match Camera::on(None) {
        Ok(c) => c,
        Err(why) => {
            println!("  카메라를 열지 못했습니다 — {why}");
            return 2;
        }
    };
    println!("  {:?} {}x{} · 여는 데 {:.0} ms · 안정화 {}",
             camera.name, camera.width, camera.height,
             opening.elapsed().as_secs_f64() * 1000.0,
             match camera.settled_in {
                 Some(s) => format!("{:.0} ms", s * 1000.0),
                 None => "끝내 안정되지 않음 — 사진이 흐릴 수 있습니다".to_string(),
             });

    //: **One photograph at the sensor's size**, shrunk for each arm afterwards. Four separate
    //: captures would be four different instants, and *the small one missed the person* would be
    //: indistinguishable from *the person moved*.
    let biggest = sizes.iter().map(|(w, _)| *w).max().unwrap_or(640);
    let tallest = sizes.iter().map(|(_, h)| *h).max().unwrap_or(360);
    let taken = match camera.photograph(biggest, tallest) {
        Ok(p) => p,
        Err(why) => {
            println!("  사진을 찍지 못했습니다 — {why}");
            return 2;
        }
    };
    //: The light goes out here rather than after the model has finished thinking, which is a minute
    //: of a camera being on for nothing.
    drop(camera);
    println!("  사진 {}x{} · {:.0} ms · 불을 껐습니다\n", taken.wide, taken.tall,
             taken.took * 1000.0);

    let mut model = Local::new(SEES);
    //: **Asked for, and measured to be ignored.** The test-card run set `think: false` three
    //: different ways and this model thought anyway, 467–1,690 characters about four squares. It is
    //: sent so the arms are identical and so the day it starts working is visible; the seconds
    //: below include whatever thinking happened.
    model.think = Some(false);
    model.keep_alive = "300s".into();

    println!("  {:>10} {:>8} {:>8} {:>7}   {}", "크기", "PNG", "토큰", "걸림", "뭐라고 하는가");
    for (wide, tall) in &sizes {
        let rgb = if (*wide, *tall) == (taken.wide, taken.tall) {
            taken.rgb.clone()
        } else {
            //: Shrunk from the one photograph, by averaging, the same way the camera reduced it.
            shrink_rgb(&taken.rgb, taken.wide, taken.tall, *wide, *tall)
        };
        let shot = png(&rgb, *wide, *tall);
        let began = std::time::Instant::now();
        let got = model.look_at(ASK, &shot);
        let took = began.elapsed().as_secs_f64();
        match got {
            Ok(r) => {
                let said = r.text.replace('\n', " ");
                //: **A cut-off caption is not a short caption.** `length` means the ceiling took it
                //: and whatever it was going to say about the room is missing, so the sentence below
                //: cannot be read as the model's answer.
                let how = if r.reason == "stop" { String::new() } else { format!(" [{}]", r.reason) };
                println!("  {:>5}x{:<4} {:>7} B {:>8} {:>6.1}s{}   {}", wide, tall, shot.len(),
                         r.prompt_tokens.map(|n| n.to_string()).unwrap_or_else(|| "?".into()),
                         took, how, said.chars().take(90).collect::<String>());
            }
            Err(e) => println!("  {:>5}x{:<4} {:>7} B                실패: {e}", wide, tall,
                               shot.len()),
        }
    }

    println!();
    println!("  ── 읽기 ──");
    println!("  숫자가 아니라 문장을 보세요. 방에 실제로 있던 것과 맞습니까?");
    println!("  · 전부 맞으면 가장 작은 크기가 쓸 크기입니다 — 토큰은 크기를 따라 줄지 않습니다.");
    println!("  · 큰 것만 맞으면 해상도가 사는 것이 있다는 뜻이고, 그 값을 아는 셈입니다.");
    println!("  · 전부 그럴듯하고 전부 틀리면 — 어느 방에나 참인 문장을 쓰고 있는 것이고,");
    println!("    그건 이 경로가 사람을 알아보는 데 쓸 수 없다는 뜻입니다.");
    0
}

/// Box-average RGB down. The camera's own `shrink` takes BGRA off the sensor; this takes what it
/// already produced, so one photograph can serve every arm.
fn shrink_rgb(rgb: &[u8], width: usize, height: usize, wide: usize, tall: usize) -> Vec<u8> {
    let mut out = vec![0u8; wide * tall * 3];
    if width == 0 || height == 0 || wide == 0 || tall == 0 {
        return out;
    }
    for ty in 0..tall {
        for tx in 0..wide {
            let x0 = tx * width / wide;
            let x1 = (((tx + 1) * width) / wide).max(x0 + 1).min(width);
            let y0 = ty * height / tall;
            let y1 = (((ty + 1) * height) / tall).max(y0 + 1).min(height);
            let (mut r, mut g, mut b, mut seen) = (0u64, 0u64, 0u64, 0u64);
            for y in y0..y1 {
                for x in x0..x1 {
                    let at = (y * width + x) * 3;
                    let (Some(rr), Some(gg), Some(bb)) =
                        (rgb.get(at), rgb.get(at + 1), rgb.get(at + 2)) else { continue };
                    r += *rr as u64;
                    g += *gg as u64;
                    b += *bb as u64;
                    seen += 1;
                }
            }
            if seen > 0 {
                let at = (ty * wide + tx) * 3;
                out[at] = (r / seen) as u8;
                out[at + 1] = (g / seen) as u8;
                out[at + 2] = (b / seen) as u8;
            }
        }
    }
    out
}
