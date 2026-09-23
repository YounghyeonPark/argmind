//! Can a vision model say anything true about **this room**, and at what size?
//!
//! ```text
//!   cargo run -p argmind-measure --bin seeing --release
//!   cargo run -p argmind-measure --bin seeing --release -- 320x180
//!   cargo run -p argmind-measure --bin seeing --release -- --repeat 3
//!   cargo run -p argmind-measure --bin seeing --release -- --control
//! ```
//!
//! **From `native/`** — the workspace manifest is there and not at the repository root. `--release`
//! because the largest arm hands 6 MB of picture to a hand-written base64, and in a debug build
//! that shows up in the seconds as if the model had spent them.
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
//! ## The control, and why the first run needed one
//!
//! The first run answered every arm correctly — *a person at a monitor, looking at the screen, and a
//! yellow cup.* **Most of that sentence is guessable without a picture.** The question names the
//! things to look for, and in front of a webcam *somebody is there and facing the screen* is the
//! answer a model would give whether or not it can see anything. `--control` is what separates
//! those: the same question about pictures with **no room in them at all** — flat grey, uniform
//! noise, and the four-coloured-square test card the 8 s figure came from. **It opens no camera.**
//! If a person appears in those captions, the prompt is doing the talking.
//!
//! `--repeat N` re-asks every size on **the same photograph**, which is what decides whether a
//! spread like 4.0 / 5.1 / 6.5 s at an identical token count is a size effect or sampling.
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

/// **The sizes below the sensor's own.** The top arm is not in this list — it is whatever the
/// camera actually gives, read at runtime, for a reason worth stating: asking for 1920x1080 from a
/// 640x480 device would **upscale**, and an upscale carries no detail the smaller arm did not have.
/// Every arm would then be the same picture at different byte counts and *resolution does not help*
/// would come out true for free. Anything here at or above the sensor is dropped.
const SMALLER: [(usize, usize); 3] = [(640, 360), (320, 180), (160, 90)];

/// **Asked so a wrong answer is visible.** *Describe this image* invites a sentence that is true of
/// any room; naming the three things makes a caption that is wrong say so.
const ASK: &str = "이 사진에 무엇이 있는지 한국어 한두 문장으로 말하라. \
사람이 있는지, 있다면 화면 쪽을 보고 있는지, 그리고 눈에 띄는 물건 하나를 포함하라.";

fn main() {
    std::process::exit(run());
}

fn run() -> i32 {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--control") {
        return control();
    }
    let repeat: usize = args.iter().position(|a| a == "--repeat")
        .and_then(|i| args.get(i + 1)).and_then(|n| n.parse().ok()).unwrap_or(1);
    let only: Option<(usize, usize)> = args.iter().find_map(|a| {
        let (w, h) = a.split_once('x')?;
        Some((w.parse().ok()?, h.parse().ok()?))
    });

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

    //: **The arms are known only once the device has answered.** The top one is the sensor's own
    //: size and the rest are the standing list with anything at or above it dropped — see `SMALLER`
    //: for why an upscaled arm would answer the question by itself.
    let sizes: Vec<(usize, usize)> = match only {
        Some(one) => vec![one],
        None => std::iter::once((camera.width, camera.height))
            .chain(SMALLER.iter().copied().filter(|(w, h)| *w < camera.width && *h < camera.height))
            .collect(),
    };
    if only.is_some_and(|(w, h)| w > camera.width || h > camera.height) {
        println!("  이 크기는 센서보다 큽니다 — 업스케일이고, 없는 디테일은 생기지 않습니다.");
    }

    //: **One photograph at the sensor's size**, shrunk for each arm afterwards. Four separate
    //: captures would be four different instants, and *the small one missed the person* would be
    //: indistinguishable from *the person moved*.
    let biggest = sizes.iter().map(|(w, _)| *w).max().unwrap_or(640).max(camera.width);
    let tallest = sizes.iter().map(|(_, h)| *h).max().unwrap_or(360).max(camera.height);
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

    println!("  {:>10} {:>9} {:>8} {:>7}   {}", "크기", "PNG", "토큰", "걸림", "뭐라고 하는가");
    for (wide, tall) in &sizes {
        let rgb = if (*wide, *tall) == (taken.wide, taken.tall) {
            taken.rgb.clone()
        } else {
            //: Shrunk from the one photograph, by averaging, the same way the camera reduced it.
            shrink_rgb(&taken.rgb, taken.wide, taken.tall, *wide, *tall)
        };
        //: **Encoding is outside the clock.** `png` here writes deflate *stored* blocks — no
        //: compression, by the licence argument the rest of the wire code is written under — so the
        //: full-size arm is about 6 MB where the smallest is 43 kB. The byte count is printed
        //: precisely so a slow arm can be checked against it rather than assumed to be the model.
        let shot = png(&rgb, *wide, *tall);
        for round in 0..repeat {
            let began = std::time::Instant::now();
            let got = model.look_at(ASK, &shot);
            let took = began.elapsed().as_secs_f64();
            //: **The same picture every round.** One photograph, re-asked, so the only thing that
            //: can move between rounds is the model — which is what makes a spread readable as
            //: sampling rather than as a property of the size.
            let size = if round == 0 { format!("{wide}x{tall}") } else { String::new() };
            match got {
                Ok(r) => {
                    let said = r.text.replace('\n', " ");
                    //: **A cut-off caption is not a short caption.** `length` means the ceiling took
                    //: it and whatever it was going to say about the room is missing, so the
                    //: sentence below cannot be read as the model's answer.
                    let how = if r.reason == "stop" { String::new() }
                              else { format!(" [{}]", r.reason) };
                    println!("  {:>10} {:>7} B {:>8} {:>6.1}s{}   {}", size, shot.len(),
                             r.prompt_tokens.map(|n| n.to_string()).unwrap_or_else(|| "?".into()),
                             took, how, said.chars().take(90).collect::<String>());
                }
                Err(e) => println!("  {:>10} {:>7} B                실패: {e}", size, shot.len()),
            }
        }
    }

    println!();
    println!("  ── 읽기 ──");
    println!("  숫자가 아니라 문장을 보세요. 방에 실제로 있던 것과 맞습니까?");
    println!("  · 전부 맞으면 가장 작은 크기가 쓸 크기입니다 — 토큰은 크기를 따라 줄지 않습니다.");
    println!("  · 큰 것만 맞으면 해상도가 사는 것이 있다는 뜻이고, 그 값을 아는 셈입니다.");
    println!("  · 전부 그럴듯하고 전부 틀리면 — 어느 방에나 참인 문장을 쓰고 있는 것이고,");
    println!("    그건 이 경로가 사람을 알아보는 데 쓸 수 없다는 뜻입니다.");
    println!("  시간은 PNG 크기와 같이 보세요 — 큰 쪽은 6 MB 를 들고 갑니다. 초가 바이트를 따라");
    println!("  움직이면 그건 모델이 아니라 전선입니다.");
    0
}

/// **The negative control: the same question about pictures with no room in them.**
///
/// **It opens no camera** — every picture here is drawn in this function, which is the point. The
/// first run's caption was *a person at a monitor, looking at the screen, and a yellow cup*, and the
/// first two thirds of that are what anybody would guess in front of a webcam. If these captions
/// also contain a person, the model is answering the question rather than the picture, and the room
/// run says nothing.
///
/// The test card is the third one on purpose: **it is what the 8 s figure was measured on**, so this
/// also re-reads that number on the machine as it stands today.
fn control() -> i32 {
    println!();
    println!("  대조군 — 방이 없는 사진에 같은 질문");
    println!("  **카메라를 열지 않습니다.** 세 장 모두 여기서 그립니다.");
    println!("  질문이 사람을 명시하므로, 여기서 사람이 나오면 모델은 사진이 아니라 질문에");
    println!("  답하고 있는 것이고 방 실험은 아무 말도 하지 않은 것이 됩니다.");
    println!("  harness {}", argmind_measure::source_print(include_str!("seeing.rs")));
    println!();

    let (w, h) = (640usize, 360usize);
    let mut cards: Vec<(&str, Vec<u8>)> = Vec::new();
    cards.push(("평평한 회색", vec![128u8; w * h * 3]));
    //: A fixed generator, so the noise arm is the same picture on every run and a changed answer is
    //: the model rather than a different picture.
    let mut seed = 0x2026_09_23u64;
    let mut noise = vec![0u8; w * h * 3];
    for byte in noise.iter_mut() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *byte = (seed >> 33) as u8;
    }
    cards.push(("균일 잡음", noise));
    //: **Four coloured squares** — the drawn test card the captioning route was first measured on.
    let mut card = vec![0u8; w * h * 3];
    const QUARTERS: [[u8; 3]; 4] = [[220, 40, 40], [40, 180, 60], [50, 80, 220], [230, 200, 40]];
    for y in 0..h {
        for x in 0..w {
            let c = QUARTERS[(y >= h / 2) as usize * 2 + (x >= w / 2) as usize];
            let at = (y * w + x) * 3;
            card[at..at + 3].copy_from_slice(&c);
        }
    }
    cards.push(("색 사각형 네 개", card));

    let mut model = Local::new(SEES);
    model.think = Some(false);
    model.keep_alive = "300s".into();
    //: **The first call carries the load.** The first control run read 11.0 s for flat grey against
    //: 3.4 s for noise at an identical 1137 tokens, which is the model arriving on the card and not
    //: anything about the picture. One throwaway ask, so every printed number is warm.
    print!("  모델을 올리는 중 ...");
    let _ = std::io::Write::flush(&mut std::io::stdout());
    let warming = std::time::Instant::now();
    let _ = model.look_at("한 단어로 답하라: 색.", &png(&[128u8; 48], 4, 4));
    println!(" {:.1}s\n", warming.elapsed().as_secs_f64());

    let mut people = 0;
    println!("  {:>16} {:>8} {:>7}   {}", "사진", "토큰", "걸림", "뭐라고 하는가");
    for (what, rgb) in &cards {
        let shot = png(rgb, w, h);
        let began = std::time::Instant::now();
        let got = model.look_at(ASK, &shot);
        let took = began.elapsed().as_secs_f64();
        match got {
            Ok(r) => {
                let said = r.text.replace('\n', " ");
                //: **Crude on purpose, and printed as crude** — and the first run caught it being
                //: too crude. `없` was the only denial it knew, so *사람이나 눈에 띄는 물건이 전혀
                //: 보이지 않습니다* was scored as a person and the control read 1 of 3 when the three
                //: sentences all declined. The Korean negations are a short list and it is written
                //: out rather than guessed at; **the sentence printed beside it is still what
                //: decides**, which is the only reason a wrong count was visible at all.
                let denied = ["없", "않", "아니", "보이지", "찾을 수"];
                if said.contains("사람") && !denied.iter().any(|d| said.contains(d)) {
                    people += 1;
                }
                println!("  {:>16} {:>8} {:>6.1}s   {}", what,
                         r.prompt_tokens.map(|n| n.to_string()).unwrap_or_else(|| "?".into()),
                         took, said.chars().take(80).collect::<String>());
            }
            Err(e) => println!("  {:>16}                  실패: {e}", what),
        }
    }

    println!();
    println!("  방이 없는 사진 {}장 중 사람이 나온 것: {}", cards.len(), people);
    println!("  (`사람` 이 있고 부정어가 없는 문장을 센 것 — 거칠기 때문에 옆의 문장이 판단입니다.");
    println!("   첫 판에서 이 셈이 틀렸고, 옆의 문장이 있었기 때문에 틀린 것이 보였습니다.)");
    if people == 0 {
        println!("  0 이면 캡션은 그림을 읽고 있습니다 — 방 실험이 증거가 됩니다.");
    } else {
        println!("  0 이 아니면 질문이 말하고 있는 것이고, 방 실험은 아무 말도 하지 않았습니다.");
    }
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
