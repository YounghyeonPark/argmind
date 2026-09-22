//! Does choosing survive a list the size of a web page? **The one measurement that decides whether
//! the sandbox is worth starting.**
//!
//! ```text
//!   crowding            one pass per length
//!   crowding 2          two, worst counted
//! ```
//!
//! The owner asked whether argmind should work in a sandbox of its own instead of on the screen you
//! are using. The strongest thing for it is that `name` and `role` together are **not a key** — two
//! buttons called `Open` cost a goal in `clicking`, and in a DOM argmind would own the ids. The
//! strongest thing against is size. Counted on three real pages: **230, 438 and 935** elements that
//! can be clicked. `choosing` is 28 of 37 against screens carrying **12 to 200** widgets, and
//! nothing says it holds four times further out.
//!
//! ## How the length is changed without changing anything else
//!
//! The same 37 goals and the same four captured screens. Each screen is **padded with widgets taken
//! from the other screens** — real names and real roles, because a list of `Button 41` is a list a
//! model can ignore, and an arm the model can ignore measures nothing.
//!
//! Three things are excluded from the padding, and each is a way this could have flattered itself:
//!
//! * any name the goal **accepts**, which would put a second right answer in the list;
//! * any name **already on the screen**, which would make two rows indistinguishable and the
//!   scoring ambiguous;
//! * nothing else — in particular the distractors are *not* chosen to be unlike the target, because
//!   a web page's other nine hundred elements are not chosen that way either.
//!
//! Padding is inserted at positions from a **fixed seed**, so a rerun is the same experiment, and
//! the target's index is recorded in every arm. **Position is the confound**: a target pushed to
//! row 700 is both further down and in a longer list, and only the index lets those come apart
//! afterwards.
//!
//! ## The other confound, and it is the one that would ruin this quietly
//!
//! `Local::ctx` is **8192** by default. A list of nine hundred widgets is more than that, and a
//! prompt that overflows is silently cut — which would make this a measurement of *truncation*
//! wearing the name of a measurement of *size*. So the context is raised and **`prompt_tokens` is
//! printed for every arm**; an arm whose prompt reached the ceiling is reported as unreadable
//! rather than scored.
//!
//! ## Written before the run
//!
//! **I do not know which way this goes, and there is a reason to expect it to be messy.** The
//! offer-slot work already measured accuracy against screen size and found it *non-monotone*: 3 of
//! 10 at 12 lines, 7 of 10 at 21, 2 of 9 at 123, 3 of 10 at 152 — with the shortest prompt among
//! the worst. So:
//!
//! * **holds** — accuracy at 900 within noise of the baseline. The size objection to the sandbox is
//!   answered and the project is worth starting.
//! * **falls** — a clear decline with length. Then a sandbox needs hard DOM filtering *before* it
//!   can choose anything, which is a different and larger job than *argmind browses for you*.
//! * **neither** — noisy and flat, which given the non-monotone result above is a real possibility
//!   and would mean 37 goals cannot see the effect. That is a finding about the instrument and not
//!   a licence to start building.
//!
//! ## Run it with argmind stopped, and the first attempt says why
//!
//! The first run of this hung: thirty-five minutes, no output, the graphics card at 2%. `argmind-app`
//! was running and holding `qwen3:8b` loaded at its own **8192**, and ollama serves the instance it
//! already has — a `num_ctx` in a request does not resize somebody else's model. So every arm was
//! being answered at a context a third of what it asked for, when it was answered at all.
//!
//! With argmind stopped, the same nine-hundred-widget prompt takes **3.8 s** at 10,626 tokens. The
//! whole experiment is about twelve minutes. **`/api/ps` reports `context_length`**, so this checks
//! it rather than trusting the request, and stops if it is not what was asked for.
//!
//! ## What it said — 2026-09-22, two padding arrangements
//!
//! | asked | median list | tokens | right of 28 answerable | declined of 9 | of 37 |
//! |---|---|---|---|---|---|
//! | as captured | 123 | 1,872 | **22 / 22** | 7 / 7 | **29 / 29** |
//! | 100 | 123 | 1,872 | 18 / 16 | 6 / 5 | 24 / 21 |
//! | 250 | 250 | 2,866 | 14 / 12 | 8 / 7 | 22 / 19 |
//! | 500 | 500 | 5,695 | 6 / 7 | 7 / 6 | 13 / 13 |
//! | 900 | 900 | 10,422 | **8 / 3** | 7 / 7 | 15 / 10 |
//!
//! Two figures per cell: seed `0x5eed` and seed `987654321`. **The control is 29 of 37 both times**,
//! against the 28 of 37 this experiment has always produced, so the padding is the only thing that
//! changed.
//!
//! **Finding the right thing collapses and declining does not.** Answerable goals fall from 22 of
//! 28 to between 3 and 8; the nine that should be refused are refused 5 to 8 times throughout,
//! flat. Whatever crowding takes away, it is not the judgement to say *nothing here does that*.
//!
//! **The tail is noisy and position is not disentangled.** At 500 the answer sits around row 200
//! and at 900 around row 390; one seed reads 6 then 8, the other 7 then 3. Further down and better,
//! further down and worse — this design cannot separate *longer list* from *lower row*, and saying
//! which one does the damage would be inventing it. The collapse itself does not depend on the
//! separation: 22 to 3–8 is the same answer either way.
//!
//! **What this does and does not rule out.** It rules out the naive sandbox — hand argmind a page
//! as a widget list and let it choose. It does not rule out a sandbox that *narrows first*: what is
//! in the viewport, what has an accessible name, what the last step was about. That filtering is
//! deterministic and so allowed, but it is a different project from *argmind browses for you*, and
//! it has to exist before the browsing does.
//!
//! Makes no sound, opens no device, presses nothing.

use std::io::Write;
use std::path::PathBuf;

use argmind_core::contract::Widget;
use argmind_core::converse::{read_reply, render};
use argmind_core::intent::Trigger;
use argmind_core::mind::Mind;
use argmind_env_ollama::Local;
use argmind_measure::choosing::{goals, score, tally};
use argmind_measure::crowding::crowd;
use argmind_measure::screens;

/// How long a list each arm renders. The first is *as captured*, which is the control.
const LENGTHS: [usize; 5] = [0, 100, 250, 500, 900];

/// Where the ceiling is put. **Well past the longest arm**, so that a prompt is cut by nothing.
const ROOM: i64 = 32_768;

fn main() {
    let repeat: usize = std::env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(1);
    //: **The padding arrangement, and it is an argument for a reason.** Running this twice
    //: unchanged gives the same numbers to the digit — temperature is 0 and the seed was fixed, so
    //: it is one experiment run twice rather than two readings, and the first rerun of it proved
    //: exactly that by agreeing in every cell. A different seed puts different distractors in
    //: different places and is the only rerun worth counting.
    let seed: u64 = std::env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(0x5eed);
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let dir = root.join("measure");
    let loaded = screens(&dir.join("screens"));
    let all = goals(&std::fs::read_to_string(dir.join("goals.json")).expect("goals.json"));
    let pool: Vec<Widget> = loaded.values().flat_map(|s| s.widgets.clone()).collect();

    let mut model = Local::new("qwen3:8b");
    model.think = Some(false);
    model.ctx = ROOM;
    model.keep_alive = "600s".into();
    for _ in 0..2 {
        let _ = model.ask("Reply with the single word: ready");
    }

    println!("\n  목록 길이만 바꿔가며 고르기 — {} goals x {repeat}, 화면 {}, 풀 {} 개",
             all.len(), loaded.len(), pool.len());
    println!("  프롬프트 한도 {ROOM} 토큰. 잘린 팔은 점수를 읽지 않습니다.\n");
    //: **Three numbers for the length and not one.** The first version printed the longest list
    //: in the arm, and the first two arms then read 152 and 152 with different scores — because
    //: padding to 100 grows the small screens and leaves the 152 one alone. A column that cannot
    //: tell two arms apart is a column that will be misread.
    println!("  {:>5} {:>6} {:>7} {:>10} {:>9} {:>9}   {}",
             "요청", "중앙", "토큰", "맞음", "거절", "합계", "답이 있던 줄 (중앙값)");
    let _ = std::io::stdout().flush();

    //: `(median length, correct, of)` per arm — the median rather than the longest, because the
    //: arms differ by what the *small* screens were grown to.
    let mut best: Vec<(usize, usize, usize)> = Vec::new();
    for want in LENGTHS {
        let mut rows = Vec::new();
        let (mut tokens, mut cut) = (0u64, 0usize);
        let mut wheres: Vec<usize> = Vec::new();
        let mut lengths: Vec<usize> = Vec::new();
        for g in &all {
            let Some(screen) = loaded.get(&g.screen) else { continue };
            let (widgets, at) = crowd(&screen.widgets, &pool, g, want, seed);
            lengths.push(widgets.len());
            if let Some(i) = at {
                wheres.push(i);
            }
            let mut chose = Vec::new();
            for _ in 0..repeat {
                let trigger = Trigger::said(&g.goal);
                //: **The limit is the arm.** `MAX_LISTED` is what ships; here the length is the
                //: independent variable, so it is passed rather than taken.
                let prompt = render(&widgets, &trigger, &[], widgets.len(), &Mind::resting());
                let got = model.ask(&prompt);
                if let Ok(r) = &got {
                    if let Some(n) = r.prompt_tokens {
                        tokens = tokens.max(n as u64);
                        if n as i64 >= ROOM {
                            cut += 1;
                        }
                    }
                }
                let reply = got.map(|r| r.text).unwrap_or_default();
                let intent = read_reply(&reply, &widgets, &trigger);
                chose.push(intent.effects.iter().find(|e| e.kind == "click")
                           .map(|e| e.target.clone()));
            }
            rows.push(score(g, &widgets, chose));
        }
        let t = tally(&rows);
        wheres.sort_unstable();
        let middle = wheres.get(wheres.len() / 2).copied().unwrap_or(0);
        let note = if cut > 0 { format!("  ** {cut} 개가 한도에 닿음 — 읽지 마세요") } else {
            String::new()
        };
        lengths.sort_unstable();
        let middling = lengths.get(lengths.len() / 2).copied().unwrap_or(0);
        println!("  {want:>5} {middling:>6} {tokens:>7} {:>7}/{:<2} {:>6}/{:<2} {:>6}/{:<2}   {middle}{note}",
                 t.right, t.answerable, t.declined, t.unanswerable,
                 t.right + t.declined, rows.len());
        //: **Flushed, because the first run of this was invisible for thirty-five minutes.**
        //: Rust buffers a pipe, and a harness whose progress cannot be seen is one nobody can tell
        //: from a hang — which is exactly what it was.
        let _ = std::io::stdout().flush();
        if cut == 0 {
            best.push((middling, t.right + t.declined, rows.len()));
        }
    }

    println!();
    println!("  ── 읽기 ──");
    if best.len() < 2 {
        println!("  읽을 수 있는 팔이 둘도 안 됩니다.");
        return;
    }
    let (short_n, short_ok, short_of) = best[0];
    let (long_n, long_ok, long_of) = *best.last().unwrap();
    println!("  {short_n} 개에서 {short_ok}/{short_of}, {long_n} 개에서 {long_ok}/{long_of}.");
    //: **The control first.** If the shortest arm does not look like the 28 of 37 this experiment
    //: has always produced, the padding is not the only thing that changed and no arm is readable.
    if short_ok + 4 < 28 || short_ok > 32 {
        println!("  대조군이 {short_ok}/{short_of} 입니다 — 늘 28/37 이던 것과 다릅니다. \
길이 말고 다른 것이 바뀌었으므로 아래 줄은 읽지 마세요.");
        return;
    }
    let fall = short_ok as i64 - long_ok as i64;
    if fall >= 6 {
        println!("  → 길어지면 무너집니다 ({fall} 만큼). 샌드박스는 무엇을 보여줄지 먼저 \
걸러내야 하고, 그것은 '대신 웹을 봐준다' 와는 다른 크기의 일입니다.");
    } else if fall <= 2 {
        println!("  → 900 개에서도 버팁니다. 크기를 이유로 한 반대는 답해졌습니다.");
    } else {
        println!("  → {fall} 만큼 떨어졌습니다 — 노이즈와 구별되지 않습니다. 37 개 목표로는 \
이 효과를 못 봅니다. 계측기에 대한 발견이지 착수해도 좋다는 뜻이 아닙니다.");
    }
}
