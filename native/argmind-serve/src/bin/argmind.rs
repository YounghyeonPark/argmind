//! `argmind` — look at what the environment sees, talk to it, or let it notice.
//!
//! ```text
//!   argmind look                what is on screen right now, and how long reading it took
//!   argmind say "닫아줘"         one exchange and exit
//!   argmind talk                a conversation. It answers; it clicks only with --live
//!   argmind watch --for build   it thinks when something happens, unasked
//!   argmind check               what it needs, what is here, and what it would do about it
//!   argmind setup               fetch the part that is safe to fetch
//! ```
//!
//! **`check` changes nothing and `setup` fetches one thing.** What `setup` will do is a model, and
//! only because that means Ollama putting a file in its own store over the socket this project
//! already uses — no installer runs and no executable is downloaded. Everything else argmind wants
//! is reported with the command to run and left alone: a program that can click your screen must
//! not also be able to install software while you are not looking.
//!
//! **`look` is the default because it costs nothing, touches nothing and needs no model**, and it
//! answers the question that decides whether any of this works on *your* screen: if the thing you
//! want is not in that list, no model will find it.
//!
//! `--live` is a separate word from anything you say, on purpose. **`--allow-unprompted` is a
//! second one**, because they are different risks: `--live` means *do what I asked*, and without
//! the second flag anything the agent decides **by itself** comes out as a sentence rather than as
//! a click. `watch` runs without either and is worth running that way first — it costs a model call
//! every time it notices something, and how often that is on your desktop is not a thing to guess.

use std::io::{BufRead, Write};

use argmind_core::contract::{Eyes, Speaks};
use argmind_core::intent::Trigger;
use argmind_core::session::Session;
use argmind_core::watch::{poll, Looking, Watch, MIN_GAP};
use argmind_env_ollama::Local;
use argmind_serve::platform::{self, Plain, Touch};

struct Model(Local);

impl Speaks for Model {
    fn ask(&mut self, prompt: &str) -> String {
        self.0.ask(prompt).map(|r| r.text).unwrap_or_else(|e| {
            eprintln!("  the model did not answer: {e}");
            String::new()
        })
    }
}

struct Args {
    live: bool,
    allow_unprompted: bool,
    model: String,
    limit: usize,
    rest: Vec<String>,
}

fn args() -> (String, Args) {
    let all: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| all.iter().any(|a| a == name);
    let value = |name: &str| all.iter().position(|a| a == name).and_then(|i| all.get(i + 1))
        .cloned();
    let rest: Vec<String> = all.iter()
        .filter(|a| !a.starts_with("--"))
        .filter(|a| Some((*a).clone()) != value("--model") && Some((*a).clone()) != value("--for")
                && Some((*a).clone()) != value("--limit"))
        .cloned().collect();
    let what = rest.first().cloned().unwrap_or_else(|| "look".into());
    (what, Args {
        live: flag("--live"),
        allow_unprompted: flag("--allow-unprompted"),
        model: value("--model").unwrap_or_else(|| "qwen3:8b".into()),
        limit: value("--limit").and_then(|v| v.parse().ok())
            .unwrap_or(argmind_core::converse::MAX_LISTED),
        rest,
    })
}

fn session(a: &Args) -> Session<Plain, Touch, Model> {
    let eyes = Plain::new().expect("the eyes would not open");
    //: **Two flags, two risks.** `--live` lets it do what it was asked; `--allow-unprompted` lets
    //: it act on something nobody asked for, and without it such an effect becomes a sentence.
    let hands = if a.live {
        Touch::unrestricted().arm()
    } else {
        Touch::unrestricted()
    };
    let mut model = Local::new(&a.model);
    model.think = Some(false);
    let mut s = Session::new(eyes, hands, Model(model));
    s.limit = a.limit;
    s.may_act_unprompted = a.allow_unprompted;
    s
}

fn show(say: &str, intent: &argmind_core::intent::Intent) {
    if !say.is_empty() {
        println!("  > {say}");
    }
    for e in &intent.effects {
        let what = if e.target.is_empty() { &e.text } else { &e.target };
        println!("    · {} {what}", e.kind);
    }
    if say.is_empty() && intent.effects.is_empty() {
        println!("  > …");
    }
}

fn look(a: &Args) -> i32 {
    let mut eyes = Plain::new().expect("the eyes would not open");
    let (widgets, took) = eyes.timed_foreground();
    if widgets.is_empty() {
        //: **A real answer, not a failure.** Canvas content, video, games and custom-drawn
        //: interfaces expose no widgets, and saying so is the whole point of this command.
        println!("  nothing named in the foreground window. That is a real answer, not a failure: \
                  canvas content, video, games and custom-drawn interfaces expose no widgets.");
        return 0;
    }
    println!("  {:?}", widgets[0].window.chars().take(96).collect::<String>());
    println!("  {} named widgets, read in {:.0} ms\n", widgets.len(), took * 1000.0);
    println!("  {:>4}  {:<16} {:>18}  name", "#", "role", "position");
    for (i, w) in widgets.iter().take(a.limit).enumerate() {
        let (x, y, ww, hh) = w.rect;
        println!("  {i:>4}  {:<16} {:>18}  {}", w.role.chars().take(16).collect::<String>(),
                 format!("{x},{y} {ww}x{hh}"), w.name.chars().take(60).collect::<String>());
    }
    if widgets.len() > a.limit {
        println!("  ... and {} more", widgets.len() - a.limit);
    }
    0
}

fn one(a: &Args, text: &str) -> i32 {
    let mut s = session(a);
    s.observe();
    let intent = s.decide(&Trigger::said(text));
    let (did, proposed, refused) = s.perform(&intent.effects);
    show(&intent.say, &intent);
    for p in &proposed {
        println!("    (would have {} {}) — pass --allow-unprompted, or say it yourself",
                 p.kind, if p.target.is_empty() { &p.text } else { &p.target });
    }
    for r in &refused {
        println!("    (gone before it could be done: {})", r.target);
    }
    let _ = did;
    0
}

fn talk(a: &Args) -> i32 {
    let mut s = session(a);
    println!("  {} | {}", a.model,
             if a.live { "LIVE — it will click" } else { "dry run — it will not touch anything" });
    println!("  say something, or an empty line to leave.\n");
    let stdin = std::io::stdin();
    loop {
        print!("  you: ");
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            println!();
            break;
        }
        let heard = line.trim();
        if heard.is_empty() {
            break;
        }
        let exchange = s.exchange(heard);
        if !exchange.said.is_empty() {
            println!("  > {}", exchange.said);
        }
        for d in &exchange.did {
            println!("    · did {} {}", d.effect.kind, d.effect.target);
        }
        for p in &exchange.proposed {
            println!("    (would have {} {}) — it decided that by itself, so it was not done",
                     p.kind, if p.target.is_empty() { &p.text } else { &p.target });
        }
        println!();
    }
    println!("    {:<20} {}", "turns", s.exchanges);
    println!("    {:<20} {}", "nothing to do", s.blind);
    println!("    {:<20} {}", "gone before acting", s.vanished);
    println!("    {:<20} {}", "keys it does not have", s.unknown_keys);
    0
}

fn watch(a: &Args) -> i32 {
    let mut watches: Vec<Watch> = Vec::new();
    let all: Vec<String> = std::env::args().skip(1).collect();
    for (i, arg) in all.iter().enumerate() {
        match arg.as_str() {
            "--for" => {
                if let Some(n) = all.get(i + 1) {
                    watches.push(Watch::new(Looking::Appeared(n.clone())));
                }
            }
            "--gone" => {
                if let Some(n) = all.get(i + 1) {
                    watches.push(Watch::new(Looking::Vanished(n.clone())));
                }
            }
            "--window" => watches.push(Watch::new(Looking::WindowChanged)),
            _ => {}
        }
    }
    if watches.is_empty() {
        println!("  nothing to watch for. --for NAME, --gone NAME, or --window");
        return 2;
    }
    let mut s = session(a);
    println!("  {} | {} watches | floor {MIN_GAP}s | {}", a.model, watches.len(),
             if a.allow_unprompted { "it may act by itself" } else {
                 "anything it decides by itself becomes a sentence" });
    println!("  ctrl-c to stop.\n");
    //: **The first look has no before**, so it reports nothing whatever is on screen — otherwise
    //: the whole desktop reads as newly appeared the moment this starts.
    let mut before = s.eyes.foreground();
    let started = std::time::Instant::now();
    loop {
        std::thread::sleep(std::time::Duration::from_millis(700));
        let after = s.eyes.foreground();
        let now = started.elapsed().as_secs_f64();
        for trigger in poll(&mut watches, &before, &after, now) {
            println!("  ! {}", trigger.text);
            let exchange = s.think(&trigger);
            if !exchange.said.is_empty() {
                println!("  > {}", exchange.said);
            }
            for p in &exchange.proposed {
                println!("    (would have {} {}) — nobody asked, so it was not done",
                         p.kind, if p.target.is_empty() { &p.text } else { &p.target });
            }
        }
        before = after;
    }
}

/// What it needs and what is here. **Reads, and changes nothing.**
fn check(a: &Args) -> i32 {
    let needs = argmind_serve::needs::every(&a.model);
    println!();
    print!("{}", argmind_serve::needs::report(&needs));
    //: The exit code says whether it would run, so this is usable from a script without anybody
    //: reading the words.
    i32::from(needs.iter().any(|n| n.required && !n.present))
}

/// Offer to fix what is missing — **one window per thing, and nothing without a yes.**
///
/// The window says in full what will happen before it happens, because an approval given to
/// *"install dependencies?"* is not an approval to anything in particular.
fn setup(a: &Args) -> i32 {
    let needs = argmind_serve::needs::every(&a.model);
    println!();
    print!("{}", argmind_serve::needs::report(&needs));

    let offers: Vec<&argmind_serve::needs::Need> =
        needs.iter().filter(|n| !n.present && n.fix.is_some()).collect();
    if offers.is_empty() {
        println!("  물어볼 게 없습니다.");
        return i32::from(needs.iter().any(|n| n.required && !n.present));
    }

    println!();
    let mut done = 0;
    let mut refused = 0;
    for need in &offers {
        let fix = need.fix.as_ref().expect("filtered on it");
        let body = argmind_serve::fix::describe(fix);
        //: **Printed as well as shown.** Somebody running this over a remote session, or reading
        //: back what happened, should not have to have seen the window.
        println!("  ── {} ──\n{}\n", need.what,
                 body.lines().map(|l| format!("  {l}")).collect::<Vec<_>>().join("\n"));
        if !platform::confirm(fix.title(), &body) {
            println!("  {} — 아니오\n", need.what);
            refused += 1;
            continue;
        }
        match argmind_serve::fix::apply(fix, &mut |said| println!("    {said}")) {
            Ok(said) => {
                println!("  {} — {said}\n", need.what);
                done += 1;
            }
            Err(why) => println!("  {} — 실패: {why}\n", need.what),
        }
    }
    println!("  {done}개 처리, {refused}개 거절.");
    if done > 0 {
        println!("  argmind check 로 다시 확인해 보세요.");
    }
    i32::from(needs.iter().any(|n| n.required && !n.present) && done == 0)
}

fn main() {
    let (what, a) = args();
    let code = match what.as_str() {
        "look" => look(&a),
        "check" => check(&a),
        "setup" => setup(&a),
        "talk" => talk(&a),
        "watch" => watch(&a),
        "say" => match a.rest.get(1) {
            Some(text) => one(&a, text),
            None => {
                println!("  say what: argmind say \"close this window\"");
                2
            }
        },
        other => {
            println!("  no such thing as {other:?}. look, say, talk, watch, check, setup.");
            2
        }
    };
    std::process::exit(code);
}
