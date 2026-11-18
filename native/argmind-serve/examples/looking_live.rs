//! **The application's own turn where the list is blind — on a Character Map this started, with an
//! argmind of its own.** The path `--example pointing` (argmind-measure) does not take: a sentence
//! sent to `/say`, the conversation model's choice, `think`'s gate, `look_inside`, `propose`, the
//! row and its marked picture as the page draws them, `/confirm`, `act`, `perform` and the hands.
//!
//! ```text
//!   LOCALAPPDATA=<an empty folder> cargo run --release -p argmind-serve --example looking_live -- <out>
//! ```
//!
//! **Never the person's argmind or their settings.** It refuses to run with the real
//! `LOCALAPPDATA`, because the server reads and writes `%LOCALAPPDATA%\argmind\` exactly as the real
//! one does, and it writes the one standing answer it needs — `looking` — into that empty folder,
//! with `notice` and `initiative` refused so nothing starts by itself. `acting` is left unanswered,
//! so every effect waits for a press, and the only press sent is for a row whose effect is a point
//! inside Character Map's grid. The person's own argmind should be stopped first: it watches window
//! changes, and a Character Map appearing is one.
//!
//! **It stops before asking anything unless the window in front is the Character Map it started**,
//! since the turn reads whatever window is in front and that would otherwise be the person's.
//!
//! Writes into `<out>`: `events.txt` (every event the page was told), `marked.png` (what
//! `/looked/<row>` served), `page.png` (the page as headless Edge draws it while the row waits).

#[cfg(all(windows, not(feature = "headless")))]
fn main() {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpStream;
    use std::path::PathBuf;
    use std::time::{Duration, Instant};

    const PORT: u16 = 7742;
    const ASKED: &str = "© 문자 눌러줘";

    let out = PathBuf::from(std::env::args().nth(1).expect("a folder for what it saw"));
    std::fs::create_dir_all(&out).expect("the out folder");
    let home = std::env::var("LOCALAPPDATA").expect("LOCALAPPDATA");
    let real = std::env::var("USERPROFILE").map(|u| format!("{u}\\AppData\\Local")).unwrap_or_default();
    assert!(!home.trim_end_matches(['\\', '/']).eq_ignore_ascii_case(&real),
            "LOCALAPPDATA is the person's own — point it at an empty folder");
    let dir = PathBuf::from(&home).join("argmind");
    std::fs::create_dir_all(&dir).unwrap();
    assert!(!dir.join("conversation.jsonl").exists(), "{} is not empty — use a fresh folder", dir.display());
    std::fs::write(dir.join("decisions.json"),
                   "{\n  \"initiative\": \"denied\",\n  \"looking\": \"allowed\",\n  \"notice\": \"denied\"\n}\n")
        .unwrap();

    std::thread::spawn(|| {
        argmind_serve::app::run(argmind_serve::app::How {
            dry: false, speak: false, hear: false, model_name: "qwen3:8b".into(), port: PORT, web: None,
        })
    });

    let ask = |method: &str, path: &str, token: &str, body: &str| -> Option<(String, Vec<u8>)> {
        let mut s = TcpStream::connect(("127.0.0.1", PORT)).ok()?;
        let _ = s.set_read_timeout(Some(Duration::from_secs(30)));
        let head = format!("{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{PORT}\r\nX-Argmind-Token: {token}\r\n\
                            Content-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                           body.len());
        s.write_all(head.as_bytes()).ok()?;
        s.write_all(body.as_bytes()).ok()?;
        let mut all = Vec::new();
        let _ = s.read_to_end(&mut all);
        let split = all.windows(4).position(|w| w == b"\r\n\r\n")?;
        Some((String::from_utf8_lossy(&all[..split]).to_string(), all[split + 4..].to_vec()))
    };

    //: The server, and the token it stamped into its page.
    let began = Instant::now();
    let token = loop {
        if let Some((_, body)) = ask("GET", "/", "", "") {
            let page = String::from_utf8_lossy(&body).to_string();
            if let Some(at) = page.find("data-token=\"") {
                let rest = &page[at + 12..];
                break rest[..rest.find('"').unwrap()].to_string();
            }
        }
        assert!(began.elapsed() < Duration::from_secs(30), "the server never answered");
        std::thread::sleep(Duration::from_millis(300));
    };

    //: Every event the page would be told, as it arrives.
    let (tx, events) = std::sync::mpsc::channel::<(String, String)>();
    {
        let token = token.clone();
        std::thread::spawn(move || {
            let mut s = TcpStream::connect(("127.0.0.1", PORT)).expect("the event stream");
            let _ = write!(s, "GET /events HTTP/1.1\r\nHost: 127.0.0.1:{PORT}\r\nX-Argmind-Token: {token}\r\n\r\n");
            let (mut event, mut data) = (String::new(), Vec::<String>::new());
            for line in BufReader::new(s).lines() {
                let Ok(line) = line else { break };
                if let Some(e) = line.strip_prefix("event: ") {
                    event = e.to_string();
                } else if let Some(d) = line.strip_prefix("data: ") {
                    data.push(d.to_string());
                } else if line.is_empty() && !event.is_empty() {
                    if tx.send((std::mem::take(&mut event), data.join("\n"))).is_err() {
                        break;
                    }
                    data.clear();
                }
            }
        });
    }
    let mut log = String::new();
    let wait_for = |what: &dyn Fn(&str, &str) -> bool, within: Duration, log: &mut String| -> Option<(String, String)> {
        let end = Instant::now() + within;
        while let Some(left) = end.checked_duration_since(Instant::now()) {
            match events.recv_timeout(left) {
                Ok((e, d)) => {
                    if !matches!(e.as_str(), "seeing" | "state" | "needs" | "tasks" | "voice-seed" | "note-kept") {
                        log.push_str(&format!("{:>6.1}s {e}: {d}\n", began.elapsed().as_secs_f64()));
                    }
                    if what(&e, &d) {
                        return Some((e, d));
                    }
                }
                Err(_) => break,
            }
        }
        None
    };
    let _ = wait_for(&|_, _| false, Duration::from_secs(2), &mut log);

    //: **A Character Map of its own, in front** — or nothing is asked.
    let mut map = argmind_env_windows::launch::Launched::start("charmap.exe").expect("charmap.exe");
    let handle = map.window(Duration::from_secs(10)).expect("Character Map never showed a window");
    let in_front = argmind_env_windows::launch::bring_to_front(handle, map.pid, Duration::from_secs(5));
    let (_, title, pid) = argmind_env_windows::launch::whats_in_front();
    if !in_front || pid != Some(map.pid) {
        println!("  stopped: the window in front is {title:?}, not the Character Map this started — nothing was asked");
        map.close();
        std::process::exit(2);
    }
    std::thread::sleep(Duration::from_millis(1500));
    let pointer = argmind_env_windows::cursor_now();

    let sent = Instant::now();
    ask("POST", "/say", &token, ASKED).expect("/say");
    let proposed = wait_for(&|e, _| e == "propose", Duration::from_secs(150), &mut log);
    //: Whatever else the turn said after its row.
    let _ = wait_for(&|_, _| false, Duration::from_secs(2), &mut log);
    let mut said = vec![format!("asked {ASKED:?} with {title:?} in front")];
    let Some((_, row)) = proposed else {
        said.push(format!("no row within 150 s — {:.1} s", sent.elapsed().as_secs_f64()));
        finish(&out, &log, &said);
        return;
    };
    said.push(format!("row after {:.1} s: {row}", sent.elapsed().as_secs_f64()));
    let number: u64 = row.split("\"token\":").nth(1).and_then(|r| r.split(',').next())
        .and_then(|n| n.trim().parse().ok()).expect("the row's number");
    let pictured = row.contains("\"picture\":true");

    //: What the page fetches beside the row, and the page as it draws it.
    match ask("GET", &format!("/looked/{number}"), &token, "") {
        Some((head, png)) if head.contains("200 OK") => {
            std::fs::write(out.join("marked.png"), &png).unwrap();
            said.push(format!("/looked/{number}: {} bytes of PNG", png.len()));
        }
        Some((head, _)) => said.push(format!("/looked/{number}: {}", head.lines().next().unwrap_or(""))),
        None => said.push(format!("/looked/{number}: no answer")),
    }
    //: **`--timeout`, not `--virtual-time-budget`**: virtual time waits for the network to go quiet,
    //: and the page's event stream never does — the first run hung there until it was killed.
    let edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe";
    let shot = std::process::Command::new(edge)
        .args(["--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=520,900", "--timeout=6000"])
        .arg(format!("--user-data-dir={}", out.join("edge").display()))
        .arg(format!("--screenshot={}", out.join("page.png").display()))
        .arg(format!("http://127.0.0.1:{PORT}/"))
        .spawn();
    said.push(format!("page screenshot: {}", match shot {
        Ok(mut child) => {
            let end = Instant::now() + Duration::from_secs(30);
            loop {
                match child.try_wait() {
                    Ok(Some(status)) => break format!("{status}"),
                    _ if Instant::now() > end => {
                        let _ = child.kill();
                        break "killed after 30 s".to_string();
                    }
                    _ => std::thread::sleep(Duration::from_millis(200)),
                }
            }
        }
        Err(e) => e.to_string(),
    }));

    //: **Pressed only when it is a point inside the grid**; anything else is reported and left.
    if !(pictured && row.contains("Character Grid")) {
        said.push("not a point inside the grid — left unpressed".into());
        let _ = ask("POST", &format!("/decline/{number}"), &token, "");
        finish(&out, &log, &said);
        return;
    }
    ask("POST", &format!("/confirm/{number}"), &token, "").expect("/confirm");
    let ended = wait_for(&|e, _| matches!(e, "ran" | "dropped" | "gone"), Duration::from_secs(20), &mut log);
    let _ = wait_for(&|_, _| false, Duration::from_secs(1), &mut log);
    said.push(format!("after the press: {ended:?}"));
    let status: Vec<String> = argmind_env_windows::WindowsEyes::new().expect("UI Automation").read(handle)
        .into_iter().filter(|w| w.name.starts_with("U+")).map(|w| w.name).collect();
    said.push(format!("Character Map's status bar: {status:?}"));
    let after = argmind_env_windows::cursor_now();
    said.push(format!("the person's pointer {pointer:?} → {after:?}: {}",
                      if pointer == after { "unchanged" } else { "MOVED" }));
    finish(&out, &log, &said);

    fn finish(out: &std::path::Path, log: &str, said: &[String]) {
        std::fs::write(out.join("events.txt"), log).unwrap();
        for line in said {
            println!("  {line}");
        }
    }
}

#[cfg(not(all(windows, not(feature = "headless"))))]
fn main() {
    eprintln!("this drives a Character Map on Windows");
    std::process::exit(2);
}
