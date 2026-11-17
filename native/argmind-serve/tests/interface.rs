//! **The page in four languages: the table is whole, and what the page can show is in it.**
//!
//! Since 2026-11-17, the owner's *인터페이스언어도 국가별로 설정할수 있게 해줘*. The page and the
//! server still write Korean; `ui/src/speak.ts` translates what reaches the screen from one table,
//! `ui/src/words.ts`, and a sentence with no row simply stays Korean. That fails quietly — a
//! Japanese page with one Korean button looks like a page somebody forgot — so this file is where
//! it fails loudly:
//!
//! * **every row is whole**: four columns, the same holes in each, and no column written in another
//!   language's script (a Korean word inside 「」 is allowed, because 「멈춰」 is what a person has
//!   to say for argmind to stop, whatever the page is written in);
//! * **every Korean sentence the page itself writes has a row** — the text and the `title`,
//!   `placeholder` and `alt` of `index.html`, and every Korean string in `app.ts`;
//! * **what the server sends to be read has a row** where a person meets it first: the setup
//!   panel (`needs.rs`), the window that asks (`fix.rs`, every variant), the permission names, and
//!   the timeline's own words. The server's other notes are counted and printed, not required.
//!
//! The matcher here mirrors `speak.ts` — exact, then templates, then line by line, a bullet, parts
//! joined by ` · ` — so *covered* means *the page would translate it*, not *a row mentions it*.
//! One weaker rule is used for pieces of a sentence the page assembles at run time (*실패: *
//! before a tool's own words): such a piece passes when a template contains it, and those passes
//! are printed so they can be read.

use std::collections::HashSet;

fn root(rel: &str) -> String {
    let at = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(rel);
    std::fs::read_to_string(&at).unwrap_or_else(|e| panic!("{}: {e}", at.display()))
}

fn rows() -> Vec<[String; 4]> {
    let text = root("ui/src/words.ts");
    let begin = text.find("// words:begin").expect("words:begin");
    let end = text.find("// words:end").expect("words:end");
    text[begin..end].lines().map(str::trim).filter(|l| l.starts_with('['))
        .map(|l| {
            let v: Vec<String> = serde_json::from_str(l.trim_end_matches(','))
                .unwrap_or_else(|e| panic!("{l}: {e}"));
            assert_eq!(v.len(), 4, "{l}");
            [v[0].clone(), v[1].clone(), v[2].clone(), v[3].clone()]
        })
        .collect()
}

fn hangul(s: &str) -> bool {
    s.chars().any(|c| ('\u{AC00}'..='\u{D7A3}').contains(&c))
}
fn kana(s: &str) -> bool {
    s.chars().any(|c| ('\u{3040}'..='\u{30FF}').contains(&c))
}
fn han(s: &str) -> bool {
    s.chars().any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c))
}

/// The holes of a template, by number — `{0}`, `{0~}`, `{0#}` are all hole 0.
fn holes(s: &str) -> Vec<char> {
    let mut out: Vec<char> = Vec::new();
    let c: Vec<char> = s.chars().collect();
    for i in 0..c.len() {
        if c[i] == '{' && i + 2 < c.len() && c[i + 1].is_ascii_digit()
            && (c[i + 2] == '}' || (i + 3 < c.len() && "~#".contains(c[i + 2]) && c[i + 3] == '}')) {
            out.push(c[i + 1]);
        }
    }
    out.sort();
    out.dedup();
    out
}

/// What is outside 「」, where a translation must not carry Korean.
fn outside_quotes(s: &str) -> String {
    let mut out = String::new();
    let mut depth = 0;
    for ch in s.chars() {
        match ch {
            '「' => depth += 1,
            '」' => depth -= 1,
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    out
}

#[test]
fn every_row_is_whole_and_written_in_its_own_script() {
    let all = rows();
    assert!(all.len() > 300, "{} rows", all.len());
    let mut seen = HashSet::new();
    for [ko, en, zh, ja] in &all {
        assert!(seen.insert(ko.clone()), "{ko} is in the table twice");
        //: A row with no Korean of its own is still a row when a hole in it is translated —
        //: `{0~} — {1#} Hz` is how *목소리 1 — 136 Hz* becomes *Voice 1 — 136 Hz*.
        assert!(hangul(ko) || ko.contains("~}"), "{ko} has nothing to translate");
        let h = holes(ko);
        for col in [en, zh, ja] {
            assert_eq!(holes(col), h, "{ko} → {col}: the holes differ");
            assert!(!col.contains("~}") && !col.contains("#}"), "{col}: only the Korean says how a hole is read");
        }
        let (en_, zh_, ja_) = (outside_quotes(en), outside_quotes(zh), outside_quotes(ja));
        assert!(!hangul(&en_) && !kana(&en_) && !han(&en_), "English is not English: {en}");
        assert!(!hangul(&zh_) && !kana(&zh_), "Chinese is not Chinese: {zh}");
        assert!(!hangul(&ja_), "Japanese is not Japanese: {ja}");
    }
}

// ---- the matcher, as speak.ts has it -----------------------------------------------------------

enum Part {
    Lit(String),
    Any,
    Number,
}

struct Table {
    exact: HashSet<String>,
    templates: Vec<Vec<Part>>,
    /// Each template with every hole written `{}`, for the weaker rule.
    shapes: Vec<String>,
}

fn table() -> Table {
    let mut t = Table { exact: HashSet::new(), templates: Vec::new(), shapes: Vec::new() };
    for [ko, ..] in rows() {
        t.shapes.push(shape(&ko));
        if holes(&ko).is_empty() {
            t.exact.insert(ko);
            continue;
        }
        let c: Vec<char> = ko.chars().collect();
        let (mut parts, mut lit, mut i) = (Vec::new(), String::new(), 0);
        while i < c.len() {
            if c[i] == '{' && i + 2 < c.len() && c[i + 1].is_ascii_digit() {
                let (kind, len) = match c[i + 2] {
                    '}' => (Part::Any, 3),
                    '~' => (Part::Any, 4),
                    '#' => (Part::Number, 4),
                    _ => { lit.push(c[i]); i += 1; continue; }
                };
                if !lit.is_empty() {
                    parts.push(Part::Lit(std::mem::take(&mut lit)));
                }
                parts.push(kind);
                i += len;
            } else {
                lit.push(c[i]);
                i += 1;
            }
        }
        if !lit.is_empty() {
            parts.push(Part::Lit(lit));
        }
        t.templates.push(parts);
    }
    t
}

/// Every hole, whatever kind, written `{}`.
fn shape(s: &str) -> String {
    let mut out = String::new();
    let c: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < c.len() {
        if c[i] == '{' {
            if let Some(j) = c[i..].iter().position(|&x| x == '}') {
                let inside: String = c[i + 1..i + j].iter().collect();
                if inside.chars().all(|x| x.is_ascii_digit() || "~#".contains(x)) && !inside.is_empty() {
                    out.push_str("{}");
                    i += j + 1;
                    continue;
                }
            }
        }
        out.push(c[i]);
        i += 1;
    }
    out
}

fn fits(parts: &[Part], text: &[char]) -> bool {
    match parts.first() {
        None => text.is_empty(),
        Some(Part::Lit(l)) => {
            let l: Vec<char> = l.chars().collect();
            text.len() >= l.len() && text[..l.len()] == l[..] && fits(&parts[1..], &text[l.len()..])
        }
        Some(hole) => (1..=text.len()).any(|end| {
            (match hole {
                Part::Number => text[..end].iter().enumerate()
                    .all(|(k, c)| c.is_ascii_digit() || ".,".contains(*c) || (k == 0 && *c == '-')),
                _ => true,
            }) && fits(&parts[1..], &text[end..])
        }),
    }
}

fn covered(t: &Table, text: &str) -> bool {
    let core = text.trim();
    if !hangul(core) || t.exact.contains(core) {
        return true;
    }
    let chars: Vec<char> = core.chars().collect();
    if t.templates.iter().any(|p| fits(p, &chars)) {
        return true;
    }
    if core.contains('\n') {
        return core.split('\n').all(|l| covered(t, l));
    }
    if let Some(rest) = core.strip_prefix("· ") {
        return covered(t, rest);
    }
    if core.contains(" · ") {
        return core.split(" · ").all(|p| covered(t, p));
    }
    false
}

/// A string from source, with its holes written `{}`: covered as the page would see it, with `1`
/// in every hole — or, more weakly, as a piece some template holds.
fn source_covered(t: &Table, written: &str, weak: &mut Vec<String>) -> bool {
    for line in written.split('\n') {
        let line = line.trim();
        if !hangul(line) {
            continue;
        }
        if covered(t, &line.replace("{}", "1")) {
            continue;
        }
        let piece = line.trim_matches(|c: char| c.is_whitespace() || "·—:(),".contains(c))
            .trim_start_matches("{}").trim_end_matches("{}").trim();
        if !piece.is_empty() && (covered(t, &piece.replace("{}", "1")) || t.shapes.iter().any(|s| s.contains(piece))) {
            weak.push(line.to_string());
            continue;
        }
        return false;
    }
    true
}

// ---- the page --------------------------------------------------------------------------------

/// The text and the three attributes of `index.html`, outside comments, styles and scripts.
fn page_text() -> Vec<String> {
    let mut html = root("ui/src/index.html");
    for (open, close) in [("<!--", "-->"), ("<style>", "</style>"), ("<script", "</script>")] {
        while let Some(a) = html.find(open) {
            let b = html[a..].find(close).map(|b| a + b + close.len()).unwrap_or(html.len());
            html.replace_range(a..b, " ");
        }
    }
    let mut out = Vec::new();
    for attr in ["title=\"", "placeholder=\"", "alt=\""] {
        for (i, _) in html.match_indices(attr) {
            let rest = &html[i + attr.len()..];
            out.push(rest[..rest.find('"').unwrap_or(0)].to_string());
        }
    }
    let mut inside = false;
    let mut text = String::new();
    for ch in html.chars() {
        match ch {
            '<' => { inside = true; out.push(std::mem::take(&mut text)); }
            '>' => inside = false,
            _ if !inside => text.push(ch),
            _ => {}
        }
    }
    out.into_iter().map(|s| s.trim().to_string()).filter(|s| hangul(s)).collect()
}

/// Every string literal in a TypeScript file, `${…}` written `{}` — and the strings inside it too.
fn ts_strings(src: &str) -> Vec<String> {
    let c: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    fn code(c: &[char], mut i: usize, until: Option<char>, out: &mut Vec<String>) -> usize {
        let mut depth = 0;
        while i < c.len() {
            match c[i] {
                '/' if c.get(i + 1) == Some(&'/') => { while i < c.len() && c[i] != '\n' { i += 1; } }
                '/' if c.get(i + 1) == Some(&'*') => {
                    i += 2;
                    while i + 1 < c.len() && !(c[i] == '*' && c[i + 1] == '/') { i += 1; }
                    i += 2;
                }
                '"' | '\'' => {
                    let q = c[i];
                    let mut s = String::new();
                    i += 1;
                    while i < c.len() && c[i] != q && c[i] != '\n' {
                        if c[i] == '\\' { i += 1; }
                        if i < c.len() { s.push(c[i]); }
                        i += 1;
                    }
                    out.push(s);
                    i += 1;
                }
                '`' => {
                    let mut s = String::new();
                    i += 1;
                    while i < c.len() && c[i] != '`' {
                        if c[i] == '\\' { i += 1; if i < c.len() { s.push(c[i]); } i += 1; continue; }
                        if c[i] == '$' && c.get(i + 1) == Some(&'{') {
                            s.push_str("{}");
                            i = code(c, i + 2, Some('}'), out);
                            continue;
                        }
                        s.push(c[i]);
                        i += 1;
                    }
                    out.push(s);
                    i += 1;
                }
                '{' => { depth += 1; i += 1; }
                '}' if until == Some('}') && depth == 0 => return i + 1,
                '}' => { depth -= 1; i += 1; }
                _ => i += 1,
            }
        }
        i
    }
    code(&c, 0, None, &mut out);
    out
}

/// Strings in `app.ts` that are never shown — sent to the server, or matched against what it sent.
const NOT_SHOWN: [&str; 7] = ["멈춰", "들음", "말함", "알아챔", "보는 창", "기억 제안", "찾아보기 제안"];

#[test]
fn every_korean_sentence_the_page_writes_has_a_row() {
    let t = table();
    let mut missing = Vec::new();
    let mut weak = Vec::new();
    for s in page_text() {
        if !covered(&t, &s) {
            missing.push(format!("index.html: {s}"));
        }
    }
    for s in ts_strings(&root("ui/src/app.ts")).into_iter().filter(|s| hangul(s)) {
        if NOT_SHOWN.contains(&s.as_str()) {
            continue;
        }
        if !source_covered(&t, &s, &mut weak) {
            missing.push(format!("app.ts: {s}"));
        }
    }
    for w in &weak {
        println!("  as a piece: {w}");
    }
    assert!(missing.is_empty(), "no row for {} of the page's sentences:\n{}", missing.len(), missing.join("\n"));
}

// ---- what the server sends ---------------------------------------------------------------------

/// Every string literal in a Rust file before its tests, unescaped, holes written `{}`.
fn rust_strings(src: &str) -> Vec<String> {
    let src = &src[..src.find("#[cfg(test)]").unwrap_or(src.len())];
    let c: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < c.len() {
        if c[i] == '/' && c.get(i + 1) == Some(&'/') {
            while i < c.len() && c[i] != '\n' { i += 1; }
        } else if c[i] == '\'' {
            //: A char literal, or a lifetime.
            if c.get(i + 1) == Some(&'\\') {
                i += 2;
                while i < c.len() && c[i] != '\'' { i += 1; }
                i += 1;
            } else if c.get(i + 2) == Some(&'\'') {
                i += 3;
            } else {
                i += 1;
            }
        } else if c[i] == '"' {
            let mut s = String::new();
            i += 1;
            while i < c.len() && c[i] != '"' {
                if c[i] == '\\' {
                    i += 1;
                    match c.get(i) {
                        Some('n') => s.push('\n'),
                        Some('t') => s.push('\t'),
                        Some('\n') => { while i + 1 < c.len() && c[i + 1].is_whitespace() { i += 1; } }
                        Some('u') => {
                            let end = c[i..].iter().position(|&x| x == '}').map(|p| i + p).unwrap_or(i);
                            let hex: String = c[i + 2..end].iter().collect();
                            if let Some(ch) = u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32) { s.push(ch); }
                            i = end;
                        }
                        Some(other) => s.push(*other),
                        None => {}
                    }
                    i += 1;
                    continue;
                }
                s.push(c[i]);
                i += 1;
            }
            i += 1;
            out.push(s);
        } else {
            i += 1;
        }
    }
    //: Format holes — `{}`, `{name}`, `{:.1}`, `{x:?}` — written `{}`; `{{` and `}}` are braces.
    out.into_iter().map(|s| {
        let mut o = String::new();
        let c: Vec<char> = s.chars().collect();
        let mut k = 0;
        while k < c.len() {
            if c[k] == '{' && c.get(k + 1) == Some(&'{') { o.push('{'); k += 2; continue; }
            if c[k] == '}' && c.get(k + 1) == Some(&'}') { o.push('}'); k += 2; continue; }
            if c[k] == '{' {
                if let Some(p) = c[k..].iter().position(|&x| x == '}') {
                    o.push_str("{}");
                    k += p + 1;
                    continue;
                }
            }
            o.push(c[k]);
            k += 1;
        }
        o
    }).collect()
}

/// Lines written for the terminal by `argmind check`, never sent to the page.
const TERMINAL_ONLY: [&str; 3] = ["필요한 건 다 있습니다.", "{} 은(는) argmind가 할 수 있는 게 없습니다.",
                                  "{}개가 없습니다. argmind setup 을 실행하면 {}개에 대해 물어봅니다."];

#[test]
fn what_a_person_meets_first_from_the_server_has_a_row() {
    let t = table();
    let mut weak = Vec::new();
    let mut missing = Vec::new();
    //: The files a person reads on the first run: what is missing, the window that asks, the
    //: builds and downloads it reports as it goes, and the timeline's own words.
    for file in ["needs.rs", "fix.rs", "decided.rs", "activity.rs", "building.rs", "fetch.rs", "ollama.rs"] {
        for s in rust_strings(&root(&format!("native/argmind-serve/src/{file}"))) {
            if !hangul(&s) || TERMINAL_ONLY.iter().any(|x| s.trim() == *x) {
                continue;
            }
            if !source_covered(&t, &s, &mut weak) {
                missing.push(format!("{file}: {s:?}"));
            }
        }
    }
    //: **The asking window as it is actually written**, every variant, from the function that
    //: writes it — the literal check above cannot see what `format!` puts together.
    for fix in every_fix() {
        for line in [fix.title().to_string(), argmind_serve::fix::describe(&fix)] {
            if !covered(&t, &line) {
                missing.push(format!("fix: {line}"));
            }
        }
    }
    for key in argmind_serve::decided::DECIDABLE {
        let what = argmind_serve::decided::what(key);
        if !covered(&t, what) {
            missing.push(format!("decided: {what}"));
        }
    }
    for (event, data) in [("them", "x"), ("say", "x"), ("noticed", "x"), ("note", "x"), ("attending", "x"),
                          ("keep-offer", "x"), ("lookup-offer", "x"),
                          ("propose", "{\"label\":\"클릭  button “OK”\"}"),
                          ("ran", "{\"label\":\"클릭  button “OK”\",\"confirmed\":true}"),
                          ("ran", "{\"label\":\"키  enter\",\"confirmed\":false}"),
                          ("gone", "{\"what\":\"OK\"}"), ("held", "1")] {
        if let Some((kind, text)) = argmind_serve::activity::reading(event, data) {
            if !covered(&t, kind) {
                missing.push(format!("activity kind: {kind}"));
            }
            if !["them", "say", "noticed", "attending", "keep-offer", "lookup-offer"].contains(&event)
                && !covered(&t, &text) {
                missing.push(format!("activity {event}: {text}"));
            }
        }
    }
    for w in &weak {
        println!("  as a piece: {w}");
    }
    assert!(missing.is_empty(), "no row for {}:\n{}", missing.len(), missing.join("\n"));
}

/// **The rest, counted.** The server's other notes — refusals, hearing, the camera, the web pane —
/// are in the table where they were found; this prints what is not, so the number is seen rather
/// than assumed. It asserts only that most are there, which is a floor, not a promise.
#[test]
fn the_servers_other_sentences_are_counted() {
    let t = table();
    let (mut have, mut all, mut weak) = (0, 0, Vec::new());
    let mut without = Vec::new();
    for file in ["app.rs", "chosen.rs", "errands.rs", "mind.rs", "platform.rs", "recogniser.rs",
                 "speaker.rs", "wonders.rs", "people.rs", "faces.rs"] {
        for s in rust_strings(&root(&format!("native/argmind-serve/src/{file}"))) {
            if !hangul(&s) {
                continue;
            }
            all += 1;
            if source_covered(&t, &s, &mut weak) {
                have += 1;
            } else {
                without.push(format!("{file}: {s:?}"));
            }
        }
    }
    println!("  the server's other Korean strings with a row: {have} of {all}");
    for w in &without {
        println!("  none: {w}");
    }
    assert!(have * 10 >= all * 8, "{have} of {all}");
}

fn every_fix() -> Vec<argmind_serve::fix::Fix> {
    use argmind_serve::fix::Fix;
    let all = vec![
        Fix::PullModel { model: "qwen3:8b".into(), about_gb: 5.2 },
        Fix::OpenPage { url: "https://ollama.com/download".into(),
                        why: "Ollama가 있어야 argmind가 동작합니다.".into() },
        Fix::RunElevated { command: "wsl --install --no-distribution".into(),
                           why: "음성 입력에는 받아쓰기 서버가 필요합니다.".into() },
        Fix::Install { url: "https://ollama.com/download/OllamaSetup.exe".into(), file: "OllamaSetup.exe".into(),
                       why: "Ollama가 있어야 argmind가 동작합니다.".into(), about_mb: 700 },
        Fix::Fetch { url: "https://huggingface.co/x/y".into(), into: "C:/models/x.bin".into(),
                     why: "받아쓰기 모델입니다.".into(), about_mb: 141, sha256: None },
        Fix::Fetch { url: "https://huggingface.co/x/y".into(), into: "C:/models/x.bin".into(),
                     why: "받아쓰기 모델입니다.".into(), about_mb: 141, sha256: Some("ab".into()) },
        Fix::BuildRecogniser { repo: "https://github.com/ggml-org/whisper.cpp.git".into(),
                               commit: "8a9a".repeat(10), why: "음성 입력에는 받아쓰기 서버가 필요합니다.".into(),
                               minutes: 10, built: false },
        Fix::BuildRecogniser { repo: "https://github.com/ggml-org/whisper.cpp.git".into(),
                               commit: "8a9a".repeat(10), why: "음성 입력에는 받아쓰기 서버가 필요합니다.".into(),
                               minutes: 10, built: true },
        Fix::BuildSpeaker { repo: "https://github.com/ServeurpersoCom/qwentts.cpp.git".into(),
                            commit: "90ef".repeat(10), why: "Windows 한국어 음성은 오래된 것 하나뿐이라 어색합니다.".into(),
                            minutes: 15, built: false },
        Fix::BuildSpeaker { repo: "https://github.com/ServeurpersoCom/qwentts.cpp.git".into(),
                            commit: "90ef".repeat(10), why: "Windows 한국어 음성은 오래된 것 하나뿐이라 어색합니다.".into(),
                            minutes: 15, built: true },
        Fix::StartOllama { why: "Ollama가 설치되어 있는데 실행 중이 아닙니다.".into() },
        Fix::FetchFaces { about_mb: argmind_serve::faces::BOTH_MB },
        Fix::MakeSandbox { name: "argmind-sandbox".into(), base: "Ubuntu-24.04".into(),
                           at: r"C:\Users\x\AppData\Local\argmind\sandbox".into(), about_mb: 390 },
    ];
    //: Exhaustive, so a variant added later fails to compile here until its window is checked.
    for fix in &all {
        match fix {
            Fix::PullModel { .. } | Fix::OpenPage { .. } | Fix::Install { .. } | Fix::Fetch { .. }
            | Fix::BuildRecogniser { .. } | Fix::BuildSpeaker { .. } | Fix::StartOllama { .. }
            | Fix::RunElevated { .. } | Fix::FetchFaces { .. } | Fix::MakeSandbox { .. } => {}
        }
    }
    all
}

// ---- the choice ------------------------------------------------------------------------------

#[test]
fn the_choice_is_written_down_and_stamped_into_the_page() {
    use argmind_serve::interface::{is_one, parse, render, stamped, LANGUAGES};
    for code in LANGUAGES {
        assert!(is_one(code));
        assert_eq!(parse(&render(code)), code);
    }
    //: Nothing chosen writes nothing, and a code nobody offers reads as nothing chosen.
    assert_eq!(render(""), "{}\n");
    assert_eq!(parse("{}\n"), "");
    assert_eq!(parse("{\n  \"화면 언어\": \"fr\"\n}\n"), "");
    assert!(argmind_serve::interface::choose("fr").is_err());
    //: The page learns it before drawing anything.
    assert_eq!(stamped("<body data-lang=\"__LANG__\">", "ja"), "<body data-lang=\"ja\">");
    assert_eq!(stamped("<body data-lang=\"__LANG__\">", "fr"), "<body data-lang=\"\">");
    let page = root("ui/src/index.html");
    assert!(page.contains("data-lang=\"__LANG__\""), "the page has no place for the language");
    //: The same four the page offers.
    let shown = root("ui/src/speak.ts");
    for code in LANGUAGES {
        assert!(shown.contains(&format!("[\"{code}\", ")), "{code} is not offered by the page");
    }
}

#[test]
fn choosing_travels_as_a_route() {
    use argmind_serve::{route, Request, Route};
    let post = |p: &str| route(&Request { method: "POST".into(), path: p.into(), token: None, body: String::new() });
    assert_eq!(post("/interface/ja"), Route::Interface("ja".into()));
    assert_eq!(post("/interface/follow"), Route::Interface(String::new()));
    assert_eq!(post("/interface/"), Route::Unknown);
}

/// **The tray says what the page says**: the choice, or the system's language, or Korean.
#[test]
fn the_tray_follows_the_same_choice_as_the_page() {
    use argmind_serve::interface::{for_the_tray, tray_words};
    assert_eq!(for_the_tray("ja", Some("en")), "ja", "a choice wins over the system");
    assert_eq!(for_the_tray("", Some("en")), "en");
    assert_eq!(for_the_tray("", Some("fr")), "ko");
    assert_eq!(for_the_tray("", None), "ko");
    assert_eq!(tray_words("ja"), ["表示", "終了"]);
    assert_eq!(tray_words(""), ["보이기", "끝내기"], "Korean, as it always was");
}
