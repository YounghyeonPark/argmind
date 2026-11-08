//! **The sentences `--bin routing` and `--bin seam` are both read against**, with their accepted
//! paths, moved out of `routing.rs` on 2026-11-09 unchanged so the two cannot drift apart.

pub struct Said {
    pub text: &'static str,
    pub source: &'static str,
    pub ok: &'static [&'static str],
}

const fn s(text: &'static str, source: &'static str, ok: &'static [&'static str]) -> Said {
    Said { text, source, ok }
}

const T: &[&str] = &["turn"];
const W: &[&str] = &["work"];
const L: &[&str] = &["lookup"];
const WEB: &[&str] = &["web"];
const K: &[&str] = &["keep"];

pub const WRITTEN: &[Said] = &[
    s("11월중에 2박3일로 일본여행계획을 세우려고 하는데 2인성인으로 가장 저렴한 항공권이랑 호텔찾아서 스케쥴 작성해줘", "real", W),
    s("11월 2박3일일정으로 일본가는 항공이랑 호텔 찾아서 스케줄 만들어줘", "real", W),
    s("11월 일본여행 계획 2박3일로 성인2명 항공권이랑 호텔 조사해서 적절한 스케쥴을 세워줘", "real", W),
    s("안녕", "real", T),
    s("오늘 날씨 알려줘", "real", L),
    s("오늘 서울 날씨 알려줘", "real", L),
    s("메일 열어줘", "real", &["web", "turn"]),
    // work
    s("다운로드 폴더에 있는 pdf 파일 이름들을 목록으로 정리해줘", "written", W),
    s("이번 달 가계부 엑셀 만들어줘", "written", W),
    s("회의록 초안 작성해줘. 주제는 신제품 출시 일정", "written", W),
    s("간단한 할 일 관리 웹페이지 만들어줘", "written", W),
    s("블로그 글 하나 써줘, 주제는 가을 캠핑", "written", W),
    s("부산 1박2일 맛집 위주로 여행 일정 짜줘", "written", W),
    s("100만원 이하 노트북 세 개 비교해서 추천해줘", "written", W),
    s("주식 공부할 책 5권 조사해서 정리해줘", "written", W),
    s("제주도 렌터카 가격 비교해줘", "written", W),
    s("엑셀 파일에서 중복된 행 지워줘", "written", W),
    s("사진 폴더를 날짜별로 정리해줘", "written", W),
    s("파이썬으로 1부터 100까지 소수 구하는 코드 짜서 돌려줘", "written", W),
    s("자기소개서 첨삭해줘", "written", &["work", "turn"]),
    s("영어로 번역해줘: 내일 회의 3시로 미뤄도 될까요", "written", &["turn", "work"]),
    s("이 논문 요약해줘", "written", &["work", "turn"]),
    s("다음 주 팀 워크숍 일정표 만들어줘", "written", W),
    s("경쟁사 세 곳 서비스를 조사해서 보고서로 써줘", "written", W),
    // lookup
    s("지금 달러 환율 얼마야", "written", L),
    s("오늘 코스피 어때", "written", L),
    s("내일 부산 날씨 알려줘", "written", &["lookup", "work"]),
    s("요즘 개봉한 영화 뭐 있어", "written", &["lookup", "work"]),
    s("지금 서울 미세먼지 어때", "written", L),
    // web
    s("유튜브 열어줘", "written", WEB),
    s("네이버에서 손흥민 검색해줘", "written", &["web", "lookup"]),
    s("쿠팡 열어서 물티슈 찾아줘", "written", &["web", "work"]),
    s("웹 닫아", "written", WEB),
    s("깃허브 들어가줘", "written", WEB),
    // keep
    s("나는 커피보다 차를 좋아한다고 기억해줘", "written", K),
    s("내 생일은 3월 2일이야, 기억해줘", "written", K),
    s("앞으로 존댓말 써", "written", &["keep", "turn"]),
    // the turn: the screen
    s("메모장 열어줘", "written", T),
    s("이 창 닫아줘", "written", T),
    s("저장 버튼 눌러", "written", T),
    s("이 문서 저장해줘", "written", T),
    s("엑셀 열어줘", "written", T),
    s("이 파일 이름 바꿔줘", "written", &["turn", "work"]),
    s("복사해서 붙여넣어줘", "written", T),
    // the turn: talk
    s("고마워", "written", T),
    s("넌 누구야", "written", T),
    s("오늘 좀 피곤하다", "written", T),
    s("파이썬이 뭐야?", "written", T),
    s("점심 뭐 먹을까", "written", T),
    s("여행 가고 싶다", "written", T),
    s("문서 작성 잘하는 법 알려줘", "written", T),
    s("코드 리뷰가 뭐야?", "written", T),
    s("일정 관리 어떻게 해?", "written", T),
    s("계획 세우는 거 잘해?", "written", T),
    s("실행이 안 돼", "written", T),
    // held: written after the guard, before any arm ran
    s("보고서 잘 쓰는 방법 알려줘", "held", T),
    s("엑셀에서 중복 지우는 법 알려줘", "held", T),
    s("여행 계획은 보통 어떻게 세워?", "held", T),
    s("코드 짜는 거 좋아해?", "held", T),
    s("파일 정리 잘하는 편이야?", "held", T),
    s("번역도 할 줄 알아?", "held", T),
    s("일정 짤 때 뭐부터 해야 돼?", "held", T),
    s("파이썬 스크립트는 어떻게 실행해?", "held", T),
    s("이 폴더 파일 목록 정리해줄 수 있어?", "held", W),
    s("다음 달 여행 일정 좀 짜줄래?", "held", W),
    s("보고서 쓰는 법을 정리해서 문서로 만들어줘", "held", W),
    s("엑셀 파일 어떻게 정리할지 계획 세워서 정리해줘", "held", W),
    s("파이썬 코드 하나 만들어 줄래?", "held", W),
    s("회의 일정표 만들어주세요", "held", W),
];

/// The title a borrowed screen had, by its capture's name.
pub fn title(screen: &str) -> &'static str {
    match screen {
        "untitled---paint" => "Untitled - Paint",
        "character-map" => "Character Map",
        "system-information" => "System Information",
        s if s.starts_with("chrome-remote-desktop") => "Chrome Remote Desktop - Google Chrome",
        _ => "Visual Studio Code",
    }
}

/// The window the shipped router is shown: each borrowed screen's title, and the editor otherwise.
pub fn borrowed_or_code(all: &[(String, &'static str, Vec<&'static str>)], borrowed: &[&'static str]) -> Vec<&'static str> {
    let mut k = 0;
    all.iter().map(|(_, src, _)| if *src == "borrowed" { k += 1; borrowed[k - 1] } else { "Visual Studio Code" }).collect()
}

pub fn set() -> Vec<(String, &'static str, Vec<&'static str>)> {
    let mut all: Vec<(String, &'static str, Vec<&'static str>)> =
        WRITTEN.iter().map(|w| (w.text.to_string(), w.source, w.ok.to_vec())).collect();
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let goals: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        root.join("measure").join("goals.json")).expect("goals.json")).expect("goals");
    for g in goals["goals"].as_array().expect("goals") {
        all.push((g["goal"].as_str().unwrap_or("").to_string(), "borrowed", vec!["turn"]));
        WINDOWS.with(|w| w.borrow_mut().push(title(g["screen"].as_str().unwrap_or(""))));
    }
    let talk: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        root.join("native").join("expectations").join("talking.json")).expect("talking.json")).expect("talk");
    for t in talk["talk"].as_array().expect("talk") {
        let text = t[1].as_str().unwrap_or("").to_string();
        //: *오늘 날씨 어때?* was chit-chat when that set was written; with a pane it is a question
        //: about now, and either is right.
        let ok = if text.contains("날씨") { vec!["turn", "lookup"] } else { vec!["turn"] };
        all.push((text, "borrowed", ok));
        let first = talk["where"][0].as_str().unwrap_or("");
        WINDOWS.with(|w| w.borrow_mut().push(title(first)));
    }
    all
}

thread_local! {
    /// The window in front for each borrowed sentence, in order; real and written have none listed.
    pub static WINDOWS: std::cell::RefCell<Vec<&'static str>> = const { std::cell::RefCell::new(Vec::new()) };
}

