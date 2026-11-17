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


/// **[`WRITTEN`] in English, Chinese and Japanese**, one row each in the same order, so every
/// sentence keeps the source and the accepted paths of the Korean it translates — since
/// 2026-11-17, the owner's *대화언어도 다국어 지원이 되어야함*. Translated **before any word rule
/// was written for those languages**, as a person would type each to an assistant; *the topic* of
/// a sentence was kept where a literal rendering would not be said (*영어로 번역해줘* became a
/// translation into another language, since nobody asks for their own). `--bin tongues` reads them.
pub const OTHER_TONGUES: &[[&str; 3]] = &[
    ["I'm planning a 3-day, 2-night trip to Japan in November for 2 adults — find the cheapest flights and hotel and write up a schedule",
     "我想11月去日本玩三天两夜，两个成人，帮我找最便宜的机票和酒店，再写一份行程安排",
     "11月に2泊3日で日本旅行を計画したいんだけど、大人2人で一番安い航空券とホテルを探してスケジュールを作って"],
    ["Find flights and a hotel to Japan for 3 days 2 nights in November and make a schedule",
     "帮我找11月去日本三天两夜的机票和酒店，做个行程表",
     "11月に2泊3日で日本へ行く航空券とホテルを探してスケジュールを作って"],
    ["Japan trip plan for November, 3 days 2 nights, 2 adults — research flights and hotels and put together a sensible schedule",
     "11月日本旅行计划，三天两夜，两个成人，调查一下机票和酒店，制定一个合适的行程",
     "11月の日本旅行の計画、2泊3日で大人2人、航空券とホテルを調べて適切なスケジュールを立てて"],
    ["hi", "你好", "こんにちは"],
    ["tell me today's weather", "告诉我今天的天气", "今日の天気を教えて"],
    ["what's the weather in Seoul today", "今天首尔天气怎么样", "今日のソウルの天気を教えて"],
    ["open my mail", "打开邮件", "メールを開いて"],
    // work
    ["make a list of the pdf file names in my Downloads folder", "把下载文件夹里的pdf文件名整理成一个列表",
     "ダウンロードフォルダにあるpdfファイルの名前を一覧にまとめて"],
    ["make me an Excel budget for this month", "帮我做一个这个月的记账Excel表", "今月の家計簿をエクセルで作って"],
    ["write a draft of the meeting minutes. The topic is the new product launch schedule",
     "帮我写一份会议纪要草稿，主题是新产品发布日程", "議事録の下書きを作成して。テーマは新製品の発売スケジュール"],
    ["make a simple to-do list web page", "做一个简单的待办事项管理网页", "簡単なToDo管理のウェブページを作って"],
    ["write a blog post, the topic is autumn camping", "写一篇博客，主题是秋季露营", "ブログ記事を一つ書いて、テーマは秋のキャンプ"],
    ["plan a 2-day, 1-night trip itinerary for Busan focused on good restaurants",
     "帮我安排一个釜山两天一夜、以美食为主的旅行行程", "釜山1泊2日、グルメ中心で旅行の日程を組んで"],
    ["compare three laptops under 1 million won and recommend one", "比较三款100万韩元以下的笔记本电脑并推荐一下",
     "100万ウォン以下のノートパソコンを3つ比較しておすすめして"],
    ["research 5 books for learning about stocks and summarize them", "调查5本学习股票的书并整理出来",
     "株の勉強になる本を5冊調べてまとめて"],
    ["compare rental car prices on Jeju Island", "比较一下济州岛租车的价格", "済州島のレンタカーの値段を比較して"],
    ["delete the duplicate rows in the Excel file", "删掉Excel文件里重复的行", "エクセルファイルの重複した行を削除して"],
    ["organize my photos folder by date", "把照片文件夹按日期整理一下", "写真フォルダを日付ごとに整理して"],
    ["write Python code that finds the primes from 1 to 100 and run it", "用Python写一段求1到100之间质数的代码并运行",
     "Pythonで1から100までの素数を求めるコードを書いて実行して"],
    ["proofread my cover letter", "帮我修改一下自我介绍信", "自己紹介書を添削して"],
    ["translate this into French: can we move tomorrow's meeting to 3?", "翻译成英文：明天的会议可以推到三点吗",
     "英語に翻訳して：明日の会議を3時に延ばしてもいいですか"],
    ["summarize this paper", "总结一下这篇论文", "この論文を要約して"],
    ["make a schedule for next week's team workshop", "做一份下周团队研讨会的日程表", "来週のチームワークショップの日程表を作って"],
    ["research three competitors' services and write it up as a report", "调查三家竞争对手的服务，写成报告",
     "競合3社のサービスを調べてレポートにまとめて"],
    // lookup
    ["what's the dollar exchange rate right now", "现在美元汇率是多少", "今のドルの為替レートはいくら"],
    ["how's the KOSPI doing today", "今天KOSPI怎么样", "今日のKOSPIはどう"],
    ["tell me tomorrow's weather in Busan", "告诉我明天釜山的天气", "明日の釜山の天気を教えて"],
    ["what movies have come out lately", "最近上映了什么电影", "最近公開された映画って何がある"],
    ["how's the fine dust in Seoul right now", "现在首尔的雾霾怎么样", "今のソウルのPM2.5はどう"],
    // web
    ["open YouTube", "打开YouTube", "YouTubeを開いて"],
    ["search for Son Heung-min on Naver", "在Naver上搜索孙兴慜", "Naverでソン・フンミンを検索して"],
    ["open Coupang and find wet wipes", "打开Coupang找一下湿巾", "Coupangを開いてウェットティッシュを探して"],
    ["close the web", "关闭网页", "ウェブを閉じて"],
    ["go to GitHub", "进入GitHub", "GitHubに入って"],
    // keep
    ["remember that I like tea more than coffee", "记住我喜欢茶胜过咖啡", "私はコーヒーより紅茶が好きだって覚えておいて"],
    ["my birthday is March 2nd, remember that", "我的生日是3月2日，记住", "私の誕生日は3月2日だよ、覚えておいて"],
    ["from now on, speak to me formally", "以后说话用敬语", "これからは敬語で話して"],
    // the turn: the screen
    ["open Notepad", "打开记事本", "メモ帳を開いて"],
    ["close this window", "关闭这个窗口", "このウィンドウを閉じて"],
    ["press the save button", "按保存按钮", "保存ボタンを押して"],
    ["save this document", "保存这个文档", "この文書を保存して"],
    ["open Excel", "打开Excel", "エクセルを開いて"],
    ["rename this file", "给这个文件改个名", "このファイルの名前を変えて"],
    ["copy and paste it", "复制粘贴一下", "コピーして貼り付けて"],
    // the turn: talk
    ["thanks", "谢谢", "ありがとう"],
    ["who are you", "你是谁", "あなたは誰"],
    ["I'm a bit tired today", "今天有点累", "今日はちょっと疲れた"],
    ["what is Python?", "Python是什么？", "Pythonって何？"],
    ["what should I have for lunch", "午饭吃什么好呢", "お昼何食べようかな"],
    ["I want to go on a trip", "好想去旅行", "旅行に行きたいな"],
    ["tell me how to write documents well", "告诉我写好文档的方法", "文書をうまく書く方法を教えて"],
    ["what is a code review?", "代码审查是什么？", "コードレビューって何？"],
    ["how do you manage a schedule?", "日程怎么管理？", "スケジュール管理ってどうやるの？"],
    ["are you good at making plans?", "你擅长做计划吗？", "計画を立てるのは得意？"],
    ["it won't run", "运行不了", "実行できない"],
    // held
    ["tell me how to write a good report", "告诉我怎么写好报告", "レポートを上手に書く方法を教えて"],
    ["tell me how to remove duplicates in Excel", "告诉我在Excel里删除重复项的方法", "エクセルで重複を削除する方法を教えて"],
    ["how do people usually plan a trip?", "旅行计划一般怎么制定？", "旅行の計画って普通どうやって立てるの？"],
    ["do you like writing code?", "你喜欢写代码吗？", "コードを書くのは好き？"],
    ["are you good at organizing files?", "你整理文件在行吗？", "ファイルの整理は得意なほう？"],
    ["do you know how to translate too?", "你也会翻译吗？", "翻訳もできるの？"],
    ["when making a schedule, what should I do first?", "排日程的时候应该先做什么？", "スケジュールを組むとき、まず何をすればいい？"],
    ["how do you run a Python script?", "Python脚本怎么运行？", "Pythonスクリプトはどうやって実行するの？"],
    ["can you put together a list of the files in this folder?", "能帮我整理一下这个文件夹的文件列表吗？",
     "このフォルダのファイル一覧をまとめてくれる？"],
    ["could you plan my trip itinerary for next month?", "能帮我安排一下下个月的旅行日程吗？", "来月の旅行の日程を組んでくれる？"],
    ["put together how to write a report and make it into a document", "把写报告的方法整理成一份文档",
     "レポートの書き方をまとめて文書にして"],
    ["make a plan for how to organize the Excel file and organize it", "制定一个整理Excel文件的计划然后整理好",
     "エクセルファイルをどう整理するか計画を立てて整理して"],
    ["will you make me some Python code?", "能给我写一段Python代码吗？", "Pythonのコードを一つ作ってくれる？"],
    ["please make a meeting schedule", "请做一份会议日程表", "会議の日程表を作ってください"],
];

/// [`WRITTEN`] in one language — `"ko"` as written, `"en"`, `"zh"` or `"ja"` from
/// [`OTHER_TONGUES`] — with each sentence's source and accepted paths.
pub fn written_in(code: &str) -> Vec<(String, &'static str, Vec<&'static str>)> {
    let col = match code { "en" => Some(0), "zh" => Some(1), "ja" => Some(2), _ => None };
    WRITTEN.iter().enumerate().map(|(i, w)| {
        let text = col.map_or(w.text, |c| OTHER_TONGUES[i][c]);
        (text.to_string(), w.source, w.ok.to_vec())
    }).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn every_written_sentence_has_its_three_translations() {
        assert_eq!(super::OTHER_TONGUES.len(), super::WRITTEN.len());
    }
}
