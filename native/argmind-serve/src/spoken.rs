//! **What argmind says that the program wrote, in the language it is being spoken to in** — since
//! 2026-11-17, the owner's *대화언어도 다국어 지원이 되어야함*, the same day the reply and the page
//! could be set to four languages and a turn the program answers itself still could not.
//!
//! *기억했습니다*, *하던 작업을 멈출게요*, *웹 칸을 닫았습니다*, a report of what argmind did by
//! itself: none of these are the model's, so neither the language line nor the rewrite in
//! `Session::decide` ever reached them, and somebody who set English heard Korean from every path
//! that is not a turn. **Each sentence is a row of four**, chosen by name where it is said rather
//! than matched from a finished string, so a sentence cannot be said without its translations
//! existing — the compiler holds the row, `tests` below hold that each column is in its own script
//! with the same holes. `{0}`, `{1}`… are filled in as given: a file name, a path, the person's own
//! words.
//!
//! **Which language** is [`crate::app`]'s `speaking`: the one set in 설정 → 생각, or else the one
//! the person last wrote in (`argmind_core::tongue::of`), Korean until they write in another —
//! the reply's own rule, *the language they used*, applied to what the program says.
//!
//! What the page shows around it is the interface's, `ui/src/words.ts`. What the work model wrote —
//! a task's closing words, an errand's result — is said as it wrote it, Korean inside a sentence
//! from here: rewriting it was measured and changed six facts in 36 (`app::REWRITES_WORK`). This
//! is the third kind: the program's own words, said.

/// Korean, English, Chinese, Japanese.
pub type Sentence = [&'static str; 4];

/// The sentence in `code`, with its holes filled — Korean for an unset or unknown code.
pub fn say(s: &Sentence, code: &str, holes: &[&str]) -> String {
    let col = match code { "en" => 1, "zh" => 2, "ja" => 3, _ => 0 };
    let mut out = s[col].to_string();
    for (i, h) in holes.iter().enumerate() {
        out = out.replace(&format!("{{{i}}}"), h);
    }
    out
}

/// **A Korean sentence another module wrote**, said in `code` when this table has it — for the
/// refusals `errands::add` and `mind::keep` return as text, which the page shows too. Anything else
/// comes back as it was.
pub fn known(korean: &str, code: &str) -> String {
    ALL.iter().find(|s| s[0] == korean).map_or_else(|| korean.to_string(), |s| say(s, code, &[]))
}

pub const DID_NOT_FOLLOW: Sentence = [
    "무슨 말인지 잘 못 알아들었어요. 다시 말해줄래요?",
    "Sorry, I didn't follow that. Could you say it again?",
    "抱歉，我没听懂。能再说一遍吗？",
    "すみません、よく分かりませんでした。もう一度言ってもらえますか？",
];
pub const KEPT: Sentence = [
    "기억했습니다 — 「{0}」. 설정 → 생각에서 보고 지울 수 있습니다",
    "Remembered — 「{0}」. You can see and delete it in Settings → Thinking",
    "记住了 — 「{0}」。可以在 设置 → 思考 里查看和删除",
    "覚えました — 「{0}」。設定 → 考え で確認・削除できます",
];
/// Each example is one the keep rule of its own language reads — a test below says so.
pub const KEEP_WHAT: Sentence = [
    "무엇을 기억할지 한 문장으로 말씀해 주세요 — 예: 「메모장 말고 VS Code를 쓴다고 기억해줘」",
    "Tell me in one sentence what to remember — e.g. 「Remember that I use VS Code, not Notepad」",
    "请用一句话告诉我要记住什么 — 例如：「记住我用VS Code，不用记事本」",
    "何を覚えておくか一文で言ってください — 例:「メモ帳じゃなくてVS Codeを使うって覚えておいて」",
];
pub const KEEP_FULL: Sentence = [
    "기억한 것이 {0}개입니다 — 하나를 지우고 다시 말씀해 주세요",
    "I'm already keeping {0} things — delete one and tell me again",
    "已经记住了 {0} 条 — 请删掉一条再告诉我",
    "覚えていることが {0} 個あります — 一つ消してからもう一度言ってください",
];
pub const UNREADABLE: Sentence = [
    "글자가 깨져서 읽을 수 없어요. 다시 말씀해 주세요.",
    "The text came through garbled and I can't read it. Could you say it again?",
    "文字乱码了，我读不出来。请再说一遍。",
    "文字が化けていて読めません。もう一度言ってください。",
];
pub const CARRYING_ON: Sentence = ["알겠습니다 — 이어서 할게요", "Got it — carrying on", "好的 — 我接着做", "わかりました — 続けます"];
pub const WEB_CLOSED: Sentence = ["웹 칸을 닫았습니다", "Closed the web pane", "已关闭网页栏", "ウェブ欄を閉じました"];
pub const WEB_OPENED: Sentence = [
    "웹 칸에 열었습니다 — {0}", "Opened in the web pane — {0}", "已在网页栏打开 — {0}", "ウェブ欄で開きました — {0}",
];
pub const WEB_SEARCHED: Sentence = [
    "웹 칸에 「{0}」 검색 결과를 띄웠습니다",
    "Showing search results for 「{0}」 in the web pane",
    "已在网页栏显示「{0}」的搜索结果",
    "ウェブ欄に「{0}」の検索結果を表示しました",
];
pub const OWN_HALTED: Sentence = [
    "스스로 하던 일을 멈출게요 — 30분 뒤에 목표로 다시 돌아갑니다",
    "I'll stop what I was doing on my own — I'll come back to the goal in 30 minutes",
    "我先停下自己在做的事 — 30 分钟后再回到目标上",
    "自分で進めていたことを止めます — 30 分後に目標に戻ります",
];
pub const WORK_HALTED: Sentence = ["하던 작업을 멈출게요", "I'll stop the task", "我停下手头的任务", "作業を止めます"];
pub const OWN_GIVES_WAY: Sentence = [
    "스스로 하던 일을 멈추고 맡기신 일부터 할게요",
    "I'll stop what I was doing on my own and do what you asked first",
    "我先停下自己在做的事，先做你交代的",
    "自分で進めていたことを止めて、頼まれたことから先にやります",
];
pub const STILL_WORKING: Sentence = [
    "하던 작업이 아직 안 끝났어요 — 끝나면 다시 말씀해 주세요",
    "The task I'm on isn't finished yet — tell me again when it is",
    "手头的任务还没做完 — 做完后请再告诉我",
    "まだ作業が終わっていません — 終わったらもう一度言ってください",
];
pub const WORK_STARTED: Sentence = [
    "작업 폴더({0})에서 해 볼게요 — 끝나면 알려 드릴게요",
    "I'll work on it in the work folder ({0}) — I'll tell you when it's done",
    "我在工作文件夹（{0}）里做 — 做完告诉你",
    "作業フォルダ（{0}）でやってみます — 終わったらお知らせします",
];
pub const NO_WORKBENCH: Sentence = [
    "이 환경에서는 파일이나 명령을 다룰 수 없습니다",
    "Files and commands can't be handled in this environment",
    "在这个环境里无法处理文件或命令",
    "この環境ではファイルやコマンドを扱えません",
];
pub const TASK_ASKS: Sentence = [
    "작업하다가 여쭤볼게요 — {0}", "A question while I work on it — {0}", "做任务时想问你一下 — {0}",
    "作業中にお尋ねします — {0}",
];
pub const TASK_NOT_FINISHED: Sentence = [
    "작업을 다 끝내지 못했습니다 — {0}", "I couldn't finish the task — {0}", "任务没能全部完成 — {0}",
    "作業を最後まで終えられませんでした — {0}",
];
pub const TASK_STOPPED: Sentence = ["작업이 멈췄습니다", "The task stopped", "任务停止了", "作業が止まりました"];
pub const LOOKED_UP: Sentence = [
    "브라우저 칸에 검색 결과를 띄웠습니다 — 읽고 정리해서 알려 드릴게요",
    "I've put the search results in the browser pane — I'll read them and tell you",
    "已在浏览器栏显示搜索结果 — 我读完整理好再告诉你",
    "ブラウザ欄に検索結果を表示しました — 読んでまとめてからお知らせします",
];
pub const NOT_LOOKED_UP: Sentence = [
    "찾아보지 못했습니다 ({0})", "I couldn't look it up ({0})", "没能查到（{0}）", "調べられませんでした（{0}）",
];
pub const ERRAND_EMPTY: Sentence = [
    "맡길 일을 적어 주세요", "Write down what to hand over", "请写下要交代的事", "頼むことを書いてください",
];
pub const ERRAND_TOO_LONG: Sentence = [
    "너무 깁니다 — 200자 안으로 적어 주세요", "That's too long — keep it under 200 characters",
    "太长了 — 请写在 200 字以内", "長すぎます — 200 文字以内で書いてください",
];
pub const ERRAND_FULL: Sentence = [
    "맡긴 일이 스무 개입니다 — 끝난 것을 지우고 다시 맡겨 주세요",
    "There are already twenty errands — delete a finished one and hand it over again",
    "交办的事已经有二十件了 — 请删掉已完成的再交给我",
    "頼んだことが二十件あります — 終わったものを消してからもう一度頼んでください",
];
pub const ERRAND_DONE: Sentence = [
    "맡기신 「{0}」 정리했습니다.", "Done with 「{0}」, which you handed me.", "你交代的「{0}」我整理好了。",
    "頼まれた「{0}」をまとめました。",
];
pub const ERRAND_ASKS: Sentence = [
    "맡기신 「{0}」 — {1}", "About 「{0}」, which you handed me — {1}", "关于你交代的「{0}」— {1}",
    "頼まれた「{0}」について — {1}",
];
pub const OWN_REPORT: Sentence = [
    "목표로 스스로 한 일 {0}가지 — {1}.", "{0} things I did by myself toward the goal — {1}.",
    "为了目标我自己做了 {0} 件事 — {1}。", "目標に向けて自分でやったこと {0} 件 — {1}。",
];
pub const OWN_FINISHED: Sentence = ["끝냈어요", "finished", "完成了", "終わりました"];
pub const OWN_GAVE_WAY: Sentence = [
    "맡기신 일 때문에 멈췄어요", "stopped for what you asked", "因为你交代的事停下了", "頼まれたことのために止めました",
];
pub const OWN_UNFINISHED: Sentence = ["다 못 했어요", "not finished", "没做完", "終わりませんでした"];
pub const OWN_FILES: Sentence = [
    " 만든 파일은 작업 폴더에 있어요.", " What I made is in the work folder.", " 做好的文件在工作文件夹里。",
    " 作ったファイルは作業フォルダにあります。",
];

/// Every sentence above — a test counts the `Sentence =` in this file so none is left out.
pub const ALL: &[&Sentence] = &[
    &DID_NOT_FOLLOW, &KEPT, &KEEP_WHAT, &KEEP_FULL, &UNREADABLE, &CARRYING_ON, &WEB_CLOSED, &WEB_OPENED,
    &WEB_SEARCHED, &OWN_HALTED, &WORK_HALTED, &OWN_GIVES_WAY, &STILL_WORKING, &WORK_STARTED, &NO_WORKBENCH,
    &TASK_ASKS, &TASK_NOT_FINISHED, &TASK_STOPPED, &LOOKED_UP, &NOT_LOOKED_UP, &ERRAND_EMPTY,
    &ERRAND_TOO_LONG, &ERRAND_FULL, &ERRAND_DONE, &ERRAND_ASKS, &OWN_REPORT, &OWN_FINISHED, &OWN_GAVE_WAY,
    &OWN_UNFINISHED, &OWN_FILES,
];

#[cfg(test)]
mod tests {
    use super::*;

    fn holes(s: &str) -> Vec<char> {
        let c: Vec<char> = s.chars().collect();
        let mut out: Vec<char> = (0..c.len().saturating_sub(2))
            .filter(|&i| c[i] == '{' && c[i + 1].is_ascii_digit() && c[i + 2] == '}').map(|i| c[i + 1]).collect();
        out.sort();
        out.dedup();
        out
    }

    /// What is outside 「」, which may quote a word as somebody has to say it.
    fn outside(s: &str) -> String {
        let mut depth = 0;
        s.chars().filter(|&c| match c {
            '「' => { depth += 1; false }
            '」' => { depth -= 1; false }
            _ => depth == 0,
        }).collect()
    }

    #[test]
    fn every_sentence_is_in_the_list_and_written_four_times_in_four_scripts() {
        let source = include_str!("spoken.rs");
        let declared = source.lines().filter(|l| l.starts_with("pub const ") && l.contains(": Sentence = [")).count();
        let distinct: std::collections::HashSet<&str> = ALL.iter().map(|s| s[0]).collect();
        assert_eq!(distinct.len(), ALL.len(), "a sentence is in ALL twice");
        assert_eq!(ALL.len(), declared, "a sentence is missing from ALL");
        let hangul = |s: &str| s.chars().any(|c| ('\u{AC00}'..='\u{D7A3}').contains(&c));
        let kana = |s: &str| s.chars().any(|c| ('\u{3040}'..='\u{30FF}').contains(&c));
        let han = |s: &str| s.chars().any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c));
        for s in ALL {
            assert!(hangul(s[0]), "{}", s[0]);
            for col in &s[1..] {
                assert_eq!(holes(col), holes(s[0]), "{} → {col}", s[0]);
            }
            let (en, zh, ja) = (outside(s[1]), outside(s[2]), outside(s[3]));
            assert!(!hangul(&en) && !kana(&en) && !han(&en), "{}", s[1]);
            assert!(!hangul(&zh) && !kana(&zh), "{}", s[2]);
            assert!(!hangul(&ja) && (kana(&ja) || s[3].trim().is_empty()), "{}", s[3]);
            //: Nothing in any column is Korean, quoted or not, except the Korean column.
            for col in &s[1..] {
                assert!(!hangul(col), "{col}");
            }
        }
    }

    #[test]
    fn the_example_of_what_to_remember_is_one_its_own_language_keeps() {
        for (code, col) in [("ko", 0), ("en", 1), ("zh", 2), ("ja", 3)] {
            let s = KEEP_WHAT[col];
            let example = s.split('「').nth(1).and_then(|r| r.split('」').next()).unwrap();
            assert!(matches!(argmind_core::mind::asked_to_keep(example), Some(argmind_core::mind::Keep::This(_))),
                    "{code}: {example}");
        }
    }

    #[test]
    fn holes_are_filled_and_an_unset_language_is_korean() {
        assert_eq!(say(&WEB_OPENED, "en", &["https://x.org"]), "Opened in the web pane — https://x.org");
        assert_eq!(say(&WEB_OPENED, "", &["https://x.org"]), "웹 칸에 열었습니다 — https://x.org");
        assert_eq!(say(&KEPT, "fr", &["x"]), say(&KEPT, "ko", &["x"]));
        assert_eq!(known("너무 깁니다 — 200자 안으로 적어 주세요", "ja"), ERRAND_TOO_LONG[3]);
        assert_eq!(known("모르는 문장", "ja"), "모르는 문장");
    }

    #[test]
    fn the_refusals_other_modules_write_are_in_the_table() {
        let mut list = Vec::new();
        assert_eq!(known(crate::errands::add(&mut list, "  ").unwrap_err(), "en"), ERRAND_EMPTY[1]);
        assert_eq!(known(crate::errands::add(&mut list, &"가".repeat(201)).unwrap_err(), "en"), ERRAND_TOO_LONG[1]);
        for i in 0..20 {
            let _ = crate::errands::add(&mut list, &format!("일 {i}"));
        }
        assert_eq!(known(crate::errands::add(&mut list, "하나 더").unwrap_err(), "en"), ERRAND_FULL[1]);
    }
}
