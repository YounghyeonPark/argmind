//! **The word rules in English, Chinese and Japanese** — since 2026-11-17, the owner's *대화언어도
//! 다국어 지원이 되어야함*. Each rule is pinned both ways: the request it must read, and a sentence
//! that looks like it and is not one. Korean is pinned where it always was, in each module's own
//! tests; what is checked here about Korean is only that a sentence with Hangul is not read by the
//! other languages' words.

use argmind_core::converse::goes_ahead;
use argmind_core::errand::{asks_to_look_up, sounds_current};
use argmind_core::mind::{asked_to_keep, worth_offering, Keep};
use argmind_core::task::{asks_about, asks_for_work};

fn kept(t: &str) -> Option<String> {
    match asked_to_keep(t) {
        Some(Keep::This(s)) => Some(s),
        Some(Keep::Vague) => Some("(vague)".into()),
        None => None,
    }
}

#[test]
fn remembering_is_read_in_four_languages_and_kept_in_the_person_s_words() {
    assert_eq!(kept("remember that I like tea more than coffee").as_deref(), Some("I like tea more than coffee"));
    assert_eq!(kept("Please remember my name is Arg.").as_deref(), Some("my name is Arg"));
    assert_eq!(kept("my birthday is March 2nd, remember that").as_deref(), Some("my birthday is March 2nd"));
    assert_eq!(kept("remember to call me boss").as_deref(), Some("call me boss"));
    assert_eq!(kept("记住我喜欢茶胜过咖啡").as_deref(), Some("我喜欢茶胜过咖啡"));
    assert_eq!(kept("我的生日是3月2日，记住").as_deref(), Some("我的生日是3月2日"));
    assert_eq!(kept("私はコーヒーより紅茶が好きだって覚えておいて").as_deref(), Some("私はコーヒーより紅茶が好きだ"));
    assert_eq!(kept("私の誕生日は3月2日だよ、覚えておいて").as_deref(), Some("私の誕生日は3月2日だよ"));
    //: Nothing that could stand alone a week later is asked back, as *이거 기억해줘* is.
    for vague in ["remember this", "remember that", "记住这个", "これを覚えておいて"] {
        assert_eq!(kept(vague).as_deref(), Some("(vague)"), "{vague}");
    }
    //: Questions, and sentences about remembering, are not requests.
    for not in ["do you remember what I said?", "Do you remember", "I remember the trip", "你记住了吗？",
                "覚えてる？", "open notepad", "thanks"] {
        assert_eq!(kept(not), None, "{not}");
    }
}

#[test]
fn a_preference_is_offered_in_its_own_language_only() {
    assert!(worth_offering("use VS Code instead of Notepad"));
    assert!(worth_offering("以后不要叫我老师"));
    assert!(worth_offering("メモ帳じゃなくてVS Codeを使って"));
    assert!(!worth_offering("what do you always do?"));
    assert!(!worth_offering("open notepad please"));
}

#[test]
fn work_is_a_thing_and_a_doing_in_each_language() {
    for work in ["write Python code that finds the primes from 1 to 100 and run it",
                 "make a list of the pdf file names in my Downloads folder",
                 "can you put together a list of the files in this folder?",
                 "open Character Map on the work screen and pick a star",
                 "用Python写一段求1到100之间质数的代码并运行", "能帮我整理一下这个文件夹的文件列表吗？",
                 "Pythonで1から100までの素数を求めるコードを書いて実行して", "このフォルダのファイル一覧をまとめてくれる？",
                 "summary.md 에 정리해줘"] {
        assert!(asks_for_work(work), "{work}");
    }
    for turn in ["open Notepad", "press the save button", "tell me how to write documents well",
                 "how do you run a Python script?", "the latest version is fine", "play my playlist",
                 "打开记事本", "告诉我写好文档的方法", "Python脚本怎么运行？",
                 "メモ帳を開いて", "文書をうまく書く方法を教えて", "Pythonスクリプトはどうやって実行するの？",
                 //: An English word inside a Korean sentence is not read as English.
                 "VS Code에서 run 버튼 눌러"] {
        assert!(!asks_for_work(turn), "{turn}");
    }
}

#[test]
fn a_question_about_how_is_not_a_request() {
    for about in ["tell me how to remove duplicates in Excel", "are you good at making plans?",
                  "how do people usually plan a trip?", "what is Python", "你擅长做计划吗？", "日程怎么管理？",
                  "スケジュール管理ってどうやるの？", "計画を立てるのは得意？"] {
        assert!(asks_about(about), "{about}");
    }
    for request in ["can you put together a list of the files in this folder?", "could you plan my trip?",
                    "will you make me some Python code?", "能给我写一段Python代码吗？", "来月の旅行の日程を組んでくれる？",
                    "make a schedule"] {
        assert!(!asks_about(request), "{request}");
    }
}

#[test]
fn a_question_about_now_is_looked_up_and_one_about_you_is_not() {
    for now in ["tell me today's weather", "what's the dollar exchange rate right now", "告诉我今天的天气",
                "现在美元汇率是多少", "今日の天気を教えて", "今のドルの為替レートはいくら"] {
        assert!(asks_to_look_up(now), "{now}");
    }
    for not in ["how are you today?", "what did you do today?", "I know the weather", "你今天心情怎么样",
                "今日の気分はどう", "what's on the screen now", "今天有点累"] {
        assert!(!asks_to_look_up(not), "{not}");
    }
    assert!(sounds_current("the latest news on the election") && !sounds_current("I know a snowy place"));
}

#[test]
fn a_bare_yes_in_each_language_and_a_request_that_starts_with_one() {
    for yes in ["yes", "Go ahead!", "do it", "sure", "好", "好的", "执行吧", "はい", "お願いします", "やって"] {
        assert!(goes_ahead(yes), "{yes}");
    }
    for not in ["yes, open notepad", "go ahead and open the file", "好想去旅行", "はい、メモ帳を開いて", "いいよ",
                "thanks"] {
        assert!(!goes_ahead(not), "{not}");
    }
}
