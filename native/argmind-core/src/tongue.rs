//! **The word rules' shared pieces, in the four languages a person can talk to argmind in** — since
//! 2026-11-17, the owner's *대화언어도 다국어 지원이 되어야함*.
//!
//! Every rule that moves a turn without asking a model — keep (`mind::asked_to_keep`), work
//! (`task::asks_for_work`), a question about how (`task::asks_about`), a look-up
//! (`errand::asks_to_look_up`), the pane (`surface::web_move`), a yes (`converse::goes_ahead`) and a
//! stop ([`stops`]) — was a list of Korean words, so *remember that I like tea* was a turn and a
//! turn cannot keep anything. `--bin tongues` measured it: the same 69 requests went where they
//! should **60 times in Korean and 35 in each of English, Chinese and Japanese**.
//!
//! **Each language's words are read only in a sentence written in that language** ([`reads_as`]).
//! Korean's are read in every sentence, exactly as before, and the others never in a sentence with
//! Hangul in it — so *VS Code에서 run 눌러* is not a request to run code, and no Korean sentence
//! can go anywhere it did not go before. That is what lets the other three be added without
//! measuring Korean again from nothing: it is held by the gate, and `--bin tongues` prints the 69
//! Korean routes to compare.

/// **Do this language's word lists apply to this sentence?** Korean's always do — they always
/// have. English's only in a sentence of Latin letters with no Hangul, kana or Han; Chinese's only
/// in Han with no Hangul or kana; Japanese's wherever there is kana, Hangul or not.
pub fn reads_as(text: &str, code: &str) -> bool {
    let script = crate::converse::written_in(text);
    match code {
        "ko" => true,
        "en" => script == Some("en"),
        "zh" => script == Some("zh"),
        "ja" => matches!(script, Some("ja") | Some("mixed")),
        _ => false,
    }
}

/// **The language somebody wrote in, for what the program says back** — `None` when a sentence
/// does not say: no letters, or a Latin word too short to be a choice (*ok*, *yes*, *go*), since a
/// Korean speaker types those too and the next program sentence would then come back in English.
/// Hangul with kana is read as Korean, the language of most of its words.
pub fn of(text: &str) -> Option<&'static str> {
    match crate::converse::written_in(text)? {
        "ko" | "mixed" => Some("ko"),
        "en" => (text.chars().filter(|c| c.is_ascii_alphabetic()).count() >= 5).then_some("en"),
        "zh" => Some("zh"),
        "ja" => Some("ja"),
        _ => None,
    }
}

/// Lower case, with the typographic apostrophe a phone keyboard puts in *what’s* made plain.
pub fn lower(text: &str) -> String {
    text.to_lowercase().replace(['’', '‘'], "'")
}

/// **A word or phrase, whole**, in lower-cased ASCII text: *now* is not in *know*, *list* is not in
/// *playlist*. A phrase is matched on the same boundaries at both ends.
pub fn has_word(lower: &str, word: &str) -> bool {
    let b = lower.as_bytes();
    let edge = |i: usize| i >= b.len() || !(b[i].is_ascii_alphanumeric() || b[i] == b'\'');
    let mut from = 0;
    while let Some(at) = lower[from..].find(word) {
        let start = from + at;
        let end = start + word.len();
        //: *today's* is *today*: a possessive ends the word — found by `--bin tongues`, whose first
        //: run sent *tell me today's weather* to a turn.
        let possessive = lower[end..].starts_with("'s") && edge(end + 2);
        if (start == 0 || edge(start - 1)) && (edge(end) || possessive) {
            return true;
        }
        from = start + 1;
        while !lower.is_char_boundary(from) {
            from += 1;
        }
    }
    false
}

/// **A word that begins with this**, in lower-cased ASCII text — *organiz* finds *organize* and
/// *organizing*, *compar* finds *compare* and *comparison*, and *test* is not found in *latest*.
pub fn has_stem(lower: &str, stem: &str) -> bool {
    let b = lower.as_bytes();
    let mut from = 0;
    while let Some(at) = lower[from..].find(stem) {
        let start = from + at;
        if start == 0 || !(b[start - 1].is_ascii_alphanumeric() || b[start - 1] == b'\'') {
            return true;
        }
        from = start + 1;
        while !lower.is_char_boundary(from) {
            from += 1;
        }
    }
    false
}

/// **Is this the person telling argmind to stop?** *멈춰* and *그만* anywhere, as it always was;
/// *stop*, *cancel* and *abort* as words; *停下*, *停止*, *别做了*, *取消* and a bare *停*; *止めて*,
/// *やめて*, *ストップ* and *中止*. As broad as the Korean on purpose — while a task runs, stopping it
/// by mistake costs a sentence, and not stopping it costs whatever it does next.
pub fn stops(text: &str) -> bool {
    if ["멈춰", "그만"].iter().any(|w| text.contains(w)) {
        return true;
    }
    if reads_as(text, "en") {
        let low = lower(text);
        return ["stop", "cancel", "abort"].iter().any(|w| has_word(&low, w));
    }
    if reads_as(text, "zh") {
        let bare = text.trim().trim_end_matches(['!', '！', '。', '.', ' ']);
        return bare == "停" || ["停下", "停止", "停一下", "别做了", "不要做了", "不用做了", "取消"]
            .iter().any(|w| text.contains(w));
    }
    if reads_as(text, "ja") {
        return ["止めて", "やめて", "ストップ", "中止", "止まって"].iter().any(|w| text.contains(w));
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_language_s_words_are_read_only_in_that_language() {
        assert!(reads_as("메모장 열어줘", "ko") && reads_as("open notepad", "ko"));
        assert!(reads_as("open notepad", "en") && !reads_as("VS Code에서 run 눌러", "en"));
        assert!(reads_as("打开记事本", "zh") && !reads_as("メモ帳を開いて", "zh"));
        assert!(reads_as("メモ帳を開いて", "ja") && reads_as("今日の天気", "ja"));
        assert!(!reads_as("打开记事本", "ja") && !reads_as("open notepad", "ja"));
    }

    #[test]
    fn the_language_said_back_needs_enough_to_go_on() {
        assert_eq!(of("기억해줘"), Some("ko"));
        assert_eq!(of("remember that"), Some("en"));
        assert_eq!(of("ok"), None);
        assert_eq!(of("yes"), None);
        assert_eq!(of("好"), Some("zh"));
        assert_eq!(of("はい"), Some("ja"));
        assert_eq!(of("メモ장"), Some("ko"));
        assert_eq!(of("123"), None);
    }

    #[test]
    fn words_are_whole_and_stems_start_words() {
        assert!(has_word("what's the weather now", "now") && !has_word("i know", "now"));
        assert!(has_word("close the web please", "close the web") && !has_word("enclose the webs", "close the web"));
        assert!(has_stem("organizing files", "organiz") && !has_stem("the latest news", "test"));
        assert!(has_word(&lower("What’s up"), "what's"));
        assert!(has_word("tell me today's weather", "today") && !has_word("todays", "today"));
    }

    #[test]
    fn stopping_in_four_languages() {
        for yes in ["멈춰", "그만해", "stop", "Stop!", "please cancel that", "停", "停下", "别做了", "止めて", "やめて", "ストップ"] {
            assert!(stops(yes), "{yes}");
        }
        for no in ["unstoppable", "고마워", "停车场在哪", "open notepad", "ありがとう"] {
            assert!(!stops(no), "{no}");
        }
    }
}
