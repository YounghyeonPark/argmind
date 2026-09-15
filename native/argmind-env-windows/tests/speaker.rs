//! Which voice speaks — **decided from the text, with no speaker opened.**
//!
//! `SapiSpeaker::new()` enumerates voice tokens and `say` makes sound, so neither belongs in a
//! suite. The decision does: it is a pure function of the string, and it is the one that turns a
//! correct reply into silence when it goes wrong.
//!
//! **The measured asymmetry, and why this is `any` and not `most`.** An English voice given pure
//! Korean emits **0 bytes, 79 of 79 strings**. Given Korean with a number in it, it emits the
//! number alone — 46 of 46, and **0 of 20 survive a round trip**: `지우개는 22번 버튼입니다.` comes
//! back from the recogniser as `03, 04.` Heami given English is merely accented: 57 of 57 audible,
//! understood 10 of 30 against Zira's 25 of 30. So the mixed case costs a great deal on the English
//! voice and a little on the Korean one, and one Hangul character is enough to decide.
//!
//! Ported from `tests/test_env_voice.py`, which held these cases against `needs_korean`.

use argmind_env_windows::speaker::has_hangul;

/// The recorded cases, verbatim from the Python's table.
#[test]
fn any_hangul_at_all_picks_the_korean_voice() {
    for (text, korean) in [("지우개는 22번 버튼입니다.", true),
                           ("빨간색을 선택할 수 있어요.", true),
                           //: **Mixed, and the expensive direction.** Latin words inside a Korean
                           //: sentence still go to Heami, because the alternative loses the
                           //: sentence entirely rather than accenting it.
                           ("Character Grid이 열려 있어요.", true),
                           ("Okay, I'll select the red colour.", false),
                           ("BIOS Version/Date is 10/22/2025", false),
                           ("22", false),
                           ("", false),
                           //: Jamo on their own — a person typing 초성 — is Hangul.
                           ("ㄱㄴㄷ", true)] {
        assert_eq!(has_hangul(text), korean, "{text:?}");
    }
}

/// **Every block Hangul lives in**, because syllables alone would miss what a keyboard produces.
///
/// Precomposed syllables are the common case, but jamo arrive on their own from an IME mid-word and
/// the two extended blocks exist for old orthography. A range dropped here is a sentence that goes
/// silently to an English voice.
#[test]
fn all_five_hangul_blocks_count() {
    for (what, sample) in [("syllables", '가'), ("syllables, last", '힣'),
                           ("jamo", 'ᄀ'), ("compatibility jamo", 'ㄱ'),
                           ("extended-A", '\u{A960}'), ("extended-B", '\u{D7B0}')] {
        assert!(has_hangul(&sample.to_string()), "{what} ({sample:?}) was not recognised");
    }
}

/// **Nothing else is Hangul**, and the neighbours are the ones a range that was one too wide
/// would swallow.
#[test]
fn nothing_that_is_not_hangul_is_taken_for_it() {
    for (what, sample) in [("Latin", "hello"), ("digits", "2025"),
                           ("punctuation", "... !?"), ("Japanese kana", "ありがとう"),
                           ("Han", "漢字"), ("Cyrillic", "привет"),
                           //: The character immediately below the syllable block and the one
                           //: immediately above it. An off-by-one at either end is invisible in
                           //: ordinary text and shows up as a wrong voice on one rare string.
                           ("just below 가", "\u{ABFF}"), ("just above 힣", "\u{D7A4}")] {
        assert!(!has_hangul(sample), "{what} ({sample:?}) was taken for Korean");
    }
}

/// One Hangul character in a long English sentence is enough.
#[test]
fn one_character_decides_it() {
    let mostly_english = "The BIOS version is 10/22/2025 and the pane is open 요";
    assert!(has_hangul(mostly_english));
    assert!(!has_hangul(&mostly_english.replace('요', "")));
}
