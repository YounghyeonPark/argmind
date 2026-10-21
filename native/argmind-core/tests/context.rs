//! **The history fitted to the context, in whole turns, and kept in a file.**

use argmind_core::session::{estimate_tokens, history_from, history_line, recent, recent_within};

fn turn(i: usize) -> Vec<(String, String)> {
    vec![("them".into(), format!("{i}번째 질문입니다 조금 긴 문장으로")),
         ("you".into(), format!("{i}번째 대답입니다"))]
}

fn talk(turns: usize) -> Vec<(String, String)> {
    (0..turns).flat_map(turn).collect()
}

#[test]
fn the_estimate_overestimates_the_measured_samples() {
    //: 2026-10-21, qwen3:8b: these three cost 800, 600 and 760 tokens.
    let korean = "안녕하세요, 오늘 회의 자료를 정리해야 하는데 어디서부터 시작하면 좋을까요? \
지난주에 이야기한 예산안도 같이 봐야 해요. ".repeat(20);
    let english = "On screen now: [12] button \"Save\" [13] link \"Open file\" They said: open the \
settings menu please. ".repeat(20);
    assert!(estimate_tokens(&korean) >= 800, "{}", estimate_tokens(&korean));
    assert!(estimate_tokens(&english) >= 600, "{}", estimate_tokens(&english));
}

#[test]
fn more_room_holds_more_turns_and_never_part_of_one() {
    let h = talk(40);
    let small = recent_within(&h, 200);
    let large = recent_within(&h, 2000);
    assert!(large.len() > small.len());
    for kept in [&small, &large] {
        assert_eq!(kept.first().map(|l| l.0.as_str()), Some("them"), "opens on a question");
        assert_eq!(kept.last(), h.last(), "ends on the newest line");
    }
    assert_eq!(recent_within(&h, 1_000_000), h, "everything when everything fits");
    assert!(recent_within(&h, 0).is_empty());
}

#[test]
fn with_room_to_spare_it_holds_more_than_twelve_turns() {
    //: The 2026-10-21 complaint, in one line: twelve turns was the ceiling whatever the room.
    let h = talk(40);
    let by_turns = recent(&h, Some(12));
    let by_room = recent_within(&h, 3000);
    assert_eq!(by_turns.len(), 24);
    assert!(by_room.len() > 24, "{}", by_room.len());
}

#[test]
fn the_file_reads_back_and_a_broken_line_loses_only_itself() {
    let mut file = String::new();
    for (i, (who, text)) in talk(3).iter().enumerate() {
        file.push_str(&history_line(i as u64, who, text));
        file.push('\n');
    }
    file.push_str("{not json\n");
    file.push_str(&history_line(9, "them", "따옴표 \"와\" 줄\n바꿈"));
    let back = history_from(&file, 100);
    assert_eq!(back.len(), 7);
    assert_eq!(back[6].1, "따옴표 \"와\" 줄\n바꿈");
    assert_eq!(history_from(&file, 2).len(), 2, "only the last ones");
}
