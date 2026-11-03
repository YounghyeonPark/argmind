//! **Where a turn goes**, decided in one place by the word rules — so the application and
//! `--bin routing` read the same function and a measured miss is a miss the program makes.
//!
//! Until 2026-11-03 the order lived only as the order of `if`s in `app::think`, each `_instead`
//! asking its own question. The owner's trip request fell through all of them three times.

/// The paths a turn can take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// *…기억해줘* — kept by the program, no model call.
    Keep,
    /// Carried to the end by the task loop, or a running task stopped.
    Work,
    /// A question about now, looked up in the pane.
    LookUp,
    /// The pane opened, closed or searched.
    Web,
    /// The ordinary turn: a reply, and at most three things done on the screen.
    Turn,
}

impl Route {
    pub fn name(self) -> &'static str {
        match self {
            Route::Keep => "keep",
            Route::Work => "work",
            Route::LookUp => "lookup",
            Route::Web => "web",
            Route::Turn => "turn",
        }
    }
}

/// **The word rules, in the order `think` asks them.** `has_pane` because looking up and moving the
/// web need one; `working` because *멈춰* stops a task only while there is one.
pub fn by_rules(text: &str, has_pane: bool, on_web: bool, working: bool) -> Route {
    if argmind_core::mind::asked_to_keep(text).is_some() {
        return Route::Keep;
    }
    if (working && ["멈춰", "그만"].iter().any(|w| text.contains(w)))
        || argmind_core::task::asks_for_work(text) {
        return Route::Work;
    }
    if has_pane && argmind_core::errand::asks_to_look_up(text) {
        return Route::LookUp;
    }
    if has_pane {
        match crate::surface::web_move(text, on_web) {
            //: *메모장* with the pane up: the desktop comes back and the turn carries on there.
            None | Some(crate::surface::WebMove::ToDesktop) => {}
            Some(_) => return Route::Web,
        }
    }
    Route::Turn
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_owner_requests_go_where_they_went() {
        assert_eq!(by_rules("불러달라고 기억해줘", true, false, false), Route::Keep);
        assert_eq!(by_rules("11월 일본여행 계획 2박3일로 성인2명 항공권이랑 호텔 조사해서 적절한 스케쥴을 세워줘",
                            true, false, false), Route::Work);
        assert_eq!(by_rules("오늘 서울 날씨 알려줘", true, false, false), Route::LookUp);
        assert_eq!(by_rules("오늘 서울 날씨 알려줘", false, false, false), Route::Turn);
        assert_eq!(by_rules("고마워", true, false, false), Route::Turn);
        assert_eq!(by_rules("멈춰", true, false, true), Route::Work);
        assert_eq!(by_rules("멈춰", true, false, false), Route::Turn);
    }
}
