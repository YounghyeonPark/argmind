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
/// web need one; `working` because *멈춰* — or *stop*, *停下*, *止めて* — stops a task only while there
/// is one.
pub fn by_rules(text: &str, has_pane: bool, on_web: bool, working: bool) -> Route {
    if argmind_core::mind::asked_to_keep(text).is_some() {
        return Route::Keep;
    }
    if (working && argmind_core::tongue::stops(text))
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

/// What the model is asked, the six-way prompt `--bin routing` measured. Only its `work` is read:
/// the other paths each have a word rule that does the moving (*…기억해줘* is kept by the program,
/// never the model; the pane moves by `web_move`), so a model naming them would name a path that
/// then does nothing.
const ASK: &str = "사람이 PC 비서에게 한 말이다. 어떤 처리로 보낼지 하나만 골라라.\n\
- reply: 대화나 지식으로 바로 답할 말 (인사, 질문, 의견, 짧은 번역)\n\
- desktop: 지금 화면의 앱에서 무언가를 누르거나 입력하는 요청 (앱 열기·닫기, 버튼, 저장, 복사)\n\
- web: 웹사이트를 열거나 닫거나, 사이트에서 검색만 해 달라는 요청\n\
- lookup: 지금·오늘의 정보(날씨, 환율, 주가, 뉴스)를 찾아 알려 달라는 요청\n\
- work: 여러 단계가 필요한 일 — 조사해서 정리, 계획 세우기, 비교, 문서·파일·코드를 만들기·고치기·실행\n\
- keep: 무엇을 기억해 달라는 요청\n\n\
한 말: ";

/// The prompt, with the window in front — **without it the model sent *글씨 쓰고 싶어* in Paint to
/// work**, and the screen goals fell to 48 of 54.
pub fn model_prompt(text: &str, window: &str) -> String {
    let front = if window.is_empty() { String::new() } else { format!("지금 앞에 있는 창: {window}\n") };
    format!("{front}{ASK}{text}\n\nJSON 하나로만 답하라: {{\"route\": \"reply|desktop|web|lookup|work|keep\"}}")
}

/// The path the model named, as it named it — `?` when it named none.
pub fn model_said(reply: &str) -> String {
    argmind_core::llm::first_object(reply)
        .and_then(|v| v.get("route").and_then(|r| r.as_str()).map(str::to_string))
        .unwrap_or_else(|| "?".into())
}

/// **The rules first, and the model only where they said `Turn`** — the arm `--bin routing` judged.
/// The model can move a turn to work and nothing else, and not a question about how
/// ([`argmind_core::task::asks_about`]), which it sent to work three times in three. Returns the
/// route and, when the model was asked, what it said and how long that took.
pub fn routed(text: &str, has_pane: bool, on_web: bool, working: bool, window: &str,
              model: &mut dyn argmind_core::contract::Speaks) -> (Route, Option<(String, f64)>) {
    let rules = by_rules(text, has_pane, on_web, working);
    if rules != Route::Turn || argmind_core::task::asks_about(text) {
        return (rules, None);
    }
    let started = std::time::Instant::now();
    let said = model_said(&model.ask(&model_prompt(text, window)));
    let took = started.elapsed().as_secs_f64();
    let route = if said == "work" { Route::Work } else { Route::Turn };
    (route, Some((said, took)))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Says(&'static str, usize);
    impl argmind_core::contract::Speaks for Says {
        fn ask(&mut self, _: &str) -> String {
            self.1 += 1;
            self.0.to_string()
        }
    }

    #[test]
    fn the_model_moves_a_turn_to_work_and_nothing_else() {
        let mut work = Says("{\"route\": \"work\"}", 0);
        assert_eq!(routed("블로그 글 하나 써줘", true, false, false, "", &mut work).0, Route::Work);
        //: Not asked where the rules already decided, nor about a question about how.
        assert_eq!(routed("불러달라고 기억해줘", true, false, false, "", &mut work).0, Route::Keep);
        assert_eq!(routed("계획 세우는 거 잘해?", true, false, false, "", &mut work).0, Route::Turn);
        assert_eq!(work.1, 1);
        //: A path with a word rule of its own is not taken on the model's word.
        for other in ["web", "lookup", "keep", "desktop", "garbage"] {
            let mut m = Says(Box::leak(format!("{{\"route\": \"{other}\"}}").into_boxed_str()), 0);
            assert_eq!(routed("그거 좀 해 봐", true, false, false, "", &mut m).0, Route::Turn, "{other}");
        }
    }

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
