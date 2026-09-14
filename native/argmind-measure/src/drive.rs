//! Driving a conversation against a fixture — what every history-carrying harness does.
//!
//! **`decide()`, not `think()`.** Nothing is performed: a harness that clicked would measure the
//! fixture rather than the model, and the fixture cannot change. The offer bookkeeping is done here
//! rather than by the core, which mirrors `app.py` — and is itself a finding, recorded in
//! `tests/test_loop_golden.py`: `think()` never writes the `offered` channel, so **every interface
//! has to remember to**.

use argmind_core::contract::{Eyes, Hands, Speaks, Widget};
use argmind_core::intent::Trigger;
use argmind_core::session::Session;

/// Eyes that always answer with the same captured screen.
pub struct Fixed(pub Vec<Widget>);

impl Eyes for Fixed {
    fn foreground(&mut self) -> Vec<Widget> {
        self.0.clone()
    }
    fn timed_foreground(&mut self) -> (Vec<Widget>, f64) {
        (self.0.clone(), 0.0)
    }
}

/// Hands that refuse to do anything, loudly if asked.
pub struct Nothing;

impl Hands for Nothing {
    fn live(&self) -> bool {
        false
    }
    fn click(&mut self, w: &Widget) {
        panic!("a measurement clicked {:?}; nothing here may touch the world", w.name);
    }
    fn type_text(&mut self, _text: &str) {
        panic!("a measurement typed");
    }
    fn press(&mut self, _key: &str) {
        panic!("a measurement pressed a key");
    }
}

/// One turn of a driven conversation, in the shape the result files carry.
#[derive(Debug, Clone)]
pub struct Turn {
    pub i: usize,
    pub said: String,
    pub reply: String,
    pub clicked: Vec<String>,
    pub clicked_roles: Vec<String>,
    pub n_effects: usize,
    pub history_lines: usize,
    pub prompt_tokens: Option<i64>,
    pub out_tokens: Option<i64>,
    pub reason: String,
    pub thought_chars: usize,
    pub seconds: f64,
}

impl Turn {
    pub fn clicks(&self) -> Vec<(String, String)> {
        self.clicked.iter().cloned().zip(self.clicked_roles.iter().cloned()).collect()
    }
}

/// Feed a conversation to the model, line by line, and record every turn.
pub fn drive<M: Speaks>(model: M, widgets: &[Widget], lines: &[String],
                        remember_turns: Option<i64>) -> Vec<Turn> {
    let mut session = Session::new(Fixed(widgets.to_vec()), Nothing, model);
    session.remember_turns = remember_turns;
    let mut turns = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        session.observe();
        let before = session.history.len();
        let t0 = std::time::Instant::now();
        let intent = session.decide(&Trigger::said(line));
        let took = t0.elapsed().as_secs_f64();
        let clicked: Vec<String> = intent.effects.iter()
            .filter(|e| e.kind == "click").map(|e| e.target.clone()).collect();
        let clicked_roles: Vec<String> = intent.effects.iter()
            .filter(|e| e.kind == "click").map(|e| e.role.clone()).collect();
        turns.push(Turn { i, said: line.clone(), reply: intent.say.clone(), clicked,
                          clicked_roles, n_effects: intent.effects.len(), history_lines: before,
                          prompt_tokens: None, out_tokens: None, reason: String::new(),
                          thought_chars: 0, seconds: (took * 1000.0).round() / 1000.0 });
        session.history.push(("them".into(), line.clone()));
        if !intent.say.is_empty() {
            session.history.push(("you".into(), intent.say.clone()));
        }
        //: **The bookkeeping the core does not do.** Every interface files its own offers, and one
        //: that forgets loses the standing-offer block without any error saying so.
        session.remember_doing(&[], &[], &intent.effects, &[]);
    }
    turns
}
