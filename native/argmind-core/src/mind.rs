//! What argmind is trying to be, as dials a person can turn.
//!
//! The owner, 2026-09-17, after asking what argmind actually thinks: **설정에서 생각 목표설정이
//! 가능하고 수치를 조정할수 있음. 재치, 유머, 봉사, 헌신 등등.**
//!
//! ## What this is not
//!
//! It is not a personality. It is a handful of numbers that reach the prompt as a line, and whether
//! the model does anything with them is a question about the model rather than about this file.
//! **A dial that moves nothing is worse than no dial** — it is an interface making a claim it
//! cannot keep — so the ones that ship are the ones shown to change behaviour, and the rest are
//! removed rather than left looking adjustable.
//!
//! ## The rule that makes it safe to add at all
//!
//! **With every dial at rest, [`Mind::line`] is `None` and the prompt is byte-for-byte what it was
//! before this file existed.** Prompt edits here are 0 for 5 — each raised one measured number and
//! lowered another — and the sixth that paid was a slot present only when there was something to
//! put in it. This is the same shape: nothing set, nothing said.
//!
//! So the feature cannot cost anything until somebody turns it on, and what it costs then is
//! theirs to weigh against what it buys, which is a matter of taste and is not this file's to
//! judge.

/// The dials, in the order a person reads them.
///
/// **Four, not ten.** Each one is a line in every prompt for the rest of the program's life, and a
/// long list is how a prompt stops being read — by the model as much as by anybody. These four are
/// the owner's own words; more can be added when these are shown to work.
pub const DIALS: [(&str, &str); 4] = [
    //: **A name and the gloss that goes with it, and both reach the model.** The page shows this
    //: same pair, so what a person reads beside the slider and what the model is told are one
    //: string rather than two that can drift.
    ("재치", "빠르고 영리하게"),
    ("유머", "가볍고 재미있게"),
    ("봉사", "먼저 도와주겠다고 나서기"),
    ("헌신", "하던 일을 끝까지 붙들기"),
];

/// **Korean, and that is measured rather than assumed.**
///
/// The first version of this line was English, on the argument that the rest of the prompt is and
/// that mixing languages inside one instruction was unmeasured. It is measured now, on the 44
/// Korean requests of the `saying` harness at one repeat and temperature 0, and the argument was
/// backwards: **the language of this slot decides the language of the reply.**
///
/// | what is in the slot | replies in Korean |
/// |---|---|
/// | nothing — the shipped prompt | 28 of 44 |
/// | the four dials, in English | 28, 30, 30, 31 of 44 |
/// | a content-free English sentence | **19 of 44** |
/// | a content-free Korean sentence | **42 of 44**, and 41 on a repeat |
///
/// So an English line here *costs* nine cases of a defect this project already records — *the agent
/// still answers some Korean in English* — and a Korean one buys fourteen. The dials in English did
/// neither, which is the honest reading of those four rows: they moved the wording and left the
/// language where they found it.
///
/// **This is a Korean program.** The dials are named in Korean because the owner named them, the
/// page is Korean, and the requests are Korean. Nothing here says what the right slot language is
/// for somebody who talks to argmind in English, and the measurement above cannot say either — it
/// has no English requests in it.
const AIM_AT: &str = "이런 태도를 목표로 하세요: ";
const KEEP_IN_MIND: &str = "사용자가 기억해 달라고 한 것: ";
/// **Said last, and it is the line that keeps the rest honest.** Every one of these dials is a
/// reason to say more than is wanted, and this project measured that 31 of 34 conversational turns
/// should produce no action at all.
const BUT: &str =
    " 그렇다고 해서 할 말이 없는데 말하거나, 부탁받지 않은 일을 할 이유가 되지는 않습니다.";

/// How far a dial can go.
///
/// **Ten, and the middle is not a default.** Zero is *do not aim for this at all*, which is what
/// every dial is until somebody moves it — see the rule above.
pub const MOST: u8 = 10;

/// What argmind is trying to be. **Every dial at zero is the state this program shipped in.**
#[derive(Debug, Clone, PartialEq)]
pub struct Mind {
    /// One number per entry of [`DIALS`], in the same order.
    pub dials: Vec<u8>,
    /// A sentence the person wrote, which is the goal the numbers cannot express.
    ///
    /// **The dials say how, this says what for.** *재치 7* cannot mean *내가 코드 리뷰하는 동안
    /// 놓친 것만 말해줘*, and a design with only numbers would push every real instruction into
    /// the conversation, where it is twelve turns from being forgotten.
    pub aim: String,
    /// **What the person asked argmind to remember**, one sentence each, in their own words —
    /// *나를 대표님이라고 불러*, *메모장 말고 VS Code*. From the dots comparison: corrections and
    /// preferences that outlive the twelve turns of history, shown one by one and deletable one by
    /// one. **Never written by the model**: added by the person in the panel, by saying
    /// *…기억해줘*, or by accepting an offer the program makes when it reads a correction.
    pub kept: Vec<String>,
}

/// How many things are kept. Chosen: each is a line in every prompt, and the dials' own rule is
/// that a long list is how a prompt stops being read.
pub const MOST_KEPT: usize = 12;

/// How long one may be, in characters. A sentence, not a document.
pub const KEPT_CHARS: usize = 200;

/// **The default is rest**, written out rather than derived: a derived one leaves `dials` empty
/// while `resting()` fills it with zeros, and two values that behave identically but compare
/// unequal is the sort of difference that surfaces a year later in a test nobody can explain.
impl Default for Mind {
    fn default() -> Self {
        Mind::resting()
    }
}

impl Mind {
    /// Nothing set. The prompt is then exactly what it was before any of this existed.
    pub fn resting() -> Self {
        Mind { dials: vec![0; DIALS.len()], aim: String::new(), kept: Vec::new() }
    }

    /// Is every dial at zero and no goal written?
    pub fn at_rest(&self) -> bool {
        self.aim.trim().is_empty() && self.dials.iter().all(|&n| n == 0)
            && self.kept.iter().all(|k| k.trim().is_empty())
    }

    /// Read one dial by its Korean name, clamped, missing meaning zero.
    pub fn of(&self, korean: &str) -> u8 {
        DIALS.iter().position(|(k, _)| *k == korean)
            .and_then(|i| self.dials.get(i).copied())
            .unwrap_or(0)
            .min(MOST)
    }

    /// Turn one dial. **An unknown name is ignored rather than added**, for the same reason the
    /// decisions file drops a key nobody can name.
    pub fn turn(&mut self, korean: &str, to: u8) {
        self.dials.resize(DIALS.len(), 0);
        if let Some(i) = DIALS.iter().position(|(k, _)| *k == korean) {
            self.dials[i] = to.min(MOST);
        }
    }

    /// The line that goes into the prompt, or **`None` when there is nothing to say.**
    ///
    /// Only dials above zero appear, with the number, because *playful 2* and *playful 9* are
    /// different instructions and a bare list of adjectives is neither.
    pub fn line(&self) -> Option<String> {
        if self.at_rest() {
            return None;
        }
        let mut parts: Vec<String> = Vec::new();
        for (i, (korean, gloss)) in DIALS.iter().enumerate() {
            let n = self.dials.get(i).copied().unwrap_or(0).min(MOST);
            if n > 0 {
                parts.push(format!("{korean}({gloss}) {n}/{MOST}"));
            }
        }
        let mut out = String::new();
        if !parts.is_empty() {
            //: **Aimed at rather than achieved.** A model told it *is* witty writes jokes; one told
            //: to aim for it keeps answering the question. The difference is measurable and this
            //: is the side worth erring on.
            out.push_str(&format!("{AIM_AT}{}.", parts.join(", ")));
        }
        //: **The goal first and the kept sentences after it, under the one label they share** —
        //: the goal was always rendered as *기억해 달라고 한 것*, and a goal alone renders byte for
        //: byte as it did, so what the dials were measured against has not moved.
        let aim = self.aim.trim();
        let said: Vec<String> = std::iter::once(aim)
            .chain(self.kept.iter().map(|k| k.trim()))
            .filter(|s| !s.is_empty())
            //: The person's own words, unedited and quoted, so a rewrite of the surrounding
            //: sentence cannot quietly change what they asked for.
            .map(|s| format!("\"{s}\""))
            .collect();
        if !said.is_empty() {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&format!("{KEEP_IN_MIND}{}", said.join(" / ")));
        }
        out.push_str(BUT);
        Some(out)
    }
}

/// What a turn asking to be remembered means.
#[derive(Debug, PartialEq)]
pub enum Keep {
    /// A sentence to keep, with the request words taken off.
    This(String),
    /// *이거 기억해줘* — a request with nothing in it that could stand alone in a prompt a week
    /// later. Asked back rather than stored, because *"이거"* in every prompt is noise.
    Vague,
}

/// The words that make a turn a request to remember. **The program reads them, not the model**:
/// a model asked to decide what to remember is a model writing its own instructions, and this
/// project's note invented having done things the first time it was asked about itself.
const KEEP_WORDS: [&str; 8] = ["기억해줘", "기억해 줘", "기억해둬", "기억해 둬", "기억해 두세요",
                               "기억해주세요", "기억해 주세요", "기억해"];

/// **Is this turn asking argmind to remember something?** Only when the request words end the
/// sentence or open it — *기억해?* is a question, and *내가 뭐라고 했는지 기억해?* is not a request.
pub fn asked_to_keep(text: &str) -> Option<Keep> {
    let t = text.trim();
    if t.ends_with('?') || t.ends_with('？') {
        return None;
    }
    let t = t.trim_end_matches(['.', '!', '~', ' ']);
    let rest = KEEP_WORDS.iter().find_map(|w| t.strip_suffix(w))
        .or_else(|| t.strip_prefix("기억해:").or_else(|| t.strip_prefix("기억해 :")))?;
    let rest = rest.trim().trim_matches([',', ':', '.', ' ']).trim();
    //: *…라고*, *…는 거* and *…것* are how Korean quotes a sentence before 기억해줘; the quote
    //: marker goes, the sentence stays.
    //: *…다고* keeps its 다 (*쓴다고* → *쓴다*); *…이라고* and *…라고* after a noun go whole
    //: (*화요일이라고* → *화요일*).
    //: **A quoted request keeps its 라** (*불러달라고* → *불러달라*, *하지 말라고* → *하지 말라*):
    //: stripping 라고 there left *불러달*, found on the headless server the day this shipped.
    let rest = if let Some(r) = rest.strip_suffix("다고") {
        format!("{r}다")
    } else if ["달라고", "하라고", "말라고", "주라고", "으라고", "오라고", "가라고", "보라고", "쓰라고"]
        .iter().any(|m| rest.ends_with(m)) {
        rest.strip_suffix('고').unwrap_or(rest).to_string()
    } else {
        ["이라고", "라고", "라는 거", "다는 거", "다는 것", "는 거", "는 것"].iter()
            .find_map(|m| rest.strip_suffix(m)).unwrap_or(rest).to_string()
    };
    let rest = rest.trim();
    let deictic = ["이거", "그거", "저거", "이것", "그것", "저것", "이걸", "그걸", "이것을", "그것을",
                   "이 말", "그 말", "방금", "방금 거", ""];
    if deictic.contains(&rest) || rest.chars().count() < 4 {
        return Some(Keep::Vague);
    }
    Some(Keep::This(rest.chars().take(KEPT_CHARS).collect()))
}

/// The words that make a turn look like a correction or a standing preference. **An offer, never
/// a store**: the program shows *기억해 둘까요?* and the person decides, so a word list that fires
/// on the wrong sentence costs one dismissed row and nothing in the prompt.
const PREFERENCE_WORDS: [&str; 10] = ["말고", "그게 아니라", "하지 마", "하지마", "싫어", "좋아해",
                                      "앞으로", "항상", "절대", "부르지 마"];

pub fn worth_offering(text: &str) -> bool {
    let t = text.trim();
    t.chars().count() >= 6 && !t.ends_with('?') && asked_to_keep(t).is_none()
        && PREFERENCE_WORDS.iter().any(|w| t.contains(w))
}
