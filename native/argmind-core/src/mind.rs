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

/// **The languages a person can choose to be answered in — since 2026-11-17**, the owner's
/// *사용자가 대화할 언어를 설정할수 있으면 함*: a code, the name the page shows, and the line the
/// model is told.
///
/// **Each line is written in its own language**, because the measurement above is the only one this
/// project has about a reply's language and it says the slot decides it: a content-free Korean
/// sentence moved Korean replies from 28 to 42 of 44, an English one down to 19. So a line asking
/// for English must not be a Korean sentence about English — the kept *항상 영어로 대답해* got
/// English **0 of 3** (`remembering`). The codes are the ones the recogniser takes.
///
/// **None of them is the default.** Unset, nothing is added and the prompt says what it always
/// said — *in the language they used* — so the six probes still hash to `r322a4c`.
///
/// **And each carries the frame around the dials and the kept sentences in its own language**, and
/// the placeholder of `say` in the reply's shape — both found by `--bin language`'s first run, where
/// the line alone held 9 of 12 Korean turns for English and **1 of 12 once a kept sentence's Korean
/// frame sat beside it** (Japanese beside a dial: 0 of 12). Korean's frame is the one that shipped,
/// character for character, so a Korean setting changes nothing but the line.
///
/// **Japanese was measured and left out on 2026-11-17, and is back the same day — by the program,
/// not by its line.** With the frame below it answered Korean turns in Japanese **8 and 7 of 12**
/// against a bar of 11; on a screen of Korean buttons it wrote *メモ장이 열렸습니다* and
/// *計算기를 열어드릴게요*, Korean grammar around a half-converted name, and as often a reply wholly
/// in Korean. The line is unchanged, character for character, so that failure is the control; what
/// is new is [`crate::converse::in_language`] reading every reply when a language is set, and a
/// reply in another script asked to be written again in [`Language::rewrite`] — one more call, only
/// on that reply. Japanese then follows 11–12 of 12 on every set, twice, with 9 of 24 Korean turns
/// and 0 of 12 Japanese turns paying for a second call — and of about twenty rewrites two changed
/// the meaning both runs (*メモ장이 떠요* → *…飛んでいます*). `--bin language` has the rest.
pub struct Language {
    pub code: &'static str,
    /// What the page shows.
    pub name: &'static str,
    /// What the model is told.
    pub line: &'static str,
    /// The placeholder for `say` in the reply's shape.
    pub say: &'static str,
    pub aim_at: &'static str,
    pub keep: &'static str,
    pub but: &'static str,
    /// The dials as this language says them, in [`DIALS`]' order.
    pub dials: [(&'static str, &'static str); 4],
    /// **What the model is asked when its reply came back in another language**, followed by that
    /// reply. Told it is *its own* words, so it rewrites rather than answers them.
    pub rewrite: &'static str,
}

pub const LANGUAGES: [Language; 4] = [
    Language { code: "ko", name: "한국어", line: "상대가 어떤 언어로 말하든 항상 한국어로 대답하세요.",
               say: "<한국어로 돌려줄 말>", aim_at: AIM_AT, keep: KEEP_IN_MIND, but: BUT, dials: DIALS,
               rewrite: "다음은 당신이 상대에게 하려던 말입니다. 뜻은 그대로 두고 덧붙이지 말고 자연스러운 한국어로 다시 쓰세요. 다시 쓴 말만 답하세요." },
    Language { code: "en", name: "English",
               line: "Always reply in English, whatever language the person writes in.",
               say: "<what you say back, in English>",
               aim_at: "Aim for this manner: ", keep: "What the person asked you to remember: ",
               but: " None of this is a reason to speak when there is nothing to say, or to do what nobody asked for.",
               dials: [("wit", "quick and clever"), ("humour", "light and fun"),
                       ("service", "offering to help first"), ("devotion", "seeing a task through to the end")],
               rewrite: "This is what you were about to say to the person. Write it again in natural English, keeping the meaning and adding nothing. Reply with the rewritten words only." },
    Language { code: "zh", name: "中文", line: "无论对方使用什么语言，请始终用中文回答。",
               say: "<用中文回复的话>",
               aim_at: "请以这样的态度为目标: ", keep: "对方请你记住的事: ",
               but: " 但这并不是在无话可说时开口、或去做没人要求的事的理由。",
               dials: [("机智", "敏捷而聪明"), ("幽默", "轻松有趣"),
                       ("服务", "主动提出帮忙"), ("投入", "把手头的事坚持做完")],
               rewrite: "下面是你正要对对方说的话。请保持意思不变、不要添加内容，用自然的中文重新写一遍。只回答重写后的话。" },
    Language { code: "ja", name: "日本語", line: "相手が何語で話しても、必ず日本語で答えてください。",
               say: "<日本語で返す言葉>",
               aim_at: "次のような話し方を目指してください: ", keep: "相手が覚えておいてほしいと言ったこと: ",
               but: " ただし、言うことがないのに話したり、頼まれていないことをしたりする理由にはなりません。",
               dials: [("機転", "素早く賢く"), ("ユーモア", "軽く楽しく"),
                       ("奉仕", "自分から手伝いを申し出る"), ("献身", "やりかけたことを最後までやり抜く")],
               rewrite: "次は、あなたが相手に言おうとした言葉です。意味を変えず、何も付け足さずに、自然な日本語で書き直してください。書き直した言葉だけを答えてください。" },
];

/// The language set, if it is one of [`LANGUAGES`] — **an unknown code is rest**, as an unknown dial
/// is: the worst a bad file can do is turn the feature off.
pub fn language(code: &str) -> Option<&'static Language> {
    LANGUAGES.iter().find(|l| l.code == code.trim())
}

/// Is this one of [`LANGUAGES`]?
pub fn is_a_language(code: &str) -> bool {
    language(code).is_some()
}

/// The line for a language, or `None` for an unset or unknown one.
pub fn language_line(code: &str) -> Option<&'static str> {
    language(code).map(|l| l.line)
}

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
    /// **The goal argmind works toward by itself when nobody needs it** — `crate::pursuit`.
    ///
    /// **Never shown to a turn, since 2026-11-09.** It was rendered into every prompt as *사용자가
    /// 기억해 달라고 한 것*, and the owner's *돈을어떻게벌수있는지생각해줘* came back in answers to
    /// things that had nothing to do with it; the owner then said what the box was for: *사용자
    /// 작업이 없을떄 스스로 목표를 설정하고 이루도록 수행하는것.* An instruction about how to answer
    /// is a kept sentence, below.
    pub aim: String,
    /// **What the person asked argmind to remember**, one sentence each, in their own words —
    /// *나를 대표님이라고 불러*, *메모장 말고 VS Code*. From the dots comparison: corrections and
    /// preferences that outlive the twelve turns of history, shown one by one and deletable one by
    /// one. **Never written by the model**: added by the person in the panel, by saying
    /// *…기억해줘*, or by accepting an offer the program makes when it reads a correction.
    pub kept: Vec<String>,
    /// **The language argmind answers in**, one of [`LANGUAGES`] by code — or empty, which is *the
    /// language they used*, what argmind has always done.
    pub language: String,
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
        Mind { dials: vec![0; DIALS.len()], aim: String::new(), kept: Vec::new(), language: String::new() }
    }

    /// Is every dial at zero and no goal written?
    pub fn at_rest(&self) -> bool {
        self.aim.trim().is_empty() && self.dials.iter().all(|&n| n == 0)
            && self.kept.iter().all(|k| k.trim().is_empty())
            && language_line(&self.language).is_none()
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

    /// **The placeholder for `say` in the reply's shape**, in the language set — `None` at rest, so
    /// the shape is what it always was.
    pub fn say_slot(&self) -> Option<&'static str> {
        language(&self.language).map(|l| l.say)
    }

    /// The line that goes into the prompt, or **`None` when there is nothing to say.**
    ///
    /// Only dials above zero appear, with the number, because *playful 2* and *playful 9* are
    /// different instructions and a bare list of adjectives is neither. **The goal does not** — see
    /// [`Mind::aim`] — so a goal alone renders nothing.
    pub fn line(&self) -> Option<String> {
        let tongue = language_line(&self.language);
        if self.dials.iter().all(|&n| n == 0) && self.kept.iter().all(|k| k.trim().is_empty()) {
            //: **A language alone is its own line**, with nothing about dials after it: the
            //: sentence that follows them is about not saying more than is wanted, and nothing
            //: here gives a reason to.
            return tongue.map(str::to_string);
        }
        //: The frame in the language set, or the Korean one that shipped.
        let (aim_at, keep, but, dials) = match language(&self.language) {
            Some(l) => (l.aim_at, l.keep, l.but, l.dials),
            None => (AIM_AT, KEEP_IN_MIND, BUT, DIALS),
        };
        let mut parts: Vec<String> = Vec::new();
        for (i, (korean, gloss)) in dials.iter().enumerate() {
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
            out.push_str(&format!("{aim_at}{}.", parts.join(", ")));
        }
        //: **The kept sentences, under the label the goal used to share with them** — a kept
        //: sentence alone renders byte for byte as a goal alone did, so what `remembering`
        //: measured has not moved.
        let said: Vec<String> = self.kept.iter().map(|k| k.trim())
            .filter(|s| !s.is_empty())
            //: The person's own words, unedited and quoted, so a rewrite of the surrounding
            //: sentence cannot quietly change what they asked for.
            .map(|s| format!("\"{s}\""))
            .collect();
        if !said.is_empty() {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(&format!("{keep}{}", said.join(" / ")));
        }
        out.push_str(but);
        //: **Last**, after the Korean of the lines above it, so the sentence nearest the turn is in
        //: the language asked for — `language` measures whether that is enough with them present.
        if let Some(t) = tongue {
            out.push(' ');
            out.push_str(t);
        }
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
