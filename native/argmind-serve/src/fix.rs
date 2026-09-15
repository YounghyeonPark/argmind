//! What argmind would do about something it is missing, in the words it will ask with.
//!
//! **Nothing here runs anything.** A [`Fix`] is a description of an action and [`describe`] is the
//! sentence a person is shown before it happens; the doing lives in `setup`, behind an answer. That
//! split is the point: the text can be read, reviewed and tested without a machine being changed,
//! and a fix whose description drifts from what it does is a fix whose approval means nothing.
//!
//! **Three grades, and the grade is the risk.** Fetching data into a program that is already
//! running is not the same as running an installer, and neither is the same as asking Windows for
//! administrator rights. They are separate variants so that a fix cannot be quietly promoted from
//! one to the next by an edit to a string.

/// Something argmind can do about a missing need, once somebody says yes.
#[derive(Debug, Clone, PartialEq)]
pub enum Fix {
    /// Ask Ollama to fetch a model into its own store. **No installer, no executable** — a program
    /// already running by the person's choice, putting a file where it keeps its files.
    PullModel { model: String, about_gb: f64 },
    /// Open a page in the browser. **This installs nothing**; it is the honest end of the road for
    /// software argmind will not fetch and run on somebody's behalf.
    OpenPage { url: String, why: String },
    /// A documented Windows command that needs administrator rights, run through the elevation
    /// prompt so Windows asks as well.
    RunElevated { command: String, why: String },
}

impl Fix {
    /// The title of the window that asks.
    pub fn title(&self) -> &'static str {
        match self {
            Fix::PullModel { .. } => "argmind — 모델을 받을까요?",
            Fix::OpenPage { .. } => "argmind — 설치 페이지를 열까요?",
            Fix::RunElevated { .. } => "argmind — 관리자 권한이 필요합니다",
        }
    }

    /// Whether saying yes changes anything on this machine by itself.
    ///
    /// Opening a page does not: it hands the decision back to the person in a browser, where they
    /// still have to download and run something themselves.
    pub fn changes_the_machine(&self) -> bool {
        !matches!(self, Fix::OpenPage { .. })
    }
}

/// **Exactly what will happen, in the sentence the person reads before it does.**
///
/// Every one of these names the thing, the size where there is one, and the words *this machine* or
/// *administrator* where they apply. A dialog that said "install dependencies?" would be answered
/// yes by everybody and would have told them nothing.
pub fn describe(fix: &Fix) -> String {
    match fix {
        Fix::PullModel { model, about_gb } => format!(
            "모델 {model} 을(를) 받습니다.\n\n\
             · 약 {about_gb:.0} GB를 내려받습니다\n\
             · 이미 실행 중인 Ollama가 자기 저장소에 넣습니다\n\
             · 설치 프로그램은 실행되지 않습니다\n\n\
             받을까요?"),
        Fix::OpenPage { url, why } => format!(
            "{why}\n\n\
             · 브라우저로 {url} 을(를) 엽니다\n\
             · argmind가 직접 설치하지는 않습니다 — 내려받고 실행하는 건 직접 하셔야 합니다\n\n\
             페이지를 열까요?"),
        Fix::RunElevated { command, why } => format!(
            "{why}\n\n\
             · 다음을 관리자 권한으로 실행합니다:\n    {command}\n\
             · Windows가 한 번 더 물어봅니다\n\
             · 몇 분 걸리고, 인터넷에서 내려받습니다\n\n\
             실행할까요?"),
    }
}
