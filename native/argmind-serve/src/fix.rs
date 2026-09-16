//! What argmind would do about something it is missing, in the words it will ask with.
//!
//! A [`Fix`] is a description of an action, [`describe`] is the sentence a person is shown before
//! it happens, and [`apply`] is the doing — **and `apply` is never reached except through an
//! answer.** The split is the point: the text can be read, reviewed and tested without a machine
//! being changed, and a fix whose description drifts from what it does is a fix whose approval
//! means nothing.
//!
//! `apply` lives here rather than in either caller because there are two — the window in the page
//! and the command line — and a second copy of "what yes means" is how the two come to mean
//! different things.
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
    /// Open a page in the browser. **This installs nothing.**
    OpenPage { url: String, why: String },
    /// Download an installer and run it.
    ///
    /// **The heaviest thing in this program**, and the one an approval has to be worth. It is
    /// fetched over TLS with the certificate verified, and the whole sequence afterwards — running
    /// it, waiting for it, checking it took — happens without asking again, because asking four
    /// times for one decision is how people learn to click through.
    Install { url: String, file: String, why: String, about_mb: u64 },
    /// Fetch a file that is data rather than code, and put it where something expects it.
    ///
    /// Separate from `Install` because **nothing is executed**: a model is read by a program that
    /// was already there.
    Fetch { url: String, into: String, why: String, about_mb: u64, sha256: Option<String> },
    /// Clone one pinned commit, compile it here, and run what comes out.
    ///
    /// **Not a generic build step, and it does not pretend to be one.** It is the recogniser, whose
    /// project publishes source rather than an installer; the constants live in
    /// [`crate::recogniser`] and this variant carries them so the sentence can name them.
    ///
    /// Compiling it yourself changes who produced the bytes, not who wrote the instructions — so
    /// this counts as running a stranger, and the window says so.
    ///
    /// `built` is whether a server compiled from this same commit is already sitting there, and it
    /// exists so the window can ask the **shorter, truer question** in the ordinary case: after the
    /// first time, saying yes downloads nothing and compiles nothing.
    BuildRecogniser { repo: String, commit: String, why: String, minutes: u64, built: bool },
    /// The same thing for the voice, and **a separate variant rather than a parameter.**
    ///
    /// They look alike and they are not the same decision. This one needs a gigabyte of weights
    /// instead of a hundred and forty megabytes, it needs a graphics driver and a development kit
    /// that the recogniser does not, and what it produces speaks out loud. A person who agreed to
    /// the recogniser has not agreed to this, and a shared variant with a flag in it is how those
    /// two approvals would come to be one.
    ///
    /// `built` means a server compiled from this same commit is already sitting there, so the
    /// window can ask the **shorter, truer question** in the ordinary case.
    BuildSpeaker { repo: String, commit: String, why: String, minutes: u64, built: bool },
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
            Fix::Install { .. } => "argmind — 내려받아 설치할까요?",
            Fix::Fetch { .. } => "argmind — 파일을 받을까요?",
            //: Two questions, so two titles. A window headed "빌드할까요?" over a sentence that
            //: says nothing will be compiled is a window that was not read.
            Fix::BuildRecogniser { built: true, .. } => "argmind — 받아쓰기 서버를 띄울까요?",
            Fix::BuildRecogniser { .. } => "argmind — 받아쓰기 서버를 빌드할까요?",
            Fix::BuildSpeaker { built: true, .. } => "argmind — 목소리 서버를 띄울까요?",
            Fix::BuildSpeaker { .. } => "argmind — 목소리 서버를 빌드할까요?",
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

    /// Does saying yes cause a program from somewhere else to run on this machine?
    ///
    /// **The sharpest line in here.** Fetching data is not the same as executing somebody's build,
    /// and an approval for one is not an approval for the other — so they are different variants
    /// and the text says which.
    pub fn runs_a_stranger(&self) -> bool {
        //: **Building it yourself does not make it yours.** The compiler turned somebody else's
        //: instructions into bytes on this machine and then those bytes run — which is the thing
        //: being agreed to, whoever pressed the compile button.
        matches!(self, Fix::Install { .. } | Fix::BuildRecogniser { .. } | Fix::BuildSpeaker { .. })
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
        Fix::Install { url, why, about_mb, .. } => format!(
            "{why}\n\n\
             · {url} 에서 약 {about_mb} MB를 내려받습니다\n\
             · 인증서를 확인하고, https가 아니면 받지 않습니다\n\
             · 받은 설치 프로그램을 실행합니다 — 만든 곳을 신뢰한다는 뜻입니다\n\
             · 설치 프로그램이 스스로 물어볼 수 있습니다\n\n\
             내려받아 설치할까요?"),
        Fix::Fetch { url, into, why, about_mb, sha256 } => format!(
            "{why}\n\n\
             · {url} 에서 약 {about_mb} MB를 내려받습니다\n\
             · {into} 에 저장합니다\n\
             · {}\n\
             · 실행하지 않습니다 — 데이터 파일입니다\n\n\
             받을까요?",
            match sha256 {
                Some(_) => "받은 뒤 해시를 확인하고, 다르면 지웁니다",
                //: **Said out loud rather than left out.** "Nobody published a hash for this" is a
                //: fact about the source, and the person deciding should have it.
                None => "게시된 해시가 없어 내용은 확인하지 못합니다",
            }),
        //: **After the first time, the honest question is a much smaller one.** Describing a
        //: download and a ten-minute build to somebody whose machine will spend five seconds
        //: starting a binary it already has is the description drifting from the doing — and it is
        //: the case that happens every time but the first.
        Fix::BuildRecogniser { commit, why, built: true, .. } => format!(
            "{why}\n\n\
             · 이미 빌드되어 있습니다 (커밋 {}) — 내려받지도, 컴파일하지도 않습니다\n\
             · 그때 빌드한 것을 실행합니다 — 남이 쓴 코드를 이 컴퓨터에서 돌린다는 뜻입니다\n\
             · 서버는 {} 에서만 듣고, argmind를 닫으면 함께 꺼집니다\n\n\
             띄울까요?",
            commit.get(..12).unwrap_or(commit), argmind_env_whisper::WHISPER),
        Fix::BuildRecogniser { repo, commit, why, minutes, .. } => format!(
            "{why}\n\n\
             · {repo} 에서 커밋 {} 하나만 받습니다\n\
             · 태그가 아니라 커밋 이름입니다 — 이름이 곧 내용이라 다른 게 오면 git이 거부합니다\n\
             · 이 컴퓨터의 컴파일러로 직접 빌드합니다 (길면 {minutes}분)\n\
             · 빌드한 것을 실행합니다 — 남이 쓴 코드를 이 컴퓨터에서 돌린다는 뜻입니다\n\
             · 받아쓰기 모델이 없으면 함께 받습니다 (약 {} MB, 해시를 확인합니다)\n\
             · 서버는 {} 에서만 듣고, argmind를 닫으면 함께 꺼집니다\n\n\
             빌드해서 띄울까요?",
            commit.get(..12).unwrap_or(commit),
            crate::recogniser::MODEL_MB,
            argmind_env_whisper::WHISPER),
        //: The same two questions as the recogniser, for the same reason: after the first time,
        //: saying yes downloads nothing and compiles nothing, and describing a gigabyte and a
        //: quarter of an hour to somebody whose machine will spend fifteen seconds loading weights
        //: it already has is the description drifting from the doing.
        Fix::BuildSpeaker { commit, why, built: true, .. } => format!(
            "{why}\n\n\
             · 이미 빌드되어 있습니다 (커밋 {}) — 내려받지도, 컴파일하지도 않습니다\n\
             · 그때 빌드한 것을 실행합니다 — 남이 쓴 코드를 이 컴퓨터에서 돌린다는 뜻입니다\n\
             · 그래픽카드에 모델을 올리는 데 몇십 초 걸립니다\n\
             · 서버는 {} 에서만 듣고, argmind를 닫으면 함께 꺼집니다\n\n\
             띄울까요?",
            commit.get(..12).unwrap_or(commit), crate::speaker::QWENTTS_HOST),
        Fix::BuildSpeaker { repo, commit, why, minutes, .. } => format!(
            "{why}\n\n\
             · {repo} 에서 커밋 {} 하나만 받습니다\n\
             · 태그가 아니라 커밋 이름입니다 — 이름이 곧 내용이라 다른 게 오면 git이 거부합니다\n\
             · 이 컴퓨터의 컴파일러로 직접 빌드합니다 (길면 {minutes}분)\n\
             · 빌드한 것을 실행합니다 — 남이 쓴 코드를 이 컴퓨터에서 돌린다는 뜻입니다\n\
             · 목소리 모델 두 개를 받습니다 (합쳐서 약 {} MB, 해시를 확인합니다)\n\
             · 그래픽카드를 씁니다 — CPU만으로는 32배 느려서 대화가 안 됩니다\n\
             · 이걸 띄우면 argmind가 이 목소리로 소리를 냅니다\n\
             · 서버는 {} 에서만 듣고, argmind를 닫으면 함께 꺼집니다\n\n\
             빌드해서 띄울까요?",
            commit.get(..12).unwrap_or(commit),
            crate::speaker::WEIGHTS_MB,
            crate::speaker::QWENTTS_HOST),
        Fix::RunElevated { command, why } => format!(
            "{why}\n\n\
             · 다음을 관리자 권한으로 실행합니다:\n    {command}\n\
             · Windows가 한 번 더 물어봅니다\n\
             · 몇 분 걸리고, 인터넷에서 내려받습니다\n\n\
             실행할까요?"),
    }
}


/// Do one approved fix, reporting as it goes.
///
/// **Reached only after a yes.** Every branch says what it did, in a sentence the person who
/// approved it can check against the one they were shown.
pub fn apply(fix: &Fix, say: &mut dyn FnMut(&str)) -> Result<String, String> {
    match fix {
        Fix::PullModel { model, .. } => {
            let mut last = 0;
            argmind_env_ollama::manage::pull(argmind_env_ollama::OLLAMA, model, |p| {
                //: Every ten per cent, not every line: a pull emits thousands, and a page that
                //: scrolls past is no more informative than one that says nothing.
                match p.percent() {
                    Some(pct) if pct >= last + 10 => {
                        last = pct - pct % 10;
                        say(&format!("{pct}%  {}", p.status));
                    }
                    None if p.status != "pulling manifest" => say(&p.status),
                    _ => {}
                }
            })?;
            Ok("받았습니다".into())
        }
        Fix::OpenPage { url, .. } => {
            //: `explorer` hands a URL to whatever the person chose as their browser. It is a shell
            //: command, not a downloader — **nothing is fetched or run here.**
            std::process::Command::new("explorer").arg(url).spawn().map_err(|e| format!("{e}"))?;
            Ok(format!("{url} 을(를) 열었습니다 — 설치는 직접 하셔야 합니다"))
        }
        Fix::Install { url, file, .. } => {
            let at = crate::fetch::shed().join(file);
            crate::fetch::get(url, &at, None, say)?;
            //: **The whole sequence, without asking again.** The approval covered downloading *and*
            //: running; stopping here to ask a second time for the half they already agreed to is
            //: how people learn to click through dialogs without reading them.
            crate::fetch::run(&at, &[], say)?;
            //: Tidied up. An installer left in a temp folder is one somebody runs by accident in
            //: six months, when it is out of date.
            let _ = std::fs::remove_file(&at);
            Ok("설치했습니다".into())
        }
        Fix::Fetch { url, into, sha256, .. } => {
            let at = std::path::PathBuf::from(into);
            if let Some(parent) = at.parent() {
                std::fs::create_dir_all(parent).map_err(|e| format!("{e}"))?;
            }
            crate::fetch::get(url, &at, sha256.as_deref(), say)?;
            Ok(format!("{into} 에 받았습니다"))
        }
        Fix::BuildRecogniser { repo, commit, .. } => {
            //: **Checked again here**, not only where the offer was made. Between the two, minutes
            //: of downloading and an approval — and a build that discovers halfway through that
            //: there is no compiler has already spent somebody's evening.
            let missing = crate::recogniser::missing_tools();
            if !missing.is_empty() {
                return Err(format!("{} 이(가) 없어 빌드할 수 없습니다", missing.join(", ")));
            }
            //: **The whole sequence on one yes.** The model half is literally the fix that would
            //: have been offered on its own, run through this same function — so what the person
            //: agreed to and what happens cannot come apart, and there is one downloader rather
            //: than two.
            if !crate::recogniser::model_at().is_file() {
                apply(&crate::recogniser::model_fix(), say)?;
            }
            let at = crate::recogniser::shed();
            crate::recogniser::clone_at(repo, commit, &at, say)?;
            crate::recogniser::build(&at, crate::recogniser::TARGET, say)?;
            //: Without this it runs here and nowhere else — see `bring_runtime`.
            crate::recogniser::bring_runtime(&crate::recogniser::server_exe(), say)?;
            //: Written only once everything above it worked, so a half-finished tree is never
            //: described as "already built".
            crate::recogniser::remember_build(commit);
            crate::recogniser::serve(say)?;
            Ok("받아쓰기 서버가 떴습니다".into())
        }
        Fix::BuildSpeaker { repo, commit, .. } => {
            //: **Checked again here**, not only where the offer was made — between the two there
            //: are minutes of downloading and an approval.
            let missing = crate::speaker::missing_tools();
            if !missing.is_empty() {
                return Err(format!("{} 이(가) 없어 빌드할 수 없습니다", missing.join(", ")));
            }
            //: **And the SDK, separately.** It is the one thing here that a person may have
            //: installed in the minute since the window opened, and the one thing this process
            //: cannot learn about by looking at its own environment.
            if crate::speaker::sdk().is_none() {
                return Err("Vulkan SDK가 없어 빌드할 수 없습니다".into());
            }
            //: **The whole sequence on one yes**, and the weights are literally the fixes that
            //: would have been offered on their own, run through this same function — so there is
            //: one downloader rather than three and no way for them to mean different things.
            for weights in crate::speaker::weight_fixes() {
                let Fix::Fetch { into, .. } = &weights else { continue };
                if !std::path::Path::new(into).is_file() {
                    apply(&weights, say)?;
                }
            }
            let at = crate::speaker::shed();
            crate::speaker::clone_at(repo, commit, &at, say)?;
            crate::speaker::build(&at, crate::speaker::TARGET, say)?;
            //: Without this it runs here and nowhere else — see `bring_runtime`.
            crate::building::bring_runtime(&crate::speaker::server_exe(), say)?;
            //: Written only once everything above it worked, so a half-finished tree is never
            //: described as "already built".
            crate::speaker::remember_build(commit);
            crate::speaker::serve(say)?;
            Ok("목소리 서버가 떴습니다".into())
        }
        Fix::RunElevated { command, .. } => {
            //: **The operating system asks again.** Elevation raises a prompt of its own, so a yes
            //: here is the first of two and the second one is not argmind's to give.
            let done = crate::platform::run_elevated(command)?;
            if done {
                Ok("실행했습니다".into())
            } else {
                //: A cancelled elevation prompt lands here, and it is a refusal rather than a
                //: fault — calling somebody changing their mind a failure is a lie.
                Err("관리자 권한을 받지 못했거나 취소되었습니다".into())
            }
        }
    }
}
