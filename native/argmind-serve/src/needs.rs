//! What argmind needs, whether it is here, and **which of it is safe to fetch.**
//!
//! Everything this program wants falls into one of three bands, and the difference between them is
//! the whole design:
//!
//! * **Already here or genuinely absent, and free to check.** A socket that answers, a voice that
//!   is installed, a microphone that exists. Looking costs nothing and changes nothing.
//! * **Something argmind can do about it, once asked.** Every one of these carries a [`Fix`], and
//!   a fix only happens after a window has said in full what it will do and somebody has answered
//!   yes. The rule that used to read *we will not install software while you are not looking* is
//!   now *you will be looking* — the same rule with the person put back into it.
//! * **Nothing argmind can do.** Reported, and that is all.
//!
//! The fix is a property of the need, written next to it, rather than a judgement made at the call
//! site — so adding a need means deciding what could be done about it, and the grades in [`Fix`]
//! mean it cannot be quietly promoted from fetching data to running an installer.

use crate::fix::Fix;

/// One thing argmind wants.
#[derive(Debug, Clone)]
pub struct Need {
    pub what: &'static str,
    /// Whether the agent works at all without it.
    pub required: bool,
    pub present: bool,
    /// What is true right now, in a sentence.
    pub found: String,
    /// What to do about it, when it is missing.
    pub how: String,
    /// **What argmind offers to do, and nothing happens until somebody answers yes.** `None` means
    /// there is nothing it could do even with permission.
    pub fix: Option<Fix>,
}

impl Need {
    pub fn mark(&self) -> &'static str {
        if self.present {
            "있음"
        } else if self.required {
            "없음 — 필요"
        } else {
            "없음 — 선택"
        }
    }
}

/// Everything, in the order a person would fix it.
pub fn every(model: &str) -> Vec<Need> {
    let mut out = Vec::new();

    let ollama_up = argmind_env_ollama::manage::awake(argmind_env_ollama::OLLAMA);
    out.push(Need {
        what: "Ollama",
        required: true,
        present: ollama_up,
        found: if ollama_up { "127.0.0.1:11434 응답함".into() } else { "응답 없음".into() },
        how: "https://ollama.com 에서 설치한 뒤 실행해 두세요".into(),
        //: **A page, not an installer.** argmind will not download and run a third party's
        //: executable on somebody's behalf, approval or no approval — the approval would be for a
        //: file nobody had seen, fetched from a name nobody had checked. Opening the page hands the
        //: decision back to where it can actually be made.
        fix: Some(Fix::OpenPage {
            url: "https://ollama.com/download".into(),
            why: "Ollama가 있어야 argmind가 동작합니다.".into(),
        }),
    });

    //: **Only asked when Ollama is up.** "The model is missing" and "nothing was there to ask" are
    //: different facts, and reporting the first when the second is true sends somebody looking for
    //: a five gigabyte download they already have.
    let has_model = ollama_up && argmind_env_ollama::manage::has(argmind_env_ollama::OLLAMA, model);
    out.push(Need {
        what: "모델",
        required: true,
        present: has_model,
        found: if !ollama_up {
            "Ollama가 없어서 확인 못 함".into()
        } else if has_model {
            format!("{model} 있음")
        } else {
            format!("{model} 없음")
        },
        how: format!("argmind setup 이 받아옵니다 (약 5 GB) — 또는 ollama pull {model}"),
        //: Only offered when Ollama is up, because there is nothing to pull with otherwise.
        fix: ollama_up.then(|| Fix::PullModel { model: model.to_string(), about_gb: 5.0 }),
    });

    //: A TCP connect and nothing else — **it is not asked to transcribe anything**, because a
    //: check that made the thing it is checking do work is a check with a side effect.
    let whisper_up = std::net::TcpStream::connect_timeout(
        &argmind_env_whisper::WHISPER.parse().expect("a literal address"),
        std::time::Duration::from_millis(400)).is_ok();
    out.push(Need {
        what: "받아쓰기",
        required: false,
        //: A server you build and run, the way Ollama is. Not ours to install.
        present: whisper_up,
        found: if whisper_up {
            format!("{} 응답함", argmind_env_whisper::WHISPER)
        } else {
            format!("{} 응답 없음 — 음성 입력만 안 됩니다", argmind_env_whisper::WHISPER)
        },
        how: "whisper.cpp 를 받아 whisper-server 를 8178 포트로 띄우세요".into(),
        //: Same reasoning as Ollama, and more so — this one is built from source, and fetching and
        //: running somebody's build is exactly what this refuses to do unseen.
        fix: Some(Fix::OpenPage {
            url: "https://github.com/ggml-org/whisper.cpp".into(),
            why: "음성 입력에는 받아쓰기 서버가 필요합니다.".into(),
        }),
    });

    //: Built once and dropped. Enumerating voices reads a registry category and opens no audio
    //: device, so this is as quiet as the two checks above it.
    let korean = argmind_env_windows::SapiSpeaker::new().map(|v| v.has_korean()).unwrap_or(false);
    out.push(Need {
        what: "한국어 음성",
        required: false,
        //: A Windows language feature, and installing one needs elevation.
        present: korean,
        found: if korean {
            "설치됨".into()
        } else {
            //: **The measured failure, named.** An English voice handed Korean produces zero bytes
            //: and reports success, which is why the router refuses rather than falling back.
            "없음 — 한국어 답변은 소리로 나오지 않습니다".into()
        },
        how: "설정 > 시간 및 언어 > 언어 > 한국어 > 옵션 > 음성 을 설치하세요".into(),
        //: **This one argmind can actually do**, because it is one documented Windows command
        //: rather than somebody's binary — and Windows asks for administrator rights itself, so
        //: two separate windows say yes before anything changes.
        fix: Some(Fix::RunElevated {
            command: "DISM /Online /Add-Capability /CapabilityName:Language.Speech~~~ko-KR~0.0.1.0".into(),
            why: "한국어 음성이 없으면 한국어 답변은 소리로 나오지 않습니다.".into(),
        }),
    });

    out
}

/// The lines `check` prints.
pub fn report(needs: &[Need]) -> String {
    let mut out = String::new();
    for n in needs {
        out.push_str(&format!("  {:<10} {:<14} {}\n", n.what, n.mark(), n.found));
        if !n.present {
            out.push_str(&format!("             {}\n", n.how));
        }
    }
    let missing: Vec<&Need> = needs.iter().filter(|n| !n.present).collect();
    out.push('\n');
    if missing.is_empty() {
        out.push_str("  필요한 건 다 있습니다.\n");
        return out;
    }
    let offered = missing.iter().filter(|n| n.fix.is_some()).count();
    out.push_str(&format!("  {}개가 없습니다. argmind setup 을 실행하면 {offered}개에 대해 \
                           물어봅니다.\n", missing.len()));
    //: **Named, not counted.** Saying that nothing can be done about one of them is only useful if
    //: it says which one.
    for n in missing.iter().filter(|n| n.fix.is_none()) {
        out.push_str(&format!("  {} 은(는) argmind가 할 수 있는 게 없습니다.\n", n.what));
    }
    out
}
