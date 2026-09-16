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

/// Where models argmind fetches are kept.
///
/// **Beside the person's own files, not beside the executable.** A folder under the executable
/// needs write access to wherever argmind was unpacked, which is a permission the program should
/// not want.
pub fn model_shed() -> std::path::PathBuf {
    let mut at = std::env::var("LOCALAPPDATA").map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    at.push("argmind");
    at.push("models");
    at
}

/// One thing argmind wants.
#[derive(Debug, Clone)]
pub struct Need {
    /// **A stable name, not a position.** A button in a page that had been open a while would
    /// otherwise fix whatever had moved into its slot — the same defect the proposal numbers are
    /// built to avoid, arriving in a different part of the program.
    pub key: &'static str,
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

/// One need by its key, out of a freshly taken reading.
///
/// **Re-taken rather than remembered.** A page open for an hour is a page whose list is an hour
/// old, and acting on it would offer to install something that is already there.
pub fn by_key(model: &str, key: &str) -> Option<Need> {
    every(model).into_iter().find(|n| n.key == key)
}

/// Everything, in the order a person would fix it.
pub fn every(model: &str) -> Vec<Need> {
    let mut out = Vec::new();

    let ollama_up = argmind_env_ollama::manage::awake(argmind_env_ollama::OLLAMA);
    out.push(Need {
        key: "ollama",
        what: "Ollama",
        required: true,
        present: ollama_up,
        found: if ollama_up { "127.0.0.1:11434 응답함".into() } else { "응답 없음".into() },
        how: "https://ollama.com 에서 설치한 뒤 실행해 두세요".into(),
        //: **Downloaded and run, on one approval.** The earlier version opened a page and refused
        //: to fetch anybody's executable; the owner's decision is that the window should ask and
        //: the agent should then do the whole thing. What makes that approval worth having is
        //: naming the source and verifying the certificate — recorded here because *"a file nobody
        //: had seen, from a name nobody had checked"* was the objection, and this is the part of
        //: it that is answered rather than overruled.
        fix: Some(Fix::Install {
            url: "https://ollama.com/download/OllamaSetup.exe".into(),
            file: "OllamaSetup.exe".into(),
            why: "Ollama가 있어야 argmind가 동작합니다.".into(),
            about_mb: 700,
        }),
    });

    //: **Only asked when Ollama is up.** "The model is missing" and "nothing was there to ask" are
    //: different facts, and reporting the first when the second is true sends somebody looking for
    //: a five gigabyte download they already have.
    let has_model = ollama_up && argmind_env_ollama::manage::has(argmind_env_ollama::OLLAMA, model);
    out.push(Need {
        key: "model",
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
    //: Asked once and used twice — for the sentence and for whether there is a button at all.
    let missing = crate::recogniser::missing_tools();
    out.push(Need {
        key: "whisper",
        what: "받아쓰기",
        required: false,
        present: whisper_up,
        found: if whisper_up {
            format!("{} 응답함", argmind_env_whisper::WHISPER)
        } else {
            format!("{} 응답 없음 — 음성 입력만 안 됩니다", argmind_env_whisper::WHISPER)
        },
        how: if missing.is_empty() {
            "argmind가 소스에서 빌드해 띄울 수 있습니다".into()
        } else {
            //: **Names the tool rather than saying it cannot.** "빌드할 수 없습니다" sends somebody
            //: looking; the name of the missing program is the whole answer.
            format!("빌드에 {} 이(가) 필요합니다 — 설치하시면 argmind가 나머지를 합니다",
                    missing.join(", "))
        },
        //: **Both halves, on one approval** — the model is data and the server is a build, and
        //: from the person's side they are one decision: make voice input work. What the sequence
        //: will do is spelled out in the window rather than split across two of them.
        //:
        //: **Offered only when it can actually be done.** A button whose first act is to discover
        //: there is no compiler teaches somebody that argmind is unreliable, and it teaches them
        //: that after they said yes.
        fix: missing.is_empty().then(|| Fix::BuildRecogniser {
            repo: crate::recogniser::REPO.into(),
            commit: crate::recogniser::COMMIT.into(),
            why: "음성 입력에는 받아쓰기 서버가 필요합니다.".into(),
            minutes: crate::recogniser::MINUTES,
            //: Cheap: a file beside the binary, written by the build that produced it.
            built: crate::recogniser::ready(),
        }),
    });

    //: Built once and dropped. Enumerating voices reads a registry category and opens no audio
    //: device, so this is as quiet as the two checks above it.
    let korean = argmind_env_windows::SapiSpeaker::new().map(|v| v.has_korean()).unwrap_or(false);
    out.push(Need {
        key: "korean-voice",
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

    //: **Its own need, not a condition on the one below it.** The SDK is a machine-wide developer
    //: kit installed with administrator rights; the voice is a build. Folding them into one key
    //: would mean a remembered *항상허용* for "the voice" silently covering an elevated install,
    //: and a decision is only worth remembering if it was about one thing.
    let vulkan = crate::speaker::sdk();
    out.push(Need {
        key: "vulkan",
        what: "Vulkan SDK",
        required: false,
        present: vulkan.is_some(),
        found: match &vulkan {
            Some(at) => format!("{}", at.display()),
            //: **Measured, and the number is the reason.** Same binary, same weights, GPU disabled
            //: with `GGML_VK_VISIBLE_DEVICES=""`: RTF 0.15 against 4.8.
            None => "없음 — 신경망 목소리는 CPU로는 32배 느려서 못 씁니다".into(),
        },
        how: "winget install --id KhronosGroup.VulkanSDK — 또는 https://vulkan.lunarg.com/sdk/home"
            .into(),
        fix: Some(crate::speaker::sdk_fix()),
    });

    let speaking = crate::speaker::answering();
    //: Asked once and used twice — for the sentence and for whether there is a button at all.
    let missing = crate::speaker::missing_tools();
    out.push(Need {
        key: "speaker",
        what: "목소리",
        required: false,
        present: speaking,
        found: if speaking {
            format!("{} 응답함", argmind_env_qwentts::QWENTTS)
        } else {
            //: **What is lost is named, and it is not everything.** Windows still speaks; this is
            //: the difference between Heami and a neural voice, not between sound and silence.
            format!("{} 응답 없음 — Windows 음성으로 말합니다",
                    argmind_env_qwentts::QWENTTS)
        },
        how: if vulkan.is_none() {
            "Vulkan SDK를 먼저 설치하세요".into()
        } else if missing.is_empty() {
            "argmind가 소스에서 빌드해 띄울 수 있습니다".into()
        } else {
            format!("빌드에 {} 이(가) 필요합니다 — 설치하시면 argmind가 나머지를 합니다",
                    missing.join(", "))
        },
        //: **Offered only when it can actually be done**, which here means the tools *and* the SDK.
        //: A button whose first act is to discover there is no compiler teaches somebody that
        //: argmind is unreliable, and it teaches them that after they said yes.
        fix: (missing.is_empty() && vulkan.is_some()).then(|| Fix::BuildSpeaker {
            repo: crate::speaker::REPO.into(),
            commit: crate::speaker::COMMIT.into(),
            why: "Windows 한국어 음성은 오래된 것 하나뿐이라 어색합니다.".into(),
            minutes: crate::speaker::MINUTES,
            built: crate::speaker::ready(),
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
