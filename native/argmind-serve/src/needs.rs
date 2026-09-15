//! What argmind needs, whether it is here, and **which of it is safe to fetch.**
//!
//! Everything this program wants falls into one of three bands, and the difference between them is
//! the whole design:
//!
//! * **Already here or genuinely absent, and free to check.** A socket that answers, a voice that
//!   is installed, a microphone that exists. Looking costs nothing and changes nothing.
//! * **Safe to fetch.** A model, pulled by Ollama into Ollama's own store, over the socket this
//!   project already uses. No installer runs, no executable is downloaded, nothing outside a
//!   program the person already chose to run is touched.
//! * **Not ours to install.** Ollama itself, a recogniser, a Windows language pack. Each means
//!   running an installer or elevating, and **a program that can click your screen must not also
//!   be able to install software while you are not looking.** These are reported with the exact
//!   command and nothing else.
//!
//! The band is a property of the need, written next to it, rather than a judgement made at the
//! call site — so adding a need means deciding which band it is in.

/// How much this program may do about a missing thing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Band {
    /// We can fetch it ourselves: no installer, no executable, nothing outside a program already
    /// running by the person's choice.
    Fetchable,
    /// Report it and print the command. **Installing it is the person's decision to make.**
    TheirCall,
}

/// One thing argmind wants.
#[derive(Debug, Clone)]
pub struct Need {
    pub what: &'static str,
    /// Whether the agent works at all without it.
    pub required: bool,
    pub band: Band,
    pub present: bool,
    /// What is true right now, in a sentence.
    pub found: String,
    /// What to do about it, when it is missing.
    pub how: String,
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
        //: An installer. Not ours to run.
        band: Band::TheirCall,
        present: ollama_up,
        found: if ollama_up { "127.0.0.1:11434 응답함".into() } else { "응답 없음".into() },
        how: "https://ollama.com 에서 설치한 뒤 실행해 두세요".into(),
    });

    //: **Only asked when Ollama is up.** "The model is missing" and "nothing was there to ask" are
    //: different facts, and reporting the first when the second is true sends somebody looking for
    //: a five gigabyte download they already have.
    let has_model = ollama_up && argmind_env_ollama::manage::has(argmind_env_ollama::OLLAMA, model);
    out.push(Need {
        what: "모델",
        required: true,
        band: Band::Fetchable,
        present: has_model,
        found: if !ollama_up {
            "Ollama가 없어서 확인 못 함".into()
        } else if has_model {
            format!("{model} 있음")
        } else {
            format!("{model} 없음")
        },
        how: format!("argmind setup 이 받아옵니다 (약 5 GB) — 또는 ollama pull {model}"),
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
        band: Band::TheirCall,
        present: whisper_up,
        found: if whisper_up {
            format!("{} 응답함", argmind_env_whisper::WHISPER)
        } else {
            format!("{} 응답 없음 — 음성 입력만 안 됩니다", argmind_env_whisper::WHISPER)
        },
        how: "whisper.cpp 를 받아 whisper-server 를 8178 포트로 띄우세요".into(),
    });

    //: Built once and dropped. Enumerating voices reads a registry category and opens no audio
    //: device, so this is as quiet as the two checks above it.
    let korean = argmind_env_windows::SapiSpeaker::new().map(|v| v.has_korean()).unwrap_or(false);
    out.push(Need {
        what: "한국어 음성",
        required: false,
        //: A Windows language feature, and installing one needs elevation.
        band: Band::TheirCall,
        present: korean,
        found: if korean {
            "설치됨".into()
        } else {
            //: **The measured failure, named.** An English voice handed Korean produces zero bytes
            //: and reports success, which is why the router refuses rather than falling back.
            "없음 — 한국어 답변은 소리로 나오지 않습니다".into()
        },
        how: "설정 > 시간 및 언어 > 언어 > 한국어 > 옵션 > 음성 을 설치하세요".into(),
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
    let missing: Vec<&Need> = needs.iter().filter(|n| !n.present && n.required).collect();
    let fetchable = missing.iter().filter(|n| n.band == Band::Fetchable).count();
    out.push('\n');
    if missing.is_empty() {
        out.push_str("  필요한 건 다 있습니다.\n");
    } else {
        out.push_str(&format!("  {}개가 없고, 그중 {fetchable}개는 argmind setup 이 받아올 수 \
                               있습니다.\n", missing.len()));
        //: **Named, not counted.** "One of these is yours to install" is only useful if it says
        //: which one.
        for n in missing.iter().filter(|n| n.band == Band::TheirCall) {
            out.push_str(&format!("  {} 은(는) 직접 설치하셔야 합니다.\n", n.what));
        }
    }
    out
}
