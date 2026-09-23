//! **The two model files the face path needs, and where they come from.**
//!
//! Constants and paths only. This names no environment crate on purpose, the same way
//! `speaker.rs` does: it is compiled into the headless build and into the Linux cross-check, where
//! `argmind-env-face` is not. What actually runs the graphs is `platform.rs`.
//!
//! ## Two keys, not one, and the project has made this mistake before
//!
//! Downloading 37 MB of weights and remembering somebody's face are different decisions:
//!
//! * **`face-models`** — fetch the files. Changes the machine, is a need with a button, and buys
//!   nothing on its own.
//! * **`faces`** — keep a row about a person. Outlives the session, and is what `people.rs` guards.
//!
//! Collapsing them would repeat `speaker` against `aloud`, where permission to *build* the
//! synthesiser was read as permission to *make a noise in the room* and the owner had to point at
//! it. The same shape again would be nobody's fault but this file's.
//!
//! ## Why these two files
//!
//! The default choice for this job is InsightFace's `buffalo_l`, whose code is MIT and whose
//! **weights are non-commercial**. The owner's rule is 오픈소스라이센스에 문제 없는것만 사용할것,
//! so the pair below is what is left that is small, fast and clean — YuNet under MIT and SFace
//! under Apache-2.0, both from OpenCV's model zoo. `docs/licences.toml` has the record.
//!
//! **Nothing here is executed.** Both are weights read by `tract`, fetched over TLS and checked
//! against the hashes below, which is why they are a `Fix::Fetch` and not a `Fix::Install`.

use std::path::PathBuf;

use crate::fix::Fix;

/// Face detection: a box and five landmarks. **232,589 bytes**, MIT.
pub const FINDER_URL: &str = "https://huggingface.co/opencv/opencv_zoo/resolve/main/\
models/face_detection_yunet/face_detection_yunet_2023mar.onnx";
pub const FINDER_SHA256: &str =
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4";
pub const FINDER_BYTES: u64 = 232_589;
pub const FINDER_FILE: &str = "face_detection_yunet_2023mar.onnx";

/// Face recognition: 128 numbers. **38,696,353 bytes**, Apache-2.0.
pub const NAMER_URL: &str = "https://huggingface.co/opencv/opencv_zoo/resolve/main/\
models/face_recognition_sface/face_recognition_sface_2021dec.onnx";
pub const NAMER_SHA256: &str =
    "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79";
pub const NAMER_BYTES: u64 = 38_696_353;
pub const NAMER_FILE: &str = "face_recognition_sface_2021dec.onnx";

/// Both together, for the sentence that has to name a size. **37 MB**, and the detector is 0.2 of
/// it — a person agreeing to this is agreeing to the recogniser.
pub const BOTH_MB: u64 = (FINDER_BYTES + NAMER_BYTES) / 1_048_576;

pub fn finder_at() -> PathBuf {
    crate::needs::model_shed().join(FINDER_FILE)
}

pub fn namer_at() -> PathBuf {
    crate::needs::model_shed().join(NAMER_FILE)
}

/// **Both files, present and the right size.**
///
/// Size rather than a hash, because this is asked every time the settings page is drawn and
/// hashing 37 MB to render a row would be a third of a second of nothing. The hash is checked
/// where it matters — on the way in, by [`crate::fetch::get`] — and a file that is the right length
/// and the wrong bytes fails at `tract` with a message rather than silently.
pub fn here() -> bool {
    [(finder_at(), FINDER_BYTES), (namer_at(), NAMER_BYTES)].iter()
        .all(|(p, n)| std::fs::metadata(p).map(|m| m.len() == *n).unwrap_or(false))
}

/// The two fixes that fetch them.
///
/// **Returned as fixes rather than as a download function** for the reason `speaker::weight_fixes`
/// gives: `fix::apply` runs exactly these, so what somebody agreed to and what happens cannot come
/// apart.
pub fn model_fixes() -> [Fix; 2] {
    [
        Fix::Fetch {
            url: FINDER_URL.into(),
            into: finder_at().to_string_lossy().to_string(),
            why: "얼굴을 찾는 모델입니다 (MIT).".into(),
            about_mb: (FINDER_BYTES / 1_048_576).max(1),
            sha256: Some(FINDER_SHA256.into()),
        },
        Fix::Fetch {
            url: NAMER_URL.into(),
            into: namer_at().to_string_lossy().to_string(),
            why: "같은 얼굴인지 판단하는 모델입니다 (Apache-2.0).".into(),
            about_mb: NAMER_BYTES / 1_048_576,
            sha256: Some(NAMER_SHA256.into()),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn both_fixes_fetch_and_neither_installs() {
        //: **Weights are data.** A `Fix::Install` downloads and *runs* something, and the window
        //: that asks says so; these are read by a library that was already here.
        for fix in model_fixes() {
            match fix {
                Fix::Fetch { sha256, url, .. } => {
                    assert!(sha256.is_some(), "a weight file with no hash is a file from nowhere");
                    assert!(url.starts_with("https://"), "over TLS or not at all");
                }
                other => panic!("a model file must be fetched, not {other:?}"),
            }
        }
    }

    #[test]
    fn the_sentence_names_the_bigger_file() {
        //: 37 MB, near enough all of it the recogniser — so a person reading the offer is reading
        //: about the part that remembers faces rather than the part that finds them.
        assert!(BOTH_MB >= 36 && BOTH_MB <= 38, "BOTH_MB drifted to {BOTH_MB}");
        assert!(NAMER_BYTES > FINDER_BYTES * 100);
    }

    #[test]
    fn the_two_files_do_not_land_on_each_other() {
        assert_ne!(finder_at(), namer_at());
        assert!(finder_at().starts_with(crate::needs::model_shed()));
    }
}
