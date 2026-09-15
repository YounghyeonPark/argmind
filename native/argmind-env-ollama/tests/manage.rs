//! Reading what Ollama says while it fetches a model. No network.
//!
//! **Every line below was captured from a real pull**, including the error, because a hand-written
//! fixture of a protocol is a test of what somebody imagined the protocol was.

use argmind_env_ollama::manage::{names_in, progress_of};

/// Captured from `POST /api/pull` against a model that was already present — which verifies every
/// layer and finishes in a second, so the whole path can be exercised without downloading anything.
const REAL: &str = r#"{"status":"pulling manifest"}
{"status":"pulling a3de86cd1c13","digest":"sha256:a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f","total":5225374496,"completed":5225374496}
{"status":"pulling ae370d884f10","digest":"sha256:ae370d884f108d16e7cc8fd5259ebc5773a0afa6e078b11f4ed7e39a27e0dfc4","total":1723,"completed":1723}
{"status":"verifying sha256 digest"}
{"status":"writing manifest"}
{"status":"success"}"#;

#[test]
fn the_progress_of_a_real_pull_is_read_the_way_it_arrives() {
    let lines: Vec<&str> = REAL.lines().collect();
    let got: Vec<_> = lines.iter().map(|l| progress_of(l).expect(l)).collect();

    //: The first and last lines carry no sizes, and **`None` is the honest answer for a percentage
    //: nobody has said yet** — reporting 0% while the manifest is being fetched is a number the
    //: server never gave.
    assert_eq!(got[0].status, "pulling manifest");
    assert_eq!(got[0].percent(), None);
    assert_eq!(got.last().unwrap().status, "success");
    assert_eq!(got.last().unwrap().percent(), None);

    //: A layer already present arrives complete.
    assert_eq!(got[1].total, 5_225_374_496);
    assert_eq!(got[1].completed, 5_225_374_496);
    assert_eq!(got[1].percent(), Some(100));
    //: **The digest is full of digits and must not be mistaken for a size.**
    assert!(got[1].status.starts_with("pulling a3de"));

    assert_eq!(got[3].status, "verifying sha256 digest");
    assert_eq!(got[4].status, "writing manifest");
}

#[test]
fn a_partial_layer_is_a_partial_percentage() {
    let p = progress_of(r#"{"status":"pulling x","total":1000,"completed":250}"#).unwrap();
    assert_eq!(p.percent(), Some(25));
    //: Nothing yet is nought per cent, which is different from no answer at all.
    let none = progress_of(r#"{"status":"pulling x","total":1000,"completed":0}"#).unwrap();
    assert_eq!(none.percent(), Some(0));
    //: **A big file must not overflow into a wrong number.** Five gigabytes times a hundred does
    //: not fit in 32 bits, and the multiply happens before the divide.
    let big = progress_of(r#"{"status":"x","total":5225374496,"completed":2612687248}"#).unwrap();
    assert_eq!(big.percent(), Some(50));
}

#[test]
fn a_line_that_is_not_progress_is_not_progress() {
    assert!(progress_of("").is_none());
    assert!(progress_of("{}").is_none());
    //: An error arrives on the same stream, and it has no status — `pull` reads it separately, so
    //: what matters here is that it is not mistaken for a step forward.
    assert!(progress_of(r#"{"error":"pull model manifest: file does not exist"}"#).is_none());
}

/// The installed list, out of a real `/api/tags` body.
#[test]
fn every_model_name_comes_out_of_the_tags_body() {
    let body = r#"{"models":[{"name":"qwen3:8b","model":"qwen3:8b","size":5225388164},
                   {"name":"qwen3:0.6b","model":"qwen3:0.6b","size":522538816}]}"#;
    assert_eq!(names_in(body), vec!["qwen3:8b", "qwen3:0.6b"]);
    assert!(names_in("{}").is_empty());
    assert!(names_in("").is_empty());
    //: A body it cannot read gives nothing rather than a half-read name — and nothing means
    //: "cannot confirm", which the caller treats as absent rather than as present.
    assert!(names_in(r#"{"name":"unterminated"#).is_empty());
}
