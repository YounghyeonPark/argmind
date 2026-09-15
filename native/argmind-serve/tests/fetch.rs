//! Fetching, and knowing what arrived.
//!
//! **The network ones are `#[ignore]`d**, so the suite still runs with no network — and they are
//! here rather than absent because the thing being checked is whether a real server, a real
//! certificate and a real hash line up, which a fake cannot answer. Run them with
//! `cargo test -p argmind-serve --test fetch -- --ignored`.

use std::path::PathBuf;

use argmind_serve::fetch::{digest, get, shed};

fn scratch(name: &str) -> PathBuf {
    let at = std::env::temp_dir().join("argmind-fetch-test");
    let _ = std::fs::create_dir_all(&at);
    at.join(name)
}

/// **Plain HTTP is refused before anything is attempted.**
///
/// A download nobody can authenticate is one an approval cannot cover, whatever the dialog said —
/// so this is decided by the address rather than by what came back.
#[test]
fn a_download_that_cannot_be_authenticated_is_refused() {
    let into = scratch("never.bin");
    let _ = std::fs::remove_file(&into);
    let why = get("http://example.com/x", &into, None, &mut |_| {}).unwrap_err();
    assert!(why.contains("https"), "{why}");
    //: And nothing was written — a refusal that left a file behind would be a refusal in name.
    assert!(!into.exists());

    //: Not a URL at all is the same answer.
    assert!(get("example.com/x", &into, None, &mut |_| {}).is_err());
    assert!(get("", &into, None, &mut |_| {}).is_err());
    assert!(get("ftp://example.com/x", &into, None, &mut |_| {}).is_err());
}

/// The digest of a file whose contents are known.
#[test]
fn a_file_hashes_to_what_it_hashes_to() {
    let at = scratch("known.txt");
    std::fs::write(&at, b"abc").unwrap();
    //: The SHA-256 of "abc" — one of the few hashes worth writing from memory, and checkable
    //: against any implementation.
    assert_eq!(digest(&at).unwrap(),
               "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    //: **An empty file has a well-known hash and `certutil` will not tell you it.** It fails with
    //: `ERROR_FILE_INVALID` rather than answering `e3b0c442…`, which was a surprise and is written
    //: down rather than worked around: `get` rejects a zero-byte download *before* it hashes
    //: anything, so nothing empty ever reaches here — and if that check were ever removed, this
    //: says what the next failure would look like.
    std::fs::write(&at, b"").unwrap();
    assert!(digest(&at).is_err());
    //: A file that is not there cannot be hashed, and says so rather than answering zero.
    assert!(digest(&scratch("no-such-file")).is_err());
    let _ = std::fs::remove_file(&at);
}

/// Downloads land somewhere ours, under the user's temp.
#[test]
fn downloads_do_not_land_beside_the_program_or_in_your_documents() {
    let where_ = shed();
    let temp = std::env::temp_dir();
    assert!(where_.starts_with(&temp), "{where_:?} is not under {temp:?}");
    assert!(where_.to_string_lossy().contains("argmind"));
}

/// **A wrong hash means the file goes.**
///
/// Leaving a failed download on disk invites somebody to run it in six months, and the name it is
/// under says nothing about having failed a check. Uses the network, so it is ignored by default.
#[ignore = "needs the network"]
#[test]
fn a_download_whose_hash_is_wrong_is_deleted() {
    let into = scratch("readme.md");
    let _ = std::fs::remove_file(&into);
    let url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/README.md";

    //: First with no hash: it arrives, because nobody published one and that is a fact rather
    //: than a failure.
    get(url, &into, None, &mut |s| println!("  {s}")).unwrap();
    assert!(into.exists());
    let real = digest(&into).unwrap();

    //: Then with the right one.
    get(url, &into, Some(&real), &mut |s| println!("  {s}")).unwrap();
    assert!(into.exists());

    //: Then with a wrong one — refused, and gone.
    let wrong = "0".repeat(64);
    let why = get(url, &into, Some(&wrong), &mut |s| println!("  {s}")).unwrap_err();
    assert!(why.contains("해시"), "{why}");
    assert!(!into.exists(), "a download that failed its hash was left on disk");
}

/// The published hash for the recogniser model is the one argmind will check against.
///
/// **A hash written from memory is a download that always fails.** This asks the server what it
/// publishes and compares it with the constant in `needs.rs`, so the two cannot drift apart
/// silently. Ignored by default: it needs the network.
#[ignore = "needs the network"]
#[test]
fn the_model_hash_in_the_source_is_the_one_the_server_publishes() {
    let out = std::process::Command::new("curl.exe")
        .args(["-sIL", "--max-time", "30",
               "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin"])
        .output().unwrap();
    let head = String::from_utf8_lossy(&out.stdout).to_lowercase();
    let published = head.lines()
        .find(|l| l.starts_with("x-linked-etag:"))
        .and_then(|l| l.split('"').nth(1))
        .expect("the server publishes a hash for it")
        .to_string();
    assert_eq!(published,
               "60ed5bc3dd14eea856493d334349b405782ddcaf0028d4b5df4088345fba2efe");
}
