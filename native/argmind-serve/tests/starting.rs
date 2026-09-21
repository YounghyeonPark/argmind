//! **Already built means start it, not build it again.**
//!
//! The recogniser and the synthesiser die with argmind — a job object, on purpose — so at every
//! launch they are not answering on their ports, which is what `present` means for them, which puts
//! them on the standing-yes list and runs their *build* fix again. Measured 2026-09-21 on a machine
//! where nothing had changed: **22 s from launch to a voice**, and the four processes that ran in
//! it were `cmake` (4.0 s), `mingw32-make`, a `git` fetch and `glslc` regenerating Vulkan shaders
//! (4.7 s). None of them produced a byte that was not already there. With the short circuit,
//! **7.1 s** and none of the four run at all.
//!
//! The owner noticed before any of that was measured: *앱이 실행될때마다 서버를 매번 설치하는것
//! 같은데 이럴 필요가 있나?*
//!
//! **Nothing here builds, downloads or starts anything.** The decision was split out of `apply` so
//! that it could be checked on a machine with no compiler, which is the only way this gets a test
//! at all — `apply` itself spends fifteen minutes and a network.

use std::fs;
use std::path::{Path, PathBuf};

use argmind_serve::fix::can_just_start;

const COMMIT: &str = "90efdc60d62f7d6c70113994d0be17e820fd8673";

/// One arm of `apply`, from its pattern to the next one. **Read out of `apply` rather than the
/// whole file**, so that `describe`'s text for the same variant cannot satisfy a rule about order.
fn arm_of(arm: &str) -> String {
    let source = include_str!("../src/fix.rs");
    let (_, apply) = source.split_once("pub fn apply(").expect("apply is gone");
    let from = apply.find(arm).unwrap_or_else(|| panic!("{arm} is gone from apply"));
    let rest = &apply[from + arm.len()..];
    let to = ["Fix::BuildRecogniser {", "Fix::BuildSpeaker {", "Fix::RunElevated {"].iter()
        .filter_map(|next| rest.find(next)).min().unwrap_or(rest.len());
    format!("{arm}{}", &rest[..to])
}

/// A scratch directory of our own, removed at the end. **Not the real shed** — a test that wrote
/// `built-at.txt` beside the owner's binary could talk them out of a rebuild they need.
struct Shed(PathBuf);

impl Shed {
    fn new(name: &str) -> Shed {
        let at = std::env::temp_dir().join(format!("argmind-starting-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&at);
        fs::create_dir_all(&at).expect("a scratch directory");
        Shed(at)
    }

    fn exe(&self) -> PathBuf {
        self.0.join("some-server.exe")
    }

    fn build(&self, commit: Option<&str>) {
        fs::write(self.exe(), b"not really a binary").unwrap();
        if let Some(commit) = commit {
            fs::write(self.0.join("built-at.txt"), commit).unwrap();
        }
    }
}

impl Drop for Shed {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// The one case that has to be fast: built from this commit, weights on disk.
#[test]
fn a_tree_built_from_this_commit_is_started_rather_than_rebuilt() {
    let shed = Shed::new("built");
    shed.build(Some(COMMIT));
    assert!(can_just_start(&shed.exe(), COMMIT, true));
}

/// **Every way it can be not-ready ends in a build**, because the cost of a needless rebuild is
/// minutes and the cost of a wrong skip is a program that does not work.
#[test]
fn anything_less_than_built_and_loadable_goes_the_long_way() {
    let nowhere = Path::new("C:/a/path/that/is/not/there/server.exe");
    assert!(!can_just_start(nowhere, COMMIT, true), "a missing binary was started");

    //: A tree half-built: the binary is there and no stamp was written. `remember_build` runs last
    //: precisely so this reads as *not built*.
    let half = Shed::new("half");
    half.build(None);
    assert!(!can_just_start(&half.exe(), COMMIT, true), "a tree with no stamp was started");

    //: **A binary from an older argmind.** The file is there and it is not the program the window
    //: described, so saying *already built* about it would be a lie that shows up only as a server
    //: behaving unlike the documented one.
    let old = Shed::new("old");
    old.build(Some("0000000000000000000000000000000000000000"));
    assert!(!can_just_start(&old.exe(), COMMIT, false), "a stale build was started");
    assert!(!can_just_start(&old.exe(), COMMIT, true), "a stale build was started");

    //: **Built, and the gigabyte it loads is gone.** The ordinary way somebody frees disk space,
    //: and the honest answer is to fetch it again rather than start a server that cannot open its
    //: model.
    let no_weights = Shed::new("weightless");
    no_weights.build(Some(COMMIT));
    assert!(!can_just_start(&no_weights.exe(), COMMIT, false), "a server with no weights started");
}

/// **Both arms go through it, and neither reaches a compiler first.**
///
/// Held by reading the source, like the rules in `argmind-rules`, because the thing being protected
/// is an *order*: a short circuit placed below the tool check would still skip the build and would
/// still refuse to start an already-built server on a machine with no compiler — which is a machine
/// that does not need one.
#[test]
fn neither_build_fix_can_reach_a_compiler_before_the_short_circuit() {
    for arm in ["Fix::BuildRecogniser {", "Fix::BuildSpeaker {"] {
        let body = arm_of(arm);
        let body = body.as_str();

        let start = body.find("can_just_start(")
            .unwrap_or_else(|| panic!("{arm} does not short circuit; it will rebuild every launch"));
        //: Everything that costs minutes has to come after it.
        for slow in ["missing_tools()", "clone_at(", "::build(", "bring_runtime(", "sdk()"] {
            if let Some(at) = body.find(slow) {
                assert!(at > start,
                        "{arm} reaches {slow} before can_just_start, so an already-built server \
                         still pays for it");
            }
        }
        //: And the skip has to actually skip — a condition with no `return` under it changes
        //: nothing at all.
        let after = &body[start..];
        let ret = after.find("return Ok(").expect("the short circuit does not return");
        assert!(ret < after.find("clone_at(").unwrap_or(usize::MAX),
                "{arm} clones before it returns");
    }
}

/// **`remember_build` is written last, and the short circuit rests on that.**
///
/// If the stamp were written before the build, a build interrupted halfway would leave a tree that
/// every later launch happily *starts* — and the failure would be a server that exits, not a
/// message about a broken build. This is the assumption, asserted where it can be seen.
#[test]
fn the_stamp_is_the_last_thing_a_build_writes() {
    for (arm, who) in [("Fix::BuildRecogniser {", "recogniser"), ("Fix::BuildSpeaker {", "speaker")] {
        let body = arm_of(arm);
        let stamp = body.find(&format!("crate::{who}::remember_build("))
            .unwrap_or_else(|| panic!("{who} no longer records what it built"));
        for earlier in ["::build(", "bring_runtime("] {
            let at = body.find(earlier)
                .unwrap_or_else(|| panic!("{who} no longer calls {earlier}"));
            assert!(stamp > at, "{who} stamps the tree before {earlier}");
        }
        //: And the stamp comes before the build path starts the server. **`rfind`, because the
        //: short circuit calls `serve` too** — the first one in this arm is the fast path, which
        //: by construction runs no build and writes no stamp.
        let served = body.rfind(&format!("crate::{who}::serve(")).expect("serve is gone");
        assert!(stamp < served, "{who} starts the server before recording the build");
    }
}
