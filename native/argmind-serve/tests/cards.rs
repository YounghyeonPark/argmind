//! **A task card says why a step failed**, not that it failed — the line from a step's result a
//! person reads on the 작업 card and in 한 일.

use argmind_serve::app::gist;

#[test]
fn a_failed_step_shows_the_line_that_names_the_error() {
    //: Word for word what the owner's Blender box returned on 2026-11-09, path shortened.
    let observed = "오류: [stderr]\nTraceback (most recent call last):\n  File \"box.py\", line 15, in <module>\n    \
bpy.ops.wm.stl_export(filepath='part.stl', use_selection=True)\n    ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^\n\
TypeError: Converting py args to operator properties:: keyword \"use_selection\" unrecognized\n\
Error: script failed, file: 'box.py', exiting.\n(종료 코드 1)";
    assert_eq!(gist(observed, true),
               "TypeError: Converting py args to operator properties:: keyword \"use_selection\" unrecognized");
    //: A one-line refusal is its own reason.
    assert_eq!(gist("오류: box.py 이(가) 작업 폴더에 없다 — write_file 로 먼저 쓴다", true),
               "오류: box.py 이(가) 작업 폴더에 없다 — write_file 로 먼저 쓴다");
}

#[test]
fn a_step_that_went_ahead_shows_its_first_line() {
    let observed = "Info: File exported successfully\nTimer 'STL Export' took 0.31 ms\n(종료 코드 0)";
    assert_eq!(gist(observed, false), "Info: File exported successfully");
    assert_eq!(gist("", false), "");
    assert_eq!(gist("", true), "");
}
