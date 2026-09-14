//! Save real widget lists, so choosing can be measured without a live desktop.
//!
//! ```text
//!   capture                    watch for 30 s and report what it would keep
//!   capture 90                 longer, if you want to alt-tab through more of them
//!   capture 90 0.5 --write     and actually write the new fixtures
//! ```
//!
//! **Writing is opt-in.** `s` is a hash of this directory and four committed results name the value
//! it has today, so adding a screen makes every result taken before this run incomparable with
//! every result taken after it. That is sometimes exactly what is wanted and it is never something
//! to do by accident, so the default reports and the flag writes.
//!
//! Alt-tab through the windows you want captured while it runs. Nothing is clicked and nothing is
//! brought to the front: it reads whatever the person puts in front of it, which is why the
//! fixtures are real screens rather than screens invented to be answerable.
//!
//! **It refuses to overwrite.** Every harness in this directory names a hash of these files, so
//! replacing one silently makes every committed result incomparable with every future one for a
//! reason nobody can see. A new capture writes beside the old and says which files it would have
//! replaced.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use argmind_core::contract::Eyes;
use argmind_env_windows::WindowsEyes;
use argmind_measure::capture::{screen_json, slug, Kept};
use argmind_measure::screen_print;

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap()
        .to_path_buf();
    let dir = root.join("measure").join("screens");
    let args = std::env::args().skip(1);
    let write = std::env::args().any(|a| a == "--write");
    let numbers: Vec<f64> = args.filter(|a| a != "--write")
        .filter_map(|a| a.parse().ok()).collect();
    let seconds = numbers.first().copied().unwrap_or(30.0);
    let every = numbers.get(1).copied().unwrap_or(1.5);

    let before = screen_print(&dir);
    println!("  screens s{before} before this run");
    println!("  watching for {seconds:.0} s — alt-tab through the windows you want captured\n");

    let mut eyes = WindowsEyes::new().expect("UI Automation would not open");
    let mut kept = Kept::default();
    let end = Instant::now() + Duration::from_secs_f64(seconds);
    while Instant::now() < end {
        let widgets = eyes.foreground();
        //: Read off the reading itself, before it is moved into `kept` — printing the largest
        //: window instead would name the wrong one whenever a smaller window was just improved.
        let seen = widgets.first().map(|w| (w.window.clone(), widgets.len()));
        if kept.offer(widgets) {
            if let Some((title, n)) = seen {
                println!("  {n:>4} widgets  {}", title.chars().take(70).collect::<String>());
            }
        }
        std::thread::sleep(Duration::from_secs_f64(every));
    }
    println!();

    std::fs::create_dir_all(&dir).unwrap();
    let (mut written, mut refused, mut would) = (0, 0, 0);
    for (title, widgets) in &kept.windows {
        let path = dir.join(format!("{}.json", slug(title)));
        //: **Never over an existing fixture.** `s` is a hash of this directory and four committed
        //: results name the value it has today; a silent replacement would leave those results
        //: describing screens that no longer exist.
        if path.exists() {
            println!("  REFUSED  {} already exists — move it aside if you mean to replace it",
                     path.file_name().unwrap().to_string_lossy());
            refused += 1;
            continue;
        }
        if !write {
            println!("  WOULD WRITE  {} ({} widgets) — pass --write to keep it",
                     path.file_name().unwrap().to_string_lossy(), widgets.len());
            would += 1;
            continue;
        }
        std::fs::write(&path, screen_json(title, widgets)).unwrap();
        written += 1;
    }

    let after = screen_print(&dir);
    println!("\n  {written} written, {would} not written, {refused} refused -> {}",
             dir.display());
    println!("  screens s{before} -> s{after}{}", if before == after { " (unchanged)" } else { "" });
    if before != after {
        println!("  **every harness names s, so results taken before this are not comparable with \
                  results taken after it.**");
    }
    if eyes.vanished > 0 {
        println!("  ({} elements died between being listed and being read — that is the staleness \
                  this design assumes, not an error)", eyes.vanished);
    }
}
