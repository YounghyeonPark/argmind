//! Making a list longer without making it easier.
//!
//! **Split out of `--bin crowding` so the construction can be tested without a model.** The binary
//! asks whether choosing survives a list the size of a web page; this is the part that builds the
//! list, and every way it could quietly flatter the answer is a way the sandbox gets started on a
//! number nobody measured. `tests/crowding.rs` is the three of them.

use argmind_core::contract::Widget;

use crate::choosing::Goal;

/// A small deterministic shuffle, so a rerun is the same experiment and no crate is added for it.
pub fn spread(seed: u64, of: usize) -> Vec<usize> {
    let mut at = Vec::with_capacity(of);
    let mut x = seed | 1;
    for _ in 0..of {
        //: xorshift64, which is four lines and enough to scatter positions.
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        at.push(x as usize);
    }
    at
}

/// The screen, padded to `want` widgets with plausible ones from elsewhere.
///
/// Returns the list and where the goal's answer ended up, or `None` for a goal whose answer is
/// *decline* — those have no index to track and are scored on refusing.
pub fn crowd(real: &[Widget], pool: &[Widget], goal: &Goal, want: usize, seed: u64)
         -> (Vec<Widget>, Option<usize>) {
    if want <= real.len() {
        let at = goal.accept.as_ref()
            .and_then(|a| real.iter().position(|w| a.iter().any(|n| n == &w.name)));
        return (real.to_vec(), at);
    }
    let here: Vec<&str> = real.iter().map(|w| w.name.as_str()).collect();
    let accepted: Vec<&str> = goal.accept.as_deref().unwrap_or(&[]).iter()
        .map(String::as_str).collect();
    let usable: Vec<&Widget> = pool.iter()
        .filter(|w| !w.name.trim().is_empty())
        .filter(|w| !here.contains(&w.name.as_str()))
        .filter(|w| !accepted.contains(&w.name.as_str()))
        .collect();
    if usable.is_empty() {
        return (real.to_vec(), None);
    }
    let mut out = real.to_vec();
    let need = want - real.len();
    let order = spread(seed, need);
    for (i, r) in order.iter().enumerate() {
        //: **Inserted rather than appended**, so the real widgets are scattered through the list
        //: instead of sitting in the first rows where a model would find them without reading.
        let where_ = r % (out.len() + 1);
        out.insert(where_, usable[i % usable.len()].clone());
    }
    let at = goal.accept.as_ref()
        .and_then(|a| out.iter().position(|w| a.iter().any(|n| n == &w.name)));
    (out, at)
}

