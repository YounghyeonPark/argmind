//! **Whether a face is somebody already known, and when a new one may be written down.**
//!
//! Pure, and here rather than beside the roster for one reason: the harness that measures the rule
//! and the application that runs it must run **the same rule**. `argmind-measure` cannot depend on the
//! application, and the roster in `argmind-serve` is compiled on every platform while the face models
//! are not — so the arithmetic lives in the one crate both can reach. It names no platform and holds
//! no coordinate; a print is 128 numbers about a face, not a place on a screen.
//!
//! ## Why a new face has to be seen twice
//!
//! The owner, 2026-09-30: *얼굴을 치웠다 다시 보이면 이름을 또 물어봄.* The roster reached nine rows
//! for one person. `--bin recognising` found where: sitting still, every print matched the centre of
//! the others (20 of 20, median 0.987); turning the head, still 20 of 20; **leaning out of the frame
//! and back, 8 of 13** — the frames of a face arriving and leaving are the ones that come out wrong.
//! A gate on head direction, frame edge and detector score was tried against the same run and **did
//! not separate them** (81% of passing frames over threshold, 88% of failing ones), so the bad frames
//! cannot be picked out one at a time from what the detector reports.
//!
//! **What they have in common is that they do not agree with each other.** All nine rows sat under
//! 0.363 against each other — **36 of 36 pairs**, the closest at 0.3628. (A first reading of the
//! rounded table said 35, taking a 0.363 on screen for one over the line.) So a face nobody knows is
//! held once,
//! and written down only when **the next unknown face agrees with it** — two frames of a real new
//! person agree; two frames of a face in the act of leaving, measured, almost never do. The print kept
//! is the mean of the two, because a single frame is a poor anchor: nine rows each built on one
//! arbitrary frame is how the roster came to hold nine people.
//!
//! **A known face clears what was held.** One unmatched frame between two that match somebody is a
//! bad frame of them, not a stranger.

/// **OpenCV's cosine threshold for SFace**, and still not measured on two people here — one person
/// is all this machine has had in front of it. With the preparation matching OpenCV's, it is at least
/// the threshold for the pipeline it was published for.
pub const ALIKE_ENOUGH: f32 = 0.363;

/// **How long a held unknown face waits for a second one to agree.** Chosen, not measured: the
/// application looks every three seconds, so this is three looks — long enough for a person who has
/// just sat down, short enough that two unrelated passing faces a minute apart are not joined into one.
pub const NEWCOMER_WITHIN: f64 = 10.0;

/// Cosine similarity, −1 to 1. `None` on a length mismatch or a zero vector, because both mean the
/// comparison is meaningless rather than zero.
pub fn alike(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let ma: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if ma == 0.0 || mb == 0.0 { None } else { Some(dot / (ma * mb)) }
}

/// **The strongest match at or above the threshold**, as an index into `known` and how alike.
///
/// The strongest and not the first: walking the list and taking the first over the line makes the
/// answer depend on the order people were enrolled in.
pub fn best<'a>(known: impl Iterator<Item = &'a [f32]>, print: &[f32]) -> Option<(usize, f32)> {
    known.enumerate()
        .filter_map(|(i, p)| alike(print, p).map(|a| (i, a)))
        .filter(|(_, a)| *a >= ALIKE_ENOUGH)
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
}

/// The mean of two prints, each normalised first so neither outweighs the other by its length.
pub fn mean(a: &[f32], b: &[f32]) -> Vec<f32> {
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    let (na, nb) = (norm(a), norm(b));
    a.iter().zip(b).map(|(x, y)| x / na + y / nb).collect()
}

/// **A face nobody knows, held until a second one agrees.**
#[derive(Clone, Debug, Default)]
pub struct Newcomer {
    held: Option<(Vec<f32>, f64)>,
}

impl Newcomer {
    /// An unknown face was seen at `when` (seconds, any monotonic clock). **Returns the print to
    /// enrol and when the face was first seen** — the held frame's time, not this one's, since that
    /// is when argmind first saw them — once it agrees with the one held; `None` while it is still
    /// one frame's word.
    pub fn unknown(&mut self, print: Vec<f32>, when: f64) -> Option<(Vec<f32>, f64)> {
        match self.held.take() {
            Some((held, at)) if when - at <= NEWCOMER_WITHIN
                && alike(&held, &print).is_some_and(|a| a >= ALIKE_ENOUGH) => {
                Some((mean(&held, &print), at))
            }
            //: Nothing held, too old, or it does not agree: this frame becomes the one held. The
            //: one it replaces was a single frame nobody else vouched for.
            _ => {
                self.held = Some((print, when));
                None
            }
        }
    }

    /// A known face was seen. **Whatever was held was a bad frame of somebody known.**
    pub fn known(&mut self) {
        self.held = None;
    }

    pub fn holding(&self) -> bool {
        self.held.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(i: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; 128];
        v[i % 128] = 1.0;
        v
    }

    /// A print near `at(i)` — alike it, but not identical.
    fn near(i: usize) -> Vec<f32> {
        let mut v = at(i);
        v[(i + 1) % 128] = 0.3;
        v
    }

    #[test]
    fn one_frame_is_not_enough_to_become_somebody() {
        let mut n = Newcomer::default();
        assert_eq!(n.unknown(at(5), 0.0), None);
        assert!(n.holding());
    }

    #[test]
    fn two_frames_that_agree_become_one_person_at_their_mean() {
        let mut n = Newcomer::default();
        n.unknown(at(5), 0.0);
        let (kept, first) = n.unknown(near(5), 3.0).expect("two agreeing frames enrol");
        assert!(alike(&kept, &at(5)).unwrap() > 0.9);
        assert_eq!(first, 0.0, "first seen is the held frame, not the one that confirmed it");
        assert!(!n.holding(), "enrolling empties the hand");
    }

    #[test]
    fn frames_that_disagree_never_enrol() {
        //: **The measured shape of a face leaving the frame**: consecutive bad frames unlike each
        //: other. Nine rows came from this, one frame each.
        let mut n = Newcomer::default();
        for (i, t) in [(5, 0.0), (40, 3.0), (80, 6.0), (110, 9.0)] {
            assert_eq!(n.unknown(at(i), t), None, "frame {i} enrolled on its own word");
        }
    }

    #[test]
    fn a_known_face_clears_what_was_held() {
        let mut n = Newcomer::default();
        n.unknown(at(5), 0.0);
        n.known();
        assert!(!n.holding());
        //: So the next unknown frame starts over rather than pairing with the stray one.
        assert_eq!(n.unknown(near(5), 3.0), None);
    }

    #[test]
    fn two_agreeing_frames_too_far_apart_are_not_joined() {
        let mut n = Newcomer::default();
        n.unknown(at(5), 0.0);
        assert_eq!(n.unknown(near(5), NEWCOMER_WITHIN + 1.0), None);
    }

    #[test]
    fn the_strongest_match_wins_and_the_threshold_holds() {
        let known = [at(1), at(9), near(9)];
        let (i, _) = best(known.iter().map(|v| v.as_slice()), &at(9)).unwrap();
        assert_eq!(i, 1, "the exact match beats the near one");
        assert!(best(known.iter().map(|v| v.as_slice()), &at(60)).is_none());
    }
}
