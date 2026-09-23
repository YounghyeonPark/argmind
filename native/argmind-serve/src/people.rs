//! **Who argmind has seen before**, as 128 numbers and three dates each.
//!
//! The owner asked on 2026-09-23: *기존에 봤던건 등록하고 기억해둘수 있으면 해.* This is that, and
//! the first thing to be plain about is what it stores, because *remember a face* sounds like a
//! photograph and this is not one.
//!
//! ## What is written down, and what deliberately is not
//!
//! **No picture, ever.** A row holds SFace's 128 floats, a name the person typed or an empty
//! string, and when it was first and last seen. The numbers are a one-way summary — you cannot
//! reconstruct a face from them, and nothing here keeps a frame long enough to.
//!
//! **A cap, and it is the point rather than housekeeping.** [`MOST_REMEMBERED`] is 20. A camera
//! that is allowed to stay on will see delivery drivers, family and whoever walks behind the desk,
//! and an uncapped roster turns *remember who I have seen* into a register of visitors. Past the
//! cap argmind stops enrolling and says so, rather than evicting the oldest quietly — because
//! *quietly* is how a limit becomes a rotating log of everyone.
//!
//! **One print per person and not a growing set.** Several would recognise better; several is also
//! more of somebody's face on a disk, and nothing has measured that the first is not enough. If a
//! measurement ever says it is not, that is the moment to store more and to say so here.
//!
//! **The file is readable and deletable.** `%LOCALAPPDATA%\argmind\faces.json`, beside
//! `decisions.json`, one person per line — the same argument `decided.rs` makes: a decision kept
//! where the person cannot read it is one they have not really made.
//!
//! ## The threshold is somebody else's and this says so
//!
//! [`ALIKE_ENOUGH`] is **0.363**, which is OpenCV's own default for SFace cosine similarity. **It
//! has not been measured on this machine and it cannot be yet.** Deciding a threshold needs two
//! distributions — the same face across sessions, and *different* faces — and the second one needs
//! a second person. The first half is measurable today and the second is not, so the number is
//! inherited, labelled, and the harness that would replace it is described in
//! `docs/findings/seeing-a-room.md` rather than pretended into existence.
//!
//! **What that costs, concretely**: too low and two people merge into one row; too high and one
//! person becomes a new row every session, which the cap then fills. Both failures are visible in
//! the list, which is the other reason the list is something a person can look at.

use std::collections::BTreeMap;
use std::path::PathBuf;

/// **OpenCV's own number for SFace, not one measured here.** See the module note: the measurement
/// that would replace it needs a face this machine has never seen.
pub const ALIKE_ENOUGH: f32 = 0.363;

/// **How many people argmind will ever hold.** Chosen, not measured, and low on purpose — see the
/// module note. Past this nothing new is enrolled and the caller is told.
pub const MOST_REMEMBERED: usize = 20;

/// How long a print is, so a hand-edited file with the wrong shape is dropped rather than compared.
pub const PRINT_LENGTH: usize = 128;

/// One person argmind has seen.
#[derive(Clone, Debug, PartialEq)]
pub struct Person {
    /// Stable across renames, because the name is the part a person edits.
    pub key: String,
    /// **Empty until somebody types one.** argmind never invents a name for a face.
    pub name: String,
    /// Unix seconds.
    pub first: i64,
    pub last: i64,
    pub times: u64,
    /// **Whether argmind has already asked who this is.** Written down rather than kept in memory
    /// because the whole point is not asking twice, and a restart is exactly when a program forgets
    /// that it already did. Set by answering *or* by skipping: both are an answer to the question.
    pub asked: bool,
    /// SFace's 128 numbers. Private: a caller that wants to compare goes through [`People::saw`].
    print: Vec<f32>,
}

impl Person {
    /// What to show in a list when nobody has named them. **Not a guess at who they are** — a
    /// number, and the date argmind first saw them, which is the only true thing it knows.
    pub fn shown(&self) -> String {
        if self.name.trim().is_empty() {
            format!("이름 없음 ({})", self.key)
        } else {
            self.name.clone()
        }
    }
}

/// What happened when a face was offered.
#[derive(Clone, Debug, PartialEq)]
pub enum Saw {
    /// Recognised, with how alike it was.
    Known { key: String, alike: f32 },
    /// Not recognised, and a row was made.
    Fresh { key: String },
    /// Not recognised, and no row was made because the roster is full.
    Full,
    /// The print was the wrong shape — a hand-edited file or a different model.
    Wrong,
}

/// Everybody argmind has seen, in the order it first saw them.
#[derive(Clone, Debug, Default)]
pub struct People {
    pub who: Vec<Person>,
    /// Counts up so a key is never reused after a forget, which would silently merge two people.
    next: u64,
}

impl People {
    /// Read the file. **A file that cannot be read is nobody**, for the same reason a corrupt
    /// `decisions.json` is no decisions: the safe direction is to have forgotten.
    pub fn read() -> People {
        std::fs::read_to_string(at()).map(|t| parse(&t)).unwrap_or_default()
    }

    /// Write it back, making the directory if it is not there.
    pub fn write(&self) -> Result<(), String> {
        let path = at();
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        std::fs::write(&path, render(self)).map_err(|e| format!("{path:?}: {e}"))
    }

    /// Offer a face. Recognises it or enrols it, and says which.
    ///
    /// **Recognition is the strongest match above the threshold, not the first one.** Walking the
    /// list and taking the first over 0.363 makes the answer depend on the order people were
    /// enrolled in, which is the kind of bug that only shows up once there are three of them.
    pub fn saw(&mut self, print: Vec<f32>, when: i64) -> Saw {
        if print.len() != PRINT_LENGTH || print.iter().any(|f| !f.is_finite()) {
            return Saw::Wrong;
        }
        let best = self.who.iter().enumerate()
            .filter_map(|(i, p)| alike(&print, &p.print).map(|a| (i, a)))
            .filter(|(_, a)| *a >= ALIKE_ENOUGH)
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        if let Some((i, a)) = best {
            self.who[i].last = when;
            self.who[i].times += 1;
            //: **The print is not updated.** A running average would drift toward whatever lighting
            //: the last session had, and there is no measurement here saying which is better.
            return Saw::Known { key: self.who[i].key.clone(), alike: a };
        }
        if self.who.len() >= MOST_REMEMBERED {
            return Saw::Full;
        }
        self.next += 1;
        let key = format!("p{}", self.next);
        self.who.push(Person {
            key: key.clone(), name: String::new(), first: when, last: when, times: 1,
            asked: false, print,
        });
        Saw::Fresh { key }
    }

    /// **Who argmind could ask about**: a row with no name that has not been asked about yet.
    ///
    /// Returns the key rather than the person, because the caller is going to raise a question and
    /// then come back through [`People::mark_asked`] — holding a borrow across that is a borrow
    /// held across a page round trip.
    pub fn worth_asking(&self) -> Option<String> {
        self.who.iter().find(|p| p.name.trim().is_empty() && !p.asked).map(|p| p.key.clone())
    }

    /// Remember that the question was put. **Answering and skipping both call this** — the fact
    /// being recorded is *argmind has asked*, not *somebody replied*.
    pub fn mark_asked(&mut self, key: &str) -> bool {
        match self.who.iter_mut().find(|p| p.key == key) {
            Some(p) => { p.asked = true; true }
            None => false,
        }
    }

    /// Give somebody a name. `false` when there is nobody by that key.
    pub fn call(&mut self, key: &str, name: &str) -> bool {
        match self.who.iter_mut().find(|p| p.key == key) {
            //: Trimmed, and an empty name is allowed — it is how somebody takes a name back off.
            Some(p) => { p.name = name.trim().chars().take(40).collect(); true }
            None => false,
        }
    }

    /// Forget one. **The key is not reused afterwards**, so the next stranger is not silently
    /// filed as the person who was just deleted.
    pub fn forget(&mut self, key: &str) -> bool {
        let before = self.who.len();
        self.who.retain(|p| p.key != key);
        before != self.who.len()
    }

    /// Forget everybody, and **delete the file** rather than leaving an empty one. Somebody who
    /// presses this means *there is nothing of mine here*, and a file saying `{"people": []}` is a
    /// worse answer than no file.
    pub fn forget_everybody(&mut self) -> Result<(), String> {
        self.who.clear();
        match std::fs::remove_file(at()) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e.to_string()),
        }
    }
}

/// Cosine similarity, −1 to 1. `None` on a length mismatch or a zero vector, because both mean the
/// comparison is meaningless rather than zero.
fn alike(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let ma: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if ma == 0.0 || mb == 0.0 { None } else { Some(dot / (ma * mb)) }
}

/// `%LOCALAPPDATA%\argmind\faces.json`, beside the decisions and the models.
pub fn at() -> PathBuf {
    let mut path = std::env::var("LOCALAPPDATA").map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    path.push("argmind");
    path.push("faces.json");
    path
}

/// **One person per line**, hand-written for the reason `decided.rs` gives: this crate takes no
/// third-party dependency for a file it also wants somebody to be able to open and read.
pub fn render(people: &People) -> String {
    if people.who.is_empty() {
        return "{\"people\": []}\n".to_string();
    }
    let rows: Vec<String> = people.who.iter().map(|p| {
        let print: Vec<String> = p.print.iter().map(|f| format!("{f:.5}")).collect();
        format!("  {{\"key\": \"{}\", \"name\": \"{}\", \"first\": {}, \"last\": {}, \
\"times\": {}, \"asked\": {}, \"print\": [{}]}}",
                p.key, escape(&p.name), p.first, p.last, p.times, p.asked, print.join(","))
    }).collect();
    format!("{{\"people\": [\n{}\n]}}\n", rows.join(",\n"))
}

/// Read it back. **Anything that does not parse is dropped rather than failing the file** — one
/// mangled row should not lose the other nineteen.
pub fn parse(text: &str) -> People {
    let mut people = People::default();
    let mut used: BTreeMap<String, ()> = BTreeMap::new();
    for line in text.lines() {
        let line = line.trim().trim_end_matches(',');
        if !line.starts_with('{') || !line.contains("\"print\"") {
            continue;
        }
        let Some(key) = field(line, "key") else { continue };
        //: A duplicated key in a hand-edited file would make `forget` ambiguous, so the second one
        //: is dropped rather than renamed — renaming would be argmind inventing an identity.
        if used.insert(key.clone(), ()).is_some() {
            continue;
        }
        let print = numbers(line);
        if print.len() != PRINT_LENGTH {
            continue;
        }
        if let Some(n) = key.strip_prefix('p').and_then(|d| d.parse::<u64>().ok()) {
            people.next = people.next.max(n);
        }
        people.who.push(Person {
            key,
            name: field(line, "name").unwrap_or_default(),
            first: whole(line, "first").unwrap_or(0),
            last: whole(line, "last").unwrap_or(0),
            times: whole(line, "times").unwrap_or(0).max(0) as u64,
            //: **Absent reads as not asked**, which re-asks once after an upgrade from a file
            //: written before this field existed. Once, for people who have no name — which is the
            //: right direction: the alternative silently loses the question for everybody already
            //: in the roster.
            asked: line.contains("\"asked\": true"),
            print,
        });
    }
    people
}

/// A `"name": "value"` string field.
fn field(line: &str, name: &str) -> Option<String> {
    let at = line.find(&format!("\"{name}\":"))? + name.len() + 3;
    let rest = line[at..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let mut out = String::new();
    let mut chars = rest.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => out.push(chars.next().unwrap_or('\\')),
            _ => out.push(c),
        }
    }
    None
}

/// A `"name": 123` integer field.
fn whole(line: &str, name: &str) -> Option<i64> {
    let at = line.find(&format!("\"{name}\":"))? + name.len() + 3;
    let rest = line[at..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '-').unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// The one array on the line.
fn numbers(line: &str) -> Vec<f32> {
    let Some(open) = line.find('[') else { return Vec::new() };
    let Some(close) = line[open..].find(']') else { return Vec::new() };
    line[open + 1..open + close].split(',')
        .filter_map(|n| n.trim().parse::<f32>().ok())
        .filter(|f| f.is_finite())
        .collect()
}

/// Enough for a name in a one-line JSON object.
fn escape(s: &str) -> String {
    s.chars().flat_map(|c| match c {
        '"' => vec!['\\', '"'],
        '\\' => vec!['\\', '\\'],
        '\n' | '\r' | '\t' => vec![' '],
        _ => vec![c],
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A print that is `PRINT_LENGTH` long and points somewhere particular.
    ///
    /// **The first version was `sin(i * 0.07 + seed)` and it collided**, because a sine is periodic
    /// and a seed stepping by 5 wraps past 2π — the roster test caught it at fixture 4 only because
    /// it asserted its own fixtures were distinct. This one puts its weight at one index, so two
    /// different seeds are orthogonal by construction and the tests below are testing the roster
    /// rather than the generator.
    fn print_of(seed: f32) -> Vec<f32> {
        let at = (seed as usize) % PRINT_LENGTH;
        let mut v = vec![0.0f32; PRINT_LENGTH];
        v[at] = 1.0;
        v[(at + 1) % PRINT_LENGTH] += 0.1;
        v
    }

    #[test]
    fn a_new_face_is_enrolled_and_the_same_face_is_recognised() {
        let mut people = People::default();
        let me = print_of(1.0);
        let Saw::Fresh { key } = people.saw(me.clone(), 100) else { panic!("first sight is fresh") };
        assert_eq!(people.who.len(), 1);
        match people.saw(me, 200) {
            Saw::Known { key: again, alike } => {
                assert_eq!(again, key);
                assert!(alike > 0.99, "the identical print should be all but 1.0, was {alike}");
            }
            other => panic!("the same face came back as {other:?}"),
        }
        assert_eq!(people.who.len(), 1, "recognising must not enrol");
        assert_eq!(people.who[0].times, 2);
        assert_eq!(people.who[0].last, 200);
        assert_eq!(people.who[0].first, 100, "the first sighting is not overwritten");
    }

    #[test]
    fn a_different_face_gets_its_own_row() {
        let mut people = People::default();
        people.saw(print_of(1.0), 100);
        //: Far enough apart to be under the threshold; the test asserts that rather than assuming.
        let other = print_of(40.0);
        let first = people.who[0].print.clone();
        assert!(alike(&other, &first).unwrap() < ALIKE_ENOUGH,
                "this fixture stopped being two people; the test below would pass for the wrong \
                 reason");
        assert!(matches!(people.saw(other, 110), Saw::Fresh { .. }));
        assert_eq!(people.who.len(), 2);
    }

    #[test]
    fn the_strongest_match_wins_rather_than_the_first() {
        let mut people = People::default();
        //: Two rows, and a probe nearer the second. Taking the first over the line would return the
        //: wrong person and the answer would depend on enrolment order.
        people.saw(print_of(1.0), 10);
        people.saw(print_of(40.0), 20);
        let mut near_second = print_of(40.0);
        near_second[0] += 0.02;
        match people.saw(near_second, 30) {
            Saw::Known { key, .. } => assert_eq!(key, "p2"),
            other => panic!("expected the second person, got {other:?}"),
        }
    }

    #[test]
    fn the_roster_fills_rather_than_evicting_quietly() {
        let mut people = People::default();
        for i in 0..MOST_REMEMBERED {
            assert!(matches!(people.saw(print_of(i as f32 * 5.0 + 1.0), 10), Saw::Fresh { .. }),
                    "fixture {i} collided with an earlier one");
        }
        assert_eq!(people.saw(print_of(999.0), 10), Saw::Full);
        assert_eq!(people.who.len(), MOST_REMEMBERED, "nobody was evicted to make room");
    }

    #[test]
    fn a_print_of_the_wrong_shape_is_refused_rather_than_compared() {
        let mut people = People::default();
        assert_eq!(people.saw(vec![0.5; 64], 10), Saw::Wrong);
        assert_eq!(people.saw(vec![f32::NAN; PRINT_LENGTH], 10), Saw::Wrong);
        assert!(people.who.is_empty(), "a refused print must not make a row");
    }

    #[test]
    fn forgetting_does_not_hand_the_key_to_the_next_stranger() {
        let mut people = People::default();
        people.saw(print_of(1.0), 10);
        assert!(people.forget("p1"));
        assert!(!people.forget("p1"), "forgetting twice is not a second success");
        let Saw::Fresh { key } = people.saw(print_of(40.0), 20) else { panic!("expected a row") };
        assert_ne!(key, "p1", "reusing p1 would file a stranger as the person just deleted");
    }

    #[test]
    fn a_person_with_no_name_is_worth_asking_about_exactly_once() {
        let mut people = People::default();
        people.saw(print_of(1.0), 10);
        assert_eq!(people.worth_asking().as_deref(), Some("p1"));
        assert!(people.mark_asked("p1"));
        assert_eq!(people.worth_asking(), None, "asking twice is the failure this field exists for");
        assert!(!people.mark_asked("nobody"));
    }

    #[test]
    fn somebody_already_named_is_never_asked_about() {
        let mut people = People::default();
        people.saw(print_of(1.0), 10);
        people.call("p1", "영현");
        assert_eq!(people.worth_asking(), None);
    }

    #[test]
    fn having_asked_survives_the_file() {
        let mut people = People::default();
        people.saw(print_of(1.0), 10);
        people.mark_asked("p1");
        let back = parse(&render(&people));
        assert!(back.who[0].asked);
        assert_eq!(back.worth_asking(), None,
                   "a restart is exactly when a program forgets it already asked");
    }

    #[test]
    fn a_file_from_before_this_field_asks_once_more_rather_than_never() {
        //: The upgrade case, written out by hand as the old version wrote it. Re-asking once is the
        //: safe direction; silently losing the question for everybody already there is not.
        let mut people = People::default();
        people.saw(print_of(1.0), 10);
        let old = render(&people).replace(", \"asked\": false", "");
        assert!(!old.contains("asked"));
        assert_eq!(parse(&old).worth_asking().as_deref(), Some("p1"));
    }

    #[test]
    fn a_name_is_the_persons_and_argmind_never_invents_one() {
        let mut people = People::default();
        people.saw(print_of(1.0), 10);
        assert_eq!(people.who[0].name, "");
        assert!(people.who[0].shown().starts_with("이름 없음"));
        assert!(people.call("p1", "  영현  "));
        assert_eq!(people.who[0].name, "영현");
        assert_eq!(people.who[0].shown(), "영현");
        assert!(people.call("p1", ""), "a name can be taken back off");
        assert_eq!(people.who[0].name, "");
        assert!(!people.call("nobody", "x"));
    }

    #[test]
    fn a_roster_survives_the_round_trip() {
        let mut people = People::default();
        people.saw(print_of(1.0), 100);
        people.saw(print_of(40.0), 110);
        people.call("p2", "누군가 \"따옴표\" 있는");
        let back = parse(&render(&people));
        assert_eq!(back.who.len(), 2);
        assert_eq!(back.who[1].name, "누군가 \"따옴표\" 있는");
        assert_eq!(back.who[0].first, 100);
        //: **And recognition still works after a round trip**, which is the half that a
        //: format test alone would not catch: five decimal places must be enough.
        match back.clone().saw(people.who[0].print.clone(), 200) {
            Saw::Known { key, .. } => assert_eq!(key, "p1"),
            other => panic!("a saved face stopped being recognisable: {other:?}"),
        }
    }

    #[test]
    fn the_counter_survives_a_round_trip_too() {
        let mut people = People::default();
        people.saw(print_of(1.0), 10);
        people.saw(print_of(40.0), 20);
        people.forget("p1");
        let mut back = parse(&render(&people));
        let Saw::Fresh { key } = back.saw(print_of(80.0), 30) else { panic!("expected a row") };
        assert_eq!(key, "p3", "after a reload the next key must still be past every key ever used");
    }

    #[test]
    fn one_mangled_row_does_not_lose_the_others() {
        let mut people = People::default();
        people.saw(print_of(1.0), 10);
        people.saw(print_of(40.0), 20);
        let text = render(&people);
        let broken: Vec<&str> = text.lines().collect();
        //: The second row truncated mid-array, which is what a half-written file looks like.
        let mangled = format!("{}\n{}\n{}\n{}", broken[0], broken[1],
                              "  {\"key\": \"p2\", \"print\": [0.1,0.2]}", "]}");
        let back = parse(&mangled);
        assert_eq!(back.who.len(), 1);
        assert_eq!(back.who[0].key, "p1");
    }

    #[test]
    fn a_duplicated_key_is_dropped_rather_than_renamed() {
        let mut people = People::default();
        people.saw(print_of(1.0), 10);
        let one = render(&people);
        let row = one.lines().nth(1).unwrap().trim_end_matches(',');
        let doubled = format!("{{\"people\": [\n{row},\n{row}\n]}}");
        let back = parse(&doubled);
        assert_eq!(back.who.len(), 1, "renaming one of them would be argmind inventing an identity");
    }

    #[test]
    fn nobody_renders_as_something_a_person_can_read() {
        assert_eq!(render(&People::default()), "{\"people\": []}\n");
        assert!(parse("{\"people\": []}\n").who.is_empty());
    }

    #[test]
    fn no_picture_is_anywhere_in_what_is_written() {
        //: **The claim the module note makes, checked.** A row is numbers and dates; if a field
        //: ever carries pixels this fails, which is the only way that promise stays true.
        let mut people = People::default();
        people.saw(print_of(1.0), 10);
        let text = render(&people);
        for word in ["png", "jpeg", "rgb", "base64", "image", "pixels", "data:"] {
            assert!(!text.to_lowercase().contains(word),
                    "faces.json grew a {word} field — the note says it stores no picture");
        }
    }
}
