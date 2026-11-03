//! **확인 없이 실행 — the one permission that changes what argmind may do.**
//!
//! Every other decidable key decides what argmind may *know* or *say*. This one decides whether
//! `Route::Confirm` is still the only way anything on the screen gets pressed, and it was
//! deliberately absent — *a click is not on it*, `decided.rs` used to say — until the owner asked
//! for it on 2026-09-22, having watched three buttons pile up in the transcript.
//!
//! **What the button was doing, and which half survives.** It did two jobs: a person checked that
//! the model had chosen the right widget, and a person read the second opinion — *그 지점에는 link
//! Cargo.lock* — when the hit test disagreed with the resolution. The first is what is being given
//! up, knowingly, against `choosing` at 28 of 37 and `talking` producing a spurious action on 9% of
//! conversational turns. The second is not: with nobody reading it, the second opinion becomes the
//! gate it is not allowed to be while somebody is.
//!
//! It stays a note when a person is going to press the button, because the hit test disagrees with
//! a *correct* pixel about three times in ten and gating on it would refuse those.
//!
//! Nothing here presses anything or writes `decisions.json`.

use argmind_serve::decided::{is_decidable, what, DECIDABLE};

/// The whole of `apply`-like source reading: one function of `app.rs`, from its `fn` line to the
/// next one at the same indent.
fn function(name: &str) -> String {
    let source = include_str!("../src/app.rs");
    let from = source.find(&format!("fn {name}("))
        .unwrap_or_else(|| panic!("{name} is gone from app.rs"));
    let rest = &source[from..];
    let end = rest[1..].find("\nfn ").map(|i| i + 1).unwrap_or(rest.len());
    rest[..end].to_string()
}

#[test]
fn it_is_a_permission_like_the_others() {
    assert!(is_decidable("acting"));
    assert_ne!(what("acting"), "알 수 없는 권한");
    //: **After everything that only changes what argmind knows or says**, which is the ordering in
    //: `DECIDABLE` saying what kind of thing this is: needs, then the switches, then the two that
    //: cost something outside the program. `camera` joined it below on the same day.
    let at = |k| DECIDABLE.iter().position(|x| *x == k).unwrap_or(usize::MAX);
    for quieter in ["ollama", "model", "whisper", "korean-voice", "vulkan", "speaker",
                    "notice", "microphone", "aloud"] {
        assert!(at("acting") > at(quieter), "acting is listed before {quieter}");
    }
}

/// **The window says what it costs, in numbers rather than in adjectives.**
///
/// A dialog reading *act automatically?* is answered yes by everybody. The measured facts about
/// this model are the whole of the argument against saying yes, so they are in the sentence: how
/// often it decides correctly, how often it acts when nobody asked, and the one recorded case of
/// the right name on the wrong button.
#[test]
fn the_window_says_how_often_the_model_is_wrong() {
    let said = argmind_serve::app::plan_for("acting");
    for number in ["28/37", "9%"] {
        assert!(said.contains(number), "the plan does not say {number}:\n{said}");
    }
    //: **And what it does *not* cover**, which is the part somebody would otherwise assume the
    //: worst about: a thought argmind had by itself still comes out as speech.
    assert!(said.contains("말로만"), "{said}");
    assert!(said.trim_end().ends_with("까요?"), "{said}");
    assert!(said.lines().filter(|l| !l.trim().is_empty()).count() >= 4, "{said}");
}

// ---- the route ---------------------------------------------------------------------------------

fn asked(path: &str) -> argmind_serve::Route {
    argmind_serve::route(&argmind_serve::Request {
        method: "POST".into(), path: path.into(), body: String::new(), token: None,
    })
}

#[test]
fn it_is_two_routes_and_not_a_toggle() {
    assert_eq!(asked("/acting/on"), argmind_serve::Route::Acting(true));
    assert_eq!(asked("/acting/off"), argmind_serve::Route::Acting(false));
    //: **No `/acting/toggle`**, for the same reason `/listen` is two routes: a switch whose meaning
    //: depends on a state the page believes it knows is a switch that gets out of step.
    for bad in ["/acting", "/acting/", "/acting/yes", "/acting/on/off"] {
        assert_eq!(asked(bad), argmind_serve::Route::Unknown, "{bad}");
    }
}

// ---- the two things that must stay true ----------------------------------------------------------

/// **Automatic is still the same two steps a person's button is.**
///
/// Going straight from `propose` to `act` would be a second path into the only thing in this
/// program that presses anything, and `Proposals` is what makes a stale token unable to confirm a
/// later action. So the offer is made, published to the page, and then taken — `confirm` on the
/// token that was just handed out.
#[test]
fn an_automatic_run_goes_through_the_same_offer_a_person_would_press() {
    let propose = function("propose");
    let offer = propose.find(".offer(").expect("propose no longer offers");
    let confirm = propose.find(".confirm(")
        .expect("propose does not confirm; it must reach `act` the way the button does");
    assert!(offer < confirm, "propose confirms before it offers");
    //: And it tells the page first, so a row appears and then resolves rather than never existing.
    let told = propose.find("\"propose\"").expect("the page is no longer told");
    assert!(told < confirm, "propose acts before the page hears about it");
    //: **`act` is not called from here directly on any other line.** One call, inside the guard.
    assert_eq!(propose.matches("act(desk, out, p)").count(), 1, "{propose}");
}

/// **Both halves of the guard, and the doubt one is the reason this is a test.**
///
/// The standing answer is the obvious half. The other is that a proposal carrying a second opinion
/// — the hit test naming a different widget at the resolved point — still waits for a person. That
/// is the half a refactor would drop as redundant, because it looks like belt and braces and is
/// the only thing left when nobody is reading the row.
#[test]
fn doubt_still_asks_even_when_acting_is_allowed() {
    let propose = function("propose");
    let guard = propose.lines()
        .find(|l| l.contains("crate::decided::standing(if its_own_idea"))
        .unwrap_or_else(|| panic!("the standing answer is not consulted:\n{propose}"));
    //: **Since 2026-11-03 only for argmind's own idea**: what a person asked for runs at the
    //: owner's word, and a doubtful idea of argmind's own is let go rather than asked about.
    assert!(guard.contains("(!its_own_idea || (!doubt && !unsure))"),
            "argmind's own idea is no longer held to the second opinion and its sureness: {guard}");
    assert!(guard.contains("Standing::Allowed"), "{guard}");
    //: **And a 항상 물어봄 rule** — since 2026-10-01, the third thing the automatic path yields to.
    assert!(guard.contains("!must_confirm"), "a rule saying always ask is ignored: {guard}");

    //: **The rule, or a reply that itself asked** — since 2026-11-03, when *찾아볼까요?* went out
    //: beside a click on a Git button and 확인 없이 실행 pressed it. Either half dropped fails here.
    assert!(propose.contains(
        "let must_confirm = rule == Some(argmind_core::rule::Then::Confirm) || asked;"),
            "must_confirm no longer means the rule said so, or the reply asked:
{propose}");
    //: **`doubt` is the hint being non-empty and nothing else.** A refactor that made it mean
    //: something looser — *the widget was found*, say — would pass the line above and gate on
    //: nothing.
    assert!(propose.contains("let doubt = !hint.is_empty();"),
            "doubt is no longer *the hit test disagreed*:\n{propose}");
}

/// **A thought argmind had by itself waits for a person, and 확인 없이 실행 does not cover it.**
///
/// **This test said the opposite yesterday and it was wrong about the design, not about the code.**
/// It asserted that nothing in the application turns `may_act_unprompted` on — which was true, and
/// which is why the 실행 button beside a self-started proposal did nothing at all. `session.rs` says
/// what that flag is for in as many words: *an interface with a confirm button turns this on and
/// gates with the button instead, which is stronger than the flag.* This application has the
/// button. It had not turned it on. So the button was drawn, could be pressed, and silently did
/// nothing — and until the day before this it reported `ran` while doing it.
///
/// So the rule being kept is not *argmind may never act on its own idea*. It is:
///
/// * **a person pressing 실행 is a permission**, for that one action, whoever thought of it;
/// * **a standing 확인 없이 실행 is not** — it was given in advance to *the thing I just asked
///   for*, and extending it to what argmind thought of by itself would be reading one permission
///   as another. That is the distinction the whole `because_asked` field exists to carry.
#[test]
fn argmind_own_idea_needs_a_person_and_not_a_standing_yes() {
    let core = include_str!("../../argmind-core/src/session.rs");
    assert!(core.contains("effect.unprompted() && !self.may_act_unprompted"),
            "the core no longer downgrades an unprompted effect at all");

    let source = include_str!("../src/app.rs");
    //: **The button's half.** Turned on once, where the `Desk` is built, and nowhere else — a
    //: second place that set it would be a second answer to who the permission came from.
    assert_eq!(source.matches("may_act_unprompted = true").count(), 1,
               "the flag is set in more than one place");

    //: **The switch's half, and the one a refactor would drop.** The automatic path has to refuse
    //: an effect argmind proposed to itself, and it has to do it by asking whether the effect was
    //: asked for rather than by any looser test.
    let propose = function("propose");
    assert!(propose.contains("let its_own_idea = effect.unprompted();"),
            "propose no longer reads whose idea it was:
{propose}");
    let guard = propose.lines()
        .find(|l| l.contains("crate::decided::standing(if its_own_idea"))
        .unwrap_or_else(|| panic!("the standing answer is not consulted:
{propose}"));
    //: **Changed 2026-10-21, at the owner's choice: argmind's own idea may run without a person —
    //: but only on its own standing answer.** The half a refactor would drop is the same as before:
    //: 확인 없이 실행 alone must never carry out what argmind began itself.
    assert!(guard.contains(r#"if its_own_idea { "initiative" } else { "acting" }"#),
            "argmind's own idea is no longer gated by its own key: {guard}");
}

/// **A window change argmind did nothing with is not a turn of the conversation.** `noticed` opens
/// a turn and the prompt keeps twelve; recorded before the check that it came to anything, silent
/// notices pushed the person's own words out of memory — the owner, 2026-10-21. Held by reading
/// `notice`: the line that records it comes after the line that drops a notice with nothing to do.
#[test]
fn a_silent_notice_does_not_take_a_turn_of_memory() {
    let notice = function("notice");
    let silent = notice.find("if intent.effects.is_empty()").expect("the silent case is gone");
    let kept = notice.find("history.push((\"noticed\"").expect("notices are no longer recorded");
    assert!(kept > silent, "a notice is recorded before it is known to come to anything");
    assert_eq!(notice.matches("history.push((\"noticed\"").count(), 1);
}
