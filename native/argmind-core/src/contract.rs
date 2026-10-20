//! What an environment must provide — the Rust half of `argmind/core/contract.py`.
//!
//! Traits, for the reason they are protocols there: **the core calls them, and nothing else about
//! the platform reaches it.** A trait here that no core module calls would be the same defect
//! `contract.py` refuses for `attending_title()` and `voice.py` refused for `Ears` — which is why
//! hearing and speaking are not in this file.

/// What one attempted action did.
///
/// **In the core because two environments now have to say the same things**, and a second copy of
/// this enum is how they would come to mean different ones — the argument `fix.rs` makes about
/// `apply` and `argmind-window` makes about the tray. It named no platform when it lived in
/// `argmind-env-windows/src/hands.rs` and it names none here: a process id is not a Windows idea,
/// and `how` is a string the environment chooses.
///
/// **It is not part of the `Hands` trait.** `click` returns nothing; this is the record a caller
/// may read afterwards to tell a refusal apart from a press that landed and did nothing, which is
/// a distinction no boolean can carry.
#[derive(Debug, Clone, PartialEq)]
pub enum Did {
    /// Sent. `false` for `live` means the hands were dry and nothing left the process.
    Sent { live: bool },
    /// The point belonged to somebody else. **Not an error and not a success** — the caller has to
    /// be able to tell this apart from a click that landed and did nothing.
    Elsewhere { owner: Option<u32> },
    /// Operated through the accessibility layer — **no pixel was aimed at and the cursor did not
    /// move.** `how` is which mechanism, in the environment's own words.
    Operated { how: &'static str, live: bool },
    /// Malformed: an unknown button or an unnamed key.
    Refused(String),
}

/// A thing on screen: a name, a role, a rectangle, and the window it belongs to.
///
/// `PartialEq` is name, role and rect — **deliberately not the window**, because `Session::confirm`
/// leans on exactly that: the same control in a different window still compares equal, and the
/// window is checked separately.
#[derive(Clone, Debug)]
pub struct Widget {
    pub name: String,
    pub role: String,
    pub rect: (i64, i64, i64, i64),
    pub window: String,
}

impl PartialEq for Widget {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.role == other.role && self.rect == other.rect
    }
}

impl Widget {
    pub fn new(name: &str, role: &str, rect: (i64, i64, i64, i64), window: &str) -> Self {
        Widget { name: name.into(), role: role.into(), rect, window: window.into() }
    }

    /// **The one place a name becomes a coordinate.** Two such places would be two chances to
    /// disagree about what "the middle of this thing" means. Floor division, as in the Python —
    /// Rust's `/` truncates toward zero and would differ on a negative rectangle.
    pub fn centre(&self) -> (i64, i64) {
        let (x, y, w, h) = self.rect;
        (x + w.div_euclid(2), y + h.div_euclid(2))
    }
}

/// Reading the screen. **Exact, and that is the contract** — 41–266 ms measured, against 24–82 s
/// for a vision model that on a dense screen never answered at all.
pub trait Eyes {
    fn foreground(&mut self) -> Vec<Widget>;
    /// The same read, with what it cost. Separate because the cost is a measurement an interface
    /// shows and the core does not act on.
    fn timed_foreground(&mut self) -> (Vec<Widget>, f64);
}

/// Touching the screen.
pub trait Hands {
    /// Whether anything actually moves. A dry run has nothing to confirm, which is why
    /// `Session::confirm` answers `None` rather than `false`.
    fn live(&self) -> bool;
    fn click(&mut self, widget: &Widget);
    fn type_text(&mut self, text: &str);
    fn press(&mut self, key: &str);
    /// The keys this environment has, or `None` when it publishes none — in which case
    /// `knows_key` answers `true`. Off the protocol proper in the Python for a stated reason: the
    /// whole vocabulary is the wrong quantity for the one line that consults it, and a phone
    /// exposing hundreds of names would be a contract paid for by everyone and used once.
    fn keys(&self) -> Option<Vec<String>> {
        None
    }
    /// **Would text typed now land in a password field?** Asked before every `type`, and a `true`
    /// is a refusal: argmind sends what it types to nobody, but it is a program that writes down
    /// what it did and shows a model its history, and a password has no business in either. The
    /// person types it themselves — that is the hand-off.
    ///
    /// **`false` by default**, which means an environment that cannot tell does not stop typing.
    /// That is the weaker direction and it is chosen: refusing every keystroke on a platform with
    /// no way to ask would make typing a feature of Windows alone. Windows answers from UI
    /// Automation's `IsPassword`; AT-SPI has a `password text` role and nobody has wired it yet.
    fn into_a_secret(&mut self) -> bool {
        false
    }
    /// **Is there somewhere for text to go?** Asked before every `type`: the focus is a field that
    /// takes text — `Some(true)` — or it is not — `Some(false)`, and the text is refused. Found in
    /// the owner's session on 2026-10-21, where nothing asked this and a sequence typed and pressed
    /// Enter with nothing checking where. `None` is *cannot tell*, and does not stop typing — the
    /// same direction as [`Hands::into_a_secret`], for the same reason.
    fn typing_lands(&mut self) -> Option<bool> {
        None
    }
    /// **Did the last action land?** `Some(false)` when the environment refused it — covered,
    /// gone, malformed — so `perform` can stop the steps after it rather than carry on into a
    /// screen the first step did not produce. `None` when it cannot say, and then nothing changes.
    fn landed(&self) -> Option<bool> {
        None
    }
}

/// Asking the model. **It is handed text and returns text** — never an image, never a coordinate.
pub trait Speaks {
    fn ask(&mut self, prompt: &str) -> String;
}
