//! Hearing and speaking — **the seam, and deliberately not part of the core.**
//!
//! `Eyes` and `Hands` are in the core's contract because the core calls them. Nothing in the core
//! could call an `Ears` or a `Mouth`: a turn goes in as a `String` and comes out as one, and both
//! ends of a voice channel attach outside that. Being I/O is not the criterion; being called by the
//! core is. So: **`Eyes` and `Hands` are how the agent touches the machine's other applications; a
//! microphone and a speaker are how it touches the person.**
//!
//! One consequence is load-bearing and is not a style choice. **A spoken *"응"* cannot run an
//! action.** The deterministic side says WHERE, and WHETHER: a button press is an event Windows
//! attributes to a physical device, a transcript is a recogniser's opinion about a waveform, and
//! putting one in the other's slot is the same failure as handing a robot's detector to `Eyes`.
//! Nothing in this crate reaches a session.

/// One finished utterance, as text.
///
/// **There is no confidence field.** The recognisers in sight publish no calibrated one, and an
/// always-`None` field invites a threshold nobody can set. `seconds` earns its place instead: it is
/// the one cheap signal a transcript was *invented*, because near-silence does not make Whisper
/// silent — it makes it say `Thank you.`
pub struct Heard {
    pub text: String,
    pub seconds: f64,
    pub at: f64,
}

impl Heard {
    /// Characters of transcript per second of audio; infinite if the audio had no length.
    ///
    /// **The one place this ratio is computed**, for the reason `Widget::centre` is the one place a
    /// name becomes a coordinate. **No threshold is named here**: the only pace measured on this
    /// machine is a synthesiser's — 5.69 characters a second for Korean — which is the wrong
    /// denominator for a person, and a guessed cut-off that silently drops real turns is worse than
    /// none.
    pub fn pace(&self) -> f64 {
        if self.seconds <= 0.0 { f64::INFINITY } else { self.text.chars().count() as f64 / self.seconds }
    }
}

/// A microphone that produces finished utterances.
///
/// Shaped so that push-to-talk today and voice-activity detection later are the *same* trait: an
/// implementation that decides for itself when speech begins ignores `hold` and returns utterances
/// from `heard`. The thing most likely to change is when capture starts, and it is the thing not
/// encoded.
pub trait Listener {
    /// Capture while `down`. Push-to-talk is the design and not a placeholder: the key is the floor
    /// an open microphone has no equivalent of, it makes self-hearing impossible because the
    /// microphone is shut while the agent talks, and it is the only thing that tells a turn
    /// addressed to the agent from a phone call in the same room.
    fn hold(&mut self, down: bool);
    /// A finished utterance, or nothing — and it **must not block**, because the caller is the
    /// thread already waiting on the model. `None` is also the answer for a transcript the
    /// implementation judged a fabrication: a rejection, not an error.
    fn heard(&mut self) -> Option<Heard>;
    fn release(&mut self);
}

/// A voice. Says a string and can be cut off mid-word.
pub trait Speaker {
    /// Begin speaking. **Returns at once, and that is forced by barge-in**: a blocking version
    /// cannot be interrupted, and a median reply is ~3.1 s of Korean speech against a 0.61 s model
    /// call — long enough that being unable to stop it is the difference between a conversation and
    /// a lecture.
    fn say(&mut self, text: &str);
    /// Cut the current utterance off mid-word and do not resume it. **This is what barge-in is.**
    fn stop(&mut self);
    fn speaking(&self) -> bool;
    fn release(&mut self);
}
