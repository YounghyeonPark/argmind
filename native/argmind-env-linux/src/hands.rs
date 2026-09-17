//! Operating a widget **through the accessibility layer rather than through a pixel.**
//!
//! This is the choice the Windows environment already makes and prefers. `patterns.rs` there uses a
//! UI Automation pattern where there is one and records `Did::Operated { how, live }` — *no pixel
//! was aimed at and the cursor did not move* — which the pointer measurement found is worth 3 of 3:
//! a click drags the cursor to the widget, a pattern leaves it where the person left it.
//!
//! AT-SPI has the same idea in `org.a11y.atspi.Action`, and the D-Bus connection the eyes already
//! need is all it takes. **Measured before it was written**: a GTK button offers one action, named
//! `Click` and described as *Clicks the button*, and `DoAction` answered `true`.
//!
//! ## What is missing, and it is not laziness
//!
//! **Typing is not here, because AT-SPI no longer synthesises input.** `DeviceEventController` used
//! to generate key events; this at-spi2-core does not expose it — checked by introspection rather
//! than assumed. So `type_text` and `press` need XTEST on X11 or `xdg-desktop-portal` on Wayland,
//! and those are the permission question that has not been settled against a real desktop. Until
//! then they refuse, and say which of the two it would take.
//!
//! **And there is no `Reach` here.** The Windows hands can be bound to one process so a measurement
//! harness cannot touch what it did not start. Nothing here can do that yet, so it is unrestricted
//! only — which is what the application uses in any case, and why the confirm button exists.

use argmind_core::contract::{Did, Widget};

use crate::atspi::{Bus, Node, DEEPEST};

/// Said the same way in both places, because it is one fact.
const NO_SYNTHESIS: &str =
    "이 운영체제에서는 아직 키를 보낼 수 없습니다 (X11은 XTEST, Wayland는 포털 동의가 필요합니다)";

pub struct Hands {
    bus: Option<Bus>,
    /// What each attempt did. **Read by the application** to tell a refusal from a press that
    /// landed and did nothing.
    pub done: Vec<Did>,
    armed: bool,
}

impl Default for Hands {
    fn default() -> Self {
        Hands::unrestricted()
    }
}

impl Hands {
    /// Dry: it will find the widget and not operate it.
    pub fn unrestricted() -> Self {
        //: **Failing to open the bus is not failing to construct.** Hands that cannot reach the
        //: accessibility bus still answer every call, with a refusal that says so.
        Hands { bus: Bus::open().ok(), done: Vec::new(), armed: false }
    }

    /// Live.
    pub fn arm(mut self) -> Self {
        self.armed = true;
        self
    }

    /// The node this widget is, **right now**.
    ///
    /// Re-resolved rather than remembered, which is the invariant the whole design rests on: a
    /// listed widget is gone 13 times in 14 over the second a call takes, so an effect carries a
    /// name and a role and is looked up again immediately before acting.
    ///
    /// Name, role *and* rectangle, because name and role together are still not a key — Character
    /// Map's advanced view has more than one button called `Open`, and that cost a goal on Windows.
    /// The rectangle separates copies; it is also the thing most likely to have moved, so all three
    /// are tried first and name-and-role is the fallback.
    fn find(&self, want: &Widget) -> Option<Node> {
        let bus = self.bus.as_ref()?;
        let frame = bus.active_frame()?;
        let mut loose = None;
        let mut layer = vec![frame];
        for _ in 0..DEEPEST {
            if layer.is_empty() {
                break;
            }
            let mut next = Vec::new();
            for node in &layer {
                if bus.name(node) == want.name && bus.role(node) == want.role {
                    let here = bus.extents(node)
                        .map(|(x, y, w, h)| (x as i64, y as i64, w as i64, h as i64));
                    if here == Some(want.rect) {
                        return Some(node.clone());
                    }
                    //: **The first same-named one, kept but not preferred.** Among copies it
                    //: settles nothing, and this is the same tie-break that is wrong on Windows —
                    //: recorded there, and no better here.
                    if loose.is_none() {
                        loose = Some(node.clone());
                    }
                }
                next.extend(bus.children(node));
            }
            layer = next;
        }
        loose
    }
}

impl argmind_core::contract::Hands for Hands {
    fn live(&self) -> bool {
        self.armed && self.bus.is_some()
    }

    fn click(&mut self, widget: &Widget) {
        let Some(node) = self.find(widget) else {
            //: **Gone between the offer and the button**, which is the case the re-resolve exists
            //: for and is a proper outcome rather than an error.
            self.done.push(Did::Refused(format!(
                "{:?} ({}) 를 찾지 못했습니다", widget.name, widget.role)));
            return;
        };
        let Some(bus) = self.bus.as_ref() else { return };

        //: **By name, not by index.** `Click` is what GTK calls it. A node with one unnamed action
        //: is taken at its word; a node with none cannot be operated this way at all, and saying so
        //: is better than pressing something else.
        let actions = bus.actions(&node);
        let which = actions
            .iter()
            .position(|a| matches!(a.to_lowercase().as_str(), "click" | "press" | "activate"))
            .or_else(|| (!actions.is_empty()).then_some(0));

        let Some(which) = which else {
            self.done.push(Did::Refused(format!(
                "{:?} 는 눌릴 수 있는 것이 아닙니다", widget.name)));
            return;
        };
        if !self.armed {
            self.done.push(Did::Operated { how: "atspi-action", live: false });
            return;
        }
        //: **`live` says it really went, and a decline is not a failure to ask.** A `DoAction` that
        //: answered false is the widget saying no.
        self.done.push(if bus.act(&node, which as i32) {
            Did::Operated { how: "atspi-action", live: true }
        } else {
            Did::Refused(format!("{:?} 가 동작을 거절했습니다", widget.name))
        });
    }

    fn type_text(&mut self, _text: &str) {
        self.done.push(Did::Refused(NO_SYNTHESIS.into()));
    }

    fn press(&mut self, _key: &str) {
        self.done.push(Did::Refused(NO_SYNTHESIS.into()));
    }
}
