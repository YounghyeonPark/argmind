//! Operating a control instead of aiming at it.
//!
//! **This path is off by default and exists so it can be measured.** Two harnesses in `measure/`
//! have no other purpose: `obstruction` asks whether a pattern reaches a control a synthetic click
//! cannot, and `pointer_theft` asks whether it avoids yanking the cursor. Both compare the pattern
//! arm against the click arm, so neither can run without it — and the earlier decision to leave it
//! out would have left two experiments with one arm each, which is the shape this project calls a
//! measurement that cannot answer its own question.
//!
//! **Both have now answered, and they disagree — 2026-09-16.**
//!
//! | axis | with patterns | without | n |
//! |---|---|---|---|
//! | task success, ordinary goals | 3 of 3 | 3 of 3 | 3 |
//! | reaching past something covering the target | missed | missed | 24 arms over 3 runs |
//! | the mouse pointer | **unmoved** | dragged to the widget | 3 |
//!
//! **Obstruction says nothing, and says it for a reason worth knowing**: [`element_for`] starts at
//! `ElementFromPoint` and walks *upwards*, so a window over the target hands it that window's
//! ancestors and the pattern falls back to a click. The hope that a control pattern *does not aim
//! at a pixel* is not true of this one. Whatever else patterns are for, they are not for that.
//!
//! **The pointer is the axis they win, every time.** A synthetic click moves the operating system's
//! cursor to the widget and presses; `Invoke` calls a method and the pointer stays where the person
//! left it. On a machine somebody is working at, that is the difference between an agent that acts
//! and one that interrupts.
//!
//! **Still off by default**, and now for a narrower reason than before: the shipped click is
//! confirmed by a person pressing a button, so the cursor was already about to move to something
//! they chose. What the measurement removes is the old justification — *they buy nothing* — which
//! was true of the two axes anybody had looked at and false of the third.
//!
//! The order is fixed and it matters. `Invoke` first, because a control that offers it means *press
//! me*; a combo box does not have it and falls through to `ExpandCollapse`, which is the measured
//! failure the whole ancestor walk was built for. Putting expand first would make a split button
//! drop its menu instead of doing its job.

use windows::core::Interface;
use windows::Win32::UI::Accessibility::{
    IUIAutomation, IUIAutomationElement, IUIAutomationExpandCollapsePattern,
    IUIAutomationInvokePattern, IUIAutomationSelectionItemPattern, IUIAutomationTogglePattern,
    UIA_ExpandCollapsePatternId, UIA_InvokePatternId, UIA_SelectionItemPatternId,
    UIA_TogglePatternId,
};
use windows::Win32::Foundation::POINT;

/// What a pattern attempt did.
#[derive(Debug, Clone, PartialEq)]
pub enum Operated {
    /// This pattern was found and used. The name is what the result files record.
    By(&'static str),
    /// A pattern was found but a dry run stopped before using it — so a dry run says which path
    /// *would* have been taken rather than leaving the two indistinguishable.
    WouldUse(&'static str),
    /// No pattern applies, so the caller should click. The reason is carried, because
    /// *"fell back"* without a reason is the same as no answer.
    Click(String),
}

/// How far up to look for the named element. Five, as in the Python — a caption button whose centre
/// resolves straight to the application is the measured case where that runs out, and a synthetic
/// click is right there anyway.
pub const HOPS: usize = 5;

/// Find the element a widget names, from its own centre and then upwards, and operate it.
///
/// **Name and role, not name alone.** Walking up on the name matched an ancestor that happened to
/// share it — a tab item named `Claude Code` inside a group of the same name resolved to the group,
/// and operating the group is not what was asked for.
pub fn operate(automation: &IUIAutomation, x: i64, y: i64, name: &str, role: &str, live: bool)
               -> Operated {
    let element = match element_for(automation, x, y, name, role) {
        Ok(e) => e,
        Err(why) => return Operated::Click(why),
    };
    let mut offered: Vec<String> = Vec::new();
    for (label, id) in [("invoke", UIA_InvokePatternId), ("expand", UIA_ExpandCollapsePatternId),
                        ("toggle", UIA_TogglePatternId), ("select", UIA_SelectionItemPatternId)] {
        let raw = match unsafe { element.GetCurrentPattern(id) } {
            Ok(r) => r,
            Err(_) => continue,
        };
        //: A dry run resolves the pattern and stops there.
        if !live {
            return Operated::WouldUse(name_of(label));
        }
        let used = unsafe {
            match label {
                "invoke" => raw.cast::<IUIAutomationInvokePattern>().and_then(|p| p.Invoke()),
                "expand" => raw.cast::<IUIAutomationExpandCollapsePattern>()
                    .and_then(|p| p.Expand()),
                "toggle" => raw.cast::<IUIAutomationTogglePattern>().and_then(|p| p.Toggle()),
                _ => raw.cast::<IUIAutomationSelectionItemPattern>().and_then(|p| p.Select()),
            }
        };
        match used {
            Ok(()) => return Operated::By(name_of(label)),
            //: **Offered and then failed is not the same as never offered**, and falling through to
            //: the next pattern would operate a different thing from the one that said it could.
            Err(e) => return Operated::Click(format!("{label} was offered and raised {e}")),
        }
    }
    offered.push("the element offers none of invoke, expand, toggle or select".into());
    Operated::Click(offered.join("; "))
}

fn name_of(label: &str) -> &'static str {
    match label {
        "invoke" => "invoke",
        "expand" => "expand",
        "toggle" => "toggle",
        _ => "select",
    }
}

fn element_for(automation: &IUIAutomation, x: i64, y: i64, name: &str, role: &str)
               -> Result<IUIAutomationElement, String> {
    unsafe {
        let walker = automation.RawViewWalker()
            .map_err(|e| format!("no tree walker ({e})"))?;
        let mut element = automation.ElementFromPoint(POINT { x: x as i32, y: y as i32 })
            .map_err(|_| "nothing at the centre".to_string())?;
        for _ in 0..HOPS {
            let (Ok(n), Ok(r)) = (element.CurrentName(), element.CurrentLocalizedControlType())
            else {
                return Err("the element could not be read".into());
            };
            if n.to_string() == name && r.to_string() == role {
                return Ok(element);
            }
            element = walker.GetParentElement(&element)
                .map_err(|_| "walked past the top without a name and role match".to_string())?;
        }
        Err(format!("no ancestor within {HOPS} hops matched the name and role"))
    }
}
