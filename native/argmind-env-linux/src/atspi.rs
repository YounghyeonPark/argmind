//! Reading the accessibility tree, and nothing else.
//!
//! Every call here is a D-Bus method on a node the registry named. The shapes are AT-SPI2's and are
//! not negotiable: `a(so)` for children, `s` for a role name, `(iiii)` for extents, `au` for state.

use std::time::{Duration, Instant};

use argmind_core::contract::{Eyes, Widget};
use dbus::arg::Variant;
use dbus::blocking::{Connection, Proxy};
use dbus::Path;

/// How long any one call may take.
///
/// **Short on purpose.** A node belongs to another program, and another program can be wedged;
/// waiting on it would make argmind look wedged too. UI Automation answers a whole screen in
/// 41–266 ms, so a second for one node is already generous.
const PATIENCE: Duration = Duration::from_millis(1000);

/// How deep to go.
///
/// A GTK window is six or seven deep and a web view can be forty. Nothing in this project needs a
/// document's interior — it needs the controls — so the walk stops before it becomes a crawl.
pub(crate) const DEEPEST: usize = 12;

/// How many widgets to bring back at most.
///
/// **A screen, not a document.** The dense screen that took the vision model 82 seconds had 152
/// widgets; a thousand is far past the point where the prompt is the problem rather than the
/// reading.
const MOST: usize = 400;

/// Where a node lives: which program on the bus, and where inside it.
pub(crate) type Node = (String, Path<'static>);

const REGISTRY: &str = "org.a11y.atspi.Registry";
const ROOT: &str = "/org/a11y/atspi/accessible/root";
const ACCESSIBLE: &str = "org.a11y.atspi.Accessible";
const COMPONENT: &str = "org.a11y.atspi.Component";
const ACTION: &str = "org.a11y.atspi.Action";

/// `ATSPI_STATE_ACTIVE`. The frame a person is typing into has it; the others do not.
const STATE_ACTIVE: u32 = 1;

/// A connection to the accessibility bus, which is **not** the session bus.
///
/// The session bus only knows how to hand out its address; everything else happens on a second bus
/// of its own, so that the traffic of a screen reader reading every keystroke is not on the bus
/// everything else is using.
pub struct Bus {
    to: Connection,
}

impl Bus {
    /// Ask the session bus where the accessibility bus is, and say that somebody is listening.
    pub fn open() -> Result<Self, String> {
        let session = Connection::new_session().map_err(|e| format!("세션 버스 ({e})"))?;
        let ask = session.with_proxy("org.a11y.Bus", "/org/a11y/bus", PATIENCE);
        let (address,): (String,) = ask
            .method_call("org.a11y.Bus", "GetAddress", ())
            .map_err(|e| format!("접근성 버스 주소를 얻지 못했습니다 ({e})"))?;

        announce(&ask);

        let channel = dbus::channel::Channel::open_private(&address)
            .and_then(|mut c| c.register().map(|_| c))
            .map_err(|e| format!("접근성 버스에 붙지 못했습니다 ({e})"))?;
        Ok(Bus { to: Connection::from(channel) })
    }

    fn at<'a>(&'a self, node: &'a Node) -> Proxy<'a, &'a Connection> {
        self.to.with_proxy(&node.0, &node.1, PATIENCE)
    }

    pub(crate) fn children(&self, node: &Node) -> Vec<Node> {
        let got: Result<(Vec<(String, Path<'static>)>,), _> =
            self.at(node).method_call(ACCESSIBLE, "GetChildren", ());
        got.map(|(kids,)| kids).unwrap_or_default()
    }

    pub(crate) fn role(&self, node: &Node) -> String {
        let got: Result<(String,), _> = self.at(node).method_call(ACCESSIBLE, "GetRoleName", ());
        got.map(|(r,)| r).unwrap_or_default()
    }

    pub(crate) fn name(&self, node: &Node) -> String {
        let got: Result<(Variant<Box<dyn dbus::arg::RefArg>>,), _> = self.at(node).method_call(
            "org.freedesktop.DBus.Properties",
            "Get",
            (ACCESSIBLE, "Name"),
        );
        got.ok()
            .and_then(|(v,)| v.0.as_str().map(str::to_string))
            .unwrap_or_default()
    }

    /// The rectangle in screen coordinates, or `None` where there is not one.
    ///
    /// `0` is `ATSPI_COORD_TYPE_SCREEN`, which is the only one the core can use — a rectangle
    /// relative to a window is not somewhere a pointer can be put.
    pub(crate) fn extents(&self, node: &Node) -> Option<(i32, i32, i32, i32)> {
        let got: Result<((i32, i32, i32, i32),), _> =
            self.at(node).method_call(COMPONENT, "GetExtents", (0u32,));
        got.ok().map(|(r,)| r).filter(|r| sane(*r))
    }

    /// The low word of the state bitset, or nothing if the node would not say.
    pub(crate) fn states(&self, node: &Node) -> Option<u32> {
        let got: Result<(Vec<u32>,), _> = self.at(node).method_call(ACCESSIBLE, "GetState", ());
        got.ok().and_then(|(bits,)| bits.first().copied())
    }

    pub(crate) fn active(&self, node: &Node) -> bool {
        self.states(node).is_some_and(|low| low & (1 << STATE_ACTIVE) != 0)
    }

    /// What this node can be asked to do, by name.
    ///
    /// **Measured**: a GTK button offers exactly one, called `Click`, described as *Clicks the
    /// button*. Asked by name rather than assuming index zero, because a node with several actions
    /// has no rule that puts the interesting one first.
    pub(crate) fn actions(&self, node: &Node) -> Vec<String> {
        let got: Result<(Vec<(String, String, String)>,), _> =
            self.at(node).method_call(ACTION, "GetActions", ());
        got.map(|(list,)| list.into_iter().map(|(name, _, _)| name).collect()).unwrap_or_default()
    }

    /// Do one, and say whether it said it worked.
    pub(crate) fn act(&self, node: &Node, which: i32) -> bool {
        let got: Result<(bool,), _> = self.at(node).method_call(ACTION, "DoAction", (which,));
        got.map(|(ok,)| ok).unwrap_or(false)
    }

    /// The frame a person is working in: the one with `ATSPI_STATE_ACTIVE`.
    ///
    /// **On the bus rather than on the eyes**, because the hands need it too — to look a widget up
    /// again immediately before operating it, which is the whole point of re-resolving.
    pub(crate) fn active_frame(&self) -> Option<Node> {
        self.applications().into_iter()
            .flat_map(|app| self.children(&app))
            .find(|frame| self.active(frame))
    }

    pub(crate) fn applications(&self) -> Vec<Node> {
        self.children(&(REGISTRY.to_string(), Path::new(ROOT).expect("a literal path")))
    }
}

/// Is a widget currently checked or pressed, if that is a thing it can be?
///
/// **Only so that a press can be shown to have worked.** A push button that worked and one that did
/// nothing look identical from outside; a toggle can be asked. `ATSPI_STATE_CHECKED` is bit 8.
///
/// Opens its own short-lived connection, because it is a measurement rather than part of the
/// contract — nothing in `Eyes` or `Hands` needs it and nothing in the core may ask.
pub fn state_of(widget: &Widget) -> Option<bool> {
    const STATE_CHECKED: u32 = 8;
    let bus = Bus::open().ok()?;
    let frame = bus.active_frame()?;
    let mut layer = vec![frame];
    for _ in 0..DEEPEST {
        let mut next = Vec::new();
        for node in &layer {
            if bus.name(node) == widget.name && bus.role(node) == widget.role {
                return bus.states(node).map(|low| low & (1 << STATE_CHECKED) != 0);
            }
            next.extend(bus.children(node));
        }
        if next.is_empty() {
            break;
        }
        layer = next;
    }
    None
}

/// Say, at most once and only if nobody has said it already, that something is listening.
///
/// **This changes the whole desktop, not just what argmind looks at.** Toolkits publish no tree
/// until an assistive technology announces itself — measured: one node against a hundred and
/// twenty-one.
///
/// **Written only when it is false, and only once per process.** The first version set it on every
/// connection, and three connections in one second — eyes, hands, and a measurement — left the
/// tree empty: every write makes the bridges re-register, and a re-registering bridge has nothing
/// to say. It looked like the hands being unable to find a widget the eyes had just listed, which
/// is a long way from the cause.
fn announce(ask: &Proxy<'_, &Connection>) {
    static SAID: std::sync::Once = std::sync::Once::new();
    SAID.call_once(|| {
        let already: Result<(Variant<bool>,), _> = ask.method_call(
            "org.freedesktop.DBus.Properties", "Get", ("org.a11y.Status", "IsEnabled"));
        if matches!(already, Ok((Variant(true),))) {
            return;
        }
        let _: Result<(), _> = ask.method_call::<(), _, _, _>(
            "org.freedesktop.DBus.Properties", "Set",
            ("org.a11y.Status", "IsEnabled", Variant(true)));
        //: Bridges need a moment to notice and register. Without this the very next read is of a
        //: desktop that has not published yet, which is the same empty answer by a different road.
        std::thread::sleep(Duration::from_millis(400));
    });
}

/// Is this a rectangle that exists on a screen?
///
/// **Measured, not defensive.** A menu item inside a closed combo box reports
/// `(-2147483648, -2147483648, 1, 1)` — `i32::MIN` — rather than failing, and handing that to the
/// core would put a widget at a coordinate no display has. A zero-sized box is the other way of
/// saying *not laid out*.
pub fn sane(rect: (i32, i32, i32, i32)) -> bool {
    let (x, y, w, h) = rect;
    w > 0 && h > 0 && x > i32::MIN / 2 && y > i32::MIN / 2 && x < i32::MAX / 2 && y < i32::MAX / 2
}

/// Which window a person is working in, and what is in it.
pub struct Sight {
    bus: Option<Bus>,
    /// Said once. A machine with no accessibility bus is a machine argmind cannot see on, and
    /// repeating that every three seconds helps nobody.
    complained: bool,
    pub watcher: Watcher,
}

/// Eyes that read whatever window is active.
///
/// **The same type twice**, because on Linux the distinction the Windows environment makes — the
/// window in front versus the window being attended to — is not available the same way: there is no
/// foreground handle to exclude, only an `active` frame. `Attending` and `Plain` are both this.
pub type Attending = Sight;
/// See [`Attending`].
pub type Plain = Sight;

/// There is nothing to sample here, and saying so is better than sampling nothing.
///
/// The Windows watcher keeps the handles the foreground has been, so that while a person is in
/// argmind it can answer with the window they were in before it. AT-SPI has no such handle, and
/// `active` already means *the one they are in*.
#[derive(Default, Clone)]
pub struct Watcher;

impl Watcher {
    pub fn saw(&self, _handle: isize) {}
    pub fn attended(&self, _in_front: isize) -> Option<isize> {
        None
    }
    pub fn watch(&self) {}
}

impl Sight {
    pub fn new(watcher: Watcher) -> Result<Self, String> {
        //: **Opened here rather than lazily**, so a machine with no accessibility bus says so at
        //: startup rather than the first time somebody asks a question.
        let bus = Bus::open()?;
        Ok(Sight { bus: Some(bus), complained: false, watcher })
    }

    /// The active frame, and what a person would call it.
    fn active_frame(&self) -> Option<(Node, String)> {
        let bus = self.bus.as_ref()?;
        let frame = bus.active_frame()?;
        let mut title = bus.name(&frame);
        if title.is_empty() {
            //: A frame with no name is common; the application's own name is the next best thing
            //: and is what a person would call the window.
            title = bus.applications().into_iter()
                .find(|app| bus.children(app).contains(&frame))
                .map(|app| bus.name(&app))
                .unwrap_or_default();
        }
        Some((frame, title))
    }

    /// Everything under a node, breadth first, stopping at [`DEEPEST`] and [`MOST`].
    ///
    /// **Breadth first on purpose.** A depth-first walk of a window whose first child is a document
    /// spends the whole budget inside it; the controls a person means are near the top.
    fn under(&self, from: Node, window: &str) -> Vec<Widget> {
        let Some(bus) = self.bus.as_ref() else { return Vec::new() };
        let mut out = Vec::new();
        let mut layer = vec![from];
        for _ in 0..DEEPEST {
            if layer.is_empty() || out.len() >= MOST {
                break;
            }
            let mut next = Vec::new();
            for node in &layer {
                if out.len() >= MOST {
                    break;
                }
                let role = bus.role(node);
                if let Some((x, y, w, h)) = bus.extents(node) {
                    out.push(Widget::new(&bus.name(node), &role,
                                         (x as i64, y as i64, w as i64, h as i64), window));
                }
                next.extend(bus.children(node));
            }
            layer = next;
        }
        out
    }

    /// The handle and title of the window being attended to.
    ///
    /// **The handle is always zero.** Nothing on this side of the seam is an operating system
    /// handle, and inventing a number that looks like one would be worse than an honest nothing.
    pub fn window(&self) -> (isize, String) {
        (0, self.active_frame().map(|(_, title)| title).unwrap_or_default())
    }

    /// The name and role of whatever is at a screen point.
    ///
    /// **`GetAccessibleAtPoint` answers with a direct child, not with the deepest thing there**,
    /// and the first version of this called it once and believed it. Asked for the middle of a
    /// button called `Menu` it answered `("", "panel")` — the container the button sits in, which
    /// is true and useless. So it is followed down until nothing claims the point.
    pub fn at_point(&self, x: i64, y: i64) -> Option<(String, String)> {
        let bus = self.bus.as_ref()?;
        let (frame, _) = self.active_frame()?;
        let mut here = frame;
        //: Bounded, because a node that answered with itself would otherwise be forever.
        for _ in 0..DEEPEST {
            let got: Result<((String, Path<'static>),), _> = bus.at(&here).method_call(
                COMPONENT,
                "GetAccessibleAtPoint",
                (x as i32, y as i32, 0u32),
            );
            let Ok((deeper,)) = got else { break };
            if deeper.1.is_empty() || deeper.1.as_bytes() == b"/org/a11y/atspi/null"
                || deeper == here {
                break;
            }
            here = deeper;
        }
        //: **The name of the deepest thing, or the nearest named thing above it.** A button's label
        //: is often a child of the button with a name of its own and a role of `label`; answering
        //: with the label's empty parent would be the same uselessness one level down.
        let (name, role) = (bus.name(&here), bus.role(&here));
        if !name.is_empty() {
            return Some((name, role));
        }
        //: One look inside: a nameless container over a single named child is the ordinary shape.
        for child in bus.children(&here) {
            let inner = bus.name(&child);
            if !inner.is_empty() {
                return Some((inner, bus.role(&child)));
            }
        }
        Some((name, role))
    }
}

impl Eyes for Sight {
    fn foreground(&mut self) -> Vec<Widget> {
        if self.bus.is_none() {
            if !self.complained {
                self.complained = true;
                eprintln!("  no accessibility bus, so nothing can be seen");
            }
            return Vec::new();
        }
        match self.active_frame() {
            Some((frame, title)) => self.under(frame, &title),
            //: **Empty rather than an error.** No active frame is a legitimate reading — nothing is
            //: focused — and the agent then has nothing to offer, which is correct.
            None => Vec::new(),
        }
    }

    fn timed_foreground(&mut self) -> (Vec<Widget>, f64) {
        let began = Instant::now();
        let widgets = self.foreground();
        (widgets, began.elapsed().as_secs_f64())
    }
}
