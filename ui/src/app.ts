// SPDX-License-Identifier: Apache-2.0
//
// The interface. TypeScript, because the plan is a Rust core and a front end that is not Rust.
//
// What it has to carry over from `argmind/app.py` is not the layout — it is the argument. **An
// effect the model returned is not something the agent did**; it is a row with two buttons, and
// until one of them is pressed nothing has happened. The server enforces that (`POST /confirm/<n>`
// is the only route that can act); this side simply never has another way to ask.
//
// Everything arrives over one server-sent event stream. There is no polling and no state kept here
// that the server does not also hold: a reload shows the truth, because the page rebuilds itself
// from `hello` and then follows the stream.

type Kind = "hello" | "them" | "say" | "note" | "propose" | "ran" | "gone" | "dropped" | "busy";


import { orb } from "./orb.js";

interface Proposal {
  token: number;
  label: string;
  hint: string;
}

interface Hello {
  model: string;
  dry: boolean;
  speaks: boolean;
  /** Which voice: "neural", "windows", "none" or "off". Named rather than a flag, because the
   *  difference between the two that speak is the whole reason the neural one was built. */
  voice: string;
  hears: boolean;
  noticing: boolean;
  noticed: number;
  attending: string;
  waiting: Proposal[];
}

/** One microphone, speaker or camera, as the server lists it. */
interface Device {
  id: string;
  name: string;
  default: boolean;
}

/** All of one kind, and which is chosen. */
interface Gear {
  kind: string;
  what: string;
  /** What was picked, which may not be in `found` — an unplugged headset is still the choice. */
  chosen: string;
  /** Why the list is empty, where it is. */
  trouble: string;
  found: Device[];
}

/** One thinking dial, as the server names it. */
interface Dial {
  name: string;
  /** The words that go into the prompt beside the number — shown here so the slider cannot mean
   *  one thing on the page and another to the model. */
  gloss: string;
  at: number;
}

/** What argmind is trying to be. */
interface Aim {
  most: number;
  aim: string;
  /** The exact line added to every prompt, or `""` when nothing is set and nothing is added. */
  line: string;
  dials: Dial[];
}

const token = document.body.dataset.token ?? "";

/** Every request carries the token. Localhost is not a permission: any program you run is on it. */
async function post(path: string, body = ""): Promise<void> {
  await fetch(path, {
    method: "POST",
    headers: { "X-Argmind-Token": token, "Content-Type": "text/plain; charset=utf-8" },
    body,
  });
}

const transcript = document.getElementById("transcript") as HTMLElement;
const pending = document.getElementById("pending") as HTMLElement;
const input = document.getElementById("said") as HTMLInputElement;
// Not `status`: `window.status` is a global string, so a bare `const status` here is a
// redeclaration the compiler rejects — and in plain JavaScript it would have silently shadowed it.
const statusLine = document.getElementById("status") as HTMLElement;
const header = document.getElementById("attending") as HTMLElement;

function line(who: string, text: string): void {
  const row = document.createElement("div");
  row.className = `line ${who}`;
  // textContent, never innerHTML: the model's reply is text, and a reply containing a tag is a
  // reply, not markup. The transcript is the one place untrusted text lands.
  row.textContent = text;
  transcript.append(row);
  transcript.scrollTop = transcript.scrollHeight;
}

/** One waiting effect, with the two buttons that are the whole permission model. */
function row(p: Proposal): void {
  const box = document.createElement("div");
  box.className = "proposal";
  box.id = `p${p.token}`;

  const what = document.createElement("span");
  what.className = "what";
  what.textContent = p.label;
  box.append(what);

  if (p.hint) {
    // A second opinion, not a gate — shown beside the button and never disabling it.
    const hint = document.createElement("span");
    hint.className = "hint";
    hint.textContent = p.hint;
    box.append(hint);
  }

  const yes = document.createElement("button");
  yes.className = "yes";
  yes.textContent = "실행";
  yes.onclick = () => {
    // Disabled on the way out, so a second press cannot ask twice. The server refuses a repeat
    // anyway; this is so the person is not left wondering whether it took.
    yes.disabled = no.disabled = true;
    void post(`/confirm/${p.token}`);
  };

  const no = document.createElement("button");
  no.className = "no";
  no.textContent = "아니";
  no.onclick = () => {
    yes.disabled = no.disabled = true;
    void post(`/decline/${p.token}`);
  };

  box.append(yes, no);
  pending.append(box);
}

function settle(tokenId: number, how: string): void {
  document.getElementById(`p${tokenId}`)?.remove();
  if (how) {
    line("note", how);
  }
}

// ---- the orb -----------------------------------------------------------------------------------
//
// **Fed by the events that were already arriving**, not by a second path of its own. Every call
// below sits beside a line that already puts the same thing into the transcript, so the sphere and
// the words cannot come to disagree — which is the only way an animation like this becomes a lie.

const eye = orb(document.getElementById("orb") as HTMLCanvasElement);

// The conversation is the content. The sphere is presence, and presence recedes once there is
// something to read.
function recede(): void {
  eye.presence(transcript.childElementCount === 0 ? 1 : 0.4);
}
recede();

const stream = new EventSource(`/events`);

/** What the status line says about the voice. Three states that matter and one that does not:
 *  speaking with the neural voice, speaking with the Windows one, asked to speak and unable to,
 *  and not asked. */
function voiceReads(d: Hello): string {
  if (!d.speaks) return d.voice === "none" ? "소리 안 나옴" : "소리 꺼짐";
  return d.voice === "neural" ? "소리 켜짐 (신경망)" : "소리 켜짐 (Windows)";
}

/** The last snapshot, kept so a single line of it can change without a whole new one.
 *
 *  The voice can be swapped mid-session — the neural one starts, and the agent takes it up — and
 *  the header said `소리 켜짐 (Windows)` for the rest of the session while the transcript said
 *  *이제 신경망 목소리로 말합니다* right above it. The page was not wrong; it had never been told. */
let lately: Hello | null = null;

function drawStatus(d: Hello): void {
  // Both settings shown, because "it did nothing" and "it said nothing" are the two things
  // a person would otherwise mistake for the agent being broken.
  statusLine.textContent =
    `${d.model}${d.dry ? " · 실행 안 함" : ""} · ${voiceReads(d)}` +
    `${d.hears ? " · 마이크 켜짐" : ""}`;
}

stream.addEventListener("voice", (e) => {
  if (!lately) return;
  lately.voice = (e as MessageEvent).data;
  lately.speaks = lately.voice !== "off" && lately.voice !== "none";
  drawStatus(lately);
  // **The box follows the server**, including when it says no: asking for a voice that cannot be
  // found leaves the switch off, which is the truth.
  aloud.checked = lately.speaks;
});

stream.addEventListener("hello", (e) => {
  const d: Hello = JSON.parse((e as MessageEvent).data);
  lately = d;
  header.textContent = d.attending || "(아무 창도 보고 있지 않음)";
  drawStatus(d);
  talk.hidden = !d.hears;
  aloud.checked = d.speaks;
  drawNoticing(d.noticing, d.noticed);
  // Rebuilt from the server's own list, so a reload is never a page showing buttons the server
  // has forgotten — or missing ones it still holds.
  pending.replaceChildren();
  for (const p of d.waiting) {
    row(p);
  }
});

for (const kind of ["them", "say", "note"] as Kind[]) {
  stream.addEventListener(kind, (e) => {
    line(kind, (e as MessageEvent).data);
    recede();
    //: A reply is the event the person is waiting for; what they typed is not news to them.
    if (kind === "say") {
      eye.jolt("said");
    }
  });
}

stream.addEventListener("propose", (e) => row(JSON.parse((e as MessageEvent).data) as Proposal));
stream.addEventListener("ran", (e) => {
  const d = JSON.parse((e as MessageEvent).data);
  settle(d.token, d.confirmed === false ? `${d.label} — 눌렀지만 화면이 그대로임` : `${d.label} — 됨`);
  //: **The two outcomes look different**, because "it pressed the button and the screen did not
  //: move" is the failure this project keeps finding and it should not read as success.
  eye.jolt(d.confirmed === false ? "refused" : "acted");
});
stream.addEventListener("gone", (e) => {
  const d = JSON.parse((e as MessageEvent).data);
  settle(d.token, `${d.what} — 그새 사라짐`);
  eye.jolt("refused");
});
stream.addEventListener("dropped", (e) => settle(Number((e as MessageEvent).data), ""));
stream.addEventListener("attending", (e) => {
  header.textContent = (e as MessageEvent).data || "(아무 창도 보고 있지 않음)";
});
stream.addEventListener("busy", (e) => {
  const thinking = (e as MessageEvent).data === "1";
  document.body.classList.toggle("busy", thinking);
  //: **Not while the key is held.** `busy` also goes up for the transcription that follows a
  //: release, and letting it overwrite `listening` would make the sphere flicker between two
  //: states during one turn.
  if (!holding) {
    eye.doing(thinking ? "thinking" : "resting");
  }
});

// The stream dying is worth saying out loud: a page that silently stops updating looks exactly
// like an agent that has stopped thinking.
stream.onerror = () => {
  statusLine.textContent = "연결 끊김 — 서버가 살아있는지 확인";
  //: **And the sphere collapses.** A page whose stream died but whose animation carries on
  //: breathing is a page that looks alive while knowing nothing, which is worse than a still one.
  eye.doing("gone");
};

//: Reconnected — `EventSource` retries on its own, and `hello` is what arrives when it succeeds.
stream.addEventListener("hello", () => eye.doing("resting"));

const form = document.getElementById("form") as HTMLFormElement;
form.onsubmit = (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) {
    return;
  }
  input.value = "";
  // Echoed locally at once. The round trip is 0.6 s at the median and several seconds at the
  // tail, and a person who cannot see what they typed types it again.
  line("them", text);
  void post("/say", text);
};

// ---- noticing -----------------------------------------------------------------------------------
//
// Off until switched on, and the count is next to the switch. Something on screen changes 13
// seconds out of 14, so this is a standing invitation to spend a model call every few seconds —
// and how often that is on *your* desktop is a thing to watch rather than guess.

const notice = document.getElementById("notice") as HTMLInputElement;
const noticing = document.getElementById("noticing") as HTMLElement;
let noticedSoFar = 0;

// The switch is in 설정 and the header only reports. **Two elements for one fact**, which is worth
// it here: watching is on by default now, so the thing a person needs at a glance is *is it
// looking right now*, and the thing they need occasionally is *make it stop*.
function drawNoticing(on: boolean, count: number): void {
  noticedSoFar = count;
  notice.checked = on;
  noticing.hidden = !on;
  noticing.textContent = count > 0 ? `지켜보는 중 · ${count}회` : "지켜보는 중";
}

// Whether argmind makes a sound. **A setting since 2026-09-17**, where it used to be `--speak` on
// the command line — a switch you can see is a switch you can turn off when somebody walks in.
const aloud = document.getElementById("aloud") as HTMLInputElement;
aloud.onchange = () => {
  // The server confirms with its own `voice` event and can fail to find a voice at all, so this
  // does not draw the new state itself.
  void post(aloud.checked ? "/aloud/on" : "/aloud/off");
};

notice.onchange = () => {
  // The server confirms with its own `noticing` event and can refuse — a standing 거부 turns the
  // box straight back off — so this does not draw the new state itself.
  void post(notice.checked ? "/notice/on" : "/notice/off");
};

stream.addEventListener("noticing", (e) => {
  drawNoticing((e as MessageEvent).data === "1", noticedSoFar);
});

stream.addEventListener("noticed", (e) => {
  noticedSoFar += 1;
  drawNoticing(true, noticedSoFar);
  // What set it off, before whatever it decided to say about it. A thought arriving with no
  // visible cause reads as the agent talking to itself.
  line("noticed", (e as MessageEvent).data);
  recede();
  //: **The one thing nobody asked for.** Sharp and short, because it is an interruption and should
  //: read as one rather than as another reply arriving.
  eye.jolt("noticed");
});

// ---- what it is looking at ----------------------------------------------------------------------
//
// The single most useful thing an interface for this can show: **the list the model was actually
// given.** If the thing you want is not in it, no model will find it — and until it is on screen
// that is a sentence in a document rather than something you can check.
//
// It is the same call that builds the prompt, so the panel cannot drift into showing a screen the
// model never saw.

const seeing = document.getElementById("seeing") as HTMLElement;
const seen = document.getElementById("seen") as HTMLElement;
const spent = document.getElementById("spent") as HTMLElement;

stream.addEventListener("seeing", (e) => {
  const text = (e as MessageEvent).data;
  seen.textContent = text;
  seeing.hidden = text.trim().length === 0;
});

interface Spent {
  widgets: number;
  listed: number;
  look_ms: number;
  think_ms: number;
}

stream.addEventListener("spent", (e) => {
  const d: Spent = JSON.parse((e as MessageEvent).data);
  // Both halves apart: reading the screen and asking the model are different things, and one
  // number covering both hides which of them is slow today.
  const cut = d.widgets > d.listed ? ` (${d.widgets}개 중 ${d.listed}개만 보여줌)` : "";
  spent.textContent = `${d.widgets}개 읽기 ${Math.round(d.look_ms)} ms · `
    + `생각 ${Math.round(d.think_ms)} ms${cut}`;
});

// ---- what is missing --------------------------------------------------------------------------
//
// The approval lives here. A button is only pressable next to the full sentence saying what it will
// do — a panel with a name and an "install" button would be asking the vague question the whole
// design exists to avoid, and everybody answers that one yes.

interface Missing {
  key: string;
  what: string;
  present: boolean;
  required: boolean;
  found: string;
  how: string;
  offer: boolean;
  // Whether saying yes causes code from somewhere else to run here. It arrives as a flag rather
  // than as emphasis inside `plan`, because a sentence can be rewritten and a property cannot.
  stranger: boolean;
  /** What was answered once and for all: `"allowed"`, `"denied"`, or `""` for never asked. */
  standing: string;
  plan: string;
}

const setup = document.getElementById("setup") as HTMLElement;

/// The four buttons, wherever a permission is being asked about.
///
/// **One builder, so the setup panel and the question above it cannot come to offer different
/// answers.** Order and spacing carry meaning: the two that are remembered sit apart from the two
/// that are not, so 항상 is not the button a finger lands on by accident.
function fourWays(key: string, pressed?: (button: HTMLButtonElement) => void): HTMLElement {
  const buttons = document.createElement("div");
  buttons.className = "ask-buttons";
  for (const [label, how, cls] of [["허용", "allow", "yes"], ["항상 허용", "always", "yes always"],
                                   ["거부", "deny", "no"], ["항상 거부", "never", "no"]] as const) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = cls;
    b.textContent = label;
    b.onclick = () => {
      if (pressed) {
        pressed(b);
      }
      void post(`/decide/${encodeURIComponent(key)}/${how}`);
    };
    buttons.append(b);
  }
  return buttons;
}

function drawNeeds(rows: Missing[]): void {
  const missing = rows.filter((n) => !n.present);
  setup.replaceChildren();
  setup.hidden = missing.length === 0;
  if (missing.length === 0) {
    return;
  }

  const head = document.createElement("div");
  head.className = "setup-head";
  head.textContent = `없는 것 ${missing.length}개`;
  const again = document.createElement("button");
  again.type = "button";
  again.className = "again";
  again.textContent = "다시 확인";
  again.onclick = () => void post("/look");
  head.append(again);
  setup.append(head);

  for (const n of missing) {
    const box = document.createElement("div");
    box.className = n.required ? "need required" : "need";

    const title = document.createElement("div");
    title.className = "need-what";
    title.textContent = `${n.what} — ${n.required ? "필요" : "선택"}`;
    box.append(title);

    const found = document.createElement("div");
    found.className = "need-found";
    found.textContent = n.found;
    box.append(found);

    if (!n.offer) {
      const none = document.createElement("div");
      none.className = "need-how";
      none.textContent = n.how;
      box.append(none);
      setup.append(box);
      continue;
    }

    // **Answered once and for all means not asked again.** It used to draw the four buttons every
    // launch for something already decided, and then act on the standing answer a few seconds
    // later — asking and then ignoring the answer, which is worse than either alone.
    if (n.standing) {
      const settled = document.createElement("div");
      settled.className = "need-how";
      settled.textContent = n.standing === "allowed"
        ? "전에 항상 허용하셨습니다 — 물어보지 않고 진행합니다"
        : "전에 항상 거부하셨습니다 — 설정에서 되돌릴 수 있습니다";
      box.append(settled);
      setup.append(box);
      continue;
    }

    // The whole plan, before the button and not behind it. `textContent` keeps the newlines the
    // server wrote, and `white-space: pre-wrap` in the stylesheet keeps them on screen.
    const plan = document.createElement("div");
    plan.className = "need-plan";
    plan.textContent = n.plan;
    box.append(plan);

    // The one line that changes what is being agreed to, drawn as its own row so it cannot be
    // skimmed past with the rest of the list.
    if (n.stranger) {
      const warn = document.createElement("div");
      warn.className = "need-stranger";
      warn.textContent = "다른 곳에서 만든 코드를 이 컴퓨터에서 실행합니다.";
      box.append(warn);
    }

    // **The same four answers as every other permission.** A missing thing argmind offers to
    // install is a permission question, and answering it 항상 허용 should mean the next machine
    // never asks — which is the whole point of having four buttons rather than one.
    box.append(fourWays(n.key, (button) => {
      // Disabled on the way out: the server re-reads what is missing before acting, so a second
      // press is harmless, but a person should not be left wondering whether the first took.
      button.disabled = true;
      button.textContent = "…";
    }));
    setup.append(box);
  }
}

stream.addEventListener("needs", (e) => drawNeeds(JSON.parse((e as MessageEvent).data) as Missing[]));

// ---- permission, with four answers --------------------------------------------------------------
//
// **Two of the four are remembered and two are not**, which is the whole reason there are four: a
// program that asks the same question every morning teaches people to press the first button
// without reading it, and that is the failure the approval text was written to prevent.
//
// **A click is not asked about here.** Actions keep their 실행/아니 buttons every single time; what
// can be settled once is changing the machine, and switching on something that then keeps running.

interface Asking {
  key: string;
  what: string;
  stranger: boolean;
  plan: string;
}

interface Decision {
  key: string;
  what: string;
  /** What was written down. `""` is nobody has answered — which is not the same as refused. */
  standing: "allowed" | "denied" | "";
  /** Whether an unanswered one is allowed anyway. Watching is, since 2026-09-17. */
  default: boolean;
}

const asking = document.getElementById("asking") as HTMLElement;
const decisions = document.getElementById("decisions") as HTMLElement;

function ask(a: Asking): void {
  asking.replaceChildren();
  const what = document.createElement("div");
  what.className = "ask-what";
  what.textContent = a.what;
  asking.append(what);

  // The whole plan, before the buttons and not behind them.
  const plan = document.createElement("div");
  plan.className = "ask-plan";
  plan.textContent = a.plan;
  asking.append(plan);

  if (a.stranger) {
    const warn = document.createElement("div");
    warn.className = "need-stranger";
    warn.textContent = "다른 곳에서 만든 코드를 이 컴퓨터에서 실행합니다.";
    asking.append(warn);
  }

  asking.append(fourWays(a.key, () => {
    // The question goes as soon as it is answered: leaving it up with its buttons dead is a page
    // that looks like it is still waiting for something.
    asking.hidden = true;
    asking.replaceChildren();
  }));
  asking.hidden = false;
}

stream.addEventListener("asking", (e) => ask(JSON.parse((e as MessageEvent).data) as Asking));

function drawDecisions(rows: Decision[]): void {
  decisions.replaceChildren();
  for (const d of rows) {
    const box = document.createElement("div");
    box.className = "decision";

    const name = document.createElement("span");
    name.textContent = d.what;
    box.append(name);

    const standing = document.createElement("span");
    standing.className = `standing ${d.standing}`;
    // **Unanswered is a state and it is shown.** A settings panel that lists only what was decided
    // cannot tell you what you will be asked about next.
    standing.textContent =
      d.standing === "allowed" ? "항상 허용"
      : d.standing === "denied" ? "항상 거부"
      // **A default is not an answer**, and the two read differently on purpose: one of them is
      // something you chose and the other is something nobody has had to choose yet.
      : d.default ? "기본값 — 허용" : "아직 안 물음";
    box.append(standing);

    if (d.standing !== "") {
      const forget = document.createElement("button");
      forget.type = "button";
      forget.textContent = "되돌리기";
      // **What it goes back to, not what it always went back to.** A key with a default does not
      // become a question again; saying it would is the interface describing a step that no longer
      // happens.
      forget.title = d.default ? "기본값(허용)으로 돌아갑니다" : "다음에 다시 물어봅니다";
      forget.onclick = () => void post(`/forget/${encodeURIComponent(d.key)}`);
      box.append(forget);
    }
    decisions.append(box);
  }
}

stream.addEventListener("decisions",
                        (e) => drawDecisions(JSON.parse((e as MessageEvent).data) as Decision[]));

// ---- the talk key ------------------------------------------------------------------------------
//
// Push-to-talk, and the two halves are two requests. Holding is the floor an open microphone has no
// equivalent of: the agent cannot hear itself, because the device is shut while it speaks, and a
// phone call in the same room is not a turn addressed to it.

const talk = document.getElementById("talk") as HTMLButtonElement;
let holding = false;

function hold(down: boolean): void {
  if (down === holding) {
    return;
  }
  holding = down;
  talk.classList.toggle("holding", down);
  void post(down ? "/listen/start" : "/listen/stop");
}

talk.onpointerdown = () => hold(true);
// `pointerup` on the window and not the button: releasing outside it must still stop the
// recording, or a slip of the mouse leaves the microphone open.
window.addEventListener("pointerup", () => hold(false));
window.addEventListener("pointercancel", () => hold(false));
// And leaving the page entirely — a tab switch mid-hold must not keep it recording.
window.addEventListener("blur", () => hold(false));

window.addEventListener("keydown", (e) => {
  // Not while typing: the space bar belongs to the text box when the text box has it.
  if (e.code === "Space" && !e.repeat && document.activeElement !== input) {
    e.preventDefault();
    hold(true);
  }
});
window.addEventListener("keyup", (e) => {
  if (e.code === "Space") {
    hold(false);
  }
});

stream.addEventListener("hearing", (e) => {
  const open = (e as MessageEvent).data === "1";
  talk.classList.toggle("listening", open);
  //: The microphone is the one state the server knows about and the page cannot infer: the key
  //: going down is a request, and this is the device actually being open.
  eye.doing(open ? "listening" : "resting");
});

input.focus();


// --- which microphone, which speaker, which camera -------------------------------------------
//
// Three rows, each a list and a button. The button **uses** the device — a tone, a second of
// recording, one frame — because a person checking their microphone wants to know whether it hears
// them, and a settings panel that only lists names cannot answer that.

const gear = document.getElementById("gear") as HTMLDetailsElement;
const devices = document.getElementById("devices") as HTMLElement;

stream.addEventListener("devices", (e) => {
  const kinds: Gear[] = JSON.parse((e as MessageEvent).data);
  devices.replaceChildren();
  for (const g of kinds) {
    devices.append(gearRow(g));
  }
  // Nothing to show is not worth a fold that opens onto nothing.
  gear.hidden = kinds.length === 0;
});

stream.addEventListener("tried", (e) => {
  const r = JSON.parse((e as MessageEvent).data) as
    { kind: string; worked: boolean; said: string };
  const said = devices.querySelector(`[data-said="${r.kind}"]`);
  if (said) {
    said.textContent = r.said;
    said.classList.toggle("bad", !r.worked);
  }
});

function gearRow(g: Gear): HTMLElement {
  const row = document.createElement("div");
  row.className = "gear-row";
  const line = document.createElement("div");
  line.className = "line";

  const what = document.createElement("span");
  what.className = "what";
  what.textContent = g.what;
  line.append(what);

  const pick = document.createElement("select");
  // **The first option is the machine's own preference, not a device.** Choosing it forgets the
  // choice rather than pinning whatever happens to be default today.
  const auto = document.createElement("option");
  auto.value = "";
  auto.textContent = "자동 (기본 장치)";
  pick.append(auto);
  for (const d of g.found) {
    const o = document.createElement("option");
    o.value = d.id;
    o.textContent = d.default ? `${d.name}  (기본)` : d.name;
    pick.append(o);
  }
  // A chosen device that is not in the list is unplugged, and saying so beats quietly showing 자동.
  if (g.chosen && !g.found.some((d) => d.id === g.chosen)) {
    const gone = document.createElement("option");
    gone.value = g.chosen;
    gone.textContent = "고른 장치 — 지금 없음";
    pick.append(gone);
  }
  pick.value = g.chosen;
  pick.onchange = () => {
    void post(`/choose/${encodeURIComponent(g.kind)}/${encodeURIComponent(pick.value)}`);
  };
  line.append(pick);

  const test = document.createElement("button");
  test.type = "button";
  test.textContent = "시험";
  test.onclick = () => {
    const said = row.querySelector(`[data-said="${g.kind}"]`) as HTMLElement;
    said.classList.remove("bad");
    // Said before it happens, because each of these makes a sound, records, or looks — and a
    // button that goes quiet for two seconds while a microphone is open is a button nobody trusts.
    said.textContent = g.kind === "speaker" ? "소리를 냅니다…"
      : g.kind === "microphone" ? "2초 동안 듣습니다 — 말해보세요…"
      : "한 장 찍습니다…";
    void post(`/try/${encodeURIComponent(g.kind)}`);
  };
  line.append(test);
  row.append(line);

  const said = document.createElement("div");
  said.className = "said";
  said.dataset.said = g.kind;
  if (g.trouble) {
    said.textContent = g.trouble;
    said.classList.add("bad");
  }
  row.append(said);
  return row;
}


// --- what it is holding in mind ------------------------------------------------------------------
//
// One panel, three lines, revised by argmind itself every half minute while it is watching. **It is
// shown rather than kept inside** for the reason the transcript shows the recogniser's guess: a
// belief that is wrong goes into every prompt until something replaces it, and the only way anybody
// notices is by being able to read it.

interface Thought {
  believes: string;
  waiting: string;
  tried: string;
  /** How many times it has been rewritten. A note that churns and one that settles look the same
   *  in a single reading, and this is the difference. */
  revisions: number;
}

const thinking = document.getElementById("holding") as HTMLDetailsElement;
const note = document.getElementById("note") as HTMLElement;

stream.addEventListener("note-kept", (e) => {
  const t: Thought = JSON.parse((e as MessageEvent).data);
  const rows: [string, string][] = [
    ["믿는 것", t.believes],
    ["기다리는 것", t.waiting],
    ["이미 한 것", t.tried],
  ];
  const filled = rows.filter(([, text]) => text.trim() !== "");
  // A fold that opens onto nothing is worse than no fold — the same rule 모델이 본 것 keeps.
  thinking.hidden = filled.length === 0;
  if (filled.length === 0) return;

  note.replaceChildren();
  const grid = document.createElement("div");
  grid.className = "thought";
  for (const [label, text] of filled) {
    const l = document.createElement("span");
    l.className = "label";
    l.textContent = label;
    const v = document.createElement("span");
    v.className = "text";
    v.textContent = text;
    grid.append(l, v);
  }
  note.append(grid);

  const n = document.createElement("div");
  n.id = "revisions";
  n.textContent = `${t.revisions}번째 고쳐 씀`;
  note.append(n);
});

// --- what it is trying to be -------------------------------------------------------------------
//
// Four sliders and a sentence. **The line the model is actually sent is printed underneath**,
// because a panel of numbers that will not say what it does with them is asking to be believed
// about the only part it could show. At rest the line is empty and so is the prompt slot: nothing
// set means the model is sent exactly what it was sent before any of this existed.

const dials = document.getElementById("dials") as HTMLElement;
const whatissent = document.getElementById("whatissent") as HTMLElement;
const aimBox = document.getElementById("aim") as HTMLTextAreaElement;

stream.addEventListener("mind", (e) => {
  const m: Aim = JSON.parse((e as MessageEvent).data);
  // Never while it is being typed in: replacing the value under the cursor loses what somebody is
  // halfway through writing, and this event arrives on every dial move as well.
  if (document.activeElement !== aimBox) {
    aimBox.value = m.aim;
  }
  dials.replaceChildren();
  for (const d of m.dials) {
    dials.append(dialRow(d, m.most));
  }
  whatissent.textContent = m.line;
});

function dialRow(d: Dial, most: number): HTMLElement {
  const row = document.createElement("div");
  row.className = d.at > 0 ? "dial" : "dial resting";

  const name = document.createElement("span");
  name.className = "name";
  name.textContent = d.name;
  // The gloss is what the model is told, word for word, rather than a tooltip inventing its own
  // wording that could drift from the prompt.
  name.title = d.gloss;
  row.append(name);

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = "0";
  slider.max = String(most);
  slider.step = "1";
  slider.value = String(d.at);
  const at = document.createElement("span");
  at.className = "at";
  at.textContent = String(d.at);
  // Two events on purpose: the number follows the thumb, the server hears about it once, when the
  // thumb is let go. Otherwise one drag is eleven writes of a file and eleven prompts rebuilt.
  slider.oninput = () => {
    at.textContent = slider.value;
    row.classList.toggle("resting", slider.value === "0");
  };
  slider.onchange = () => {
    void post(`/dial/${encodeURIComponent(d.name)}/${slider.value}`);
  };
  row.append(slider, at);
  return row;
}

// Saved when you leave the box, not on every keystroke: a goal is a sentence and half of one is
// not worth writing to disk, let alone putting in front of the model.
aimBox.addEventListener("change", () => {
  void post("/aim", aimBox.value);
});
