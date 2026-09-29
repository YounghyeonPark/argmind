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
  /** Whether a proposal is carried out without the button. */
  acts: boolean;
  /** Whether the camera is open right now — the fact, not the permission. */
  watching_you: boolean;
  /** Which voice: "neural", "windows", "none" or "off". Named rather than a flag, because the
   *  difference between the two that speak is the whole reason the neural one was built. */
  voice: string;
  hears: boolean;
  noticing: boolean;
  noticed: number;
  attending: string;
  waiting: Proposal[];
  /** The one thing argmind has asked about and nobody has answered, or null. */
  wonder: Wonder | null;
  /** Whether the head direction is being worked out — the switch, not whether the camera is open.
   *  Nothing is stored either way; the reading is a word with an age and it evaporates. */
  attending_to_you: boolean;
  /** 눈치, 0–10. Behaviour, never prompt. */
  nunchi: number;
  /** Whether new faces are being written down — the switch, not whether the camera is open. */
  remembering: boolean;
  /** Whether the two ONNX files are on disk. Without them the switch is on and nothing happens. */
  face_models: boolean;
  people: Person[];
}

/** One person argmind has seen. **No print** — the 128 numbers never leave the server, so a page
 *  in a screenshot or a bug report carries nobody's face. */
/** Something argmind does not know. **Its text is a constant on the server**, not a model's
 *  sentence — the deterministic side noticed a gap it could name, so no prompt was edited to make
 *  this happen and none of the 1-for-7 risk is being taken. */
interface Wonder {
  id: number;
  asked: string;
  /** Placeholder for the box, when the answer is typed. Empty when it is chosen. */
  hint: string;
  /** One line per candidate, in the order the eyes gave them. The answer is an index into this,
   *  which the server re-resolves against a fresh reading — and refuses if the count moved. */
  among: string[];
}

interface Person {
  key: string;
  name: string;
  /** Unix seconds. */
  first: number;
  last: number;
  times: number;
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
  acting.checked = d.acts;
  camera.checked = d.watching_you;
  askRow(d.wonder);
  // **The switch, not the device** — unlike the camera above. Nothing happens while the models are
  // missing, so the rows say so rather than looking on and doing nothing.
  const noModels = d.face_models ? "" : "얼굴 모델이 없습니다 — 필요한 것에서 받으세요";
  drawNunchi(d.nunchi);
  attention.checked = d.attending_to_you;
  attention.title = noModels;
  faces.checked = d.remembering;
  faces.title = noModels;
  drawPeople(d.people);
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

stream.addEventListener("wonder", (e) => {
  const raw = (e as MessageEvent).data;
  askRow(raw === "null" ? null : (JSON.parse(raw) as Wonder));
});

// **One at a time, and it replaces rather than stacks.** The server holds at most one open
// question; this mirrors that instead of trusting it, because a column of questions is the pile of
// confirm buttons the owner pointed at once already.
function askRow(w: Wonder | null): void {
  document.getElementById("wonder")?.remove();
  if (!w) return;

  const box = document.createElement("div");
  box.className = "proposal";
  box.id = "wonder";

  const what = document.createElement("span");
  what.className = "what";
  what.textContent = w.asked;
  box.append(what);

  const parts: HTMLElement[] = [];
  const send = (text: string) => {
    // Disabled on the way out so a second press cannot answer twice; the server refuses a repeat
    // anyway, and this is so the person is not left wondering whether it took.
    for (const p of parts) { (p as HTMLInputElement | HTMLButtonElement).disabled = true; }
    void post(`/wonder/${w.id}`, text);
  };

  let first: HTMLElement | null = null;
  if (w.among.length > 0) {
    // **Candidates are buttons, not a number to type.** A free-text answer to *which one* would be
    // a second way to say an index and a second way to get it wrong. Each carries its position,
    // because that is the only thing telling two identically named buttons apart — and the person
    // looking at their own screen is exactly who a position helps.
    w.among.forEach((line, i) => {
      const pick = document.createElement("button");
      pick.className = "yes";
      pick.textContent = line;
      pick.onclick = () => send(String(i));
      parts.push(pick);
      box.append(pick);
      if (!first) first = pick;
    });
  } else {
    const answer = document.createElement("input");
    answer.type = "text";
    answer.placeholder = w.hint;
    answer.maxLength = 40;
    answer.onkeydown = (ev) => { if (ev.key === "Enter") { send(answer.value); } };
    parts.push(answer);
    box.append(answer);
    first = answer;

    const ok = document.createElement("button");
    ok.className = "yes";
    ok.textContent = "그래요";
    ok.onclick = () => send(answer.value);
    parts.push(ok);
    box.append(ok);
  }

  // **Skipping is a real answer and looks like one.** Nothing here is asked twice, so a question
  // with no way out would be a question somebody has to answer to make it go away — and for a
  // choice it is also how you say *none of them*, which declines the action.
  const skip = document.createElement("button");
  skip.className = "no";
  skip.textContent = "넘어가기";
  skip.onclick = () => send("");
  parts.push(skip);
  box.append(skip);

  pending.append(box);
  first?.focus();
}
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
// ---- what the camera reads ---------------------------------------------------------------------
const sense = document.getElementById("sense") as HTMLElement;
const held = document.getElementById("held") as HTMLElement;

stream.addEventListener("sense", (e) => {
  const raw = (e as MessageEvent).data;
  if (raw === "null") {
    sense.hidden = true;
    return;
  }
  const d = JSON.parse(raw) as { camera: string; present: boolean };
  sense.hidden = false;
  sense.replaceChildren();
  sense.append(`카메라 · ${d.camera}`);
  if (!d.present) {
    // **The gate, said out loud.** This is what stops argmind speaking first at all, and it was
    // silent: five minutes of sitting still and every unprompted remark stopped with no sign why.
    const away = document.createElement("span");
    away.className = "away";
    away.textContent = " · 자리 비움으로 봄 — 먼저 말 걸지 않음";
    sense.append(away);
  }
});

// 눈치's only effect is a remark not yet made, so it has to be seen or it cannot be told from
// broken.
stream.addEventListener("held", (e) => {
  held.hidden = (e as MessageEvent).data !== "1";
});

function drawNoticing(on: boolean, count: number): void {
  noticedSoFar = count;
  notice.checked = on;
  noticing.hidden = !on;
  noticing.textContent = count > 0 ? `지켜보는 중 · ${count}회` : "지켜보는 중";
}

// Whether argmind makes a sound. **A setting since 2026-09-17**, where it used to be `--speak` on
// the command line — a switch you can see is a switch you can turn off when somebody walks in.
// Whether argmind presses the button itself. **The checkbox is not the state** — the server owns
// it as a standing answer and can refuse, so this asks and then waits to be told, exactly like
// 지켜보기. Turning it on with nothing answered raises the four-button window; the checkbox goes
// back off until that window is answered, which is why it is set from the event and not here.
// The camera. **The checkbox follows the device, not the permission** — a standing 허용 with a
// camera that will not open leaves the first true and the second false, and the switch has to show
// the second or it is telling you a light is on when it is not.
const camera = document.getElementById("camera") as HTMLInputElement;
camera.onchange = () => {
  const want = camera.checked;
  camera.checked = false;
  void post(want ? "/camera/on" : "/camera/off");
};

stream.addEventListener("camera", (e) => {
  camera.checked = (e as MessageEvent).data === "1";
});

// Whether argmind works out where the head is pointed. **The same shape as the camera above** —
// the server owns the standing answer and can raise the four-button window, so the checkbox is set
// from the event rather than from the click. Turning it on opens nothing and stores nothing.
const attention = document.getElementById("attention") as HTMLInputElement;
attention.onchange = () => {
  const want = attention.checked;
  attention.checked = false;
  void post(want ? "/attention/on" : "/attention/off");
};

stream.addEventListener("attention", (e) => {
  attention.checked = (e as MessageEvent).data === "1";
});

// Remembering faces. **The same shape as the camera above** — the server owns the standing answer
// and can raise the four-button window, so the checkbox is set from the event rather than from the
// click. Turning it on does not open anything: three things have to be true before a face is
// written down, and this is one of them.
const faces = document.getElementById("faces") as HTMLInputElement;
const people = document.getElementById("people") as HTMLElement;
const peopleRows = document.getElementById("people-rows") as HTMLElement;

faces.onchange = () => {
  const want = faces.checked;
  faces.checked = false;
  void post(want ? "/faces/on" : "/faces/off");
};

stream.addEventListener("faces", (e) => {
  faces.checked = (e as MessageEvent).data === "1";
});

stream.addEventListener("people", (e) => {
  drawPeople(JSON.parse((e as MessageEvent).data) as Person[]);
});

/** A date somebody can read, from Unix seconds. */
function on(seconds: number): string {
  if (!seconds) return "";
  const d = new Date(seconds * 1000);
  return `${d.getMonth() + 1}월 ${d.getDate()}일`;
}

// **The list is the safeguard, not decoration.** Both ways this can go wrong are visible in it: a
// threshold too low merges two people into one row, and one too high makes the same person a new
// row every session. The threshold is OpenCV's rather than measured here, so somebody has to be
// able to see which is happening.
function drawPeople(who: Person[]): void {
  people.hidden = who.length === 0;
  peopleRows.textContent = "";
  for (const p of who) {
    const row = document.createElement("div");
    row.className = "row";

    const name = document.createElement("input");
    name.type = "text";
    name.value = p.name;
    name.placeholder = "이름 없음";
    name.maxLength = 40;
    // **On blur and on Enter, not on every keystroke.** A write per character would be a file
    // rewritten forty times for one name.
    const save = () => void post(`/face/name/${encodeURIComponent(p.key)}`, name.value);
    name.onblur = save;
    name.onkeydown = (ev) => { if (ev.key === "Enter") { name.blur(); } };

    const when = document.createElement("span");
    when.className = "cost";
    when.textContent = `${on(p.first)}부터 · ${p.times}번 · 마지막 ${on(p.last)}`;

    const drop = document.createElement("button");
    drop.type = "button";
    drop.className = "plain";
    drop.textContent = "잊기";
    drop.onclick = () => void post(`/face/forget/${encodeURIComponent(p.key)}`);

    row.append(name, when, drop);
    peopleRows.append(row);
  }
}

// **Asks first.** Everything else in this panel is reversible by pressing it again; this is not.
const forgetFaces = document.getElementById("forget-faces") as HTMLButtonElement;
forgetFaces.onclick = () => {
  if (window.confirm("기억한 얼굴을 모두 지우고 파일도 지웁니다. 되돌릴 수 없습니다.")) {
    void post("/faces/forget-all");
  }
};

const acting = document.getElementById("acting") as HTMLInputElement;
acting.onchange = () => {
  const want = acting.checked;
  acting.checked = false;
  void post(want ? "/acting/on" : "/acting/off");
};

stream.addEventListener("acting", (e) => {
  acting.checked = (e as MessageEvent).data === "1";
});

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
  // **Brought to where the person is looking.** The question lives above the settings panel and
  // the switches that raise it live deep inside it, so ticking one after scrolling down put the
  // question off screen and unticked the box — which reads exactly as *I turned it on and nothing
  // happened*. The owner reported that on 2026-09-29 with a screenshot of a camera line that never
  // grew a head direction.
  asking.scrollIntoView({ block: "center", behavior: "smooth" });
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


// --- which voice ---------------------------------------------------------------------------------

// The neural talker is a *base* model: it ships with no speakers, and the timbre is whatever the
// sampler drew. A seed fixes that draw — measured 2026-09-21, four unseeded calls gave four
// different voices and `seed = 7` gave the same bytes four times — but only for *one sentence*:
// across four different sentences a seed held to 119 Hz of itself while different seeds spanned
// 137 Hz, which is no ratio at all. Since every reply is a different sentence, a seed alone is a
// different person every turn.
//
// What holds a person is a cloned voice: argmind draws one reference sentence at a fixed seed and
// registers it with the server, and the clone then tracks that reference at +0.99 and holds to a
// median 26 Hz across sentences. So this is a list of five of those, ordered by measured pitch —
// and the measurement is printed beside each one, because a person choosing a voice is entitled to
// see the number the ordering rests on. **Every change speaks**, since a name and a number in a
// list are not a voice until you have heard one.

interface Preset {
  key: string;
  what: string;
  /** Median f0 of the clone over three sentences, Hz. */
  hz: number;
  /** How far that moved between them — the honest limit of the label. */
  wobble: number;
}

interface Voice {
  chosen: string;
  /** "neural" | "windows" | "off" | "none" — what is speaking right now. */
  is: string;
  /** Whether the choice changes anything. False for SAPI, which has one voice and no registry. */
  matters: boolean;
  presets: Preset[];
}

const voiceRow = document.getElementById("voice") as HTMLElement;

stream.addEventListener("voice-seed", (e) => {
  const v: Voice = JSON.parse((e as MessageEvent).data);
  voiceRow.replaceChildren(drawVoice(v));
});

function drawVoice(v: Voice): HTMLElement {
  const row = document.createElement("div");
  row.className = v.matters ? "gear-row" : "gear-row inert";

  const line = document.createElement("div");
  line.className = "line";
  const what = document.createElement("span");
  what.className = "what";
  what.textContent = "목소리";

  const pick = document.createElement("select");
  for (const p of v.presets) {
    const o = document.createElement("option");
    o.value = p.key;
    // The Hz is in the label rather than a tooltip: it is the ordering, and an ordering nobody can
    // see is one nobody can catch being wrong.
    o.textContent = `${p.what} — ${p.hz} Hz`;
    o.title = `문장에 따라 ±${p.wobble} Hz 움직입니다`;
    o.selected = p.key === v.chosen;
    pick.append(o);
  }
  pick.disabled = !v.matters;
  // Choosing is hearing: the server remembers it and says one sentence in it, so the list is
  // pressed through rather than read.
  pick.onchange = () => void post(`/voice/${encodeURIComponent(pick.value)}`);

  const again = document.createElement("button");
  again.type = "button";
  again.textContent = "다시 듣기";
  again.disabled = !v.matters;
  again.onclick = () => void post(`/voice/${encodeURIComponent(pick.value)}`);

  line.append(what, pick, again);
  row.append(line);

  const said = document.createElement("span");
  said.className = "said";
  said.textContent = v.matters
    ? "고르면 그 목소리로 한 문장 말합니다 — 숫자는 잰 음높이입니다"
    : v.is === "windows"
      ? "지금은 Windows 목소리로 말하고 있어서 이 목록은 쓰이지 않습니다"
      : "소리 내어 말하기가 꺼져 있습니다";
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
  /** What argmind judges is worth doing, as a thing on screen the program named from the widget
   *  the model chose — never a sentence the model wrote. Usually empty, and that is right. */
  worth: string;
  /** How many ticks that judgement has been held without anybody acting on it. */
  stuck: number;
  tried: string;
  /** How many ticks the belief has held unchanged. Counted by the program, never asked of the
   *  model, for the same reason `tried` is. */
  held: number;
  /** How many times it has been rewritten. A note that churns and one that settles look the same
   *  in a single reading, and this is the difference. */
  revisions: number;
}

const thinking = document.getElementById("holding") as HTMLDetailsElement;
const note = document.getElementById("note") as HTMLElement;

stream.addEventListener("note-kept", (e) => {
  const t: Thought = JSON.parse((e as MessageEvent).data);
  const rows: [string, string][] = [
    // The belief carries how long it has held: "he is reading" and "he has been reading for six of
    // my looks" are the same sentence otherwise, and only the second is worth anything.
    [t.held > 1 ? `믿는 것 (${t.held}번째 그대로)` : "믿는 것", t.believes],
    // A judgement nobody acted on says how long it has waited — the number the next tick sees.
    [t.stuck > 1 ? `할 만한 것 (${t.stuck}번째인데 아무 일도 없음)` : "할 만한 것", t.worth],
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

// 눈치. **The line under the slider is the whole contract**, in the terms somebody would check
// against what argmind actually does — because a dial whose effect cannot be seen is the kind this
// project removes rather than keeps.
const nunchi = document.getElementById("nunchi") as HTMLInputElement;
const nunchiAt = document.getElementById("nunchi-at") as HTMLElement;
const nunchiRow = document.getElementById("nunchi-row") as HTMLElement;
const nunchiSays = document.getElementById("nunchi-says") as HTMLElement;

function drawNunchi(n: number): void {
  nunchi.value = String(n);
  nunchiAt.textContent = String(n);
  nunchiRow.classList.toggle("resting", n === 0);
  nunchiSays.textContent = n === 0
    ? "먼저 말을 거는 때를 따로 조절하지 않습니다"
    : `먼저 거는 말 사이 ${n * 10}초 이상 · 고개를 돌리고 있으면 다시 볼 때까지 기다립니다`;
}

nunchi.oninput = () => {
  nunchiAt.textContent = nunchi.value;
  nunchiRow.classList.toggle("resting", nunchi.value === "0");
};
nunchi.onchange = () => void post(`/nunchi/${nunchi.value}`);
stream.addEventListener("nunchi", (e) => drawNunchi(Number((e as MessageEvent).data)));

// Saved when you leave the box, not on every keystroke: a goal is a sentence and half of one is
// not worth writing to disk, let alone putting in front of the model.
aimBox.addEventListener("change", () => {
  void post("/aim", aimBox.value);
});
