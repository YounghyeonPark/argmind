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

interface Proposal {
  token: number;
  label: string;
  hint: string;
}

interface Hello {
  model: string;
  dry: boolean;
  speaks: boolean;
  hears: boolean;
  noticing: boolean;
  noticed: number;
  attending: string;
  waiting: Proposal[];
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

const stream = new EventSource(`/events`);

stream.addEventListener("hello", (e) => {
  const d: Hello = JSON.parse((e as MessageEvent).data);
  header.textContent = d.attending || "(아무 창도 보고 있지 않음)";
  // Both settings shown, because "it did nothing" and "it said nothing" are the two things
  // a person would otherwise mistake for the agent being broken.
  statusLine.textContent =
    `${d.model}${d.dry ? " · 실행 안 함" : ""}${d.speaks ? " · 소리 켜짐" : " · 소리 꺼짐"}` +
    `${d.hears ? " · 마이크 켜짐" : ""}`;
  talk.hidden = !d.hears;
  drawNoticing(d.noticing, d.noticed);
  // Rebuilt from the server's own list, so a reload is never a page showing buttons the server
  // has forgotten — or missing ones it still holds.
  pending.replaceChildren();
  for (const p of d.waiting) {
    row(p);
  }
});

for (const kind of ["them", "say", "note"] as Kind[]) {
  stream.addEventListener(kind, (e) => line(kind, (e as MessageEvent).data));
}

stream.addEventListener("propose", (e) => row(JSON.parse((e as MessageEvent).data) as Proposal));
stream.addEventListener("ran", (e) => {
  const d = JSON.parse((e as MessageEvent).data);
  settle(d.token, d.confirmed === false ? `${d.label} — 눌렀지만 화면이 그대로임` : `${d.label} — 됨`);
});
stream.addEventListener("gone", (e) => {
  const d = JSON.parse((e as MessageEvent).data);
  settle(d.token, `${d.what} — 그새 사라짐`);
});
stream.addEventListener("dropped", (e) => settle(Number((e as MessageEvent).data), ""));
stream.addEventListener("attending", (e) => {
  header.textContent = (e as MessageEvent).data || "(아무 창도 보고 있지 않음)";
});
stream.addEventListener("busy", (e) => {
  document.body.classList.toggle("busy", (e as MessageEvent).data === "1");
});

// The stream dying is worth saying out loud: a page that silently stops updating looks exactly
// like an agent that has stopped thinking.
stream.onerror = () => {
  statusLine.textContent = "연결 끊김 — 서버가 살아있는지 확인";
};

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

const notice = document.getElementById("notice") as HTMLButtonElement;
let noticedSoFar = 0;

function drawNoticing(on: boolean, count: number): void {
  noticedSoFar = count;
  notice.classList.toggle("on", on);
  notice.textContent = on ? `지켜보는 중 · ${count}회` : "지켜보기";
  notice.title = on
    ? "화면이 바뀌면 먼저 말을 겁니다. 알아챌 때마다 모델 호출 한 번."
    : "켜면 창이 바뀔 때 먼저 말을 겁니다.";
}

notice.onclick = () => {
  const turningOn = !notice.classList.contains("on");
  // Drawn at once rather than waiting for the round trip: the server confirms with its own event,
  // and a button that does nothing for a moment is a button people press twice.
  drawNoticing(turningOn, noticedSoFar);
  void post(turningOn ? "/notice/on" : "/notice/off");
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
  plan: string;
}

const setup = document.getElementById("setup") as HTMLElement;

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

    const yes = document.createElement("button");
    yes.type = "button";
    yes.className = "yes";
    yes.textContent = "예";
    yes.onclick = () => {
      // Disabled on the way out: the server re-reads what is missing before acting, so a second
      // press is harmless, but a person should not be left wondering whether the first took.
      yes.disabled = true;
      yes.textContent = "…";
      void post(`/fix/${encodeURIComponent(n.key)}`);
    };
    box.append(yes);
    setup.append(box);
  }
}

stream.addEventListener("needs", (e) => drawNeeds(JSON.parse((e as MessageEvent).data) as Missing[]));

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
  talk.classList.toggle("listening", (e as MessageEvent).data === "1");
});

input.focus();
