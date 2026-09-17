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
import { orb } from "./orb.js";
const token = document.body.dataset.token ?? "";
/** Every request carries the token. Localhost is not a permission: any program you run is on it. */
async function post(path, body = "") {
    await fetch(path, {
        method: "POST",
        headers: { "X-Argmind-Token": token, "Content-Type": "text/plain; charset=utf-8" },
        body,
    });
}
const transcript = document.getElementById("transcript");
const pending = document.getElementById("pending");
const input = document.getElementById("said");
// Not `status`: `window.status` is a global string, so a bare `const status` here is a
// redeclaration the compiler rejects — and in plain JavaScript it would have silently shadowed it.
const statusLine = document.getElementById("status");
const header = document.getElementById("attending");
function line(who, text) {
    const row = document.createElement("div");
    row.className = `line ${who}`;
    // textContent, never innerHTML: the model's reply is text, and a reply containing a tag is a
    // reply, not markup. The transcript is the one place untrusted text lands.
    row.textContent = text;
    transcript.append(row);
    transcript.scrollTop = transcript.scrollHeight;
}
/** One waiting effect, with the two buttons that are the whole permission model. */
function row(p) {
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
function settle(tokenId, how) {
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
const eye = orb(document.getElementById("orb"));
// The conversation is the content. The sphere is presence, and presence recedes once there is
// something to read.
function recede() {
    eye.presence(transcript.childElementCount === 0 ? 1 : 0.4);
}
recede();
const stream = new EventSource(`/events`);
/** What the status line says about the voice. Three states that matter and one that does not:
 *  speaking with the neural voice, speaking with the Windows one, asked to speak and unable to,
 *  and not asked. */
function voiceReads(d) {
    if (!d.speaks)
        return d.voice === "none" ? "소리 안 나옴" : "소리 꺼짐";
    return d.voice === "neural" ? "소리 켜짐 (신경망)" : "소리 켜짐 (Windows)";
}
/** The last snapshot, kept so a single line of it can change without a whole new one.
 *
 *  The voice can be swapped mid-session — the neural one starts, and the agent takes it up — and
 *  the header said `소리 켜짐 (Windows)` for the rest of the session while the transcript said
 *  *이제 신경망 목소리로 말합니다* right above it. The page was not wrong; it had never been told. */
let lately = null;
function drawStatus(d) {
    // Both settings shown, because "it did nothing" and "it said nothing" are the two things
    // a person would otherwise mistake for the agent being broken.
    statusLine.textContent =
        `${d.model}${d.dry ? " · 실행 안 함" : ""} · ${voiceReads(d)}` +
            `${d.hears ? " · 마이크 켜짐" : ""}`;
}
stream.addEventListener("voice", (e) => {
    if (!lately)
        return;
    lately.voice = e.data;
    lately.speaks = lately.voice !== "off" && lately.voice !== "none";
    drawStatus(lately);
});
stream.addEventListener("hello", (e) => {
    const d = JSON.parse(e.data);
    lately = d;
    header.textContent = d.attending || "(아무 창도 보고 있지 않음)";
    drawStatus(d);
    talk.hidden = !d.hears;
    drawNoticing(d.noticing, d.noticed);
    // Rebuilt from the server's own list, so a reload is never a page showing buttons the server
    // has forgotten — or missing ones it still holds.
    pending.replaceChildren();
    for (const p of d.waiting) {
        row(p);
    }
});
for (const kind of ["them", "say", "note"]) {
    stream.addEventListener(kind, (e) => {
        line(kind, e.data);
        recede();
        //: A reply is the event the person is waiting for; what they typed is not news to them.
        if (kind === "say") {
            eye.jolt("said");
        }
    });
}
stream.addEventListener("propose", (e) => row(JSON.parse(e.data)));
stream.addEventListener("ran", (e) => {
    const d = JSON.parse(e.data);
    settle(d.token, d.confirmed === false ? `${d.label} — 눌렀지만 화면이 그대로임` : `${d.label} — 됨`);
    //: **The two outcomes look different**, because "it pressed the button and the screen did not
    //: move" is the failure this project keeps finding and it should not read as success.
    eye.jolt(d.confirmed === false ? "refused" : "acted");
});
stream.addEventListener("gone", (e) => {
    const d = JSON.parse(e.data);
    settle(d.token, `${d.what} — 그새 사라짐`);
    eye.jolt("refused");
});
stream.addEventListener("dropped", (e) => settle(Number(e.data), ""));
stream.addEventListener("attending", (e) => {
    header.textContent = e.data || "(아무 창도 보고 있지 않음)";
});
stream.addEventListener("busy", (e) => {
    const thinking = e.data === "1";
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
const form = document.getElementById("form");
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
const notice = document.getElementById("notice");
let noticedSoFar = 0;
function drawNoticing(on, count) {
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
    drawNoticing(e.data === "1", noticedSoFar);
});
stream.addEventListener("noticed", (e) => {
    noticedSoFar += 1;
    drawNoticing(true, noticedSoFar);
    // What set it off, before whatever it decided to say about it. A thought arriving with no
    // visible cause reads as the agent talking to itself.
    line("noticed", e.data);
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
const seeing = document.getElementById("seeing");
const seen = document.getElementById("seen");
const spent = document.getElementById("spent");
stream.addEventListener("seeing", (e) => {
    const text = e.data;
    seen.textContent = text;
    seeing.hidden = text.trim().length === 0;
});
stream.addEventListener("spent", (e) => {
    const d = JSON.parse(e.data);
    // Both halves apart: reading the screen and asking the model are different things, and one
    // number covering both hides which of them is slow today.
    const cut = d.widgets > d.listed ? ` (${d.widgets}개 중 ${d.listed}개만 보여줌)` : "";
    spent.textContent = `${d.widgets}개 읽기 ${Math.round(d.look_ms)} ms · `
        + `생각 ${Math.round(d.think_ms)} ms${cut}`;
});
const setup = document.getElementById("setup");
/// The four buttons, wherever a permission is being asked about.
///
/// **One builder, so the setup panel and the question above it cannot come to offer different
/// answers.** Order and spacing carry meaning: the two that are remembered sit apart from the two
/// that are not, so 항상 is not the button a finger lands on by accident.
function fourWays(key, pressed) {
    const buttons = document.createElement("div");
    buttons.className = "ask-buttons";
    for (const [label, how, cls] of [["허용", "allow", "yes"], ["항상 허용", "always", "yes always"],
        ["거부", "deny", "no"], ["항상 거부", "never", "no"]]) {
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
function drawNeeds(rows) {
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
stream.addEventListener("needs", (e) => drawNeeds(JSON.parse(e.data)));
const asking = document.getElementById("asking");
const decisions = document.getElementById("decisions");
function ask(a) {
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
stream.addEventListener("asking", (e) => ask(JSON.parse(e.data)));
function drawDecisions(rows) {
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
            d.standing === "allowed" ? "항상 허용" : d.standing === "denied" ? "항상 거부" : "아직 안 물음";
        box.append(standing);
        if (d.standing !== "") {
            const forget = document.createElement("button");
            forget.type = "button";
            forget.textContent = "되돌리기";
            forget.title = "다음에 다시 물어봅니다";
            forget.onclick = () => void post(`/forget/${encodeURIComponent(d.key)}`);
            box.append(forget);
        }
        decisions.append(box);
    }
}
stream.addEventListener("decisions", (e) => drawDecisions(JSON.parse(e.data)));
// ---- the talk key ------------------------------------------------------------------------------
//
// Push-to-talk, and the two halves are two requests. Holding is the floor an open microphone has no
// equivalent of: the agent cannot hear itself, because the device is shut while it speaks, and a
// phone call in the same room is not a turn addressed to it.
const talk = document.getElementById("talk");
let holding = false;
function hold(down) {
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
    const open = e.data === "1";
    talk.classList.toggle("listening", open);
    //: The microphone is the one state the server knows about and the page cannot infer: the key
    //: going down is a request, and this is the device actually being open.
    eye.doing(open ? "listening" : "resting");
});
input.focus();
