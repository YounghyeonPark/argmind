// SPDX-License-Identifier: Apache-2.0
//
// **The page in the language a person chose — since 2026-11-17**, the owner's *인터페이스언어도
// 국가별로 설정할수 있게 해줘*.
//
// The page and the server both write Korean, as they always have. What changes is what reaches the
// screen: every text node and every `title`, `placeholder` and `alt` the page shows is looked up in
// one table (`words.ts`) the moment it appears, and the Korean it came from is remembered so a
// change of language can write the whole page again. **Nothing that writes the page had to change
// to be translated**, so nothing written later can forget to be — it is either in the table or it
// stays Korean, and `native/argmind-serve/tests/interface.rs` counts which.
//
// **What a person, a model or a screen wrote is left alone**: the conversation, a window's title, a
// button's name inside a row, the model's own note, an errand's result. Those carry
// `translate="no"` or sit under one of the selectors in `KEEP` — the same attribute a browser's own
// translator honours, for the same reason. And a hole in a template is copied as it came unless the
// table marks it `{n~}`, because *클릭 “저장”* names a button that is really called 저장, and
// rendering it *Click “Save”* would send somebody looking for a button that is not there.
import { WORDS } from "./words.js";
/** What the page offers, in the language each is written in. */
export const SHOWN = [["ko", "한국어"], ["en", "English"], ["zh", "中文"], ["ja", "日本語"]];
const COLUMN = { ko: 0, en: 1, zh: 2, ja: 3 };
const exact = new Map();
const templates = [];
// `{n}` takes anything and copies it; `{n~}` takes anything and translates it too; `{n#}` takes
// only a number, so a short template like `{0#}회` cannot swallow a sentence that ends in 회.
const HOLE = /\{(\d)([~#]?)\}/g;
const HAS_HOLE = /\{\d[~#]?\}/;
for (const row of WORDS) {
    const ko = row[0];
    if (!HAS_HOLE.test(ko)) {
        exact.set(ko, row);
        continue;
    }
    const holes = [];
    let source = "";
    let at = 0;
    let weight = 0;
    for (const m of ko.matchAll(HOLE)) {
        const lit = ko.slice(at, m.index);
        weight += lit.length;
        source += lit.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + (m[2] === "#" ? "(-?[\\d.,]+)" : "([\\s\\S]+?)");
        holes.push({ n: Number(m[1]), deep: m[2] === "~" });
        at = (m.index ?? 0) + m[0].length;
    }
    const tail = ko.slice(at);
    weight += tail.length;
    source += tail.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    templates.push({ re: new RegExp(`^${source}$`), holes, row, weight });
}
templates.sort((a, b) => b.weight - a.weight);
const HANGUL = /[가-힣]/;
/** The text in `code`, or the text itself when the table has nothing for it. */
export function translate(text, code) {
    if (code === "ko" || !HANGUL.test(text))
        return text;
    const [, lead = "", core = "", trail = ""] = /^(\s*)([\s\S]*?)(\s*)$/.exec(text) ?? [];
    return lead + inner(core, code) + trail;
}
function inner(core, code) {
    if (!HANGUL.test(core))
        return core;
    const col = COLUMN[code];
    const hit = exact.get(core);
    if (hit)
        return hit[col];
    for (const t of templates) {
        const m = t.re.exec(core);
        if (!m)
            continue;
        const got = new Map();
        t.holes.forEach((h, i) => {
            const v = m[i + 1] ?? "";
            got.set(h.n, h.deep ? translate(v, code) : v);
        });
        return t.row[col].replace(/\{(\d)\}/g, (_, n) => got.get(Number(n)) ?? "");
    }
    //: **A list of lines is translated line by line** — the setup panel's plans are, and the server
    //: writes each line from its own sentence.
    if (core.includes("\n"))
        return core.split("\n").map((l) => translate(l, code)).join("\n");
    //: **A bullet, and a line of parts joined by a middle dot** — the status line, the camera line,
    //: a task card — are each translated part by part.
    if (core.startsWith("· "))
        return "· " + translate(core.slice(2), code);
    if (core.includes(" · "))
        return core.split(" · ").map((p) => translate(p, code)).join(" · ");
    return core;
}
// ---- the page --------------------------------------------------------------------------------
/** Where nothing is translated, because a person, a model or a screen wrote it. */
const KEEP = [
    '[translate="no"]', "style", "script", "#seen", "#whatissent", "#subtitle",
    "#transcript .line:not(.note)", ".errand .what", ".errand .body", ".task .what", ".kept .text",
    ".thought .text",
].join(", ");
const ATTRS = ["title", "placeholder", "alt", "aria-label"];
let now = "ko";
/** The Korean each text node was given, and what was written over it. */
const said = new WeakMap();
const wrote = new WeakMap();
const saidAttr = new WeakMap();
function kept(n) {
    const el = n.nodeType === Node.ELEMENT_NODE ? n : n.parentElement;
    return !el || el.closest(KEEP) !== null;
}
function onText(n, again) {
    if (kept(n))
        return;
    const data = n.data;
    if (!again && wrote.get(n) === data)
        return;
    const source = again ? (said.get(n) ?? data) : data;
    said.set(n, source);
    const out = translate(source, now);
    wrote.set(n, out);
    if (out !== data)
        n.data = out;
}
function onAttr(el, name, again) {
    if (kept(el))
        return;
    const value = el.getAttribute(name);
    if (value === null)
        return;
    let map = saidAttr.get(el);
    if (!map)
        saidAttr.set(el, (map = new Map()));
    const before = map.get(name);
    if (!again && before && before[1] === value)
        return;
    const source = again && before ? before[0] : value;
    const out = translate(source, now);
    map.set(name, [source, out]);
    if (out !== value)
        el.setAttribute(name, out);
}
function walk(root, again) {
    if (root.nodeType === Node.TEXT_NODE) {
        onText(root, again);
        return;
    }
    if (root.nodeType !== Node.ELEMENT_NODE && root.nodeType !== Node.DOCUMENT_NODE)
        return;
    if (root.nodeType === Node.ELEMENT_NODE)
        for (const a of ATTRS)
            onAttr(root, a, again);
    const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
    for (let n = w.nextNode(); n; n = w.nextNode()) {
        if (n.nodeType === Node.TEXT_NODE)
            onText(n, again);
        else
            for (const a of ATTRS)
                onAttr(n, a, again);
    }
}
/** The page's language now. */
export function current() {
    return now;
}
/** A sentence the page writes outside the document — `confirm()` — in the language now. */
export function say(korean) {
    return translate(korean, now);
}
/** Write the whole page in `code`, and everything put on it from now on. */
export function switchTo(code) {
    now = code;
    document.documentElement.lang = code;
    //: The one word drawn by a stylesheet rather than a node: ` · 생각 중` after the status.
    document.body.style.setProperty("--thinking", JSON.stringify(` · ${translate("생각 중", code)}`));
    walk(document.body, true);
}
/** **The language to start in**: the one chosen, or the system's when it is one of ours. */
export function starting(chosen) {
    const known = (c) => SHOWN.some(([k]) => k === c);
    if (known(chosen))
        return chosen;
    const system = (navigator.languages?.[0] ?? navigator.language ?? "").slice(0, 2).toLowerCase();
    return known(system) ? system : "ko";
}
/** Start translating, and keep doing so as the page changes. */
export function watch(code) {
    switchTo(code);
    new MutationObserver((records) => {
        for (const r of records) {
            if (r.type === "characterData")
                onText(r.target, false);
            else if (r.type === "attributes" && r.attributeName)
                onAttr(r.target, r.attributeName, false);
            else
                r.addedNodes.forEach((n) => walk(n, false));
        }
    }).observe(document.body, {
        subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ATTRS,
    });
}
