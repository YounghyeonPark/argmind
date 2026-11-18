// Draws the five screens `--bin looking` asks about: for each html/<screen>.html, a picture as it is
// (<screen>.png), the same picture with every listed widget numbered (<screen>-marked.png), and
// what a list would say (<screen>.json: the window, the listed widgets in order with their
// rectangles, and the rectangle of every goal's target, listed or not).
//
//   node measure/looking/draw.mjs
//
// Headless Edge over its DevTools socket, 1280×800 at a device scale of 1, so a CSS pixel is a
// picture pixel and a rectangle needs no conversion. Run once; the pictures are committed, because a
// font update would otherwise move every box under the harness.
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const SCREENS = ["explorer", "dialog", "paint", "charmap", "remote"];
const edge = "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe";
const debug = 9400 + Math.floor(Math.random() * 400);
const profile = mkdtempSync(join(tmpdir(), "argmind-looking-"));
const proc = spawn(edge, ["--headless=new", `--remote-debugging-port=${debug}`, `--user-data-dir=${profile}`,
  "--window-size=1280,800", "--no-first-run", "--hide-scrollbars", "--force-device-scale-factor=1",
  "--allow-file-access-from-files", "about:blank"], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let targets;
for (let i = 0; i < 50 && !targets; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${debug}/json/list`)).json(); } catch { await sleep(200); }
}
const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((r) => (ws.onopen = r));
let id = 0;
const waiting = new Map();
ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && waiting.has(d.id)) { waiting.get(d.id)(d); waiting.delete(d.id); } };
const send = (method, params = {}) => new Promise((res) => { const i = ++id; waiting.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
const run = async (expression) => (await send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true })).result?.result?.value;

await send("Page.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1280, height: 800, deviceScaleFactor: 1, mobile: false });

const READ = `(async () => {
  await document.fonts.ready;
  const box = (e) => { const b = e.getBoundingClientRect();
    return [Math.round(b.left), Math.round(b.top), Math.round(b.width), Math.round(b.height)]; };
  const widgets = [...document.querySelectorAll("[data-name]")].map((e) => ({
    name: e.dataset.name, role: e.dataset.role, rect: box(e), target: e.dataset.target || null }));
  const targets = {};
  for (const e of document.querySelectorAll("[data-target]")) targets[e.dataset.target] = box(e);
  return { window: document.title, widgets, targets };
})()`;

// A number on every listed widget: a 2-pixel outline and a tag at its top-left, inside the box so
// a tag never sits over the neighbour it does not belong to.
const MARK = `(() => {
  const all = [...document.querySelectorAll("[data-name]")];
  all.forEach((e, i) => {
    const b = e.getBoundingClientRect();
    const o = document.createElement("div");
    o.style.cssText = "position:fixed;pointer-events:none;z-index:99999;border:2px solid #ff00aa;" +
      "left:" + b.left + "px;top:" + b.top + "px;width:" + b.width + "px;height:" + b.height + "px";
    const t = document.createElement("span");
    t.textContent = String(i + 1);
    t.style.cssText = "position:absolute;left:0;top:0;background:#ffe600;color:#000;font:bold 11px/13px Segoe UI;" +
      "padding:0 3px;border:1px solid #000";
    o.appendChild(t);
    document.body.appendChild(o);
  });
  return all.length;
})()`;

for (const screen of SCREENS) {
  await send("Page.navigate", { url: pathToFileURL(join(here, "html", `${screen}.html`)).href });
  await sleep(1500);
  const read = await run(READ);
  const plain = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(here, `${screen}.png`), Buffer.from(plain.result.data, "base64"));
  const marked = await run(MARK);
  await sleep(200);
  const shot = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(join(here, `${screen}-marked.png`), Buffer.from(shot.result.data, "base64"));
  writeFileSync(join(here, `${screen}.json`), JSON.stringify(read, null, 1) + "\n");
  console.log(`${screen}: ${read.widgets.length} listed, ${Object.keys(read.targets).length} targets, ${marked} marked`);
}
ws.close();
proc.kill();
