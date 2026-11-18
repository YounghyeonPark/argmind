# Looking at the screen — fast when asked properly, and not more exact than the list

*2026-11-18. `native/argmind-measure/src/bin/looking.rs`, screens and goals in `measure/looking/`.*

The owner, reading the public page that sold *it reads the screen instead of looking at a picture*:
*장점이 아닌거같음 보면서 속도를 빠르게 할수는 없을까* — it does not look like an advantage; can it
look and be fast? And: is Jev, or an open model like it, useful?

## The old number measured arithmetic

**24–82 s** for a vision model to give a pixel was taken with Ollama's `qwen3-vl:8b`, which is the
**thinking** build (renderer `qwen3-vl-thinking`, the same digest as `8b-thinking`), asked for a
pixel. Its log was 13,000 characters of `row × row height`. Run again on these screens the same way,
as the control: **59.2–63.0 s, 10,093–10,907 characters of thinking, 0 of 3 right** (two ran out,
one wrong). Predicted ≥ 20 s before the run; so these screens are not easier than the old one, and
the speeds below do not flatter for that reason.

Qwen3-VL is trained to answer a box in 0–1000 coordinates, and there is an **instruct** build that
answers without the essay (`qwen3-vl:8b-instruct`, 6.2 GB; `4b-instruct`, 3.3 GB; both Apache-2.0).
Nobody here had asked it the question it was trained on. A one-goal pilot confirmed the convention
before anything was scored: *저장해* → `[430, 491, 507, 527]`, centre (600, 407), inside 저장.

## What was asked

Five screens **drawn** in HTML and rendered by headless Edge at 1280×800 (`draw.mjs`), so every
answer's box is known to the pixel and no picture holds anything from this machine: a file list
with 32 rows of 19 px, a save dialog, a paint toolbar with two icons that have no name, a Character
Map with three 열기 buttons and its grid listed as one pane (as the real one is), and a remote
desktop whose contents are pixels. 26 requests, written before any model saw a screen: 13 the list
can do, 2 where two listed widgets share a name, 6 the list cannot see, 5 that nothing on the screen
does. Three repeats at temperature 0, worst counted — and **the repeats were identical in every
arm**, so this is one answer per goal, not three.

## Results

Measured, warm, wall time around one call with the picture uploaded; 1,172 prompt tokens a picture.

| | named, unique (13) | same name (2) | not in the list (6) | nothing does it, declined (5) | wrong, of 26 | median | slowest |
|---|---|---|---|---|---|---|---|
| **list** — `qwen3:8b`, the shipped prompt | **12** | 0 (clicked the combo box) | 0 — 4 declined, 2 clicked the pane | 3 | 7 | **0.47 s** | 0.88 s |
| **box** — `qwen3-vl:8b-instruct` | 9 | 1 | **5** | **5** | 6 | **0.59 s** | 1.83 s |
| **box** — `qwen3-vl:4b-instruct` | 9 | 1 | **6** | 2 | 8 | 0.40 s | 0.88 s |
| **marks** — 8b-instruct, numbered picture | 9 | 1 | 0 — 4 wrong numbers | 5 | 6 | 0.21 s | 1.17 s |
| **marks** — 4b-instruct | 5 | 2 | 0 — 5 wrong numbers | 3 | 14 | 0.16 s | 0.60 s |
| **ocr** — Windows.Media.Ocr, Korean | text found inside its target: **5 of 16** (remote 2 of 3) | | | | | 0.088 s a screen | 0.60 s (the file list) |
| **control** — `qwen3-vl:8b` thinking, asked for a pixel, 3 goals | 0 of 3 | | | | | 61.2 s | 63.0 s |

Against the bars written first:

- **box, 8b**: *fast enough to look on every turn* (≤ 1.5 s median, ≤ 3 s slowest) — **met**.
  *Sure enough to press without the list* (≥ 12 of 13, ≥ 4 of 5 declined, ≤ 1 wrong) — **not met**,
  9 of 13 and 6 wrong. *Useful where the list is blind* (≥ 5 of 6 and 2 of 2) — **not met by the
  same-name half**: 5 of 6, 1 of 2.
- **box, 4b**: fast — met. Without the list — not met, and it declined only 2 of 5 impossible
  requests (*인쇄해줘* pressed 저장, *휴지통 비우기* pressed 삭제). Where the list is blind — 6 of 6, 1 of 2,
  not met.
- **marks**: not met either size. It does not decline when the target has no number — it gives a
  wrong one — and it was no better than the list on what the list can do.
- **ocr**: fast — met; 5 of 16 — not met.
- **control**: met — the old way is slow here too.

## Why the box missed what the list gets

- **Dense rows, by a row.** All three file names: 보고서_최종_v3 → v3_수정 (the centre 3 px into the next
  row), 예산표_2026 → 2026_백업 (7 px), 회의록_1104 → 1117 (8b, two rows) or 1103 (4b). On 19-px rows
  the box is about a row out. The list got two of these three by name.
- **One sentence read wrongly**: *저장하지 말고 닫아줘* → 취소, both 8b arms. The list got it.
- **An icon confused**: 톱니바퀴 → the 회전 tool.
- **The same-name miss is defensible**: *문자 집합 목록 열어줘* pressed the 문자 집합 combo box, which in the
  real Character Map also opens that list. The list arm made the same choice for both same-name goals.
  The answer key was written first and accepts only the 열기 button, so it stands as scored — but the
  same-name column understates every arm, the list's included.

Where the list was blind, the box was not: ★ inside the grid, 지금 설치 / 나중에 / 휴지통 inside the
remote desktop, and the unnamed 휴지통 icon — 5 of 6 for 8b, 6 of 6 for 4b.

## OCR

**It runs from an ordinary unpackaged process** — Microsoft's page says the API needs package
identity, and this is the test of that — with Korean installed here, at 0.045–0.60 s a screen. But
on 12–13 px Korean UI text it misreads: 회의록 → *회의를*, 열기 → *별기*, every **2026 → 2025**, underscores
dropped, and 취소, 연결 끊기 and the white 휴지통 label not read at all. Upscaling the picture first was
not tried.

## What it means

- **Looking is fast now** — 0.4–0.6 s, the same order as the list's 0.47 s. The page's old comparison
  is withdrawn (commit 170d320).
- **Looking does not replace the list.** Where the list has the thing by name, it is more exact (12
  against 9 of 13), and the picture's misses are the dangerous kind: the row next to the right one.
- **Looking does what the list cannot**: 5–6 of 6 where the list sees nothing. Inside a remote
  desktop, a grid, an unnamed icon. The bar for that was not met as written, because of a same-name
  answer that is arguably right.
- So the shape that fits is **the list first, the picture only when the list cannot answer** — and
  that is the rule this project hangs on, *a model never emits a coordinate*, giving way inside one
  container. That is the owner's decision, not this measurement's.
- **The card is the cost nobody measured**: argmind's work model, voice and recogniser already use
  12.6 GB of 16 (measured 2026-11-16); an 8B vision model wants about 6 GB more and the 4B about
  3.5 GB (estimated from the file sizes).

## Jev and the open models like it

From a web search the same day (primary pages where marked; no independent evaluation of any of
them was found):

- **Jev** is TypeSafe AI's: a cloud API on a waitlist. Every decision would send the screen and the
  conversation off the machine. Not usable here.
- **They do not look.** Laya, Kev and openjev take text: a state, a question and up to about twenty
  options, and return probabilities. None of them makes looking faster.
- **Laya** (Apache-2.0, 322–421M, ~33–40 ms a question on a T4, from its card): about 20 options and
  1,024 tokens, against argmind's 12–200 widgets; zero-shot it scores *below* a majority-class
  baseline (0.36 against 0.46, its own card), and 0.49 on Korean intent. Its makers call it *a base to
  specialise, not a zero-shot decision engine*. A fine-tuned narrow gate — *is this work?*, today
  0.15 s on qwen3:8b — is the only place it could fit, for about 0.1 s a turn.
- **Kev-9B** needs 21.9 GB resident (its card) and a PyTorch or MLX server; **openjev** is MIT but
  Python, and its author reports one adversarial line taking it from 0.83 to 0.47; **Clef-flash** is
  9B in BF16; **d1-3B**'s licence bars commercial use above a revenue line. None fits the rules here.
- **argmind already has Jev's idea**: `action_sureness` reads the probability of the action's tokens
  from qwen3:8b — AUC 0.78, 15 of 16 right at or above 0.94 on held-out pages (2026-10-21).

The models that would make *looking* better are the grounding specialists: **UI-Venus-1.5-8B**
(Apache-2.0, Qwen3-VL based, ScreenSpot-Pro 68.4 on its card) and Holo2-8B (58.9), against about 57
for Qwen3-VL-8B. Whether either loads in Ollama was not checked.

## Not established

- Drawn screens, 26 goals, one answer each (the repeats were identical). Real Windows at another
  scale, dark mode, or a 1920×1080 screen were not asked.
- Whether a crop of the region the list points at, sent larger, fixes the off-by-a-row misses.
- Whether the vision model fits on the card beside argmind's other models, and what a turn costs
  when it has to be loaded.
