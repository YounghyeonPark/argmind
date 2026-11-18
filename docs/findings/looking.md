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
  container. That was the owner's decision, not this measurement's, and **the owner took it the
  same day** (below).
- The card was the cost nobody had measured; it is measured below.

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

## The owner's choice, and what was built for it

*1번* — the owner, the same evening: **try the picture only where the list is blind, always with a
press, and measure the card first.** What that became:

- **A point inside a named place, not a coordinate.** `argmind_core::intent::Inside` is thousandths
  of a container's width and height. The container is still found by name and role at the press,
  like every target; the point is placed in its fresh rectangle, so it moves with the container and
  cannot leave it. `Hands::click_inside` is `false` by default, so an environment that has not said
  it can press a point presses nothing. On Windows it is `click_point`: the reach guard, then a
  message to the window under the point — no pattern, because a pattern operates the container,
  and the middle of a remote desktop is not the button inside it. `argmind-core/tests/inside.rs`.
- **When to look, and what to keep** — `argmind-serve/src/looking.rs`:
  - it looks only when the person asked for a press, there is a *blind place* on the screen (a
    listed region of at least 128×96 with nothing listed inside), and the turn chose no click, or
    chose that place;
  - the picture is of that place, cut out and made 1280 wide;
  - the point is kept only when the smallest listed thing under it is a region. A point on a listed
    button or row is the list's to answer and is dropped.
- **Always a press.** `propose` counts a point inside among what must wait for a person, whatever
  *Act without confirming* says.
- **One maker and one call.** `argmind-rules`, `a_picture_points_only_where_the_list_is_blind.rs`:
  the picture is asked for and shown to a model once (`app::look_inside`, called by one turn), and
  a point is made only in `looking.rs` and never in the core.
  - Checked by breaking it four ways: a second `look_at(`, a `window_picture(` in `route.rs`, an
    `inside: Some(` in `converse.rs`, and `must_confirm` without the clause.
  - The first `route.rs` break was written into a doc comment and checked nothing. Written as code,
    it was caught.
- **Off until asked.** `looking` is a decidable key of its own and `vision-model` a need of its
  own, and the window that asks prints the measured numbers.

## Fitting on the card — `--fit`

Measured with argmind running as the owner runs it: `qwen3:8b`, the voice and the recogniser on a
16 GB card, 8.9 GB used before the run. A turn is the shipped prompt; a look is one box. Bars written
first: both models stay `100% GPU`, a turn after a look ≤ 1.5 s, a warm look ≤ 3 s.

| | beside the conversation | beside the conversation and argmind's own work (`qwen3:14b`, 24 layers) |
|---|---|---|
| `qwen3-vl:8b-instruct` | **fits**: 15.8 of 16.4 GB, turns 0.38–0.54 s, looks 0.59–0.66 s | **does not**: each call put another model out, 9.4 s and 10.4 s, until the work model was gone |
| `qwen3-vl:4b-instruct` | both stayed, looks 0.34–0.38 s; one turn at **2.57 s** fails the bar as written, with no reload (cause not established — argmind's own calls share the model) | **fits**: all three on the card, 15.9 GB, turns ≤ 0.82 s, looks ≤ 0.44 s |

## The shipped path on the drawn screens — `--arms gated`

The whole path, run as the application runs it: the list's turn, the look when `wants_a_look` says
so, `landed`, and a row. Bars written first:
- listed ≥ 12 of 13;
- unlisted ≥ 3 of the 4 reachable (★, and three things inside the remote desktop; the two paint
  icons have no listed container in this drawing);
- a row on ≤ 1 of 5 impossible requests;
- ≤ 1 row in the wrong place;
- overall wrong ≤ 4;
- look median ≤ 1.0 s.

| run | model | listed | unlisted | impossible, rows | rows in the wrong place | wrong | look median |
|---|---|---|---|---|---|---|---|
| 1 — the whole picture, regions from 48×32 | 4b | 12 | **4 of 4** | 1 | 1 | **5** — misses the bar by one | 0.39 s |
| | 8b | 12 | 3 (★ a pixel outside its cell) | 0 | 1 | 5 | 0.58 s |
| 2 — the place cut out, regions from 128×96 | 4b | 12 | 3 (★ answered with the grid's middle) | 1 (*계산기* → 지금 설치) | 2 | 6 | 0.67 s |
| | **8b** | 12 | **4 of 4** | **0** | **0** | **4 — every bar met** | 0.81 s |

The design changed between the two runs because of the real window below, and the bars were
written down again before the second.

## On a real window — `--example pointing`

Character Map at 150%, on a desktop of its own, a fresh window for every request, ten characters on
its first page and ★, which is not on it. The answer is read off Character Map's own status bar.
The person's pointer and foreground window were unchanged in every run, and nothing went down their
input: every press was posted.

| picture | 4b | 8b |
|---|---|---|
| the window as it is | 5 of 10 | — |
| the whole window made 1280 wide | 6 (★ declined) | — |
| **the grid cut out, made 1280 wide** | **8 and 8, ★ pressed both times** | **8 and 8, ★ declined both times** |

**Three earlier runs measured the harness instead:**
- The first read 0, 0 and 1 of 10. The example was not per-monitor aware, so the window's
  rectangle came back in scaled pixels while the picture was drawn at its real size: the picture
  held two thirds of the grid, and every point was read against a rectangle 1.5 times too small.
  The application is per-monitor aware; the example is now too.
- At 48×32, 150% made every button and label a "region".
- With one Character Map reused, a posted press sometimes left Character Map's enlarged character
  drawn over the grid. The next request then pointed at it: the 8b answered *"AE symbol"* for ß.
  **That leftover is real for the person too**, and four probe clicks did not reproduce it.

**So the application looks with the 8b**: it met every bar on the drawn screens and declined the
character that was not there. It does not fit beside argmind's own work, and a look then costs
about 10 s of reloading. A look happens only on a request to press inside a place the list cannot
see into.

## The picture beside the row — 2026-11-19

The row said *click inside "Character Grid" — the picture showed "paragraph symbol"*. That is what
the model believed, often in English, and nothing about where the press would land. The person is
the only check left between the two, so **the row now carries the picture the model was shown, with
the press point ringed on it** — under the label, on a line of its own.

- **Made after the model answered, from the same pixels**, at the point the press will use
  (`Inside::within` of the place) rather than the model's own number. 640 wide.
- **Memory only, one at a time.** It waits with its row's number and goes when the row is answered,
  turned down or cleared. The page asks `/looked/<row>` with the token every request carries.
- **`argmind-rules` holds it:** it is made once, in `look_inside`, after the one `look_at(`, and a
  model is never shown it; it is read once, by the route that writes it to the page. Checked by
  breaking it three ways: a read in `think`, `look_at` given the marked picture, and a
  `marked_picture(` in `route.rs`.

**On the real Character Map** (`--example pointing` with `POINTING_SAVE`, which now also writes the
marked picture and the place after the press): the same 8 of 10 and ★ declined as before, twice.
**Both misses, ¶ and Ø, came out wrong both times, and in both the ring sits on the character that
was actually chosen** (¦ and ì, read off the status bar and seen in the picture after the press). So
a wrong row is visible as wrong before it is pressed. Checked by eye on four of the ten: the two
misses and two hits.

**The first mark failed at what it was for**: a dot at the point hid the glyph under it. On the
Ø request the ringed cell held ì, and ì could not be read. The point itself is now left untouched:
a ring that sits round the glyph (20 px at 640 wide, where a cell is about 37) and four ticks
pointing in from outside it.

## The application's own path — `--example looking_live`, 2026-11-19

The owner said yes to running it. `argmind-serve --example looking_live` starts an argmind of its
own on port 7742 with `LOCALAPPDATA` in a scratch folder (it refuses the real one). There it writes
`looking` allowed and `notice` and `initiative` refused, and leaves `acting` unanswered. It starts a
Character Map and stops unless that is the window in front. Then it sends *© 문자 눌러줘* to `/say`,
fetches `/looked/<row>`, draws the page in headless Edge, and presses 실행 only when the row is a
point inside the grid. The owner's argmind was stopped for the run.

**Two runs failed, each on a defect only this path could show:**
- **The picture held the top left of the grid.** The server without its window was not DPI aware,
  so at 150% the rectangles were scaled and the picture was not — the failure `--example pointing`
  had in itself. `argmind-app`'s window library made the process aware; the console server and
  this example had nothing. `app::run` now makes the process per-monitor aware before anything
  reads the screen (`argmind_env_windows::per_monitor_pixels`).
- **The press came back *여기서는 칸 안의 한 점을 누를 수 없습니다*.** The application's hands are
  `surface::Touching`, which did not forward `click_inside`, so it answered the trait's *cannot*.
  **It had not forwarded `typing_lands` or `landed` either**, so the screen checks of 2026-10-21
  never ran in the application. All three are forwarded now, and `argmind-rules`
  (`the_surface_hides_no_hand`) reads the wrapper for every method the trait has.

**The third run passed every step**:
- the row came 8.2 s after the request (the look itself 0.99 s), with the program's sentence and
  the picture under the label, the grid whole and © ringed;
- 실행 pressed ©, and Character Map's status bar read *U+00A9: Copyright Sign*;
- the person's pointer was where it had been. The second run gave the row in 13.8 s, with the
  model loading.

One request, one program. The headless Edge page was English, because an unset interface language
follows the browser's.

## Not established

- **Only the one request ran through the application's path**, once each of three runs. The ten
  characters ran through the pieces (`--example pointing`).
- **No real remote desktop.** Whether a posted click reaches `mstsc`'s input window is unknown, and
  a control that reads `GetCursorPos` would see the person's pointer instead.
- Ten characters in one program, two runs; drawn screens otherwise. Dark mode, other scales and
  other programs were not asked.
- What the row says is still the vision model's word for the place (*"copyright symbol"*), often in
  English. The picture under it is what shows where. The row and its picture were seen on the
  application's own page in headless Edge, not in `argmind-app`'s window.
- An unnamed icon in a toolbar is not reachable: 128×96 keeps toolbars from being places.
- Earlier, on the drawn screens: whether a crop fixed the off-by-a-row misses on dense lists was not
  asked, because a list row is never a place, so the picture is never asked about one.
