//! **Documents: Word and Excel files made, changed and read by the task loop.**
//!
//! ```text
//!   cargo run -p argmind-measure --bin documents --release -- qwen3:14b
//!   cargo run -p argmind-measure --bin documents --release -- --control
//! ```
//!
//! From `native/`. **Stop argmind first.** The owner asked for *문서수정이나 작성*; `--bin complex`
//! measured documents only as Markdown. This asks it of `.docx` and `.xlsx`, which the workbench
//! makes in the sandbox with `python-docx` and `openpyxl` and reads back as text. Every input file is
//! made in the sandbox before the task, and **every check runs there too, with the same libraries,
//! against the file — never against the model's account.**
//!
//! ## Written before the run
//!
//! Six tasks, all written on 2026-11-06 before any model saw them, so none is fitted:
//!
//! 1. **a report**: `report.docx` has a heading-styled `분기 매출 보고` and a table with each quarter's
//!    total (6,600,000 and 7,200,000) in a row of its own.
//! 2. **an edit**: `contract.docx`'s paragraphs are exactly the expected ones — the company renamed
//!    in all three places, the amount changed, a closing paragraph added — and the title keeps its
//!    heading style.
//! 3. **a summary sheet**: `budget.xlsx` gains a `요약` sheet with each item beside its total, and
//!    `지출` is cell for cell what it was.
//! 4. **reading a sheet**: `answer.md` names 하은 and 95.
//! 5. **Word into Excel**: `todo.xlsx` has a header row and exactly the three to-dos from
//!    `minutes.docx`, each owner beside their own date — the decision and the postponed item left out.
//! 6. **a notice from nothing**: `notice.docx` has a heading and three list-styled paragraphs holding
//!    the date, the place and the fee.
//!
//! **Documents are usable** if at least **5 of 6** pass with **no false finish**, on `qwen3:14b` with
//! reasoning on as the application runs it. `--control` first: each check must pass on a
//! hand-written answer and fail on the untouched files.
//!
//! **The first run read 3 of 6 with three false finishes, and the three failures were one shape**:
//! the model never read the document it started from — a script written blind guessed `Task:` at
//! the start of the minutes' lines and the first column as the item — ran it, got nothing printed
//! and exit 0, and finished. So `run` now shows the text of every document it made or changed,
//! which is **fitted to those three**. Four more tasks were written **after that run and before the
//! change was run**, and the change stands on them:
//!
//! 7. **prices up**: every price in `prices.xlsx` 10% higher, rounded to the won, names untouched.
//! 8. **numbered headings**: `report.docx`'s three headings read `1. `, `2. `, `3. ` in order, still
//!    headings, and every body paragraph is what it was.
//! 9. **Excel into Word**: `directory.docx` has a heading per department and under it a table with
//!    exactly that department's people.
//! 10. **reading Word**: `answer.md` holds 15 and 3일 — the annual leave and how early to ask — and
//!     none of the other policies' numbers.
//!
//! The change stays if the first six read **at least 5 of 6**, the four held out **at least 3 of
//! 4**, and **no false finish** across the ten.
//!
//! **It did not: 2 of 6 and 3 of 4, three false finishes, and it was reverted.** The text was there
//! and was not read: *제2조 계약 금액은 5,000,000원* in the result of the command meant to change it,
//! and a table of one header row, each followed by a finish. Showing a model its own failure is
//! not the program checking anything. **What the failures did share is earlier**: in six of seven,
//! no document the request named was ever read before the script that changed it was written.
//! So the program reads them itself — `Tools::opening`, the calls made before the model's first
//! step, recorded as steps like any other — which is the shape of every other fix here: a question
//! about now is looked up by the program, and the model is handed the page. **Fitted to the ten**,
//! so three more were written after that run and before this one:
//!
//! 11. **deleting rows**: `stock.xlsx` keeps exactly the rows whose quantity is not 0, in order.
//! 12. **a table cell**: `schedule.docx`'s table reads exactly as before but for Wednesday's name.
//! 13. **reading a Word table**: `answer.md` holds the invoice's total, 171,500.
//!
//! The opening read stays if the ten read **at least 7 of 10**, the three **at least 2 of 3**, and
//! **no false finish** across the thirteen.
//!
//! **First run: 7 of 10, 2 of 3, two false finishes — one of them the reader's.** It printed a
//! heading as `[Heading 1] 배경`, and the script looked for paragraphs *starting* with
//! `[Heading 1]`, changed none and finished. The style now follows the text in brackets the header
//! line names as not text. **Re-run, same bar: 9 of 10, 3 of 3, one false finish** — a regular
//! expression that wrote `1. \1. 배경` — so **the bar written first is missed by one**, and the
//! opening read is kept anyway, on the comparison over tasks 1–10: **5 of 10 with three false
//! finishes without it, 9 of 10 with one**. That is a decision against a bar this file set, made
//! by its author, and is written here so it can be overruled. The thirteenth passed on the file
//! and was stopped by `why_not_done`: the script wrote 171,500 to a file and printed nothing, and
//! the finish quoted a number no result had shown — right, and unchecked, so the stop is correct.

use std::path::PathBuf;
use std::time::Instant;

use argmind_core::contract::Speaks;
use argmind_core::task::{run, Ended, Task};
use argmind_env_ollama::Local;
use argmind_env_windows::workspace::Workspace;

struct Model(Local);

impl Speaks for Model {
    fn ask(&mut self, prompt: &str) -> String {
        let (reply, why) = self.0.ask(prompt).map(|r| (r.text, r.reason))
            .unwrap_or_else(|e| (String::new(), format!("오류: {e}")));
        if std::env::var("DOCUMENTS_RAW").is_ok() {
            println!("         « {} » ({why})", reply.replace('\n', " ⏎ ").chars().take(400).collect::<String>());
        }
        reply
    }
}

struct Case {
    name: &'static str,
    what: &'static str,
    /// Plain files, written from here.
    files: &'static [(&'static str, &'static str)],
    /// Python run in the sandbox before the task, to make the documents it starts from.
    setup: &'static str,
    /// **A right answer, as Python** — the control.
    solve: &'static str,
    /// Python that prints `OK` as its last line, or why not.
    check: &'static str,
}

/// Python run in the sandbox's `/work`, with nothing left behind in the folder.
fn python(w: &Workspace, code: &str) -> Result<String, String> {
    w.run_sandboxed(&format!("python3 - <<'ARGMIND_PY'\n{code}\nARGMIND_PY"))
}

const CASES: [Case; 13] = [
    Case { name: "보고서",
           what: "sales.csv 의 분기별 매출 합계를 표로 담은 report.docx 를 만들어줘. 맨 위에 '분기 매출 보고'라는 제목을 달아줘",
           files: &[("sales.csv", "월,매출\n1,1200000\n2,2300000\n3,3100000\n4,2500000\n5,1800000\n6,2900000\n")],
           setup: "",
           solve: r#"
import docx
d = docx.Document()
d.add_heading("분기 매출 보고", 1)
t = d.add_table(rows=3, cols=2)
for r, (a, b) in enumerate([("분기", "합계"), ("1분기", "6,600,000"), ("2분기", "7,200,000")]):
    t.cell(r, 0).text, t.cell(r, 1).text = a, b
d.save("report.docx")
"#,
           check: r#"
import docx
d = docx.Document("report.docx")
heads = [p for p in d.paragraphs if "분기 매출 보고" in p.text and p.style.name.startswith(("Heading", "Title"))]
rows = [" ".join(c.text for c in r.cells).replace(",", "") for t in d.tables for r in t.rows]
missing = [v for v in ("6600000", "7200000") if not any(v in r for r in rows)]
both = [r for r in rows if "6600000" in r and "7200000" in r]
print("OK" if heads and not missing and not both else f"FAIL: 제목 {len(heads)}개, 표에 없는 합계 {missing}, 행 {rows}")
"# },
    Case { name: "계약서 고치기",
           what: "contract.docx 에서 '가나상사'를 모두 '다라상사'로 바꾸고, 계약 금액을 6,000,000원으로 고치고, 맨 끝에 '부칙: 이 계약은 2027년 1월 1일부터 효력이 있다.' 문단을 추가해줘. 나머지는 그대로 둬",
           files: &[],
           setup: r#"
import docx
d = docx.Document()
d.add_heading("용역 계약서", 1)
for t in ["갑: 가나상사", "을: 마바소프트", "제1조 가나상사는 을에게 홈페이지 제작을 의뢰한다.",
          "제2조 계약 금액은 5,000,000원으로 한다.", "제3조 가나상사는 대금을 납품 후 30일 이내에 지급한다."]:
    d.add_paragraph(t)
d.save("contract.docx")
"#,
           solve: r#"
import docx
d = docx.Document("contract.docx")
for p in d.paragraphs:
    for r in p.runs:
        r.text = r.text.replace("가나상사", "다라상사").replace("5,000,000원", "6,000,000원")
d.add_paragraph("부칙: 이 계약은 2027년 1월 1일부터 효력이 있다.")
d.save("contract.docx")
"#,
           check: r#"
import docx
d = docx.Document("contract.docx")
got = [p.text.strip() for p in d.paragraphs if p.text.strip()]
want = ["용역 계약서", "갑: 다라상사", "을: 마바소프트", "제1조 다라상사는 을에게 홈페이지 제작을 의뢰한다.",
        "제2조 계약 금액은 6,000,000원으로 한다.", "제3조 다라상사는 대금을 납품 후 30일 이내에 지급한다.",
        "부칙: 이 계약은 2027년 1월 1일부터 효력이 있다."]
title = [p for p in d.paragraphs if p.text.strip() == "용역 계약서"]
styled = bool(title) and title[0].style.name.startswith(("Heading", "Title"))
print("OK" if got == want and styled else f"FAIL: 제목 서식 {styled}, 문단 {got}")
"# },
    Case { name: "요약 시트",
           what: "budget.xlsx 의 '지출' 시트에서 항목별 금액 합계를 구해서, 같은 파일에 '요약' 시트를 새로 만들고 항목과 합계 값을 적어줘. '지출' 시트는 그대로 둬",
           files: &[],
           setup: r#"
import openpyxl
w = openpyxl.Workbook(); s = w.active; s.title = "지출"
for row in [("날짜", "항목", "금액"), ("11-01", "식비", 12000), ("11-01", "교통", 2800), ("11-02", "통신", 55000),
            ("11-03", "식비", 9000), ("11-04", "교통", 2800), ("11-05", "식비", 15500), ("11-06", "교통", 14000)]:
    s.append(row)
w.save("budget.xlsx")
"#,
           solve: r#"
import openpyxl
w = openpyxl.load_workbook("budget.xlsx"); s = w.create_sheet("요약")
for row in [("항목", "합계"), ("식비", 36500), ("교통", 19600), ("통신", 55000)]:
    s.append(row)
w.save("budget.xlsx")
"#,
           check: r#"
import openpyxl
w = openpyxl.load_workbook("budget.xlsx")
want_raw = [("날짜", "항목", "금액"), ("11-01", "식비", 12000), ("11-01", "교통", 2800), ("11-02", "통신", 55000),
            ("11-03", "식비", 9000), ("11-04", "교통", 2800), ("11-05", "식비", 15500), ("11-06", "교통", 14000)]
raw = [tuple(r) for r in w["지출"].iter_rows(values_only=True) if any(v is not None for v in r)] if "지출" in w.sheetnames else []
same = raw == want_raw
rows = [r for r in w["요약"].iter_rows(values_only=True)] if "요약" in w.sheetnames else []
def total(item):
    for r in rows:
        if item in [str(v).strip() for v in r]:
            nums = [v for v in r if isinstance(v, (int, float))]
            return nums[0] if nums else None
bad = {k: total(k) for k, v in {"식비": 36500, "교통": 19600, "통신": 55000}.items() if total(k) != v}
print("OK" if same and rows and not bad else f"FAIL: 지출 그대로 {same}, 요약 {rows}, 틀린 합계 {bad}")
"# },
    Case { name: "엑셀 읽기",
           what: "scores.xlsx 에서 세 과목 평균이 가장 높은 학생의 이름과 그 평균을 answer.md 에 적어줘",
           files: &[],
           setup: r#"
import openpyxl
w = openpyxl.Workbook(); s = w.active; s.title = "성적"
for row in [("이름", "국어", "수학", "영어"), ("민준", 88, 76, 90), ("서준", 100, 90, 92), ("하은", 90, 95, 100),
            ("지우", 70, 85, 80), ("도윤", 95, 92, 81)]:
    s.append(row)
w.save("scores.xlsx")
"#,
           solve: r#"
open("answer.md", "w").write("평균이 가장 높은 학생: 하은 (95점)\n")
"#,
           check: r#"
import os
a = open("answer.md").read() if os.path.exists("answer.md") else ""
others = [n for n in ("민준", "서준", "지우", "도윤") if n in a]
print("OK" if "하은" in a and "95" in a and not others else f"FAIL: {a!r}")
"# },
    Case { name: "회의록→할 일",
           what: "minutes.docx 회의록에서 할 일만 골라 todo.xlsx 에 담당자, 할 일, 마감일 세 열로 정리해줘. 첫 줄은 머리글로",
           files: &[],
           setup: r#"
import docx
d = docx.Document()
d.add_heading("11월 3일 주간 회의", 1)
for t in ["결정: 서비스 출시일은 12월 5일로 확정한다.", "할 일: 김민수 — 로그인 화면 디자인, 11월 15일까지",
          "할 일: 이서연 — 결제 모듈 테스트, 11월 20일까지", "보류: 마케팅 예산은 다음 회의에서 다시 논의한다.",
          "할 일: 박지훈 — 출시 공지문 작성, 12월 1일까지"]:
    d.add_paragraph(t)
d.save("minutes.docx")
"#,
           solve: r#"
import openpyxl
w = openpyxl.Workbook(); s = w.active
for row in [("담당자", "할 일", "마감일"), ("김민수", "로그인 화면 디자인", "11월 15일"),
            ("이서연", "결제 모듈 테스트", "11월 20일"), ("박지훈", "출시 공지문 작성", "12월 1일")]:
    s.append(row)
w.save("todo.xlsx")
"#,
           check: r#"
import openpyxl, os, re
if not os.path.exists("todo.xlsx"):
    print("FAIL: todo.xlsx 없음"); raise SystemExit
s = openpyxl.load_workbook("todo.xlsx").active
rows = [[("" if v is None else str(v)) for v in r] for r in s.iter_rows(values_only=True) if any(v is not None for v in r)]
head = " ".join(rows[0]) if rows else ""
header = all(k in head for k in ("담당", "할 일", "마감")) or all(k in head for k in ("담당", "할일", "마감"))
def date(cells, m, d):
    t = " ".join(cells)
    return bool(re.search(rf"\b{m}\s*월\s*{d}\s*일|-{m:02d}-{d:02d}\b|\b{m}/{d}\b|\b{m}\.\s*{d}\b", t))
want = {"김민수": (11, 15), "이서연": (11, 20), "박지훈": (12, 1)}
body = rows[1:]
ok = len(body) == 3 and all(any(n in r and date(r, m, d) for r in body) for n, (m, d) in want.items())
print("OK" if header and ok else f"FAIL: 머리글 {header}, 행 {rows}")
"# },
    Case { name: "안내문",
           what: "회사 송년회 안내문을 notice.docx 로 만들어줘. 제목을 하나 달고, 일시·장소·회비 세 가지는 글머리표 목록으로 써줘. 일시는 12월 19일 금요일 저녁 7시, 장소는 본사 12층 라운지, 회비는 없어",
           files: &[],
           setup: "",
           solve: r#"
import docx
d = docx.Document()
d.add_heading("송년회 안내", 1)
for t in ["일시: 12월 19일(금) 저녁 7시", "장소: 본사 12층 라운지", "회비: 없음"]:
    d.add_paragraph(t, style="List Bullet")
d.save("notice.docx")
"#,
           check: r#"
import docx, os
if not os.path.exists("notice.docx"):
    print("FAIL: notice.docx 없음"); raise SystemExit
d = docx.Document("notice.docx")
heads = [p.text for p in d.paragraphs if p.text.strip() and p.style.name.startswith(("Heading", "Title"))]
items = [p.text for p in d.paragraphs if p.text.strip() and "List" in p.style.name]
found = [any(all(k in t for k in ks) for t in items) for ks in (("19", "7"), ("12층", "라운지"), ("회비",))]
print("OK" if heads and len(items) == 3 and all(found) else f"FAIL: 제목 {heads}, 목록 {items}")
"# },
    Case { name: "가격 올리기",
           what: "prices.xlsx 의 모든 가격을 10% 올려서 저장해줘. 원 단위에서 반올림해",
           files: &[],
           setup: r#"
import openpyxl
w = openpyxl.Workbook(); s = w.active; s.title = "가격"
for row in [("품목", "가격"), ("아메리카노", 4500), ("라테", 5000), ("케이크", 6800), ("쿠키", 2300), ("샌드위치", 7900)]:
    s.append(row)
w.save("prices.xlsx")
"#,
           solve: r#"
import openpyxl
w = openpyxl.load_workbook("prices.xlsx"); s = w["가격"]
for r in range(2, 7):
    s.cell(r, 2).value = round(s.cell(r, 2).value * 1.1)
w.save("prices.xlsx")
"#,
           check: r#"
import openpyxl
s = openpyxl.load_workbook("prices.xlsx").worksheets[0]
got = [tuple(r) for r in s.iter_rows(values_only=True) if any(v is not None for v in r)]
want = [("품목", "가격"), ("아메리카노", 4950), ("라테", 5500), ("케이크", 7480), ("쿠키", 2530), ("샌드위치", 8690)]
fine = len(got) == len(want) and got[0] == want[0] and all(
    g[0] == w[0] and isinstance(g[1], (int, float)) and abs(g[1] - w[1]) <= 1 for g, w in zip(got[1:], want[1:]))
print("OK" if fine else f"FAIL: {got}")
"# },
    Case { name: "제목 번호",
           what: "report.docx 의 큰 제목(Heading 1) 앞에 순서대로 '1. ', '2. ', '3. ' 번호를 붙여줘. 본문은 건드리지 마",
           files: &[],
           setup: r#"
import docx
d = docx.Document()
for h, body in [("배경", "올해 고객 문의가 30% 늘었다."), ("문제", "응답까지 평균 2일이 걸린다."), ("제안", "자동 분류를 도입한다.")]:
    d.add_heading(h, 1)
    d.add_paragraph(body)
d.save("report.docx")
"#,
           solve: r#"
import docx
d = docx.Document("report.docx")
n = 0
for p in d.paragraphs:
    if p.style.name == "Heading 1":
        n += 1
        p.runs[0].text = f"{n}. " + p.runs[0].text
d.save("report.docx")
"#,
           check: r#"
import docx
d = docx.Document("report.docx")
heads = [p.text.strip() for p in d.paragraphs if p.style.name.startswith("Heading")]
body = [p.text.strip() for p in d.paragraphs if p.text.strip() and not p.style.name.startswith("Heading")]
fine = heads == ["1. 배경", "2. 문제", "3. 제안"] and body == ["올해 고객 문의가 30% 늘었다.", "응답까지 평균 2일이 걸린다.", "자동 분류를 도입한다."]
print("OK" if fine else f"FAIL: 제목 {heads}, 본문 {body}")
"# },
    Case { name: "엑셀→워드 명단",
           what: "members.xlsx 의 명단을 부서별로 나눠서 directory.docx 로 만들어줘. 부서마다 부서 이름을 제목으로 달고 그 아래에 이름과 내선번호 표를 넣어줘",
           files: &[],
           setup: r#"
import openpyxl
w = openpyxl.Workbook(); s = w.active; s.title = "명단"
for row in [("이름", "부서", "내선"), ("강하늘", "개발", 2101), ("윤소희", "디자인", 2201), ("정우진", "개발", 2102),
            ("한지민", "영업", 2301), ("오세훈", "디자인", 2202), ("임채원", "개발", 2103)]:
    s.append(row)
w.save("members.xlsx")
"#,
           solve: r#"
import docx
d = docx.Document()
for dept, people in [("개발", [("강하늘", "2101"), ("정우진", "2102"), ("임채원", "2103")]),
                     ("디자인", [("윤소희", "2201"), ("오세훈", "2202")]), ("영업", [("한지민", "2301")])]:
    d.add_heading(dept, 2)
    t = d.add_table(rows=1, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "이름", "내선"
    for n, x in people:
        c = t.add_row().cells
        c[0].text, c[1].text = n, x
d.save("directory.docx")
"#,
           check: r#"
import docx, os
from docx.table import Table
from docx.text.paragraph import Paragraph
if not os.path.exists("directory.docx"):
    print("FAIL: directory.docx 없음"); raise SystemExit
d = docx.Document("directory.docx")
want = {"개발": {"강하늘": "2101", "정우진": "2102", "임채원": "2103"}, "디자인": {"윤소희": "2201", "오세훈": "2202"}, "영업": {"한지민": "2301"}}
got, dept = {}, None
for el in d.element.body.iterchildren():
    tag = el.tag.split("}")[-1]
    if tag == "p":
        p = Paragraph(el, d)
        if p.style.name.startswith(("Heading", "Title")):
            dept = next((k for k in want if k in p.text), None)
    elif tag == "tbl" and dept:
        for r in Table(el, d).rows:
            cells = [c.text.strip() for c in r.cells]
            for n in [n for k in want for n in want[k]]:
                if n in cells:
                    got.setdefault(dept, {})[n] = next((c for c in cells if c.isdigit()), "")
print("OK" if got == want else f"FAIL: {got}")
"# },
    Case { name: "워드 읽기",
           what: "policy.docx 에서 연차 휴가가 1년에 며칠인지, 그리고 며칠 전까지 신청해야 하는지를 answer.md 에 적어줘",
           files: &[],
           setup: r#"
import docx
d = docx.Document()
d.add_heading("휴가 규정", 1)
for t in ["제1조 병가는 1년에 10일까지 쓸 수 있으며, 당일 아침까지 알린다.",
          "제2조 연차 휴가는 1년에 15일이며, 사용 3일 전까지 신청해야 한다.",
          "제3조 경조 휴가는 사유에 따라 5일까지이며, 7일 전까지 신청한다."]:
    d.add_paragraph(t)
d.save("policy.docx")
"#,
           solve: r#"
open("answer.md", "w").write("연차 휴가: 1년에 15일, 사용 3일 전까지 신청\n")
"#,
           check: r#"
import os, re
a = open("answer.md").read() if os.path.exists("answer.md") else ""
nums = set(re.findall(r"\d+", a.replace("2026", "")))
print("OK" if "15" in nums and "3" in nums and not ({"10", "5", "7"} & nums) else f"FAIL: {a!r}")
"# },
    Case { name: "행 지우기",
           what: "stock.xlsx 에서 수량이 0인 줄을 지워줘. 나머지는 순서 그대로 둬",
           files: &[],
           setup: r#"
import openpyxl
w = openpyxl.Workbook(); s = w.active; s.title = "재고"
for row in [("품목", "수량"), ("볼펜", 120), ("지우개", 0), ("공책", 45), ("자", 0), ("가위", 8), ("풀", 0)]:
    s.append(row)
w.save("stock.xlsx")
"#,
           solve: r#"
import openpyxl
w = openpyxl.load_workbook("stock.xlsx"); s = w["재고"]
for r in range(s.max_row, 1, -1):
    if s.cell(r, 2).value == 0:
        s.delete_rows(r)
w.save("stock.xlsx")
"#,
           check: r#"
import openpyxl
s = openpyxl.load_workbook("stock.xlsx").worksheets[0]
got = [tuple(r) for r in s.iter_rows(values_only=True) if any(v is not None for v in r)]
want = [("품목", "수량"), ("볼펜", 120), ("공책", 45), ("가위", 8)]
print("OK" if got == want else f"FAIL: {got}")
"# },
    Case { name: "표 칸 고치기",
           what: "schedule.docx 의 당번 표에서 수요일 담당을 '최유나'로 바꿔줘. 다른 칸은 그대로 둬",
           files: &[],
           setup: r#"
import docx
d = docx.Document()
d.add_heading("11월 둘째 주 청소 당번", 1)
t = d.add_table(rows=6, cols=2)
for r, (a, b) in enumerate([("요일", "담당"), ("월", "김도현"), ("화", "이수빈"), ("수", "박민재"), ("목", "정다은"), ("금", "한승우")]):
    t.cell(r, 0).text, t.cell(r, 1).text = a, b
d.save("schedule.docx")
"#,
           solve: r#"
import docx
d = docx.Document("schedule.docx")
for row in d.tables[0].rows:
    if row.cells[0].text == "수":
        row.cells[1].text = "최유나"
d.save("schedule.docx")
"#,
           check: r#"
import docx
d = docx.Document("schedule.docx")
got = [[c.text.strip() for c in r.cells] for t in d.tables for r in t.rows]
want = [["요일", "담당"], ["월", "김도현"], ["화", "이수빈"], ["수", "최유나"], ["목", "정다은"], ["금", "한승우"]]
print("OK" if got == want else f"FAIL: {got}")
"# },
    Case { name: "청구서 합계",
           what: "invoice.docx 의 품목 표를 보고 전체 합계 금액을 answer.md 에 적어줘",
           files: &[],
           setup: r#"
import docx
d = docx.Document()
d.add_heading("청구서 2026-118", 1)
d.add_paragraph("아래와 같이 청구합니다. 결제는 30일 이내에 부탁드립니다.")
t = d.add_table(rows=5, cols=3)
for r, row in enumerate([("품목", "수량", "단가"), ("A4 용지", "10", "4,500"), ("토너", "2", "38,000"), ("파일철", "25", "1,200"), ("택배비", "1", "20,500")]):
    for c, v in enumerate(row):
        t.cell(r, c).text = v
d.save("invoice.docx")
"#,
           solve: r#"
open("answer.md", "w").write("전체 합계: 171,500원\n")
"#,
           check: r#"
import os
a = open("answer.md").read() if os.path.exists("answer.md") else ""
print("OK" if "171500" in a.replace(",", "").replace(" ", "") else f"FAIL: {a!r}")
"# },
];

fn fresh(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("argmind-documents-{name}-{}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A workspace with a case's starting files in it.
fn prepared(case: &Case, tag: &str) -> (PathBuf, Workspace) {
    let dir = fresh(tag);
    for (f, body) in case.files {
        std::fs::write(dir.join(f), body).unwrap();
    }
    let w = Workspace::new(dir.clone());
    if !case.setup.is_empty() {
        python(&w, case.setup).expect("the starting documents are made");
    }
    (dir, w)
}

fn checked(case: &Case, w: &Workspace) -> Result<(), String> {
    //: The workbench adds the exit code as a last line of its own; the verdict is the line before.
    let said = |o: &str| o.lines().rev().find(|l| !l.trim().is_empty() && !l.starts_with("(종료 코드"))
        .unwrap_or("").to_string();
    match python(w, case.check) {
        Ok(o) if said(&o) == "OK" => Ok(()),
        Ok(o) | Err(o) => Err(said(&o)),
    }
}

fn control() {
    println!("\n  대조: 맞는 답에서 통과하고, 손대지 않은 파일에서 실패해야 한다\n");
    let mut broken = 0;
    for (i, case) in CASES.iter().enumerate() {
        let (raw_dir, raw) = prepared(case, &format!("control-raw-{i}"));
        let (ok_dir, right) = prepared(case, &format!("control-ok-{i}"));
        python(&right, case.solve).expect("the right answer runs");
        //: And the reader the model is given has to read what was made.
        let text: String = std::fs::read_dir(&ok_dir).unwrap().filter_map(Result::ok)
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|f| argmind_env_windows::workspace::is_document(f))
            .map(|f| right.read_document(&f).unwrap_or_else(|e| format!("읽지 못함: {e}"))).collect();
        let (ok, untouched) = (checked(case, &right), checked(case, &raw));
        let fine = ok.is_ok() && untouched.is_err();
        broken += !fine as usize;
        println!("    {} {}. {:<10} 맞는 답 {:?} / 손대지 않음 {:?}", if fine { "●" } else { "✗" }, i + 1, case.name,
                 ok.err().unwrap_or_else(|| "통과".into()), untouched.err().unwrap_or_else(|| "통과".into()));
        println!("         읽은 글: {}", text.replace('\n', " ⏎ ").chars().take(200).collect::<String>());
        let _ = std::fs::remove_dir_all(&raw_dir);
        let _ = std::fs::remove_dir_all(&ok_dir);
    }
    println!("\n    → 대조가 깨진 검사 {broken}개\n");
}

fn main() {
    if std::env::args().any(|a| a == "--control") {
        return control();
    }
    let models: Vec<String> = std::env::args().skip(1).filter(|a| !a.starts_with("--")).collect();
    let models = if models.is_empty() { vec!["qwen3:14b".to_string()] } else { models };
    println!("\n  문서 — Word 와 Excel 을 만들고, 고치고, 읽는다");
    println!("  harness {}\n", argmind_measure::source_print(include_str!("documents.rs")));
    for name in &models {
        let mut local = Local::new(name);
        //: As the application runs a task: reasoning on (`TASK_THINKS`).
        local.think = Some(std::env::var("DOCUMENTS_NO_THINK").is_err());
        local.keep_alive = "300s".into();
        for _ in 0..2 {
            let _ = local.ask("Reply with the single word: ready");
        }
        let mut model = Model(local);
        println!("  ── {name} ──");
        let (mut passed, mut false_done, mut secs_all, mut fitted, mut held) = (0, 0, 0.0, 0, 0);
        for (i, case) in CASES.iter().enumerate() {
            let (dir, bench) = prepared(case, &format!("{}-{i}", name.replace(':', "_")));
            let mut tools = Workspace::new(dir.clone());
            let mut task = Task::new(case.what);
            let began = Instant::now();
            run(&mut task, &mut model, &mut tools, "2026년 11월 6일", &|| true);
            let secs = began.elapsed().as_secs_f64();
            secs_all += secs;
            let verdict = checked(case, &bench);
            let said_done = matches!(task.ended, Some(Ended::Done(_)));
            passed += verdict.is_ok() as usize;
            if i < 10 { fitted += verdict.is_ok() as usize } else { held += verdict.is_ok() as usize }
            false_done += (said_done && verdict.is_err()) as usize;
            let ended = match &task.ended {
                Some(Ended::Done(_)) => "끝".to_string(),
                Some(Ended::Stopped(w)) => format!("멈춤({})", w.chars().take(40).collect::<String>()),
                None => "?".into(),
            };
            let steps: Vec<String> = task.steps.iter()
                .map(|s| format!("{}{}", s.tool, if s.failed { "✗" } else { "" })).collect();
            println!("    {} {}. {:<10} {:>2}단계 {secs:>4.0}초 {ended}{}", if verdict.is_ok() { "●" } else { "○" },
                     i + 1, case.name, task.steps.len(),
                     match &verdict { Err(why) => format!(" — {}", why.chars().take(160).collect::<String>()), Ok(()) => String::new() });
            println!("         [{}]", steps.join(" "));
            if std::env::var("DOCUMENTS_VERBOSE").is_ok() {
                for s in &task.steps {
                    println!("         · {} {} → {}", s.tool,
                             s.args.to_string().chars().take(300).collect::<String>(),
                             s.observed.replace('\n', " ⏎ ").chars().take(300).collect::<String>());
                }
            }
            let _ = std::fs::remove_dir_all(&dir);
        }
        let runs = CASES.len();
        println!("\n    → {passed}/{runs} 통과, 끝났다고 하고 틀린 것 {false_done}, 평균 {:.0}초", secs_all / runs as f64);
        println!("      앞의 열(1–10) {fitted}/10 · 따로 둔 셋(11–13) {held}/3");
        println!("    ── 돌리기 전에 적어둔 대로: {} ──\n",
                 if fitted >= 7 && held >= 2 && false_done == 0 { "문서를 다룰 수 있다 — 먼저 읽기가 남는다" }
                 else { "문서를 다룰 수 있다고 할 수 없다" });
    }
}
