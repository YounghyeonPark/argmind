//! **작업 화면** — a Windows desktop of argmind's own, for work that has no API and no script: the
//! third way in, since 2026-11-08.
//!
//! The owner: *사용자의 컴퓨터 사용과 충돌하지 않게 별도의 샌드박스 창에서 작업을 수행하는것 고려*,
//! then *작업 수행시 접근 우선순위는 mcp또는 api, 코드또는 스크립트 작성및 실행, 그래도 안되면
//! 샌드박스작업 수행*, and *설치된건 아무거나 띄워도 됨*. So this is ranked last
//! (`task::Rank::Screen`), and `examples/hidden-desktop.rs` is the probe it rests on: a desktop
//! made with `CreateDesktopW` is one the person never sees, a program started there with
//! `lpDesktop` draws there, and UI Automation reads and operates it **through control patterns
//! only** — `SendInput` reaches the desktop the person is on, which is the one thing this must not
//! touch. The probe read Character Map in 40 ms, toggled, set a value, invoked, and left the
//! foreground and the physical pointer where they were, three runs of three.
//!
//! **The model picks a number; the worker turns it into the element** — `Pressing`'s rule: a look
//! numbers what can be read or pressed, with no rectangle, and a press or a write forgets the
//! numbers, since what they named may have gone.
//!
//! **What it refuses, and why.** A program already running somewhere: a single-instance program
//! — Word — joins the running one on whichever desktop that is, so argmind's document would open
//! on the person's screen or theirs on argmind's. A packaged launcher (Notepad, Paint, Calculator
//! on Windows 11): the stub exits and the real window belongs to another process, which may well
//! open on the person's desktop — not tried, because trying it is the harm. A password box. **And
//! there are no keys**: `Enter` or `Ctrl+S` would be input, so a save is the 저장 button, and text
//! goes in by `ValuePattern`, which *replaces* a box's value.
//!
//! UI Automation objects are apartment-bound, and a thread is on one desktop, so everything runs
//! on one worker thread that owns the desktop, the programs and the last look.
//!
//! **And a person can watch it — since 2026-11-08, the owner's 작업화면 보여주는 칸.** The worker
//! draws the work screen's windows into a [`View`] after every job and once a second while idle,
//! with `PrintWindow` from the worker's own thread, which is on that desktop. **The picture is for
//! the person and never for a model**: no tool result carries it, `screen_look` stays the numbered
//! text it was, and `argmind-rules` holds that one route in the application reads it. It is
//! argmind's desktop, not the person's — the worker is not on the person's screen and cannot draw it.

use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::{Duration, Instant};

use argmind_core::task::{ToolSpec, Tools};
use serde_json::{json, Value};
use windows::core::{Interface, BSTR, PCWSTR, PWSTR};
use windows::Win32::Foundation::{CloseHandle, BOOL, HANDLE, HWND, LPARAM};
use windows::Win32::System::Com::{CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED};
use windows::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Process32FirstW, Process32NextW,
                                                    PROCESSENTRY32W, TH32CS_SNAPPROCESS};
use windows::Win32::System::StationsAndDesktops::{CloseDesktop, CreateDesktopW, EnumDesktopWindows, SetThreadDesktop,
                                                  DESKTOP_CONTROL_FLAGS, HDESK};
use windows::Win32::System::JobObjects::AssignProcessToJobObject;
use windows::Win32::System::Threading::{CreateProcessW, ResumeThread, TerminateProcess, WaitForSingleObject,
                                        CREATE_SUSPENDED, PROCESS_INFORMATION, STARTUPINFOW};
use windows::Win32::UI::Accessibility::*;
use windows::Win32::UI::WindowsAndMessaging::{GetWindowTextW, IsWindowVisible};

/// Things one look shows, most — `task::MOST_PRESSABLE`, for the same reason.
pub const MOST_SHOWN: usize = argmind_core::task::MOST_PRESSABLE;

/// Waiting for a started program's window. Word took seconds in the probe; a first start more.
pub const WINDOW_SECONDS: u64 = 40;

/// Waiting for the worker to answer one request. A pattern call that never returns — a modal
/// dialog some programs open inside `Invoke` — is reported rather than waited on for ever.
pub const ANSWER_SECONDS: u64 = 30;

/// Launchers whose window is somebody else's process — on Windows 11 these are packaged apps.
const PACKAGED: [&str; 6] = ["notepad.exe", "mspaint.exe", "calc.exe", "snippingtool.exe", "wt.exe", "explorer.exe"];

/// How often an idle work screen is drawn again — a program that is loading changes by itself.
pub const REDRAWN: Duration = Duration::from_millis(1000);

/// The widest picture handed out: enough to read a program's buttons by eye, about a megabyte.
pub const MOST_WIDE: usize = 800;

/// What the work screen looks like now, for a person.
#[derive(Clone)]
pub struct Picture {
    /// Goes up only when the picture changed, so a page that has this one asks for nothing more.
    pub seq: u64,
    /// The windows' titles, front first.
    pub titles: Vec<String>,
    /// A complete `.bmp` file.
    pub bmp: Vec<u8>,
}

/// Shared between the worker that draws and whoever shows it. `None` while nothing is open.
pub type View = std::sync::Arc<std::sync::Mutex<Option<Picture>>>;

type Reply = Sender<Result<String, String>>;

/// **What the person decided about acting, asked before every press and every write** — the
/// window's title, the kind (`click` or `type`, `rule::ACTIONS`'s words) and what it is in the
/// person's words. `Err` is a refusal and its sentence becomes the tool's result. The application
/// answers it with the person's 규칙 and 확인 없이 실행, and may take as long as a person does:
/// it runs on the task's thread, not the worker's, so no `ANSWER_SECONDS` applies to it.
pub type Gate = Box<dyn FnMut(&str, &str, &str) -> Result<(), String> + Send>;

enum Job {
    Open(String),
    Look,
    /// The window and the name of a numbered thing, for the gate.
    Which(usize),
    Press(usize),
    Type(usize, String),
    Close,
}

/// The tools. Nothing is made until `screen_open`; dropping it closes what it opened.
#[derive(Default)]
pub struct Screen {
    worker: Option<(Sender<(Job, Reply)>, std::thread::JoinHandle<()>)>,
    /// Processes started, as raw handles, so a stuck worker's programs can still be ended here.
    started: std::sync::Arc<std::sync::Mutex<Vec<isize>>>,
    view: View,
    gate: Option<Gate>,
}

impl Screen {
    pub fn new() -> Self {
        Self::default()
    }

    /// The same, drawing into `view` for somebody to watch.
    pub fn shown(view: View) -> Self {
        Self { worker: None, started: Default::default(), view, gate: None }
    }

    /// Every press and write asks `gate` first.
    pub fn gated(mut self, gate: Gate) -> Self {
        self.gate = Some(gate);
        self
    }

    /// The gate's answer for item `i`, or `Ok` with no gate.
    fn passes(&mut self, i: usize, kind: &str, text: Option<&str>) -> Result<(), String> {
        if self.gate.is_none() {
            return Ok(());
        }
        let which = self.ask(Job::Which(i))?;
        let (window, name) = which.split_once('\t').unwrap_or(("", &which));
        let what = match text {
            Some(t) => format!("「{name}」에 \"{}\" 넣기", t.chars().take(60).collect::<String>()),
            None => format!("「{name}」 누르기"),
        };
        (self.gate.as_mut().unwrap())(window, kind, &what)
    }

    fn ask(&mut self, job: Job) -> Result<String, String> {
        if self.worker.is_none() {
            let (tx, rx) = channel::<(Job, Reply)>();
            let started = std::sync::Arc::clone(&self.started);
            let view = std::sync::Arc::clone(&self.view);
            let handle = std::thread::spawn(move || work(rx, started, view));
            self.worker = Some((tx, handle));
        }
        let (tx, _) = self.worker.as_ref().unwrap();
        let (reply, answer) = channel();
        tx.send((job, reply)).map_err(|_| "작업 화면이 닫혔다".to_string())?;
        answer.recv_timeout(Duration::from_secs(ANSWER_SECONDS))
            .unwrap_or_else(|_| Err(format!("작업 화면이 {ANSWER_SECONDS}초 안에 답하지 않았다 — 대화상자가 떠서 막혔을 수 있다")))
    }
}

impl Drop for Screen {
    fn drop(&mut self) {
        if let Some((tx, handle)) = self.worker.take() {
            let (reply, answer) = channel();
            if tx.send((Job::Close, reply)).is_ok() && answer.recv_timeout(Duration::from_secs(10)).is_ok() {
                let _ = handle.join();
                return;
            }
        }
        //: The worker did not answer: end what it started from here.
        if let Ok(s) = self.started.lock() {
            for &h in s.iter() {
                unsafe { let _ = TerminateProcess(HANDLE(h as *mut _), 0); }
            }
        }
    }
}

impl Tools for Screen {
    fn list(&self) -> Vec<ToolSpec> {
        let t = |name: &str, does: &str, args: Value| ToolSpec { name: name.into(), does: does.into(), args };
        vec![
            t("screen_open", "사람 화면과 따로 떨어진 작업 화면에서 설치된 프로그램을 연다 (실행 파일 이름이나 경로). \
사람의 마우스·키보드·창은 건드리지 않는다", json!({"program": "charmap"})),
            t("screen_look", "작업 화면의 창에서 읽거나 누를 수 있는 것을 번호와 함께 본다. 누르기 전에 먼저 본다", json!({})),
            t("screen_press", "screen_look 이 보여 준 번호의 것을 누른다 (단추, 확인란, 탭, 목록 항목, 메뉴). 키보드는 없다 — \
저장도 저장 단추로 한다", json!({"item": 3})),
            t("screen_type", "번호의 입력칸 값을 이 글로 바꾼다 (덧붙이지 않고 통째로 바꾼다)", json!({"item": 2, "text": "A-1042"})),
            t("screen_close", "작업 화면의 프로그램을 닫는다. 저장하지 않은 것은 사라진다", json!({})),
        ]
    }
    fn call(&mut self, tool: &str, args: &Value) -> Result<String, String> {
        let item = || args.get("item").and_then(|v| v.as_u64().or_else(|| v.as_str()?.trim().parse().ok()))
            .map(|n| n as usize).ok_or_else(|| "item 은 screen_look 의 번호다".to_string());
        match tool {
            "screen_open" => {
                let p = args.get("program").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty())
                    .ok_or("program 이 없다")?;
                self.ask(Job::Open(p.to_string()))
            }
            "screen_look" => self.ask(Job::Look),
            "screen_press" => {
                let i = item()?;
                self.passes(i, "click", None)?;
                self.ask(Job::Press(i))
            }
            "screen_type" => {
                let text = args.get("text").and_then(Value::as_str).ok_or("text 가 없다")?.to_string();
                let i = item()?;
                self.passes(i, "type", Some(&text))?;
                self.ask(Job::Type(i, text))
            }
            "screen_close" => {
                if self.worker.is_none() {
                    return Ok("열린 작업 화면이 없다".into());
                }
                let r = self.ask(Job::Close);
                if let Some((_, h)) = self.worker.take() {
                    let _ = h.join();
                }
                r
            }
            _ => Err(format!("그런 도구는 없다: {tool}")),
        }
    }
}

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain([0]).collect()
}

/// The executable a command line starts, and its file name, lower case.
fn image_of(program: &str) -> (String, String) {
    let p = program.trim();
    let exe = if let Some(rest) = p.strip_prefix('"') {
        rest.split('"').next().unwrap_or("").to_string()
    } else if std::path::Path::new(p).is_file() {
        p.to_string()
    } else if let Some(i) = p.to_ascii_lowercase().find(".exe") {
        p[..i + 4].to_string()
    } else {
        p.split_whitespace().next().unwrap_or("").to_string()
    };
    let mut name = std::path::Path::new(&exe).file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    if !name.contains('.') {
        name.push_str(".exe");
    }
    (exe, name)
}

/// Process ids running this image, anywhere.
fn running(image: &str) -> Vec<u32> {
    let mut out = Vec::new();
    unsafe {
        let Ok(snap) = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) else { return out };
        let mut e = PROCESSENTRY32W { dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32, ..Default::default() };
        let mut ok = Process32FirstW(snap, &mut e).is_ok();
        while ok {
            let n = String::from_utf16_lossy(&e.szExeFile[..e.szExeFile.iter().position(|&c| c == 0).unwrap_or(0)]);
            if n.eq_ignore_ascii_case(image) {
                out.push(e.th32ProcessID);
            }
            ok = Process32NextW(snap, &mut e).is_ok();
        }
        let _ = CloseHandle(snap);
    }
    out
}

fn windows_on(desk: HDESK) -> Vec<HWND> {
    unsafe extern "system" fn each(h: HWND, l: LPARAM) -> BOOL {
        let v = &mut *(l.0 as *mut Vec<HWND>);
        if IsWindowVisible(h).as_bool() {
            v.push(h);
        }
        BOOL(1)
    }
    let mut v: Vec<HWND> = Vec::new();
    unsafe { let _ = EnumDesktopWindows(desk, Some(each), LPARAM(&mut v as *mut Vec<HWND> as isize)); }
    v
}

fn title(h: HWND) -> String {
    let mut b = [0u16; 256];
    let n = unsafe { GetWindowTextW(h, &mut b) }.max(0) as usize;
    String::from_utf16_lossy(&b[..n])
}

struct Worker {
    desk: HDESK,
    ua: IUIAutomation,
    procs: Vec<PROCESS_INFORMATION>,
    seen: Vec<(IUIAutomationElement, String)>,
}

fn work(jobs: Receiver<(Job, Reply)>, started: std::sync::Arc<std::sync::Mutex<Vec<isize>>>, view: View) {
    static MADE: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let name = format!("argmind-work-{}-{}", std::process::id(), MADE.fetch_add(1, std::sync::atomic::Ordering::SeqCst));
    let made = (|| -> Result<Worker, String> {
        let n = wide(&name);
        let desk = unsafe { CreateDesktopW(PCWSTR(n.as_ptr()), None, None, DESKTOP_CONTROL_FLAGS(0), 0x1000_0000, None) }
            .map_err(|e| format!("작업 화면을 만들지 못했다 — {e}"))?;
        unsafe { SetThreadDesktop(desk) }.map_err(|e| format!("작업 화면에 붙지 못했다 — {e}"))?;
        unsafe { let _ = CoInitializeEx(None, COINIT_MULTITHREADED); }
        let ua: IUIAutomation = unsafe { CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER) }
            .map_err(|e| e.to_string())?;
        Ok(Worker { desk, ua, procs: Vec::new(), seen: Vec::new() })
    })();
    let mut w = match made {
        Ok(w) => w,
        Err(why) => {
            for (_, reply) in jobs.iter() {
                let _ = reply.send(Err(why.clone()));
            }
            return;
        }
    };
    loop {
        let (job, reply) = match jobs.recv_timeout(REDRAWN) {
            Ok(j) => j,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                draw(w.desk, &view);
                continue;
            }
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        };
        let closing = matches!(job, Job::Close);
        let got = match job {
            Job::Open(p) => w.open(&p, &name, &started),
            Job::Look => w.look(),
            Job::Which(i) => w.element(i).map(|e| {
                let name = unsafe { e.CurrentName() }.map(|s| s.to_string()).unwrap_or_default();
                format!("{}\t{name}", w.seen[i].1)
            }),
            Job::Press(i) => w.press(i),
            Job::Type(i, t) => w.type_(i, &t),
            Job::Close => w.close(),
        };
        let _ = reply.send(got);
        if closing {
            break;
        }
        draw(w.desk, &view);
    }
    let _ = w.close();
    if let Ok(mut v) = view.lock() {
        *v = None;
    }
    unsafe { let _ = CloseDesktop(w.desk); }
}

/// The work screen's titled windows drawn where they stand, back to front, into `view` — or
/// `None` when there are none. The sequence number moves only when the picture did.
fn draw(desk: HDESK, view: &View) {
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Gdi::*;
    use windows::Win32::Storage::Xps::{PrintWindow, PRINT_WINDOW_FLAGS};
    use windows::Win32::UI::WindowsAndMessaging::GetWindowRect;
    let wins: Vec<(HWND, String, RECT)> = windows_on(desk).into_iter().filter_map(|h| {
        let t = title(h);
        let mut r = RECT::default();
        (!t.trim().is_empty() && unsafe { GetWindowRect(h, &mut r) }.is_ok() && r.right > r.left && r.bottom > r.top)
            .then_some((h, t, r))
    }).take(4).collect();
    let made = (|| -> Option<Picture> {
        let (l, t) = (wins.iter().map(|w| w.2.left).min()?, wins.iter().map(|w| w.2.top).min()?);
        let wide = (wins.iter().map(|w| w.2.right).max()? - l).min(4000);
        let tall = (wins.iter().map(|w| w.2.bottom).max()? - t).min(4000);
        let mut pixels = vec![0u8; (wide * tall * 4) as usize];
        unsafe {
            let screen = GetDC(None);
            let canvas = CreateCompatibleDC(screen);
            let whole = CreateCompatibleBitmap(screen, wide, tall);
            let old = SelectObject(canvas, whole);
            //: Back to front: `EnumDesktopWindows` lists the front one first.
            for (h, _, r) in wins.iter().rev() {
                let (w_, h_) = (r.right - r.left, r.bottom - r.top);
                let one = CreateCompatibleDC(screen);
                let bmp = CreateCompatibleBitmap(screen, w_, h_);
                let was = SelectObject(one, bmp);
                if PrintWindow(*h, one, PRINT_WINDOW_FLAGS(2)).as_bool() {
                    let _ = BitBlt(canvas, r.left - l, r.top - t, w_, h_, one, 0, 0, SRCCOPY);
                }
                SelectObject(one, was);
                let _ = DeleteObject(bmp);
                let _ = DeleteDC(one);
            }
            SelectObject(canvas, old);
            let mut info = BITMAPINFO { bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32, biWidth: wide, biHeight: -tall,
                biPlanes: 1, biBitCount: 32, biCompression: BI_RGB.0, ..Default::default() }, ..Default::default() };
            let got = GetDIBits(canvas, whole, 0, tall as u32, Some(pixels.as_mut_ptr() as *mut _), &mut info,
                                DIB_RGB_COLORS);
            let _ = DeleteObject(whole);
            let _ = DeleteDC(canvas);
            ReleaseDC(None, screen);
            if got == 0 {
                return None;
            }
        }
        let (small, sw, sh) = reduced(&pixels, wide as usize, tall as usize, MOST_WIDE);
        Some(Picture { seq: 0, titles: wins.iter().map(|w| w.1.clone()).collect(), bmp: bmp_file(&small, sw, sh) })
    })();
    if let Ok(mut v) = view.lock() {
        *v = match (made, v.take()) {
            (None, _) => None,
            (Some(p), Some(was)) if p.bmp == was.bmp && p.titles == was.titles => Some(was),
            (Some(mut p), _) => {
                //: One count for the whole process, so a later task's first picture is never
                //: mistaken for an earlier one the page already has.
                static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
                p.seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Some(p)
            }
        };
    }
}

/// Top-down BGRA made at most `most` wide by averaging boxes — read by eye, so a box filter does.
fn reduced(bgra: &[u8], wide: usize, tall: usize, most: usize) -> (Vec<u8>, usize, usize) {
    if wide <= most {
        return (bgra.to_vec(), wide, tall);
    }
    let (nw, nh) = (most, (tall * most / wide).max(1));
    let mut out = vec![0u8; nw * nh * 4];
    for y in 0..nh {
        let (y0, y1) = (y * tall / nh, ((y + 1) * tall / nh).max(y * tall / nh + 1));
        for x in 0..nw {
            let (x0, x1) = (x * wide / nw, ((x + 1) * wide / nw).max(x * wide / nw + 1));
            let mut sum = [0u32; 3];
            for yy in y0..y1 {
                for xx in x0..x1 {
                    let i = (yy * wide + xx) * 4;
                    for (c, s) in sum.iter_mut().enumerate() {
                        *s += bgra[i + c] as u32;
                    }
                }
            }
            let n = ((y1 - y0) * (x1 - x0)) as u32;
            let o = (y * nw + x) * 4;
            for (c, s) in sum.iter().enumerate() {
                out[o + c] = (s / n) as u8;
            }
            out[o + 3] = 255;
        }
    }
    (out, nw, nh)
}

/// A 24-bit `.bmp` from top-down BGRA — every browser shows one and it needs no encoder.
fn bmp_file(bgra: &[u8], wide: usize, tall: usize) -> Vec<u8> {
    let row = (wide * 3).div_ceil(4) * 4;
    let size = 54 + row * tall;
    let mut f = Vec::with_capacity(size);
    f.extend_from_slice(b"BM");
    f.extend_from_slice(&(size as u32).to_le_bytes());
    f.extend_from_slice(&[0; 4]);
    f.extend_from_slice(&54u32.to_le_bytes());
    f.extend_from_slice(&40u32.to_le_bytes());
    f.extend_from_slice(&(wide as i32).to_le_bytes());
    f.extend_from_slice(&(tall as i32).to_le_bytes());
    f.extend_from_slice(&1u16.to_le_bytes());
    f.extend_from_slice(&24u16.to_le_bytes());
    f.extend_from_slice(&[0; 24]);
    //: Bottom-up, as a positive height says.
    for y in (0..tall).rev() {
        for x in 0..wide {
            let i = (y * wide + x) * 4;
            f.extend_from_slice(&bgra[i..i + 3]);
        }
        f.resize(f.len() + row - wide * 3, 0);
    }
    f
}

impl Worker {
    fn open(&mut self, program: &str, desk_name: &str, started: &std::sync::Mutex<Vec<isize>>) -> Result<String, String> {
        let (exe, image) = image_of(program);
        if PACKAGED.contains(&image.as_str()) || exe.to_ascii_lowercase().contains("\\windowsapps\\") {
            return Err(format!("{image} 은 창을 다른 프로세스가 띄우는 앱이라 사람 화면에 뜰 수 있어 작업 화면에서 열지 않는다"));
        }
        let ours: Vec<u32> = self.procs.iter().map(|p| p.dwProcessId).collect();
        if running(&image).iter().any(|pid| !ours.contains(pid)) {
            return Err(format!("{image} 이(가) 이미 켜져 있다 — 한 번에 하나만 뜨는 프로그램이면 사람 화면의 것과 섞이므로 \
작업 화면에서 열지 않는다. 사람이 닫은 뒤에 다시 하거나 다른 방법을 써라"));
        }
        let before = windows_on(self.desk).len();
        let mut where_ = wide(&format!("WinSta0\\{desk_name}"));
        let mut cmd = wide(program);
        let si = STARTUPINFOW { cb: std::mem::size_of::<STARTUPINFOW>() as u32, lpDesktop: PWSTR(where_.as_mut_ptr()),
                                ..Default::default() };
        let mut pi = PROCESS_INFORMATION::default();
        //: **Started suspended and put in argmind's job before it runs**, so it dies with argmind
        //: however argmind ends — a killed harness once left three Character Maps running on a
        //: desktop nobody could see. Suspended so a child it starts joins the job too.
        unsafe { CreateProcessW(None, PWSTR(cmd.as_mut_ptr()), None, None, false, CREATE_SUSPENDED,
                                None, None, &si, &mut pi) }
            .map_err(|e| format!("{program} 을(를) 시작하지 못했다 — {e}"))?;
        if let Some(job) = crate::supervise::ours() {
            unsafe { let _ = AssignProcessToJobObject(job, pi.hProcess); }
        }
        unsafe { ResumeThread(pi.hThread); }
        if let Ok(mut s) = started.lock() {
            s.push(pi.hProcess.0 as isize);
        }
        self.procs.push(pi);
        let until = Instant::now() + Duration::from_secs(WINDOW_SECONDS);
        while Instant::now() < until {
            std::thread::sleep(Duration::from_millis(200));
            let now = windows_on(self.desk);
            if now.len() > before {
                //: A moment for the window to fill in before anybody reads it.
                std::thread::sleep(Duration::from_millis(500));
                let titles: Vec<String> = windows_on(self.desk).iter().map(|&h| title(h)).filter(|t| !t.is_empty()).collect();
                self.seen.clear();
                return Ok(format!("작업 화면에 열었다: {}. screen_look 으로 보라", titles.join(", ")));
            }
            if unsafe { WaitForSingleObject(pi.hProcess, 0) }.0 == 0 {
                return Err(format!("{image} 이(가) 창을 띄우지 않고 끝났다"));
            }
        }
        Err(format!("{image} 의 창이 {WINDOW_SECONDS}초 안에 작업 화면에 뜨지 않았다"))
    }

    fn look(&mut self) -> Result<String, String> {
        let wins = windows_on(self.desk);
        if wins.is_empty() {
            return Err("작업 화면에 열린 창이 없다 — screen_open 으로 먼저 연다".into());
        }
        self.seen.clear();
        let mut lines = Vec::new();
        let mut titles = Vec::new();
        let all_cond = unsafe { self.ua.CreateTrueCondition() }.map_err(|e| e.to_string())?;
        for &h in wins.iter().take(4) {
            //: Untitled windows are the input method's own; listed by nothing.
            let t = title(h);
            if t.trim().is_empty() {
                continue;
            }
            titles.push(format!("'{t}'"));
            let Ok(root) = (unsafe { self.ua.ElementFromHandle(h) }) else { continue };
            let Ok(found) = (unsafe { root.FindAll(TreeScope_Descendants, &all_cond) }) else { continue };
            for i in 0..unsafe { found.Length() }.unwrap_or(0) {
                if self.seen.len() >= MOST_SHOWN {
                    break;
                }
                let Ok(e) = (unsafe { found.GetElement(i) }) else { continue };
                let kind = unsafe { e.CurrentControlType() }.unwrap_or_default();
                let shown = [UIA_ButtonControlTypeId, UIA_CheckBoxControlTypeId, UIA_RadioButtonControlTypeId,
                             UIA_EditControlTypeId, UIA_ComboBoxControlTypeId, UIA_ListItemControlTypeId,
                             UIA_MenuItemControlTypeId, UIA_TabItemControlTypeId, UIA_HyperlinkControlTypeId,
                             UIA_TreeItemControlTypeId, UIA_DocumentControlTypeId, UIA_TextControlTypeId,
                             UIA_SplitButtonControlTypeId, UIA_DataItemControlTypeId];
                if !shown.contains(&kind) {
                    continue;
                }
                let name = unsafe { e.CurrentName() }.map(|s| s.to_string()).unwrap_or_default();
                let value = value_of(&e);
                let typed = [UIA_EditControlTypeId, UIA_DocumentControlTypeId, UIA_ComboBoxControlTypeId].contains(&kind);
                if name.trim().is_empty() && !typed {
                    continue;
                }
                let role = unsafe { e.CurrentLocalizedControlType() }.map(|s| s.to_string()).unwrap_or_default();
                let off = if unsafe { e.CurrentIsEnabled() }.map(|b| b.as_bool()).unwrap_or(true) { "" } else { " (꺼짐)" };
                let v = match value {
                    Some(v) if !v.is_empty() && v != name => format!(" = \"{}\"", v.chars().take(120).collect::<String>()),
                    _ => String::new(),
                };
                lines.push(format!("{}: {role} \"{}\"{v}{off}", self.seen.len(), name.chars().take(80).collect::<String>()));
                self.seen.push((e, t.clone()));
            }
        }
        Ok(format!("작업 화면의 창: {}\n(번호: 종류 \"이름\" = 값)\n{}", titles.join(", "),
                   if lines.is_empty() { "(읽을 수 있는 것이 없다)".into() } else { lines.join("\n") }))
    }

    fn element(&self, i: usize) -> Result<IUIAutomationElement, String> {
        self.seen.get(i).map(|s| s.0.clone()).ok_or_else(|| if self.seen.is_empty() {
            "먼저 screen_look 으로 보라 — 번호는 마지막으로 본 목록의 것이다".to_string()
        } else {
            format!("{i} 번은 마지막 목록에 없다 (0–{})", self.seen.len() - 1)
        })
    }

    fn press(&mut self, i: usize) -> Result<String, String> {
        let e = self.element(i)?;
        self.seen.clear();
        let name = unsafe { e.CurrentName() }.map(|s| s.to_string()).unwrap_or_default();
        let how = unsafe {
            if let Ok(p) = e.GetCurrentPattern(UIA_InvokePatternId).and_then(|p| p.cast::<IUIAutomationInvokePattern>()) {
                p.Invoke().map(|_| "눌렀다")
            } else if let Ok(p) = e.GetCurrentPattern(UIA_TogglePatternId).and_then(|p| p.cast::<IUIAutomationTogglePattern>()) {
                p.Toggle().map(|_| "바꿨다")
            } else if let Ok(p) = e.GetCurrentPattern(UIA_SelectionItemPatternId).and_then(|p| p.cast::<IUIAutomationSelectionItemPattern>()) {
                p.Select().map(|_| "골랐다")
            } else if let Ok(p) = e.GetCurrentPattern(UIA_ExpandCollapsePatternId).and_then(|p| p.cast::<IUIAutomationExpandCollapsePattern>()) {
                p.Expand().map(|_| "펼쳤다")
            } else if let Ok(p) = e.GetCurrentPattern(UIA_LegacyIAccessiblePatternId).and_then(|p| p.cast::<IUIAutomationLegacyIAccessiblePattern>()) {
                p.DoDefaultAction().map(|_| "기본 동작을 했다")
            } else {
                return Err(format!("\"{name}\" 은(는) 누를 방법이 없다"));
            }
        }.map_err(|err| format!("\"{name}\" 을(를) 누르지 못했다 — {err}"))?;
        std::thread::sleep(Duration::from_millis(600));
        Ok(format!("\"{name}\": {how}. 화면이 바뀌었을 수 있으니 screen_look 으로 다시 보라"))
    }

    fn type_(&mut self, i: usize, text: &str) -> Result<String, String> {
        let e = self.element(i)?;
        self.seen.clear();
        let name = unsafe { e.CurrentName() }.map(|s| s.to_string()).unwrap_or_default();
        //: **A password is the person's to type** — `Hands::into_a_secret`, here too.
        if unsafe { e.CurrentIsPassword() }.map(|b| b.as_bool()).unwrap_or(false) {
            return Err("비밀번호 칸이다 — 비밀번호는 사람이 넣는다".into());
        }
        let v = unsafe { e.GetCurrentPattern(UIA_ValuePatternId) }.and_then(|p| p.cast::<IUIAutomationValuePattern>())
            .map_err(|_| format!("\"{name}\" 은(는) 글을 넣을 수 있는 칸이 아니다"))?;
        if unsafe { v.CurrentIsReadOnly() }.map(|b| b.as_bool()).unwrap_or(false) {
            return Err(format!("\"{name}\" 은(는) 읽기 전용이다"));
        }
        unsafe { v.SetValue(&BSTR::from(text)) }.map_err(|err| format!("\"{name}\" 에 넣지 못했다 — {err}"))?;
        std::thread::sleep(Duration::from_millis(200));
        let now = unsafe { v.CurrentValue() }.map(|s| s.to_string()).unwrap_or_default();
        Ok(format!("\"{name}\" 의 값이 이제 \"{}\"", now.chars().take(200).collect::<String>()))
    }

    fn close(&mut self) -> Result<String, String> {
        self.seen.clear();
        let mut closed = 0;
        for h in windows_on(self.desk) {
            if let Ok(root) = unsafe { self.ua.ElementFromHandle(h) } {
                let w = unsafe { root.GetCurrentPattern(UIA_WindowPatternId) }.and_then(|p| p.cast::<IUIAutomationWindowPattern>());
                if let Ok(w) = w {
                    if unsafe { w.Close() }.is_ok() {
                        closed += 1;
                    }
                }
            }
        }
        let mut ended = 0;
        for p in self.procs.drain(..) {
            unsafe {
                if WaitForSingleObject(p.hProcess, 3000).0 != 0 {
                    let _ = TerminateProcess(p.hProcess, 0);
                    ended += 1;
                }
                let _ = CloseHandle(p.hProcess);
                let _ = CloseHandle(p.hThread);
            }
        }
        Ok(format!("작업 화면을 닫았다 (창 {closed}개를 닫고, 끝나지 않은 프로그램 {ended}개를 멈췄다)"))
    }
}

fn value_of(e: &IUIAutomationElement) -> Option<String> {
    unsafe { e.GetCurrentPattern(UIA_ValuePatternId) }.and_then(|p| p.cast::<IUIAutomationValuePattern>()).ok()
        .and_then(|v| unsafe { v.CurrentValue() }.ok()).map(|s| s.to_string().trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_picture_is_made_narrower_by_averaging_and_written_as_a_bmp() {
        //: 4×2, the left half black and the right half white.
        let mut px = Vec::new();
        for _ in 0..2 {
            for x in 0..4 {
                let c = if x < 2 { 0 } else { 255 };
                px.extend_from_slice(&[c, c, c, 255]);
            }
        }
        let (small, w, h) = reduced(&px, 4, 2, 2);
        assert_eq!((w, h), (2, 1));
        assert_eq!(&small[..3], &[0, 0, 0]);
        assert_eq!(&small[4..7], &[255, 255, 255]);
        let (same, w, _) = reduced(&px, 4, 2, 800);
        assert_eq!((w, same.len()), (4, px.len()));
        //: 3 wide is 9 bytes a row, padded to 12.
        let f = bmp_file(&[10u8; 3 * 2 * 4], 3, 2);
        assert_eq!(&f[..2], b"BM");
        assert_eq!(f.len(), 54 + 12 * 2);
        assert_eq!(u32::from_le_bytes(f[2..6].try_into().unwrap()) as usize, f.len());
    }

    #[test]
    fn the_program_a_command_line_starts_is_read() {
        assert_eq!(image_of("charmap").1, "charmap.exe");
        assert_eq!(image_of("\"C:\\Program Files\\Office\\WINWORD.EXE\" /n /q").1, "winword.exe");
        assert_eq!(image_of("C:\\Program Files\\Office\\WINWORD.EXE /n").1, "winword.exe");
        assert_eq!(image_of("C:\\Windows\\System32\\notepad.exe").1, "notepad.exe");
    }

    #[test]
    fn a_launcher_and_a_program_already_running_are_refused_before_anything_starts() {
        let mut s = Screen::new();
        let r = s.call("screen_open", &json!({"program": "notepad"}));
        assert!(r.as_ref().is_err_and(|e| e.contains("사람 화면에 뜰 수")), "{r:?}");
        //: This test binary is running, so it is a program already running.
        let me = std::env::current_exe().unwrap().display().to_string();
        let r = s.call("screen_open", &json!({"program": me}));
        assert!(r.as_ref().is_err_and(|e| e.contains("이미 켜져 있다")), "{r:?}");
        assert!(s.call("screen_look", &json!({})).is_err());
    }
}
