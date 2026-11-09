//! **Every command a task runs starts on a desktop of argmind's own** — since 2026-11-09.
//!
//! `--bin screen`'s *order* case: the first `screen_open` was refused by the owner's order of ways
//! in, so the model ran `Start-Process order-window.exe` in `run_windows` — and the window opened
//! **on the person's desktop**, in front of what they were doing, and stayed there until the harness
//! exited. `run_windows` had been called *weaker, and said so* since the day it was built; that was
//! the weakness arriving. A process draws on the desktop it was started on and its children inherit
//! it, so a command started on a desktop nobody sees cannot put a window where the person works,
//! whatever it runs. `std::process::Command` has no way to name a desktop, which is why this module
//! starts processes itself, with `CreateProcessW` and `lpDesktop`.
//!
//! **And what a command leaves running ends with it.** Each command gets a job of its own, nested in
//! argmind's, and the job is emptied when the command returns — a program `Start-Process` left
//! behind would otherwise sit invisible on that desktop until argmind exits, and the work screen
//! would then refuse to open it as *already running*. The count is returned so the result can say
//! so, and say where a program with a window belongs: `screen_open`.
//!
//! Only the three pipe ends are inherited (`PROC_THREAD_ATTRIBUTE_HANDLE_LIST`), so a command
//! cannot hold open a pipe some other thread of argmind is spawning with at the same moment.

use std::ffi::OsStr;
use std::fs::File;
use std::io::{Read, Write};
use std::os::windows::ffi::OsStrExt;
use std::os::windows::io::FromRawHandle;
use std::process::Command;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use windows::core::{PCWSTR, PWSTR};
use windows::Win32::Foundation::{CloseHandle, SetHandleInformation, HANDLE, HANDLE_FLAGS, HANDLE_FLAG_INHERIT};
use windows::Win32::Security::SECURITY_ATTRIBUTES;
use windows::Win32::System::JobObjects::{AssignProcessToJobObject, JobObjectBasicProcessIdList,
                                         QueryInformationJobObject, TerminateJobObject};
use windows::Win32::System::Pipes::CreatePipe;
use windows::Win32::System::StationsAndDesktops::{CreateDesktopW, DESKTOP_CONTROL_FLAGS};
use windows::Win32::System::Threading::{CreateProcessW, DeleteProcThreadAttributeList, GetExitCodeProcess,
                                        InitializeProcThreadAttributeList, ResumeThread, UpdateProcThreadAttribute,
                                        WaitForSingleObject, CREATE_NO_WINDOW, CREATE_SUSPENDED,
                                        CREATE_UNICODE_ENVIRONMENT, EXTENDED_STARTUPINFO_PRESENT,
                                        LPPROC_THREAD_ATTRIBUTE_LIST, PROCESS_INFORMATION,
                                        PROC_THREAD_ATTRIBUTE_HANDLE_LIST, STARTF_USESTDHANDLES, STARTUPINFOEXW};

/// What came of a command.
pub(crate) struct Ran {
    /// The exit code, or `None` when it ran past its time and was ended.
    pub code: Option<i32>,
    pub out: String,
    pub err: String,
    /// Programs the command left running, by name, ended with it.
    pub left: Vec<String>,
}

/// The desktop commands start on, as `lpDesktop` names it — made once, kept for the process's life.
pub(crate) fn desktop() -> Option<&'static str> {
    static DESK: OnceLock<Option<String>> = OnceLock::new();
    DESK.get_or_init(|| {
        let name = format!("argmind-commands-{}", std::process::id());
        let n: Vec<u16> = name.encode_utf16().chain([0]).collect();
        //: The handle is never closed: a desktop lasts while a handle to it is open, and this one
        //: is wanted for as long as argmind runs.
        unsafe { CreateDesktopW(PCWSTR(n.as_ptr()), None, None, DESKTOP_CONTROL_FLAGS(0), 0x1000_0000, None) }
            .ok().map(|_| format!("WinSta0\\{name}"))
    }).as_deref()
}

/// `arg` as one argument on a Windows command line — the rule `CommandLineToArgvW` reads.
fn push_quoted(line: &mut Vec<u16>, arg: &OsStr, always: bool) {
    let a: Vec<u16> = arg.encode_wide().collect();
    let plain = !a.is_empty() && !a.iter().any(|&c| c == ' ' as u16 || c == '\t' as u16 || c == '"' as u16);
    if plain && !always {
        line.extend(a);
        return;
    }
    line.push('"' as u16);
    let mut slashes = 0;
    for &c in &a {
        if c == '\\' as u16 {
            slashes += 1;
        } else {
            if c == '"' as u16 {
                line.extend(std::iter::repeat('\\' as u16).take(slashes + 1));
            }
            slashes = 0;
        }
        line.push(c);
    }
    line.extend(std::iter::repeat('\\' as u16).take(slashes));
    line.push('"' as u16);
}

/// The program as a path: a bare name is looked up on `PATH` — never in argmind's own folder or the
/// current one, which is where `CreateProcessW` would look first.
fn located(program: &OsStr) -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(program);
    if p.components().count() > 1 {
        return Some(p.to_path_buf());
    }
    let with_exe = if p.extension().is_some() { p.to_path_buf() } else { p.with_extension("exe") };
    std::env::var_os("PATH").and_then(|path| std::env::split_paths(&path).map(|d| d.join(&with_exe)).find(|f| f.is_file()))
}

/// The environment block when the command changes any variable, else `None` (inherit).
fn environment(c: &Command) -> Option<Vec<u16>> {
    if c.get_envs().next().is_none() {
        return None;
    }
    let mut vars: std::collections::BTreeMap<String, (std::ffi::OsString, std::ffi::OsString)> = std::env::vars_os()
        .map(|(k, v)| (k.to_string_lossy().to_uppercase(), (k, v))).collect();
    for (k, v) in c.get_envs() {
        let key = k.to_string_lossy().to_uppercase();
        match v {
            Some(v) => { vars.insert(key, (k.to_os_string(), v.to_os_string())); }
            None => { vars.remove(&key); }
        }
    }
    let mut block = Vec::new();
    for (k, v) in vars.values() {
        block.extend(k.encode_wide());
        block.push('=' as u16);
        block.extend(v.encode_wide());
        block.push(0);
    }
    block.push(0);
    Some(block)
}

/// The programs still in `job`, by name, leaving out `main` and the console hosts Windows puts in a
/// console program's job, which go when it does — counted as left behind, `findstr` read as one —
/// and any that ended between the two reads, which have no name by then.
unsafe fn left_in(job: HANDLE, main: u32) -> Vec<String> {
    let mut room = vec![0usize; 2 + 256];
    if QueryInformationJobObject(job, JobObjectBasicProcessIdList, room.as_mut_ptr() as *mut _,
                                 (room.len() * std::mem::size_of::<usize>()) as u32, None).is_err() {
        return Vec::new();
    }
    //: Two u32 counts and then the list, on x86_64: the second count, the ids listed, is the high half.
    let listed = (room[0] >> 32) as usize;
    let pids: Vec<u32> = room[1..1 + listed.min(256)].iter().map(|&p| p as u32).collect();
    let names = names();
    pids.into_iter().filter(|&p| p != main)
        .map(|p| names.get(&p).cloned().unwrap_or_default())
        .filter(|n| !n.is_empty() && !n.eq_ignore_ascii_case("conhost.exe"))
        .collect()
}

/// Every running process's image name, by id.
fn names() -> std::collections::HashMap<u32, String> {
    use windows::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, Process32FirstW, Process32NextW,
                                                        PROCESSENTRY32W, TH32CS_SNAPPROCESS};
    let mut out = std::collections::HashMap::new();
    unsafe {
        let Ok(snap) = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) else { return out };
        let mut e = PROCESSENTRY32W { dwSize: std::mem::size_of::<PROCESSENTRY32W>() as u32, ..Default::default() };
        let mut ok = Process32FirstW(snap, &mut e).is_ok();
        while ok {
            let n = String::from_utf16_lossy(&e.szExeFile[..e.szExeFile.iter().position(|&c| c == 0).unwrap_or(0)]);
            out.insert(e.th32ProcessID, n);
            ok = Process32NextW(snap, &mut e).is_ok();
        }
        let _ = CloseHandle(snap);
    }
    out
}

/// A pipe whose `keep` end stays here and is not inherited; the other end is the child's.
unsafe fn pipe(keep_read: bool) -> Result<(HANDLE, HANDLE), String> {
    let sa = SECURITY_ATTRIBUTES { nLength: std::mem::size_of::<SECURITY_ATTRIBUTES>() as u32,
                                   lpSecurityDescriptor: std::ptr::null_mut(), bInheritHandle: true.into() };
    let (mut r, mut w) = (HANDLE::default(), HANDLE::default());
    CreatePipe(&mut r, &mut w, Some(&sa), 0).map_err(|e| format!("파이프를 만들지 못했다 — {e}"))?;
    let (keep, theirs) = if keep_read { (r, w) } else { (w, r) };
    let _ = SetHandleInformation(keep, HANDLE_FLAG_INHERIT.0, HANDLE_FLAGS(0));
    Ok((keep, theirs))
}

/// Run `c` on the commands' desktop in a job of its own, feed it `stdin`, and wait at most
/// `seconds`. `Err` only when it could not be started.
pub(crate) fn run(c: &Command, stdin: Option<&str>, seconds: u64) -> Result<Ran, String> {
    let desk = desktop().ok_or("명령을 사람 화면과 떨어진 곳에서 시작할 수 없어 실행하지 않았다")?;
    let exe = located(c.get_program())
        .ok_or_else(|| format!("{} 을(를) 찾지 못했다", c.get_program().to_string_lossy()))?;
    let mut line = Vec::new();
    push_quoted(&mut line, exe.as_os_str(), true);
    for a in c.get_args() {
        line.push(' ' as u16);
        push_quoted(&mut line, a, false);
    }
    line.push(0);
    let app: Vec<u16> = exe.as_os_str().encode_wide().chain([0]).collect();
    let dir: Option<Vec<u16>> = c.get_current_dir().map(|d| d.as_os_str().encode_wide().chain([0]).collect());
    let env = environment(c);
    let mut desk_w: Vec<u16> = desk.encode_utf16().chain([0]).collect();
    unsafe {
        let (in_keep, in_theirs) = pipe(false)?;
        let (out_keep, out_theirs) = pipe(true)?;
        let (err_keep, err_theirs) = pipe(true)?;
        let theirs = [in_theirs, out_theirs, err_theirs];
        let close_theirs = || for h in theirs { let _ = CloseHandle(h); };
        //: Only these three are inherited.
        let mut size = 0usize;
        let _ = InitializeProcThreadAttributeList(LPPROC_THREAD_ATTRIBUTE_LIST(std::ptr::null_mut()), 1, 0, &mut size);
        let mut room = vec![0u8; size];
        let list = LPPROC_THREAD_ATTRIBUTE_LIST(room.as_mut_ptr() as *mut _);
        let mut made = false;
        let started = (|| -> Result<PROCESS_INFORMATION, String> {
            InitializeProcThreadAttributeList(list, 1, 0, &mut size).map_err(|e| e.to_string())?;
            made = true;
            UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST as usize,
                                      Some(theirs.as_ptr() as *const _), std::mem::size_of_val(&theirs), None, None)
                .map_err(|e| e.to_string())?;
            let mut si = STARTUPINFOEXW::default();
            si.StartupInfo.cb = std::mem::size_of::<STARTUPINFOEXW>() as u32;
            si.StartupInfo.lpDesktop = PWSTR(desk_w.as_mut_ptr());
            si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
            si.StartupInfo.hStdInput = in_theirs;
            si.StartupInfo.hStdOutput = out_theirs;
            si.StartupInfo.hStdError = err_theirs;
            si.lpAttributeList = list;
            let mut pi = PROCESS_INFORMATION::default();
            let mut cmd = line.clone();
            CreateProcessW(PCWSTR(app.as_ptr()), PWSTR(cmd.as_mut_ptr()), None, None, true,
                           CREATE_NO_WINDOW | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT,
                           env.as_ref().map(|b| b.as_ptr() as *const _),
                           dir.as_ref().map(|d| PCWSTR(d.as_ptr())).unwrap_or(PCWSTR::null()),
                           &si.StartupInfo, &mut pi)
                .map_err(|e| format!("{} 을(를) 시작하지 못했다 — {e}", exe.display()))?;
            Ok(pi)
        })();
        if made {
            DeleteProcThreadAttributeList(list);
        }
        close_theirs();
        let mut input = File::from_raw_handle(in_keep.0 as _);
        let out = File::from_raw_handle(out_keep.0 as _);
        let err = File::from_raw_handle(err_keep.0 as _);
        let pi = started?;
        //: argmind's job first, then the command's own: an empty job assigned second nests inside
        //: whatever the process is already in.
        if let Some(job) = crate::supervise::ours() {
            let _ = AssignProcessToJobObject(job, pi.hProcess);
        }
        let own = crate::supervise::kill_on_close().ok();
        if let Some(j) = own {
            let _ = AssignProcessToJobObject(j, pi.hProcess);
        }
        ResumeThread(pi.hThread);
        let _ = CloseHandle(pi.hThread);
        if let Some(text) = stdin {
            let _ = input.write_all(text.as_bytes());
            let _ = input.write_all(b"\n");
        }
        drop(input);
        let read = |mut f: File| std::thread::spawn(move || { let mut v = Vec::new(); let _ = f.read_to_end(&mut v); v });
        let (a, b) = (read(out), read(err));
        let began = Instant::now();
        let mut timed_out = false;
        while WaitForSingleObject(pi.hProcess, 50).0 != 0 {
            if began.elapsed() > Duration::from_secs(seconds) {
                timed_out = true;
                break;
            }
        }
        let mut left = Vec::new();
        if let Some(j) = own {
            left = left_in(j, pi.dwProcessId);
            let _ = TerminateJobObject(j, 1);
            let _ = CloseHandle(j);
        }
        //: The job may not have taken it — a process already in a job that forbids nesting — so a
        //: command past its time is ended by its own handle as well.
        if timed_out {
            let _ = windows::Win32::System::Threading::TerminateProcess(pi.hProcess, 1);
            let _ = WaitForSingleObject(pi.hProcess, 2000);
        }
        let mut code = 0u32;
        let _ = GetExitCodeProcess(pi.hProcess, &mut code);
        let _ = CloseHandle(pi.hProcess);
        let out = String::from_utf8_lossy(&a.join().unwrap_or_default()).to_string();
        let err = String::from_utf8_lossy(&b.join().unwrap_or_default()).to_string();
        Ok(Ran { code: if timed_out { None } else { Some(code as i32) }, out, err, left })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_argument_is_quoted_as_command_line_to_argv_reads_it() {
        let q = |s: &str| { let mut v = Vec::new(); push_quoted(&mut v, OsStr::new(s), false); String::from_utf16(&v).unwrap() };
        assert_eq!(q("plain"), "plain");
        assert_eq!(q("two words"), "\"two words\"");
        assert_eq!(q(""), "\"\"");
        assert_eq!(q("say \"hi\""), "\"say \\\"hi\\\"\"");
        assert_eq!(q("C:\\a b\\"), "\"C:\\a b\\\\\"");
    }

    #[test]
    fn input_the_environment_an_exit_code_and_a_time_limit_still_work() {
        let r = run(Command::new("findstr").arg("x"), Some("axb"), 20).unwrap();
        assert_eq!((r.code, r.out.trim(), r.left.len()), (Some(0), "axb", 0), "{:?}", r.left);
        let r = run(Command::new("cmd").args(["/c", "echo %ARGMIND_APART%& exit 3"]).env("ARGMIND_APART", "예"), None, 20).unwrap();
        assert_eq!(r.code, Some(3));
        assert!(!r.out.contains("%ARGMIND_APART%"), "{}", r.out);
        let began = Instant::now();
        let r = run(Command::new("powershell").args(["-NoProfile", "-Command", "Start-Sleep -Seconds 20"]), None, 1).unwrap();
        assert_eq!(r.code, None);
        assert!(began.elapsed() < Duration::from_secs(10));
        assert!(run(&Command::new("no-such-program-argmind"), None, 5).is_err_and(|e| e.contains("찾지 못했다")));
    }

    /// **The case it exists for**: `Start-Process winver` from a command, while this thread — on
    /// the desktop the test was started on, the person's — looks for a window of `winver.exe` the
    /// whole time. None appears, and the program is gone when the command returns.
    #[test]
    fn a_window_a_command_opens_never_reaches_this_desktop_and_ends_with_the_command() {
        use windows::Win32::Foundation::{BOOL, HWND, LPARAM};
        use windows::Win32::UI::WindowsAndMessaging::{EnumWindows, GetWindowThreadProcessId, IsWindowVisible};
        let before = crate::screen::running("winver.exe");
        let job = std::thread::spawn(|| run(Command::new("powershell").args(["-NoProfile", "-Command",
            "Start-Process winver; Start-Sleep -Seconds 3"]), None, 30));
        unsafe extern "system" fn each(h: HWND, l: LPARAM) -> BOOL {
            let found = &mut *(l.0 as *mut Vec<u32>);
            let mut pid = 0;
            GetWindowThreadProcessId(h, Some(&mut pid));
            if IsWindowVisible(h).as_bool() {
                found.push(pid);
            }
            true.into()
        }
        let mut seen_here = false;
        let mut seen_running = false;
        while !job.is_finished() {
            let theirs: Vec<u32> = crate::screen::running("winver.exe").into_iter().filter(|p| !before.contains(p)).collect();
            seen_running |= !theirs.is_empty();
            let mut shown = Vec::<u32>::new();
            unsafe { let _ = EnumWindows(Some(each), LPARAM(&mut shown as *mut _ as isize)); }
            seen_here |= shown.iter().any(|p| theirs.contains(p));
            std::thread::sleep(Duration::from_millis(100));
        }
        let r = job.join().unwrap().unwrap();
        assert!(seen_running, "winver never ran, so nothing was tested: {:?} {}", r.code, r.err);
        assert!(!seen_here, "a window of the command's winver was on this desktop");
        assert!(r.left.iter().any(|n| n.eq_ignore_ascii_case("winver.exe")), "{:?}", r.left);
        let after: Vec<u32> = crate::screen::running("winver.exe").into_iter().filter(|p| !before.contains(p)).collect();
        assert!(after.is_empty(), "winver outlived the command: {after:?}");
    }
}
