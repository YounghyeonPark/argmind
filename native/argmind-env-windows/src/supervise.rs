//! A child process that cannot outlive argmind.
//!
//! **A server argmind started is argmind's to clean up.** The recogniser is a separate program
//! listening on a port; if argmind starts one and then exits, the port stays taken, the process
//! stays in the task list, and the next argmind finds a server it did not start and cannot vouch
//! for. Nothing on the screen says any of this happened.
//!
//! `Drop` is not enough and this project has already been bitten by that: **`std::process::exit`
//! skips destructors**, a panic in another thread can take the process without unwinding this one,
//! and Task Manager asks nobody. A cleanup that only runs on the tidy path is a cleanup that is
//! missing exactly when it is needed.
//!
//! So the guarantee is the operating system's rather than ours. Every supervised child joins one
//! **job object** created with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`, and Windows closes every
//! handle a process held when it dies, however it died. The last handle to the job closing is what
//! kills the members — so the children go when argmind goes, including when argmind is killed.
//!
//! No new dependency: job objects are in the same `windows` crate as the tray and the eyes.

use std::os::windows::io::AsRawHandle;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::OnceLock;

use windows::Win32::Foundation::HANDLE;
use windows::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, SetInformationJobObject,
    JobObjectExtendedLimitInformation, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

/// A fresh job whose members are killed when the last handle to it closes.
///
/// Public because **the one thing worth testing here is that the flag does what it says**, and a
/// test cannot observe the process-wide job without exiting the test. A test makes its own, puts a
/// child in it, closes it, and watches the child die — the same call, the same flag.
pub fn kill_on_close() -> windows::core::Result<HANDLE> {
    unsafe {
        let job = CreateJobObjectW(None, None)?;
        let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(
            job,
            JobObjectExtendedLimitInformation,
            &limits as *const _ as *const core::ffi::c_void,
            std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
        )?;
        Ok(job)
    }
}

/// The one job every supervised child joins, made once and never closed by us.
///
/// Stored as an `isize` because a raw handle is not `Sync`, the same way [`crate::tray`] keeps its
/// window. `None` means the job could not be made — the child still starts and is still killed on
/// `Drop`, and [`Supervised::supervised`] says the weaker promise is the one in force rather than
/// letting a caller assume the stronger one.
fn ours() -> Option<HANDLE> {
    static JOB: OnceLock<Option<isize>> = OnceLock::new();
    (*JOB.get_or_init(|| kill_on_close().ok().map(|h| h.0 as isize)))
        .map(|h| HANDLE(h as *mut core::ffi::c_void))
}

/// A running child argmind is responsible for.
pub struct Supervised {
    child: Child,
    pub pid: u32,
    pub exe: String,
    /// Whether the operating system will kill it if argmind dies without tidying up.
    supervised: bool,
}

impl Supervised {
    /// Start something and take responsibility for it.
    ///
    /// `dir` is the working directory — a server that reads files relative to its own source tree
    /// needs one. `log` catches both output streams: **argmind has no console**, so a child's
    /// complaint would otherwise go nowhere, and "it did not start and there is no reason why" is
    /// the worst shape a failure can take. Standard input is closed: nothing here is interactive,
    /// and a child waiting on a keystroke that will never come looks exactly like a hung one.
    pub fn start(exe: &Path, args: &[&str], dir: Option<&Path>, log: Option<&Path>)
                 -> std::io::Result<Self> {
        let mut command = Command::new(exe);
        command.args(args).stdin(Stdio::null());
        if let Some(dir) = dir {
            command.current_dir(dir);
        }
        match log {
            Some(at) => {
                let out = std::fs::File::create(at)?;
                let err = out.try_clone()?;
                command.stdout(Stdio::from(out)).stderr(Stdio::from(err));
            }
            //: Not inherited. A console argmind does not own is not a place to write to.
            None => {
                command.stdout(Stdio::null()).stderr(Stdio::null());
            }
        }
        let child = command.spawn()?;
        let supervised = match ours() {
            Some(job) => {
                unsafe { AssignProcessToJobObject(job, HANDLE(child.as_raw_handle())) }.is_ok()
            }
            None => false,
        };
        Ok(Supervised { pid: child.id(), exe: exe.display().to_string(), child, supervised })
    }

    /// Whether the operating system will clean this up even if argmind cannot.
    ///
    /// **A caller that wants to claim the guarantee has to ask for it.** If the job could not be
    /// made, the child is still killed on the ordinary path and only on the ordinary path, which is
    /// a different promise and should not be reported as the same one.
    pub fn supervised(&self) -> bool {
        self.supervised
    }

    /// Still running? `false` the moment it has exited, for whatever reason.
    pub fn alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// What it exited with, or `None` while it is still going.
    pub fn finished(&mut self) -> Option<Option<i32>> {
        match self.child.try_wait() {
            Ok(Some(status)) => Some(status.code()),
            _ => None,
        }
    }

    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Supervised {
    fn drop(&mut self) {
        self.stop();
    }
}
