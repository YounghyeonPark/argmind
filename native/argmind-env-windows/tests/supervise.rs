//! Whether a child argmind started can outlive it.
//!
//! **The whole module rests on one operating-system promise**, and a promise nobody checked is a
//! comment. So this puts a real process in a real job, closes the job, and watches. It spawns
//! `ping 127.0.0.1` — loopback, no network, no window, and it would run for half a minute if
//! nothing killed it, which is what makes its death evidence rather than a coincidence.

use std::os::windows::io::AsRawHandle;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use argmind_env_windows::supervise::{kill_on_close, Supervised};
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::System::JobObjects::AssignProcessToJobObject;

/// Something that will keep running until somebody stops it.
fn a_long_job() -> Command {
    let mut it = Command::new("ping.exe");
    it.args(["-n", "30", "127.0.0.1"]).stdout(Stdio::null()).stderr(Stdio::null());
    it
}

/// **Closing the job kills what is in it.**
///
/// This is the guarantee the recogniser's server depends on: argmind can be killed outright — no
/// unwinding, no `Drop`, no chance to tidy — and Windows still closes its handles, and the last
/// handle to this job closing is what takes the child with it.
#[test]
fn a_child_in_a_job_dies_when_the_job_closes() {
    let job = kill_on_close().expect("a job object");
    let mut child = a_long_job().spawn().expect("ping");
    unsafe { AssignProcessToJobObject(job, HANDLE(child.as_raw_handle())) }.expect("assigned");

    //: Still going — otherwise the death below would prove nothing.
    assert!(matches!(child.try_wait(), Ok(None)), "it was not running to begin with");

    unsafe { CloseHandle(job) }.expect("closed");

    let by = Instant::now() + Duration::from_secs(5);
    let mut gone = false;
    while Instant::now() < by {
        if matches!(child.try_wait(), Ok(Some(_))) {
            gone = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let _ = child.kill();
    assert!(gone, "the child outlived the job it was in");
}

/// A child **not** in a job carries on, which is the thing being prevented.
///
/// Without this the test above could pass because `ping` happened to exit — a control, in the
/// project's usual sense, and cheap.
#[test]
fn a_child_in_no_job_is_still_running_a_moment_later() {
    let mut child = a_long_job().spawn().expect("ping");
    std::thread::sleep(Duration::from_millis(400));
    let running = matches!(child.try_wait(), Ok(None));
    let _ = child.kill();
    let _ = child.wait();
    assert!(running, "ping stopped on its own, so the other test's evidence is worthless");
}

/// A supervised child starts, is in the job, and stops when it is dropped.
#[test]
fn dropping_a_supervised_child_stops_it() {
    let pid;
    {
        let mut it = Supervised::start(std::path::Path::new("ping.exe"),
                                       &["-n", "30", "127.0.0.1"], None, None)
            .expect("started");
        pid = it.pid;
        assert!(it.alive());
        //: The stronger promise is claimed only when it is actually in force.
        assert!(it.supervised(), "the job object was not made, so nothing guarantees cleanup");
        assert_eq!(it.finished(), None);
    }
    //: Gone with the value. `tasklist` is asked rather than the `Child`, because the `Child` is
    //: what was just dropped — the question is whether the *operating system* still has it.
    let by = Instant::now() + Duration::from_secs(5);
    let mut gone = false;
    while Instant::now() < by {
        let out = Command::new("tasklist.exe")
            .args(["/FI", &format!("PID eq {pid}"), "/NH"]).output().expect("tasklist");
        if !String::from_utf8_lossy(&out.stdout).contains(&pid.to_string()) {
            gone = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    assert!(gone, "pid {pid} is still running after the Supervised holding it was dropped");
}

/// **The output goes somewhere it can be read.**
///
/// argmind has no console, so a child that failed with a good explanation would otherwise fail
/// silently — which is the shape of failure this project keeps finding.
#[test]
fn what_the_child_says_lands_in_the_log() {
    let log = std::env::temp_dir().join("argmind-supervise-test.log");
    let _ = std::fs::remove_file(&log);
    {
        let mut it = Supervised::start(std::path::Path::new("cmd.exe"),
                                       &["/c", "echo", "안녕"], None, Some(&log)).expect("started");
        let by = Instant::now() + Duration::from_secs(5);
        while it.alive() && Instant::now() < by {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_eq!(it.finished(), Some(Some(0)));
    }
    let said = std::fs::read(&log).expect("a log file");
    assert!(!said.is_empty(), "the child said something and the log is empty");
    let _ = std::fs::remove_file(&log);
}
