//! What cameras this machine has — **without opening one.**
//!
//!     cargo run -p argmind-env-windows --example cameras
//!
//! **Enumerating is not capturing**, and the difference matters here in the way it matters for the
//! microphone: no stream is started, no frame is read, and the light does not come on. This is the
//! cheapest answer to *is a camera reachable at all, and with what we already link against* — asked
//! before any of the expensive questions, the same way whisper.cpp was built by hand before
//! anything automated the build.
//!
//! Measured 2026-09-16 on this machine: **1 camera, `Integrated Camera`**, through the `windows`
//! crate that was already a dependency. No new crate, and none needed.
use windows::core::PWSTR;
use windows::Win32::Media::MediaFoundation::{
    MFCreateAttributes, MFEnumDeviceSources, MFStartup, MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME,
    MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID,
    MF_VERSION,
};

fn main() {
    unsafe {
        if let Err(e) = MFStartup(MF_VERSION, 0) {
            println!("  MFStartup failed: {e}");
            return;
        }
        let mut attrs = None;
        if let Err(e) = MFCreateAttributes(&mut attrs, 1) {
            println!("  attributes: {e}");
            return;
        }
        let attrs = attrs.expect("attributes");
        if let Err(e) = attrs.SetGUID(&MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
                                      &MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID) {
            println!("  set: {e}");
            return;
        }
        let mut found = std::ptr::null_mut();
        let mut count = 0u32;
        match MFEnumDeviceSources(&attrs, &mut found, &mut count) {
            Ok(()) => println!("  cameras: {count}"),
            Err(e) => {
                println!("  enumerate: {e}");
                return;
            }
        }
        for i in 0..count as isize {
            let Some(source) = (*found.offset(i)).as_ref() else { continue };
            let mut name = PWSTR::null();
            let mut len = 0u32;
            if source.GetAllocatedString(&MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &mut name,
                                         &mut len).is_ok() {
                println!("    {}", name.to_string().unwrap_or_default());
            }
        }
    }
}
