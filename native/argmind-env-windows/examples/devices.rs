//! Every microphone, speaker and camera this machine has. **Opens none of them.**
//!
//!     cargo run -p argmind-env-windows --example devices
//!
//! Enumerating reads a property store and a device list; it does not start a stream, so nothing
//! here records, plays or sees. That distinction is the reason this can be run without warning
//! anybody.

use windows::Win32::System::Com::{CoInitializeEx, COINIT_APARTMENTTHREADED};

fn main() {
    //: The enumerator is a COM object like everything else here.
    unsafe {
        let _ = CoInitializeEx(None, COINIT_APARTMENTTHREADED);
    }

    for (what, found) in [("마이크", argmind_env_windows::devices::microphones()),
                          ("스피커", argmind_env_windows::devices::speakers())] {
        println!("\n  === {what} ===");
        match found {
            Ok(list) if list.is_empty() => println!("    없음"),
            Ok(list) => {
                for d in &list {
                    println!("    {} {}", if d.default { "*" } else { " " }, d.name);
                    println!("        {}", d.id);
                }
                println!("    {}개, 기본 {}개", list.len(),
                         list.iter().filter(|d| d.default).count());
            }
            Err(why) => println!("    열거하지 못했습니다: {why}"),
        }
    }

    println!("\n  === 카메라 ===");
    match argmind_env_windows::camera::cameras() {
        Ok(list) if list.is_empty() => println!("    없음"),
        //: **Cameras come back as names only.** Media Foundation's stable handle is a symbolic
        //: link, and nothing here has measured whether the friendly name is unstable — so the name
        //: is what gets remembered, and this says so rather than leaving it to be discovered.
        Ok(list) => for name in list {
            println!("      {name}");
        },
        Err(why) => println!("    열거하지 못했습니다: {why}"),
    }
    println!();
}
