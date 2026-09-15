//! Asking the person, in a window they cannot miss.
//!
//! **An approval is only worth having if it was informed.** A dialog saying *"Install
//! dependencies?"* buys nothing: the person clicks yes because they want the program to work, and
//! they have authorised something they could not name. So [`confirm`] takes the whole sentence and
//! the callers write out exactly what will happen — which program, how big, where from, and
//! whether it will ask for administrator rights.
//!
//! Topmost, because the one failure that makes a confirmation dialog worse than no dialog is the
//! person never seeing it: a program that appears to hang while a hidden window waits for a click
//! teaches people to kill it.

use windows::core::{HSTRING, PCWSTR};
use windows::Win32::Foundation::HWND;
use windows::Win32::UI::WindowsAndMessaging::{MessageBoxW, IDYES, MB_ICONQUESTION, MB_SETFOREGROUND,
                                              MB_SYSTEMMODAL, MB_YESNO};

/// Yes or no, from a window on top of everything.
///
/// **Anything that is not a clear yes is a no.** A dialog dismissed with the escape key, closed
/// with its own button, or destroyed because the session ended all answer the same way, and the
/// answer is not *go ahead*.
pub fn confirm(title: &str, body: &str) -> bool {
    unsafe {
        let answer = MessageBoxW(
            HWND::default(),
            PCWSTR(HSTRING::from(body).as_ptr()),
            PCWSTR(HSTRING::from(title).as_ptr()),
            MB_YESNO | MB_ICONQUESTION | MB_SETFOREGROUND | MB_SYSTEMMODAL,
        );
        answer == IDYES
    }
}
