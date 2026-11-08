//! **A program with no API and no script** — for `--bin screen`: a window, an order-number box, a
//! 조회 button, and the answer drawn only after the button is pressed. Nothing on disk holds it, so
//! the only way to the answer is through the window.
//!
//!     cargo build --release -p argmind-env-windows --example order-window

#[cfg(windows)]
fn main() {
    use windows::core::{w, PCWSTR};
    use windows::Win32::Foundation::{HWND, LPARAM, LRESULT, WPARAM};
    use windows::Win32::System::LibraryLoader::GetModuleHandleW;
    use windows::Win32::UI::WindowsAndMessaging::*;

    static mut EDIT: HWND = HWND(std::ptr::null_mut());
    static mut RESULT: HWND = HWND(std::ptr::null_mut());

    unsafe extern "system" fn proc_(h: HWND, m: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
        match m {
            WM_COMMAND if (wp.0 & 0xffff) == 2 => {
                let mut b = [0u16; 64];
                let n = GetWindowTextW(EDIT, &mut b).max(0) as usize;
                let typed = String::from_utf16_lossy(&b[..n]);
                let answer = if typed.trim().eq_ignore_ascii_case("A-1042") {
                    "A-1042: 출고 완료 — 운송장 6038-2271-9954"
                } else {
                    "그런 주문은 없습니다"
                };
                let a: Vec<u16> = answer.encode_utf16().chain([0]).collect();
                let _ = SetWindowTextW(RESULT, PCWSTR(a.as_ptr()));
                LRESULT(0)
            }
            WM_DESTROY => {
                PostQuitMessage(0);
                LRESULT(0)
            }
            _ => DefWindowProcW(h, m, wp, lp),
        }
    }

    unsafe {
        let inst = GetModuleHandleW(None).unwrap();
        let class = WNDCLASSW { lpfnWndProc: Some(proc_), hInstance: inst.into(), lpszClassName: w!("argmind-order"),
                                hbrBackground: windows::Win32::Graphics::Gdi::HBRUSH(16 as _), ..Default::default() };
        RegisterClassW(&class);
        let win = CreateWindowExW(Default::default(), w!("argmind-order"), w!("주문 조회"), WS_OVERLAPPEDWINDOW | WS_VISIBLE,
                                  100, 100, 460, 200, None, None, inst, None).unwrap();
        let child = |class: PCWSTR, text: PCWSTR, style: u32, x, y, w_, h_, id: usize| {
            CreateWindowExW(Default::default(), class, text, WINDOW_STYLE(style) | WS_CHILD | WS_VISIBLE,
                            x, y, w_, h_, win, HMENU(id as _), inst, None).unwrap()
        };
        child(w!("STATIC"), w!("주문번호"), 0, 10, 14, 70, 22, 10);
        EDIT = child(w!("EDIT"), w!(""), 0x0080 | 0x0080_0000, 90, 10, 200, 24, 1);
        child(w!("BUTTON"), w!("조회"), 0, 300, 10, 80, 26, 2);
        RESULT = child(w!("STATIC"), w!("주문번호를 넣고 조회를 누르세요"), 0, 10, 60, 420, 24, 3);
        let mut msg = MSG::default();
        while GetMessageW(&mut msg, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
}

#[cfg(not(windows))]
fn main() {}
