//! **A picture of one window on the person's desktop** — for the one place in the application that
//! shows a model where the list is blind (`argmind-serve`'s `looking.rs`), and for nothing else.
//!
//! Drawn by `PrintWindow` with `PW_RENDERFULLCONTENT`, as the work screen's picture is, so a window
//! partly covered by another is still drawn as itself; a point on the covered part is refused at
//! the press by the reach guard, which reads who owns the pixel *then*. Cut to a part and sized by a
//! box filter or a bilinear one, and written as a PNG with stored blocks — no dependency. **Never
//! saved**: the bytes go to the model's socket and are dropped, and the one copy kept is the same
//! picture with the point the model chose drawn on it ([`marked`]), held in memory for the person
//! to see beside the 실행 row until that row is answered.
//!
//! The rectangle is `GetWindowRect`'s, in the same physical pixels UI Automation reports, so a
//! point in thousandths of the picture is a point on the screen without any scaling in between.

use windows::Win32::Foundation::{HWND, RECT};
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::Storage::Xps::{PrintWindow, PRINT_WINDOW_FLAGS};
use windows::Win32::UI::WindowsAndMessaging::GetWindowRect;

/// The picture and the screen rectangle it covers — and the pixels the PNG holds, top-down BGRA,
/// so the same picture can be shown to the person with a point marked on it.
pub struct Shot {
    pub png: Vec<u8>,
    pub rect: (i64, i64, i64, i64),
    pub bgra: Vec<u8>,
    pub wide: usize,
    pub tall: usize,
}

/// A window's pixels as it draws itself — top-down BGRA — and the screen rectangle they cover.
pub struct Drawn {
    pub bgra: Vec<u8>,
    pub wide: usize,
    pub tall: usize,
    pub rect: (i64, i64, i64, i64),
}

/// The window `handle` as it draws itself. `None` for no window, an empty one, or one that would
/// not draw.
pub fn drawn(handle: isize) -> Option<Drawn> {
    if handle == 0 {
        return None;
    }
    let h = HWND(handle as _);
    let mut r = RECT::default();
    unsafe { GetWindowRect(h, &mut r) }.ok()?;
    let (wide, tall) = ((r.right - r.left).min(8000), (r.bottom - r.top).min(8000));
    if wide <= 0 || tall <= 0 {
        return None;
    }
    let mut pixels = vec![0u8; (wide * tall * 4) as usize];
    unsafe {
        let screen = GetDC(None);
        let canvas = CreateCompatibleDC(screen);
        let bmp = CreateCompatibleBitmap(screen, wide, tall);
        let was = SelectObject(canvas, bmp);
        let drew = PrintWindow(h, canvas, PRINT_WINDOW_FLAGS(2)).as_bool();
        SelectObject(canvas, was);
        let mut info = BITMAPINFO { bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32, biWidth: wide, biHeight: -tall,
            biPlanes: 1, biBitCount: 32, biCompression: BI_RGB.0, ..Default::default() }, ..Default::default() };
        let got = if drew {
            GetDIBits(canvas, bmp, 0, tall as u32, Some(pixels.as_mut_ptr() as *mut _), &mut info, DIB_RGB_COLORS)
        } else {
            0
        };
        let _ = DeleteObject(bmp);
        let _ = DeleteDC(canvas);
        ReleaseDC(None, screen);
        if got == 0 {
            return None;
        }
    }
    Some(Drawn { bgra: pixels, wide: wide as usize, tall: tall as usize,
                 rect: (r.left as i64, r.top as i64, wide as i64, tall as i64) })
}

impl Drawn {
    /// **The part of the window over `part` (screen pixels), made `width` wide** — shrunk by a box
    /// filter or enlarged by a bilinear one — as a PNG, with the screen rectangle it covers. `part`
    /// is clipped to the window; `None` when nothing of it is left.
    pub fn shot(&self, part: Option<(i64, i64, i64, i64)>, width: usize) -> Option<Shot> {
        let (wx, wy, ww, wh) = self.rect;
        let (px, py, pw, ph) = part.unwrap_or(self.rect);
        let (x0, y0) = (px.max(wx), py.max(wy));
        let (x1, y1) = ((px + pw).min(wx + ww), (py + ph).min(wy + wh));
        if x1 <= x0 || y1 <= y0 || width == 0 {
            return None;
        }
        let (cw, ch) = ((x1 - x0) as usize, (y1 - y0) as usize);
        let mut cut = Vec::with_capacity(cw * ch * 4);
        for y in (y0 - wy) as usize..(y1 - wy) as usize {
            let from = (y * self.wide + (x0 - wx) as usize) * 4;
            cut.extend_from_slice(&self.bgra[from..from + cw * 4]);
        }
        let (out, ow, oh) = if width <= cw {
            crate::screen::reduced(&cut, cw, ch, width)
        } else {
            enlarged(&cut, cw, ch, width)
        };
        let mut rgb = Vec::with_capacity(ow * oh * 3);
        for px in out.chunks_exact(4) {
            rgb.extend_from_slice(&[px[2], px[1], px[0]]);
        }
        Some(Shot { png: png(&rgb, ow, oh), rect: (x0, y0, x1 - x0, y1 - y0), bgra: out, wide: ow, tall: oh })
    }
}

/// **The picture a model was shown, with the point it chose drawn on it** — a red ring with a white
/// edge and four ticks pointing in, at `at` (pixels of that picture), made at most `most_wide` wide,
/// as a PNG. **Nothing on the point itself**: the first version put a dot there, and on a real
/// Character Map the dot hid the very glyph it marked (ì, under a request for Ø) — the thing a
/// person reads the picture to check. For
/// the person, beside the row that waits for their press: *the picture showed “©”* says what the
/// model believed, and this says where it will land.
pub fn marked(bgra: &[u8], wide: usize, tall: usize, at: (usize, usize), most_wide: usize) -> Vec<u8> {
    let (mut out, ow, oh) = crate::screen::reduced(bgra, wide, tall, most_wide.max(1));
    let (cx, cy) = (at.0 as f64 * ow as f64 / wide.max(1) as f64, at.1 as f64 * oh as f64 / tall.max(1) as f64);
    ring(&mut out, ow, oh, cx, cy);
    let mut rgb = Vec::with_capacity(ow * oh * 3);
    for px in out.chunks_exact(4) {
        rgb.extend_from_slice(&[px[2], px[1], px[0]]);
    }
    png(&rgb, ow, oh)
}

/// **Radius of the ring**, in pixels of the picture it is drawn on: a thirty-second of its width and
/// never under 12 — 20 at 640 wide, where a Character Map cell at 150% is about 37 across, so the
/// ring sits round the glyph rather than on it.
fn radius(wide: usize) -> f64 {
    (wide as f64 / 32.0).max(12.0)
}

/// Red on white so it reads on a light window and a dark one; drawn by distance from the centre,
/// so a ring that runs off an edge is cut there rather than wrapping.
fn ring(bgra: &mut [u8], wide: usize, tall: usize, cx: f64, cy: f64) {
    const RED: [u8; 3] = [0x30, 0x30, 0xe8];
    const WHITE: [u8; 3] = [0xff, 0xff, 0xff];
    let r = radius(wide);
    let (tick_from, tick_to) = (r + 2.0, r + 9.0);
    let reach = (tick_to + 2.0).ceil() as i64;
    //: A tick: within `half` of an axis through the point, between the ring and `tick_to`.
    let tick = |dx: f64, dy: f64, half: f64, grow: f64| {
        let (along, across) = if dx.abs() > dy.abs() { (dx.abs(), dy.abs()) } else { (dy.abs(), dx.abs()) };
        across <= half && along >= tick_from - grow && along <= tick_to + grow
    };
    for y in (cy as i64 - reach).max(0)..=(cy as i64 + reach).min(tall as i64 - 1) {
        for x in (cx as i64 - reach).max(0)..=(cx as i64 + reach).min(wide as i64 - 1) {
            let (dx, dy) = (x as f64 + 0.5 - cx, y as f64 + 0.5 - cy);
            let d = (dx * dx + dy * dy).sqrt();
            let paint = if (d - r).abs() <= 1.5 || tick(dx, dy, 1.0, 0.0) {
                RED
            } else if (d - r).abs() <= 2.7 || tick(dx, dy, 2.2, 1.2) {
                WHITE
            } else {
                continue;
            };
            let o = (y as usize * wide + x as usize) * 4;
            bgra[o..o + 3].copy_from_slice(&paint);
        }
    }
}

/// The window `handle`, at most `most_wide` pixels wide — never enlarged.
pub fn window(handle: isize, most_wide: usize) -> Option<Shot> {
    let d = drawn(handle)?;
    d.shot(None, most_wide.min(d.wide))
}

/// Top-down BGRA made `width` wide, bilinear — for a picture smaller than the size measured.
fn enlarged(bgra: &[u8], wide: usize, tall: usize, width: usize) -> (Vec<u8>, usize, usize) {
    let (nw, nh) = (width, (tall * width / wide).max(1));
    let mut out = vec![0u8; nw * nh * 4];
    let at = |x: usize, y: usize, c: usize| bgra[(y * wide + x) * 4 + c] as f64;
    for y in 0..nh {
        let fy = ((y as f64 + 0.5) * tall as f64 / nh as f64 - 0.5).clamp(0.0, (tall - 1) as f64);
        let (y0, ty) = (fy.floor() as usize, fy - fy.floor());
        let y1 = (y0 + 1).min(tall - 1);
        for x in 0..nw {
            let fx = ((x as f64 + 0.5) * wide as f64 / nw as f64 - 0.5).clamp(0.0, (wide - 1) as f64);
            let (x0, tx) = (fx.floor() as usize, fx - fx.floor());
            let x1 = (x0 + 1).min(wide - 1);
            let o = (y * nw + x) * 4;
            for c in 0..3 {
                let top = at(x0, y0, c) * (1.0 - tx) + at(x1, y0, c) * tx;
                let low = at(x0, y1, c) * (1.0 - tx) + at(x1, y1, c) * tx;
                out[o + c] = (top * (1.0 - ty) + low * ty).round() as u8;
            }
            out[o + 3] = 255;
        }
    }
    (out, nw, nh)
}

/// A PNG with **deflate's stored blocks** — the same hand-written encoder `argmind-measure` uses for
/// its test cards, because the licence rule is cheapest to keep with nothing new to check.
pub fn png(rgb: &[u8], w: usize, h: usize) -> Vec<u8> {
    //: Each row is prefixed with filter byte 0, which is what makes stored blocks legal.
    let mut raw = Vec::with_capacity(h * (1 + w * 3));
    for y in 0..h {
        raw.push(0);
        raw.extend_from_slice(&rgb[y * w * 3..(y + 1) * w * 3]);
    }
    let mut z = vec![0x78, 0x01];
    for (i, part) in raw.chunks(65_535).enumerate() {
        let last = (i + 1) * 65_535 >= raw.len();
        z.push(if last { 1 } else { 0 });
        z.extend_from_slice(&(part.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(part.len() as u16)).to_le_bytes());
        z.extend_from_slice(part);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for byte in &raw {
        a = (a + *byte as u32) % 65_521;
        b = (b + a) % 65_521;
    }
    z.extend_from_slice(&((b << 16) | a).to_be_bytes());
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&(w as u32).to_be_bytes());
    ihdr.extend_from_slice(&(h as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
    chunk(&mut out, b"IHDR", &ihdr);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut c = 0xffff_ffffu32;
    for byte in kind.iter().chain(data) {
        c ^= *byte as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    out.extend_from_slice(&(!c).to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The same bytes `argmind-measure`'s encoder makes**, whose files Ollama and Edge have both
    /// read — a second encoder that drifted would be a picture nobody could open.
    #[test]
    fn a_two_by_two_png_is_what_the_measured_encoder_writes() {
        let rgb = [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255];
        let made = png(&rgb, 2, 2);
        assert_eq!(&made[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
        //: IHDR's CRC for a 2×2 8-bit RGB image — `zlib.crc32` gives 0xfdd49a73.
        assert_eq!(&made[29..33], &[0xfd, 0xd4, 0x9a, 0x73]);
        assert_eq!(&made[made.len() - 8..], &[0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82]);
    }

    #[test]
    fn no_window_is_no_picture() {
        assert!(window(0, 1280).is_none());
    }

    fn checker(wide: usize, tall: usize) -> Drawn {
        let mut bgra = Vec::new();
        for y in 0..tall {
            for x in 0..wide {
                let v = if (x + y) % 2 == 0 { 255 } else { 0 };
                bgra.extend_from_slice(&[v, v, v, 255]);
            }
        }
        Drawn { bgra, wide, tall, rect: (100, 50, wide as i64, tall as i64) }
    }

    #[test]
    fn a_part_is_cut_on_screen_pixels_and_clipped_to_the_window() {
        let d = checker(40, 20);
        let s = d.shot(Some((110, 60, 10, 5)), 10).unwrap();
        assert_eq!(s.rect, (110, 60, 10, 5));
        assert_eq!(&s.png[16..24], &[0, 0, 0, 10, 0, 0, 0, 5]);
        let clipped = d.shot(Some((130, 40, 100, 100)), 10).unwrap();
        assert_eq!(clipped.rect, (130, 50, 10, 20), "clipped to the window it was cut from");
        assert!(d.shot(Some((0, 0, 10, 10)), 10).is_none(), "nothing of it inside the window");
    }

    #[test]
    fn the_point_is_ringed_where_it_was_chosen_and_nothing_else_is_touched() {
        let (wide, tall) = (200, 100);
        let mut bgra = vec![0x80; wide * tall * 4];
        ring(&mut bgra, wide, tall, 50.0, 40.0);
        let px = |x: usize, y: usize| [bgra[(y * wide + x) * 4], bgra[(y * wide + x) * 4 + 1], bgra[(y * wide + x) * 4 + 2]];
        let r = radius(wide) as usize;
        assert_eq!(px(50, 40), [0x80; 3], "the point itself is left as it is, so its glyph can be read");
        assert_eq!(px(50 + r / 2, 40), [0x80; 3], "and so is everything inside the ring");
        assert_eq!(px(50 + r, 40), [0x30, 0x30, 0xe8], "the ring at its radius");
        assert_eq!(px(50, 40 - r - 5), [0x30, 0x30, 0xe8], "a tick above it, pointing in");
        assert_eq!(px(150, 80), [0x80; 3], "far from the point nothing changed");
        //: Cut at an edge, never wrapped to the other side.
        let mut edge = vec![0x80; wide * tall * 4];
        ring(&mut edge, wide, tall, 1.0, 1.0);
        assert_eq!(&edge[(wide - 1) * 4..wide * 4 - 1], &[0x80; 3]);
    }

    #[test]
    fn a_marked_picture_is_a_png_no_wider_than_asked() {
        let d = checker(1280, 400);
        let s = d.shot(None, 1280).unwrap();
        let made = marked(&s.bgra, s.wide, s.tall, (640, 200), 640);
        assert_eq!(&made[16..24], &[0, 0, 2, 128, 0, 0, 0, 200]);
    }

    #[test]
    fn a_small_part_is_enlarged_to_the_width_asked() {
        let d = checker(40, 20);
        let s = d.shot(Some((100, 50, 20, 10)), 80).unwrap();
        assert_eq!(&s.png[16..24], &[0, 0, 0, 80, 0, 0, 0, 40]);
        assert_eq!(s.rect, (100, 50, 20, 10), "the rectangle is the screen's, not the picture's");
    }
}
