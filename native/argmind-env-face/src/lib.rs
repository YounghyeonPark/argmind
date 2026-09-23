//! **Finding a face, and which way it is pointed — without asking anything that can talk.**
//!
//! The vision model can already do this: measured 2026-09-23, `qwen3-vl:8b` reads *looking at the
//! screen* or not correctly at 320×180 and above, 4 of 4 across two room states. It costs **4–5
//! seconds** and a model call, and at full resolution it also invented a detail it could not see.
//! `docs/findings/seeing-a-room.md` has it. This crate is the other way of getting the same fact,
//! and the whole argument for it is the price.
//!
//! ## What is in here and what is deliberately not
//!
//! Two graphs, both run by [`tract`], both permissively licensed, neither able to produce a
//! sentence:
//!
//! | | what it gives | licence |
//! |---|---|---|
//! | **YuNet** `face_detection_yunet_2023mar` | a box and five points per face | MIT |
//! | **SFace** `face_recognition_sface_2021dec` | 128 numbers per face | Apache-2.0 |
//!
//! **The obvious choice was refused on licence.** InsightFace's `buffalo_l` and the ArcFace family
//! generally are the default everywhere, the code is MIT — and the *weights* are non-commercial,
//! research-only, because of what they were trained on. The owner's rule is
//! 오픈소스라이센스에 문제 없는것만 사용할것, so they are out, and the pair above is what is left
//! that is small, fast and clean.
//!
//! **`tract` rather than ONNX Runtime**, for a reason this repository has already paid for once:
//! `argmind-app.exe` links its WebView2 loader statically because a `-gnu` build exits on a
//! double-click with `0xC0000135` and no message, for want of one DLL. ONNX Runtime is a DLL beside
//! the executable. `tract` is pure Rust and there is nothing to ship.
//!
//! ## What this crate will not tell you
//!
//! **It does not name anybody and it cannot describe a room.** It gives a rectangle, five points and
//! — with [`Faces::embed`] — an array of numbers. Whether two arrays are the same person is a
//! comparison a caller makes; whether there is a coffee cup on the desk is a question for the model
//! that can see one.
//!
//! **And what it measures is a head, not a gaze.** [`Face::head_on`] is the nose's offset between
//! the eyes, which moves when somebody turns their head and does not move when they keep their head
//! still and look sideways. It is named for what it is. **No threshold is shipped here**: the number
//! is returned raw, because `camera::MOVING` got its 0.15 from a measurement with a person doing
//! both things on purpose and this one has not had that yet.

use std::path::Path;

use tract_onnx::prelude::*;

/// A graph that has been read, optimised and made runnable.
type Ready = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// **The one size YuNet accepts, and it is not a choice.**
///
/// The published ONNX has a shape baked into the output of its first strided convolution, so tract
/// refuses every other input: 320×320 and 180×320 both fail with *Unifying shapes 1,16,320,320 and
/// …*. Measured rather than read off a document, and worth writing down because it looks like a
/// tunable and is not.
pub const SQUARE: usize = 640;

/// The three strides YuNet's head is written over — 80×80, 40×40 and 20×20 cells of `SQUARE`.
const STRIDES: [usize; 3] = [8, 16, 32];

/// Boxes overlapping more than this, keeping the stronger. **Chosen, not measured** — it is
/// OpenCV's own default for this model and there is no reading here that would improve on it.
const OVERLAP: f32 = 0.3;

/// A face, in the coordinates of the picture that was handed in.
#[derive(Clone, Debug)]
pub struct Face {
    /// `x`, `y`, `width`, `height`.
    pub rect: (f32, f32, f32, f32),
    /// **YuNet's order: right eye, left eye, nose tip, right mouth corner, left mouth corner** —
    /// right and left as the *picture* has them, so the viewer's left is index 0.
    pub marks: [(f32, f32); 5],
    pub score: f32,
}

impl Face {
    /// **How square-on the head is: 0 facing the camera, ±1 fully turned.**
    ///
    /// The nose sits midway between the eyes when somebody faces the lens and slides toward one of
    /// them as they turn. Dividing by the eye separation makes it independent of how far away they
    /// are, which is the only reason it can be one number rather than a table.
    ///
    /// **It is a head, not a gaze, and the difference is the whole caveat.** Somebody who holds
    /// their head still and looks off to the side reads 0 here and is not looking at the screen; the
    /// vision model gets that case and this does not. Whether that case matters is a measurement
    /// nobody has taken.
    ///
    /// `None` when the eyes land on top of each other, which is a detection that should not be
    /// trusted rather than a head turned exactly 90°.
    pub fn head_on(&self) -> Option<f32> {
        let (right, left, nose) = (self.marks[0], self.marks[1], self.marks[2]);
        let apart = left.0 - right.0;
        if apart.abs() < 1.0 {
            return None;
        }
        let middle = (right.0 + left.0) / 2.0;
        Some((nose.0 - middle) / apart * 2.0)
    }

    /// **How far up or down the head is tipped**, by the same construction: the nose between the
    /// eye line and the mouth line, scaled by the distance between them. Positive is tipped down.
    ///
    /// Included because a person reading something on their desk is the case `head_on` reads as
    /// facing the camera, and it is the commonest way of not looking at a screen.
    pub fn head_down(&self) -> Option<f32> {
        let eyes = ((self.marks[0].1) + (self.marks[1].1)) / 2.0;
        let mouth = ((self.marks[3].1) + (self.marks[4].1)) / 2.0;
        let tall = mouth - eyes;
        if tall.abs() < 1.0 {
            return None;
        }
        Some(((self.marks[2].1 - eyes) / tall - 0.5) * 2.0)
    }
}

/// The detector, and optionally the thing that turns a face into numbers.
pub struct Faces {
    finder: Ready,
    namer: Option<Ready>,
}

impl Faces {
    /// Read YuNet off disk. **0.03 s**, measured, and it is done once.
    pub fn load(finder: &Path) -> Result<Faces, String> {
        Ok(Faces { finder: read(finder, [1, 3, SQUARE, SQUARE])?, namer: None })
    }

    /// Add SFace, so [`Faces::embed`] works. **0.12 s** to read, measured, and 38.7 MB on disk —
    /// which is why it is separate: finding a face is not remembering one, and a caller that only
    /// wants to know whether somebody is at the desk should not pay for the second.
    pub fn also_remember(mut self, namer: &Path) -> Result<Faces, String> {
        self.namer = Some(read(namer, [1, 3, 112, 112])?);
        Ok(self)
    }

    /// Every face in an **RGB** picture, at or above `least`.
    ///
    /// **103 ms median, 122 ms worst**, ten passes at `SQUARE`, measured on noise rather than on
    /// zeros because a constant tensor is the one input a graph might shortcut.
    pub fn find(&self, rgb: &[u8], width: usize, height: usize, least: f32)
                -> Result<Vec<Face>, String> {
        if width == 0 || height == 0 {
            return Ok(Vec::new());
        }
        //: **Letterboxed rather than stretched.** A squashed face moves the nose relative to the
        //: eyes, which is precisely the quantity `head_on` reads, so stretching would put a false
        //: turn into every wide picture.
        let scale = (SQUARE as f32 / width as f32).min(SQUARE as f32 / height as f32);
        let (into_w, into_h) = ((width as f32 * scale) as usize, (height as f32 * scale) as usize);

        //: BGR and 0–255, which is what OpenCV's `blobFromImage` hands this model with no mean and
        //: no scaling. Getting either wrong gives a graph that runs and finds nothing.
        let mut plane = vec![0.0f32; 3 * SQUARE * SQUARE];
        for y in 0..into_h.min(SQUARE) {
            let from_y = ((y as f32 / scale) as usize).min(height - 1);
            for x in 0..into_w.min(SQUARE) {
                let from_x = ((x as f32 / scale) as usize).min(width - 1);
                let at = (from_y * width + from_x) * 3;
                let to = y * SQUARE + x;
                plane[to] = rgb[at + 2] as f32;
                plane[SQUARE * SQUARE + to] = rgb[at + 1] as f32;
                plane[2 * SQUARE * SQUARE + to] = rgb[at] as f32;
            }
        }
        let input: Tensor = tract_ndarray::Array4::from_shape_vec((1, 3, SQUARE, SQUARE), plane)
            .map_err(|e| e.to_string())?.into();
        let out = self.finder.run(tvec!(input.into())).map_err(|e| e.to_string())?;
        if out.len() < 12 {
            return Err(format!("YuNet gave {} outputs, expected 12", out.len()));
        }

        let mut found = Vec::new();
        for (i, stride) in STRIDES.iter().enumerate() {
            let cls = out[i].to_array_view::<f32>().map_err(|e| e.to_string())?;
            let obj = out[3 + i].to_array_view::<f32>().map_err(|e| e.to_string())?;
            let box_ = out[6 + i].to_array_view::<f32>().map_err(|e| e.to_string())?;
            let kps = out[9 + i].to_array_view::<f32>().map_err(|e| e.to_string())?;
            let across = SQUARE / stride;
            for cell in 0..across * across {
                //: **The geometric mean of the two heads.** OpenCV's own decoder does this and the
                //: two numbers mean different things — *is this a face* and *is anything here* —
                //: so multiplying and rooting is not a shortcut for averaging them.
                let score = (cls[[0, cell, 0]] * obj[[0, cell, 0]]).max(0.0).sqrt();
                if score < least {
                    continue;
                }
                let (col, row) = ((cell % across) as f32, (cell / across) as f32);
                let s = *stride as f32;
                let cx = (col + box_[[0, cell, 0]]) * s;
                let cy = (row + box_[[0, cell, 1]]) * s;
                let w = box_[[0, cell, 2]].exp() * s;
                let h = box_[[0, cell, 3]].exp() * s;
                let mut marks = [(0.0f32, 0.0f32); 5];
                for (k, mark) in marks.iter_mut().enumerate() {
                    *mark = (((col + kps[[0, cell, k * 2]]) * s) / scale,
                             ((row + kps[[0, cell, k * 2 + 1]]) * s) / scale);
                }
                found.push(Face {
                    rect: ((cx - w / 2.0) / scale, (cy - h / 2.0) / scale, w / scale, h / scale),
                    marks,
                    score,
                });
            }
        }
        Ok(thin(found))
    }

    /// **128 numbers for one face**, from SFace. `None` unless [`Faces::also_remember`] was called.
    ///
    /// **45 ms median**, measured. The numbers mean nothing on their own — two of them are the same
    /// person when their cosine similarity is high, and *how high* is a threshold this crate does
    /// not ship for the same reason it ships no threshold for [`Face::head_on`].
    pub fn embed(&self, rgb: &[u8], width: usize, height: usize, face: &Face)
                 -> Result<Option<Vec<f32>>, String> {
        let Some(namer) = &self.namer else { return Ok(None) };
        //: **Aligned on the eyes rather than cropped to the box.** SFace was trained on faces put
        //: through a five-point similarity transform, and handing it a plain crop is the mistake
        //: that produces embeddings which are consistently wrong in a way nothing complains about.
        let plane = align(rgb, width, height, &face.marks);
        let input: Tensor = tract_ndarray::Array4::from_shape_vec((1, 3, 112, 112), plane)
            .map_err(|e| e.to_string())?.into();
        let out = namer.run(tvec!(input.into())).map_err(|e| e.to_string())?;
        let view = out[0].to_array_view::<f32>().map_err(|e| e.to_string())?;
        Ok(Some(view.iter().copied().collect()))
    }
}

/// **How alike two faces are, −1 to 1.** Cosine similarity, which is what SFace's embeddings are
/// built to be compared with. No threshold: see [`Faces::embed`].
pub fn alike(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let ma: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if ma == 0.0 || mb == 0.0 { None } else { Some(dot / (ma * mb)) }
}

/// Read one graph and fix its input shape, which tract needs before it can optimise.
fn read(path: &Path, shape: [usize; 4]) -> Result<Ready, String> {
    onnx()
        .model_for_path(path)
        .and_then(|m| m.with_input_fact(0, f32::fact(shape).into()))
        .and_then(|m| m.into_optimized())
        .and_then(|m| m.into_runnable())
        .map_err(|e| format!("{path:?}: {e}"))
}

/// **Where SFace's training put the five points in a 112×112 crop.** Copied from ArcFace's canonical
/// template, which is what every implementation of this family aligns to.
const CANON: [(f32, f32); 5] = [(38.29, 51.70), (73.53, 51.50), (56.02, 71.74),
                                (41.55, 92.37), (70.73, 92.20)];

/// Warp the five points onto [`CANON`] and sample a 112×112 BGR plane.
///
/// **A similarity transform and not a full affine**: four numbers — rotation, one scale and a
/// shift — fitted least-squares over the five pairs. A full affine can shear a face into the
/// template and would hide a bad detection instead of producing a bad embedding, which is worse.
fn align(rgb: &[u8], width: usize, height: usize, marks: &[(f32, f32); 5]) -> Vec<f32> {
    //: Umeyama, restricted to similarity: solve for (a, b, tx, ty) where
    //: [x'] = [a -b][x] + [tx]
    //: [y']   [b  a][y]   [ty]
    let n = 5.0f32;
    let (mut sx, mut sy, mut dx, mut dy) = (0.0, 0.0, 0.0, 0.0);
    for (i, m) in marks.iter().enumerate() {
        sx += m.0; sy += m.1; dx += CANON[i].0; dy += CANON[i].1;
    }
    let (sx, sy, dx, dy) = (sx / n, sy / n, dx / n, dy / n);
    let (mut num_a, mut num_b, mut den) = (0.0f32, 0.0f32, 0.0f32);
    for (i, m) in marks.iter().enumerate() {
        let (px, py) = (m.0 - sx, m.1 - sy);
        let (qx, qy) = (CANON[i].0 - dx, CANON[i].1 - dy);
        num_a += px * qx + py * qy;
        num_b += px * qy - py * qx;
        den += px * px + py * py;
    }
    let (a, b) = if den.abs() < 1e-6 { (1.0, 0.0) } else { (num_a / den, num_b / den) };
    //: Inverted, because sampling walks the *destination* and asks where each pixel came from.
    let det = a * a + b * b;
    let (ia, ib) = if det.abs() < 1e-9 { (1.0, 0.0) } else { (a / det, -b / det) };

    let mut plane = vec![0.0f32; 3 * 112 * 112];
    for y in 0..112usize {
        for x in 0..112usize {
            let (ox, oy) = (x as f32 - dx, y as f32 - dy);
            let fx = ia * ox - ib * oy + sx;
            let fy = ib * ox + ia * oy + sy;
            if fx < 0.0 || fy < 0.0 {
                continue;
            }
            let (px, py) = (fx as usize, fy as usize);
            if px >= width || py >= height {
                continue;
            }
            let at = (py * width + px) * 3;
            let to = y * 112 + x;
            plane[to] = rgb[at + 2] as f32;
            plane[112 * 112 + to] = rgb[at + 1] as f32;
            plane[2 * 112 * 112 + to] = rgb[at] as f32;
        }
    }
    plane
}

/// Strongest first, then drop anything overlapping one already kept.
fn thin(mut found: Vec<Face>) -> Vec<Face> {
    found.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<Face> = Vec::new();
    for face in found {
        if !kept.iter().any(|k| over(&k.rect, &face.rect) > OVERLAP) {
            kept.push(face);
        }
    }
    kept
}

/// Intersection over union of two `x, y, w, h` boxes.
fn over(a: &(f32, f32, f32, f32), b: &(f32, f32, f32, f32)) -> f32 {
    let x = (a.0 + a.2).min(b.0 + b.2) - a.0.max(b.0);
    let y = (a.1 + a.3).min(b.1 + b.3) - a.1.max(b.1);
    if x <= 0.0 || y <= 0.0 {
        return 0.0;
    }
    let both = x * y;
    both / (a.2 * a.3 + b.2 * b.3 - both)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face_at(nose_x: f32) -> Face {
        Face {
            rect: (0.0, 0.0, 100.0, 100.0),
            //: eyes 40 apart, mouth 30 below the eye line
            marks: [(30.0, 40.0), (70.0, 40.0), (nose_x, 55.0), (35.0, 70.0), (65.0, 70.0)],
            score: 0.9,
        }
    }

    #[test]
    fn a_nose_between_the_eyes_is_facing_the_camera() {
        assert!(face_at(50.0).head_on().unwrap().abs() < 0.01);
    }

    #[test]
    fn a_nose_over_one_eye_is_turned() {
        //: Signed, and the sign says which way: toward the picture's left eye is positive.
        assert!(face_at(70.0).head_on().unwrap() > 0.9);
        assert!(face_at(30.0).head_on().unwrap() < -0.9);
    }

    #[test]
    fn eyes_on_top_of_each_other_is_refused_rather_than_answered() {
        let mut f = face_at(50.0);
        f.marks[1] = f.marks[0];
        assert!(f.head_on().is_none(), "a detection this bad must not produce a confident number");
    }

    #[test]
    fn the_nose_halfway_down_is_level() {
        //: eyes at 40, mouth at 70, nose at 55 is exactly between.
        assert!(face_at(50.0).head_down().unwrap().abs() < 0.01);
    }

    #[test]
    fn alike_is_one_with_itself_and_refuses_a_mismatch() {
        let a = vec![0.3f32, -0.5, 0.8];
        assert!((alike(&a, &a).unwrap() - 1.0).abs() < 1e-5);
        assert_eq!(alike(&a, &[0.1, 0.2]), None);
        assert_eq!(alike(&a, &[]), None);
    }

    #[test]
    fn overlapping_boxes_collapse_to_the_stronger_one() {
        let strong = Face { score: 0.9, ..face_at(50.0) };
        let weak = Face { score: 0.4, rect: (2.0, 2.0, 100.0, 100.0), ..face_at(50.0) };
        let apart = Face { score: 0.8, rect: (500.0, 500.0, 50.0, 50.0), ..face_at(50.0) };
        let kept = thin(vec![weak, strong, apart]);
        assert_eq!(kept.len(), 2, "the overlapping pair should collapse and the far one survive");
        assert!((kept[0].score - 0.9).abs() < 1e-6, "the stronger one is the one kept");
    }
}
