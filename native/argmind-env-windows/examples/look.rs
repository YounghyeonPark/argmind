//! What the camera costs and how still a still room is. **This turns the camera on.**
//!
//!     cargo run -p argmind-env-windows --example look
use std::time::Instant;
fn main() {
    let began = Instant::now();
    let mut cam = match argmind_env_windows::camera::Camera::open() {
        Ok(c) => c,
        Err(e) => { println!("  open: {e}"); return; }
    };
    println!("  {:?} {}x{}", cam.name, cam.width, cam.height);
    println!("  open and settle: {:.0} ms (settling alone {:?})",
             began.elapsed().as_secs_f64() * 1000.0,
             cam.settled_in.map(|s| format!("{:.0} ms", s * 1000.0)));
    let (mut before, mut grabs, mut worst) = (None, Vec::new(), 0.0f64);
    for _ in 0..25 {
        let Ok(f) = cam.look() else { continue };
        if let Some(b) = &before { worst = worst.max(f.differs_from(b)); }
        grabs.push(f.took * 1000.0);
        before = Some(f);
    }
    grabs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("  a look: median {:.0} ms, worst {:.0} ms, over {} looks",
             grabs[grabs.len() / 2], grabs[grabs.len() - 1], grabs.len());
    println!("  a still room differs from itself by at most {worst:.2}");
}
