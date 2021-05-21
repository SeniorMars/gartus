use curves_rs::graphics::display::Canvas;
use curves_rs::graphics::display::Pixel;
fn main() {
    let mut img = Canvas::empty(256, 256, 255);
    let (width, height) = (img.width(), img.height());
    let mut data: Vec<Pixel> = Vec::with_capacity((width * height) as usize);
    (0..height).rev().for_each(|j| {
        eprintln!("Scanlines reminaing: {}", height - j - 1);
        (0..width).for_each(|i| {
            data.push(Pixel {
                red: (255.99 * (i as f64 / (width - 1) as f64)) as u8,
                green: (255.99 * (j as f64 / (height - 1) as f64)) as u8,
                blue: (255.99 * 0.25) as u8,
            })
        });
    });
    eprintln!("Done.");
    img.fill_canvas(data);
    img.display().expect("Could not render image")
}
