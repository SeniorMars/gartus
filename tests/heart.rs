use curves_rs::gmath::matrix::*;
use curves_rs::graphics::colors::*;
use curves_rs::graphics::display::*;

#[test]
fn main() {
    let mut heart = Canvas::new(610, 610, 255, Pixel::RGB(RGB::default()));
    heart.upper_left_system = true;
    let mut matrix = Matrix::new(4, 0, Vec::with_capacity(120));
    let corrs = [
        365, 341, 376, 315, 376, 315, 404, 289, 404, 289, 429, 263, 429, 263, 458, 239, 458, 239,
        485, 211, 485, 211, 511, 178, 511, 178, 525, 137, 525, 137, 520, 101, 520, 101, 493, 72,
        493, 72, 449, 49, 449, 49, 411, 59, 411, 59, 390, 77, 390, 77, 370, 104, 370, 104, 365,
        124, 365, 124, 365, 341,
    ];
    for i in corrs.chunks_exact(2) {
        matrix.add_point(i[0] as f64, i[1] as f64, 0.0)
    }
    let color = Pixel::RGB(RGB::new(188, 0, 45));
    let mut translate1 = Matrix::translate(-60.0, 99.0, 0.0);
    let mut rotatey = Matrix::reflect_yz();
    translate1 *= matrix;
    rotatey *= translate1.clone();
    heart.set_line_pixel(&color);
    heart.draw_lines(&rotatey);
    heart.draw_lines(&translate1);
    heart.fill(359, 237, &color, &color);
    heart.fill(267, 224, &color, &color);
    heart
        .save_extension("./pics/amit_i_love_you.png")
        .expect("could not save image");
    heart.display().expect("Could not display image")
}
