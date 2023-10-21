use gartus::prelude::{Canvas, EdgeMatrix, Matrix, Rgb};

fn main() {
    let mut heart = Canvas::new(610, 610, Rgb::default());
    heart.upper_left_origin = true;
    let mut points = EdgeMatrix::new();
    let corrs = [
        365, 341, 376, 315, 376, 315, 404, 289, 404, 289, 429, 263, 429, 263, 458, 239, 458, 239,
        485, 211, 485, 211, 511, 178, 511, 178, 525, 137, 525, 137, 520, 101, 520, 101, 493, 72,
        493, 72, 449, 49, 449, 49, 411, 59, 411, 59, 390, 77, 390, 77, 370, 104, 370, 104, 365,
        124, 365, 124, 365, 341,
    ];
    for i in corrs.chunks_exact(2) {
        points.push_point(i[0] as f64, i[1] as f64, 0.0);
    }
    let color = Rgb::new(188, 0, 45);
    let translated = points.apply(&Matrix::translate(-60.0, 99.0, 0.0));
    let reflected = translated.apply(&Matrix::reflect_yz());
    heart.set_line_pixel(color);
    heart.draw_lines(&reflected);
    heart.draw_lines(&translated);
    heart.fill(359, 237, &color, &color);
    heart.fill(267, 224, &color, &color);
    heart
        .save_extension("./pics/amit_i_love_you.png")
        .expect("could not save image");
    heart.display().expect("Could not display image")
}
