use gartus::prelude::{Canvas, EdgeMatrix, FrameRecorder, Matrix, Rgb};

fn geass() {
    let mut img = Canvas::new_with_bg(800, 800, Rgb::new(24, 26, 27));
    img.upper_left_origin = true;

    let geass_corrs = [
        170, 216, 190, 249, 190, 249, 220, 274, 220, 274, 250, 295, 250, 295, 289, 318, 289, 318,
        347, 349, 347, 349, 347, 421, 347, 421, 400, 449, 400, 449, 453, 421, 453, 421, 453, 349,
        453, 349, 511, 318, 511, 318, 550, 295, 550, 295, 580, 274, 580, 274, 606, 249, 606, 249,
        630, 216, 630, 216, 601, 285, 601, 285, 571, 323, 571, 323, 525, 358, 525, 358, 492, 388,
        492, 388, 489, 448, 489, 448, 441, 475, 441, 475, 400, 499, 400, 499, 359, 475, 359, 475,
        311, 448, 311, 448, 308, 388, 308, 388, 275, 358, 275, 358, 229, 323, 229, 323, 199, 285,
        199, 285, 170, 216,
    ];

    let geass = EdgeMatrix::from_xy_pairs(&geass_corrs, 0.0);

    let center = Matrix::translate(-400.0, -400.0, 0.0);
    let base = geass.apply(&center);
    let reflect_pts = base.apply(&Matrix::reflect_xz());
    let half_pts = base.apply(&Matrix::reflect_45());
    let last_half_pts = half_pts.apply(&Matrix::reflect_yz());

    let mut combined = base;
    combined.extend(&reflect_pts);
    combined.extend(&half_pts);
    combined.extend(&last_half_pts);

    let white = Rgb::new(255, 255, 255);
    img.set_line_pixel(white);

    let off_center =
        Matrix::translate(360.0, 370.0, 0.0).mult_matrix(&Matrix::scale(0.1, 0.1, 0.1));
    img.draw_transformed(&geass, &off_center);
    img.fill(406, 413, white, white);
    img.set_line_pixel(Rgb::new(191, 70, 61));

    let back_translation = Matrix::translate(400.0, 410.0, 0.0);
    let mut recorder = FrameRecorder::new("anim", "geass").with_delay(2);
    for i in 0..180 {
        let transform = back_translation.mult_matrix(&Matrix::rotate_y(i as f64 * 2.0));
        recorder
            .capture_drawn(&img, &combined, &transform)
            .expect("Could not save frame");
    }
    std::fs::create_dir_all("final").expect("could not create final output directory");
    let file_name = "final/geass.gif";
    recorder
        .encode_gif(file_name)
        .expect("Could not make animation");
}

pub fn main() {
    geass()
}
