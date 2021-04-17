mod graphics;
use graphics::display::*;
use graphics::matrix::*;
use std::io;

fn main() -> io::Result<()> {
    let bg = Pixel::new(24, 26, 27);
    let mut img = Canvas::new_with_bg(800, 800, 255, bg);
    let mut geass = Matrix::new(0, 4, Vec::new());
    img.upper_left_system = true;

    let geass_corrs = [
        170, 216, 190, 249, 190, 249, 220, 274, 220, 274, 250, 295, 250, 295, 289, 318, 289, 318,
        347, 349, 347, 349, 347, 421, 347, 421, 400, 449, 400, 449, 453, 421, 453, 421, 453, 349,
        453, 349, 511, 318, 511, 318, 550, 295, 550, 295, 580, 274, 580, 274, 606, 249, 606, 249,
        630, 216, 630, 216, 601, 285, 601, 285, 571, 323, 571, 323, 525, 358, 525, 358, 492, 388,
        492, 388, 489, 448, 489, 448, 441, 475, 441, 475, 400, 499, 400, 499, 359, 475, 359, 475,
        311, 448, 311, 448, 308, 388, 308, 388, 275, 358, 275, 358, 229, 323, 229, 323, 199, 285,
        199, 285, 170, 216,
    ];

    for corr in geass_corrs.chunks(2) {
        geass.add_point(corr[0] as f64, corr[1] as f64, 0.0)
    }

    let mut dilate = Matrix::scale(0.1, 0.1, 0.1);
    let mut translate = Matrix::translate(360.0, 370.0, 0.0);
    let mut rfp1p1 = Matrix::translate(-400.0, -400.0, 0.0);
    let mut reflect = Matrix::reflect_xz();
    let mut half = Matrix::reflect_45();
    let mut last_half = Matrix::reflect_yz();
    let redish = Pixel::new(191, 70, 61);

    // small
    dilate *= geass.clone();
    translate *= dilate;

    rfp1p1 *= geass.clone();
    reflect *= rfp1p1.clone();
    last_half *= half.clone();
    half *= rfp1p1.clone();
    last_half *= rfp1p1.clone();

    let white = Pixel::new(255, 255, 255);
    img.set_line_pixel(white);
    img.draw_lines(&translate);
    img.fill(406, 413, white, white);
    img.set_line_pixel(redish);
    for i in 0..360 {
        let mut copy = img.clone();
        let rotate = Matrix::rotate_y(i as f64);
        let rfp1p2 = Matrix::translate(400.0, 400.0, 0.0);
        // apply rotate matrix - probably a better way of doing this
        let old_geass = rfp1p1.clone() * rotate.clone();
        let old_reflect = reflect.clone() * rotate.clone();
        let old_half = half.clone() * rotate.clone();
        let old_last = last_half.clone() * rotate.clone();
        // fix rotation to be at center
        let new_geass = old_geass * rfp1p2.clone();
        let new_reflect = old_reflect * rfp1p2.clone();
        let new_half = old_half * rfp1p2.clone();
        let new_last = old_last * rfp1p2.clone();
        // apply the matrices
        copy.draw_lines(&new_geass);
        copy.draw_lines(&new_reflect);
        copy.draw_lines(&new_half);
        copy.draw_lines(&new_last);
        // copy.display()?;
        copy.save_binary(&format!("anim/geass{:04}.ppm", i))?;
    }
    // utils::animation("h", "h.gif")?;
    Ok(())
}
