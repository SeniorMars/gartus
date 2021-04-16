mod graphics;
use graphics::display::*;
use graphics::matrix::*;
use std::io;

fn main() -> io::Result<()> {
    let bg = Pixel {
        red: 246,
        green: 199,
        blue: 183,
    };
    let mut img = Canvas::new_with_bg(400, 400, 255, bg);
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

    let mut dilate = Matrix::scale(0.5, 0.5, 0.5);
    dilate *= geass;
    let mut reflect = Matrix::reflect_xz();
    img.set_line_color(191, 70, 61);
    // for i in 0..360 {
    //     let mut copy = img.clone();
    //     let mut rotate = Matrix::rotate_z(i as f64);
    //     rotate *= dilate.clone();
    //     copy.draw_lines(&rotate);
    //     copy.save_binary(&format!("pics/a{:04}.ppm", i))?;
    // }
    // let mut shear = Matrix::shearing_y(0.5, 1.0);
    let file_name = "geass";
    // shear *= dilate.clone();
    reflect *= dilate.clone();

    // img.draw_lines(&rotate);
    // img.display()?;
    img.draw_lines_for_animation(&dilate, &file_name)?;
    img.draw_lines_for_animation(&reflect, &file_name)?;
    img.animation(&file_name)?;
    // img.view_animation("./geass.gif")?;
    Ok(())
}
