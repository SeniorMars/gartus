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
    let mut reflect = Matrix::reflect_xz();
    let file_name = "geass";
    dilate *= geass;
    reflect *= dilate.clone();
    img.set_line_color(191, 70, 61);
    img.draw_lines_for_animation(&dilate, &file_name)?;
    img.draw_lines_for_animation(&reflect, &file_name)?;
    img.animation(&file_name)
}

#[cfg(test)]
mod tests {
    use super::graphics::display::*;
    use super::graphics::matrix::*;
    use std::io;

    #[test]
    fn matrix_test() {
        let mut edge_matrix = Matrix::new(0, 4, Vec::with_capacity(4 * 2));
        println!("{}", edge_matrix);
        println!("Testing add_edge. Adding (1, 2, 3), (4, 5, 6) m2");
        edge_matrix.add_edge(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        println!("{}", edge_matrix);
        let mut ident = Matrix::identity_matrix(4);
        println!("Testing ident. m1 =");
        println!("{}", ident);
        println!("Testing Matrix mult. m1 * m2 =");
        ident *= edge_matrix.clone();
        println!("{}", ident);
        println!("Testing Matrix mult. m1 =");
        let mut m1 = Matrix::new(
            4,
            4,
            vec![
                1.0, 2.0, 3.0, 1.0, 4.0, 5.0, 6.0, 1.0, 7.0, 8.0, 9.0, 1.0, 10.0, 11.0, 12.0, 1.0,
            ],
        );
        println!("{}", m1);
        println!("Testing Matrix mult. m1 * m2 =");
        m1 *= edge_matrix.clone();
        println!("{}", m1);
        assert_eq!(
            m1,
            Matrix::new(2, 4, vec![40.0, 47.0, 54.0, 7.0, 76.0, 92.0, 108.0, 16.0])
        )
    }

    #[test]
    fn dw_line_test() -> io::Result<()> {
        let xres: f64 = 750.0;
        let yres: f64 = 750.0;
        let mut screen = Canvas::new(xres as u32, yres as u32, 255);
        // screen.upper_left_system = true;
        screen.set_line_color(0, 255, 0);

        // octants 1 and 5
        screen.draw_line(screen.line, 0.0, 0.0, xres - 1.0, yres - 1.0);
        screen.draw_line(screen.line, 0.0, 0.0, xres - 1.0, yres / 2.0);
        screen.draw_line(screen.line, xres - 1.0, yres - 1.0, 0.0, yres / 2.0);

        // octants 8 and 4
        screen.line.blue = 255;
        screen.draw_line(screen.line, 0.0, yres - 1.0, xres - 1.0, 0.0);
        screen.draw_line(screen.line, 0.0, yres - 1.0, xres - 1.0, yres / 2.0);
        screen.draw_line(screen.line, xres - 1.0, 0.0, 0.0, yres / 2.0);

        // octants 2 and 6
        screen.set_line_color(255, 0, 0);
        screen.draw_line(screen.line, 0.0, 0.0, xres / 2.0, yres - 1.0);
        screen.draw_line(screen.line, xres - 1.0, yres - 1.0, xres / 2.0, 0.0);

        // octants 7 and 3
        screen.line.blue = 255;
        screen.draw_line(screen.line, 0.0, yres - 1.0, xres / 2.0, 0.0);
        screen.draw_line(screen.line, xres - 1.0, 0.0, xres / 2.0, yres - 1.0);

        // horizontal and vertical
        screen.set_line_color(255, 255, 0);
        screen.draw_line(screen.line, 0.0, yres / 2.0, xres - 1.0, yres / 2.0);
        screen.draw_line(screen.line, xres / 2.0, 0.0, xres / 2.0, yres - 1.0);

        // saving
        // screen.animation("test")
        screen.display()?;
        screen.save_binary("pics/binary.ppm")?;
        screen.save_ascii("pics/ascii.ppm")?;
        screen.save_extension("img.png")
    }
}
