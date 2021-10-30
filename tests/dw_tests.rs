use curves_rs::gmath::matrix::*;
use curves_rs::graphics::colors::*;
use curves_rs::graphics::display::*;
use curves_rs::parser::Parser;
use std::io;

#[test]
fn script_test() {
    let mut dw = Parser::new(
        "./tests/script_transform",
        500,
        500,
        255,
        &Pixel::RGB(RGB::new(0, 255, 0)),
    );
    dw.parse_file()
}

#[test]
fn matrix_test() {
    let mut edge_matrix = Matrix::new(4, 0, Vec::with_capacity(4 * 2));
    // println!("{}", edge_matrix);
    println!("Testing add_edge. Adding (1, 2, 3), (4, 5, 6) m2");
    edge_matrix.add_edge(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    println!("{}", edge_matrix);
    let mut ident = Matrix::identity_matrix(4);
    println!("Testing ident. m1 =");
    println!("{:?}", ident);
    println!("Testing Matrix mult. m1 * m2 =");
    ident *= edge_matrix.clone();
    println!("{:#?}", ident);
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
    m1 *= edge_matrix;
    println!("{}", m1);
    assert_eq!(
        m1,
        Matrix::new(4, 2, vec![40.0, 47.0, 54.0, 7.0, 76.0, 92.0, 108.0, 16.0])
    )
}

#[test]
fn dw_line_test() -> io::Result<()> {
    let xres: f64 = 750.0;
    let yres: f64 = 750.0;
    let mut screen = Canvas::new(xres as u32, yres as u32, 255, Pixel::RGB(RGB::default()));
    // screen.upper_left_system = true;
    screen.set_line_color_rgb(0, 255, 0);

    // octants 1 and 5
    screen.draw_line(screen.line, 0.0, 0.0, xres - 1.0, yres - 1.0);
    screen.draw_line(screen.line, 0.0, 0.0, xres - 1.0, yres / 2.0);
    screen.draw_line(screen.line, xres - 1.0, yres - 1.0, 0.0, yres / 2.0);

    // octants 8 and 4
    screen.set_line_color_rgb(0, 255, 255);
    screen.draw_line(screen.line, 0.0, yres - 1.0, xres - 1.0, 0.0);
    screen.draw_line(screen.line, 0.0, yres - 1.0, xres - 1.0, yres / 2.0);
    screen.draw_line(screen.line, xres - 1.0, 0.0, 0.0, yres / 2.0);

    // octants 2 and 6
    screen.set_line_color_rgb(255, 0, 0);
    screen.draw_line(screen.line, 0.0, 0.0, xres / 2.0, yres - 1.0);
    screen.draw_line(screen.line, xres - 1.0, yres - 1.0, xres / 2.0, 0.0);

    // octants 7 and 3
    screen.set_line_color_rgb(255, 0, 255);
    screen.draw_line(screen.line, 0.0, yres - 1.0, xres / 2.0, 0.0);
    screen.draw_line(screen.line, xres - 1.0, 0.0, xres / 2.0, yres - 1.0);

    // horizontal and vertical
    screen.set_line_color_rgb(255, 255, 0);
    screen.draw_line(screen.line, 0.0, yres / 2.0, xres - 1.0, yres / 2.0);
    screen.draw_line(screen.line, xres / 2.0, 0.0, xres / 2.0, yres - 1.0);

    // saving
    // screen.animation("test")
    screen.display()?;
    screen.save_binary("./pics/binary2.ppm")?;
    screen.save_ascii("./pics/ascii2.ppm")?;
    screen.save_extension("./pics/img.png")
}
