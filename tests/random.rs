use gartus::{
    gmath::helpers::polar_to_xy,
    gmath::matrix::Matrix,
    graphics::{
        colors::Rgb,
        display::Canvas,
    },
};

#[cfg(feature = "colors")]
use gartus::graphics::colors::Hsl;

#[cfg(feature = "colors")]
#[test]
fn circle() {
    let mut circle = Canvas::new_with_bg(500, 500, 255, Hsl::new(0, 100, 100));
    circle.set_line_color_hsl(5, 99, 26);
    let color = Hsl::new(5, 99, 26);
    let mut matrix = Matrix::new(4, 0, Vec::new());
    matrix.add_circle(249.0, 249.0, 249.0, 50.0, 0.0001);
    circle.draw_lines(&matrix);
    circle.fill(249, 249, &color, &color);
    circle.display().expect("Could not draw circle")
}

#[test]
fn donut() {
    let mut t = 0.0;
    let mut donut = Canvas::new(500, 500, 255, Rgb::default());
    let colors = vec![
        Rgb::RED,
        Rgb::MAGENTA,
        Rgb::BLUE,
        Rgb::CYAN,
        Rgb::GREEN,
        Rgb::WHITE,
        Rgb::YELLOW,
    ];
    for _ in 0..6 {
        for color in &colors {
            // very inefficient
            let mut matrix = Matrix::new(4, 0, Vec::new());
            let mut x = 249.0;
            let mut y = 249.0;
            t += 10.0;
            let (dx, dy) = polar_to_xy(10.0, t);
            x += dx;
            y += dy;
            matrix.add_circle(x, y, 0.0, 100.0, 0.0001);
            donut.set_line_rgb(*color);
            donut.draw_lines(&matrix);
        }
    }
    donut.display().expect("Could not draw circle");
    donut
        .save_extension("./pics/donut.png")
        .expect("Could not save donut")
}

#[test]
fn spirograph() {
    let mut circle = Canvas::new(500, 500, 255, Rgb::default());
    let colors = vec![
        Rgb::RED,
        Rgb::MAGENTA,
        Rgb::BLUE,
        Rgb::CYAN,
        Rgb::GREEN,
        Rgb::WHITE,
        Rgb::YELLOW,
    ];
    let mut t = 0.0;
    let mut x = 249.0;
    let mut y = 300.0;
    for _ in 0..6 {
        for color in &colors {
            // very inefficient
            let mut matrix = Matrix::new(4, 0, Vec::new());
            let (dx, dy) = polar_to_xy(10.0, t);
            x += dx;
            y += dy;
            t -= 10.0;
            matrix.add_circle(x, y, 0.0, 100.0, 0.0001);
            circle.set_line_rgb(*color);
            circle.draw_lines(&matrix);
        }
    }
    circle.display().expect("Could not draw circle");
    circle
        .save_extension("spiro.png")
        .expect("Could not save spiro")
}

#[test]
#[cfg(feature = "colors")]
fn hermite_test() {
    let color = Hsl::new(5, 99, 26);
    let mut hermite = Canvas::new(500, 500, 255, color);
    let mut matrix = Matrix::new(4, 0, Vec::new());
    matrix.add_hermite(
        (150.0, 150.0),
        (350.0, 150.0),
        (-100.0, -100.0),
        (100.0, 150.0),
    );
    hermite.draw_lines(&matrix);
    hermite.display().expect("Could not draw circle")
}
