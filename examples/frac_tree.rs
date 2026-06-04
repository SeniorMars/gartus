#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use gartus::{
    graphics::turtle::Turtle,
    prelude::{Canvas, Rgb},
};
use std::{error::Error, fs};

const WIDTH: u32 = 1000;
const HEIGHT: u32 = 900;
const MAX_DEPTH: u32 = 13;

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final/fractals")?;

    let mut canvas = Canvas::new_with_bg(WIDTH, HEIGHT, Rgb::new(5, 8, 14));
    canvas.upper_left_origin = true;

    let mut turtle = Turtle::new(
        Rgb::new(126, 78, 42),
        -90.0,
        f64::from(WIDTH) * 0.5,
        f64::from(HEIGHT) - 42.0,
    );
    turtle.pen_down();

    draw_fractal_tree(&mut turtle, &mut canvas, 190.0, MAX_DEPTH, MAX_DEPTH);
    canvas.save_extension("final/fractals/fractal_tree.png")?;
    println!("saved final/fractals/fractal_tree.png");

    Ok(())
}

fn draw_fractal_tree(
    turtle: &mut Turtle,
    canvas: &mut Canvas,
    branch_length: f64,
    depth: u32,
    max_depth: u32,
) {
    if depth == 0 || branch_length < 2.0 {
        return;
    }

    let progress = f64::from(max_depth - depth) / f64::from(max_depth);
    turtle.set_color(branch_color(progress));
    canvas.set_line_width((6.0 * (1.0 - progress)).max(1.0));
    turtle.draw_forward(canvas, branch_length);

    let branch_sway = 8.0 * (progress * std::f64::consts::PI).sin();
    let left_angle = 18.0 + branch_sway + f64::from(depth % 3) * 3.0;
    let right_angle = 24.0 - branch_sway * 0.35 + f64::from((depth + 1) % 4) * 2.0;

    turtle.push_state();
    turtle.rotate_left(left_angle);
    draw_fractal_tree(turtle, canvas, branch_length * 0.71, depth - 1, max_depth);
    turtle
        .pop_state()
        .expect("left branch state should be present");

    turtle.push_state();
    turtle.rotate_right(right_angle);
    draw_fractal_tree(turtle, canvas, branch_length * 0.67, depth - 1, max_depth);
    turtle
        .pop_state()
        .expect("right branch state should be present");

    if depth < 5 {
        turtle.push_state();
        turtle.rotate_left(left_angle * 0.35 - right_angle * 0.25);
        draw_fractal_tree(turtle, canvas, branch_length * 0.42, depth - 1, max_depth);
        turtle
            .pop_state()
            .expect("center branch state should be present");
    }
}

fn branch_color(progress: f64) -> Rgb {
    let trunk = (1.0 - progress).powf(1.8);
    let leaf = progress.powf(0.75);
    Rgb::new(
        channel(91.0 * trunk + 82.0 * leaf),
        channel(54.0 * trunk + 204.0 * leaf),
        channel(32.0 * trunk + 126.0 * leaf),
    )
}

fn channel(value: f64) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}
