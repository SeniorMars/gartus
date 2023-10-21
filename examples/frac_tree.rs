use gartus::{
    graphics::turtle::Turtle,
    prelude::{Canvas, Rgb},
};


fn draw_fractal_tree(turtle: &mut Turtle<Rgb>, canvas: &mut Canvas<Rgb>, branch_length: f64, angle: f64, depth: u32) {
    if depth == 0 {
        return;
    }

    // Draw the current branch
    turtle.move_turtle(canvas, branch_length);
    turtle.set_color(Rgb::new(139, 69, 19));  // Brown color (adjust as needed)

    // Recursive left branch
    turtle.push_state();  // Save the current turtle state
    turtle.rotate_left(angle);
    draw_fractal_tree(turtle, canvas, branch_length * 0.7, angle, depth - 1);
    turtle.pop_state();  // Restore the previous turtle state

    // Recursive right branch
    turtle.push_state();  // Save the current turtle state
    turtle.rotate_right(angle);
    draw_fractal_tree(turtle, canvas, branch_length * 0.7, angle, depth - 1);
    turtle.pop_state();  // Restore the previous turtle state
}

fn main() {
    let width = 800;  // Adjust the canvas size as needed
    let height = 800;
    let mut canvas = Canvas::new_with_bg(width, height, 255, Rgb::new(0, 0, 0));
    let mut turtle = Turtle::new(Rgb::new(0, 128, 0), 90.0, width / 2, height);
    turtle.set_draw_mode(true);

    // Set initial parameters
    let branch_length = 200.0;
    let angle = 25.0;
    let depth = 20;  // Adjust the depth for more or fewer iterations

    // Position the turtle at the bottom-center of the canvas
    turtle.set_position(width / 2, height);

    // Draw the fractal tree
    draw_fractal_tree(&mut turtle, &mut canvas, branch_length, angle, depth);

    // Save the canvas to a file
    canvas.display().expect("Image is displayable");
    // canvas.save_binary("./pics/fractal_tree.png").expect("Image is writeable");
}

