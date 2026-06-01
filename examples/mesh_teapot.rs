use gartus::external;
use gartus::prelude::*;

fn main() {
    let width = 800;
    let height = 800;
    let mut canvas = Canvas::new_with_bg(width, height, Rgb::new(24, 26, 27));
    canvas.wrapped = false;

    println!("Loading Utah Teapot mesh...");
    let matrix = external::meshify("examples/data/meshes/teapot.obj").expect("Could not load mesh");

    let transform = Matrix::translate(375.0, 240.0, 0.0)
        * Matrix::rotate_x(-90.0)
        * Matrix::rotate_y(15.0)
        * Matrix::scale(100.0, 100.0, 100.0);

    let final_matrix = matrix.apply(&transform);

    canvas.set_line_pixel(Rgb::new(255, 200, 100));
    canvas.draw_polygons(&final_matrix);

    println!("Rendering and saving...");
    canvas
        .save_extension("pics/mesh_teapot.png")
        .expect("Could not save mesh_teapot.png");
    println!("Done! Saved to pics/mesh_teapot.png");
}
