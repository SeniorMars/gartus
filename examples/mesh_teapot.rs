use gartus::external;
use gartus::prelude::*;

fn main() {
    let width = 800;
    let height = 800;
    let mut canvas = Canvas::new_with_bg(width, height, Rgb::new(24, 26, 27));
    canvas.set_wrapped(false);

    println!("Loading Utah Teapot mesh...");
    let matrix = external::meshify("examples/data/meshes/teapot.obj").expect("Could not load mesh");

    let transform = Matrix::translate(f64::from(width) * 0.5, f64::from(height) * 0.5, 0.0)
        * Matrix::rotate_y(15.0)
        * external::normalize_mesh_transform(&matrix, 560.0, external::MeshUpAxis::Z);

    let final_matrix = matrix.apply(&transform);

    canvas.set_line_pixel(Rgb::new(255, 200, 100));
    canvas.draw_polygons(&final_matrix);

    println!("Rendering and saving...");
    canvas
        .save_extension("pics/mesh_teapot.png")
        .expect("Could not save mesh_teapot.png");
    println!("Done! Saved to pics/mesh_teapot.png");
}
