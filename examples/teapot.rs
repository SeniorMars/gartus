use gartus::prelude::*;
#[path = "data/teapot.rs"]
mod teapot_data;
use teapot_data::{TEAPOT_PATCHES, TEAPOT_VERTICES};

fn main() {
    let width = 800;
    let height = 800;
    let mut canvas = Canvas::new_with_bg(width, height, Rgb::new(24, 26, 27));
    canvas.set_wrapped(false);

    println!("Generating Utah Teapot...");

    let mut matrix = PolygonMatrix::new();

    for patch_indices in &TEAPOT_PATCHES {
        let mut controls = [[(0.0, 0.0, 0.0); 4]; 4];
        for (i, row) in controls.iter_mut().enumerate() {
            for (j, control) in row.iter_mut().enumerate() {
                // Indices are already 0-based in TEAPOT_PATCHES
                *control = TEAPOT_VERTICES[patch_indices[i * 4 + j]];
            }
        }
        // Use 8 steps for a smooth surface
        matrix.add_bezier_surface(controls, 8);
    }

    // Teapot is naturally ~6.5 units wide (spout to handle) and ~3 units high.
    // After rotate_x(-90), Z (0..3.15) becomes Y (height).
    // X (-3..3.5) remains X (width).
    // To center:
    // X center is approx 0.25. Scaled (100) -> 25. To get to 400: 400 - 25 = 375.
    // Y center is approx 1.57. Scaled (100) -> 157. To get to 400: 400 - 157 = 243.
    let transform = Matrix::translate(375.0, 240.0, 0.0)
        * Matrix::rotate_x(-90.0)
        * Matrix::rotate_y(15.0)
        * Matrix::scale(100.0, 100.0, 100.0);

    let final_matrix = matrix.apply(&transform);

    canvas.set_line_pixel(Rgb::new(255, 200, 100)); // Golden teapot
    canvas.draw_polygons(&final_matrix);

    println!("Rendering and saving...");
    canvas
        .save_extension("pics/teapot.png")
        .expect("Could not save teapot.png");
    println!("Done! Saved to pics/teapot.png");
}
