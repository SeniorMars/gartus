use gartus::prelude::*;
use std::f64::consts::PI;

fn main() {
    let width = 800;
    let height = 800;
    let mut canvas = Canvas::new_with_bg(width, height, Rgb::new(24, 26, 27));
    canvas.wrapped = false;

    println!("Generating Advanced Shapes...");

    // 1. Surface of Revolution (Vase/Wine Glass profile)
    println!("Building Revolution Surface (Vase)...");
    let mut vase_edges = PolygonMatrix::new();
    let mut profile = Vec::new();
    for i in 0..=50 {
        let t = i as f64 / 50.0;
        let y = t * 300.0 - 150.0;
        // Vase-like profile: x = sin(y) + constant
        let x = (y * 0.05).sin() * 40.0 + 60.0;
        profile.push((x, y));
    }
    vase_edges.add_revolution_surface(&profile, 40);
    
    let vase_transform = Matrix::translate(200.0, 400.0, 0.0) * Matrix::rotate_x(20.0);
    let vase_final = vase_edges.apply(&vase_transform);
    canvas.set_line_pixel(Rgb::new(0, 200, 255));
    canvas.draw_polygons(&vase_final);

    // 2. Icosahedron
    println!("Building Icosahedron...");
    let mut ico_edges = PolygonMatrix::new();
    ico_edges.add_icosahedron((0.0, 0.0, 0.0), 80.0);
    let ico_transform = Matrix::translate(600.0, 200.0, 0.0) * Matrix::rotate_y(45.0) * Matrix::rotate_x(20.0);
    let ico_final = ico_edges.apply(&ico_transform);
    canvas.set_line_pixel(Rgb::new(255, 100, 0));
    canvas.draw_polygons(&ico_final);

    // 3. Dodecahedron
    println!("Building Dodecahedron...");
    let mut dodeca_edges = PolygonMatrix::new();
    dodeca_edges.add_dodecahedron((0.0, 0.0, 0.0), 70.0);
    let dodeca_transform = Matrix::translate(600.0, 600.0, 0.0) * Matrix::rotate_z(30.0) * Matrix::rotate_x(30.0);
    let dodeca_final = dodeca_edges.apply(&dodeca_transform);
    canvas.set_line_pixel(Rgb::new(100, 255, 100));
    canvas.draw_polygons(&dodeca_final);

    // 4. Torus Knot (Parametric Revolution Surface)
    println!("Building Torus Knot...");
    let mut knot_edges = PolygonMatrix::new();
    let mut knot_profile = Vec::new();
    for i in 0..=30 {
        let t = i as f64 / 30.0 * 2.0 * PI;
        let x = (t.cos() * 2.0 + 3.0) * 15.0;
        let y = t.sin() * 30.0;
        knot_profile.push((x, y));
    }
    knot_edges.add_revolution_surface(&knot_profile, 60);
    let knot_transform = Matrix::translate(200.0, 700.0, 0.0) * Matrix::rotate_x(45.0);
    let knot_final = knot_edges.apply(&knot_transform);
    canvas.set_line_pixel(Rgb::new(255, 200, 0));
    canvas.draw_polygons(&knot_final);

    println!("Rendering and saving...");
    canvas.save_extension("pics/advanced_shapes.png").expect("Could not save image");
    println!("Done! Saved to advanced_shapes.png");
}
