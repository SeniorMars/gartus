use gartus::prelude::*;
use rand::Rng;

const GRID_SIZE: usize = 65; // 2^6 + 1

fn main() {
    let width = 800;
    let height = 800;
    // Dark background to make peaks pop
    let mut canvas = Canvas::new_with_bg(width, height, Rgb::BLACK);
    canvas.wrapped = false; // Disable wrapping to prevent top-to-bottom overlap

    // 1. Generate height map using Diamond-Square
    println!("Generating random terrain...");
    let mut height_map = generate_terrain(GRID_SIZE, 1.2);

    // Normalize height map to [0, 1] for predictable scaling
    let mut min_h = f64::MAX;
    let mut max_h = f64::MIN;
    for row in &height_map {
        for &h in row {
            if h < min_h {
                min_h = h;
            }
            if h > max_h {
                max_h = h;
            }
        }
    }
    for row in &mut height_map {
        for h in row {
            *h = (*h - min_h) / (max_h - min_h);
        }
    }

    // 2. Build the terrain mesh
    println!("Building grid...");
    let scale = 650.0; // Increased scale from 500.0
    let h_scale = 350.0; // Increased height from 300.0
    let polygons =
        PolygonMatrix::from_height_map(&height_map, HeightMapOptions::new(scale, scale, h_scale));

    println!("Applying transformations and rendering...");
    // 3. Manual Projection and Rendering with Height-based Coloring
    let cabinet = Matrix::cabinet_projection(45.0);

    let transformed = polygons.apply(&cabinet);

    for (p0, p1, p2) in transformed.iter_triangles() {
        // Center the larger terrain
        let project = |p: &[f64]| (p[0] + 400.0, 700.0 - p[1]);

        let (x0, y0) = project(p0);
        let (x1, y1) = project(p1);
        let (x2, y2) = project(p2);

        // Average height for coloring
        let avg_h = (p0[1] + p1[1] + p2[1]) / (3.0 * h_scale);
        let color = determine_color(avg_h);

        canvas.draw_line(color, x0, y0, x1, y1);
        canvas.draw_line(color, x1, y1, x2, y2);
        canvas.draw_line(color, x2, y2, x0, y0);
    }

    println!("Rendering and saving...");
    canvas
        .save_extension("pics/terrain.png")
        .expect("Could not save to terrain.png");
    println!("Done! Saved to terrain.png");
}

/// Simple Diamond-Square implementation for terrain generation
fn generate_terrain(size: usize, roughness: f64) -> Vec<Vec<f64>> {
    let mut map = vec![vec![0.0; size]; size];
    let mut rng = rand::thread_rng();

    map[0][0] = rng.gen_range(-1.0..1.0);
    map[0][size - 1] = rng.gen_range(-1.0..1.0);
    map[size - 1][0] = rng.gen_range(-1.0..1.0);
    map[size - 1][size - 1] = rng.gen_range(-1.0..1.0);

    let mut step = size - 1;
    let mut r = roughness;
    while step > 1 {
        let half = step / 2;
        for x in (half..size).step_by(step) {
            for y in (half..size).step_by(step) {
                let avg = (map[x - half][y - half]
                    + map[x + half][y - half]
                    + map[x - half][y + half]
                    + map[x + half][y + half])
                    / 4.0;
                map[x][y] = avg + rng.gen_range(-r..r);
            }
        }
        for x in (0..size).step_by(half) {
            let start_y = if (x / half).is_multiple_of(2) {
                half
            } else {
                0
            };
            for y in (start_y..size).step_by(step) {
                let mut sum = 0.0;
                let mut count = 0.0;
                if x >= half {
                    sum += map[x - half][y];
                    count += 1.0;
                }
                if x + half < size {
                    sum += map[x + half][y];
                    count += 1.0;
                }
                if y >= half {
                    sum += map[x][y - half];
                    count += 1.0;
                }
                if y + half < size {
                    sum += map[x][y + half];
                    count += 1.0;
                }
                map[x][y] = (sum / count) + rng.gen_range(-r..r);
            }
        }
        step /= 2;
        r /= 2.0;
    }
    map
}

fn determine_color(height: f64) -> Rgb {
    let ramp = ColorRamp::new(vec![
        (0.0, Rgb::new(0, 50, 200)),    // Water (Deep blue)
        (0.2, Rgb::new(0, 150, 0)),     // Lowlands (Green)
        (0.4, Rgb::new(100, 100, 50)),  // Mountains (Brownish)
        (0.7, Rgb::new(180, 180, 180)), // High peaks (Gray)
        (1.0, Rgb::new(255, 255, 255)), // Snow (White)
    ]);
    ramp.sample(height)
}
