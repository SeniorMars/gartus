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

    // 2. Build EdgeMatrix for the grid
    println!("Building grid...");
    let mut edges = EdgeMatrix::new();
    let scale = 500.0; // Horizontal extent
    let h_scale = 300.0; // Vertical extent (peaks)

    for y in 0..GRID_SIZE {
        for x in 0..GRID_SIZE {
            let px = (x as f64 / (GRID_SIZE - 1) as f64 - 0.5) * scale;
            let pz = (y as f64 / (GRID_SIZE - 1) as f64 - 0.5) * scale;
            let py = height_map[x][y] * h_scale;

            if x + 1 < GRID_SIZE {
                let nx = ((x + 1) as f64 / (GRID_SIZE - 1) as f64 - 0.5) * scale;
                let ny = height_map[x + 1][y] * h_scale;
                edges.push_edge(px, py, pz, nx, ny, pz);
            }
            if y + 1 < GRID_SIZE {
                let nz = ((y + 1) as f64 / (GRID_SIZE - 1) as f64 - 0.5) * scale;
                let ny = height_map[x][y + 1] * h_scale;
                edges.push_edge(px, py, pz, px, ny, nz);
            }
        }
    }

    println!("Applying transformations and rendering...");
    // 3. Manual Projection and Rendering with Height-based Coloring
    let mut iter = edges.iter_points();
    while let Some(p0) = iter.next() {
        let p1 = iter.next().unwrap();

        // Cabinet projection: x' = x + 0.5 * z * cos(45), y' = y + 0.5 * z * sin(45)
        let project = |p: &[f64]| {
            let x = p[0] + 0.5 * p[2] * 0.707;
            let y = p[1] + 0.5 * p[2] * 0.707;
            // Shift base further down (y=650) and center X
            (x + 400.0, 650.0 - y)
        };

        let (x0, y0) = project(p0);
        let (x1, y1) = project(p1);

        // Calculate color based on height (p[1] is the vertical axis)
        // Since we normalized height_map to [0, 1] and scaled by h_scale,
        // we can just use the normalized height directly.
        let avg_h = (p0[1] + p1[1]) / (2.0 * h_scale);
        let color = determine_color(avg_h);

        canvas.draw_line(color, x0, y0, x1, y1);
    }

    println!("Rendering and saving...");
    canvas
        .save_extension("terrain.png")
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
    let color_stops = [
        (0.0, Rgb::new(0, 50, 200)),    // Water (Deep blue)
        (0.2, Rgb::new(0, 150, 0)),     // Lowlands (Green)
        (0.4, Rgb::new(100, 100, 50)),  // Mountains (Brownish)
        (0.7, Rgb::new(180, 180, 180)), // High peaks (Gray)
        (1.0, Rgb::new(255, 255, 255)), // Snow (White)
    ];

    let mut low = color_stops[0];
    let mut high = color_stops[color_stops.len() - 1];

    for i in 0..color_stops.len() - 1 {
        if height >= color_stops[i].0 && height <= color_stops[i + 1].0 {
            low = color_stops[i];
            high = color_stops[i + 1];
            break;
        }
    }

    let t = if high.0 == low.0 {
        0.0
    } else {
        (height - low.0) / (high.0 - low.0)
    };
    let r = (low.1.red as f64 + t * (high.1.red as f64 - low.1.red as f64)) as u8;
    let g = (low.1.green as f64 + t * (high.1.green as f64 - low.1.green as f64)) as u8;
    let b = (low.1.blue as f64 + t * (high.1.blue as f64 - low.1.blue as f64)) as u8;
    Rgb::new(r, g, b)
}
