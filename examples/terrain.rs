use gartus::prelude::*;
use rand::Rng;

const MIN_HEIGHT: f64 = 50.0;
const MAX_HEIGHT: f64 = 500.0;

fn main() {
    // Terrain dimensions (width and height)
    let width = 500;
    let height = 500;
    // Create a Canvas for rendering
    let mut canvas = Canvas::new_with_bg(width, height, 255, Rgb::WHITE);

    let mut points = Matrix::with_capacity(500, 500);

    // Generate a height map using the Diamond-Square algorithm
    let height_map = generate_terrain_height_map_diamond_square(
        width.try_into().unwrap(),
        height.try_into().unwrap(),
        0.75,
    );

    let height_map = height_map.iter().flatten().copied().collect::<Vec<f64>>();
    points.fill_data(height_map);

    // dbg!(&height_map);

    // // Create a height map (you'll need to implement or import this)
    // let terrain_height_map = generate_terrain_height_map(width, height);
    //
    // // Render the terrain based on the height map
    // render_terrain(&mut canvas, &height_map);

    // rotate the terrain
    // let mut rotatex = Matrix::rotate_x(0.5);
    // let new = rotatex * points;

    // // Display or save the canvas with the rendered terrain
    // canvas.display();
    // canvas.save_ascii("what.ppm");
}

// fn generate_terrain_height_map(width: usize, height: usize) -> Vec<Vec<f32>> {
//     // Implement your terrain generation algorithm here
//     // Populate a 2D array (height map) with elevation values
//     // Example: Diamond-Square, Perlin Noise, or custom algorithm
//     // Return the height map
//     // ...
// }
//
fn render_terrain(canvas: &mut Canvas<Rgb>, height_map: &Vec<Vec<f64>>) {
    // Loop through the height map and render the terrain
    for y in 0..canvas.height() {
        for x in 0..canvas.width() {
            // Get the height value from the height map
            let height = height_map[y as usize][x as usize];

            // Use the height value to determine terrain color or elevation
            // You can use color mapping or other rendering techniques here
            let color = determine_color(height);

            // Set the pixel color on the canvas
            canvas.plot(&color, (x as i32).into(), (y as i32).into());
        }
    }
}

fn generate_terrain_height_map_diamond_square(
    height: usize,
    width: usize,
    roughness: f64,
) -> Vec<Vec<f64>> {
    let mut height_map: Vec<Vec<f64>> = vec![vec![0.0; height]; width];
    let mut rng = rand::thread_rng();

    let max_index = height - 1;
    // let half_size = max_index / 2;

    // Initialize corner heights
    height_map[0][0] = rng.gen_range(0.0..1.0);
    height_map[0][max_index] = rng.gen_range(0.0..1.0);
    height_map[max_index][0] = rng.gen_range(0.0..1.0);
    height_map[max_index][max_index] = rng.gen_range(0.0..1.0);

    diamond_square_recursive(
        &mut height_map,
        0,
        0,
        max_index,
        max_index,
        roughness,
        &mut rng,
    );

    height_map
}

fn diamond_square_recursive(
    height_map: &mut Vec<Vec<f64>>,
    x1: usize,
    y1: usize,
    x2: usize,
    y2: usize,
    roughness: f64,
    rng: &mut impl Rng,
) {
    if x2 - x1 < 2 {
        return;
    }

    let half_x = (x1 + x2) / 2;
    let half_y = (y1 + y2) / 2;

    // Diamond step
    let diamond_avg =
        (height_map[x1][y1] + height_map[x2][y1] + height_map[x1][y2] + height_map[x2][y2]) / 4.0;
    let diamond_value = diamond_avg + rng.gen_range(-roughness..roughness);
    height_map[half_x][half_y] = diamond_value;

    // Square step
    let square_avg = (height_map[x1][half_y]
        + height_map[x2][half_y]
        + height_map[half_x][y1]
        + height_map[half_x][y2])
        / 4.0;
    let square_value = square_avg + rng.gen_range(-roughness..roughness);
    height_map[x1][half_y] = square_value;
    height_map[x2][half_y] = square_value;
    height_map[half_x][y1] = square_value;
    height_map[half_x][y2] = square_value;

    // Recursive calls
    diamond_square_recursive(height_map, x1, y1, half_x, half_y, roughness, rng);
    diamond_square_recursive(height_map, half_x, y1, x2, half_y, roughness, rng);
    diamond_square_recursive(height_map, x1, half_y, half_x, y2, roughness, rng);
    diamond_square_recursive(height_map, half_x, half_y, x2, y2, roughness, rng);
}

fn perlin_noise(x: f64, y: f64) -> f64 {
    fn fade(t: f64) -> f64 {
        t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    }

    fn lerp(t: f64, a: f64, b: f64) -> f64 {
        a + t * (b - a)
    }

    fn grad(hash: i32, x: f64, y: f64) -> f64 {
        let h = hash & 15;
        let u = if h < 8 { x } else { y };
        let v = if h < 4 {
            y
        } else if h == 12 || h == 14 {
            x
        } else {
            0.0
        };
        (if h & 1 == 0 { u } else { -u }) + (if h & 2 == 0 { v } else { -v })
    }

    const P: [i32; 256] = [
        151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30,
        69, 142, 8, 99, 37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94,
        252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171,
        168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60,
        211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1,
        216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86,
        164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118,
        126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170,
        213, 119, 248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39,
        253, 19, 98, 108, 110, 79, 113, 224, 232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34,
        242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249, 14, 239, 107, 49,
        192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254,
        138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180,
    ];

    let X = x.floor() as i32 & 255;
    let Y = y.floor() as i32 & 255;

    let x_frac = x - x.floor();
    let y_frac = y - y.floor();

    let u = fade(x_frac);
    let v = fade(y_frac);

    let n00 = grad(P[P[X as usize] as usize + Y as usize], x_frac, y_frac);
    let n01 = grad(
        P[P[X as usize] as usize + Y as usize + 1],
        x_frac,
        y_frac - 1.0,
    );
    let n10 = grad(
        P[P[X as usize + 1] as usize + Y as usize],
        x_frac - 1.0,
        y_frac,
    );
    let n11 = grad(
        P[P[X as usize + 1] as usize + Y as usize + 1],
        x_frac - 1.0,
        y_frac - 1.0,
    );

    lerp(v, lerp(u, n00, n10), lerp(u, n01, n11))
}

fn determine_color(height: f64) -> Rgb {
    // Define color stops with heights and corresponding colors
    let color_stops = [
        (0.0, Rgb::new(0, 128, 0)),     // Low terrain (green)
        (0.4, Rgb::new(128, 128, 128)), // Mid terrain (gray)
        (1.0, Rgb::new(255, 255, 255)), // High terrain (white)
    ];

    // Find the two nearest color stops based on the normalized height
    let (low_height, low_color) =
        color_stops
            .iter()
            .fold((0.0, Rgb::default()), |acc, &(stop_height, stop_color)| {
                if stop_height <= height {
                    (stop_height, stop_color)
                } else {
                    acc
                }
            });

    let (high_height, high_color) =
        color_stops
            .iter()
            .fold((1.0, Rgb::default()), |acc, &(stop_height, stop_color)| {
                if stop_height >= height {
                    (stop_height, stop_color)
                } else {
                    acc
                }
            });

    // Interpolate between the two nearest colors using a smoothstep function
    let t = smoothstep(low_height, high_height, height);

    // Interpolate RGB components
    let r = (low_color.red as f64 + t * (high_color.red as f64 - low_color.red as f64)) as u8;
    let g = (low_color.green as f64 + t * (high_color.green as f64 - low_color.green as f64)) as u8;
    let b = (low_color.blue as f64 + t * (high_color.blue as f64 - low_color.blue as f64)) as u8;

    Rgb::new(r, g, b)
}

// Smoothstep function for smooth interpolation
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// Clamp a value within a range
fn clamp(x: f64, min: f64, max: f64) -> f64 {
    if x < min {
        min
    } else if x > max {
        max
    } else {
        x
    }
}
