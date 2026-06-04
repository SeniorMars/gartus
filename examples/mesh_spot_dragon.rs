use gartus::{external, prelude::*};
use std::{error::Error, f64::consts::PI, fs};

const WIDTH: u32 = 1000;
const HEIGHT: u32 = 900;
const FRAMES: usize = 72;
const CAMERA_DISTANCE: f64 = 900.0;
const FOCAL_LENGTH: f64 = 720.0;

struct Model {
    name: &'static str,
    mesh: PolygonMatrix,
    normalize: Matrix,
    placement: Matrix,
    color: Rgb,
    stride: usize,
}

#[derive(Clone, Copy)]
struct ScreenPoint {
    x: f64,
    y: f64,
    depth: f64,
}

struct Segment {
    a: ScreenPoint,
    b: ScreenPoint,
    color: Rgb,
}

fn main() {
    if let Err(err) = render() {
        eprintln!("could not render Spot and XYZ Dragon:\n{err}");
        std::process::exit(1);
    }
}

fn render() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final")?;

    let models = load_models()?;
    let scanner = build_scanner_rig();
    let mut recorder = FrameRecorder::new("anim", "spot-dragon-").with_delay(3);
    let mut preview = Canvas::new_with_bg(WIDTH, HEIGHT, background());

    for frame in 0..FRAMES {
        let canvas = render_frame(frame, &models, &scanner);
        if frame == 19 {
            preview = canvas.clone();
        }
        recorder.capture(&canvas)?;
    }

    preview.save_extension("final/spot_dragon_mesh.png")?;
    recorder.encode_gif("final/spot_dragon_mesh.gif")?;

    println!("Saved final/spot_dragon_mesh.png and final/spot_dragon_mesh.gif");
    Ok(())
}

fn load_models() -> Result<Vec<Model>, Box<dyn Error>> {
    let spot = external::meshify("examples/data/meshes/spot.obj")?;
    let dragon = external::meshify("examples/data/meshes/xyzrgb_dragon.obj")?;

    println!(
        "Loaded Spot: {} triangles; XYZ Dragon: {} triangles",
        spot.triangle_count(),
        dragon.triangle_count()
    );

    Ok(vec![
        Model {
            name: "spot",
            normalize: external::normalize_mesh_transform(&spot, 280.0, external::MeshUpAxis::Z),
            placement: Matrix::translate(-245.0, 24.0, -25.0) * Matrix::rotate_y(-18.0),
            mesh: spot,
            color: Rgb::new(255, 211, 96),
            stride: 2,
        },
        Model {
            name: "dragon",
            normalize: external::normalize_mesh_transform(&dragon, 345.0, external::MeshUpAxis::Z),
            placement: Matrix::translate(245.0, 12.0, 35.0) * Matrix::rotate_y(148.0),
            mesh: dragon,
            color: Rgb::new(84, 230, 255),
            stride: 13,
        },
    ])
}

fn render_frame(frame: usize, models: &[Model], scanner: &EdgeMatrix) -> Canvas {
    let mut canvas = Canvas::new_with_bg(WIDTH, HEIGHT, background());
    canvas.wrapped = false;
    canvas.upper_left_origin = true;
    canvas.set_line_width(1.0);

    let t = frame as f64 / FRAMES as f64;

    let camera_orbit = Matrix::rotate_y(26.0 * (t * PI * 2.0).sin())
        * Matrix::rotate_z(2.5 * (t * PI * 2.0).cos());

    draw_scanner_rig(&mut canvas, scanner, &camera_orbit, t);
    draw_sweep_bands(&mut canvas, &camera_orbit, t);

    let mut segments = Vec::new();
    for model in models {
        let model_spin = if model.name == "dragon" {
            -180.0
        } else {
            120.0
        };
        let transform = camera_orbit.clone()
            * model.placement.clone()
            * Matrix::rotate_y(t * model_spin)
            * model.normalize.clone();
        collect_mesh_segments(
            &mut segments,
            &model.mesh.apply(&transform),
            model.color,
            model.stride,
            t,
        );
    }

    segments.sort_by(|a, b| {
        let da = (a.a.depth + a.b.depth) * 0.5;
        let db = (b.a.depth + b.b.depth) * 0.5;
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
    });

    for segment in segments {
        canvas.draw_line(
            segment.color,
            segment.a.x,
            segment.a.y,
            segment.b.x,
            segment.b.y,
        );
    }

    canvas
}

fn collect_mesh_segments(
    segments: &mut Vec<Segment>,
    mesh: &PolygonMatrix,
    base_color: Rgb,
    stride: usize,
    t: f64,
) {
    for (idx, (p0, p1, p2)) in mesh.iter_triangles().enumerate() {
        if idx % stride != 0 {
            continue;
        }
        let Some(a) = project(p0) else {
            continue;
        };
        let Some(b) = project(p1) else {
            continue;
        };
        let Some(c) = project(p2) else {
            continue;
        };
        let color = depth_color(base_color, (a.depth + b.depth + c.depth) / 3.0, t);
        segments.push(Segment { a, b, color });
        segments.push(Segment { a: b, b: c, color });
        segments.push(Segment { a: c, b: a, color });
    }
}

fn project(point: &[f64]) -> Option<ScreenPoint> {
    let depth = point[2] + CAMERA_DISTANCE;
    if depth < 80.0 {
        return None;
    }
    let scale = FOCAL_LENGTH / depth;
    Some(ScreenPoint {
        x: f64::from(WIDTH) * 0.5 + point[0] * scale,
        y: f64::from(HEIGHT) * 0.54 - point[1] * scale,
        depth,
    })
}

fn build_scanner_rig() -> EdgeMatrix {
    let mut rig = EdgeMatrix::new();
    add_grid(&mut rig, -210.0, 470.0, 18);
    add_cage(&mut rig, (-245.0, -8.0, -25.0), (190.0, 245.0, 190.0));
    add_cage(&mut rig, (245.0, -2.0, 35.0), (245.0, 210.0, 210.0));
    add_wave_rail(&mut rig, -440.0, 282.0, -160.0, 0.0);
    add_wave_rail(&mut rig, -440.0, 282.0, 160.0, PI);
    rig
}

fn add_grid(rig: &mut EdgeMatrix, y: f64, extent: f64, steps: usize) {
    for i in 0..=steps {
        let p = -extent + i as f64 * extent * 2.0 / steps as f64;
        rig.push_edge(-extent, y, p, extent, y, p);
        rig.push_edge(p, y, -extent, p, y, extent);
    }
}

fn add_cage(rig: &mut EdgeMatrix, center: (f64, f64, f64), size: (f64, f64, f64)) {
    let (cx, cy, cz) = center;
    let (sx, sy, sz) = (size.0 * 0.5, size.1 * 0.5, size.2 * 0.5);
    let corners = [
        (cx - sx, cy - sy, cz - sz),
        (cx + sx, cy - sy, cz - sz),
        (cx + sx, cy - sy, cz + sz),
        (cx - sx, cy - sy, cz + sz),
        (cx - sx, cy + sy, cz - sz),
        (cx + sx, cy + sy, cz - sz),
        (cx + sx, cy + sy, cz + sz),
        (cx - sx, cy + sy, cz + sz),
    ];
    for &(a, b) in &[
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ] {
        rig.push_edge_tuple(corners[a], corners[b]);
    }
}

fn add_wave_rail(rig: &mut EdgeMatrix, x0: f64, x1: f64, z: f64, phase: f64) {
    let steps = 54;
    let mut prev = None;
    for i in 0..=steps {
        let u = i as f64 / steps as f64;
        let x = x0 + (x1 - x0) * u;
        let y = 118.0 + (u * PI * 6.0 + phase).sin() * 24.0;
        let p = (x, y, z);
        if let Some(prev) = prev {
            rig.push_edge_tuple(prev, p);
        }
        prev = Some(p);
    }
}

fn draw_scanner_rig(canvas: &mut Canvas, rig: &EdgeMatrix, transform: &Matrix, t: f64) {
    let transformed = rig.apply(transform);
    for (p0, p1) in transformed.iter_edges() {
        let Some(a) = project(p0) else {
            continue;
        };
        let Some(b) = project(p1) else {
            continue;
        };
        canvas.draw_line(
            depth_color(Rgb::new(77, 98, 174), (a.depth + b.depth) * 0.5, t),
            a.x,
            a.y,
            b.x,
            b.y,
        );
    }
}

fn draw_sweep_bands(canvas: &mut Canvas, transform: &Matrix, t: f64) {
    let mut bands = EdgeMatrix::new();
    for model_x in [-245.0, 245.0] {
        let sweep = -120.0 + ((t * 2.0 + if model_x < 0.0 { 0.0 } else { 0.5 }) % 1.0) * 240.0;
        for radius in [72.0, 112.0, 152.0] {
            add_ellipse(
                &mut bands,
                model_x,
                sweep,
                if model_x < 0.0 { -25.0 } else { 35.0 },
                radius,
                radius * 0.56,
            );
        }
    }
    let transformed = bands.apply(transform);
    for (p0, p1) in transformed.iter_edges() {
        let Some(a) = project(p0) else {
            continue;
        };
        let Some(b) = project(p1) else {
            continue;
        };
        canvas.draw_line(Rgb::new(116, 255, 191), a.x, a.y, b.x, b.y);
    }
}

fn add_ellipse(rig: &mut EdgeMatrix, cx: f64, cy: f64, cz: f64, rx: f64, rz: f64) {
    let steps = 48;
    let mut prev = (cx + rx, cy, cz);
    for i in 1..=steps {
        let a = i as f64 / steps as f64 * PI * 2.0;
        let p = (cx + a.cos() * rx, cy, cz + a.sin() * rz);
        rig.push_edge_tuple(prev, p);
        prev = p;
    }
}

fn depth_color(color: Rgb, depth: f64, t: f64) -> Rgb {
    let pulse = 0.82 + 0.16 * (t * PI * 2.0).sin();
    let factor = (1.14 - depth / 1500.0).clamp(0.38, 1.0) * pulse;
    Rgb::new(
        scale_channel(color.red, factor),
        scale_channel(color.green, factor),
        scale_channel(color.blue, factor),
    )
}

fn scale_channel(channel: u8, factor: f64) -> u8 {
    (f64::from(channel) * factor).round().clamp(0.0, 255.0) as u8
}

fn background() -> Rgb {
    Rgb::new(4, 7, 15)
}
