use gartus::{external, prelude::*};
use std::{error::Error, f64::consts::PI};

const WIDTH: u32 = 900;
const HEIGHT: u32 = 900;
const FRAMES: usize = 72;
const TAU: f64 = PI * 2.0;
const CAMERA_DISTANCE: f64 = 980.0;
const FOCAL_LENGTH: f64 = 760.0;
const CHARGE_LINE_Z: f64 = -5600.0;
const LIGHT_DIR: (f64, f64, f64) = (-0.28, -0.54, 0.8);

struct Scene {
    mesh: PolygonMatrix,
    normalize: Matrix,
}

fn main() {
    if let Err(err) = render() {
        eprintln!("could not render Joltik scanline lair:\n{err}");
        std::process::exit(1);
    }
}

fn render() -> Result<(), Box<dyn Error>> {
    let mesh = external::meshify("examples/data/meshes/joltik.obj")?;
    println!("loaded Joltik: {} triangles", mesh.triangle_count());

    let scene = Scene {
        normalize: external::normalize_mesh_transform(&mesh, 565.0, external::MeshUpAxis::Y),
        mesh,
    };

    FrameRecorder::render_gif(
        AnimationRenderOptions::new("anim", "mesh-joltik-", FRAMES, "final/mesh_joltik.gif")
            .delay_cs(3)
            .preview(18, "final/mesh_joltik.png")
            .unique_frame_dir(true),
        |frame| Ok(render_frame(frame, &scene)),
    )?;

    println!("Saved final/mesh_joltik.png and final/mesh_joltik.gif");
    Ok(())
}

fn render_frame(frame: usize, scene: &Scene) -> Canvas {
    let mut canvas = Canvas::new_with_bg(WIDTH, HEIGHT, Rgb::new(5, 7, 12));
    canvas.wrapped = false;
    canvas.set_polygon_color_mode(PolygonColorMode::LineColor);

    let t = frame as f64 / FRAMES as f64;
    draw_web_lair(&mut canvas, t);

    let phase = t * TAU;
    let body_transform = Matrix::translate(0.0, 188.0 + phase.sin() * 8.0, 115.0)
        * Matrix::rotate_z(374.0 + 1.7 * (phase * 1.6).sin())
        * Matrix::rotate_y(12.0 + 5.0 * phase.sin())
        * Matrix::rotate_x(9.0 + 2.5 * (phase * 1.2).cos())
        * scene.normalize.clone();

    draw_lit_projected_triangles(&mut canvas, &scene.mesh, &body_transform, t);
    draw_charge_lines(&mut canvas, t);

    canvas
}

fn draw_web_lair(canvas: &mut Canvas, t: f64) {
    draw_web_rings(canvas, t);
    draw_anchor_spokes(canvas, t);
    draw_web_glints(canvas, t);
}

fn draw_web_rings(canvas: &mut Canvas, t: f64) {
    let center = web_center(t);
    let spokes = 22;
    canvas.set_line_width(1.0);

    for ring in 1..=7 {
        let z = -6200.0 + ring as f64 * 22.0;
        let color = web_color(ring, 0.72);
        let mut prev = web_point(center, ring, spokes - 1, spokes, t);
        for spoke in 0..spokes {
            let next = web_point(center, ring, spoke, spokes, t);
            canvas.draw_line_z(color, screen_point(prev, z), screen_point(next, z));
            prev = next;
        }
    }
}

fn draw_anchor_spokes(canvas: &mut Canvas, t: f64) {
    let center = web_center(t);
    let spokes = 22;
    for spoke in 0..spokes {
        let angle = spoke as f64 / spokes as f64 * TAU;
        if angle.cos().abs() < 0.45 {
            continue;
        }

        let color = if spoke % 3 == 0 {
            Rgb::new(45, 126, 148)
        } else {
            Rgb::new(22, 58, 82)
        };
        let mut prev = web_spoke_point(center, 1, spoke, spokes, t);
        for ring in 2..=8 {
            let next = web_spoke_point(center, ring, spoke, spokes, t);
            canvas.draw_line_z(
                color,
                screen_point(prev, -6120.0 + ring as f64 * 16.0),
                screen_point(next, -6120.0 + ring as f64 * 16.0),
            );
            prev = next;
        }
    }
}

fn draw_web_glints(canvas: &mut Canvas, t: f64) {
    let center = web_center(t);
    let spokes = 22;
    canvas.set_line_width(1.0);

    for i in 0..18 {
        let ring = 2 + i % 6;
        let spoke = (i * 5 + (t * 12.0) as usize) % spokes;
        let p0 = web_point(center, ring, spoke, spokes, t);
        let p1 = web_point(center, ring, (spoke + 1) % spokes, spokes, t);
        let u0 = 0.24 + 0.28 * ((i * 3) % 5) as f64 / 4.0;
        let u1 = (u0 + 0.11).min(0.9);
        let a = lerp_point(p0, p1, u0);
        let b = lerp_point(p0, p1, u1);
        let flicker = 0.45 + 0.4 * (t * TAU * 2.0 + i as f64 * 1.4).sin().max(0.0);
        canvas.draw_line_z(
            dim(Rgb::new(132, 244, 255), flicker),
            screen_point(a, -5750.0),
            screen_point(b, -5750.0),
        );
    }
}

fn web_center(t: f64) -> (f64, f64) {
    (
        f64::from(WIDTH) * 0.5 + (t * TAU * 0.7).cos() * 8.0,
        f64::from(HEIGHT) * 0.58 + (t * TAU).sin() * 10.0,
    )
}

fn web_point(center: (f64, f64), ring: usize, spoke: usize, spokes: usize, t: f64) -> (f64, f64) {
    let angle = spoke as f64 / spokes as f64 * TAU;
    let radius = 52.0 + ring as f64 * 48.0;
    let wobble = 1.0 + 0.055 * (t * TAU * 2.0 + ring as f64 * 1.7 + spoke as f64 * 0.6).sin();
    (
        center.0 + angle.cos() * radius * wobble,
        center.1 + angle.sin() * radius * 0.55 * wobble,
    )
}

fn web_spoke_point(
    center: (f64, f64),
    ring: usize,
    spoke: usize,
    spokes: usize,
    t: f64,
) -> (f64, f64) {
    let mut point = web_point(center, ring, spoke, spokes, t);
    let angle = spoke as f64 / spokes as f64 * TAU;
    let drift = (t * TAU + ring as f64 * 1.17 + spoke as f64 * 0.41).sin() * 5.5;
    point.0 += -angle.sin() * drift * 0.45;
    point.1 += angle.cos() * drift;
    point
}

fn web_color(ring: usize, brightness: f64) -> Rgb {
    let pulse = if ring % 2 == 0 { 1.0 } else { 0.72 };
    dim(Rgb::new(58, 183, 210), brightness * pulse)
}

fn draw_lit_projected_triangles(
    canvas: &mut Canvas,
    mesh: &PolygonMatrix,
    transform: &Matrix,
    t: f64,
) {
    let mut triangle = PolygonMatrix::with_capacity(3);
    for (idx, (p0, p1, p2)) in mesh.transformed_triangles(transform).enumerate() {
        let Some(p0_screen) = project_body_point(&p0) else {
            continue;
        };
        let Some(p1_screen) = project_body_point(&p1) else {
            continue;
        };
        let Some(p2_screen) = project_body_point(&p2) else {
            continue;
        };

        let color = joltik_color(idx, &p0, &p1, &p2, t);
        triangle.clear();
        triangle.add_polygon(p0_screen, p1_screen, p2_screen);
        canvas.set_line_pixel(color);
        canvas.draw_polygons(&triangle);
    }
}

fn project_body_point(point: &[f64]) -> Option<(f64, f64, f64)> {
    let depth = point[2] + CAMERA_DISTANCE;
    if depth < 80.0 {
        return None;
    }

    let scale = FOCAL_LENGTH / depth;
    Some((
        f64::from(WIDTH) * 0.5 + point[0] * scale,
        f64::from(HEIGHT) * 0.43 + point[1] * scale,
        -depth,
    ))
}

fn joltik_color(idx: usize, p0: &[f64], p1: &[f64], p2: &[f64], t: f64) -> Rgb {
    let n = normal(p0, p1, p2);
    let facing = dot(n, normalized(LIGHT_DIR)).max(0.0);
    let center = centroid(p0, p1, p2);
    let upper_body = (1.0 - center.1 / f64::from(HEIGHT)).clamp(0.0, 1.0);
    let pulse = 0.5 + 0.5 * (t * TAU * 3.0 + center.0 * 0.018 + center.2 * 0.01).sin();
    let speckle = if (idx + (t * 41.0) as usize).is_multiple_of(29) {
        0.18
    } else {
        0.0
    };
    let intensity = (0.22 + facing * 0.56 + upper_body * 0.22 + pulse * speckle).clamp(0.0, 1.0);

    color_ramp(
        Rgb::new(74, 65, 20),
        Rgb::new(242, 205, 48),
        Rgb::new(255, 251, 171),
        intensity,
    )
}

fn draw_charge_lines(canvas: &mut Canvas, t: f64) {
    let center = web_center(t);
    canvas.set_line_width(1.0);

    for arc in 0..6 {
        let a0 = arc as f64 / 6.0 * TAU + t * TAU * 0.9;
        let a1 = a0 + PI * (0.55 + 0.08 * (arc % 3) as f64);
        let start = (
            center.0 + a0.cos() * (250.0 + arc as f64 * 18.0),
            center.1 + a0.sin() * (126.0 + arc as f64 * 8.0),
        );
        let end = (
            center.0 + a1.cos() * (250.0 + arc as f64 * 14.0),
            center.1 + a1.sin() * (126.0 + arc as f64 * 9.0),
        );
        let color = if arc % 2 == 0 {
            Rgb::new(116, 242, 255)
        } else {
            Rgb::new(255, 241, 109)
        };
        draw_jagged_arc(canvas, start, end, color, arc, t);
    }

    draw_web_sparks(canvas, t);
    canvas.set_line_width(1.0);
}

fn draw_jagged_arc(
    canvas: &mut Canvas,
    start: (f64, f64),
    end: (f64, f64),
    color: Rgb,
    seed: usize,
    t: f64,
) {
    let steps = 13;
    let dx = end.0 - start.0;
    let dy = end.1 - start.1;
    let length = (dx * dx + dy * dy).sqrt().max(1.0);
    let normal = (-dy / length, dx / length);
    let mut prev = start;

    for step in 1..=steps {
        let u = step as f64 / steps as f64;
        let snap = ((step * 37 + seed * 19) % 11) as f64 - 5.0;
        let wave = (t * TAU * 5.0 + seed as f64 + step as f64 * 1.9).sin();
        let offset = (snap * 2.8 + wave * 10.0) * (1.0 - (u - 0.5).abs() * 1.4).max(0.25);
        let next = (
            start.0 + dx * u + normal.0 * offset,
            start.1 + dy * u + normal.1 * offset,
        );
        canvas.draw_line_z(
            color,
            screen_point(prev, CHARGE_LINE_Z),
            screen_point(next, CHARGE_LINE_Z),
        );
        prev = next;
    }
}

fn draw_web_sparks(canvas: &mut Canvas, t: f64) {
    let center = web_center(t);
    for i in 0..10 {
        let angle = i as f64 / 10.0 * TAU + t * TAU * 1.15;
        let radius = 195.0 + ((i * 37) % 6) as f64 * 22.0;
        let start = (
            center.0 + angle.cos() * radius,
            center.1 + angle.sin() * radius * 0.55,
        );
        let end = (
            center.0 + (angle + 0.16).cos() * (radius + 38.0),
            center.1 + (angle + 0.16).sin() * (radius + 24.0) * 0.55,
        );
        let color = if i % 3 == 0 {
            Rgb::new(130, 248, 255)
        } else {
            Rgb::new(255, 236, 82)
        };
        draw_jagged_arc(canvas, start, end, dim(color, 0.78), i + 11, t);
    }
}

fn screen_point(point: (f64, f64), z: f64) -> (f64, f64, f64) {
    (point.0, screen_y(point.1), z)
}

fn lerp_point(a: (f64, f64), b: (f64, f64), t: f64) -> (f64, f64) {
    (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
}

fn screen_y(y: f64) -> f64 {
    f64::from(HEIGHT) - y
}

fn color_ramp(shadow: Rgb, mid: Rgb, highlight: Rgb, t: f64) -> Rgb {
    if t < 0.66 {
        shadow.lerp(mid, t / 0.66)
    } else {
        mid.lerp(highlight, (t - 0.66) / 0.34)
    }
}

fn dim(color: Rgb, factor: f64) -> Rgb {
    Rgb::new(
        scale_channel(color.red, factor),
        scale_channel(color.green, factor),
        scale_channel(color.blue, factor),
    )
}

fn scale_channel(channel: u8, factor: f64) -> u8 {
    (f64::from(channel) * factor).round().clamp(0.0, 255.0) as u8
}

fn centroid(p0: &[f64], p1: &[f64], p2: &[f64]) -> (f64, f64, f64) {
    (
        (p0[0] + p1[0] + p2[0]) / 3.0,
        (p0[1] + p1[1] + p2[1]) / 3.0,
        (p0[2] + p1[2] + p2[2]) / 3.0,
    )
}

fn normal(p0: &[f64], p1: &[f64], p2: &[f64]) -> (f64, f64, f64) {
    normalized(cross(
        (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]),
        (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]),
    ))
}

fn cross(a: (f64, f64, f64), b: (f64, f64, f64)) -> (f64, f64, f64) {
    (
        a.1 * b.2 - a.2 * b.1,
        a.2 * b.0 - a.0 * b.2,
        a.0 * b.1 - a.1 * b.0,
    )
}

fn dot(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

fn normalized(v: (f64, f64, f64)) -> (f64, f64, f64) {
    let length = (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt();
    if length <= f64::EPSILON {
        return (0.0, 0.0, 1.0);
    }
    (v.0 / length, v.1 / length, v.2 / length)
}
