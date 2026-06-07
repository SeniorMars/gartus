use gartus::prelude::*;
use std::f64::consts::PI;
use std::fs;
use std::io;

const WIDTH: u32 = 900;
const HEIGHT: u32 = 900;
const FRAMES: usize = 96;

fn main() -> io::Result<()> {
    fs::create_dir_all("final")?;

    let core = build_core();
    let halo = build_halo();
    let spires = build_spires();
    let starfield = build_starfield();

    let mut recorder = FrameRecorder::new("anim", "polygon-orbit-").with_delay(3);
    let mut preview = Canvas::new_with_bg(WIDTH, HEIGHT, background());

    for frame in 0..FRAMES {
        let canvas = render_frame(frame, &core, &halo, &spires, &starfield);
        if frame == 18 {
            preview = canvas.clone();
        }
        recorder.capture(&canvas)?;
    }

    preview.save_extension("final/polygon_orbit.png")?;
    recorder.encode_gif("final/polygon_orbit.gif")?;

    println!("Saved final/polygon_orbit.png and final/polygon_orbit.gif");
    Ok(())
}

fn render_frame(
    frame: usize,
    core: &PolygonMatrix,
    halo: &PolygonMatrix,
    spires: &PolygonMatrix,
    starfield: &EdgeMatrix,
) -> Canvas {
    let mut canvas = Canvas::new_with_bg(WIDTH, HEIGHT, background());
    canvas.set_wrapped(false);
    canvas.set_line_width(1.0);

    let t = frame as f64 / FRAMES as f64;
    draw_starfield(&mut canvas, starfield, t);

    let base = Matrix::translate(450.0, 450.0, 0.0);
    let breathe = Matrix::scale(
        1.0 + 0.045 * (t * PI * 2.0).sin(),
        1.0 + 0.045 * (t * PI * 2.0).sin(),
        1.0,
    );

    let halo_transform =
        &(&base * &Matrix::rotate_z(t * 360.0)) * &(&Matrix::rotate_x(68.0) * &breathe);
    draw_wire_polygons(&mut canvas, halo, &halo_transform, Rgb::new(68, 210, 255));

    let counter_halo =
        &(&base * &Matrix::rotate_z(-t * 270.0)) * &(&Matrix::rotate_y(42.0) * &breathe);
    draw_wire_polygons(&mut canvas, halo, &counter_halo, Rgb::new(255, 88, 187));

    let core_transform = &(&base * &Matrix::rotate_y(t * 360.0 + 20.0)) * &Matrix::rotate_x(24.0);
    draw_wire_polygons(&mut canvas, core, &core_transform, Rgb::new(238, 242, 255));

    let spire_transform = &base * &Matrix::rotate_z(-t * 180.0);
    draw_wire_polygons(
        &mut canvas,
        spires,
        &spire_transform,
        Rgb::new(255, 198, 82),
    );

    draw_orbit_ticks(&mut canvas, t);
    canvas
}

fn build_core() -> PolygonMatrix {
    let mut polygons = PolygonMatrix::new();
    polygons.add_dodecahedron((0.0, 0.0, 0.0), 118.0);
    polygons.add_icosahedron((0.0, 0.0, 0.0), 92.0);
    polygons.add_sphere((0.0, 0.0, 0.0), 56.0, 14);
    polygons
}

fn build_halo() -> PolygonMatrix {
    let mut polygons = PolygonMatrix::new();
    polygons.add_torus((0.0, 0.0, 0.0), 18.0, 245.0, 48);
    polygons.add_torus((0.0, 0.0, 0.0), 8.0, 168.0, 36);
    polygons
}

fn build_spires() -> PolygonMatrix {
    let mut polygons = PolygonMatrix::new();
    for i in 0..12 {
        let a = i as f64 / 12.0 * PI * 2.0;
        let (sin_a, cos_a) = a.sin_cos();
        let x = cos_a * 292.0;
        let y = sin_a * 292.0;
        polygons.add_pyramid((x - 24.0, y - 24.0, 0.0), 48.0, 92.0);
    }
    polygons
}

fn build_starfield() -> EdgeMatrix {
    let mut stars = EdgeMatrix::new();
    for i in 0..72 {
        let a = i as f64 * 2.399_963_229_728_653;
        let r = 120.0 + (i % 19) as f64 * 19.0;
        let x = a.cos() * r;
        let y = a.sin() * r;
        let s = 3.0 + (i % 5) as f64;
        stars.push_edge(x - s, y, 0.0, x + s, y, 0.0);
        stars.push_edge(x, y - s, 0.0, x, y + s, 0.0);
    }
    stars
}

fn draw_starfield(canvas: &mut Canvas, stars: &EdgeMatrix, t: f64) {
    canvas.set_line_pixel(Rgb::new(65, 78, 119));
    let drift = &Matrix::translate(450.0, 450.0, 0.0) * &Matrix::rotate_z(t * 18.0);
    canvas.draw_transformed(stars, &drift);
}

fn draw_wire_polygons(
    canvas: &mut Canvas,
    polygons: &PolygonMatrix,
    transform: &Matrix,
    color: Rgb,
) {
    let transformed = polygons.apply(transform);
    canvas.set_line_pixel(color);
    for (p0, p1, p2) in transformed.iter_triangles() {
        canvas.draw_line(color, p0[0], p0[1], p1[0], p1[1]);
        canvas.draw_line(color, p1[0], p1[1], p2[0], p2[1]);
        canvas.draw_line(color, p2[0], p2[1], p0[0], p0[1]);
    }
}

fn draw_orbit_ticks(canvas: &mut Canvas, t: f64) {
    let color = Rgb::new(142, 255, 168);
    for i in 0..36 {
        let a = i as f64 / 36.0 * PI * 2.0 + t * PI * 2.0;
        let inner = 328.0;
        let outer = if i % 3 == 0 { 354.0 } else { 342.0 };
        let (sin_a, cos_a) = a.sin_cos();
        canvas.draw_line(
            color,
            450.0 + cos_a * inner,
            450.0 + sin_a * inner,
            450.0 + cos_a * outer,
            450.0 + sin_a * outer,
        );
    }
}

fn background() -> Rgb {
    Rgb::new(7, 10, 18)
}
