use gartus::prelude::*;
use std::{error::Error, f64::consts::PI, fs};

const WIDTH: u32 = 960;
const HEIGHT: u32 = 960;
const FRAMES: usize = 96;
const CAMERA_DISTANCE: f64 = 980.0;
const FOCAL_LENGTH: f64 = 760.0;

#[derive(Clone)]
struct MeshObject {
    mesh: PolygonMatrix,
    base: Matrix,
    color: Rgb,
}

fn main() {
    if let Err(err) = render() {
        eprintln!("could not render 3D gallery piece:\n{err}");
        std::process::exit(1);
    }
}

fn render() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final")?;

    let objects = build_scene();
    let mut recorder = FrameRecorder::new("anim", "gallery-3d-").with_delay(3);
    let mut preview = Canvas::new_with_bg(WIDTH, HEIGHT, background());

    for frame in 0..FRAMES {
        let canvas = render_frame(frame, &objects);
        if frame == 22 {
            preview = canvas.clone();
        }
        recorder.capture(&canvas)?;
    }

    preview.save_extension("final/gallery_3d.png")?;
    recorder.encode_gif("final/gallery_3d.gif")?;

    println!("Saved final/gallery_3d.png and final/gallery_3d.gif");
    Ok(())
}

fn build_scene() -> Vec<MeshObject> {
    let mut objects = Vec::new();

    objects.push(MeshObject {
        mesh: make_ground_grid(),
        base: Matrix::identity_matrix(4),
        color: Rgb::new(51, 214, 255),
    });

    objects.extend(make_towers());

    let mut halo = PolygonMatrix::new();
    halo.add_torus((0.0, 0.0, 0.0), 14.0, 210.0, 42);
    halo.add_torus((0.0, 0.0, 0.0), 8.0, 130.0, 32);
    objects.push(MeshObject {
        mesh: halo,
        base: Matrix::translate(0.0, 96.0, 0.0) * Matrix::rotate_x(72.0),
        color: Rgb::new(255, 72, 185),
    });

    let mut core = PolygonMatrix::new();
    core.add_dodecahedron((0.0, 0.0, 0.0), 78.0);
    core.add_icosahedron((0.0, 0.0, 0.0), 58.0);
    core.add_sphere((0.0, 0.0, 0.0), 34.0, 10);
    objects.push(MeshObject {
        mesh: core,
        base: Matrix::translate(0.0, 96.0, 0.0),
        color: Rgb::new(242, 248, 255),
    });

    objects.extend(make_orbiting_crystals());
    objects
}

fn make_ground_grid() -> PolygonMatrix {
    let mut mesh = PolygonMatrix::new();
    let size = 620.0;
    let step = 62.0;
    let half = size / 2.0;

    for z_idx in 0..10 {
        for x_idx in 0..10 {
            let x0 = -half + x_idx as f64 * step;
            let x1 = x0 + step;
            let z0 = -half + z_idx as f64 * step;
            let z1 = z0 + step;
            let y00 = ripple_height(x0, z0);
            let y10 = ripple_height(x1, z0);
            let y01 = ripple_height(x0, z1);
            let y11 = ripple_height(x1, z1);
            mesh.add_polygon((x0, y00, z0), (x0, y01, z1), (x1, y11, z1));
            mesh.add_polygon((x0, y00, z0), (x1, y11, z1), (x1, y10, z0));
        }
    }

    mesh
}

fn make_towers() -> Vec<MeshObject> {
    let mut objects = Vec::new();
    for i in 0..14 {
        let angle = i as f64 / 14.0 * PI * 2.0;
        let radius = if i % 2 == 0 { 255.0 } else { 188.0 };
        let x = angle.cos() * radius;
        let z = angle.sin() * radius;
        let height = 96.0 + (i % 5) as f64 * 34.0;
        let width = 34.0 + (i % 3) as f64 * 8.0;

        let mut tower = PolygonMatrix::new();
        tower.add_box((-width / 2.0, height, width / 2.0), width, height, width);
        if i % 2 == 0 {
            tower.add_pyramid((-width / 2.0, height + 74.0, width / 2.0), width, 74.0);
        } else {
            tower.add_cone((0.0, height + 56.0, 0.0), width * 0.58, 56.0, 9);
        }

        objects.push(MeshObject {
            mesh: tower,
            base: Matrix::translate(x, -60.0, z) * Matrix::rotate_y(-angle.to_degrees()),
            color: if i % 2 == 0 {
                Rgb::new(255, 200, 72)
            } else {
                Rgb::new(92, 255, 171)
            },
        });
    }
    objects
}

fn make_orbiting_crystals() -> Vec<MeshObject> {
    let mut objects = Vec::new();
    for i in 0..8 {
        let angle = i as f64 / 8.0 * PI * 2.0;
        let mut crystal = PolygonMatrix::new();
        add_crystal(&mut crystal, 38.0, 72.0);
        objects.push(MeshObject {
            mesh: crystal,
            base: Matrix::translate(
                angle.cos() * 342.0,
                96.0 + (i % 2) as f64 * 52.0,
                angle.sin() * 342.0,
            ) * Matrix::rotate_y(angle.to_degrees()),
            color: Rgb::new(166, 126, 255),
        });
    }
    objects
}

fn add_crystal(mesh: &mut PolygonMatrix, radius: f64, height: f64) {
    let top = (0.0, height * 0.5, 0.0);
    let bottom = (0.0, -height * 0.5, 0.0);
    let points = [
        (radius, 0.0, 0.0),
        (0.0, 0.0, radius),
        (-radius, 0.0, 0.0),
        (0.0, 0.0, -radius),
    ];

    for i in 0..4 {
        let curr = points[i];
        let next = points[(i + 1) % 4];
        mesh.add_polygon(top, curr, next);
        mesh.add_polygon(bottom, next, curr);
    }
}

fn render_frame(frame: usize, objects: &[MeshObject]) -> Canvas {
    let mut canvas = Canvas::new_with_bg(WIDTH, HEIGHT, background());
    canvas.wrapped = false;
    canvas.set_line_width(1.0);

    let t = frame as f64 / FRAMES as f64;
    draw_star_tunnel(&mut canvas, t);

    let world = Matrix::rotate_y(t * 360.0)
        * Matrix::rotate_x(18.0 + 7.0 * (t * PI * 2.0).sin())
        * Matrix::rotate_z(2.5 * (t * PI * 4.0).cos());

    let camera = Camera3D::new(WIDTH, HEIGHT)
        .with_camera_distance(CAMERA_DISTANCE)
        .with_focal_length(FOCAL_LENGTH)
        .with_center_y_factor(0.58);
    let mut segments = Vec::new();
    for object in objects {
        let animation = Matrix::rotate_y(t * 360.0 * object_spin(object.color));
        let transform = world.clone() * object.base.clone() * animation;
        segments.extend(camera.project_mesh_wireframe_segments(
            &object.mesh,
            &transform,
            1,
            |_, depth| depth_color(object.color, depth, t),
        ));
    }

    sort_segments_back_to_front(&mut segments);
    canvas.draw_projected_segments(segments);

    canvas
}

fn draw_star_tunnel(canvas: &mut Canvas, t: f64) {
    let color = Rgb::new(45, 59, 104);
    for i in 0..92 {
        let a = i as f64 * 2.399_963_229_728_653 + t * PI * 0.35;
        let r = 68.0 + ((i * 37) % 430) as f64;
        let z = 160.0 + ((i * 83) % 720) as f64;
        let flicker = 2.0 + ((i * 11) % 5) as f64;
        let x = a.cos() * r * FOCAL_LENGTH / (z + 480.0) + f64::from(WIDTH) * 0.5;
        let y = a.sin() * r * FOCAL_LENGTH / (z + 480.0) + f64::from(HEIGHT) * 0.5;
        canvas.draw_line(color, x - flicker, y, x + flicker, y);
        canvas.draw_line(color, x, y - flicker, x, y + flicker);
    }
}

fn depth_color(color: Rgb, depth: f64, t: f64) -> Rgb {
    let pulse = 0.78 + 0.18 * (t * PI * 2.0).sin();
    let depth_factor = (1.16 - depth / 1600.0).clamp(0.42, 1.0) * pulse;
    Rgb::new(
        scale_channel(color.red, depth_factor),
        scale_channel(color.green, depth_factor),
        scale_channel(color.blue, depth_factor),
    )
}

fn scale_channel(channel: u8, factor: f64) -> u8 {
    (f64::from(channel) * factor).round().clamp(0.0, 255.0) as u8
}

fn ripple_height(x: f64, z: f64) -> f64 {
    ((x * 0.025).sin() + (z * 0.031).cos()) * 14.0 - 86.0
}

fn object_spin(color: Rgb) -> f64 {
    if color == Rgb::new(242, 248, 255) {
        1.35
    } else if color == Rgb::new(255, 72, 185) {
        -0.55
    } else {
        0.25
    }
}

fn background() -> Rgb {
    Rgb::new(4, 7, 16)
}
