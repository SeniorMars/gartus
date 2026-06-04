use gartus::{external, prelude::*};
use std::{error::Error, f64::consts::PI};

const WIDTH: u32 = 1000;
const HEIGHT: u32 = 900;
const FRAMES: usize = 96;
const CAMERA_DISTANCE: f64 = 1120.0;
const FOCAL_LENGTH: f64 = 820.0;

struct Model {
    mesh: PolygonMatrix,
    normalize: Matrix,
    color: Rgb,
    stride: usize,
}

struct Scene {
    spider: Model,
    penguin: Model,
    duck: Model,
    creature: Model,
    ring: EdgeMatrix,
    arm: EdgeMatrix,
}

fn main() {
    if let Err(err) = render() {
        eprintln!("could not render cstack model theater:\n{err}");
        std::process::exit(1);
    }
}

fn render() -> Result<(), Box<dyn Error>> {
    let scene = load_scene()?;
    FrameRecorder::render_gif(
        AnimationRenderOptions::new(
            "anim",
            "cstack-model-theater-",
            FRAMES,
            "final/cstack_model_theater.gif",
        )
        .delay_cs(3)
        .preview(28, "final/cstack_model_theater.png")
        .unique_frame_dir(true),
        |frame| Ok(render_frame(frame, &scene)),
    )?;

    println!("Saved final/cstack_model_theater.png and final/cstack_model_theater.gif");
    Ok(())
}

fn load_scene() -> Result<Scene, Box<dyn Error>> {
    Ok(Scene {
        spider: load_model(
            "examples/data/meshes/scotty/arzhang.obj",
            380.0,
            Rgb::new(255, 68, 119),
            4,
        )?,
        penguin: load_model(
            "examples/data/meshes/scotty/bamado.obj",
            230.0,
            Rgb::new(255, 224, 92),
            2,
        )?,
        duck: load_model(
            "examples/data/meshes/scotty/aylu.obj",
            245.0,
            Rgb::new(126, 255, 185),
            8,
        )?,
        creature: load_model(
            "examples/data/meshes/scotty/ashung.obj",
            210.0,
            Rgb::new(186, 133, 255),
            7,
        )?,
        ring: EdgeMatrix::great_circle(245.0, 44),
        arm: make_arm_edges(),
    })
}

fn load_model(path: &str, size: f64, color: Rgb, stride: usize) -> Result<Model, Box<dyn Error>> {
    let mesh = external::meshify(path)?;
    println!("loaded {path}: {} triangles", mesh.triangle_count());
    let normalize = external::normalize_mesh_transform(&mesh, size, external::MeshUpAxis::Y);
    Ok(Model {
        mesh,
        normalize,
        color,
        stride,
    })
}

fn render_frame(frame: usize, scene: &Scene) -> Canvas {
    let mut canvas = Canvas::new_with_bg(WIDTH, HEIGHT, background());
    canvas.wrapped = false;
    canvas.upper_left_origin = true;
    canvas.set_line_width(1.0);

    let t = frame as f64 / FRAMES as f64;
    let mut root = MatrixStack::new();
    let camera_phase = t * PI * 2.0;
    root.apply(Matrix::translate(
        0.0,
        12.0 * camera_phase.cos(),
        92.0 * camera_phase.sin(),
    ));
    root.apply(Matrix::rotate_y(18.0 * camera_phase.sin()));
    root.apply(Matrix::rotate_x(20.0 + 12.0 * camera_phase.cos()));
    root.apply(Matrix::rotate_z(4.0 * (camera_phase * 1.7).sin()));

    let camera = Camera3D::new(WIDTH, HEIGHT)
        .with_camera_distance(CAMERA_DISTANCE)
        .with_focal_length(FOCAL_LENGTH)
        .with_center_y_factor(0.58);
    let mut segments = Vec::new();
    draw_great_circle_shell(&mut segments, &mut root.clone(), scene, &camera, t);
    draw_portal(&mut segments, &mut root.clone(), scene, &camera, t);
    draw_spider(&mut segments, &mut root.clone(), scene, &camera, t);
    draw_hologram_arm(
        &mut segments,
        &mut root.clone(),
        &scene.penguin,
        &scene.arm,
        &camera,
        -1.0,
        t,
    );
    draw_hologram_arm(
        &mut segments,
        &mut root.clone(),
        &scene.duck,
        &scene.arm,
        &camera,
        1.0,
        t,
    );
    draw_overhead_creature(&mut segments, &mut root, scene, &camera, t);

    sort_segments_back_to_front(&mut segments);
    canvas.draw_projected_segments(segments);

    canvas
}

fn draw_portal(
    segments: &mut Vec<ProjectedSegment>,
    stack: &mut MatrixStack,
    scene: &Scene,
    camera: &Camera3D,
    t: f64,
) {
    stack.push();
    let orbit = t * PI * 2.0;
    stack.apply(Matrix::translate(
        92.0 * orbit.sin(),
        56.0 + 28.0 * (orbit * 1.3).cos(),
        -180.0 + 72.0 * orbit.cos(),
    ));
    stack.apply(Matrix::rotate_y(t * 310.0));
    stack.apply(Matrix::rotate_x(66.0 + 14.0 * orbit.sin()));
    stack.apply(Matrix::rotate_z(t * -120.0));
    collect_edge_segments(
        segments,
        &scene.ring,
        stack.top(),
        camera,
        Rgb::new(55, 155, 255),
        0.62,
    );
    let _ = stack.pop();
}

fn draw_great_circle_shell(
    segments: &mut Vec<ProjectedSegment>,
    stack: &mut MatrixStack,
    scene: &Scene,
    camera: &Camera3D,
    t: f64,
) {
    let configs = [
        (0.0, 0.0, 0.0, Rgb::new(36, 132, 255), 0.44),
        (90.0, 0.0, 0.0, Rgb::new(36, 132, 255), 0.4),
        (0.0, 90.0, 0.0, Rgb::new(146, 93, 255), 0.38),
        (58.0, 24.0, 0.0, Rgb::new(62, 210, 255), 0.34),
        (-42.0, 0.0, 38.0, Rgb::new(104, 130, 255), 0.3),
        (18.0, 68.0, -24.0, Rgb::new(92, 255, 198), 0.26),
        (74.0, -34.0, 52.0, Rgb::new(255, 98, 183), 0.22),
    ];

    for (idx, (rx, ry, rz, color, brightness)) in configs.iter().enumerate() {
        stack.push();
        let phase = t * 360.0;
        stack.apply(Matrix::translate(0.0, -14.0, -18.0));
        stack.apply(Matrix::rotate_y(phase * (0.28 + idx as f64 * 0.035)));
        stack.apply(Matrix::rotate_x(rx + phase * 0.08));
        stack.apply(Matrix::rotate_y(*ry));
        stack.apply(Matrix::rotate_z(rz + phase * -0.11));
        stack.apply(Matrix::scale(
            1.0 + idx as f64 * 0.045,
            1.0 + idx as f64 * 0.045,
            1.0,
        ));
        collect_edge_segments(
            segments,
            &scene.ring,
            stack.top(),
            camera,
            *color,
            *brightness,
        );
        let _ = stack.pop();
    }
}

fn draw_spider(
    segments: &mut Vec<ProjectedSegment>,
    stack: &mut MatrixStack,
    scene: &Scene,
    camera: &Camera3D,
    t: f64,
) {
    stack.push();
    stack.apply(Matrix::translate(0.0, -84.0, 28.0));
    stack.apply(Matrix::rotate_y(t * 50.0 - 25.0));
    stack.apply(Matrix::rotate_x(6.0 * (t * PI * 4.0).sin()));
    stack.apply(scene.spider.normalize.clone());
    collect_mesh_segments(segments, &scene.spider, stack.top(), camera, 1.0, t);
    let _ = stack.pop();
}

fn draw_hologram_arm(
    segments: &mut Vec<ProjectedSegment>,
    stack: &mut MatrixStack,
    model: &Model,
    arm: &EdgeMatrix,
    camera: &Camera3D,
    side: f64,
    t: f64,
) {
    stack.push();
    stack.apply(Matrix::translate(side * 255.0, -28.0, 28.0));
    stack.apply(Matrix::rotate_z(side * (9.0 + 5.0 * (t * PI * 2.0).sin())));
    collect_edge_segments(
        segments,
        arm,
        stack.top(),
        camera,
        dim(model.color, 0.55),
        1.0,
    );

    stack.push();
    stack.apply(Matrix::translate(side * 92.0, 12.0, 12.0));
    stack.apply(Matrix::rotate_y(side * -t * 260.0));
    stack.apply(Matrix::rotate_x(8.0 * (t * PI * 2.0).cos()));
    stack.apply(model.normalize.clone());
    collect_mesh_segments(segments, model, stack.top(), camera, 0.86, t);
    let _ = stack.pop();
    let _ = stack.pop();
}

fn draw_overhead_creature(
    segments: &mut Vec<ProjectedSegment>,
    stack: &mut MatrixStack,
    scene: &Scene,
    camera: &Camera3D,
    t: f64,
) {
    stack.push();
    stack.apply(Matrix::translate(0.0, 216.0, 12.0));
    stack.apply(Matrix::rotate_y(t * 360.0));
    stack.apply(Matrix::rotate_x(-18.0));
    collect_edge_segments(
        segments,
        &scene.ring,
        stack.top(),
        camera,
        dim(scene.creature.color, 0.45),
        1.0,
    );

    stack.push();
    stack.apply(Matrix::scale(0.72, 0.72, 0.72));
    stack.apply(Matrix::rotate_y(t * -540.0));
    stack.apply(scene.creature.normalize.clone());
    collect_mesh_segments(segments, &scene.creature, stack.top(), camera, 0.7, t);
    let _ = stack.pop();
    let _ = stack.pop();
}

fn collect_mesh_segments(
    segments: &mut Vec<ProjectedSegment>,
    model: &Model,
    transform: &Matrix,
    camera: &Camera3D,
    brightness: f64,
    t: f64,
) {
    for (idx, (p0, p1, p2)) in model.mesh.transformed_triangles(transform).enumerate() {
        if idx % model.stride != 0 {
            continue;
        }
        let Some(a) = camera.project(&p0) else {
            continue;
        };
        let Some(b) = camera.project(&p1) else {
            continue;
        };
        let Some(c) = camera.project(&p2) else {
            continue;
        };
        let color = depth_color(
            model.color,
            (a.depth + b.depth + c.depth) / 3.0,
            brightness,
            t,
        );
        segments.push(ProjectedSegment { a, b, color });
        segments.push(ProjectedSegment { a: b, b: c, color });
        segments.push(ProjectedSegment { a: c, b: a, color });
    }
}

fn collect_edge_segments(
    segments: &mut Vec<ProjectedSegment>,
    edges: &EdgeMatrix,
    transform: &Matrix,
    camera: &Camera3D,
    color: Rgb,
    brightness: f64,
) {
    for (p0, p1) in edges.transformed_edges(transform) {
        if let Some(segment) =
            ProjectedSegment::from_points(camera, &p0, &p1, dim(color, brightness))
        {
            segments.push(segment);
        }
    }
}

fn make_arm_edges() -> EdgeMatrix {
    let mut arm = EdgeMatrix::new();
    arm.push_edge(-128.0, 0.0, 0.0, 128.0, 0.0, 0.0);
    arm.push_edge(-128.0, 0.0, 0.0, -96.0, 28.0, 0.0);
    arm.push_edge(128.0, 0.0, 0.0, 96.0, 28.0, 0.0);
    arm.push_edge(-96.0, 28.0, 0.0, 96.0, 28.0, 0.0);
    arm.push_edge(-90.0, -28.0, 0.0, 90.0, -28.0, 0.0);
    arm.push_edge(-90.0, -28.0, 0.0, -128.0, 0.0, 0.0);
    arm.push_edge(90.0, -28.0, 0.0, 128.0, 0.0, 0.0);
    arm
}

fn depth_color(color: Rgb, depth: f64, brightness: f64, t: f64) -> Rgb {
    let pulse = 0.9 + 0.1 * (t * PI * 2.0).sin();
    let depth_factor = (1.22 - depth / 1800.0).clamp(0.42, 1.0) * brightness * pulse;
    dim(color, depth_factor)
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

fn background() -> Rgb {
    Rgb::new(3, 5, 12)
}
