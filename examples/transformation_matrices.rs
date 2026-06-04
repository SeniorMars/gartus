#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use gartus::prelude::*;
use std::{f64::consts::PI, io};

const WIDTH: u32 = 1000;
const HEIGHT: u32 = 900;
const FRAMES: usize = 96;

#[derive(Clone)]
struct Scene {
    core: EdgeMatrix,
    ribbon: EdgeMatrix,
    basis_glyphs: Vec<(Matrix, Rgb)>,
}

fn main() -> io::Result<()> {
    let scene = Scene {
        core: build_core_glyph(),
        ribbon: build_ribbon(),
        basis_glyphs: vec![
            (Matrix::hermite(), Rgb::new(255, 204, 84)),
            (Matrix::inverse_hermite(), Rgb::new(88, 230, 255)),
            (Matrix::inverse_bezier(), Rgb::new(255, 96, 184)),
        ],
    };

    FrameRecorder::render_gif(
        AnimationRenderOptions::new(
            "anim",
            "transformation-matrices-",
            FRAMES,
            "final/transformation_matrices.gif",
        )
        .delay_cs(3)
        .preview(24, "final/transformation_matrices.png")
        .unique_frame_dir(true),
        |frame| Ok(render_frame(frame, &scene)),
    )?;

    println!("Saved final/transformation_matrices.png and final/transformation_matrices.gif");
    Ok(())
}

fn render_frame(frame: usize, scene: &Scene) -> Canvas {
    let mut canvas = Canvas::new_with_bg(WIDTH, HEIGHT, Rgb::new(4, 8, 17));
    canvas.wrapped = false;
    canvas.upper_left_origin = true;
    canvas.set_line_width(1.0);

    let t = frame as f64 / FRAMES as f64;
    let view = Matrix::look_at(
        [
            360.0 * (t * PI * 2.0).cos(),
            190.0 + 34.0 * (t * PI * 4.0).sin(),
            520.0 * (t * PI * 2.0).sin() + 620.0,
        ],
        [0.0, 24.0, 0.0],
        [0.0, 1.0, 0.0],
    );
    let perspective =
        Matrix::perspective_projection(52.0, f64::from(WIDTH) / f64::from(HEIGHT), 1.0, 2200.0);
    let viewport = Matrix::viewport(0, 0, WIDTH as usize, HEIGHT as usize);
    let ortho = Matrix::orthographic_projection(-480.0, 480.0, -360.0, 360.0, -1200.0, 1200.0);
    let mini_viewport = Matrix::viewport(690, 42, 260, 210);

    let mut segments = Vec::new();
    draw_reflection_ghosts(&mut segments, scene, &view, &perspective, &viewport, t);
    draw_shear_ribbons(&mut segments, scene, &view, &perspective, &viewport, t);
    draw_core(&mut segments, scene, &view, &perspective, &viewport, t);
    draw_orthographic_blueprint(&mut segments, scene, &ortho, &mini_viewport, t);
    sort_segments_back_to_front(&mut segments);
    canvas.draw_projected_segments(segments);

    draw_basis_glyphs(&mut canvas, &scene.basis_glyphs, t);
    draw_viewport_frame(&mut canvas);
    canvas
}

fn draw_core(
    segments: &mut Vec<ProjectedSegment>,
    scene: &Scene,
    view: &Matrix,
    projection: &Matrix,
    viewport: &Matrix,
    t: f64,
) {
    let pulse = 1.0 + 0.08 * (t * PI * 2.0).sin();
    let model = Matrix::identity_matrix(4)
        * Matrix::translate(0.0, 18.0 * (t * PI * 2.0).sin(), 0.0)
        * Matrix::scale(pulse, 1.0 / pulse, 1.0 + 0.04 * (t * PI * 4.0).cos())
        * Matrix::rotate_x(t * 360.0)
        * Matrix::rotate_y(t * -270.0 + 28.0)
        * Matrix::rotate_z(t * 180.0)
        * Matrix::rotate_point(t * 420.0, 1.0, 1.35, 0.65);

    collect_projected_edges(
        segments,
        &scene.core,
        &model,
        view,
        projection,
        viewport,
        Rgb::new(237, 245, 255),
    );
}

fn draw_reflection_ghosts(
    segments: &mut Vec<ProjectedSegment>,
    scene: &Scene,
    view: &Matrix,
    projection: &Matrix,
    viewport: &Matrix,
    t: f64,
) {
    let reflections = [
        (
            Matrix::reflect_yz(),
            Rgb::new(76, 201, 255),
            (-255.0, 0.0, 0.0),
        ),
        (
            Matrix::reflect_xz(),
            Rgb::new(125, 255, 177),
            (255.0, 0.0, 0.0),
        ),
        (
            Matrix::reflect_xy(),
            Rgb::new(255, 216, 92),
            (0.0, 0.0, -250.0),
        ),
        (
            Matrix::reflect_45(),
            Rgb::new(255, 112, 198),
            (-190.0, 0.0, 210.0),
        ),
        (
            Matrix::reflect_neg45(),
            Rgb::new(176, 125, 255),
            (190.0, 0.0, 210.0),
        ),
        (
            Matrix::reflect_origin(),
            Rgb::new(255, 111, 92),
            (0.0, -145.0, 135.0),
        ),
    ];

    for (idx, (reflect, color, offset)) in reflections.into_iter().enumerate() {
        let drift = t * PI * 2.0 + idx as f64 * 0.7;
        let model = Matrix::translate(
            offset.0,
            offset.1 + 22.0 * drift.sin(),
            offset.2 + 20.0 * drift.cos(),
        ) * Matrix::scale(0.38, 0.38, 0.38)
            * Matrix::rotate_y(t * 180.0 + idx as f64 * 22.0)
            * reflect;
        collect_projected_edges(
            segments,
            &scene.core,
            &model,
            view,
            projection,
            viewport,
            color,
        );
    }
}

fn draw_shear_ribbons(
    segments: &mut Vec<ProjectedSegment>,
    scene: &Scene,
    view: &Matrix,
    projection: &Matrix,
    viewport: &Matrix,
    t: f64,
) {
    let wave = (t * PI * 2.0).sin();
    let shears = [
        (
            Matrix::shearing_x(0.45 * wave, -0.18),
            Rgb::new(80, 232, 255),
            -145.0,
        ),
        (
            Matrix::shearing_y(-0.35, 0.35 * (t * PI * 2.0).cos()),
            Rgb::new(255, 213, 79),
            0.0,
        ),
        (
            Matrix::shearing_z(0.22, 0.42 * wave),
            Rgb::new(255, 102, 184),
            145.0,
        ),
    ];

    for (shear, color, z) in shears {
        let model = Matrix::translate(0.0, -210.0, z)
            * Matrix::scale(1.0, 0.74, 1.0)
            * Matrix::rotate_y(t * 360.0)
            * shear;
        collect_projected_edges(
            segments,
            &scene.ribbon,
            &model,
            view,
            projection,
            viewport,
            color,
        );
    }
}

fn draw_orthographic_blueprint(
    segments: &mut Vec<ProjectedSegment>,
    scene: &Scene,
    ortho: &Matrix,
    viewport: &Matrix,
    t: f64,
) {
    let model = Matrix::translate(0.0, 0.0, 0.0)
        * Matrix::scale(0.92, 0.92, 0.92)
        * Matrix::rotate_x(62.0)
        * Matrix::rotate_z(t * -180.0);
    collect_projected_edges(
        segments,
        &scene.core,
        &model,
        &Matrix::identity_matrix(4),
        ortho,
        viewport,
        Rgb::new(79, 103, 154),
    );
}

fn collect_projected_edges(
    segments: &mut Vec<ProjectedSegment>,
    edges: &EdgeMatrix,
    model: &Matrix,
    view: &Matrix,
    projection: &Matrix,
    viewport: &Matrix,
    color: Rgb,
) {
    let view_model = view.clone() * model.clone();
    for (p0, p1) in edges.transformed_edges(&view_model) {
        let Some(a) = project_point(&p0, projection, viewport) else {
            continue;
        };
        let Some(b) = project_point(&p1, projection, viewport) else {
            continue;
        };
        segments.push(ProjectedSegment { a, b, color });
    }
}

fn project_point(point: &[f64], projection: &Matrix, viewport: &Matrix) -> Option<ScreenPoint> {
    let clip = projection.transform_homogeneous_point(point);
    if clip[3].abs() < 1e-9 {
        return None;
    }
    let ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3], 1.0];
    if !ndc[0].is_finite() || !ndc[1].is_finite() || ndc[2].abs() > 1.25 {
        return None;
    }
    let screen = viewport.transform_homogeneous_point(&ndc);
    Some(ScreenPoint {
        x: screen[0],
        y: screen[1],
        depth: ndc[2],
    })
}

fn draw_basis_glyphs(canvas: &mut Canvas, glyphs: &[(Matrix, Rgb)], t: f64) {
    let origin_y = 742.0;
    for (glyph_idx, (basis, color)) in glyphs.iter().enumerate() {
        let x0 = 100.0 + glyph_idx as f64 * 220.0;
        let phase = t * PI * 2.0 + glyph_idx as f64;
        for row in 0..4 {
            for col in 0..4 {
                let value = basis[(row, col)];
                let cx = x0 + col as f64 * 34.0;
                let cy = origin_y + row as f64 * 30.0;
                let radius = 4.0 + value.abs() * 4.0;
                let tilt = value.signum() * (18.0 + 10.0 * phase.sin());
                canvas.draw_line(*color, cx - radius, cy - tilt, cx + radius, cy + tilt);
                canvas.draw_line(
                    dim(*color, 0.55),
                    cx - tilt * 0.18,
                    cy + radius,
                    cx + tilt * 0.18,
                    cy - radius,
                );
            }
        }
    }
}

fn draw_viewport_frame(canvas: &mut Canvas) {
    let frame = Rgb::new(45, 65, 105);
    canvas.draw_line(frame, 690.0, 42.0, 950.0, 42.0);
    canvas.draw_line(frame, 950.0, 42.0, 950.0, 252.0);
    canvas.draw_line(frame, 950.0, 252.0, 690.0, 252.0);
    canvas.draw_line(frame, 690.0, 252.0, 690.0, 42.0);
}

fn build_core_glyph() -> EdgeMatrix {
    let mut edges = EdgeMatrix::new();
    add_box_edges(&mut edges, 145.0);
    add_octahedron_edges(&mut edges, 125.0);

    let ring = EdgeMatrix::great_circle(104.0, 60);
    edges.extend(&ring);
    edges.extend(&ring.apply(&Matrix::rotate_x(90.0)));
    edges.extend(&ring.apply(&Matrix::rotate_z(90.0)));
    edges
}

fn add_box_edges(edges: &mut EdgeMatrix, size: f64) {
    let h = size * 0.5;
    let p = [
        (-h, -h, -h),
        (h, -h, -h),
        (h, h, -h),
        (-h, h, -h),
        (-h, -h, h),
        (h, -h, h),
        (h, h, h),
        (-h, h, h),
    ];
    for (a, b) in [
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
        edges.push_edge_tuple(p[a], p[b]);
    }
}

fn add_octahedron_edges(edges: &mut EdgeMatrix, radius: f64) {
    let top = (0.0, radius, 0.0);
    let bottom = (0.0, -radius, 0.0);
    let ring = [
        (radius, 0.0, 0.0),
        (0.0, 0.0, radius),
        (-radius, 0.0, 0.0),
        (0.0, 0.0, -radius),
    ];
    for i in 0..ring.len() {
        let next = ring[(i + 1) % ring.len()];
        edges.push_edge_tuple(ring[i], next);
        edges.push_edge_tuple(top, ring[i]);
        edges.push_edge_tuple(bottom, ring[i]);
    }
}

fn build_ribbon() -> EdgeMatrix {
    let mut ribbon = EdgeMatrix::new();
    for i in 0..34 {
        let x = -210.0 + i as f64 * 12.0;
        let y = 18.0 * (i as f64 * 0.52).sin();
        let z = 30.0 * (i as f64 * 0.41).cos();
        ribbon.push_edge(x, -28.0 + y, z, x, 28.0 + y, -z);
        if i > 0 {
            let px = -210.0 + (i - 1) as f64 * 12.0;
            let py = 18.0 * ((i - 1) as f64 * 0.52).sin();
            let pz = 30.0 * ((i - 1) as f64 * 0.41).cos();
            ribbon.push_edge(px, -28.0 + py, pz, x, -28.0 + y, z);
            ribbon.push_edge(px, 28.0 + py, -pz, x, 28.0 + y, -z);
        }
    }
    ribbon
}

fn dim(color: Rgb, factor: f64) -> Rgb {
    Rgb::new(
        (f64::from(color.red) * factor).round().clamp(0.0, 255.0) as u8,
        (f64::from(color.green) * factor).round().clamp(0.0, 255.0) as u8,
        (f64::from(color.blue) * factor).round().clamp(0.0, 255.0) as u8,
    )
}
