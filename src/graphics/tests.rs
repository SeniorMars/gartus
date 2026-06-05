//! Tests for the graphics module.

use super::{
    animation::FrameRecorder,
    colors::Rgb,
    display::{Canvas, PolygonColorMode, ShadingMode},
    draw::{triangle_color, vertex_normal, vertex_normals},
    lighting::Lighting,
};
use crate::gmath::{
    edge_matrix::EdgeMatrix, matrix::Matrix, polygon_matrix::PolygonMatrix, vector::Vector,
};
use std::{collections::BTreeSet, fs};

#[test]
fn from_pixels_preserves_exact_pixel_data() {
    let canvas = Canvas::from_pixels(2, 1, vec![Rgb::new(1, 2, 3), Rgb::new(4, 5, 6)]);

    assert_eq!(canvas.width(), 2);
    assert_eq!(canvas.height(), 1);
    assert_eq!(canvas.pixels(), &[Rgb::new(1, 2, 3), Rgb::new(4, 5, 6)]);
}

#[test]
fn map_pixels_preserves_canvas_metadata() {
    let canvas = Canvas::builder(1, 1)
        .background(Rgb::new(10, 20, 30))
        .line_color(Rgb::new(40, 50, 60))
        .line_width(3.0)
        .polygon_color_mode(PolygonColorMode::DeterministicRandom)
        .shading_mode(ShadingMode::Gouraud)
        .lighting(Lighting {
            specular_exponent: 8,
            ..Lighting::default()
        })
        .upper_left_origin(true)
        .wrapped(false)
        .build();

    let mapped = canvas.map_pixels(|pixel| Rgb::new(pixel.blue, pixel.green, pixel.red));

    assert_eq!(mapped.pixels(), &[Rgb::new(30, 20, 10)]);
    assert_eq!(mapped.line_color(), Rgb::new(40, 50, 60));
    assert!((mapped.line_width() - 3.0).abs() < f64::EPSILON);
    assert_eq!(
        mapped.polygon_color_mode(),
        PolygonColorMode::DeterministicRandom
    );
    assert_eq!(mapped.shading_mode(), ShadingMode::Gouraud);
    assert_eq!(mapped.lighting().specular_exponent, 8);
    assert!(mapped.upper_left_origin);
    assert!(!mapped.wrapped);
}

#[test]
fn polygon_color_mode_defaults_to_line_color_and_can_change() {
    let mut canvas = Canvas::new(1, 1, Rgb::WHITE);

    assert_eq!(canvas.polygon_color_mode(), PolygonColorMode::LineColor);
    canvas.set_polygon_color_mode(PolygonColorMode::TintedFromLine);
    assert_eq!(
        canvas.polygon_color_mode(),
        PolygonColorMode::TintedFromLine
    );
    assert_eq!(canvas.shading_mode(), ShadingMode::Flat);
    canvas.set_shading_mode(ShadingMode::Phong);
    assert_eq!(canvas.shading_mode(), ShadingMode::Phong);
}

#[test]
fn empty_canvas_coordinates_are_clipped_without_panic() {
    let mut canvas = Canvas::default();

    assert_eq!(canvas.normalize_coords(0, 0), None);
    assert_eq!(canvas.get_pixel(0, 0), None);
    canvas.plot(&Rgb::WHITE, 0, 0);
    assert!(canvas.is_empty());
}

#[test]
fn plot_z_only_replaces_farther_pixels() {
    let mut canvas = Canvas::new_with_bg(1, 1, Rgb::WHITE);
    canvas.upper_left_origin = true;
    canvas.wrapped = false;

    canvas.plot_z(&Rgb::RED, 0, 0, 5.0);
    canvas.plot_z(&Rgb::BLUE, 0, 0, 4.0);
    assert_eq!(canvas.get_pixel(0, 0), Some(&Rgb::RED));
    assert_eq!(canvas.get_zbuffer(0, 0), Some(5.0));

    canvas.plot_z(&Rgb::GREEN, 0, 0, 6.0);
    assert_eq!(canvas.get_pixel(0, 0), Some(&Rgb::GREEN));
    assert_eq!(canvas.get_zbuffer(0, 0), Some(6.0));
}

#[test]
fn clear_canvas_resets_zbuffer() {
    let mut canvas = Canvas::new_with_bg(1, 1, Rgb::WHITE);
    canvas.upper_left_origin = true;
    canvas.wrapped = false;

    canvas.plot_z(&Rgb::RED, 0, 0, 5.0);
    canvas.clear_canvas();

    assert_eq!(canvas.get_pixel(0, 0), Some(&Rgb::BLACK));
    assert_eq!(canvas.get_zbuffer(0, 0), Some(f64::NEG_INFINITY));
}

fn line_points(x0: f64, y0: f64, x1: f64, y1: f64) -> BTreeSet<(i64, i64)> {
    let mut canvas = Canvas::new_with_bg(8, 8, Rgb::WHITE);
    canvas.upper_left_origin = true;
    canvas.wrapped = false;
    canvas.draw_line(Rgb::BLACK, x0, y0, x1, y1);
    black_points(&canvas)
}

fn black_points(canvas: &Canvas) -> BTreeSet<(i64, i64)> {
    let mut points = BTreeSet::new();
    for y in 0..canvas.height() {
        for x in 0..canvas.width() {
            if canvas.get_pixel(x.into(), y.into()) == Some(&Rgb::BLACK) {
                points.insert((x.into(), y.into()));
            }
        }
    }
    points
}

fn non_background_points(canvas: &Canvas, background: Rgb) -> BTreeSet<(i64, i64)> {
    let mut points = BTreeSet::new();
    for y in 0..canvas.height() {
        for x in 0..canvas.width() {
            if canvas.get_pixel(x.into(), y.into()) != Some(&background) {
                points.insert((x.into(), y.into()));
            }
        }
    }
    points
}

fn points<const N: usize>(items: [(i64, i64); N]) -> BTreeSet<(i64, i64)> {
    BTreeSet::from(items)
}

#[test]
fn draw_line_covers_horizontal_vertical_and_single_point() {
    assert_eq!(
        line_points(1.0, 2.0, 5.0, 2.0),
        points([(1, 2), (2, 2), (3, 2), (4, 2), (5, 2)])
    );
    assert_eq!(
        line_points(3.0, 1.0, 3.0, 5.0),
        points([(3, 1), (3, 2), (3, 3), (3, 4), (3, 5)])
    );
    assert_eq!(line_points(4.0, 4.0, 4.0, 4.0), points([(4, 4)]));
}

#[test]
fn draw_line_covers_shallow_and_steep_octants() {
    assert_eq!(
        line_points(1.0, 1.0, 5.0, 3.0),
        points([(1, 1), (2, 1), (3, 2), (4, 2), (5, 3)])
    );
    assert_eq!(
        line_points(1.0, 5.0, 5.0, 3.0),
        points([(1, 5), (2, 5), (3, 4), (4, 4), (5, 3)])
    );
    assert_eq!(
        line_points(1.0, 1.0, 3.0, 5.0),
        points([(1, 1), (1, 2), (2, 3), (2, 4), (3, 5)])
    );
    assert_eq!(
        line_points(1.0, 5.0, 3.0, 1.0),
        points([(1, 5), (1, 4), (2, 3), (2, 2), (3, 1)])
    );
}

#[test]
fn draw_line_reverse_directions_match_forward_lines() {
    assert_eq!(
        line_points(5.0, 3.0, 1.0, 1.0),
        line_points(1.0, 1.0, 5.0, 3.0)
    );
    assert_eq!(
        line_points(5.0, 3.0, 1.0, 5.0),
        line_points(1.0, 5.0, 5.0, 3.0)
    );
    assert_eq!(
        line_points(3.0, 5.0, 1.0, 1.0),
        line_points(1.0, 1.0, 3.0, 5.0)
    );
    assert_eq!(
        line_points(3.0, 1.0, 1.0, 5.0),
        line_points(1.0, 5.0, 3.0, 1.0)
    );
}

#[test]
fn draw_line_uses_odd_width_radius() {
    let mut canvas = Canvas::new_with_bg(5, 5, Rgb::WHITE);
    canvas.upper_left_origin = true;
    canvas.wrapped = false;
    canvas.set_line_width(2.0);
    canvas.draw_line(Rgb::BLACK, 2.0, 2.0, 2.0, 2.0);

    assert_eq!(black_points(&canvas), points([(2, 1), (2, 2), (2, 3)]));
}

#[test]
fn draw_line_z_interpolates_depth_along_driving_axis() {
    let mut canvas = Canvas::new_with_bg(5, 1, Rgb::WHITE);
    canvas.upper_left_origin = true;
    canvas.wrapped = false;

    canvas.draw_line_z(Rgb::BLACK, (0.0, 0.0, 0.0), (4.0, 0.0, 8.0));

    assert_eq!(canvas.get_zbuffer(0, 0), Some(0.0));
    assert_eq!(canvas.get_zbuffer(2, 0), Some(4.0));
    assert_eq!(canvas.get_zbuffer(4, 0), Some(8.0));
}

#[test]
fn thick_steep_lines_use_horizontal_brush() {
    let mut canvas = Canvas::new_with_bg(5, 5, Rgb::WHITE);
    canvas.upper_left_origin = true;
    canvas.wrapped = false;
    canvas.set_line_width(3.0);
    canvas.draw_line(Rgb::BLACK, 2.0, 1.0, 2.0, 3.0);

    assert_eq!(
        black_points(&canvas),
        points([
            (1, 1),
            (2, 1),
            (3, 1),
            (1, 2),
            (2, 2),
            (3, 2),
            (1, 3),
            (2, 3),
            (3, 3)
        ])
    );
}

#[test]
fn fill_uses_clipped_coordinates_even_when_canvas_wraps() {
    let mut canvas = Canvas::new_with_bg(3, 1, Rgb::WHITE);
    canvas.upper_left_origin = true;
    canvas.wrapped = true;
    canvas.plot(&Rgb::BLACK, 1, 0);
    canvas.fill(2, 0, Rgb::new(255, 0, 0), Rgb::BLACK);

    assert_eq!(canvas.get_pixel(0, 0), Some(&Rgb::WHITE));
    assert_eq!(canvas.get_pixel(1, 0), Some(&Rgb::BLACK));
    assert_eq!(canvas.get_pixel(2, 0), Some(&Rgb::new(255, 0, 0)));
    assert!(canvas.wrapped);
}

#[test]
#[should_panic(expected = "edge matrix must contain pairs of points")]
fn draw_lines_rejects_odd_point_count() {
    let mut edges = EdgeMatrix::new();
    edges.push_point(1.0, 1.0, 0.0);
    let mut canvas = Canvas::new_with_bg(4, 4, Rgb::WHITE);
    canvas.draw_lines(&edges);
}

#[test]
fn draw_transformed_applies_matrix_before_drawing() {
    let mut edges = EdgeMatrix::new();
    edges.push_edge(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);

    let mut canvas = Canvas::new_with_bg(4, 4, Rgb::WHITE);
    canvas.upper_left_origin = true;
    canvas.wrapped = false;
    canvas.draw_transformed(&edges, &Matrix::translate(1.0, 2.0, 0.0));

    assert_eq!(black_points(&canvas), points([(1, 2), (2, 2)]));
}

#[test]
fn draw_polygons_scanline_fills_flat_bottom_triangle() {
    let mut polygons = PolygonMatrix::new();
    polygons.add_polygon((1.0, 1.0, 0.0), (5.0, 1.0, 0.0), (3.0, 5.0, 0.0));

    let mut canvas = Canvas::new_with_bg(7, 7, Rgb::WHITE);
    canvas.upper_left_origin = true;
    canvas.wrapped = false;
    canvas.line = Rgb::BLACK;
    canvas.draw_polygons(&polygons);

    assert_eq!(
        non_background_points(&canvas, Rgb::WHITE),
        points([
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
            (5, 1),
            (2, 2),
            (3, 2),
            (4, 2),
            (5, 2),
            (2, 3),
            (3, 3),
            (4, 3),
            (3, 4),
            (4, 4),
            (3, 5)
        ])
    );
}

#[test]
fn draw_polygons_scanline_fills_flat_top_triangle() {
    let mut polygons = PolygonMatrix::new();
    polygons.add_polygon((3.0, 1.0, 0.0), (5.0, 5.0, 0.0), (1.0, 5.0, 0.0));

    let mut canvas = Canvas::new_with_bg(7, 7, Rgb::WHITE);
    canvas.upper_left_origin = true;
    canvas.wrapped = false;
    canvas.line = Rgb::BLACK;
    canvas.draw_polygons(&polygons);

    assert_eq!(
        non_background_points(&canvas, Rgb::WHITE),
        points([
            (3, 1),
            (3, 2),
            (4, 2),
            (2, 3),
            (3, 3),
            (4, 3),
            (2, 4),
            (3, 4),
            (4, 4),
            (5, 4),
            (1, 5),
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 5)
        ])
    );
}

#[test]
fn draw_polygons_scanline_keeps_backface_culling() {
    let mut polygons = PolygonMatrix::new();
    polygons.add_polygon((1.0, 1.0, 0.0), (3.0, 5.0, 0.0), (5.0, 1.0, 0.0));

    let mut canvas = Canvas::new_with_bg(7, 7, Rgb::WHITE);
    canvas.upper_left_origin = true;
    canvas.wrapped = false;
    canvas.line = Rgb::BLACK;
    canvas.draw_polygons(&polygons);

    assert!(black_points(&canvas).is_empty());
}

#[test]
fn draw_polygons_uses_zbuffer_for_overlapping_triangles() {
    let mut near = PolygonMatrix::new();
    near.add_polygon((1.0, 1.0, 10.0), (5.0, 1.0, 10.0), (3.0, 5.0, 10.0));
    let mut far = PolygonMatrix::new();
    far.add_polygon((1.0, 1.0, 1.0), (5.0, 1.0, 1.0), (3.0, 5.0, 1.0));

    let mut canvas = Canvas::new_with_bg(7, 7, Rgb::WHITE);
    canvas.upper_left_origin = true;
    canvas.wrapped = false;

    canvas.line = Rgb::BLUE;
    canvas.draw_polygons(&near);
    canvas.line = Rgb::RED;
    canvas.draw_polygons(&far);

    assert_eq!(canvas.get_pixel(3, 3), Some(&Rgb::BLUE));
    assert_eq!(canvas.get_zbuffer(3, 3), Some(10.0));
}

#[test]
fn draw_polygons_can_flat_shade_with_phong_reflection() {
    let mut polygons = PolygonMatrix::new();
    polygons.add_polygon((1.0, 1.0, 0.0), (5.0, 1.0, 0.0), (3.0, 5.0, 0.0));

    let mut canvas = Canvas::new_with_bg(7, 7, Rgb::WHITE);
    canvas.upper_left_origin = true;
    canvas.wrapped = false;
    canvas.set_polygon_color_mode(PolygonColorMode::PhongReflection);
    canvas.draw_polygons(&polygons);

    assert_eq!(canvas.get_pixel(3, 3), Some(&Rgb::new(139, 139, 139)));
}

fn smooth_test_polygons() -> PolygonMatrix {
    let mut polygons = PolygonMatrix::new();
    polygons.add_polygon((1.0, 1.0, 0.0), (5.0, 1.0, 0.0), (3.0, 5.0, 0.0));
    polygons.add_polygon((1.0, 1.0, 0.0), (3.0, 5.0, 0.0), (1.0, 1.0, 4.0));
    polygons
}

#[test]
fn vertex_normals_average_shared_surface_normals() {
    let polygons = smooth_test_polygons();
    let normals = vertex_normals(polygons.as_matrix().data());
    let shared = vertex_normal(&normals, (1.0, 1.0, 0.0));
    let unshared = vertex_normal(&normals, (5.0, 1.0, 0.0));

    assert!((shared.length() - 1.0).abs() < 1e-12);
    assert_ne!(shared, unshared);
    assert_eq!(unshared, Vector::new(0.0, 0.0, 1.0));
}

#[test]
fn draw_polygons_can_gouraud_shade_from_vertex_normals() {
    let mut canvas = Canvas::new_with_bg(7, 7, Rgb::WHITE);
    canvas.upper_left_origin = true;
    canvas.wrapped = false;
    canvas.set_shading_mode(ShadingMode::Gouraud);
    canvas.draw_polygons(&smooth_test_polygons());

    assert_ne!(canvas.get_pixel(3, 3), Some(&Rgb::WHITE));
    assert_ne!(canvas.get_pixel(3, 3), Some(&Rgb::new(150, 63, 91)));
}

#[test]
fn draw_polygons_can_phong_shade_from_interpolated_normals() {
    let mut canvas = Canvas::new_with_bg(7, 7, Rgb::WHITE);
    canvas.upper_left_origin = true;
    canvas.wrapped = false;
    canvas.set_shading_mode(ShadingMode::Phong);
    canvas.draw_polygons(&smooth_test_polygons());

    assert_ne!(canvas.get_pixel(3, 3), Some(&Rgb::WHITE));
    assert_ne!(canvas.get_pixel(3, 3), Some(&Rgb::new(150, 63, 91)));
}

#[test]
fn draw_polygons_can_toon_shade_with_banded_lighting() {
    let mut phong = Canvas::new_with_bg(7, 7, Rgb::WHITE);
    phong.upper_left_origin = true;
    phong.wrapped = false;
    phong.set_shading_mode(ShadingMode::Phong);
    phong.draw_polygons(&smooth_test_polygons());

    let mut toon = Canvas::new_with_bg(7, 7, Rgb::WHITE);
    toon.upper_left_origin = true;
    toon.wrapped = false;
    toon.set_shading_mode(ShadingMode::Toon);
    toon.draw_polygons(&smooth_test_polygons());

    assert_ne!(toon.get_pixel(3, 3), Some(&Rgb::WHITE));
    assert_ne!(toon.get_pixel(3, 3), phong.get_pixel(3, 3));
}

#[test]
fn draw_polygons_rejects_non_finite_y_before_smooth_scan_conversion() {
    for shading_mode in [ShadingMode::Gouraud, ShadingMode::Phong, ShadingMode::Toon] {
        for invalid_y in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut polygons = PolygonMatrix::new();
            polygons.add_polygon((1.0, invalid_y, 0.0), (5.0, 1.0, 0.0), (3.0, 5.0, 0.0));

            let mut canvas = Canvas::new_with_bg(7, 7, Rgb::WHITE);
            canvas.upper_left_origin = true;
            canvas.wrapped = false;
            canvas.set_shading_mode(shading_mode);
            canvas.draw_polygons(&polygons);

            assert!(
                canvas.pixels().iter().all(|pixel| *pixel == Rgb::WHITE),
                "{shading_mode:?} should reject invalid y {invalid_y}"
            );
        }
    }
}

#[test]
fn draw_polygons_can_render_wireframe_edges() {
    let mut polygons = PolygonMatrix::new();
    polygons.add_polygon((1.0, 1.0, 0.0), (5.0, 1.0, 0.0), (3.0, 5.0, 0.0));

    let mut canvas = Canvas::new_with_bg(7, 7, Rgb::WHITE);
    canvas.upper_left_origin = true;
    canvas.wrapped = false;
    canvas.line = Rgb::BLACK;
    canvas.set_shading_mode(ShadingMode::Wireframe);
    canvas.draw_polygons(&polygons);

    assert_eq!(canvas.get_pixel(3, 3), Some(&Rgb::WHITE));
    assert!(black_points(&canvas).contains(&(1, 1)));
    assert!(black_points(&canvas).contains(&(5, 1)));
    assert!(black_points(&canvas).contains(&(3, 5)));
}

#[test]
fn triangle_color_modes_are_stable_and_varied() {
    let base = Rgb::GREEN;
    let colors: Vec<_> = (0..12)
        .map(|index| triangle_color(PolygonColorMode::DeterministicRandom, base, index))
        .collect();
    let unique: BTreeSet<_> = colors.iter().map(Rgb::values).collect();

    assert_eq!(triangle_color(PolygonColorMode::LineColor, base, 7), base);
    assert_eq!(
        triangle_color(PolygonColorMode::PhongReflection, base, 7),
        base
    );
    assert_eq!(
        triangle_color(PolygonColorMode::DeterministicRandom, base, 7),
        colors[7]
    );
    assert_eq!(
        triangle_color(PolygonColorMode::TintedFromLine, base, 0),
        base
    );
    assert!(unique.len() > 8);
}

#[test]
fn draw_lines_no_longer_saves_animation_frames() {
    fs::create_dir_all("anim").expect("create animation dir");
    let prefix = format!("test-frame-count-{}-", std::process::id());
    let mut edges = EdgeMatrix::new();
    edges.push_edge(0.0, 0.0, 0.0, 1.0, 1.0, 0.0);
    edges.push_edge(1.0, 1.0, 0.0, 2.0, 2.0, 0.0);

    let mut canvas = Canvas::new_with_bg(4, 4, Rgb::WHITE);
    canvas.try_draw_lines(&edges);

    assert!(!std::path::Path::new(&format!("anim/{prefix}00000000.ppm")).exists());
}

#[test]
fn frame_recorder_captures_explicit_frames() {
    let prefix = format!("test-recorder-{}-", std::process::id());
    let mut recorder = FrameRecorder::new("anim", prefix.clone());
    let canvas = Canvas::new_with_bg(2, 2, Rgb::WHITE);

    recorder.capture(&canvas).expect("capture frame");

    assert_eq!(recorder.frame_index(), 1);
    let _ = fs::remove_file(format!("anim/{prefix}00000000.ppm"));
}

#[test]
fn frame_recorder_can_capture_drawn_transformed_edges() {
    let prefix = format!("test-recorder-drawn-{}-", std::process::id());
    let mut recorder = FrameRecorder::new("anim", prefix.clone());
    let canvas = Canvas::new_with_bg(3, 3, Rgb::WHITE);
    let mut edges = EdgeMatrix::new();
    edges.push_edge(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);

    recorder
        .capture_drawn(&canvas, &edges, &Matrix::translate(1.0, 1.0, 0.0))
        .expect("capture transformed frame");

    assert_eq!(recorder.frame_index(), 1);
    let _ = fs::remove_file(format!("anim/{prefix}00000000.ppm"));
}
