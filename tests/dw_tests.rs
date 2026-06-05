#![cfg(feature = "old_parser")]

use gartus::gmath::edge_matrix::{DEFAULT_CURVE_STEP, EdgeMatrix};
use gartus::gmath::matrix::*;
use gartus::gmath::polygon_matrix::PolygonMatrix;
use gartus::gmath::vector::Vector;
use gartus::graphics::colors::*;
use gartus::graphics::display::{Canvas, PolygonColorMode, ShadingMode};
use gartus::graphics::lighting::{Lighting, PointLight, ReflectionConstants};
use gartus::parser::Parser;

const REFERENCE_3D_STEPS: usize = 100;

fn pixels_eq(a: &Canvas, b: &Canvas) -> bool {
    a.pixels()
        .iter()
        .zip(b.pixels().iter())
        .all(|(p, q)| p == q)
}

fn course_lighting_values() -> Lighting {
    Lighting {
        view: Vector::new(0.0, 0.0, 1.0),
        ambient: Rgb::new(50, 50, 50),
        point_light: PointLight::new(Vector::new(0.75, 0.75, 1.0), Rgb::WHITE),
        point_lights: Vec::new(),
        ambient_reflection: ReflectionConstants::new(0.1, 0.1, 0.1),
        diffuse_reflection: ReflectionConstants::new(0.75, 0.25, 0.25),
        specular_reflection: ReflectionConstants::new(0.25, 0.25, 0.75),
        specular_exponent: gartus::graphics::lighting::DEFAULT_SPECULAR_EXPONENT,
    }
}

fn old_parser_reference_canvas(width: u32, height: u32, line: Rgb, bg: Rgb) -> Canvas {
    let mut canvas = Canvas::new_with_bg(width, height, bg);
    canvas.line = line;
    canvas.set_shading_mode(ShadingMode::Flat);
    canvas.set_polygon_color_mode(PolygonColorMode::PhongReflection);
    canvas
}

fn draw_manual_cstack_robot(manual: &mut Canvas) {
    // script starts at identity; first command is push (saves identity), then body CS:
    //   move 250 250 0  → T(250,250,0)
    //   rotate y -30    → * Ry(-30)
    let body = &Matrix::identity_matrix(4)
        * &(&Matrix::translate(250.0, 250.0, 0.0) * &Matrix::rotate_y(-30.0));

    // BODY box
    let mut pm = PolygonMatrix::new();
    pm.add_box((-100.0, 125.0, 50.0), 200.0, 250.0, 100.0);
    manual.draw_polygons(&pm.apply(&body));

    // HEAD: body * T(0,175,0) * Ry(90)
    let head = &body * &(&Matrix::translate(0.0, 175.0, 0.0) * &Matrix::rotate_y(90.0));
    let mut pm = PolygonMatrix::new();
    pm.add_sphere((0.0, 0.0, 0.0), 50.0, REFERENCE_3D_STEPS);
    manual.draw_polygons(&pm.apply(&head));

    // LEFT ARM: body * T(-100,125,0) * Rx(-45)
    let left_arm = &body * &(&Matrix::translate(-100.0, 125.0, 0.0) * &Matrix::rotate_x(-45.0));
    let mut pm = PolygonMatrix::new();
    pm.add_box((-40.0, 0.0, 40.0), 40.0, 100.0, 80.0);
    manual.draw_polygons(&pm.apply(&left_arm));

    // LEFT LOWER ARM: left_arm * T(-20,-100,0)
    let left_lower = &left_arm * &Matrix::translate(-20.0, -100.0, 0.0);
    let mut pm = PolygonMatrix::new();
    pm.add_box((-10.0, 0.0, 10.0), 20.0, 125.0, 20.0);
    manual.draw_polygons(&pm.apply(&left_lower));

    // RIGHT ARM: body * T(100,125,0) * Rx(-45)
    let right_arm = &body * &(&Matrix::translate(100.0, 125.0, 0.0) * &Matrix::rotate_x(-45.0));
    let mut pm = PolygonMatrix::new();
    pm.add_box((0.0, 0.0, 40.0), 40.0, 100.0, 80.0);
    manual.draw_polygons(&pm.apply(&right_arm));

    // RIGHT LOWER ARM: right_arm * T(20,-100,0) * Rx(-20)
    let right_lower =
        &right_arm * &(&Matrix::translate(20.0, -100.0, 0.0) * &Matrix::rotate_x(-20.0));
    let mut pm = PolygonMatrix::new();
    pm.add_box((-10.0, 0.0, 10.0), 20.0, 125.0, 20.0);
    manual.draw_polygons(&pm.apply(&right_lower));

    // LEFT LEG: body * T(-100,-125,0)
    let left_leg = &body * &Matrix::translate(-100.0, -125.0, 0.0);
    let mut pm = PolygonMatrix::new();
    pm.add_box((0.0, 0.0, 40.0), 50.0, 120.0, 80.0);
    manual.draw_polygons(&pm.apply(&left_leg));

    // RIGHT LEG: body * T(100,-125,0)
    let right_leg = &body * &Matrix::translate(100.0, -125.0, 0.0);
    let mut pm = PolygonMatrix::new();
    pm.add_box((-50.0, 0.0, 40.0), 50.0, 120.0, 80.0);
    manual.draw_polygons(&pm.apply(&right_leg));
}

#[test]
fn course_lighting_face_scene() {
    let color = Rgb::new(0, 255, 0);
    let bg = Rgb::WHITE;
    let lighting = course_lighting_values();
    const W: u32 = 500;
    const H: u32 = 500;

    let script = "\
push
move
250 300 0
push
rotate
x 30
rotate
y -20
box
-37 37 37 75 75 75
pop
sphere
-125 130 0 60
sphere
125 130 0 60
push
rotate
x 30
rotate
y 20
scale
1.5 1 1
torus
0 -200 0 25 125
pop
pop
display";

    let mut dw = Parser::new_with_bg("test", W, H, &color, &bg);
    dw.set_display_enabled(false);
    // The 10_mdl C/Python reference computes Phong reflection once per triangle
    // normal, then fills the triangle. That maps to our flat shading mode plus
    // PhongReflection color mode; our ShadingMode::Phong is smooth per-pixel
    // normal interpolation and makes box faces look too shiny for this fixture.
    dw.set_shading_mode(ShadingMode::Flat);
    dw.set_polygon_color_mode(PolygonColorMode::PhongReflection);
    dw.set_lighting(lighting.clone());
    dw.parse_string(script)
        .expect("course lighting script valid");

    let mut manual = old_parser_reference_canvas(W, H, color, bg);
    manual.set_lighting(lighting);

    let base = Matrix::translate(250.0, 300.0, 0.0);

    let cube_cs = &base * &(&Matrix::rotate_x(30.0) * &Matrix::rotate_y(-20.0));
    let mut pm = PolygonMatrix::new();
    pm.add_box((-37.0, 37.0, 37.0), 75.0, 75.0, 75.0);
    manual.draw_polygons(&pm.apply(&cube_cs));

    let mut pm = PolygonMatrix::new();
    pm.add_sphere((-125.0, 130.0, 0.0), 60.0, REFERENCE_3D_STEPS);
    manual.draw_polygons(&pm.apply(&base));

    let mut pm = PolygonMatrix::new();
    pm.add_sphere((125.0, 130.0, 0.0), 60.0, REFERENCE_3D_STEPS);
    manual.draw_polygons(&pm.apply(&base));

    let torus_cs = &base
        * &(&(&Matrix::rotate_x(30.0) * &Matrix::rotate_y(20.0)) * &Matrix::scale(1.5, 1.0, 1.0));
    let mut pm = PolygonMatrix::new();
    pm.add_torus((0.0, -200.0, 0.0), 25.0, 125.0, REFERENCE_3D_STEPS);
    manual.draw_polygons(&pm.apply(&torus_cs));

    assert!(
        pixels_eq(dw.canvas(), &manual),
        "old parser should render the course lighting face scene with the provided lighting values"
    );
}

#[test]
fn script_light() {
    let color = Rgb::new(0, 255, 0);
    const W: u32 = 500;
    const H: u32 = 500;

    let mut dw = Parser::new("./tests/script_light", W, H, &color);
    dw.set_display_enabled(false);
    dw.set_shading_mode(ShadingMode::Flat);
    dw.set_polygon_color_mode(PolygonColorMode::PhongReflection);
    dw.parse_file().expect("Script is valid");
    assert!(
        std::path::Path::new("light.png").exists(),
        "script should save light.png"
    );
    let _ = std::fs::remove_file("light.png");

    let mut manual = old_parser_reference_canvas(W, H, color, Rgb::BLACK);
    draw_manual_cstack_robot(&mut manual);

    assert!(
        pixels_eq(dw.canvas(), &manual),
        "script_light final canvas should match manually drawn flat-lit robot"
    );
}

#[test]
fn script_solid() {
    let green = Rgb::new(0, 255, 0);
    const W: u32 = 500;
    const H: u32 = 500;

    let mut dw = Parser::new("./tests/script_solid", W, H, &green);
    dw.set_display_enabled(false);
    dw.set_polygon_color_mode(PolygonColorMode::DeterministicRandom);
    dw.parse_file().expect("Script is valid");
    assert!(
        std::path::Path::new("solid.png").exists(),
        "script should save solid.png"
    );
    let _ = std::fs::remove_file("solid.png");

    let mut manual = old_parser_reference_canvas(W, H, green, Rgb::BLACK);
    manual.set_polygon_color_mode(PolygonColorMode::DeterministicRandom);
    draw_manual_cstack_robot(&mut manual);

    assert!(
        pixels_eq(dw.canvas(), &manual),
        "script_solid final canvas should match manual robot after clear/pop sequence"
    );
}

#[test]
fn script_cstack() {
    let green = Rgb::new(0, 255, 0);
    const W: u32 = 500;
    const H: u32 = 500;

    // --- parser side ---
    let mut dw = Parser::new("./tests/script_cstack", W, H, &green);
    dw.set_display_enabled(false);
    dw.parse_file().expect("Script is valid");
    assert!(
        std::path::Path::new("robot.png").exists(),
        "script should save robot.png"
    );
    let _ = std::fs::remove_file("robot.png");

    // --- manual side: replicate every CS step from script_cstack ---
    // Parser::new uses the old-parser reference render state with black background.
    let mut manual = old_parser_reference_canvas(W, H, green, Rgb::BLACK);
    draw_manual_cstack_robot(&mut manual);

    assert!(
        pixels_eq(dw.canvas(), &manual),
        "parser script_cstack should match manual CS construction"
    );
}

#[test]
fn script_polygons() {
    // box + sphere drawn, then clear, then torus — final canvas = torus only.
    // CS = I * Rx(20) * Ry(20) * T(150,200,0) persists through clear.
    let color = Rgb::new(0, 255, 0);
    let bg = Rgb::default();
    const W: u32 = 500;
    const H: u32 = 500;

    // --- parser side ---
    let mut dw = Parser::new_with_bg("script_polygons", W, H, &color, &bg);
    dw.set_display_enabled(false);
    dw.parse_string(
        "box\n0 0 0 200 100 400\nrotate\nx 20\nrotate\ny 20\nmove\n150 200 0\nsphere\n0 0 0 200\nclear\ntorus\n0 0 0 25 150",
    )
    .unwrap();

    // --- manual side: only torus survives the clear ---
    let mut manual = old_parser_reference_canvas(W, H, color, bg);
    // CS = Rx(20) * Ry(20) * T(150,200,0)
    let cs = &(&Matrix::rotate_x(20.0) * &Matrix::rotate_y(20.0))
        * &Matrix::translate(150.0, 200.0, 0.0);
    let mut pm = PolygonMatrix::new();
    pm.add_torus((0.0, 0.0, 0.0), 25.0, 150.0, REFERENCE_3D_STEPS);
    manual.draw_polygons(&pm.apply(&cs));

    assert!(
        pixels_eq(dw.canvas(), &manual),
        "script_polygons final canvas should match manual torus"
    );
}

#[test]
fn script_transform() {
    // Cube wireframe at T(250,250,0), then push/scale/cube again/pop.
    // Rotations after pop draw nothing; no final clear so canvas holds all lines.
    // Parity: manual replicates both draw passes then compares pixel-for-pixel.
    let color = Rgb::new(0, 255, 0);
    let bg = Rgb::default();
    const W: u32 = 500;
    const H: u32 = 500;

    const CUBE: &[&str] = &[
        "0 0 0 100 0 0",
        "100 0 0 100 100 0",
        "100 100 0 0 100 0",
        "0 100 0 0 0 0",
        "0 0 100 100 0 100",
        "100 0 100 100 100 100",
        "100 100 100 0 100 100",
        "0 100 100 0 0 100",
        "0 0 0 0 0 100",
        "0 100 0 0 100 100",
        "100 100 0 100 100 100",
        "100 0 0 100 0 100",
    ];

    // --- parser side ---
    let mut dw = Parser::new_with_bg("test", W, H, &color, &bg);
    dw.set_display_enabled(false);
    let mut script = String::from("move\n250 250 0\n");
    for e in CUBE {
        script.push_str(&format!("line\n{e}\n"));
    }
    script.push_str("push\nscale\n2 2 2\n");
    for e in CUBE {
        script.push_str(&format!("line\n{e}\n"));
    }
    script.push_str("pop\nrotate\nz 20\nrotate\nx 20\nrotate\ny 20");
    dw.parse_string(&script).unwrap();

    // --- manual side ---
    let mut manual = old_parser_reference_canvas(W, H, color, bg);
    let cs1 = &Matrix::identity_matrix(4) * &Matrix::translate(250.0, 250.0, 0.0);
    let cs2 = &cs1 * &Matrix::scale(2.0, 2.0, 2.0);

    let cube_pts: &[(f64, f64, f64, f64, f64, f64)] = &[
        (0., 0., 0., 100., 0., 0.),
        (100., 0., 0., 100., 100., 0.),
        (100., 100., 0., 0., 100., 0.),
        (0., 100., 0., 0., 0., 0.),
        (0., 0., 100., 100., 0., 100.),
        (100., 0., 100., 100., 100., 100.),
        (100., 100., 100., 0., 100., 100.),
        (0., 100., 100., 0., 0., 100.),
        (0., 0., 0., 0., 0., 100.),
        (0., 100., 0., 0., 100., 100.),
        (100., 100., 0., 100., 100., 100.),
        (100., 0., 0., 100., 0., 100.),
    ];
    let mut em1 = EdgeMatrix::new();
    for &(x0, y0, z0, x1, y1, z1) in cube_pts {
        em1.push_edge(x0, y0, z0, x1, y1, z1);
    }
    manual.draw_lines(&em1.apply(&cs1));
    let mut em2 = EdgeMatrix::new();
    for &(x0, y0, z0, x1, y1, z1) in cube_pts {
        em2.push_edge(x0, y0, z0, x1, y1, z1);
    }
    manual.draw_lines(&em2.apply(&cs2));

    assert!(
        pixels_eq(dw.canvas(), &manual),
        "script_transform canvas should match manual cube wireframes"
    );
}

#[test]
fn curve_script() {
    // clear erases earlier curves; final canvas = 5 curves drawn in second batch.
    // Parity: manual replicates those 5 edge draws at identity CS.
    let color = Rgb::new(0, 255, 0);
    let bg = Rgb::default();
    const W: u32 = 500;
    const H: u32 = 500;

    // --- parser side ---
    let mut dw = Parser::new_with_bg("test", W, H, &color, &bg);
    dw.set_display_enabled(false);
    dw.parse_string(
        "circle\n250 250 0 200\ncircle\n175 325 0 50\nhermite\n150 150 350 150 -100 -100 100 150\nbezier\n200 250 150 50 300 250 300 250\nclear\ncircle\n250 250 0 200\ncircle\n175 325 0 50\ncircle\n325 325 0 50\nhermite\n150 150 350 150 -100 -100 100 150\nbezier\n200 250 150 50 300 250 300 250\nsave\nface.png",
    )
    .unwrap();
    assert!(
        std::path::Path::new("face.png").exists(),
        "save should create face.png"
    );
    let _ = std::fs::remove_file("face.png");

    // --- manual side: only the post-clear batch survives ---
    let mut manual = old_parser_reference_canvas(W, H, color, bg);
    let cs = Matrix::identity_matrix(4);
    let mut em = EdgeMatrix::new();
    em.add_circle(250.0, 250.0, 0.0, 200.0, DEFAULT_CURVE_STEP);
    em.add_circle(175.0, 325.0, 0.0, 50.0, DEFAULT_CURVE_STEP);
    em.add_circle(325.0, 325.0, 0.0, 50.0, DEFAULT_CURVE_STEP);
    em.add_hermite(
        (150.0, 150.0),
        (350.0, 150.0),
        (-100.0, -100.0),
        (100.0, 150.0),
    );
    em.add_bezier3(
        (200.0, 250.0),
        (150.0, 50.0),
        (300.0, 250.0),
        (300.0, 250.0),
    );
    manual.draw_lines(&em.apply(&cs));

    assert!(
        pixels_eq(dw.canvas(), &manual),
        "curve_script final canvas should match manual edge draws"
    );
}

#[test]
fn script_3d() {
    // Box + sphere + torus each in their own push/pop CS, all visible in final canvas.
    let color = Rgb::new(0, 255, 0);
    const W: u32 = 500;
    const H: u32 = 500;

    // --- parser side ---
    let mut dw = Parser::new("./tests/script_3d", W, H, &color);
    dw.set_display_enabled(false);
    dw.parse_file().unwrap();
    assert!(
        std::path::Path::new("scene_3d.png").exists(),
        "script should save scene_3d.png"
    );
    let _ = std::fs::remove_file("scene_3d.png");

    // --- manual side ---
    let mut manual = old_parser_reference_canvas(W, H, color, Rgb::BLACK);

    // box: T(130,350,0) * Ry(20) * Rx(15)
    let box_cs = &(&Matrix::translate(130.0, 350.0, 0.0) * &Matrix::rotate_y(20.0))
        * &Matrix::rotate_x(15.0);
    let mut pm = PolygonMatrix::new();
    pm.add_box((-50.0, -50.0, -50.0), 100.0, 100.0, 100.0);
    manual.draw_polygons(&pm.apply(&box_cs));

    // sphere: T(250,250,0)
    let sphere_cs = Matrix::translate(250.0, 250.0, 0.0);
    let mut pm = PolygonMatrix::new();
    pm.add_sphere((0.0, 0.0, 0.0), 80.0, REFERENCE_3D_STEPS);
    manual.draw_polygons(&pm.apply(&sphere_cs));

    // torus: T(370,150,0) * Rx(60)
    let torus_cs = &Matrix::translate(370.0, 150.0, 0.0) * &Matrix::rotate_x(60.0);
    let mut pm = PolygonMatrix::new();
    pm.add_torus((0.0, 0.0, 0.0), 20.0, 60.0, REFERENCE_3D_STEPS);
    manual.draw_polygons(&pm.apply(&torus_cs));

    assert!(
        pixels_eq(dw.canvas(), &manual),
        "script_3d final canvas should match manual construction"
    );
}

// ---------------------------------------------------------------------------
// Parity tests: parser output == manual construction via library APIs
// ---------------------------------------------------------------------------

#[test]
fn parity_line_identity_cs() {
    let color = Rgb::new(200, 100, 50);
    let bg = Rgb::default();
    const W: u32 = 200;
    const H: u32 = 200;

    let mut dw = Parser::new_with_bg("test", W, H, &color, &bg);
    dw.set_display_enabled(false);
    dw.parse_string("line\n10 10 0 150 180 0").unwrap();

    let mut manual = old_parser_reference_canvas(W, H, color, bg);
    let cs = Matrix::identity_matrix(4);
    let mut em = EdgeMatrix::new();
    em.push_edge(10.0, 10.0, 0.0, 150.0, 180.0, 0.0);
    manual.draw_lines(&em.apply(&cs));

    assert!(
        pixels_eq(dw.canvas(), &manual),
        "parser line should match manual draw_lines"
    );
}

#[test]
fn parity_line_after_translate() {
    let color = Rgb::new(0, 200, 255);
    let bg = Rgb::default();
    const W: u32 = 300;
    const H: u32 = 300;

    let mut dw = Parser::new_with_bg("test", W, H, &color, &bg);
    dw.set_display_enabled(false);
    dw.parse_string("move\n50 80 0\nline\n0 0 0 100 100 0")
        .unwrap();

    let mut manual = old_parser_reference_canvas(W, H, color, bg);
    let cs = &Matrix::identity_matrix(4) * &Matrix::translate(50.0, 80.0, 0.0);
    let mut em = EdgeMatrix::new();
    em.push_edge(0.0, 0.0, 0.0, 100.0, 100.0, 0.0);
    manual.draw_lines(&em.apply(&cs));

    assert!(
        pixels_eq(dw.canvas(), &manual),
        "parser line+move should match manual"
    );
}

#[test]
fn parity_box_identity_cs() {
    let color = Rgb::new(255, 128, 0);
    let bg = Rgb::default();
    const W: u32 = 300;
    const H: u32 = 300;

    let mut dw = Parser::new_with_bg("test", W, H, &color, &bg);
    dw.set_display_enabled(false);
    dw.parse_string("box\n50 50 0 100 80 60").unwrap();

    let mut manual = old_parser_reference_canvas(W, H, color, bg);
    let cs = Matrix::identity_matrix(4);
    let mut pm = PolygonMatrix::new();
    pm.add_box((50.0, 50.0, 0.0), 100.0, 80.0, 60.0);
    manual.draw_polygons(&pm.apply(&cs));

    assert!(
        pixels_eq(dw.canvas(), &manual),
        "parser box should match manual draw_polygons"
    );
}

#[test]
fn parity_sphere_after_translate() {
    let color = Rgb::new(100, 200, 50);
    let bg = Rgb::default();
    const W: u32 = 400;
    const H: u32 = 400;

    let mut dw = Parser::new_with_bg("test", W, H, &color, &bg);
    dw.set_display_enabled(false);
    dw.parse_string("move\n200 200 0\nsphere\n0 0 0 100")
        .unwrap();

    let mut manual = old_parser_reference_canvas(W, H, color, bg);
    let cs = &Matrix::identity_matrix(4) * &Matrix::translate(200.0, 200.0, 0.0);
    let mut pm = PolygonMatrix::new();
    pm.add_sphere((0.0, 0.0, 0.0), 100.0, REFERENCE_3D_STEPS);
    manual.draw_polygons(&pm.apply(&cs));

    assert!(
        pixels_eq(dw.canvas(), &manual),
        "parser sphere+move should match manual"
    );
}

#[test]
fn parity_torus_after_translate() {
    let color = Rgb::new(180, 60, 220);
    let bg = Rgb::default();
    const W: u32 = 400;
    const H: u32 = 400;

    let mut dw = Parser::new_with_bg("test", W, H, &color, &bg);
    dw.set_display_enabled(false);
    dw.parse_string("move\n200 200 0\ntorus\n0 0 0 30 100")
        .unwrap();

    let mut manual = old_parser_reference_canvas(W, H, color, bg);
    let cs = &Matrix::identity_matrix(4) * &Matrix::translate(200.0, 200.0, 0.0);
    let mut pm = PolygonMatrix::new();
    pm.add_torus((0.0, 0.0, 0.0), 30.0, 100.0, REFERENCE_3D_STEPS);
    manual.draw_polygons(&pm.apply(&cs));

    assert!(
        pixels_eq(dw.canvas(), &manual),
        "parser torus+move should match manual"
    );
}

#[test]
fn parity_push_pop_two_boxes() {
    // Draw box at T(30,30,0), push, move to T(30,30,0)*T(100,0,0), draw second box, pop.
    // Parser and manual should produce identical canvases.
    let color = Rgb::new(0, 180, 180);
    let bg = Rgb::default();
    const W: u32 = 400;
    const H: u32 = 300;

    let mut dw = Parser::new_with_bg("test", W, H, &color, &bg);
    dw.set_display_enabled(false);
    dw.parse_string(
        "move\n30 30 0\nbox\n0 0 0 60 60 1\npush\nmove\n100 0 0\nbox\n0 0 0 60 60 1\npop",
    )
    .unwrap();

    let mut manual = old_parser_reference_canvas(W, H, color, bg);
    let cs1 = &Matrix::identity_matrix(4) * &Matrix::translate(30.0, 30.0, 0.0);
    let mut pm1 = PolygonMatrix::new();
    pm1.add_box((0.0, 0.0, 0.0), 60.0, 60.0, 1.0);
    manual.draw_polygons(&pm1.apply(&cs1));

    let cs2 = &cs1 * &Matrix::translate(100.0, 0.0, 0.0);
    let mut pm2 = PolygonMatrix::new();
    pm2.add_box((0.0, 0.0, 0.0), 60.0, 60.0, 1.0);
    manual.draw_polygons(&pm2.apply(&cs2));

    assert!(
        pixels_eq(dw.canvas(), &manual),
        "parser push/pop two boxes should match manual"
    );
}

#[test]
fn parity_rotate_then_box() {
    // CS = I * R_y(45), then draw a box.
    let color = Rgb::new(255, 0, 128);
    let bg = Rgb::default();
    const W: u32 = 300;
    const H: u32 = 300;

    let mut dw = Parser::new_with_bg("test", W, H, &color, &bg);
    dw.set_display_enabled(false);
    dw.parse_string("move\n150 150 0\nrotate\ny 45\nbox\n-50 -50 -50 100 100 100")
        .unwrap();

    let mut manual = old_parser_reference_canvas(W, H, color, bg);
    let cs = &(&Matrix::identity_matrix(4) * &Matrix::translate(150.0, 150.0, 0.0))
        * &Matrix::rotate_y(45.0);
    let mut pm = PolygonMatrix::new();
    pm.add_box((-50.0, -50.0, -50.0), 100.0, 100.0, 100.0);
    manual.draw_polygons(&pm.apply(&cs));

    assert!(
        pixels_eq(dw.canvas(), &manual),
        "parser rotate+box should match manual"
    );
}

#[test]
fn parity_circle_identity_cs() {
    let color = Rgb::new(0, 150, 255);
    let bg = Rgb::default();
    const W: u32 = 400;
    const H: u32 = 400;

    let mut dw = Parser::new_with_bg("test", W, H, &color, &bg);
    dw.set_display_enabled(false);
    dw.parse_string("circle\n200 200 0 150").unwrap();

    let mut manual = old_parser_reference_canvas(W, H, color, bg);
    let cs = Matrix::identity_matrix(4);
    let mut em = EdgeMatrix::new();
    em.add_circle(200.0, 200.0, 0.0, 150.0, DEFAULT_CURVE_STEP);
    manual.draw_lines(&em.apply(&cs));

    assert!(
        pixels_eq(dw.canvas(), &manual),
        "parser circle should match manual"
    );
}
