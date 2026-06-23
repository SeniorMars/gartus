#[cfg(feature = "external")]
use gartus::mdl::run_source;
use gartus::{
    gmath::{matrix::Matrix, polygon_matrix::PolygonMatrix},
    graphics::colors::Rgb,
    graphics::{
        display::{Canvas, PolygonColorMode, ShadingMode},
        lighting::ReflectionConstants,
    },
    mdl::executor::execute_compiled_frame,
    mdl::{Command, RenderConfig, compile_file, parse_file, run_file},
};

const REFERENCE_3D_STEPS: usize = 100;

#[derive(Clone, Copy)]
struct TestMaterial {
    ambient: ReflectionConstants,
    diffuse: ReflectionConstants,
    specular: ReflectionConstants,
}

fn pixels_eq(a: &Canvas, b: &Canvas) -> bool {
    a.pixels()
        .iter()
        .zip(b.pixels().iter())
        .all(|(p, q)| p == q)
}

fn manual_face_canvas() -> Canvas {
    let mut canvas = Canvas::new_with_bg(500, 500, Rgb::WHITE);
    canvas.set_line_color(Rgb::BLACK);
    canvas.set_shading_mode(ShadingMode::Flat);
    canvas.set_polygon_color_mode(PolygonColorMode::PhongReflection);

    let shiny_purple = TestMaterial {
        ambient: ReflectionConstants::new(0.3, 0.3, 0.3),
        diffuse: ReflectionConstants::new(0.2, 0.0, 0.2),
        specular: ReflectionConstants::new(1.0, 1.0, 1.0),
    };
    let shiny_teal = TestMaterial {
        ambient: ReflectionConstants::new(0.3, 0.3, 0.3),
        diffuse: ReflectionConstants::new(0.0, 0.2, 0.2),
        specular: ReflectionConstants::new(0.0, 0.8, 0.8),
    };
    let dull_yellow = TestMaterial {
        ambient: ReflectionConstants::new(0.3, 0.3, 0.0),
        diffuse: ReflectionConstants::new(0.8, 0.8, 0.0),
        specular: ReflectionConstants::new(0.2, 0.2, 0.0),
    };
    let default_material = TestMaterial {
        ambient: ReflectionConstants::new(0.1, 0.1, 0.1),
        diffuse: ReflectionConstants::new(0.5, 0.5, 0.5),
        specular: ReflectionConstants::new(0.5, 0.5, 0.5),
    };

    let base = Matrix::translate(250.0, 300.0, 0.0);

    let cube_cs = &base * &(&Matrix::rotate_x(30.0) * &Matrix::rotate_y(-20.0));
    let mut polygons = PolygonMatrix::new();
    polygons.add_box((-40.0, 40.0, 40.0), 80.0, 80.0, 80.0);
    draw_polygons_with_material(&mut canvas, &polygons.apply(&cube_cs), default_material);

    let mut polygons = PolygonMatrix::new();
    polygons.add_sphere((-125.0, 130.0, 0.0), 60.0, REFERENCE_3D_STEPS);
    draw_polygons_with_material(&mut canvas, &polygons.apply(&base), shiny_purple);

    let mut polygons = PolygonMatrix::new();
    polygons.add_sphere((125.0, 130.0, 0.0), 60.0, REFERENCE_3D_STEPS);
    draw_polygons_with_material(&mut canvas, &polygons.apply(&base), shiny_teal);

    let torus_cs = &base
        * &(&(&Matrix::rotate_x(30.0) * &Matrix::rotate_y(20.0)) * &Matrix::scale(1.5, 1.0, 1.0));
    let mut polygons = PolygonMatrix::new();
    polygons.add_torus((0.0, -200.0, 0.0), 25.0, 125.0, REFERENCE_3D_STEPS);
    draw_polygons_with_material(&mut canvas, &polygons.apply(&torus_cs), dull_yellow);

    canvas
}

fn draw_polygons_with_material(
    canvas: &mut Canvas,
    polygons: &PolygonMatrix,
    material: TestMaterial,
) {
    let previous = canvas.lighting();
    let mut lighting = previous.clone();
    lighting.ambient_reflection = material.ambient;
    lighting.diffuse_reflection = material.diffuse;
    lighting.specular_reflection = material.specular;
    canvas.set_lighting(lighting);
    canvas.draw_polygons(polygons);
    canvas.set_lighting(previous);
}

#[test]
fn test_mdl_reference_validation_file_parses() {
    let file = "tests/test.mdl";
    let program = parse_file(file).expect("test.mdl should parse all 11_anim command forms");

    assert!(
        program
            .commands
            .iter()
            .any(|command| matches!(command.node, Command::Output(_))),
        "test.mdl should include output commands from the reference validation script"
    );

    let errors = compile_file(file).expect_err("test.mdl is parser-validation, not runnable MDL");
    assert!(
        errors
            .iter()
            .any(|error| error.message.contains("unknown tween knob list `list1`")),
        "compile diagnostics should explain the unresolved tween list"
    );
    assert!(
        errors
            .iter()
            .any(|error| error.message.contains("require a `frames` command")),
        "compile diagnostics should explain the missing frames command"
    );
}

#[test]
fn face_mdl_compiles_runs_and_saves() {
    let file = "tests/face.mdl";
    let output = std::env::temp_dir().join(format!("gartus-face-{}.ppm", std::process::id()));
    let _ = std::fs::remove_file(&output);

    let compiled = compile_file(file).expect("face.mdl compiles");
    assert_eq!(compiled.animation().frames(), 1);
    assert!(
        compiled
            .commands()
            .iter()
            .any(|command| matches!(command.node, Command::Render(_) | Command::Shape(_))),
        "face.mdl should contain render and shape commands"
    );

    let frames = run_file(
        file,
        RenderConfig::new_with_bg(500, 500, Rgb::BLACK, Rgb::WHITE)
            .wrapped(false)
            .display_enabled(false)
            .save_override(&output),
    )
    .expect("face.mdl runs");

    assert_eq!(frames.len(), 1);
    assert!(
        output.exists(),
        "face.mdl should save the overridden frame output"
    );
    assert!(
        frames[0]
            .canvas()
            .pixels()
            .iter()
            .any(|pixel| *pixel != Rgb::WHITE),
        "face.mdl should draw visible pixels"
    );
    let manual = manual_face_canvas();
    assert!(
        pixels_eq(frames[0].canvas(), &manual),
        "face.mdl parser output should match manually built face scene"
    );

    let _ = std::fs::remove_file(&output);
}

#[test]
fn simple_anim_mdl_compiles_and_executes_sample_frames() {
    let file = "tests/simple_anim.mdl";
    let compiled = compile_file(file).expect("simple_anim.mdl compiles");

    assert_eq!(compiled.animation().basename(), "simple_50");
    assert_eq!(compiled.animation().frames(), 50);
    assert_eq!(
        compiled
            .animation()
            .knobs_for_frame(0)
            .unwrap()
            .get("spinny"),
        Some(&0.0)
    );
    assert_eq!(
        compiled
            .animation()
            .knobs_for_frame(49)
            .unwrap()
            .get("spinny"),
        Some(&1.0)
    );
    assert_eq!(
        compiled
            .animation()
            .knobs_for_frame(0)
            .unwrap()
            .get("bigenator"),
        Some(&0.0)
    );
    assert_eq!(
        compiled
            .animation()
            .knobs_for_frame(24)
            .unwrap()
            .get("bigenator"),
        Some(&1.0)
    );
    assert_eq!(
        compiled
            .animation()
            .knobs_for_frame(25)
            .unwrap()
            .get("bigenator"),
        Some(&1.0)
    );
    assert_eq!(
        compiled
            .animation()
            .knobs_for_frame(49)
            .unwrap()
            .get("bigenator"),
        Some(&0.0)
    );

    for frame in [0, 24, 49] {
        execute_compiled_frame(
            &compiled,
            &RenderConfig::new_with_bg(500, 500, Rgb::BLACK, Rgb::WHITE)
                .wrapped(false)
                .display_enabled(false)
                .save_enabled(false),
            frame,
        )
        .unwrap_or_else(|error| panic!("simple_anim.mdl frame {frame} should execute: {error}"));
    }
}

#[cfg(feature = "external")]
#[test]
fn mdl_mesh_uses_obj_mtl_diffuse_colors() {
    let dir = std::env::temp_dir().join(format!("gartus-mtl-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temp dir");
    std::fs::write(
        dir.join("tri.mtl"),
        b"newmtl red\nKd 1 0 0\nnewmtl green\nKd 0 1 0\n",
    )
    .expect("write mtl");
    std::fs::write(
        dir.join("tri.obj"),
        b"mtllib tri.mtl\n\
v 2 2 0\n\
v 8 2 0\n\
v 2 8 0\n\
v 12 2 0\n\
v 18 2 0\n\
v 12 8 0\n\
usemtl red\n\
f 1 2 3\n\
usemtl green\n\
f 4 5 6\n",
    )
    .expect("write obj");

    let source = "constants mat 0 1 0 0 1 0 0 1 0 255 255 255\nmesh mat :tri.obj\n";
    let frames = run_source(
        source,
        Some(&dir),
        RenderConfig::new_with_bg(24, 24, Rgb::BLACK, Rgb::WHITE)
            .display_enabled(false)
            .save_enabled(false),
    )
    .expect("run material mesh");

    let pixels = frames[0].canvas().pixels();
    assert!(
        pixels
            .iter()
            .any(|pixel| pixel.red > pixel.green.saturating_add(20) && pixel.red > pixel.blue),
        "material mesh should draw red diffuse pixels"
    );
    assert!(
        pixels
            .iter()
            .any(|pixel| pixel.green > pixel.red.saturating_add(20) && pixel.green > pixel.blue),
        "material mesh should draw green diffuse pixels"
    );

    let _ = std::fs::remove_dir_all(dir);
}
