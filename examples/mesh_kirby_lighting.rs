use gartus::{external, gmath::vector::Vector, prelude::*};
use std::{error::Error, f64::consts::PI};

const WIDTH: u32 = 960;
const HEIGHT: u32 = 960;
const CELL: f64 = 480.0;
const MODEL_SIZE: f64 = 330.0;
const FRAMES: usize = 48;
const MODEL_PATH: &str = "examples/data/meshes/scotty/dtorresr.obj";
const OUTPUT_GIF: &str = "final/mesh_kirby_lighting.gif";
const OUTPUT_PREVIEW: &str = "final/mesh_kirby_lighting.png";
const PANELS: [Panel; 4] = [
    Panel {
        label: "FLAT",
        shading: ShadingMode::Flat,
        center: (CELL * 0.5, CELL * 1.5),
    },
    Panel {
        label: "GOURAUD",
        shading: ShadingMode::Gouraud,
        center: (CELL * 1.5, CELL * 1.5),
    },
    Panel {
        label: "PHONG",
        shading: ShadingMode::Phong,
        center: (CELL * 0.5, CELL * 0.5),
    },
    Panel {
        label: "TOON",
        shading: ShadingMode::Toon,
        center: (CELL * 1.5, CELL * 0.5),
    },
];

struct Scene {
    normalized_mesh: PolygonMatrix,
}

#[derive(Clone, Copy)]
struct Panel {
    label: &'static str,
    shading: ShadingMode,
    center: (f64, f64),
}

fn main() {
    if let Err(err) = render() {
        eprintln!("could not render Kirby lighting animation:\n{err}");
        std::process::exit(1);
    }
}

fn render() -> Result<(), Box<dyn Error>> {
    println!("Loading Kirby mesh from {MODEL_PATH}...");
    let mesh = external::meshify(MODEL_PATH)?;
    let triangle_count = mesh.triangle_count();
    let normalize = external::normalize_mesh_transform(&mesh, MODEL_SIZE, external::MeshUpAxis::Y);
    let scene = Scene {
        normalized_mesh: mesh.apply(&normalize),
    };

    let options = AnimationRenderOptions::new(
        "final/mesh_kirby_lighting_frames",
        "kirby-lighting-",
        FRAMES,
        OUTPUT_GIF,
    )
    .delay_cs(4)
    .preview(FRAMES / 4, OUTPUT_PREVIEW)
    .unique_frame_dir(true);

    FrameRecorder::render_gif_auto(options, |frame| Ok(render_frame(&scene, frame)))?;

    println!("Loaded {triangle_count} triangles.");
    println!("Saved preview to {OUTPUT_PREVIEW}");
    println!("Saved animation to {OUTPUT_GIF}");
    Ok(())
}

fn render_frame(scene: &Scene, frame: usize) -> Canvas {
    let mut canvas = Canvas::new_with_bg(WIDTH, HEIGHT, Rgb::new(12, 13, 19));
    canvas.set_wrapped(false);
    canvas.set_polygon_color_mode(PolygonColorMode::PhongReflection);

    draw_backdrop(&mut canvas);

    let phase = frame as f64 / FRAMES as f64 * PI * 2.0;
    let light = light_for_phase(phase);
    let posed_mesh = pose_mesh(scene, phase);

    for panel in PANELS {
        draw_panel(&mut canvas, &posed_mesh, light, panel, phase);
    }

    draw_labels(&mut canvas);
    canvas
}

fn pose_mesh(scene: &Scene, phase: f64) -> PolygonMatrix {
    let spin = phase.to_degrees();
    let transform = Matrix::rotate_x(-12.0) * Matrix::rotate_y(spin);
    scene.normalized_mesh.apply(&transform)
}

fn light_for_phase(phase: f64) -> Vector {
    Vector::new(
        phase.cos() * 1.15,
        0.72 + (phase * 1.7).sin() * 0.28,
        1.0 + phase.sin() * 0.45,
    )
}

fn draw_panel(
    canvas: &mut Canvas,
    posed_mesh: &PolygonMatrix,
    light: Vector,
    panel: Panel,
    phase: f64,
) {
    canvas.set_shading_mode(panel.shading);
    canvas.set_lighting(kirby_lighting(light, panel.shading));

    let transformed = translate_mesh(posed_mesh, panel.center.0, panel.center.1 - 8.0, 0.0);

    canvas.set_line_pixel(Rgb::new(247, 116, 156));
    canvas.draw_polygons(&transformed);

    draw_light_marker(canvas, panel, phase);
}

fn translate_mesh(mesh: &PolygonMatrix, dx: f64, dy: f64, dz: f64) -> PolygonMatrix {
    let mut translated = PolygonMatrix::with_capacity(mesh.cols());
    for (p0, p1, p2) in mesh.iter_triangles() {
        translated.add_polygon(
            (p0[0] + dx, p0[1] + dy, p0[2] + dz),
            (p1[0] + dx, p1[1] + dy, p1[2] + dz),
            (p2[0] + dx, p2[1] + dy, p2[2] + dz),
        );
    }
    translated
}

fn kirby_lighting(light: Vector, mode: ShadingMode) -> Lighting {
    let material = PhongMaterial::RUBY;
    let specular_exponent = match mode {
        ShadingMode::Toon => 10,
        ShadingMode::Phong => 18,
        ShadingMode::Gouraud => 14,
        ShadingMode::Flat | ShadingMode::Wireframe => 8,
    };

    Lighting {
        ambient: Rgb::new(58, 48, 64),
        point_light: PointLight::new(light, Rgb::new(255, 248, 235)),
        ambient_reflection: material.ambient,
        diffuse_reflection: material.diffuse,
        specular_reflection: material.specular,
        specular_exponent,
        ..Lighting::default()
    }
}

fn draw_backdrop(canvas: &mut Canvas) {
    let divider = Rgb::new(49, 55, 71);
    canvas.set_line_width(2.0);
    canvas.draw_line(divider, CELL, 34.0, CELL, HEIGHT as f64 - 34.0);
    canvas.draw_line(divider, 34.0, CELL, WIDTH as f64 - 34.0, CELL);
    canvas.set_line_width(1.0);

    for panel in PANELS {
        draw_panel_frame(canvas, panel);
    }
}

fn draw_panel_frame(canvas: &mut Canvas, panel: Panel) {
    let (cx, cy) = panel.center;
    let left = cx - CELL * 0.47;
    let right = cx + CELL * 0.47;
    let bottom = cy - CELL * 0.47;
    let top = cy + CELL * 0.47;
    canvas.draw_rect(left, bottom, right, top, Rgb::new(33, 38, 52));
}

fn draw_light_marker(canvas: &mut Canvas, panel: Panel, phase: f64) {
    let x = panel.center.0 + phase.cos() * 150.0;
    let y = panel.center.1 + 164.0 + phase.sin() * 22.0;
    let color = Rgb::new(255, 243, 168);
    canvas.fill_disc(x.round() as i64, y.round() as i64, 8, color);
    canvas.fill_disc(
        x.round() as i64,
        y.round() as i64,
        3,
        Rgb::new(255, 255, 245),
    );
}

fn draw_labels(canvas: &mut Canvas) {
    for panel in PANELS {
        canvas.draw_text_centered(
            panel.label,
            panel.center.0.round() as i64,
            (panel.center.1 - 196.0).round() as i64,
            4,
            Rgb::new(225, 228, 238),
        );
    }
}
