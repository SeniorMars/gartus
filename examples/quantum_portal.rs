//! This example uses `gartus`'s raster rendering engine with Toon cell-shading
//! and Phong reflection. It renders:
//! 1. A space-time warping gravity grid in the background that pulses over time.
//! 2. A counter-rotating core made of nested Platonic solids (icosahedron and dodecahedron)
//!    surrounding a pulsing singularity core.
//! 3. Concentric gear-like astrolabe rings with radial clockwork teeth/glyphs.
//! 4. Orbiting quantum crystals that trace glowing particle trails.
//! 5. Moving colored point lights that sweep across the scene.
//!
//! Outputs a GIF and a PNG preview to `final/`.

use gartus::gmath::vector::Vector;
use gartus::prelude::*;
use std::{error::Error, f64::consts::PI, fs};

const WIDTH: u32 = 1000;
const HEIGHT: u32 = 1000;
const FRAMES: usize = 96;
const CENTER_X: f64 = WIDTH as f64 * 0.5;
const CENTER_Y: f64 = HEIGHT as f64 * 0.5;

struct Assets {
    sphere: PolygonMatrix,
    torus_inner: PolygonMatrix,
    torus_middle: PolygonMatrix,
    torus_outer: PolygonMatrix,
    icosahedron: PolygonMatrix,
    dodecahedron: PolygonMatrix,
    cone: PolygonMatrix,
    pyramid: PolygonMatrix,
    crystal: PolygonMatrix,
}

#[derive(Clone, Copy)]
struct Material {
    line: Rgb,
    ambient: ReflectionConstants,
    diffuse: ReflectionConstants,
    specular: ReflectionConstants,
    specular_exponent: u32,
}

fn main() {
    if let Err(err) = render() {
        eprintln!("could not render quantum portal:\n{err}");
        std::process::exit(1);
    }
}

fn render() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final")?;
    let assets = build_assets();

    let options = AnimationRenderOptions::new(
        "anim",
        "quantum-portal-",
        FRAMES,
        "final/quantum_portal.gif",
    )
    .delay_cs(3)
    .preview(24, "final/quantum_portal.png")
    .unique_frame_dir(true);

    println!("Rendering {} frames...", FRAMES);

    FrameRecorder::render_gif_auto(options, |frame| Ok(render_frame(frame, &assets)))?;

    println!("Saved final/quantum_portal.png and final/quantum_portal.gif");
    Ok(())
}

// ── assets ────────────────────────────────────────────────────────────────────

fn build_assets() -> Assets {
    let mut sphere = PolygonMatrix::new();
    sphere.add_sphere((0.0, 0.0, 0.0), 1.0, 24);

    let mut torus_inner = PolygonMatrix::new();
    torus_inner.add_torus((0.0, 0.0, 0.0), 10.0, 160.0, 48);

    let mut torus_middle = PolygonMatrix::new();
    torus_middle.add_torus((0.0, 0.0, 0.0), 12.0, 250.0, 56);

    let mut torus_outer = PolygonMatrix::new();
    torus_outer.add_torus((0.0, 0.0, 0.0), 14.0, 340.0, 64);

    let mut icosahedron = PolygonMatrix::new();
    icosahedron.add_icosahedron((0.0, 0.0, 0.0), 1.0);

    let mut dodecahedron = PolygonMatrix::new();
    dodecahedron.add_dodecahedron((0.0, 0.0, 0.0), 1.0);

    let mut cone = PolygonMatrix::new();
    cone.add_cone((0.0, 0.0, 0.0), 1.0, 2.5, 12);

    let mut pyramid = PolygonMatrix::new();
    pyramid.add_pyramid((0.0, 0.0, 0.0), 1.0, 1.8);

    let mut crystal = PolygonMatrix::new();
    crystal.add_crystal((0.0, 0.0, 0.0), 4, 1.0, 3.2);

    Assets {
        sphere,
        torus_inner,
        torus_middle,
        torus_outer,
        icosahedron,
        dodecahedron,
        cone,
        pyramid,
        crystal,
    }
}

// ── rendering ─────────────────────────────────────────────────────────────────

fn render_frame(frame: usize, assets: &Assets) -> Canvas {
    let mut canvas = Canvas::new_with_bg(WIDTH, HEIGHT, Rgb::new(4, 5, 12));
    canvas.set_wrapped(false);
    canvas.set_line_width(1.0);

    // Use stylized Toon cell-shading with Phong specular reflection
    canvas.set_shading_mode(ShadingMode::Toon);
    canvas.set_polygon_color_mode(PolygonColorMode::PhongReflection);

    let t = frame as f64 / FRAMES as f64;

    // 1. Background elements
    draw_warped_grid(&mut canvas, t);
    draw_stars(&mut canvas, t);

    // 2. Set moving lighting
    canvas.set_lighting(scene_lighting(t));

    // 3. Render 3D objects
    draw_core(&mut canvas, assets, t);
    draw_rings(&mut canvas, assets, t);
    draw_crystals(&mut canvas, assets, t);

    canvas
}

fn scene_lighting(t: f64) -> Lighting {
    let pulse = 0.5 + 0.5 * (t * PI * 2.0).sin();

    // Central pulsing light
    let center_light_pos = Vector::new(CENTER_X, CENTER_Y, 120.0);
    let center_light_color = Rgb::new(
        255,
        (160.0 + 80.0 * pulse) as u8,
        (50.0 + 80.0 * (t * PI * 2.0 + PI / 3.0).cos()) as u8,
    );

    // Orbiting key light
    let side_light_pos = Vector::new(
        CENTER_X + 450.0 * (t * PI * 2.0).cos(),
        CENTER_Y + 450.0 * (t * PI * 2.0).sin(),
        320.0,
    );
    let side_light_color = Rgb::new(50, 180, 255);

    // Dim directional fill
    let dir_light = PointLight::directional(Vector::new(1.0, 1.0, 1.0), Rgb::new(30, 20, 50));

    Lighting {
        ambient: Rgb::new(6, 6, 12),
        point_lights: vec![
            PointLight::positional(center_light_pos, center_light_color)
                .with_inverse_linear_attenuation(900.0),
            PointLight::positional(side_light_pos, side_light_color)
                .with_inverse_linear_attenuation(1100.0),
            dir_light,
        ],
        ambient_reflection: ReflectionConstants::new(0.08, 0.08, 0.12),
        diffuse_reflection: ReflectionConstants::new(0.80, 0.75, 0.85),
        specular_reflection: ReflectionConstants::new(0.35, 0.35, 0.35),
        specular_exponent: 16,
        ..Lighting::default()
    }
}

// ── background ────────────────────────────────────────────────────────────────

fn warp_point(x: f64, y: f64, t: f64) -> (f64, f64) {
    let dx = x - CENTER_X;
    let dy = y - CENTER_Y;
    let r = (dx * dx + dy * dy).sqrt();
    if r < 10.0 {
        return (CENTER_X, CENTER_Y);
    }
    // Pulsing gravitational vortex warp
    let gravity = 16000.0 * (1.0 + 0.18 * (t * PI * 2.0).sin());
    let factor = 1.0 - gravity / (r * r + 2500.0);
    let factor = factor.clamp(0.0, 1.6);
    (CENTER_X + dx * factor, CENTER_Y + dy * factor)
}

fn grid_color_at_distance(dist: f64) -> Rgb {
    if dist < 85.0 {
        Rgb::new(0, 0, 0)
    } else if dist < 220.0 {
        let factor = (dist - 85.0) / 135.0;
        Rgb::new(
            (255.0 * factor) as u8,
            (20.0 * (1.0 - factor)) as u8,
            (128.0 + 127.0 * factor) as u8,
        )
    } else if dist < 450.0 {
        let factor = (dist - 220.0) / 230.0;
        Rgb::new((255.0 * (1.0 - factor)) as u8, (180.0 * factor) as u8, 255)
    } else {
        let factor = ((dist - 450.0) / 350.0).min(1.0);
        Rgb::new(
            0,
            (180.0 * (1.0 - factor)) as u8,
            (255.0 * (1.0 - 0.75 * factor)) as u8,
        )
    }
}

fn draw_warped_grid(canvas: &mut Canvas, t: f64) {
    let steps = 40;
    let grid_spacing = WIDTH as f64 / steps as f64;

    // Vertical-ish grid lines
    for i in 1..steps {
        let x = i as f64 * grid_spacing;
        let mut prev_p = warp_point(x, 0.0, t);
        for j in 1..=80 {
            let y = j as f64 * (HEIGHT as f64 / 80.0);
            let next_p = warp_point(x, y, t);

            let avg_x = 0.5 * (prev_p.0 + next_p.0);
            let avg_y = 0.5 * (prev_p.1 + next_p.1);
            let dist = ((avg_x - CENTER_X).powi(2) + (avg_y - CENTER_Y).powi(2)).sqrt();
            let color = grid_color_at_distance(dist);

            canvas.draw_line(color, prev_p.0, prev_p.1, next_p.0, next_p.1);
            prev_p = next_p;
        }
    }

    // Horizontal-ish grid lines
    for j in 1..steps {
        let y = j as f64 * grid_spacing;
        let mut prev_p = warp_point(0.0, y, t);
        for i in 1..=80 {
            let x = i as f64 * (WIDTH as f64 / 80.0);
            let next_p = warp_point(x, y, t);

            let avg_x = 0.5 * (prev_p.0 + next_p.0);
            let avg_y = 0.5 * (prev_p.1 + next_p.1);
            let dist = ((avg_x - CENTER_X).powi(2) + (avg_y - CENTER_Y).powi(2)).sqrt();
            let color = grid_color_at_distance(dist);

            canvas.draw_line(color, prev_p.0, prev_p.1, next_p.0, next_p.1);
            prev_p = next_p;
        }
    }
}

fn draw_stars(canvas: &mut Canvas, t: f64) {
    let mut seed = 0x1234_5678_9abc_def0_u64;
    let lcg = |s: &mut u64| {
        *s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *s
    };

    for i in 0..150 {
        let x_seed = lcg(&mut seed);
        let y_seed = lcg(&mut seed);
        let z_seed = lcg(&mut seed);

        let x = (x_seed % WIDTH as u64) as i64;
        let y = (y_seed % HEIGHT as u64) as i64;
        let z = -600.0 - (z_seed % 400) as f64;

        let twinkle = ((i as f64 * 1.7 + t * PI * 2.0).sin() + 1.0) * 0.5;
        if twinkle > 0.38 {
            let bright = (110.0 + 145.0 * twinkle) as u8;
            let color = if i % 7 == 0 {
                Rgb::new(bright, (bright as f64 * 0.6) as u8, bright)
            } else if i % 11 == 0 {
                Rgb::new((bright as f64 * 0.5) as u8, bright, bright)
            } else {
                Rgb::new(bright, bright, bright)
            };
            canvas.plot_z(&color, x, y, z);
        }
    }
}

// ── 3D meshes ─────────────────────────────────────────────────────────────────

fn draw_core(canvas: &mut Canvas, assets: &Assets, t: f64) {
    let base = Matrix::translate(CENTER_X, CENTER_Y, 0.0);

    // Pulsing white singularity
    let sing_mat = Material {
        line: Rgb::new(255, 240, 200),
        ambient: ReflectionConstants::new(0.9, 0.9, 0.7),
        diffuse: ReflectionConstants::new(0.95, 0.95, 0.85),
        specular: ReflectionConstants::new(0.6, 0.6, 0.6),
        specular_exponent: 32,
    };
    let sing_scale = 32.0 + 4.0 * (t * PI * 4.0).sin();
    let sing_transform = base.clone() * Matrix::scale(sing_scale, sing_scale, sing_scale);
    draw_lit_mesh(canvas, &assets.sphere, &sing_transform, sing_mat);

    // Counter-rotating magenta dodecahedron
    let dodec_mat = Material {
        line: Rgb::new(255, 60, 170),
        ambient: ReflectionConstants::new(0.15, 0.02, 0.08),
        diffuse: ReflectionConstants::new(0.80, 0.10, 0.45),
        specular: ReflectionConstants::new(0.45, 0.10, 0.35),
        specular_exponent: 12,
    };
    let dodec_scale = 68.0;
    let dodec_transform = base.clone()
        * Matrix::rotate_y(t * 360.0 * 2.2)
        * Matrix::rotate_x(t * 360.0 * 0.9)
        * Matrix::scale(dodec_scale, dodec_scale, dodec_scale);
    draw_lit_mesh(canvas, &assets.dodecahedron, &dodec_transform, dodec_mat);

    // Counter-rotating cyan icosahedron
    let ico_mat = Material {
        line: Rgb::new(60, 245, 230),
        ambient: ReflectionConstants::new(0.02, 0.15, 0.12),
        diffuse: ReflectionConstants::new(0.10, 0.55, 0.70),
        specular: ReflectionConstants::new(0.10, 0.45, 0.50),
        specular_exponent: 16,
    };
    let ico_scale = 104.0;
    let ico_transform = base.clone()
        * Matrix::rotate_y(-t * 360.0 * 1.5)
        * Matrix::rotate_z(t * 360.0 * 1.1)
        * Matrix::scale(ico_scale, ico_scale, ico_scale);
    draw_lit_mesh(canvas, &assets.icosahedron, &ico_transform, ico_mat);
}

fn draw_rings(canvas: &mut Canvas, assets: &Assets, t: f64) {
    let base = Matrix::translate(CENTER_X, CENTER_Y, 0.0);

    // ─────────────────────────────────────────────────────────────────────────
    // 1. Inner Ring
    // ─────────────────────────────────────────────────────────────────────────
    let inner_mat = Material {
        line: Rgb::new(0, 225, 255),
        ambient: ReflectionConstants::new(0.01, 0.12, 0.15),
        diffuse: ReflectionConstants::new(0.05, 0.55, 0.70),
        specular: ReflectionConstants::new(0.10, 0.40, 0.50),
        specular_exponent: 24,
    };

    let inner_ring_tilt = Matrix::rotate_x(35.0) * Matrix::rotate_y(-25.0);
    let inner_spin = Matrix::rotate_z(t * 360.0);
    let inner_transform = base.clone() * inner_ring_tilt * inner_spin;

    draw_lit_mesh(canvas, &assets.torus_inner, &inner_transform, inner_mat);

    // 6 cones pointing outward on the inner ring
    let cone_count = 6;
    let cone_mat = Material {
        line: Rgb::new(255, 225, 90),
        ambient: ReflectionConstants::new(0.12, 0.10, 0.02),
        diffuse: ReflectionConstants::new(0.75, 0.62, 0.10),
        specular: ReflectionConstants::new(0.45, 0.38, 0.10),
        specular_exponent: 16,
    };
    for i in 0..cone_count {
        let angle = i as f64 / cone_count as f64 * PI * 2.0;
        let radius = 160.0;
        let tooth_loc = Matrix::translate(radius * angle.cos(), radius * angle.sin(), 0.0)
            * Matrix::rotate_z(angle.to_degrees())
            * Matrix::rotate_y(90.0)
            * Matrix::scale(12.0, 12.0, 12.0);

        let tooth_transform = inner_transform.clone() * tooth_loc;
        draw_lit_mesh(canvas, &assets.cone, &tooth_transform, cone_mat);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2. Middle Ring
    // ─────────────────────────────────────────────────────────────────────────
    let middle_mat = Material {
        line: Rgb::new(225, 60, 255),
        ambient: ReflectionConstants::new(0.12, 0.02, 0.15),
        diffuse: ReflectionConstants::new(0.55, 0.10, 0.70),
        specular: ReflectionConstants::new(0.40, 0.10, 0.50),
        specular_exponent: 20,
    };

    let middle_ring_tilt = Matrix::rotate_x(-45.0) * Matrix::rotate_z(30.0);
    let middle_spin = Matrix::rotate_z(-t * 360.0 * 0.7);
    let middle_transform = base.clone() * middle_ring_tilt * middle_spin;

    draw_lit_mesh(canvas, &assets.torus_middle, &middle_transform, middle_mat);

    // 8 spheres sitting on the middle ring
    let sphere_count = 8;
    let sphere_mat = Material {
        line: Rgb::new(255, 120, 50),
        ambient: ReflectionConstants::new(0.15, 0.05, 0.02),
        diffuse: ReflectionConstants::new(0.85, 0.38, 0.10),
        specular: ReflectionConstants::new(0.35, 0.15, 0.05),
        specular_exponent: 10,
    };
    for i in 0..sphere_count {
        let angle = i as f64 / sphere_count as f64 * PI * 2.0;
        let radius = 250.0;
        let tooth_loc = Matrix::translate(radius * angle.cos(), radius * angle.sin(), 0.0)
            * Matrix::scale(16.0, 16.0, 16.0);

        let tooth_transform = middle_transform.clone() * tooth_loc;
        draw_lit_mesh(canvas, &assets.sphere, &tooth_transform, sphere_mat);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 3. Outer Ring
    // ─────────────────────────────────────────────────────────────────────────
    let outer_mat = Material {
        line: Rgb::new(255, 180, 50),
        ambient: ReflectionConstants::new(0.15, 0.10, 0.02),
        diffuse: ReflectionConstants::new(0.70, 0.52, 0.10),
        specular: ReflectionConstants::new(0.50, 0.40, 0.15),
        specular_exponent: 32,
    };

    let outer_ring_tilt = Matrix::rotate_y(55.0) * Matrix::rotate_x(15.0);
    let outer_spin = Matrix::rotate_z(t * 360.0 * 0.4);
    let outer_transform = base.clone() * outer_ring_tilt * outer_spin;

    draw_lit_mesh(canvas, &assets.torus_outer, &outer_transform, outer_mat);

    // 12 pyramids pointing radially inward on the outer ring
    let pyr_count = 12;
    let pyr_mat = Material {
        line: Rgb::new(80, 255, 175),
        ambient: ReflectionConstants::new(0.02, 0.12, 0.08),
        diffuse: ReflectionConstants::new(0.10, 0.65, 0.40),
        specular: ReflectionConstants::new(0.10, 0.40, 0.30),
        specular_exponent: 16,
    };
    for i in 0..pyr_count {
        let angle = i as f64 / pyr_count as f64 * PI * 2.0;
        let radius = 340.0;
        let tooth_loc = Matrix::translate(radius * angle.cos(), radius * angle.sin(), 0.0)
            * Matrix::rotate_z(angle.to_degrees() + 90.0)
            * Matrix::scale(15.0, 22.0, 15.0);

        let tooth_transform = outer_transform.clone() * tooth_loc;
        draw_lit_mesh(canvas, &assets.pyramid, &tooth_transform, pyr_mat);
    }
}

fn get_crystal_position(
    t: f64,
    orbit_r: f64,
    speed: f64,
    phase: f64,
    inc_x: f64,
    inc_z: f64,
) -> Vector {
    let angle = t * PI * 2.0 * speed + phase;
    let local_pos = Vector::new(orbit_r * angle.cos(), 0.0, orbit_r * angle.sin());
    let transform = Matrix::rotate_z(inc_z) * Matrix::rotate_x(inc_x);
    transform.mult_vector(local_pos)
}

fn draw_crystals(canvas: &mut Canvas, assets: &Assets, t: f64) {
    let base = Matrix::translate(CENTER_X, CENTER_Y, 0.0);

    for c_idx in 0..4 {
        let orbit_r = 130.0 + c_idx as f64 * 60.0;
        let speed = 1.6 - c_idx as f64 * 0.35;
        let phase = c_idx as f64 * 1.57;
        let inc_x = 20.0 + c_idx as f64 * 15.0;
        let inc_z = -15.0 - c_idx as f64 * 20.0;
        let c_scale = 10.0 + c_idx as f64 * 3.0;

        let (c_color, line_color) = match c_idx {
            0 => (Rgb::new(255, 60, 100), Rgb::new(255, 120, 160)),
            1 => (Rgb::new(50, 255, 120), Rgb::new(140, 255, 180)),
            2 => (Rgb::new(0, 150, 255), Rgb::new(100, 200, 255)),
            _ => (Rgb::new(255, 200, 50), Rgb::new(255, 230, 140)),
        };

        let c_mat = Material {
            line: line_color,
            ambient: ReflectionConstants::new(
                c_color.red as f64 / 255.0 * 0.15,
                c_color.green as f64 / 255.0 * 0.15,
                c_color.blue as f64 / 255.0 * 0.15,
            ),
            diffuse: ReflectionConstants::new(
                c_color.red as f64 / 255.0 * 0.75,
                c_color.green as f64 / 255.0 * 0.75,
                c_color.blue as f64 / 255.0 * 0.75,
            ),
            specular: ReflectionConstants::new(0.4, 0.4, 0.4),
            specular_exponent: 16,
        };

        // A. Draw trails
        let trail_steps = 8;
        for step in 1..=trail_steps {
            let t_trail = (t - step as f64 * 0.005 + 1.0) % 1.0;
            let pos = get_crystal_position(t_trail, orbit_r, speed, phase, inc_x, inc_z);

            let trail_scale = c_scale * (1.0 - step as f64 / (trail_steps + 1) as f64) * 0.5;
            let fade = 1.0 - step as f64 / trail_steps as f64;

            let trail_col = Rgb::new(
                (c_color.red as f64 * fade * 0.7) as u8,
                (c_color.green as f64 * fade * 0.7) as u8,
                (c_color.blue as f64 * fade * 0.7) as u8,
            );

            let trail_mat = Material {
                line: trail_col,
                ambient: ReflectionConstants::new(0.01, 0.01, 0.01),
                diffuse: ReflectionConstants::new(
                    trail_col.red as f64 / 255.0 * 0.5,
                    trail_col.green as f64 / 255.0 * 0.5,
                    trail_col.blue as f64 / 255.0 * 0.5,
                ),
                specular: ReflectionConstants::new(0.0, 0.0, 0.0),
                specular_exponent: 2,
            };

            let trail_transform = base.clone()
                * Matrix::translate(pos.x(), pos.y(), pos.z())
                * Matrix::scale(trail_scale, trail_scale, trail_scale);

            draw_lit_mesh(canvas, &assets.sphere, &trail_transform, trail_mat);
        }

        // B. Draw main crystal
        let pos = get_crystal_position(t, orbit_r, speed, phase, inc_x, inc_z);
        let spin_y = t * 360.0 * 2.0 + c_idx as f64 * 45.0;
        let spin_x = t * 360.0 * 0.5;

        let c_transform = base.clone()
            * Matrix::translate(pos.x(), pos.y(), pos.z())
            * Matrix::rotate_y(spin_y)
            * Matrix::rotate_x(spin_x)
            * Matrix::scale(c_scale, c_scale, c_scale);

        draw_lit_mesh(canvas, &assets.crystal, &c_transform, c_mat);
    }
}

fn draw_lit_mesh(
    canvas: &mut Canvas,
    mesh: &PolygonMatrix,
    transform: &Matrix,
    material: Material,
) {
    canvas.set_line_pixel(material.line);
    {
        let l = canvas.lighting_mut();
        l.ambient_reflection = material.ambient;
        l.diffuse_reflection = material.diffuse;
        l.specular_reflection = material.specular;
        l.specular_exponent = material.specular_exponent;
    }
    canvas.draw_polygons(&mesh.apply(transform));
}
