//! Celestial Dragon: a toon-shaded sky dragon swallowing a little sun under
//! aurora light.
//!
//! This example uses gartus' raster engine, z-buffered polygon meshes, Phong
//! materials, and `ShadingMode::Toon` for the dragon and sun. The aurora,
//! clouds, stars, and glyph-like scale sparks are procedural 2D raster passes.

use gartus::{gmath::procedural::TAU, prelude::*};
use std::{error::Error, f64::consts::PI, fs};

const WIDTH: u32 = 960;
const HEIGHT: u32 = 640;
const FRAMES: usize = 64;
const SUN_X: f64 = 694.0;
const SUN_Y: f64 = 285.0;
const HEAD_X: f64 = 640.0;
const HEAD_Y: f64 = 286.0;

#[derive(Clone)]
struct Assets {
    sphere: PolygonMatrix,
    crystal: PolygonMatrix,
    stars: Vec<Star>,
}

#[derive(Clone, Copy)]
struct Star {
    x: f64,
    y: f64,
    phase: f64,
    hue: f64,
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
        eprintln!("could not render celestial dragon:\n{err}");
        std::process::exit(1);
    }
}

fn render() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final")?;
    let assets = build_assets();
    let options = AnimationRenderOptions::new(
        "anim",
        "celestial-dragon-",
        FRAMES,
        "final/celestial_dragon.gif",
    )
    .delay_cs(3)
    .preview(18, "final/celestial_dragon.png")
    .unique_frame_dir(true);

    FrameRecorder::render_gif_auto(options, |frame| Ok(render_frame(frame, &assets)))?;

    println!("Saved final/celestial_dragon.png and final/celestial_dragon.gif");
    Ok(())
}

fn build_assets() -> Assets {
    let mut sphere = PolygonMatrix::new();
    sphere.add_sphere((0.0, 0.0, 0.0), 1.0, 28);

    let mut crystal = PolygonMatrix::new();
    crystal.add_crystal((0.0, 0.0, 0.0), 5, 1.0, 2.2);

    Assets {
        sphere,
        crystal,
        stars: build_stars(),
    }
}

fn build_stars() -> Vec<Star> {
    let mut stars = Vec::new();
    let mut seed = 0xCADA_551A_57ED_5EED_u64;
    for i in 0..230 {
        seed = lcg(seed);
        let x = 18.0 + (seed % u64::from(WIDTH - 36)) as f64;
        seed = lcg(seed);
        let y = 18.0 + (seed % u64::from(HEIGHT - 120)) as f64;
        seed = lcg(seed);
        let phase = (seed % 10_000) as f64 / 10_000.0 * TAU;
        let hue = if i % 17 == 0 {
            48.0
        } else if i % 11 == 0 {
            185.0
        } else if i % 7 == 0 {
            318.0
        } else {
            220.0
        };
        stars.push(Star { x, y, phase, hue });
    }
    stars
}

fn render_frame(frame: usize, assets: &Assets) -> Canvas {
    let t = frame as f64 / FRAMES as f64;
    let breath = (t * TAU).sin();
    let mut canvas = sky(t);
    canvas.wrapped = false;
    canvas.upper_left_origin = true;
    canvas.set_line_width(1.0);
    canvas.set_shading_mode(ShadingMode::Toon);
    canvas.set_polygon_color_mode(PolygonColorMode::PhongReflection);
    canvas.set_lighting(scene_lighting(t));

    draw_stars(&mut canvas, &assets.stars, t);
    draw_aurora(&mut canvas, t);
    draw_cloud_sea(&mut canvas, t);
    draw_tail_wisps(&mut canvas, t);
    draw_aurora_wings(&mut canvas, t);
    draw_dragon_body(&mut canvas, assets, t);
    draw_tail_tip(&mut canvas, t);
    draw_neck_bridge(&mut canvas, t);
    draw_nebula_mane(&mut canvas, t);
    draw_sun_glow(&mut canvas, t);
    draw_head_base(&mut canvas, assets, t);
    draw_mouth_shadow(&mut canvas);
    draw_little_sun(&mut canvas, assets, t);
    draw_jaws_and_horns(&mut canvas, assets, t);
    draw_scale_sparks(&mut canvas, t);
    draw_face_lines(&mut canvas, breath);
    draw_foreground_aurora(&mut canvas, t);

    canvas
}

fn scene_lighting(t: f64) -> Lighting {
    let pulse = 0.5 + 0.5 * (t * TAU).sin();
    Lighting {
        ambient: Rgb::new(7, 8, 18),
        point_lights: vec![
            PointLight::positional(
                Vector::new(SUN_X, SUN_Y, 260.0),
                Rgb::new(255, (205.0 + 35.0 * pulse) as u8, 90),
            )
            .with_inverse_linear_attenuation(720.0),
            PointLight::positional(Vector::new(360.0, 84.0, 340.0), Rgb::new(80, 235, 218))
                .with_inverse_linear_attenuation(900.0),
            PointLight::positional(Vector::new(830.0, 92.0, 310.0), Rgb::new(210, 92, 255))
                .with_inverse_linear_attenuation(840.0),
        ],
        ambient_reflection: ReflectionConstants::new(0.08, 0.09, 0.16),
        diffuse_reflection: ReflectionConstants::new(0.62, 0.74, 0.90),
        specular_reflection: ReflectionConstants::new(0.34, 0.42, 0.58),
        specular_exponent: 18,
        ..Lighting::default()
    }
}

fn sky(t: f64) -> Canvas {
    Canvas::from_fn_independent_with_options(
        WIDTH,
        HEIGHT,
        move |x, y| {
            let nx = f64::from(x) / f64::from(WIDTH);
            let ny = f64::from(y) / f64::from(HEIGHT);
            let horizon = (1.0 - ((ny - 0.70) * 2.8).abs()).clamp(0.0, 1.0);
            let aurora = ((nx * 6.2 - ny * 8.4 + t * TAU).sin() * 0.5 + 0.5).powf(3.0);
            let violet = ((nx * 4.0 + ny * 3.0 - t * 4.0).cos() * 0.5 + 0.5).powf(2.4);
            let base = hsl(
                224.0 - 36.0 * horizon + 72.0 * aurora,
                78,
                4 + (13.0 * horizon) as u16,
            );
            let glow = hsl(164.0 + 96.0 * violet, 82, (5.0 + 18.0 * aurora) as u16);
            base.lerp(glow, (0.22 + aurora * 0.28).clamp(0.0, 0.54))
        },
        true,
        false,
    )
}

fn draw_stars(canvas: &mut Canvas, stars: &[Star], t: f64) {
    for (i, star) in stars.iter().enumerate() {
        let twinkle = ((star.phase + t * TAU * 1.8).sin() * 0.5 + 0.5).powf(1.8);
        if twinkle < 0.16 {
            continue;
        }
        let color = hsl(star.hue, 72, (22.0 + 62.0 * twinkle) as u16);
        let radius = if i % 23 == 0 { 2 } else { 1 };
        plot_cross(canvas, star.x, star.y, -820.0, radius, color);
    }
}

fn draw_aurora(canvas: &mut Canvas, t: f64) {
    for ribbon in 0..7 {
        let hue = 134.0 + ribbon as f64 * 23.0;
        for band in 0..6 {
            let offset = band as f64 * 7.0 - 19.0;
            let color = hsl(hue + offset * 1.7, 88, 16 + (6 - band) as u16 * 4);
            let z = -710.0 + band as f64;
            let mut prev = aurora_point(0.0, ribbon, offset, t);
            for step in 1..=160 {
                let u = step as f64 / 160.0;
                let next = aurora_point(u, ribbon, offset, t);
                canvas.draw_line_z(color.scale(0.78), (prev.0, prev.1, z), (next.0, next.1, z));
                prev = next;
            }
        }
    }
}

fn aurora_point(u: f64, ribbon: usize, offset: f64, t: f64) -> (f64, f64) {
    let x = u * WIDTH as f64;
    let phase = t * TAU * (0.18 + ribbon as f64 * 0.015) + ribbon as f64 * 0.83;
    let y = 78.0
        + ribbon as f64 * 34.0
        + offset
        + (u * TAU * 2.0 + phase).sin() * (18.0 + ribbon as f64 * 3.0)
        + (u * TAU * 5.0 - phase * 1.3).cos() * 9.0;
    (x, y)
}

fn draw_cloud_sea(canvas: &mut Canvas, t: f64) {
    for layer in 0..4 {
        let y = 508.0 + layer as f64 * 22.0;
        let color = match layer {
            0 => Rgb::new(16, 34, 64),
            1 => Rgb::new(22, 48, 82),
            2 => Rgb::new(32, 58, 92),
            _ => Rgb::new(46, 66, 104),
        };
        let mut prev = None;
        for i in 0..=120 {
            let x = i as f64 / 120.0 * WIDTH as f64;
            let wave = (i as f64 * 0.22 + t * TAU + layer as f64).sin() * 16.0
                + (i as f64 * 0.071 - t * TAU * 0.7).cos() * 24.0;
            let p = (x, y + wave, -180.0 + layer as f64);
            if let Some(last) = prev {
                canvas.set_line_width(8.0 + layer as f64 * 3.0);
                canvas.draw_line_z(color, last, p);
            }
            prev = Some(p);
        }
    }
    canvas.set_line_width(1.0);
}

fn draw_tail_wisps(canvas: &mut Canvas, t: f64) {
    for wisp in 0..9 {
        let color = hsl(178.0 + wisp as f64 * 17.0, 88, 34);
        let mut prev = None;
        for step in 0..90 {
            let u = step as f64 / 89.0;
            let x = 152.0 + u * 214.0 + (t * TAU + wisp as f64).sin() * 7.0;
            let y = 522.0
                + (u * TAU * 0.88 + wisp as f64 * 0.7 - t * TAU * 0.4).sin() * 56.0
                + wisp as f64 * 7.0;
            let z = 40.0 + wisp as f64 * 0.4;
            if let Some(last) = prev {
                canvas.draw_line_z(color.scale(0.58 - u * 0.25), last, (x, y, z));
            }
            prev = Some((x, y, z));
        }
    }
}

fn draw_tail_tip(canvas: &mut Canvas, t: f64) {
    let root = dragon_path(0.012, t);
    let tangent = unit_tangent(dragon_tangent(0.025, t));
    let normal = unit_normal(tangent);
    let sway = (t * TAU).sin();
    let back = (-tangent.0, -tangent.1);
    let mid = (
        root.0 + back.0 * 34.0 + normal.0 * (3.0 + sway * 2.0),
        root.1 + back.1 * 34.0 + normal.1 * (3.0 + sway * 2.0),
    );
    let tip = (
        root.0 + back.0 * 72.0 + normal.0 * (7.0 + sway * 3.0),
        root.1 + back.1 * 72.0 + normal.1 * (7.0 + sway * 3.0),
    );

    let root_w = 11.0;
    let mid_w = 5.0;
    let root_l = (
        root.0 + normal.0 * root_w,
        root.1 + normal.1 * root_w,
        292.0,
    );
    let root_r = (
        root.0 - normal.0 * root_w,
        root.1 - normal.1 * root_w,
        292.0,
    );
    let mid_l = (mid.0 + normal.0 * mid_w, mid.1 + normal.1 * mid_w, 306.0);
    let mid_r = (mid.0 - normal.0 * mid_w, mid.1 - normal.1 * mid_w, 306.0);
    let tip_p = (tip.0, tip.1, 318.0);

    draw_quad(canvas, root_l, mid_l, mid_r, root_r, Rgb::new(78, 63, 56));
    canvas.draw_triangle(Rgb::new(236, 218, 158), mid_l, tip_p, mid_r);

    let root_hi = (
        root.0 + normal.0 * root_w * 0.36,
        root.1 + normal.1 * root_w * 0.36,
        330.0,
    );
    let mid_hi = (
        mid.0 + normal.0 * mid_w * 0.30,
        mid.1 + normal.1 * mid_w * 0.30,
        332.0,
    );
    canvas.draw_triangle(Rgb::new(255, 245, 196), root_hi, mid_hi, tip_p);

    let fin_color = Rgb::new(236, 218, 158);
    canvas.draw_triangle(
        fin_color.scale(0.78),
        (mid.0 + normal.0 * 6.0, mid.1 + normal.1 * 6.0, 322.0),
        (
            mid.0 + normal.0 * 20.0 - tangent.0 * 5.0,
            mid.1 + normal.1 * 20.0 - tangent.1 * 5.0,
            322.0,
        ),
        (tip.0 + normal.0 * 3.0, tip.1 + normal.1 * 3.0, 322.0),
    );
    canvas.draw_triangle(
        Rgb::new(78, 63, 56),
        (mid.0 - normal.0 * 5.0, mid.1 - normal.1 * 5.0, 321.0),
        (
            mid.0 - normal.0 * 18.0 - tangent.0 * 4.0,
            mid.1 - normal.1 * 18.0 - tangent.1 * 4.0,
            321.0,
        ),
        (tip.0 - normal.0 * 3.0, tip.1 - normal.1 * 3.0, 321.0),
    );

    canvas.set_line_width(2.0);
    canvas.draw_line_z(
        Rgb::new(255, 245, 196),
        (root.0, root.1, 340.0),
        (tip.0, tip.1, 340.0),
    );
    canvas.set_line_width(1.0);
    plot_cross(canvas, tip.0, tip.1, 350.0, 1, Rgb::new(242, 226, 170));
}

fn draw_aurora_wings(canvas: &mut Canvas, t: f64) {
    for wing in 0..2 {
        let root_u = 0.68 + wing as f64 * 0.08;
        let (root_x, root_y) = dragon_path(root_u, t);
        for strand in 0..8 {
            let s = strand as f64 / 7.0;
            let color = hsl(136.0 + s * 104.0 + wing as f64 * 18.0, 88, 34);
            let mut prev = None;
            for step in 0..58 {
                let k = step as f64 / 57.0;
                let sweep = 118.0 + wing as f64 * 52.0 + s * 36.0;
                let lift = 118.0 + s * 56.0;
                let x = root_x - k * sweep + (k * TAU * 1.1 + s * PI + t * TAU * 0.35).sin() * 18.0;
                let y = root_y - k * lift + (k * TAU * 0.9 + strand as f64).cos() * 20.0;
                let z = 58.0 + wing as f64 * 4.0 + strand as f64 * 0.2;
                if let Some(last) = prev {
                    canvas.draw_line_z(color.scale(0.72 - k * 0.28), last, (x, y, z));
                }
                prev = Some((x, y, z));
            }
        }
    }
}

fn draw_body_ribbon(canvas: &mut Canvas, t: f64) {
    draw_body_band(canvas, t, -0.92, 0.92, 108.0, |u| {
        hsl(214.0 + u * 28.0, 72, 10)
    });
    draw_body_band(canvas, t, -0.78, 0.78, 132.0, |u| {
        hsl(204.0 + u * 58.0, 72, (28.0 + u * 11.0) as u16)
    });
    draw_body_band(canvas, t, 0.14, 0.60, 156.0, |u| {
        hsl(186.0 + u * 74.0, 82, (39.0 + u * 11.0) as u16)
    });
    draw_body_band(canvas, t, -0.56, -0.18, 166.0, |u| {
        hsl(212.0 + u * 26.0, 50, (22.0 + u * 6.0) as u16)
    });
}

fn draw_body_band<F>(
    canvas: &mut Canvas,
    t: f64,
    side_a: f64,
    side_b: f64,
    z_base: f64,
    color_at: F,
) where
    F: Fn(f64) -> Rgb,
{
    let samples = 96;
    for i in 0..samples {
        let u0 = i as f64 / samples as f64;
        let u1 = (i + 1) as f64 / samples as f64;
        let a0 = body_side_point(u0, t, side_a, z_base);
        let b0 = body_side_point(u0, t, side_b, z_base);
        let a1 = body_side_point(u1, t, side_a, z_base);
        let b1 = body_side_point(u1, t, side_b, z_base);
        let color = color_at((u0 + u1) * 0.5);
        draw_quad(canvas, a0, a1, b1, b0, color);
    }
}

fn body_side_point(u: f64, t: f64, side: f64, z_base: f64) -> (f64, f64, f64) {
    let (x, y) = dragon_path(u, t);
    let normal = unit_normal(dragon_tangent(u, t));
    let radius = dragon_radius(u);
    let radius = radius * coil_width_factor(u);
    (
        x + normal.0 * radius * side,
        y + normal.1 * radius * side,
        z_base + u * 96.0 + side.abs() * 2.0,
    )
}

fn draw_dragon_body(canvas: &mut Canvas, assets: &Assets, t: f64) {
    draw_body_ribbon(canvas, t);
    let base = dragon_material();
    for i in (0..38).rev() {
        let u = 0.035 + i as f64 / 37.0 * 0.92;
        let (x, y) = dragon_path(u, t);
        let tangent = dragon_tangent(u, t);
        let angle = tangent.1.atan2(tangent.0).to_degrees();
        let normal = unit_normal(tangent);
        let radius = dragon_radius(u);
        let pulse = 1.0 + 0.035 * (t * TAU + i as f64 * 0.58).sin();
        let sx = radius * 0.60 * pulse;
        let sy = radius * 0.24 * pulse;
        let sz = 7.0 + radius * 0.16;
        let mat = Material {
            line: base.line.lerp(hsl(188.0 + u * 88.0, 88, 62), 0.34),
            diffuse: ReflectionConstants::new(0.26 + u * 0.22, 0.48 + u * 0.16, 0.72),
            ..base
        };
        let transform = Matrix::translate(
            x + normal.0 * radius * 0.22,
            y + normal.1 * radius * 0.22,
            210.0 + u * 90.0,
        ) * Matrix::rotate_z(angle)
            * Matrix::rotate_y(18.0 * (u * TAU + t * TAU).sin())
            * Matrix::scale(sx, sy, sz);
        draw_lit_mesh(canvas, &assets.sphere, &transform, mat);

        if i % 4 == 0 && i > 5 {
            draw_dorsal_fin(canvas, assets, x, y, angle, normal, radius, u);
        }
    }
    draw_belly_scutes(canvas, t);
}

fn draw_belly_scutes(canvas: &mut Canvas, t: f64) {
    canvas.set_line_width(2.0);
    for i in 0..34 {
        let u = 0.06 + i as f64 / 33.0 * 0.86;
        let (x, y) = dragon_path(u, t);
        let tangent = dragon_tangent(u, t);
        let normal = unit_normal(tangent);
        let tangent = unit_tangent(tangent);
        let radius = dragon_radius(u);
        let color = hsl(202.0 + u * 42.0, 62, (32.0 + u * 8.0) as u16);
        let start = (
            x - normal.0 * radius * 0.72 - tangent.0 * radius * 0.12,
            y - normal.1 * radius * 0.72 - tangent.1 * radius * 0.12,
            278.0 + u * 82.0,
        );
        let end = (
            x - normal.0 * radius * 0.18 + tangent.0 * radius * 0.18,
            y - normal.1 * radius * 0.18 + tangent.1 * radius * 0.18,
            278.0 + u * 82.0,
        );
        canvas.draw_line_z(color.scale(0.82), start, end);
    }
    canvas.set_line_width(1.0);
}

fn dragon_path(u: f64, t: f64) -> (f64, f64) {
    let points = [
        (122.0, 474.0),
        (206.0, 502.0),
        (318.0, 446.0),
        (390.0, 310.0),
        (356.0, 224.0),
        (488.0, 250.0),
        (560.0, 336.0),
        (622.0, 324.0),
    ];
    let (mut x, mut y) = catmull_rom_path(&points, u);
    let wobble = 5.5 * (t * TAU + u * TAU * 2.2).sin() * (0.35 + u * 0.65);
    y += wobble;
    x += 3.0 * (t * TAU * 0.7 + u * TAU).cos();
    (x, y)
}

fn dragon_tangent(u: f64, t: f64) -> (f64, f64) {
    let du = 0.01;
    let a = dragon_path((u - du).clamp(0.0, 1.0), t);
    let b = dragon_path((u + du).clamp(0.0, 1.0), t);
    (b.0 - a.0, b.1 - a.1)
}

fn dragon_radius(u: f64) -> f64 {
    let grow = smoothstep_unit((u - 0.04) / 0.72);
    let neck = smoothstep_unit((u - 0.72) / 0.28);
    let tail_tip = 1.0 - smoothstep_unit(u / 0.12);
    (13.0 + 37.0 * grow + 10.0 * neck - 5.0 * tail_tip).clamp(8.0, 61.0)
}

fn coil_width_factor(u: f64) -> f64 {
    let crossing = smoothstep_unit((u - 0.42) / 0.18) * (1.0 - smoothstep_unit((u - 0.68) / 0.18));
    1.0 - crossing * 0.24
}

fn unit_tangent(tangent: (f64, f64)) -> (f64, f64) {
    let len = (tangent.0 * tangent.0 + tangent.1 * tangent.1).sqrt();
    if len <= f64::EPSILON {
        (1.0, 0.0)
    } else {
        (tangent.0 / len, tangent.1 / len)
    }
}

fn unit_normal(tangent: (f64, f64)) -> (f64, f64) {
    let tangent = unit_tangent(tangent);
    (-tangent.1, tangent.0)
}

fn smoothstep_unit(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn catmull_rom_path(points: &[(f64, f64)], u: f64) -> (f64, f64) {
    let last = points.len() - 1;
    let scaled = u.clamp(0.0, 1.0) * last as f64;
    let i = (scaled.floor() as usize).min(last - 1);
    let local = scaled - i as f64;
    let p0 = points[i.saturating_sub(1)];
    let p1 = points[i];
    let p2 = points[(i + 1).min(last)];
    let p3 = points[(i + 2).min(last)];
    (
        catmull(p0.0, p1.0, p2.0, p3.0, local),
        catmull(p0.1, p1.1, p2.1, p3.1, local),
    )
}

fn catmull(p0: f64, p1: f64, p2: f64, p3: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

#[allow(clippy::too_many_arguments)]
fn draw_dorsal_fin(
    canvas: &mut Canvas,
    assets: &Assets,
    x: f64,
    y: f64,
    angle: f64,
    normal: (f64, f64),
    radius: f64,
    u: f64,
) {
    let mat = mat(
        hsl(174.0 + u * 96.0, 92, 54),
        (0.02, 0.10, 0.12),
        (0.14, 0.54, 0.68),
        (0.08, 0.30, 0.38),
        12,
    );
    let transform = Matrix::translate(
        x + normal.0 * radius * 0.88,
        y + normal.1 * radius * 0.88,
        250.0 + u * 88.0,
    ) * Matrix::rotate_z(angle - 86.0)
        * Matrix::rotate_y(18.0)
        * Matrix::scale(6.0 + u * 7.0, 14.0 + u * 18.0, 6.0);
    draw_lit_mesh(canvas, &assets.crystal, &transform, mat);
}

fn draw_nebula_mane(canvas: &mut Canvas, t: f64) {
    for strand in 0..18 {
        let u0 = 0.55 + strand as f64 * 0.021;
        let color = hsl(286.0 + strand as f64 * 7.0, 86, 38);
        let mut prev = None;
        for step in 0..42 {
            let k = step as f64 / 41.0;
            let (bx, by) = dragon_path((u0 + k * 0.10).clamp(0.0, 1.0), t);
            let x = bx - k * 26.0 + (t * TAU + k * TAU + strand as f64).sin() * 10.0;
            let y = by - 28.0 - k * 50.0 + (k * TAU * 1.2 + strand as f64).cos() * 12.0;
            if let Some(last) = prev {
                canvas.draw_line_z(color.scale(0.72 - k * 0.32), last, (x, y, 222.0));
            }
            prev = Some((x, y, 222.0));
        }
    }
}

fn draw_sun_glow(canvas: &mut Canvas, t: f64) {
    let pulse = 0.5 + 0.5 * (t * TAU).sin();
    for r in (24..=62).rev() {
        let k = 1.0 - (r - 24) as f64 / 38.0;
        let color = Rgb::new(
            (42.0 + 168.0 * k + 18.0 * pulse) as u8,
            (16.0 + 96.0 * k + 12.0 * pulse) as u8,
            (4.0 + 20.0 * k) as u8,
        );
        draw_disk(
            canvas,
            SUN_X,
            SUN_Y,
            r as f64,
            color,
            150.0 - r as f64 * 0.12,
        );
    }
    for ray in 0..22 {
        let a = ray as f64 / 22.0 * TAU + t * TAU * 0.08;
        let len = 36.0 + (ray % 5) as f64 * 6.0;
        let c = hsl(42.0 + ray as f64 * 2.0, 96, 42);
        canvas.draw_line_z(
            c.scale(0.72),
            (SUN_X + a.cos() * 23.0, SUN_Y + a.sin() * 23.0, 155.0),
            (SUN_X + a.cos() * len, SUN_Y + a.sin() * len, 155.0),
        );
    }
}

fn draw_little_sun(canvas: &mut Canvas, assets: &Assets, t: f64) {
    let mat = mat(
        Rgb::new(255, 218, 70),
        (0.65, 0.42, 0.08),
        (0.96, 0.72, 0.18),
        (0.68, 0.48, 0.12),
        20,
    );
    let spin = Matrix::translate(SUN_X, SUN_Y, 370.0)
        * Matrix::rotate_y(t * 360.0)
        * Matrix::rotate_x(18.0)
        * Matrix::scale(25.0, 25.0, 25.0);
    draw_lit_mesh(canvas, &assets.sphere, &spin, mat);
    draw_disk(canvas, SUN_X, SUN_Y, 14.0, Rgb::new(255, 178, 34), 404.0);
    draw_disk(
        canvas,
        SUN_X - 5.0,
        SUN_Y - 6.0,
        8.0,
        Rgb::new(255, 246, 190),
        410.0,
    );
}

fn draw_neck_bridge(canvas: &mut Canvas, t: f64) {
    let shoulder = dragon_path(0.86, t);
    draw_quad(
        canvas,
        (shoulder.0 + 8.0, shoulder.1 - 26.0, 326.0),
        (632.0, 274.0, 360.0),
        (654.0, 294.0, 360.0),
        (shoulder.0 + 34.0, shoulder.1 + 6.0, 326.0),
        hsl(204.0, 70, 24),
    );
    draw_quad(
        canvas,
        (shoulder.0 + 28.0, shoulder.1 - 12.0, 372.0),
        (638.0, 286.0, 388.0),
        (650.0, 298.0, 388.0),
        (shoulder.0 + 44.0, shoulder.1 + 4.0, 372.0),
        hsl(184.0, 84, 40),
    );
    canvas.draw_line_z(
        hsl(182.0, 96, 55),
        (shoulder.0 + 28.0, shoulder.1 - 12.0, 404.0),
        (654.0, 292.0, 404.0),
    );
}

fn draw_head_base(canvas: &mut Canvas, assets: &Assets, t: f64) {
    let base = dragon_material();
    let cranium = Matrix::translate(HEAD_X - 18.0, HEAD_Y - 10.0, 296.0)
        * Matrix::rotate_z(-12.0)
        * Matrix::rotate_y(-8.0 + 5.0 * (t * TAU).sin())
        * Matrix::scale(48.0, 41.0, 44.0);
    draw_lit_mesh(canvas, &assets.sphere, &cranium, base);

    let snout_mat = Material {
        line: Rgb::new(105, 178, 232),
        ambient: ReflectionConstants::new(0.05, 0.08, 0.15),
        diffuse: ReflectionConstants::new(0.34, 0.56, 0.82),
        specular: ReflectionConstants::new(0.26, 0.34, 0.48),
        specular_exponent: 22,
    };
    let snout = Matrix::translate(686.0, 276.0, 330.0)
        * Matrix::rotate_z(-10.0)
        * Matrix::rotate_y(-12.0)
        * Matrix::scale(70.0, 25.0, 27.0);
    draw_lit_mesh(canvas, &assets.sphere, &snout, snout_mat);

    let muzzle = Matrix::translate(724.0, 272.0, 356.0)
        * Matrix::rotate_z(-6.0)
        * Matrix::rotate_y(-14.0)
        * Matrix::scale(42.0, 15.0, 18.0);
    draw_lit_mesh(canvas, &assets.sphere, &muzzle, snout_mat);

    let brow = mat(
        Rgb::new(130, 190, 238),
        (0.08, 0.12, 0.20),
        (0.44, 0.64, 0.88),
        (0.28, 0.36, 0.48),
        18,
    );
    let cheek = Matrix::translate(660.0, 307.0, 330.0)
        * Matrix::rotate_z(-12.0)
        * Matrix::scale(37.0, 23.0, 23.0);
    draw_lit_mesh(canvas, &assets.sphere, &cheek, brow);

    let brow_ridge = Matrix::translate(646.0, 254.0, 384.0)
        * Matrix::rotate_z(-8.0)
        * Matrix::rotate_y(-10.0)
        * Matrix::scale(50.0, 11.0, 12.0);
    draw_lit_mesh(canvas, &assets.sphere, &brow_ridge, brow);

    let throat = Matrix::translate(636.0, 326.0, 304.0)
        * Matrix::rotate_z(-6.0)
        * Matrix::scale(52.0, 22.0, 24.0);
    draw_lit_mesh(canvas, &assets.sphere, &throat, base);
    draw_head_facets(canvas);
}

fn draw_head_facets(canvas: &mut Canvas) {
    draw_quad(
        canvas,
        (596.0, 266.0, 414.0),
        (642.0, 238.0, 414.0),
        (684.0, 254.0, 414.0),
        (641.0, 281.0, 414.0),
        hsl(198.0, 76, 34),
    );
    draw_quad(
        canvas,
        (642.0, 276.0, 418.0),
        (720.0, 255.0, 418.0),
        (748.0, 274.0, 418.0),
        (666.0, 292.0, 418.0),
        hsl(188.0, 84, 42),
    );
    draw_quad(
        canvas,
        (626.0, 294.0, 416.0),
        (682.0, 292.0, 416.0),
        (654.0, 334.0, 416.0),
        (604.0, 326.0, 416.0),
        hsl(214.0, 72, 23),
    );
    canvas.draw_triangle(
        hsl(202.0, 84, 53),
        (680.0, 244.0, 422.0),
        (748.0, 270.0, 422.0),
        (708.0, 276.0, 422.0),
    );
    canvas.draw_triangle(
        hsl(222.0, 64, 18),
        (650.0, 318.0, 421.0),
        (722.0, 304.0, 421.0),
        (676.0, 340.0, 421.0),
    );
    canvas.draw_line_z(
        Rgb::new(14, 28, 58),
        (600.0, 267.0, 424.0),
        (684.0, 254.0, 424.0),
    );
    canvas.draw_line_z(
        Rgb::new(14, 28, 58),
        (644.0, 276.0, 424.0),
        (746.0, 274.0, 424.0),
    );
    canvas.draw_line_z(
        Rgb::new(84, 176, 220),
        (620.0, 304.0, 425.0),
        (705.0, 290.0, 425.0),
    );
}

fn draw_mouth_shadow(canvas: &mut Canvas) {
    let dark = Rgb::new(9, 12, 24);
    draw_quad(
        canvas,
        (639.0, 281.0, 310.0),
        (726.0, 232.0, 310.0),
        (738.0, 275.0, 310.0),
        (646.0, 326.0, 310.0),
        dark,
    );
    draw_quad(
        canvas,
        (672.0, 286.0, 311.0),
        (735.0, 272.0, 311.0),
        (722.0, 310.0, 311.0),
        (664.0, 322.0, 311.0),
        Rgb::new(25, 13, 26),
    );
}

fn draw_jaws_and_horns(canvas: &mut Canvas, assets: &Assets, t: f64) {
    let jaw_mat = mat(
        Rgb::new(170, 224, 246),
        (0.10, 0.12, 0.18),
        (0.56, 0.72, 0.92),
        (0.34, 0.44, 0.58),
        22,
    );
    let upper = Matrix::translate(696.0, 251.0, 438.0)
        * Matrix::rotate_z(-11.0)
        * Matrix::rotate_y(-12.0)
        * Matrix::scale(58.0, 16.0, 20.0);
    draw_lit_mesh(canvas, &assets.sphere, &upper, jaw_mat);

    let lower = Matrix::translate(690.0, 314.0, 440.0)
        * Matrix::rotate_z(1.0)
        * Matrix::rotate_y(-8.0)
        * Matrix::scale(54.0, 16.0, 20.0);
    draw_lit_mesh(canvas, &assets.sphere, &lower, jaw_mat);

    draw_teeth(canvas);
    draw_horns(canvas, t);
    draw_whiskers(canvas, t);
}

fn draw_horns(canvas: &mut Canvas, t: f64) {
    let sway = 2.5 * (t * TAU).sin();
    draw_crescent_horn(
        canvas,
        (610.0, 245.0),
        (585.0, 213.0 + sway),
        (558.0, 202.0 - sway),
        (544.0, 166.0),
        508.0,
        0.78,
    );
    draw_crescent_horn(
        canvas,
        (637.0, 242.0),
        (616.0, 207.0 - sway),
        (593.0, 193.0 + sway),
        (584.0, 160.0),
        548.0,
        1.0,
    );
}

fn draw_crescent_horn(
    canvas: &mut Canvas,
    root: (f64, f64),
    c1: (f64, f64),
    c2: (f64, f64),
    tip: (f64, f64),
    z: f64,
    brightness: f64,
) {
    draw_horn_curve(
        canvas,
        root,
        c1,
        c2,
        tip,
        z - 2.0,
        7.0,
        Rgb::new(78, 63, 56),
    );
    draw_horn_curve(
        canvas,
        root,
        c1,
        c2,
        tip,
        z,
        4.0,
        Rgb::new(236, 218, 158).scale(brightness),
    );
    draw_horn_curve(
        canvas,
        (root.0 + 3.0, root.1 - 2.0),
        (c1.0 + 3.0, c1.1 - 3.0),
        (c2.0 + 2.0, c2.1 - 3.0),
        (tip.0 + 1.0, tip.1 + 2.0),
        z + 2.0,
        1.5,
        Rgb::new(255, 245, 196).scale(brightness),
    );

    for branch in 0..3 {
        let k = 0.34 + branch as f64 * 0.19;
        let base = cubic_point(root, c1, c2, tip, k);
        let tangent = unit_tangent(cubic_tangent(root, c1, c2, tip, k));
        let normal = unit_normal(tangent);
        let length = 20.0 - branch as f64 * 3.0;
        let end = (
            base.0 + normal.0 * length - tangent.0 * 0.22 * length,
            base.1 + normal.1 * length - tangent.1 * 0.22 * length,
        );
        draw_horn_curve(
            canvas,
            base,
            (
                base.0 + normal.0 * length * 0.35,
                base.1 + normal.1 * length * 0.35,
            ),
            (
                end.0 - tangent.0 * length * 0.12,
                end.1 - tangent.1 * length * 0.12,
            ),
            end,
            z + 3.0,
            2.5,
            Rgb::new(242, 226, 170).scale(brightness),
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_horn_curve(
    canvas: &mut Canvas,
    root: (f64, f64),
    c1: (f64, f64),
    c2: (f64, f64),
    tip: (f64, f64),
    z: f64,
    width: f64,
    color: Rgb,
) {
    canvas.set_line_width(width);
    let mut previous = cubic_point(root, c1, c2, tip, 0.0);
    for step in 1..=36 {
        let u = step as f64 / 36.0;
        let next = cubic_point(root, c1, c2, tip, u);
        canvas.draw_line_z(color, (previous.0, previous.1, z), (next.0, next.1, z));
        previous = next;
    }
    canvas.set_line_width(1.0);
}

fn cubic_point(
    p0: (f64, f64),
    p1: (f64, f64),
    p2: (f64, f64),
    p3: (f64, f64),
    t: f64,
) -> (f64, f64) {
    let a = (1.0 - t).powi(3);
    let b = 3.0 * (1.0 - t).powi(2) * t;
    let c = 3.0 * (1.0 - t) * t * t;
    let d = t.powi(3);
    (
        p0.0 * a + p1.0 * b + p2.0 * c + p3.0 * d,
        p0.1 * a + p1.1 * b + p2.1 * c + p3.1 * d,
    )
}

fn cubic_tangent(
    p0: (f64, f64),
    p1: (f64, f64),
    p2: (f64, f64),
    p3: (f64, f64),
    t: f64,
) -> (f64, f64) {
    let a = 3.0 * (1.0 - t).powi(2);
    let b = 6.0 * (1.0 - t) * t;
    let c = 3.0 * t * t;
    (
        (p1.0 - p0.0) * a + (p2.0 - p1.0) * b + (p3.0 - p2.0) * c,
        (p1.1 - p0.1) * a + (p2.1 - p1.1) * b + (p3.1 - p2.1) * c,
    )
}

fn draw_teeth(canvas: &mut Canvas) {
    let tooth = Rgb::new(244, 246, 224);
    let teeth = [
        [
            (682.0, 244.0, 468.0),
            (692.0, 254.0, 468.0),
            (686.0, 279.0, 468.0),
        ],
        [
            (718.0, 244.0, 470.0),
            (730.0, 252.0, 470.0),
            (718.0, 283.0, 470.0),
        ],
        [
            (671.0, 311.0, 469.0),
            (681.0, 301.0, 469.0),
            (681.0, 279.0, 469.0),
        ],
        [
            (713.0, 312.0, 471.0),
            (725.0, 300.0, 471.0),
            (714.0, 278.0, 471.0),
        ],
    ];
    for tri in teeth {
        canvas.draw_triangle(tooth, tri[0], tri[1], tri[2]);
        canvas.draw_line_z(Rgb::new(130, 140, 170), tri[0], tri[2]);
    }
}

fn draw_whiskers(canvas: &mut Canvas, t: f64) {
    for side in [-1.0, 1.0] {
        for line in 0..3 {
            let color = hsl(182.0 + line as f64 * 18.0, 88, 58);
            let mut prev = None;
            for step in 0..50 {
                let u = step as f64 / 49.0;
                let x = 640.0 + u * (176.0 + line as f64 * 20.0);
                let y = 292.0
                    + side * (18.0 + line as f64 * 10.0)
                    + (u * TAU * 0.8 + t * TAU + line as f64).sin() * 15.0
                    + u * u * side * 40.0;
                if let Some(last) = prev {
                    canvas.draw_line_z(color.scale(0.75 - u * 0.28), last, (x, y, 390.0));
                }
                prev = Some((x, y, 390.0));
            }
        }
    }
}

fn draw_scale_sparks(canvas: &mut Canvas, t: f64) {
    for i in 0..42 {
        let u = 0.12 + i as f64 / 48.0 * 0.78;
        let (x, y) = dragon_path(u, t);
        let side = if i % 2 == 0 { -1.0 } else { 1.0 };
        let color = hsl(178.0 + i as f64 * 9.0, 94, 58);
        let x = x + side * (16.0 + (i % 5) as f64 * 4.0);
        let y = y - 6.0 + side * (i % 7) as f64 * 2.0;
        plot_cross(canvas, x, y, 420.0, 1 + (i % 3 == 0) as i64, color);
    }
}

fn draw_face_lines(canvas: &mut Canvas, breath: f64) {
    let eye = Rgb::new(255, 238, 150);
    draw_disk(canvas, 653.0, 263.0, 8.5 + breath * 1.0, eye, 526.0);
    draw_disk(canvas, 656.0, 263.0, 3.5, Rgb::new(7, 9, 24), 527.0);
    draw_disk(canvas, 719.0, 276.0, 3.4, Rgb::new(6, 11, 24), 526.0);
    canvas.draw_line_z(
        Rgb::new(18, 35, 72),
        (617.0, 264.0, 528.0),
        (683.0, 249.0, 528.0),
    );
    canvas.draw_line_z(
        Rgb::new(18, 35, 72),
        (621.0, 302.0, 528.0),
        (675.0, 316.0, 528.0),
    );
    canvas.draw_line_z(
        Rgb::new(175, 226, 246),
        (646.0, 277.0, 529.0),
        (724.0, 258.0, 529.0),
    );
    canvas.draw_line_z(
        Rgb::new(118, 170, 220),
        (646.0, 322.0, 529.0),
        (716.0, 303.0, 529.0),
    );
    canvas.draw_line_z(
        Rgb::new(68, 142, 204),
        (672.0, 268.0, 530.0),
        (716.0, 276.0, 530.0),
    );
}

fn draw_foreground_aurora(canvas: &mut Canvas, t: f64) {
    for i in 0..12 {
        let color = hsl(128.0 + i as f64 * 18.0, 90, 32);
        let mut prev = None;
        for step in 0..72 {
            let u = step as f64 / 71.0;
            let x = 92.0 + u * 800.0;
            let y =
                56.0 + i as f64 * 7.0 + (u * TAU * 1.35 + t * TAU * 0.4 + i as f64).sin() * 24.0;
            if let Some(last) = prev {
                canvas.draw_line_z(color.scale(0.34), last, (x, y, 520.0 + i as f64));
            }
            prev = Some((x, y, 520.0 + i as f64));
        }
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
        let lighting = canvas.lighting_mut();
        lighting.ambient_reflection = material.ambient;
        lighting.diffuse_reflection = material.diffuse;
        lighting.specular_reflection = material.specular;
        lighting.specular_exponent = material.specular_exponent;
    }
    canvas.draw_polygons(&mesh.apply(transform));
}

fn draw_disk(canvas: &mut Canvas, cx: f64, cy: f64, radius: f64, color: Rgb, z: f64) {
    let r = radius.ceil() as i64;
    for dy in -r..=r {
        let yy = cy.round() as i64 + dy;
        let span = (radius * radius - (dy as f64).powi(2)).max(0.0).sqrt() as i64;
        for dx in -span..=span {
            canvas.plot_z(&color, cx.round() as i64 + dx, yy, z);
        }
    }
}

fn draw_quad(
    canvas: &mut Canvas,
    a: (f64, f64, f64),
    b: (f64, f64, f64),
    c: (f64, f64, f64),
    d: (f64, f64, f64),
    color: Rgb,
) {
    canvas.draw_triangle(color, a, b, c);
    canvas.draw_triangle(color, a, c, d);
}

fn plot_cross(canvas: &mut Canvas, x: f64, y: f64, z: f64, radius: i64, color: Rgb) {
    let x = x.round() as i64;
    let y = y.round() as i64;
    canvas.plot_z(&color, x, y, z);
    for d in 1..=radius {
        let c = color.scale(0.72);
        canvas.plot_z(&c, x + d, y, z);
        canvas.plot_z(&c, x - d, y, z);
        canvas.plot_z(&c, x, y + d, z);
        canvas.plot_z(&c, x, y - d, z);
    }
}

fn dragon_material() -> Material {
    mat(
        Rgb::new(116, 184, 236),
        (0.05, 0.08, 0.16),
        (0.34, 0.56, 0.86),
        (0.30, 0.40, 0.56),
        24,
    )
}

fn mat(
    line: Rgb,
    ambient: (f64, f64, f64),
    diffuse: (f64, f64, f64),
    specular: (f64, f64, f64),
    specular_exponent: u32,
) -> Material {
    Material {
        line,
        ambient: ReflectionConstants::new(ambient.0, ambient.1, ambient.2),
        diffuse: ReflectionConstants::new(diffuse.0, diffuse.1, diffuse.2),
        specular: ReflectionConstants::new(specular.0, specular.1, specular.2),
        specular_exponent,
    }
}

fn hsl(hue: f64, saturation: u16, light: u16) -> Rgb {
    Rgb::from_hsl_f64(hue, f64::from(saturation) / 100.0, f64::from(light) / 100.0)
}

fn lcg(seed: u64) -> u64 {
    seed.wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}
