//! The Nocturne Atlas: a floating isometric city under a glass weather dome.
//!
//! It uses gartus' canvas, triangle rasterizer, z-buffered lines, color tools,
//! and animation recorder to build a dense procedural scene from scratch.

use gartus::{gmath::procedural::hash01_2d as hash01, prelude::*};
use std::{cmp::Ordering, error::Error, f64::consts::PI, fs};

const WIDTH: u32 = 900;
const HEIGHT: u32 = 900;
const FRAMES: usize = 108;
const CENTER_X: f64 = WIDTH as f64 * 0.5;
const BASE_Y: f64 = HEIGHT as f64 * 0.68;
const ISO_SCALE: f64 = 32.5;

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final")?;

    let scene = Scene::build();
    let options =
        AnimationRenderOptions::new("anim", "nocturne-atlas-", FRAMES, "final/cosmic_loom.gif")
            .delay_cs(3)
            .preview(37, "final/cosmic_loom.png")
            .unique_frame_dir(true);

    FrameRecorder::render_gif_auto(options, |frame| Ok(render_frame(frame, &scene)))?;

    println!("Saved final/cosmic_loom.png and final/cosmic_loom.gif");
    Ok(())
}

fn render_frame(frame: usize, scene: &Scene) -> Canvas {
    let t = frame as f64 / FRAMES as f64;
    let mut canvas = sky(t);
    canvas.set_wrapped(false);
    canvas.set_upper_left_origin(true);

    draw_sky_signals(&mut canvas, scene, t);
    draw_glass_dome(&mut canvas, t);
    draw_weather(&mut canvas, scene, t);
    draw_reflections(&mut canvas, scene, t);
    draw_island(&mut canvas, t);
    draw_city_grid(&mut canvas, t);
    draw_monorail_track(&mut canvas, t);
    draw_city(&mut canvas, scene, t);
    draw_central_loom(&mut canvas, t);
    draw_light_threads(&mut canvas, scene, t);
    draw_floating_panes(&mut canvas, t);
    draw_monorail_train(&mut canvas, t);
    draw_foreground_gleam(&mut canvas, t);

    canvas
}

#[derive(Clone)]
struct Scene {
    buildings: Vec<Building>,
    stars: Vec<Star>,
}

#[derive(Clone, Copy)]
struct Building {
    x: f64,
    z: f64,
    w: f64,
    d: f64,
    h: f64,
    hue: f64,
    seed: f64,
    roof: u8,
}

#[derive(Clone, Copy)]
struct Star {
    x: f64,
    y: f64,
    phase: f64,
    hue: f64,
}

#[derive(Clone, Copy)]
struct WorldPoint {
    x: f64,
    y: f64,
    z: f64,
}

impl WorldPoint {
    const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

impl Scene {
    fn build() -> Self {
        let mut buildings = Vec::new();
        for gx in -5_i32..=5 {
            for gz in -5_i32..=5 {
                if gx.abs() <= 1 && gz.abs() <= 1 {
                    continue;
                }
                if gx == 0 && gz.abs() < 5 {
                    continue;
                }
                if gz == 0 && gx.abs() < 5 {
                    continue;
                }
                if hash01(gx, gz, 11) < 0.16 {
                    continue;
                }

                let base = hash01(gx, gz, 19);
                let tower = hash01(gx, gz, 23).powf(2.2);
                let h = 0.75 + base * 3.2 + tower * 4.8;
                let w = 0.72 + hash01(gx, gz, 29) * 0.22;
                let d = 0.72 + hash01(gx, gz, 31) * 0.22;
                let hue_pick = hash01(gx, gz, 37);
                let hue = if hue_pick < 0.28 {
                    33.0 + hash01(gx, gz, 41) * 35.0
                } else if hue_pick < 0.56 {
                    142.0 + hash01(gx, gz, 43) * 54.0
                } else if hue_pick < 0.80 {
                    188.0 + hash01(gx, gz, 47) * 48.0
                } else {
                    310.0 + hash01(gx, gz, 53) * 42.0
                };
                let x = gx as f64 * 1.08 - w * 0.5;
                let z = gz as f64 * 1.08 - d * 0.5;

                buildings.push(Building {
                    x,
                    z,
                    w,
                    d,
                    h,
                    hue,
                    seed: hash01(gx, gz, 59),
                    roof: (hash01(gx, gz, 61) * 4.0) as u8,
                });
            }
        }

        buildings.sort_by(|a, b| {
            (a.x + a.z)
                .partial_cmp(&(b.x + b.z))
                .unwrap_or(Ordering::Equal)
        });

        let stars = (0..180)
            .map(|i| {
                let i = i as f64;
                Star {
                    x: (hash01(i as i32, 3, 71) * WIDTH as f64).round(),
                    y: 20.0 + hash01(i as i32, 5, 73) * 420.0,
                    phase: hash01(i as i32, 7, 79) * TAU,
                    hue: 28.0 + hash01(i as i32, 11, 83) * 210.0,
                }
            })
            .collect();

        Self { buildings, stars }
    }
}

fn sky(t: f64) -> Canvas {
    Canvas::from_fn_independent_with_options(
        WIDTH,
        HEIGHT,
        move |x, y| {
            let nx = f64::from(x) / f64::from(WIDTH);
            let ny = f64::from(y) / f64::from(HEIGHT);
            let horizon = (1.0 - (ny - 0.58).abs() * 2.2).clamp(0.0, 1.0);
            let aurora = ((nx * 12.0 + ny * 7.0 + t * TAU).sin() * 0.5 + 0.5).powf(2.4);
            let amber = ((nx * 4.0 - ny * 11.0 - t * 3.0).cos() * 0.5 + 0.5).powf(5.0);
            let hue = 204.0 - 44.0 * horizon + 82.0 * aurora + 34.0 * amber;
            let light = 5.0 + 18.0 * horizon + 10.0 * aurora + 12.0 * amber;
            hsl(hue, 78, light as u16)
        },
        true,
        false,
    )
}

fn draw_sky_signals(canvas: &mut Canvas, scene: &Scene, t: f64) {
    for star in &scene.stars {
        let twinkle = ((t * TAU * 1.8 + star.phase).sin() * 0.5 + 0.5).powf(2.0);
        if twinkle > 0.18 {
            let color = hsl(star.hue, 94, (25.0 + twinkle * 58.0) as u16);
            plot_cross(
                canvas,
                star.x,
                star.y,
                -950.0,
                1 + (twinkle * 2.0) as i64,
                color,
            );
        }
    }

    for band in 0..8 {
        let phase = t * TAU + band as f64 * 0.71;
        let color = hsl(136.0 + band as f64 * 15.0, 72, 22 + band as u16);
        let mut previous = None;
        for x in (0..=WIDTH).step_by(18) {
            let xf = f64::from(x);
            let y = 120.0
                + band as f64 * 38.0
                + (xf * 0.017 + phase).sin() * 22.0
                + (xf * 0.006 - phase * 1.7).cos() * 18.0;
            if let Some((px, py)) = previous {
                canvas.draw_line_z(color.scale(0.62), (px, py, -900.0), (xf, y, -900.0));
            }
            previous = Some((xf, y));
        }
    }
}

fn draw_glass_dome(canvas: &mut Canvas, t: f64) {
    canvas.set_line_width(1.0);
    let color = hsl(180.0 + 25.0 * (t * TAU).sin(), 78, 48);
    draw_ellipse(
        canvas,
        CENTER_X,
        548.0,
        414.0,
        428.0,
        color.scale(0.82),
        1500.0,
    );
    draw_ellipse(
        canvas,
        CENTER_X,
        556.0,
        360.0,
        368.0,
        hsl(35.0, 92, 36).scale(0.55),
        1490.0,
    );

    for i in 0..9 {
        let y = 210.0 + i as f64 * 58.0 + (t * TAU + i as f64).sin() * 5.0;
        let span = (1.0 - ((y - 548.0) / 428.0).powi(2)).max(0.0).sqrt();
        let x0 = CENTER_X - 414.0 * span;
        let x1 = CENTER_X + 414.0 * span;
        canvas.draw_line_z(
            hsl(196.0 + i as f64 * 12.0, 72, 26).scale(0.58),
            (x0, y, 1480.0),
            (x1, y, 1480.0),
        );
    }
}

fn draw_weather(canvas: &mut Canvas, scene: &Scene, t: f64) {
    for (i, star) in scene.stars.iter().take(110).enumerate() {
        let fall = (t * 760.0 + i as f64 * 23.0 + star.phase * 11.0) % 760.0;
        let x = (star.x + i as f64 * 17.0 + t * 90.0) % WIDTH as f64;
        let y = 120.0 + fall;
        let color = hsl(188.0 + star.hue * 0.1, 72, 40).scale(0.52);
        canvas.draw_line_z(color, (x, y, 1540.0), (x - 16.0, y + 34.0, 1540.0));
    }
}

fn draw_reflections(canvas: &mut Canvas, scene: &Scene, t: f64) {
    for building in &scene.buildings {
        let base = project(
            WorldPoint::new(
                building.x + building.w * 0.5,
                0.0,
                building.z + building.d * 0.5,
            ),
            t,
            -140.0,
        );
        let height = 18.0 + building.h * 13.0;
        let color = hsl(building.hue, 84, 23).scale(0.46);
        canvas.draw_line_z(color, base, (base.0, base.1 + height, base.2 - 20.0));
    }
}

fn draw_island(canvas: &mut Canvas, t: f64) {
    let top = hsl(174.0, 52, 18);
    let side_a = hsl(190.0, 58, 10);
    let side_b = hsl(33.0, 80, 14);
    let rim = hsl(48.0 + 18.0 * (t * TAU).sin(), 94, 48);

    let p00 = WorldPoint::new(-6.75, 0.0, -6.75);
    let p10 = WorldPoint::new(6.75, 0.0, -6.75);
    let p11 = WorldPoint::new(6.75, 0.0, 6.75);
    let p01 = WorldPoint::new(-6.75, 0.0, 6.75);
    let b00 = WorldPoint::new(-6.3, -1.12, -6.3);
    let b10 = WorldPoint::new(6.3, -1.12, -6.3);
    let b11 = WorldPoint::new(6.3, -1.12, 6.3);
    let b01 = WorldPoint::new(-6.3, -1.12, 6.3);

    draw_world_quad(canvas, p00, p10, p11, p01, top, t, 0.0);
    draw_world_quad(canvas, p01, p11, b11, b01, side_a, t, -4.0);
    draw_world_quad(canvas, p10, p11, b11, b10, side_b, t, -3.0);
    draw_world_quad(canvas, p00, p10, b10, b00, side_a.scale(0.7), t, -6.0);
    draw_world_quad(canvas, p00, p01, b01, b00, side_b.scale(0.55), t, -7.0);

    for &(a, b) in &[(p00, p10), (p10, p11), (p11, p01), (p01, p00)] {
        draw_world_line(canvas, a, b, rim, t, 34.0);
    }
}

fn draw_city_grid(canvas: &mut Canvas, t: f64) {
    let road = hsl(178.0, 88, 38).scale(0.55);
    let arterial = hsl(34.0 + 16.0 * (t * TAU).sin(), 96, 50);

    for i in -6_i32..=6 {
        let x = i as f64 * 1.08;
        draw_world_line(
            canvas,
            WorldPoint::new(x, 0.035, -6.15),
            WorldPoint::new(x, 0.035, 6.15),
            if i == 0 { arterial } else { road },
            t,
            26.0,
        );
        let z = i as f64 * 1.08;
        draw_world_line(
            canvas,
            WorldPoint::new(-6.15, 0.035, z),
            WorldPoint::new(6.15, 0.035, z),
            if i == 0 { arterial } else { road },
            t,
            26.0,
        );
    }

    for i in 0..24 {
        let u = i as f64 / 24.0;
        let pulse = (u + t).fract();
        let x = -5.8 + 11.6 * pulse;
        let color = hsl(46.0, 100, (34.0 + 42.0 * (1.0 - u)) as u16);
        draw_world_line(
            canvas,
            WorldPoint::new(x - 0.16, 0.055, 0.0),
            WorldPoint::new(x + 0.16, 0.055, 0.0),
            color,
            t,
            55.0,
        );
    }
}

fn draw_monorail_track(canvas: &mut Canvas, t: f64) {
    let track = hsl(205.0, 80, 44);
    let glow = hsl(325.0 + 40.0 * (t * TAU).sin(), 96, 48);
    let corners = [
        WorldPoint::new(-5.8, 0.45, -5.8),
        WorldPoint::new(5.8, 0.45, -5.8),
        WorldPoint::new(5.8, 0.45, 5.8),
        WorldPoint::new(-5.8, 0.45, 5.8),
    ];
    for i in 0..4 {
        draw_world_line(canvas, corners[i], corners[(i + 1) % 4], track, t, 70.0);
    }
    for i in 0..16 {
        let u = i as f64 / 16.0;
        let p = rail_point(u);
        draw_world_line(
            canvas,
            WorldPoint::new(p.x, p.y - 0.28, p.z),
            WorldPoint::new(p.x, p.y + 0.28, p.z),
            glow.scale(0.44),
            t,
            66.0,
        );
    }
}

fn draw_city(canvas: &mut Canvas, scene: &Scene, t: f64) {
    for building in &scene.buildings {
        draw_building(canvas, *building, t);
    }
}

fn draw_building(canvas: &mut Canvas, b: Building, t: f64) {
    let face_front = hsl(b.hue + 14.0, 72, (13.0 + b.seed * 9.0) as u16);
    let face_side = hsl(b.hue - 22.0, 68, (10.0 + b.seed * 7.0) as u16);
    let face_top = hsl(b.hue + 50.0, 76, (24.0 + b.seed * 14.0) as u16);

    let x0 = b.x;
    let x1 = b.x + b.w;
    let z0 = b.z;
    let z1 = b.z + b.d;
    let y0 = 0.0;
    let y1 = b.h;

    draw_world_quad(
        canvas,
        WorldPoint::new(x0, y0, z1),
        WorldPoint::new(x1, y0, z1),
        WorldPoint::new(x1, y1, z1),
        WorldPoint::new(x0, y1, z1),
        face_front,
        t,
        4.0,
    );
    draw_world_quad(
        canvas,
        WorldPoint::new(x1, y0, z0),
        WorldPoint::new(x1, y0, z1),
        WorldPoint::new(x1, y1, z1),
        WorldPoint::new(x1, y1, z0),
        face_side,
        t,
        5.0,
    );
    draw_world_quad(
        canvas,
        WorldPoint::new(x0, y1, z0),
        WorldPoint::new(x1, y1, z0),
        WorldPoint::new(x1, y1, z1),
        WorldPoint::new(x0, y1, z1),
        face_top,
        t,
        8.0,
    );

    let edge = hsl(b.hue + 140.0, 88, 36);
    draw_world_line(
        canvas,
        WorldPoint::new(x0, y1, z1),
        WorldPoint::new(x1, y1, z1),
        edge,
        t,
        32.0,
    );
    draw_world_line(
        canvas,
        WorldPoint::new(x1, y0, z1),
        WorldPoint::new(x1, y1, z1),
        edge,
        t,
        32.0,
    );
    draw_world_line(
        canvas,
        WorldPoint::new(x1, y1, z0),
        WorldPoint::new(x1, y1, z1),
        edge,
        t,
        32.0,
    );
    draw_windows(canvas, b, t);
    draw_rooftop(canvas, b, t);
}

fn draw_windows(canvas: &mut Canvas, b: Building, t: f64) {
    let floors = (b.h / 0.36).floor().max(2.0) as usize;
    let front_cols = (b.w / 0.16).floor().max(2.0) as usize;
    let side_cols = (b.d / 0.16).floor().max(2.0) as usize;

    for floor in 1..floors {
        let y = floor as f64 / floors as f64 * b.h * 0.9;
        for col in 0..front_cols {
            let lit = ((b.seed * 17.0 + floor as f64 * 0.73 + col as f64 * 1.37 + t * TAU).sin()
                * 0.5
                + 0.5)
                > 0.34;
            if lit {
                let x = b.x + 0.12 + col as f64 / front_cols as f64 * (b.w - 0.22);
                let color = hsl(42.0 + b.hue * 0.15, 100, 54);
                draw_world_line(
                    canvas,
                    WorldPoint::new(x, y, b.z + b.d + 0.012),
                    WorldPoint::new(x + 0.075, y, b.z + b.d + 0.012),
                    color,
                    t,
                    64.0,
                );
            }
        }

        for col in 0..side_cols {
            let lit = ((b.seed * 11.0 + floor as f64 * 0.91 + col as f64 * 0.63 - t * TAU * 1.3)
                .cos()
                * 0.5
                + 0.5)
                > 0.48;
            if lit {
                let z = b.z + 0.12 + col as f64 / side_cols as f64 * (b.d - 0.22);
                let color = hsl(168.0 + b.hue * 0.08, 92, 45);
                draw_world_line(
                    canvas,
                    WorldPoint::new(b.x + b.w + 0.012, y, z),
                    WorldPoint::new(b.x + b.w + 0.012, y, z + 0.075),
                    color,
                    t,
                    64.0,
                );
            }
        }
    }
}

fn draw_rooftop(canvas: &mut Canvas, b: Building, t: f64) {
    let cx = b.x + b.w * 0.5;
    let cz = b.z + b.d * 0.5;
    let y = b.h + 0.04;
    let color = hsl(b.hue + 95.0, 96, 44);

    match b.roof {
        0 => {
            let r = 0.12 + b.seed * 0.12;
            draw_world_line(
                canvas,
                WorldPoint::new(cx - r, y, cz),
                WorldPoint::new(cx + r, y, cz),
                color,
                t,
                76.0,
            );
            draw_world_line(
                canvas,
                WorldPoint::new(cx, y, cz - r),
                WorldPoint::new(cx, y, cz + r),
                color,
                t,
                76.0,
            );
        }
        1 => {
            draw_world_line(
                canvas,
                WorldPoint::new(cx, y, cz),
                WorldPoint::new(cx, y + 0.7 + b.seed * 0.8, cz),
                color,
                t,
                88.0,
            );
        }
        2 => {
            for i in 0..5 {
                let a = i as f64 / 5.0 * TAU + t * TAU * 0.25;
                let p = WorldPoint::new(cx + a.cos() * 0.18, y, cz + a.sin() * 0.18);
                draw_world_line(
                    canvas,
                    WorldPoint::new(cx, y + 0.18, cz),
                    p,
                    color.scale(0.8),
                    t,
                    84.0,
                );
            }
        }
        _ => {}
    }
}

fn draw_central_loom(canvas: &mut Canvas, t: f64) {
    let body = Building {
        x: -0.48,
        z: -0.48,
        w: 0.96,
        d: 0.96,
        h: 7.7,
        hue: 48.0,
        seed: 0.8,
        roof: 3,
    };
    draw_building(canvas, body, t);

    let crown_y = 8.15 + 0.22 * (t * TAU).sin();
    for blade in 0..10 {
        let a = blade as f64 / 10.0 * TAU + t * TAU * 0.33;
        let radius = 0.78 + 0.12 * (t * TAU + blade as f64).sin();
        let p0 = WorldPoint::new(a.cos() * 0.18, crown_y, a.sin() * 0.18);
        let p1 = WorldPoint::new(a.cos() * radius, crown_y + 0.42, a.sin() * radius);
        let p2 = WorldPoint::new(
            (a + 0.25).cos() * radius,
            crown_y + 0.12,
            (a + 0.25).sin() * radius,
        );
        let p3 = WorldPoint::new(
            (a + 0.25).cos() * 0.18,
            crown_y - 0.12,
            (a + 0.25).sin() * 0.18,
        );
        draw_world_quad(
            canvas,
            p0,
            p1,
            p2,
            p3,
            hsl(184.0 + blade as f64 * 13.0, 94, 33),
            t,
            120.0,
        );
        draw_world_line(canvas, p0, p1, hsl(44.0, 100, 60), t, 170.0);
    }

    for i in 0..26 {
        let u = i as f64 / 25.0;
        let a = u * TAU * 2.4 + t * TAU;
        let r = 0.25 + u * 1.1;
        let y = 1.0 + u * 6.7;
        let p0 = WorldPoint::new(a.cos() * r, y, a.sin() * r);
        let p1 = WorldPoint::new((a + 0.42).cos() * r, y + 0.24, (a + 0.42).sin() * r);
        draw_world_line(canvas, p0, p1, hsl(300.0 - u * 120.0, 100, 55), t, 145.0);
    }
}

fn draw_light_threads(canvas: &mut Canvas, scene: &Scene, t: f64) {
    let origin = WorldPoint::new(0.0, 7.9, 0.0);
    for (i, building) in scene.buildings.iter().enumerate().step_by(3) {
        if i % 2 == 0 && building.h < 2.2 {
            continue;
        }
        let target = WorldPoint::new(
            building.x + building.w * 0.5,
            building.h + 0.16,
            building.z + building.d * 0.5,
        );
        let phase = t * TAU + building.seed * TAU + i as f64 * 0.17;
        let color = hsl(building.hue + 70.0, 96, 44);
        let mut previous = origin;
        for step in 1..=18 {
            let u = step as f64 / 18.0;
            let lift = (PI * u).sin() * (1.5 + building.seed * 1.2);
            let swirl = (phase + u * TAU * 1.5).sin() * 0.18 * (1.0 - u);
            let point = WorldPoint::new(
                lerp(origin.x, target.x, u) + swirl,
                lerp(origin.y, target.y, u) + lift,
                lerp(origin.z, target.z, u) + (phase + u * TAU).cos() * 0.18 * (1.0 - u),
            );
            draw_world_line(
                canvas,
                previous,
                point,
                color.scale(0.74 + 0.24 * (1.0 - u)),
                t,
                115.0,
            );
            previous = point;
        }
    }
}

fn draw_floating_panes(canvas: &mut Canvas, t: f64) {
    for i in 0..18 {
        let u = i as f64 / 18.0;
        let a = u * TAU + t * TAU * (0.08 + u * 0.04);
        let radius = 4.2 + (i % 5) as f64 * 0.34;
        let cx = a.cos() * radius;
        let cz = a.sin() * radius;
        let cy = 3.6 + (i % 7) as f64 * 0.72 + 0.35 * (t * TAU + i as f64).sin();
        let w = 0.34 + (i % 3) as f64 * 0.1;
        let h = 0.72 + (i % 4) as f64 * 0.18;
        let hue = 178.0 + i as f64 * 17.0;
        let p0 = WorldPoint::new(cx - w, cy - h, cz);
        let p1 = WorldPoint::new(cx + w, cy - h * 0.82, cz + 0.18);
        let p2 = WorldPoint::new(cx + w, cy + h, cz + 0.18);
        let p3 = WorldPoint::new(cx - w, cy + h * 0.82, cz);
        draw_world_quad(
            canvas,
            p0,
            p1,
            p2,
            p3,
            hsl(hue, 74, 18 + (i % 5) as u16),
            t,
            104.0,
        );
        draw_world_line(canvas, p0, p1, hsl(hue, 100, 50), t, 170.0);
        draw_world_line(canvas, p1, p2, hsl(hue, 100, 42), t, 170.0);
        draw_world_line(canvas, p2, p3, hsl(hue, 100, 50), t, 170.0);
        draw_world_line(canvas, p3, p0, hsl(hue, 100, 42), t, 170.0);
    }
}

fn draw_monorail_train(canvas: &mut Canvas, t: f64) {
    for car in 0..5 {
        let u = (t * 0.68 + car as f64 * 0.013) % 1.0;
        let p = rail_point(u);
        let next = rail_point((u + 0.004) % 1.0);
        let heading_x = (next.x - p.x).signum();
        let heading_z = (next.z - p.z).signum();
        let car_color = hsl(30.0 + car as f64 * 18.0, 100, 56);
        let width = if heading_x.abs() > heading_z.abs() {
            0.34
        } else {
            0.18
        };
        let depth = if heading_z.abs() >= heading_x.abs() {
            0.34
        } else {
            0.18
        };
        draw_box(
            canvas,
            WorldPoint::new(p.x - width * 0.5, p.y, p.z - depth * 0.5),
            width,
            0.26,
            depth,
            car_color,
            t,
        );
    }
}

fn draw_foreground_gleam(canvas: &mut Canvas, t: f64) {
    for i in 0..28 {
        let a = i as f64 / 28.0 * TAU + t * TAU * 0.4;
        let radius = 302.0 + (i % 6) as f64 * 16.0;
        let x = CENTER_X + a.cos() * radius;
        let y = 650.0 + a.sin() * 98.0;
        let color = hsl(42.0 + i as f64 * 8.0, 96, 48);
        canvas.draw_line_z(
            color.scale(0.6),
            (x - 14.0, y, 1620.0),
            (x + 14.0, y, 1620.0),
        );
    }
}

fn draw_box(
    canvas: &mut Canvas,
    origin: WorldPoint,
    width: f64,
    height: f64,
    depth: f64,
    color: Rgb,
    t: f64,
) {
    let x0 = origin.x;
    let x1 = origin.x + width;
    let y0 = origin.y;
    let y1 = origin.y + height;
    let z0 = origin.z;
    let z1 = origin.z + depth;
    draw_world_quad(
        canvas,
        WorldPoint::new(x0, y0, z1),
        WorldPoint::new(x1, y0, z1),
        WorldPoint::new(x1, y1, z1),
        WorldPoint::new(x0, y1, z1),
        color.scale(0.68),
        t,
        8.0,
    );
    draw_world_quad(
        canvas,
        WorldPoint::new(x1, y0, z0),
        WorldPoint::new(x1, y0, z1),
        WorldPoint::new(x1, y1, z1),
        WorldPoint::new(x1, y1, z0),
        color.scale(0.54),
        t,
        8.0,
    );
    draw_world_quad(
        canvas,
        WorldPoint::new(x0, y1, z0),
        WorldPoint::new(x1, y1, z0),
        WorldPoint::new(x1, y1, z1),
        WorldPoint::new(x0, y1, z1),
        color,
        t,
        12.0,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_world_quad(
    canvas: &mut Canvas,
    a: WorldPoint,
    b: WorldPoint,
    c: WorldPoint,
    d: WorldPoint,
    color: Rgb,
    t: f64,
    depth_bias: f64,
) {
    let a = project(a, t, depth_bias);
    let b = project(b, t, depth_bias);
    let c = project(c, t, depth_bias);
    let d = project(d, t, depth_bias);
    canvas.draw_triangle(color, a, b, c);
    canvas.draw_triangle(color, a, c, d);
}

fn draw_world_line(
    canvas: &mut Canvas,
    a: WorldPoint,
    b: WorldPoint,
    color: Rgb,
    t: f64,
    depth_bias: f64,
) {
    let a = project(a, t, depth_bias);
    let b = project(b, t, depth_bias);
    canvas.draw_line_z(color, a, b);
}

fn project(point: WorldPoint, t: f64, depth_bias: f64) -> (f64, f64, f64) {
    let yaw = 0.12 * (t * TAU).sin();
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    let x = point.x * cos_yaw - point.z * sin_yaw;
    let z = point.x * sin_yaw + point.z * cos_yaw;
    let sx = CENTER_X + (x - z) * ISO_SCALE;
    let sy = BASE_Y + (x + z) * ISO_SCALE * 0.49 - point.y * ISO_SCALE;
    let depth = (x + z) * 122.0 + point.y * 21.0 + depth_bias;
    (sx, sy, depth)
}

fn rail_point(u: f64) -> WorldPoint {
    let u = u.fract();
    let side = (u * 4.0).floor() as usize;
    let local = u * 4.0 - side as f64;
    let y = 0.62;
    match side {
        0 => WorldPoint::new(lerp(-5.8, 5.8, local), y, -5.8),
        1 => WorldPoint::new(5.8, y, lerp(-5.8, 5.8, local)),
        2 => WorldPoint::new(lerp(5.8, -5.8, local), y, 5.8),
        _ => WorldPoint::new(-5.8, y, lerp(5.8, -5.8, local)),
    }
}

fn draw_ellipse(canvas: &mut Canvas, cx: f64, cy: f64, rx: f64, ry: f64, color: Rgb, z: f64) {
    let mut previous = (cx + rx, cy, z);
    for i in 1..=192 {
        let a = i as f64 / 192.0 * TAU;
        let next = (cx + a.cos() * rx, cy + a.sin() * ry, z);
        canvas.draw_line_z(color, previous, next);
        previous = next;
    }
}

fn plot_cross(canvas: &mut Canvas, x: f64, y: f64, z: f64, radius: i64, color: Rgb) {
    let x = x.round() as i64;
    let y = y.round() as i64;
    canvas.plot_z(&color, x, y, z);
    for d in 1..=radius {
        let color = color.scale(0.72);
        canvas.plot_z(&color, x + d, y, z);
        canvas.plot_z(&color, x - d, y, z);
        canvas.plot_z(&color, x, y + d, z);
        canvas.plot_z(&color, x, y - d, z);
    }
}

fn hsl(hue: f64, saturation: u16, light: u16) -> Rgb {
    Rgb::from_hsl_f64(hue, f64::from(saturation) / 100.0, f64::from(light) / 100.0)
}
