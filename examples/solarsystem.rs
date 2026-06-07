use gartus::gmath::vector::Vector;
use gartus::prelude::*;
use std::{error::Error, f64::consts::PI, fs};

const WIDTH: u32 = 1080;
const HEIGHT: u32 = 1080;
const FRAMES: usize = 120;
const CENTER_X: f64 = WIDTH as f64 * 0.5;
const CENTER_Y: f64 = HEIGHT as f64 * 0.5;
const ORBIT_TILT_DEG: f64 = 16.0;
// Vertical foreshortening of orbits. Higher = rounder orbits that swing the
// front/back of each ring clear of the sun's glow instead of diving through it.
// At 0.6 every orbit's closest screen approach (R * ORBIT_SQUISH) stays outside
// SUN_GLOW_RADIUS, so planets arc around the sun rather than smearing through it.
const ORBIT_SQUISH: f64 = 0.6;
// Outer edge of the sun's glow disk. Both the corona rings and every planet
// orbit are kept clear of this radius.
const SUN_GLOW_RADIUS: i64 = 112;
const SUN_CORE_RADIUS: f64 = 72.0;

// Habitable zone centered on Terra's orbit.
// Gaussian falloff: sigma=82 → Caldus(290)~67%, Borial(430)~67%, Pyralis(215)~21%, Glacius(488)~30%
const HABITABLE_CENTER: f64 = 360.0;
const HABITABLE_SIGMA: f64 = 82.0;

struct Assets {
    sphere: PolygonMatrix,
    ring_mesh: PolygonMatrix,
    moon_mesh: PolygonMatrix,
    stars: Vec<Star>,
}

#[derive(Clone, Copy)]
struct Star {
    x: i64,
    y: i64,
    z: f64,
    color: Rgb,
}

#[derive(Clone, Copy)]
struct Material {
    line: Rgb,
    ambient: ReflectionConstants,
    diffuse: ReflectionConstants,
    specular: ReflectionConstants,
    specular_exponent: u32,
}

#[derive(Clone, Copy)]
struct Ring {
    scale: f64,
    tilt_x: f64,
    tilt_y: f64,
    mat: Material,
}

#[derive(Clone, Copy)]
struct Planet {
    orbit_radius: f64,
    depth_radius: f64,
    base_radius: f64,
    speed: f64,
    phase: f64,
    dead: Material,
    alive: Material,
    atmo_color: Option<Rgb>,
    ring: Option<Ring>,
    moons: usize,
}

struct PlanetFrame {
    planet: Planet,
    screen_x: f64,
    screen_y: f64,
    depth_z: f64,
    angle: f64,
    life: f64,
}

fn main() {
    if let Err(err) = render() {
        eprintln!("could not render solar system:\n{err}");
        std::process::exit(1);
    }
}

fn render() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final")?;
    let assets = build_assets();
    let planets = build_planets();
    let mut recorder = FrameRecorder::new("anim", "solarsystem-").with_delay(3);
    let mut preview = Canvas::new_with_bg(WIDTH, HEIGHT, Rgb::BLACK);

    for frame in 0..FRAMES {
        let canvas = render_frame(frame, &assets, &planets);
        if frame == 30 {
            preview = canvas.clone();
        }
        recorder.capture(&canvas)?;
    }

    preview.save_extension("final/solarsystem.png")?;
    recorder.encode_gif("final/solarsystem.gif")?;
    println!("Saved final/solarsystem.png and final/solarsystem.gif");
    Ok(())
}

// ── maths ─────────────────────────────────────────────────────────────────────

fn life_factor(orbit_radius: f64) -> f64 {
    let d = (orbit_radius - HABITABLE_CENTER) / HABITABLE_SIGMA;
    (-0.5 * d * d).exp()
}

fn lerp_u8(a: u8, b: u8, t: f64) -> u8 {
    (a as f64 + (b as f64 - a as f64) * t.clamp(0.0, 1.0)) as u8
}

fn lerp_rgb(a: Rgb, b: Rgb, t: f64) -> Rgb {
    Rgb::new(
        lerp_u8(a.red, b.red, t),
        lerp_u8(a.green, b.green, t),
        lerp_u8(a.blue, b.blue, t),
    )
}

fn lerp_rc(a: ReflectionConstants, b: ReflectionConstants, t: f64) -> ReflectionConstants {
    let t = t.clamp(0.0, 1.0);
    ReflectionConstants::new(
        a.red + (b.red - a.red) * t,
        a.green + (b.green - a.green) * t,
        a.blue + (b.blue - a.blue) * t,
    )
}

fn lerp_material(dead: Material, alive: Material, t: f64) -> Material {
    Material {
        line: lerp_rgb(dead.line, alive.line, t),
        ambient: lerp_rc(dead.ambient, alive.ambient, t),
        diffuse: lerp_rc(dead.diffuse, alive.diffuse, t),
        specular: lerp_rc(dead.specular, alive.specular, t),
        specular_exponent: (dead.specular_exponent as f64
            + (alive.specular_exponent as f64 - dead.specular_exponent as f64) * t.clamp(0.0, 1.0))
            as u32,
    }
}

fn lcg(seed: u64) -> u64 {
    seed.wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

// ── assets ────────────────────────────────────────────────────────────────────

fn build_assets() -> Assets {
    let mut sphere = PolygonMatrix::new();
    sphere.add_sphere((0.0, 0.0, 0.0), 1.0, 64);

    let mut ring_mesh = PolygonMatrix::new();
    ring_mesh.add_torus((0.0, 0.0, 0.0), 2.3, 30.0, 52);

    let mut moon_mesh = PolygonMatrix::new();
    moon_mesh.add_sphere((0.0, 0.0, 0.0), 1.0, 24);

    Assets {
        sphere,
        ring_mesh,
        moon_mesh,
        stars: build_stars(),
    }
}

fn build_stars() -> Vec<Star> {
    let mut stars = Vec::new();
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    for i in 0..200u32 {
        seed = lcg(seed);
        let x = (seed % u64::from(WIDTH)) as i64;
        seed = lcg(seed);
        let y = (seed % u64::from(HEIGHT)) as i64;
        seed = lcg(seed);
        let z = -500.0 - f64::from((seed % 200) as u16);
        let bright = 28 + (seed % 95) as u8;
        let color = if i % 11 == 0 {
            Rgb::new(55, 95, 165)
        } else if i % 19 == 0 {
            Rgb::new(195, 75, 38)
        } else if i % 31 == 0 {
            Rgb::new(215, 215, 170)
        } else {
            Rgb::new(bright, bright, bright.saturating_add(16))
        };
        stars.push(Star { x, y, z, color });
    }
    stars
}

// ── materials ─────────────────────────────────────────────────────────────────

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

fn mat_dead(line: Rgb) -> Material {
    mat(
        line,
        (0.04, 0.04, 0.04),
        (0.38, 0.38, 0.38),
        (0.06, 0.06, 0.07),
        7,
    )
}

fn mat_pyralis_alive() -> Material {
    mat(
        Rgb::new(208, 68, 12),
        (0.14, 0.03, 0.00),
        (0.68, 0.22, 0.03),
        (0.55, 0.18, 0.04),
        18,
    )
}

fn mat_caldus_alive() -> Material {
    mat(
        Rgb::new(235, 215, 158),
        (0.12, 0.10, 0.04),
        (0.76, 0.68, 0.44),
        (0.56, 0.50, 0.32),
        32,
    )
}

fn mat_terra_alive() -> Material {
    mat(
        Rgb::new(42, 155, 215),
        (0.01, 0.08, 0.14),
        (0.18, 0.60, 0.86),
        (0.20, 0.42, 0.68),
        22,
    )
}

fn mat_borial_alive() -> Material {
    mat(
        Rgb::new(178, 72, 40),
        (0.10, 0.03, 0.01),
        (0.62, 0.28, 0.12),
        (0.28, 0.12, 0.06),
        14,
    )
}

fn mat_glacius_alive() -> Material {
    mat(
        Rgb::new(138, 190, 235),
        (0.08, 0.10, 0.14),
        (0.42, 0.58, 0.84),
        (0.50, 0.64, 0.80),
        50,
    )
}

fn mat_glacius_ring() -> Material {
    mat(
        Rgb::new(160, 195, 230),
        (0.10, 0.12, 0.16),
        (0.35, 0.46, 0.62),
        (0.55, 0.65, 0.78),
        60,
    )
}

// ── planets ───────────────────────────────────────────────────────────────────

fn build_planets() -> Vec<Planet> {
    vec![
        Planet {
            // Pyralis — scorched, too close
            orbit_radius: 215.0,
            depth_radius: 110.0,
            base_radius: 18.0,
            speed: 3.2,
            phase: 0.8,
            dead: mat_dead(Rgb::new(32, 30, 28)),
            alive: mat_pyralis_alive(),
            atmo_color: None,
            ring: None,
            moons: 0,
        },
        Planet {
            // Caldus — warm, inner edge of habitable zone
            orbit_radius: 290.0,
            depth_radius: 150.0,
            base_radius: 26.0,
            speed: 2.0,
            phase: 2.1,
            dead: mat_dead(Rgb::new(55, 52, 46)),
            alive: mat_caldus_alive(),
            atmo_color: None,
            ring: None,
            moons: 0,
        },
        Planet {
            // Terra — the one the sun chose
            orbit_radius: 360.0,
            depth_radius: 168.0,
            base_radius: 38.0,
            speed: 1.0,
            phase: 0.3,
            dead: mat_dead(Rgb::new(86, 92, 100)),
            alive: mat_terra_alive(),
            atmo_color: Some(Rgb::new(55, 170, 230)),
            ring: None,
            moons: 1,
        },
        Planet {
            // Borial — cooling, outer edge
            orbit_radius: 430.0,
            depth_radius: 200.0,
            base_radius: 30.0,
            speed: 0.65,
            phase: 3.5,
            dead: mat_dead(Rgb::new(44, 40, 38)),
            alive: mat_borial_alive(),
            atmo_color: None,
            ring: None,
            moons: 2,
        },
        Planet {
            // Glacius — frozen giant, barely touched by the sun
            orbit_radius: 488.0,
            depth_radius: 228.0,
            base_radius: 44.0,
            speed: 0.38,
            phase: 1.8,
            dead: mat_dead(Rgb::new(16, 16, 24)),
            alive: mat_glacius_alive(),
            atmo_color: None,
            ring: Some(Ring {
                scale: 0.90,
                tilt_x: 76.0,
                tilt_y: 28.0,
                mat: mat_glacius_ring(),
            }),
            moons: 0,
        },
    ]
}

// ── rendering ─────────────────────────────────────────────────────────────────

fn render_frame(frame: usize, assets: &Assets, planets: &[Planet]) -> Canvas {
    let mut canvas = Canvas::new_with_bg(WIDTH, HEIGHT, Rgb::BLACK);
    canvas.set_wrapped(false);
    canvas.set_line_width(1.0);
    canvas.set_shading_mode(ShadingMode::Phong);
    canvas.set_polygon_color_mode(PolygonColorMode::PhongReflection);
    canvas.set_lighting(scene_lighting());

    let t = frame as f64 / FRAMES as f64;

    draw_stars(&mut canvas, &assets.stars, t);
    draw_orbit_paths(&mut canvas, planets);
    draw_sun_glow(&mut canvas, t);
    draw_emissive_sun(&mut canvas, t);
    draw_sun_ring(&mut canvas, t);
    draw_planets(&mut canvas, assets, planets, frame);
    canvas
}

fn scene_lighting() -> Lighting {
    Lighting {
        ambient: Rgb::new(4, 4, 6),
        point_lights: vec![
            PointLight::positional(
                Vector::new(CENTER_X, CENTER_Y, 180.0),
                Rgb::new(255, 248, 210),
            )
            .with_inverse_linear_attenuation(1100.0),
        ],
        ambient_reflection: ReflectionConstants::new(0.05, 0.05, 0.06),
        diffuse_reflection: ReflectionConstants::new(0.82, 0.78, 0.68),
        specular_reflection: ReflectionConstants::new(0.18, 0.15, 0.10),
        specular_exponent: 10,
        ..Lighting::default()
    }
}

// ── starfield ─────────────────────────────────────────────────────────────────

fn draw_stars(canvas: &mut Canvas, stars: &[Star], t: f64) {
    for (idx, star) in stars.iter().enumerate() {
        let twinkle = ((idx as f64 * 1.37 + t * PI * 2.0).sin() + 1.0) * 0.5;
        if twinkle > 0.38 {
            canvas.plot_z(&star.color, star.x, star.y, star.z);
        }
    }
}

// ── orbit paths ───────────────────────────────────────────────────────────────

fn orbit_screen_xy(angle: f64, orbit_r: f64, tilt: f64) -> (f64, f64) {
    let lx = angle.cos() * orbit_r;
    let ly = angle.sin() * orbit_r * ORBIT_SQUISH;
    let x = CENTER_X + lx * tilt.cos() - ly * tilt.sin();
    let y = CENTER_Y + lx * tilt.sin() + ly * tilt.cos();
    (x, y)
}

fn draw_orbit_paths(canvas: &mut Canvas, planets: &[Planet]) {
    let tilt = ORBIT_TILT_DEG.to_radians();
    let steps = 240usize;
    for planet in planets {
        let c = planet.alive.line;
        let color = Rgb::new(c.red / 14, c.green / 14, c.blue / 14);
        for i in 0..steps {
            if i % 6 != 0 {
                continue;
            }
            let angle = i as f64 / steps as f64 * PI * 2.0;
            let (x, y) = orbit_screen_xy(angle, planet.orbit_radius, tilt);
            canvas.plot_z(&color, x as i64, y as i64, -350.0);
        }
    }
}

// ── sun ───────────────────────────────────────────────────────────────────────

fn draw_sun_glow(canvas: &mut Canvas, t: f64) {
    let pulse = 0.5 + 0.5 * (t * PI * 2.0).sin();
    let span = (SUN_GLOW_RADIUS - 78) as f64;
    for ring in (78_i64..=SUN_GLOW_RADIUS).rev() {
        let warmth = 1.0 - (ring - 78) as f64 / span;
        let color = Rgb::new(
            (44.0 + 88.0 * warmth + 20.0 * pulse) as u8,
            (16.0 + 44.0 * warmth + 12.0 * pulse) as u8,
            (1.0 + 8.0 * warmth) as u8,
        );
        draw_sun_disk(canvas, ring, color, -82.0 - ring as f64 * 0.01);
    }
}

// Three glowing golden rings wrap the sun on different great-circle planes. They
// share the same projected poles, avoiding a triangular tangle in the middle.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn draw_sun_ring(canvas: &mut Canvas, t: f64) {
    let bands = [
        SunRingBand {
            vertical_radius: 78.0,
            width_factor: 0.34,
            depth_amp: 58.0,
            tube: 3.0,
            phase_offset: 0.0,
            core: Rgb::new(255, 242, 190),
            edge: Rgb::new(245, 145, 20),
        },
        SunRingBand {
            vertical_radius: 86.0,
            width_factor: 0.58,
            depth_amp: 64.0,
            tube: 2.0,
            phase_offset: 0.48,
            core: Rgb::new(255, 226, 130),
            edge: Rgb::new(220, 96, 10),
        },
        SunRingBand {
            vertical_radius: 94.0,
            width_factor: 0.78,
            depth_amp: 70.0,
            tube: 2.0,
            phase_offset: 0.96,
            core: Rgb::new(255, 198, 86),
            edge: Rgb::new(152, 58, 5),
        },
    ];

    for band in bands {
        draw_sun_ring_band(canvas, t, band);
    }
}

#[derive(Clone, Copy)]
struct SunRingBand {
    vertical_radius: f64,
    width_factor: f64,
    depth_amp: f64,
    tube: f64,
    phase_offset: f64,
    core: Rgb,
    edge: Rgb,
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn draw_sun_ring_band(canvas: &mut Canvas, t: f64, band: SunRingBand) {
    let breath = 1.0 + 0.02 * (t * PI * 2.0).sin();
    let ry = band.vertical_radius * breath;
    let rx = ry * band.width_factor;
    let base_z = 42.0;
    let spin = t * PI * 2.0 + band.phase_offset;

    let steps = 480;
    let tr = band.tube.ceil() as i64;
    for i in 0..steps {
        let theta = i as f64 / steps as f64 * PI * 2.0;
        let phase = theta + spin * 0.22;
        let cx = CENTER_X + phase.cos() * rx;
        let cy = CENTER_Y + theta.sin() * ry;
        let z = base_z + (theta + spin * 0.22 + band.phase_offset).sin() * band.depth_amp;
        // shimmer travels around the ring; always >= 0.7 so it is always lit
        let shimmer = 0.78 + 0.22 * (theta * 3.0 - spin).sin();
        for dy in -tr..=tr {
            for dx in -tr..=tr {
                let d = ((dx * dx + dy * dy) as f64).sqrt();
                if d > band.tube {
                    continue;
                }
                let glow = (1.0 - d / band.tube).powf(1.3) * shimmer;
                if glow < 0.12 {
                    continue;
                }
                let px = cx + dx as f64;
                let py = cy + dy as f64;
                let sun_dx = px - CENTER_X;
                let sun_dy = py - CENTER_Y;
                if sun_dx * sun_dx + sun_dy * sun_dy < SUN_CORE_RADIUS * SUN_CORE_RADIUS {
                    continue;
                }
                let color = lerp_rgb(band.edge, band.core, glow.min(1.0));
                canvas.plot_z(&color, px as i64, py as i64, z);
            }
        }
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn draw_emissive_sun(canvas: &mut Canvas, time: f64) {
    let r = SUN_CORE_RADIUS as i64;
    let spin = time * PI * 2.0 * 1.8;
    for y in -r..=r {
        for x in -r..=r {
            let dist = ((x * x + y * y) as f64).sqrt();
            if dist > r as f64 {
                continue;
            }
            let t = 1.0 - dist / r as f64;
            let hot = t.powf(0.42);
            let nx = x as f64 / r as f64;
            let ny = y as f64 / r as f64;
            let nz = (1.0 - nx * nx - ny * ny).max(0.0).sqrt();
            let longitude = nx.atan2(nz) + spin;
            let latitude = ny.asin();
            let banding = 0.5 + 0.5 * (longitude * 7.0 + latitude.sin() * 3.5).sin();
            let swirl = 0.5 + 0.5 * (longitude * 2.0 - spin * 1.4 + latitude * 5.0).sin();
            let spot_lon = (longitude - spin * 0.45).cos();
            let spot_lat = (latitude + 0.25).cos();
            let moving_spot = (spot_lon * spot_lat).max(0.0).powf(18.0);
            let filament = (banding.powf(2.2) * 0.75 + swirl.powf(3.0) * 0.45) * (0.35 + 0.65 * t);
            let color = Rgb::new(
                (220.0 + 31.0 * hot + 34.0 * filament - 26.0 * moving_spot).clamp(0.0, 255.0) as u8,
                (105.0 + 136.0 * hot + 28.0 * filament - 38.0 * moving_spot).clamp(0.0, 255.0)
                    as u8,
                (5.0 + 116.0 * hot + 10.0 * filament - 8.0 * moving_spot).clamp(0.0, 255.0) as u8,
            );
            canvas.plot_z(
                &color,
                CENTER_X as i64 + x,
                CENTER_Y as i64 + y,
                42.0 - dist * 0.02,
            );
        }
    }
    draw_sun_disk(canvas, 10, Rgb::new(255, 252, 215), 44.0);
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn draw_sun_disk(canvas: &mut Canvas, radius: i64, color: Rgb, z: f64) {
    for y in -radius..=radius {
        for x in -radius..=radius {
            if x * x + y * y <= radius * radius {
                canvas.plot_z(&color, CENTER_X as i64 + x, CENTER_Y as i64 + y, z);
            }
        }
    }
}

// ── planets ───────────────────────────────────────────────────────────────────

fn draw_planets(canvas: &mut Canvas, assets: &Assets, planets: &[Planet], frame: usize) {
    let t = frame as f64 / FRAMES as f64;
    let tilt = ORBIT_TILT_DEG.to_radians();

    let mut frames: Vec<PlanetFrame> = planets
        .iter()
        .map(|p| {
            let angle = t * PI * 2.0 * p.speed + p.phase;
            let (sx, sy) = orbit_screen_xy(angle, p.orbit_radius, tilt);
            let depth_z = angle.sin() * p.depth_radius;
            PlanetFrame {
                planet: *p,
                screen_x: sx,
                screen_y: sy,
                depth_z,
                angle,
                life: life_factor(p.orbit_radius),
            }
        })
        .collect();

    frames.sort_by(|a, b| {
        a.depth_z
            .partial_cmp(&b.depth_z)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for (idx, pf) in frames.iter().enumerate() {
        let spin = t * 360.0 * (1.0 + idx as f64 * 0.22);
        let mat = lerp_material(pf.planet.dead, pf.planet.alive, pf.life);
        let radius = pf.planet.base_radius * (1.0 + 0.12 * pf.life);

        draw_lit_mesh(
            canvas,
            &assets.sphere,
            Matrix::translate(pf.screen_x, pf.screen_y, pf.depth_z)
                * Matrix::rotate_y(spin)
                * Matrix::rotate_x(12.0 + idx as f64 * 7.0)
                * Matrix::scale(radius, radius, radius),
            mat,
        );

        if let Some(ring) = pf.planet.ring {
            draw_lit_mesh(
                canvas,
                &assets.ring_mesh,
                Matrix::translate(pf.screen_x, pf.screen_y, pf.depth_z + 1.0)
                    * Matrix::rotate_x(ring.tilt_x)
                    * Matrix::rotate_y(ring.tilt_y + t * 60.0)
                    * Matrix::scale(ring.scale, ring.scale, ring.scale),
                ring.mat,
            );
        }

        // atmosphere halo drawn after planet so z-buffer clips interior correctly
        if let Some(atmo) = pf.planet.atmo_color {
            draw_atmosphere_halo(canvas, pf, atmo, radius);
        }

        draw_moons(canvas, assets, pf, t);
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn draw_atmosphere_halo(canvas: &mut Canvas, pf: &PlanetFrame, color: Rgb, planet_r: f64) {
    if pf.life < 0.25 {
        return;
    }
    let cx = pf.screen_x as i64;
    let cy = pf.screen_y as i64;
    let inner = planet_r as i64;
    let outer = (planet_r * 1.48 * pf.life.sqrt()) as i64;
    if outer <= inner {
        return;
    }

    for dy in -outer..=outer {
        for dx in -outer..=outer {
            let d = ((dx * dx + dy * dy) as f64).sqrt();
            if d > outer as f64 || d < inner as f64 {
                continue;
            }
            let fade = 1.0 - (d - inner as f64) / (outer - inner) as f64;
            let brightness = fade * fade * pf.life * 0.52;
            let c = Rgb::new(
                (color.red as f64 * brightness) as u8,
                (color.green as f64 * brightness) as u8,
                (color.blue as f64 * brightness) as u8,
            );
            // z behind planet surface so sphere body occludes center
            canvas.plot_z(&c, cx + dx, cy + dy, pf.depth_z + planet_r * 0.35);
        }
    }
}

fn draw_moons(canvas: &mut Canvas, assets: &Assets, pf: &PlanetFrame, t: f64) {
    for moon_idx in 0..pf.planet.moons {
        let ma = t * PI * 2.0 * (3.0 + moon_idx as f64 * 0.8) + pf.angle;
        let dist = pf.planet.base_radius + 18.0 + moon_idx as f64 * 10.0;
        let mx = pf.screen_x + ma.cos() * dist;
        let my = pf.screen_y + ma.sin() * dist * 0.45;
        let mz = pf.depth_z + ma.sin() * dist * 0.55 + 3.0;
        let mr = 5.0 + moon_idx as f64 * 1.5;
        draw_lit_mesh(
            canvas,
            &assets.moon_mesh,
            Matrix::translate(mx, my, mz) * Matrix::scale(mr, mr, mr),
            mat(
                Rgb::new(195, 208, 224),
                (0.05, 0.05, 0.05),
                (0.40, 0.40, 0.42),
                (0.03, 0.03, 0.04),
                7,
            ),
        );
    }
}

fn draw_lit_mesh(canvas: &mut Canvas, mesh: &PolygonMatrix, transform: Matrix, material: Material) {
    canvas.set_line_pixel(material.line);
    {
        let l = canvas.lighting_mut();
        l.ambient_reflection = material.ambient;
        l.diffuse_reflection = material.diffuse;
        l.specular_reflection = material.specular;
        l.specular_exponent = material.specular_exponent;
    }
    canvas.draw_polygons(&mesh.apply(&transform));
}
