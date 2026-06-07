//! This example uses `gartus`'s path-tracing engine to render a looping
//! animation of fireflies over a lake at night:
//! 1. A dark, semi-reflective water surface (Metal with light fuzz for ripples).
//! 2. A large warm moon low on the horizon casting a soft light path across the water.
//! 3. Reeds and grass (silhouetted boxes) clustered along the banks of the lake.
//! 4. 36 fireflies moving in organic sinusoidal 3D paths, flashing and pulsing their
//!    greenish-yellow light intensities dynamically.
//! 5. A low-angle camera that slowly drifts across the water, creating deep parallax
//!    between the reeds, fireflies, and moon.
//!
//! Outputs a GIF and a PNG preview to `final/raytracing/`.

use gartus::prelude::*;
use std::{error::Error, f64::consts::PI, fs, sync::Arc};

const IMAGE_WIDTH: u32 = 400;
const STRATIFIED_GRID_WIDTH: u32 = 14; // 14x14 = 196 samples per pixel for a clean final render
const MAX_DEPTH: u32 = 6; // low depth is sufficient for diffuse volumes/mirrors, rendering much faster
const FRAMES: usize = 24;

struct Firefly {
    base_x: f64,
    base_y: f64,
    base_z: f64,
    speed_x: f64,
    speed_y: f64,
    speed_z: f64,
    phase_x: f64,
    phase_y: f64,
    phase_z: f64,
    flash_speed: f64,
    flash_phase: f64,
}

struct Reed {
    x: f64,
    z: f64,
    height: f64,
}

fn main() {
    if let Err(err) = render() {
        eprintln!("could not render firefly lake:\n{err}");
        std::process::exit(1);
    }
}

fn render() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final/raytracing")?;

    let options = AnimationRenderOptions::new(
        "anim",
        "raytracing-fireflies-",
        FRAMES,
        "final/raytracing/firefly_lake.gif",
    )
    .delay_cs(5)
    .preview(12, "final/raytracing/firefly_lake.png")
    .unique_frame_dir(true);

    println!("Rendering {} path-traced frames in parallel...", FRAMES);

    FrameRecorder::render_gif_auto(options, |frame| Ok(render_frame(frame)))?;

    println!("Saved final/raytracing/firefly_lake.png and final/raytracing/firefly_lake.gif");
    Ok(())
}

fn render_frame(frame: usize) -> Canvas {
    let t = frame as f64 / FRAMES as f64;

    // 1. Build the stable list of reeds and firefly configurations
    let (fireflies, reeds) = build_scene_assets();

    // 2. Build the world state and camera for this frame
    let (world, sampling_targets, camera) = build_scene(t, &fireflies, &reeds);

    // 3. Path trace the frame
    PathTracer::new(camera).render_with_lights(&world, &sampling_targets)
}

fn build_scene_assets() -> (Vec<Firefly>, Vec<Reed>) {
    let mut seed = 0xabcdef1234_u64;
    let lcg = |s: &mut u64| {
        *s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*s as f64) / (u64::MAX as f64)
    };

    // Deterministically generate 36 fireflies hovering above the lake
    let mut fireflies = Vec::with_capacity(36);
    for _ in 0..36 {
        fireflies.push(Firefly {
            base_x: -160.0 + lcg(&mut seed) * 320.0,
            base_y: 5.0 + lcg(&mut seed) * 55.0,
            base_z: 60.0 + lcg(&mut seed) * 340.0,
            speed_x: 0.8 + lcg(&mut seed) * 1.2,
            speed_y: 1.2 + lcg(&mut seed) * 1.5,
            speed_z: 0.7 + lcg(&mut seed) * 1.0,
            phase_x: lcg(&mut seed) * PI * 2.0,
            phase_y: lcg(&mut seed) * PI * 2.0,
            phase_z: lcg(&mut seed) * PI * 2.0,
            flash_speed: 1.5 + lcg(&mut seed) * 2.5,
            flash_phase: lcg(&mut seed) * PI * 2.0,
        });
    }

    // Deterministically generate 40 reeds clustered near the side banks
    let mut reeds = Vec::with_capacity(40);
    for _ in 0..40 {
        let side = if lcg(&mut seed) > 0.5 { 1.0 } else { -1.0 };
        let rx = side * (75.0 + lcg(&mut seed) * 105.0);
        let rz = 40.0 + lcg(&mut seed) * 320.0;
        let rheight = 40.0 + lcg(&mut seed) * 90.0;
        reeds.push(Reed {
            x: rx,
            z: rz,
            height: rheight,
        });
    }

    (fireflies, reeds)
}

fn build_scene(
    t: f64,
    fireflies: &[Firefly],
    reeds: &[Reed],
) -> (HittableList, WeightedSamplingTargetList, RayCamera) {
    let mut world = HittableList::with_capacity(85);

    // ── Materials ────────────────────────────────────────────────────────────
    let water_albedo = LinearColor::new(0.01, 0.02, 0.04);
    let water = Metal::new(water_albedo, 0.14); // semi-reflective water surface
    let reed_mat: MaterialRef = Arc::new(Lambertian::new(LinearColor::new(0.02, 0.03, 0.015))); // dark green/brown
    let moon_light = DiffuseLight::new(LinearColor::new(6.0, 5.2, 4.0)); // soft warm light

    // 1. Water Plane (the Lake)
    world.add(Quad::with_material(
        Point::new(-1000.0, 0.0, -100.0),
        Vector::new(2000.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 1500.0),
        water,
    ));

    // 2. The Moon (distant warm light)
    let moon_center = Point::new(0.0, 160.0, 850.0);
    let moon_radius = 50.0;
    world.add(Sphere::with_material(moon_center, moon_radius, moon_light));

    // 3. Reeds (silhouetted vegetation near the banks)
    for reed in reeds {
        let r_min = Point::new(reed.x - 2.0, 0.0, reed.z - 2.0);
        let r_max = Point::new(reed.x + 2.0, reed.height, reed.z + 2.0);
        world.add(box_object(r_min, r_max, reed_mat.clone()));
    }

    // 4. Fireflies (pulsing light sources)
    let mut sampling_targets = WeightedSamplingTargetList::with_capacity(fireflies.len() + 1);
    sampling_targets.add_sphere_weighted(moon_center, moon_radius, 6.0);

    for ff in fireflies {
        // Organic sinusoidal motion
        let ox = 12.0 * (t * 2.0 * PI * ff.speed_x + ff.phase_x).sin();
        let oy = 8.0 * (t * 2.0 * PI * ff.speed_y + ff.phase_y).sin();
        let oz = 10.0 * (t * 2.0 * PI * ff.speed_z + ff.phase_z).sin();
        let x = ff.base_x + ox;
        let y = ff.base_y + oy;
        let z = ff.base_z + oz;
        let f_pos = Point::new(x, y, z);

        // Flashing animation
        let flash = 0.5 + 0.5 * (t * 2.0 * PI * ff.flash_speed + ff.flash_phase).sin();
        let intensity = 18.0 * flash.powf(2.5); // sharp flashing peak

        // Greenish-yellow bioluminescent color
        let glow_color = LinearColor::new(intensity * 0.72, intensity * 0.96, intensity * 0.12);
        let light_mat = DiffuseLight::new(glow_color);

        let radius = 2.4;
        world.add(Sphere::with_material(f_pos, radius, light_mat));

        // Dynamically add firefly to sampling target list when glowing
        if intensity > 1.5 {
            sampling_targets.add_sphere_weighted(f_pos, radius, intensity.max(1.0));
        }
    }

    // ── Camera Animation ─────────────────────────────────────────────────────
    // Drifts slowly forward and bobs vertically
    let cam_x = -15.0 + 15.0 * (t * 2.0 * PI).sin();
    let cam_y = 10.0 + 3.0 * (t * 2.0 * PI).cos();
    let cam_z = 2.0 + 25.0 * t;
    let lookfrom = Point::new(cam_x, cam_y, cam_z);

    // Peering low across the lake surface
    let lookat = Point::new(0.0, 24.0, 300.0);

    let camera = RayCamera::new(IMAGE_WIDTH, 1.0)
        .with_stratified_grid_width(STRATIFIED_GRID_WIDTH)
        .with_max_depth(MAX_DEPTH)
        .with_background(LinearColor::new(0.001, 0.002, 0.004)) // dark night sky
        .with_vertical_fov(42.0)
        .with_look_at(lookfrom, lookat)
        .with_view_up(Vector::new(0.0, 1.0, 0.0));

    (world, sampling_targets, camera)
}
