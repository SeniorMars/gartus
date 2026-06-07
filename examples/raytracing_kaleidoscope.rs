//! This example constructs a physical 3D kaleidoscope using the raytracing library:
//! 1. Three perfectly reflective mirror walls form an equilateral triangular prism tube.
//! 2. A dark back cap closes the far end of the tube.
//! 3. Inside the tube, a refractive glass sphere and a rotating mirror cube float.
//! 4. Four bright neon stars (magenta, cyan, yellow, lime) orbit the Z-axis, casting
//!    refracted caustics and bouncing infinitely off the mirror walls.
//! 5. The camera is placed at the entrance, looking down the tube. It slowly rolls (spins)
//!    around its optical axis over time, which causes the entire infinite hexagonal lattice
//!    of reflections to spin dynamically on screen, mimicking a real handheld kaleidoscope.
//!
//! Outputs a GIF and a PNG preview to `final/raytracing/`.

use gartus::prelude::*;
use std::{error::Error, f64::consts::PI, fs, sync::Arc};

const IMAGE_WIDTH: u32 = 400;
const STRATIFIED_GRID_WIDTH: u32 = 16; // 16x16 = 256 samples per pixel for clean reflections
const MAX_DEPTH: u32 = 10; // deep reflections are crucial for the kaleidoscope effect
const FRAMES: usize = 24;

fn main() {
    if let Err(err) = render() {
        eprintln!("could not render kaleidoscope:\n{err}");
        std::process::exit(1);
    }
}

fn render() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final/raytracing")?;

    let options = AnimationRenderOptions::new(
        "anim",
        "raytracing-kaleidoscope-",
        FRAMES,
        "final/raytracing/kaleidoscope.gif",
    )
    .delay_cs(5)
    .preview(12, "final/raytracing/kaleidoscope.png")
    .unique_frame_dir(true);

    println!(
        "Rendering {} path-traced kaleidoscope frames in parallel...",
        FRAMES
    );

    FrameRecorder::render_gif_auto(options, |frame| Ok(render_frame(frame)))?;

    println!("Saved final/raytracing/kaleidoscope.png and final/raytracing/kaleidoscope.gif");
    Ok(())
}

fn render_frame(frame: usize) -> Canvas {
    let t = frame as f64 / FRAMES as f64;

    // 1. Build the kaleidoscope world and camera
    let (world, sampling_targets, camera) = build_scene(t);

    // 2. Path trace the frame
    PathTracer::new(camera).render_with_lights(&world, &sampling_targets)
}

fn build_scene(t: f64) -> (HittableList, WeightedSamplingTargetList, RayCamera) {
    let mut world = HittableList::with_capacity(10);

    // ── Materials ────────────────────────────────────────────────────────────
    // Perfect mirror for the kaleidoscope walls
    let mirror = Metal::new(LinearColor::new(0.99, 0.99, 0.99), 0.0);
    // Dark matte cap for the back
    let cap_mat = Lambertian::new(LinearColor::new(0.04, 0.04, 0.06));
    // Refractive glass for the core
    let glass = Dielectric::new(RefractiveIndex::GLASS);
    // Gold mirror for the rotating box
    let gold_shared: MaterialRef = Arc::new(Metal::new(LinearColor::new(0.95, 0.75, 0.25), 0.02));

    // ── Triangular Prism Tube ────────────────────────────────────────────────
    // Side length L = 200, Height H = 173.205
    // Centered at (0, 0) in X-Y. Vertices:
    // A (top): (0.0, 115.47)
    // B (bottom-left): (-100.0, -57.735)
    // C (bottom-right): (100.0, -57.735)
    let length = 500.0;

    // Wall 1: Bottom wall (between C and B, normal points +Y)
    world.add(Quad::with_material(
        Point::new(100.0, -57.735, 0.0),
        Vector::new(-200.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, length),
        mirror,
    ));

    // Wall 2: Right wall (between A and C, normal points down-left)
    world.add(Quad::with_material(
        Point::new(0.0, 115.47, 0.0),
        Vector::new(100.0, -173.205, 0.0),
        Vector::new(0.0, 0.0, length),
        mirror,
    ));

    // Wall 3: Left wall (between B and A, normal points down-right)
    world.add(Quad::with_material(
        Point::new(-100.0, -57.735, 0.0),
        Vector::new(100.0, 173.205, 0.0),
        Vector::new(0.0, 0.0, length),
        mirror,
    ));

    // Back Cap (large square closing the tube at z = 500)
    world.add(Quad::with_material(
        Point::new(-300.0, -300.0, length),
        Vector::new(600.0, 0.0, 0.0),
        Vector::new(0.0, 600.0, 0.0),
        cap_mat,
    ));

    // ── Floating Central Objects ─────────────────────────────────────────────
    // Central Glass Sphere (bobs slightly up/down)
    let glass_center = Point::new(0.0, 12.0 * (t * 2.0 * PI).sin(), 250.0);
    world.add(Sphere::with_material(glass_center, 35.0, glass));

    // Rotating Gold Mirror Box at z = 320
    let box_center = Point::new(0.0, 15.0 * (t * 2.0 * PI + PI / 2.0).cos(), 330.0);
    world.add(Translate::new(
        RotateY::new(
            box_object(
                Point::new(-16.0, -16.0, -16.0),
                Point::new(16.0, 16.0, 16.0),
                gold_shared,
            ),
            t * 360.0 * 2.0,
        ),
        Vector::new(box_center.x(), box_center.y(), box_center.z()),
    ));

    // ── Orbiting Colorful Stars ──────────────────────────────────────────────
    // Star 1: Glowing Pink/Magenta
    let r1 = 40.0;
    let a1 = t * 2.0 * PI;
    let s1_center = Point::new(r1 * a1.cos(), r1 * a1.sin(), 150.0);
    let light_magenta = DiffuseLight::new(LinearColor::new(30.0, 1.0, 18.0));
    world.add(Sphere::with_material(s1_center, 12.0, light_magenta));

    // Star 2: Glowing Cyan
    let r2 = 42.0;
    let a2 = -t * 2.0 * PI + PI / 2.0;
    let s2_center = Point::new(r2 * a2.cos(), r2 * a2.sin(), 200.0);
    let light_cyan = DiffuseLight::new(LinearColor::new(1.0, 30.0, 30.0));
    world.add(Sphere::with_material(s2_center, 10.0, light_cyan));

    // Star 3: Glowing Golden Yellow
    let r3 = 38.0;
    let a3 = t * 2.0 * PI + PI;
    let s3_center = Point::new(r3 * a3.cos(), r3 * a3.sin(), 300.0);
    let light_yellow = DiffuseLight::new(LinearColor::new(30.0, 24.0, 1.0));
    world.add(Sphere::with_material(s3_center, 11.0, light_yellow));

    // Star 4: Glowing Lime Green
    let r4 = 45.0;
    let a4 = -t * 2.0 * PI + 3.0 * PI / 2.0;
    let s4_center = Point::new(r4 * a4.cos(), r4 * a4.sin(), 370.0);
    let light_lime = DiffuseLight::new(LinearColor::new(6.0, 30.0, 1.0));
    world.add(Sphere::with_material(s4_center, 13.0, light_lime));

    // ── Importance Sampling Targets ──────────────────────────────────────────
    let mut sampling_targets = WeightedSamplingTargetList::with_capacity(5);
    sampling_targets.add_sphere_weighted(s1_center, 12.0, 12.0 * 12.0);
    sampling_targets.add_sphere_weighted(s2_center, 10.0, 10.0 * 10.0);
    sampling_targets.add_sphere_weighted(s3_center, 11.0, 11.0 * 11.0);
    sampling_targets.add_sphere_weighted(s4_center, 13.0, 13.0 * 13.0);
    sampling_targets.add_sphere_weighted(glass_center, 35.0, 35.0 * 35.0);

    // ── Camera Roll Animation ────────────────────────────────────────────────
    // Look straight down the Z-axis of the triangular tube
    let lookfrom = Point::new(0.0, 0.0, -45.0);
    let lookat = Point::new(0.0, 0.0, 200.0);
    // Roll the camera (spin around Z-axis) over time
    let roll_angle = t * 2.0 * PI;
    let view_up = Vector::new(roll_angle.sin(), roll_angle.cos(), 0.0);

    let camera = RayCamera::new(IMAGE_WIDTH, 1.0)
        .with_stratified_grid_width(STRATIFIED_GRID_WIDTH)
        .with_max_depth(MAX_DEPTH)
        .with_background(LinearColor::default())
        .with_vertical_fov(72.0)
        .with_look_at(lookfrom, lookat)
        .with_view_up(view_up);

    (world, sampling_targets, camera)
}
