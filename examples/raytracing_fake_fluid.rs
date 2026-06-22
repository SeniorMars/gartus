//! Fake fluid volume example using `FnDensityField` and `NonUniformMedium`.
//!
//! This is not a fluid solver. It authors a time-varying density field directly, then lets the
//! existing Woodcock-tracked volume renderer turn that field into smoke/plasma-like scattering.
//! Render different `FRAME` values, or drive `render_frame(frame)` from `FrameRecorder`, to animate
//! the wave field through a fixed shutter time.
//!
//! Outputs a PNG to `final/raytracing/fake_fluid.png`.

use gartus::prelude::*;
use std::{error::Error, fs};

const IMAGE_WIDTH: u32 = 360;
const STRATIFIED_GRID_WIDTH: u32 = 8;
const MAX_DEPTH: u32 = 18;
const FRAME: usize = 12;
const FPS: f64 = 24.0;

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final/raytracing")?;

    let canvas = render_frame(FRAME);
    let path = "final/raytracing/fake_fluid.png";
    canvas.save_extension(path)?;
    println!("saved {path}");

    Ok(())
}

fn render_frame(frame: usize) -> Canvas {
    let time = frame as f64 / FPS;
    let (world, lights) = build_scene(time);
    let camera = RayCamera::new(IMAGE_WIDTH, 16.0 / 9.0)
        .with_stratified_grid_width(STRATIFIED_GRID_WIDTH)
        .with_max_depth(MAX_DEPTH)
        .with_background(LinearColor::new(0.006, 0.010, 0.018))
        .with_vertical_fov(36.0)
        .with_look_at(Point::new(6.0, 3.0, 7.0), Point::new(0.0, 1.0, 0.0))
        .with_view_up(Vector::new(0.0, 1.0, 0.0))
        .with_shutter_interval(time, time)
        .with_direct_lighting_mode(DirectLightingMode::NextEventEstimation);

    PathTracer::new(camera).render_with_lights(&world, &lights)
}

fn build_scene(time: f64) -> (HittableList, WeightedSamplingTargetList) {
    let mut world = HittableList::with_capacity(6);
    let mut lights = WeightedSamplingTargetList::with_capacity(2);

    let floor = Lambertian::checker(
        0.8,
        LinearColor::new(0.055, 0.060, 0.065),
        LinearColor::new(0.025, 0.030, 0.038),
    );
    world.add(Quad::with_material(
        Point::new(-8.0, -1.1, -6.0),
        Vector::new(16.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 12.0),
        floor,
    ));

    let back_wall = Lambertian::new(LinearColor::new(0.018, 0.030, 0.046));
    world.add(Quad::with_material(
        Point::new(-8.0, -1.1, -4.0),
        Vector::new(16.0, 0.0, 0.0),
        Vector::new(0.0, 8.0, 0.0),
        back_wall,
    ));

    let key_light_corner = Point::new(-2.8, 5.2, 0.5);
    let key_light_u = Vector::new(5.6, 0.0, 0.0);
    let key_light_v = Vector::new(0.0, 0.0, 3.0);
    world.add(Quad::with_material(
        key_light_corner,
        key_light_u,
        key_light_v,
        DiffuseLight::new(LinearColor::new(4.5, 5.7, 8.5)),
    ));
    lights.add_quad_weighted(key_light_corner, key_light_u, key_light_v, 10.0);

    let accent_center = Point::new(-3.0, 0.4 + 0.2 * time.sin(), 1.6);
    world.add(Sphere::with_material(
        accent_center,
        0.22,
        DiffuseLight::new(LinearColor::new(0.8, 2.4, 7.5)),
    ));
    lights.add_sphere_weighted(accent_center, 0.22, 2.0);

    let fluid_center = Point::new(0.0, 1.0, 0.0);
    let fluid_radius = 3.0;
    let fluid_density = FnDensityField::new(0.8, move |p: Point, sample_time: f64| {
        let local = p - fluid_center;
        let falloff = (1.0 - local.length() / fluid_radius).clamp(0.0, 1.0);

        let wave_a = (p.x() * 2.4 + p.y() * 0.7 + sample_time * 1.2).sin();
        let wave_b = (p.y() * 3.1 - p.z() * 0.9 - sample_time * 0.8).sin();
        let wave_c = (p.z() * 2.7 + p.x() * 0.5 + sample_time * 1.6).sin();
        let wisps = 0.5 + 0.5 * ((wave_a + wave_b + wave_c) / 3.0);

        0.8 * falloff * wisps.powf(2.0)
    });

    world.add(NonUniformMedium::new(
        Sphere::new(fluid_center, fluid_radius),
        fluid_density,
        LinearColor::new(0.35, 0.65, 1.0),
    ));

    (world, lights)
}
