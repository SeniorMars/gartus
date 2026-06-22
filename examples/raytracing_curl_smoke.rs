//! Curl-noise smoke example using `DomainWarpedDensityField`.
//!
//! This is still procedural density, not a fluid solver. `CurlNoiseField` warps the smoke density's
//! sample coordinates so the volume twists and rolls over time while staying renderer-friendly.
//!
//! Outputs a PNG to `final/raytracing/curl_smoke.png`.

use gartus::prelude::*;
use std::{error::Error, fs};

const IMAGE_WIDTH: u32 = 320;
const STRATIFIED_GRID_WIDTH: u32 = 6;
const MAX_DEPTH: u32 = 18;
const FRAME: usize = 16;
const FPS: f64 = 24.0;

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final/raytracing")?;

    let canvas = render_frame(FRAME);
    let path = "final/raytracing/curl_smoke.png";
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
        .with_look_at(Point::new(5.5, 2.7, 6.5), Point::new(0.0, 0.8, 0.0))
        .with_view_up(Vector::new(0.0, 1.0, 0.0))
        .with_shutter_interval(time, time)
        .with_direct_lighting_mode(DirectLightingMode::NextEventEstimation);

    PathTracer::new(camera).render_with_lights(&world, &lights)
}

fn build_scene(time: f64) -> (HittableList, WeightedSamplingTargetList) {
    let mut world = HittableList::with_capacity(5);
    let mut lights = WeightedSamplingTargetList::with_capacity(2);

    world.add(Quad::with_material(
        Point::new(-7.0, -1.0, -5.0),
        Vector::new(14.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 10.0),
        Lambertian::checker(
            0.7,
            LinearColor::new(0.05, 0.055, 0.060),
            LinearColor::new(0.025, 0.030, 0.038),
        ),
    ));

    let key_light_corner = Point::new(-2.5, 4.6, -0.6);
    let key_light_u = Vector::new(5.0, 0.0, 0.0);
    let key_light_v = Vector::new(0.0, 0.0, 2.8);
    world.add(Quad::with_material(
        key_light_corner,
        key_light_u,
        key_light_v,
        DiffuseLight::new(LinearColor::new(5.0, 6.4, 8.8)),
    ));
    lights.add_quad_weighted(key_light_corner, key_light_u, key_light_v, 10.0);

    let accent_center = Point::new(-2.7, 0.35 + 0.15 * time.sin(), 1.4);
    world.add(Sphere::with_material(
        accent_center,
        0.18,
        DiffuseLight::new(LinearColor::new(0.8, 2.2, 7.0)),
    ));
    lights.add_sphere_weighted(accent_center, 0.18, 2.0);

    let smoke_center = Point::new(0.0, 0.9, 0.0);
    let smoke_radius = 2.6;
    let smoke_density = ProceduralDensityField::smoke()
        .with_seed(10)
        .with_scale(1.5)
        .with_speed(0.35)
        .with_max_density(0.45)
        .domain_warped()
        .with_warp_seed(99)
        .with_warp_strength(0.75)
        .with_warp_scale(1.3)
        .with_warp_speed(0.5);

    world.add(NonUniformMedium::new(
        Sphere::new(smoke_center, smoke_radius),
        smoke_density,
        LinearColor::new(0.7, 0.82, 1.0),
    ));

    (world, lights)
}
