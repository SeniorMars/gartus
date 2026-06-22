//! Particle splat volume example.
//!
//! This uses `ParticleSplatField` to turn overlapping density particles into a soft volumetric
//! blob. It is a density field, not a simulation.
//!
//! Outputs a PNG to `final/raytracing/particle_splats.png`.

use gartus::prelude::*;
use std::{error::Error, fs};

const IMAGE_WIDTH: u32 = 320;
const STRATIFIED_GRID_WIDTH: u32 = 6;
const MAX_DEPTH: u32 = 18;

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final/raytracing")?;

    let canvas = render();
    let path = "final/raytracing/particle_splats.png";
    canvas.save_extension(path)?;
    println!("saved {path}");

    Ok(())
}

fn render() -> Canvas {
    let (world, lights) = build_scene();
    let camera = RayCamera::new(IMAGE_WIDTH, 16.0 / 9.0)
        .with_stratified_grid_width(STRATIFIED_GRID_WIDTH)
        .with_max_depth(MAX_DEPTH)
        .with_background(LinearColor::new(0.006, 0.010, 0.018))
        .with_vertical_fov(35.0)
        .with_look_at(Point::new(5.6, 2.6, 6.4), Point::new(0.0, 0.75, 0.0))
        .with_view_up(Vector::new(0.0, 1.0, 0.0))
        .with_direct_lighting_mode(DirectLightingMode::NextEventEstimation);

    PathTracer::new(camera).render_with_lights(&world, &lights)
}

fn build_scene() -> (HittableList, WeightedSamplingTargetList) {
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

    let light_corner = Point::new(-2.5, 4.5, -0.7);
    let light_u = Vector::new(5.0, 0.0, 0.0);
    let light_v = Vector::new(0.0, 0.0, 2.8);
    world.add(Quad::with_material(
        light_corner,
        light_u,
        light_v,
        DiffuseLight::new(LinearColor::new(5.5, 6.4, 8.5)),
    ));
    lights.add_quad_weighted(light_corner, light_u, light_v, 10.0);

    let accent_center = Point::new(-2.6, 0.3, 1.4);
    world.add(Sphere::with_material(
        accent_center,
        0.18,
        DiffuseLight::new(LinearColor::new(0.6, 2.0, 7.5)),
    ));
    lights.add_sphere_weighted(accent_center, 0.18, 2.0);

    let splats = ParticleSplatField::new(blob_particles())
        .with_kernel(SplatKernel::Poly6)
        .with_max_density(2.8)
        .with_cell_size(0.28);

    world.add(NonUniformMedium::new(
        Sphere::new(Point::new(0.0, 0.75, 0.0), 2.3),
        splats,
        LinearColor::new(0.7, 0.85, 1.0),
    ));

    (world, lights)
}

fn blob_particles() -> Vec<FluidParticle> {
    let mut particles = Vec::new();

    for y_layer in 0..5 {
        let y = 0.05 + 0.32 * f64::from(y_layer);
        let layer_radius = 0.9 - 0.08 * f64::from(y_layer);
        for index in 0..10 {
            let angle = f64::from(index) * std::f64::consts::TAU / 10.0 + 0.55 * f64::from(y_layer);
            let radius = layer_radius * (0.5 + 0.12 * f64::from((index + y_layer) % 3));
            let x = radius * angle.cos();
            let z = radius * angle.sin();
            particles.push(FluidParticle::new(
                Point::new(x, y, z),
                0.36,
                0.20 + 0.04 * f64::from(y_layer),
            ));
        }
    }

    particles.push(FluidParticle::new(Point::new(0.0, 0.8, 0.0), 0.72, 0.75));
    particles
}
