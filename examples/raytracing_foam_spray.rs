//! Foam and spray volume example using particle splats.
//!
//! The particles are hand-authored along an arc, which gives a liquid-ish plume without running a
//! fluid solver.
//!
//! Outputs a PNG to `final/raytracing/foam_spray.png`.

use gartus::prelude::*;
use std::{error::Error, fs};

const IMAGE_WIDTH: u32 = 320;
const STRATIFIED_GRID_WIDTH: u32 = 6;
const MAX_DEPTH: u32 = 16;

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final/raytracing")?;

    let canvas = render();
    let path = "final/raytracing/foam_spray.png";
    canvas.save_extension(path)?;
    println!("saved {path}");

    Ok(())
}

fn render() -> Canvas {
    let (world, lights) = build_scene();
    let camera = RayCamera::new(IMAGE_WIDTH, 16.0 / 9.0)
        .with_stratified_grid_width(STRATIFIED_GRID_WIDTH)
        .with_max_depth(MAX_DEPTH)
        .with_background(LinearColor::new(0.010, 0.014, 0.022))
        .with_vertical_fov(38.0)
        .with_look_at(Point::new(5.8, 2.4, 6.0), Point::new(0.1, 0.5, 0.0))
        .with_view_up(Vector::new(0.0, 1.0, 0.0))
        .with_direct_lighting_mode(DirectLightingMode::NextEventEstimation);

    PathTracer::new(camera).render_with_lights(&world, &lights)
}

fn build_scene() -> (HittableList, WeightedSamplingTargetList) {
    let mut world = HittableList::with_capacity(6);
    let mut lights = WeightedSamplingTargetList::with_capacity(2);

    world.add(Quad::with_material(
        Point::new(-7.0, -0.8, -5.0),
        Vector::new(14.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 10.0),
        Lambertian::new(LinearColor::new(0.035, 0.045, 0.055)),
    ));

    let light_corner = Point::new(-2.4, 4.2, -0.6);
    let light_u = Vector::new(5.0, 0.0, 0.0);
    let light_v = Vector::new(0.0, 0.0, 2.8);
    world.add(Quad::with_material(
        light_corner,
        light_u,
        light_v,
        DiffuseLight::new(LinearColor::new(5.8, 6.8, 8.2)),
    ));
    lights.add_quad_weighted(light_corner, light_u, light_v, 10.0);

    let glow_center = Point::new(-2.8, 0.15, 1.5);
    world.add(Sphere::with_material(
        glow_center,
        0.16,
        DiffuseLight::new(LinearColor::new(0.5, 1.8, 6.5)),
    ));
    lights.add_sphere_weighted(glow_center, 0.16, 1.8);

    world.add(Sphere::with_material(
        Point::new(-0.6, -0.42, 0.0),
        0.38,
        Metal::new(LinearColor::new(0.45, 0.52, 0.58), 0.12),
    ));

    let splats = ParticleSplatField::new(spray_particles())
        .with_kernel(SplatKernel::Gaussian)
        .with_max_density(1.6)
        .with_cell_size(0.18);

    world.add(NonUniformMedium::new(
        Sphere::new(Point::new(0.2, 0.65, 0.0), 2.5),
        splats,
        LinearColor::new(0.78, 0.92, 1.0),
    ));

    (world, lights)
}

fn spray_particles() -> Vec<FluidParticle> {
    let mut particles = Vec::new();

    for index in 0..36 {
        let t = f64::from(index) / 35.0;
        let x = -0.8 + 2.2 * t;
        let y = -0.15 + 1.65 * (1.0 - (2.0 * t - 1.0).powi(2));
        let z = 0.32 * (t * std::f64::consts::TAU * 2.5).sin();
        let radius = 0.16 - 0.06 * t;
        let density = 0.16 + 0.18 * (1.0 - t);
        particles.push(FluidParticle::new(Point::new(x, y, z), radius, density));
    }

    for index in 0..24 {
        let t = f64::from(index) / 23.0;
        let angle = t * std::f64::consts::TAU * 3.0;
        particles.push(FluidParticle::new(
            Point::new(-0.65 + 0.75 * t, -0.28 + 0.22 * t, 0.45 * angle.sin()),
            0.08,
            0.10,
        ));
    }

    particles
}
