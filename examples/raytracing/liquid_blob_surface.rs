//! Particle-splat liquid blob extracted to triangles and rendered as water.
//!
//! Outputs a PNG to `final/raytracing/liquid_blob_surface.png`.

use gartus::prelude::*;
use std::{error::Error, fs};

const IMAGE_WIDTH: u32 = 320;
const STRATIFIED_GRID_WIDTH: u32 = 5;
const MAX_DEPTH: u32 = 18;

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final/raytracing")?;

    let canvas = render();
    let path = "final/raytracing/liquid_blob_surface.png";
    canvas.save_extension(path)?;
    println!("saved {path}");

    Ok(())
}

fn render() -> Canvas {
    let scene = build_scene();
    let camera = RayCamera::new(IMAGE_WIDTH, 16.0 / 9.0)
        .with_stratified_grid_width(STRATIFIED_GRID_WIDTH)
        .with_max_depth(MAX_DEPTH)
        .with_background(LinearColor::new(0.006, 0.010, 0.018))
        .with_vertical_fov(35.0)
        .with_look_at(Point::new(4.8, 2.6, 5.5), Point::new(0.0, 0.35, 0.0))
        .with_view_up(Vector::new(0.0, 1.0, 0.0))
        .with_direct_lighting_mode(DirectLightingMode::NextEventEstimation);

    PathTracer::new(camera).render_ray_scene(&scene)
}

fn build_scene() -> RayScene {
    let surface = LiquidSurface::from_particles(blob_particles())
        .with_resolution([28, 24, 28])
        .with_iso_value(0.35)
        .with_kernel(SplatKernel::Poly6)
        .build_triangle_mesh();

    RayScene::builder()
        .material(
            "floor",
            RayMaterial::lambertian(LinearColor::new(0.055, 0.060, 0.070)),
        )
        .material(
            "light",
            RayMaterial::diffuse_light(LinearColor::new(5.6, 6.6, 8.8)),
        )
        .material("water", RayMaterial::dielectric(RefractiveIndex::WATER))
        .quad(
            Point::new(-6.0, -0.65, -4.5),
            Vector::new(12.0, 0.0, 0.0),
            Vector::new(0.0, 0.0, 9.0),
            "floor",
        )
        .quad(
            Point::new(-2.5, 4.2, -0.7),
            Vector::new(5.0, 0.0, 0.0),
            Vector::new(0.0, 0.0, 2.8),
            "light",
        )
        .triangles(surface.triangles().iter().copied(), "water")
        .build_bvh()
}

fn blob_particles() -> Vec<FluidParticle> {
    vec![
        FluidParticle::new(Point::new(-0.35, 0.05, 0.0), 0.62, 1.0),
        FluidParticle::new(Point::new(0.30, 0.08, 0.02), 0.58, 1.0),
        FluidParticle::new(Point::new(0.0, 0.52, 0.0), 0.50, 0.9),
        FluidParticle::new(Point::new(-0.05, -0.22, 0.35), 0.38, 0.7),
        FluidParticle::new(Point::new(0.14, -0.18, -0.32), 0.36, 0.7),
    ]
}
