//! Marching-cubes density sphere rendered as a glassy water surface.
//!
//! Outputs a PNG to `final/raytracing/marching_cubes_sphere.png`.

use gartus::prelude::*;
use std::{error::Error, fs};

const IMAGE_WIDTH: u32 = 320;
const STRATIFIED_GRID_WIDTH: u32 = 5;
const MAX_DEPTH: u32 = 18;

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final/raytracing")?;

    let canvas = render();
    let path = "final/raytracing/marching_cubes_sphere.png";
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
        .with_vertical_fov(34.0)
        .with_look_at(Point::new(4.8, 2.8, 5.8), Point::new(0.0, 0.45, 0.0))
        .with_view_up(Vector::new(0.0, 1.0, 0.0))
        .with_direct_lighting_mode(DirectLightingMode::NextEventEstimation);

    PathTracer::new(camera).render_ray_scene(&scene)
}

fn build_scene() -> RayScene {
    let surface = MarchingCubes::new()
        .with_iso_value(0.5)
        .extract(&sphere_density());

    RayScene::builder()
        .material(
            "floor",
            RayMaterial::lambertian(LinearColor::new(0.06, 0.065, 0.072)),
        )
        .material(
            "light",
            RayMaterial::diffuse_light(LinearColor::new(5.4, 6.5, 8.5)),
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

fn sphere_density() -> GridDensityField {
    let bounds = GridBounds::new(Point::new(-1.4, -1.4, -1.4), Point::new(1.4, 1.4, 1.4));
    GridDensityField::from_fn(bounds, [30, 30, 30], |point| {
        let radius = (point - Point::new(0.0, 0.25, 0.0)).length();
        (1.0 - radius / 1.05).clamp(0.0, 1.0)
    })
}
