//! GGX/Trowbridge-Reitz glossy BRDF roughness sweep.
//!
//! Outputs a PNG to `final/raytracing/ggx_microfacet.png`.

use gartus::prelude::*;
use std::{error::Error, fs};

const IMAGE_WIDTH: u32 = 360;
const STRATIFIED_GRID_WIDTH: u32 = 8;
const MAX_DEPTH: u32 = 16;

fn main() -> Result<(), Box<dyn Error>> {
    fs::create_dir_all("final/raytracing")?;

    let (world, lights) = build_scene();
    let camera = RayCamera::new(IMAGE_WIDTH, 16.0 / 9.0)
        .with_stratified_grid_width(STRATIFIED_GRID_WIDTH)
        .with_max_depth(MAX_DEPTH)
        .with_background(LinearColor::new(0.01, 0.014, 0.022))
        .with_vertical_fov(34.0)
        .with_look_at(Point::new(4.7, 2.1, 5.0), Point::new(0.0, 0.55, -0.5))
        .with_view_up(Vector::new(0.0, 1.0, 0.0))
        .with_direct_lighting_mode(DirectLightingMode::NextEventEstimation);

    let canvas = PathTracer::new(camera).render_with_lights(&world, &lights);
    let path = "final/raytracing/ggx_microfacet.png";
    canvas.save_extension(path)?;
    println!("saved {path}");

    Ok(())
}

fn build_scene() -> (HittableList, WeightedSamplingTargetList) {
    let mut world = HittableList::with_capacity(8);
    let mut lights = WeightedSamplingTargetList::with_capacity(2);

    world.add(Quad::with_material(
        Point::new(-5.5, -0.05, -4.2),
        Vector::new(11.0, 0.0, 0.0),
        Vector::new(0.0, 0.0, 8.0),
        Lambertian::checker(
            0.7,
            LinearColor::new(0.12, 0.12, 0.13),
            LinearColor::new(0.035, 0.04, 0.05),
        ),
    ));

    let light_corner = Point::new(-2.5, 4.0, -1.5);
    let light_u = Vector::new(5.0, 0.0, 0.0);
    let light_v = Vector::new(0.0, 0.0, 2.0);
    world.add(Quad::with_material(
        light_corner,
        light_u,
        light_v,
        DiffuseLight::new(LinearColor::new(6.5, 6.1, 5.4)),
    ));
    lights.add_quad_weighted(light_corner, light_u, light_v, 12.0);

    let accent_center = Point::new(-2.8, 0.65, 1.5);
    world.add(Sphere::with_material(
        accent_center,
        0.16,
        DiffuseLight::new(LinearColor::new(1.2, 2.3, 6.0)),
    ));
    lights.add_sphere_weighted(accent_center, 0.16, 2.0);

    let roughness_values = [0.08, 0.22, 0.45, 0.72];
    for (index, roughness) in roughness_values.into_iter().enumerate() {
        let x = -1.8 + f64::from(u32::try_from(index).expect("index fits u32")) * 1.2;
        let color = LinearColor::new(0.95 - 0.1 * roughness, 0.75, 0.48 + 0.3 * roughness);
        world.add(Sphere::with_material(
            Point::new(x, 0.55, -0.8),
            0.55,
            GgxMicrofacet::new(color, roughness),
        ));
    }

    (world, lights)
}
